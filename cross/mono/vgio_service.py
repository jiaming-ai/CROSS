"""GPU side of the VGGT-Omega + IMU odometry (cross/mono/vggt_imu_frontend.py): the frontend's own forward passes, and
what its local pose graph needs from the passes' depth maps, reduced to a few numbers.

The frontend (IMU, tracked corners, local pose graph) asks for a visual measurement of frame b with a request: the last
measured frame m, the keyframe (or the frame measured before m) as a third view, its motion since m (the temporal anchor
of the back end's pass), and the corners it detected on b.  The service keeps the images of the requested frames and
each one's depth in the pass where it was the current frame.  For a request it uses the back end's forward pass when
that pass carried the request's anchor, else a pass of its own [b, m, third], and summarizes it (measure):

    c2w_curr / prev / kf / right   the views' camera poses (principal-point corrected)
    link, link_kf                  depth ratio of m (and of the keyframe) between this pass and its own pass, with its
                                   spread: the gauge links of the graph
    da3 / stereo                   the pass's metric scale from learned depth (mono) or stereo depth (stereo)
    corner_z                       the stereo depth at the corners detected on b (the next measurement's PnP)
    log_median_depth               the first node's gauge

In-process for a local session (the frontend owns one); on the GPU server for a remote session (cross.remote), where
the summary is all that travels back to the edge."""

from time import perf_counter

import numpy as np
import torch

from .geometry import inverse
from .scale import observe_scale


class VgioPassService:
    def __init__(self, K, config, backend=None, rgb_transform=None, depth_transform=None, metric_model=None,
                 T_right_in_left=None, depth_every: int = 3):
        self.config = config
        self.K = np.asarray(K, dtype=np.float64).copy()
        self.backend = backend                   # the VGGT-Omega backend (cross.cv.pose_est_ff), shared with the back end
        self.rgb_transform = rgb_transform       # the back end's image transform (same tensors: token-cache hits)
        self.depth_transform = depth_transform
        self.metric = metric_model               # DA3 metric depth (scale prior of the mono mode), or None
        self.T_rl = None if T_right_in_left is None else np.asarray(T_right_in_left, dtype=np.float64).copy()
        self.depth_every = int(depth_every)      # frames between learned-depth observations (0: config.scale.interval)
        self.last_depth_index = -10 ** 9
        self.frames = {}                         # index -> {"rgb", "right", "pass_depth", "conf"} of requested frames
        self._sgbm = None                        # stereo matcher (vgio_stereo_source depth)
        self._last_sgbm = None                   # (index, depth) of the last stereo depth (corners detected later)
        self._pp = None                          # principal-point correction of the passes' camera poses
        self.stats = dict(own_calls=0, backend_measurements=0, depth_priors=0, model_seconds=0.0)

    # ------------------------------------------------------------------ frames
    def add_frame(self, index, rgb, right=None):
        """The images of a requested frame (it is the current view of its request, and m or the keyframe of later ones)."""
        self.frames[int(index)] = {"rgb": rgb, "right": right, "pass_depth": None, "conf": None}

    def prune(self, keep_from):
        """Forget the frames no later request can reference (indices below keep_from)."""
        if keep_from is None:
            return
        for k in [k for k in self.frames if k < keep_from]:
            del self.frames[k]

    def anchor(self, req):
        """The temporal anchor of the back end's forward pass for a request (cross.core.system: frontend_anchor): the
        last measured frame with the frontend's motion since then (and the keyframe as one more view), or None."""
        if req is None or not req.get("anchor") or req.get("m") is None or req["m"] not in self.frames:
            return None
        out = {"token": req["token"], "rgb": self.frames[req["m"]]["rgb"], "T_prev_curr": req["T_prev_curr"],
               "metric": bool(req["metric"])}
        if req.get("kf") is not None and req["kf"] in self.frames:
            out["extra_rgb"] = [self.frames[req["kf"]]["rgb"]]      # the keyframe rides along (its pairs in the graph)
        return out

    # ------------------------------------------------------------------ the measurement
    def measure(self, req, backend_obs=None):
        """The visual measurement of a request: the back end's pass when it carried the request's anchor (backend_obs:
        cross.cv.pose_est_ff last_frontend_obs), else a pass of its own when the request allows one.  Returns the summary,
        or None when nothing was measured."""
        if backend_obs is not None and req.get("anchor") and req.get("m") is not None \
                and backend_obs.get("token") == req["token"]:
            self.stats["backend_measurements"] += 1
            return self._summarize(req, backend_obs, "backend")
        if req.get("own", True):
            return self._summarize(req, self._own_pass(req), "own")
        return None

    def _own_pass(self, req):
        """Forward pass [current, last measured (, keyframe or the frame measured before)] (the current image alone for
        the first measurement), and the stereo pair's right image when the baseline source needs it."""
        if self.backend is None:
            return None
        start = perf_counter()
        b, m = self.frames[req["index"]], req.get("m")
        views = [self.rgb_transform(b["rgb"])]
        if m is not None:
            views.append(self.rgb_transform(self.frames[m]["rgb"]))
        third = None
        if m is not None and req.get("kf") is not None:
            third = "kf"
            views.append(self.rgb_transform(self.frames[req["kf"]]["rgb"]))
        elif m is not None and req.get("m_prev") is not None:
            third = "m_prev"
            views.append(self.rgb_transform(self.frames[req["m_prev"]]["rgb"]))
        right = b["right"] if self.T_rl is not None and self._stereo_views() else None
        if right is not None:
            views.append(self.rgb_transform(right))                 # the stereo pair's right image, last
        images = torch.stack(views).float().to(self.backend.device if hasattr(self.backend, "device") else "cuda")
        if images.max() > 1.5:
            images = images / 255.0
        with torch.inference_mode():
            pred = self.backend.infer(images, n_depth=None)
        self.stats["own_calls"] += 1
        self.stats["model_seconds"] += perf_counter() - start
        out = {"c2w_curr": pred.c2w[0], "depth_curr": pred.depth[0], "conf_curr": _conf(pred, 0)}
        if m is not None:
            out.update(c2w_prev=pred.c2w[1], depth_prev=pred.depth[1], conf_prev=_conf(pred, 1))
        if third == "m_prev":
            out.update(depth_prev2=pred.depth[2], conf_prev2=_conf(pred, 2))
        elif third == "kf":
            out.update(depth_kf=pred.depth[2], conf_kf=_conf(pred, 2), c2w_kf=pred.c2w[2])
        if right is not None:
            out["c2w_right"] = pred.c2w[len(views) - 1]
        return out

    def _stereo_views(self):
        """The right image rides along in the own passes (only the baseline source needs it)."""
        return self.config.imu.vgio_stereo_source in ("baseline", "both")

    def _summarize(self, req, obs, source):
        b = int(req["index"])
        out = {"token": req["token"], "index": b, "source": source, "finite": False}
        if obs is None:
            return out
        obs = self._model_to_cam(obs, self.frames[b]["rgb"].shape[:2])
        if not np.isfinite(np.asarray(obs["c2w_curr"])).all():
            return out
        out["finite"] = True
        for key in ("c2w_curr", "c2w_prev", "c2w_kf", "c2w_right"):
            out[key] = None if obs.get(key) is None else np.asarray(obs[key], dtype=np.float64)
        frame = self.frames[b]
        depth_curr = obs["depth_curr"].float()
        conf_curr = obs.get("conf_curr")
        observed = self._learned_depth(frame["rgb"], depth_curr, b)
        out["da3"] = (float(observed.log_scale), float(np.sqrt(observed.variance))) \
            if observed is not None and observed.accepted else None
        out["stereo"], out["stereo_info"], sgbm = self._stereo_scale(obs, depth_curr, conf_curr, frame)
        self._last_sgbm = (b, sgbm)
        out["corner_z"] = self.corner_depth(b, req.get("corners"))
        out["log_median_depth"] = float(-np.log(max(_median_depth(depth_curr, conf_curr), 1e-6))) \
            if req.get("m") is None else None
        m = self.frames.get(req.get("m"))
        out["link"] = (float("nan"), float("inf"))
        if m is not None and m["pass_depth"] is not None and obs.get("depth_prev") is not None:
            out["link"] = _depth_ratio([(m["pass_depth"], m["conf"], obs["depth_prev"].float(), obs.get("conf_prev"))])
        kf = self.frames.get(req.get("kf"))
        out["link_kf"] = None
        if kf is not None and kf["pass_depth"] is not None and obs.get("depth_kf") is not None:
            out["link_kf"] = _depth_ratio([(kf["pass_depth"], kf["conf"], obs["depth_kf"].float(), obs.get("conf_kf"))])
        frame["pass_depth"], frame["conf"] = depth_curr, conf_curr
        return out

    def corner_depth(self, index, corners):
        """The stereo depth of frame `index` (its last measurement's) at the corners detected on it (pixel (u, v), the
        nearest pixel; 0 outside the image or without depth), or None without stereo depth."""
        if corners is None or self._last_sgbm is None or self._last_sgbm[0] != index or self._last_sgbm[1] is None:
            return None
        depth = self._last_sgbm[1]
        a = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
        h, w = depth.shape
        ui, vi = np.round(a[:, 0]).astype(int), np.round(a[:, 1]).astype(int)
        inside = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        z = np.zeros(len(a))
        z[inside] = depth[vi[inside], ui[inside]]
        return z

    def _model_to_cam(self, obs, hw):
        """The pass's camera poses in the calibrated camera frame: VGGT-Omega places the principal point at the image
        centre (the back end's correction, cross.cv.pose_est_ff.principal_point_rotation; KITTI ~0.9 deg, OpenLORIS
        ~1.4 deg), which would otherwise act as a camera-IMU rotation error of that size."""
        if obs is None or not self.config.imu.vgio_pp_correction:
            return obs
        if self._pp is None:
            from cross.cv.pose_est_ff import principal_point_rotation
            h, w = hw
            self._pp = np.eye(4)
            self._pp[:3, :3] = principal_point_rotation(self.K, w, h).T
        out = dict(obs)
        for k in ("c2w_curr", "c2w_prev", "c2w_kf", "c2w_right"):
            if out.get(k) is not None:
                out[k] = np.asarray(out[k], dtype=np.float64) @ self._pp
        return out

    # ------------------------------------------------------------------ metric scale of a pass
    def _learned_depth(self, rgb, pass_depth, index):
        """Learned metric depth of this frame against a depth map of it (same pixels): a scale observation, at most
        every depth_every frames."""
        every = self.depth_every if self.depth_every else self.config.scale.interval
        if self.metric is None or not self.config.imu.depth_prior or index - self.last_depth_index < every:
            return None
        self.last_depth_index = index
        t0 = perf_counter()
        metric = self.metric.predict_metric(rgb, self.K, rgb.shape[:2])
        self.stats["t_da3"] = self.stats.get("t_da3", 0.0) + perf_counter() - t0
        observed = self._metric_depth_scale(metric, pass_depth, self.config.scale)
        if observed is not None:
            self.stats["depth_priors"] += 1
        return observed

    def _metric_depth_scale(self, metric, pass_depth, scale_config):
        """A metric depth map of the current image (full resolution; learned or stereo) against the pass's depth of it:
        the log metres per unit of the pass (cross.mono.scale.observe_scale), or None if the sizes differ."""
        metric_v = self.depth_transform(torch.from_numpy(np.asarray(metric, dtype=np.float32))[None])[0].numpy()
        source = pass_depth.float().cpu().numpy()
        if metric_v.shape != source.shape:
            return None
        metric_v[~np.isfinite(metric_v)] = 0.0
        return observe_scale(metric_v, source, None, scale_config)

    def _stereo_scale(self, obs, depth, conf, frame):
        """The metric scale of a pass from the current stereo pair: (log metres per pass unit, its std) of the source
        vgio_stereo_source, or None; a log of both sources; the stereo depth map (depth source) or None.  Neither
        source has a bias state: the rig is calibrated."""
        if self.T_rl is None or obs is None:
            return None, None, None
        source = self.config.imu.vgio_stereo_source
        info, out, sgbm = {}, {}, None
        if source in ("depth", "both"):
            out["depth"], info["depth"], sgbm = self._stereo_depth_scale(depth, frame["rgb"], frame["right"])
        if source in ("baseline", "both"):
            out["baseline"], info["baseline"] = self._stereo_baseline_scale(obs, depth, conf)
        use = out.get("baseline" if source == "baseline" else "depth")
        self.stats["stereo_used"] = self.stats.get("stereo_used", 0) + int(use is not None)
        if use is not None:
            info |= {"log": round(use[0], 4), "std": round(use[1], 4)}
        return use, info, sgbm

    def _stereo_depth_scale(self, pass_depth, rgb, right):
        """Classical stereo depth (SGBM) of the current pair against the pass's depth of the left image, as learned depth
        is in the monocular case (cross.mono.scale.observe_scale: tiles, inliers; its std, floored at vgio_stereo_std).
        Pixels with less than vgio_stereo_min_disparity px of disparity are left out (their depth is mostly noise).
        Returns (observation or None, info, the stereo depth map or None)."""
        if right is None:
            return None, {"ok": False, "reason": "no right image"}, None
        from cross.dataloader.stereo_loader import SGBMDepth
        from cross.mono.config import ScaleConfig
        ic = self.config.imu
        if self._sgbm is None:
            self._sgbm = SGBMDepth(rgb.shape[1])
        t0 = perf_counter()
        fx, nb = float(self.K[0, 0]), float(np.linalg.norm(self.T_rl[:3, 3]))
        metric = self._sgbm(rgb, np.asarray(right), fx, nb)
        metric[metric > fx * nb / max(ic.vgio_stereo_min_disparity, 1e-3)] = 0.0
        self.stats["t_sgbm"] = self.stats.get("t_sgbm", 0.0) + perf_counter() - t0
        o = self._metric_depth_scale(metric, pass_depth, ScaleConfig(observation_std_floor=ic.vgio_stereo_std))
        if o is None:
            return None, {"ok": False, "reason": "shape"}, metric
        info = {"ok": bool(o.accepted), "log": round(float(o.log_scale), 4), "std": round(float(np.sqrt(o.variance)), 4)
                if np.isfinite(o.variance) else None, "pixels": int(o.pixels), "mad": round(float(o.log_mad), 3),
                "inl": round(float(o.inlier_fraction), 3)}
        if not o.accepted:
            self.stats["stereo_depth_rejected"] = self.stats.get("stereo_depth_rejected", 0) + 1
            return None, info, metric
        return (float(o.log_scale), float(np.sqrt(o.variance))), info, metric

    def _stereo_baseline_scale(self, obs, depth, conf):
        """The right image as one more view of the pass: its centre in the left camera's frame along the calibrated
        baseline, against the baseline's length.  Not used when the pass's rotation between the two cameras or its
        baseline direction disagrees with the calibration (a wrongly registered right view carries no scale).  The std
        grows with the scene depth over the baseline, both in the pass's units (scale-free).  VGGT-Omega's baseline is
        biased by scene (KITTI 07: 13 % too long against the pass's own translations, KITTI 01: 30 %; OpenLORIS T265
        2-7 % short), while its depths agree with its translations (cross_mono_ff_vgio analysis: 0.99-1.02)."""
        if obs.get("c2w_right") is None:
            return None, {"ok": False, "reason": "no right view"}
        ic = self.config.imu
        T = inverse(np.asarray(obs["c2w_curr"], dtype=np.float64)) @ np.asarray(obs["c2w_right"], dtype=np.float64)
        b = self.T_rl[:3, 3]
        nb = float(np.linalg.norm(b))
        t = T[:3, 3]
        along = float(t @ b) / nb
        cos = along / max(float(np.linalg.norm(t)), 1e-12)
        rot = _angle_deg(self.T_rl[:3, :3].T @ T[:3, :3])
        d_over_b = _median_depth(depth, conf) / max(along, 1e-12)
        info = {"rot_deg": round(rot, 3), "cos": round(cos, 4), "d_over_b": round(float(d_over_b), 2)}
        if along <= 0 or rot > ic.vgio_stereo_rot_gate_deg or cos < ic.vgio_stereo_dir_cos or not np.isfinite(d_over_b):
            self.stats["stereo_baseline_rejected"] = self.stats.get("stereo_baseline_rejected", 0) + 1
            return None, info | {"ok": False}
        std = float(np.hypot(ic.vgio_stereo_std, ic.vgio_stereo_depth_k * d_over_b))
        log_obs = float(np.log(nb) - np.log(along))
        return (log_obs, std), info | {"ok": True, "log": round(log_obs, 4), "std": round(std, 4)}


def _angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def _conf(pred, i):
    return None if pred.depth_conf is None else pred.depth_conf[i]


def _median_depth(depth, conf=None):
    d = depth.float()
    mask = torch.isfinite(d) & (d > 0)
    if conf is not None:
        cf = conf.float()
        mask &= cf >= torch.quantile(cf[mask].float(), 0.5) if mask.any() else mask
    return float(d[mask].median()) if mask.any() else float("nan")


def _depth_ratio(pairs):
    """Median of chain / pass depth over the confident pixels of each shared image (chain depth, chain confidence, pass
    depth, pass confidence), pooled, and its robust log spread."""
    logs = []
    for chain_depth, chain_conf, pass_depth, pass_conf in pairs:
        a, b = chain_depth.float(), pass_depth.float().to(chain_depth.device)
        if a.shape != b.shape:
            continue
        mask = torch.isfinite(a) & torch.isfinite(b) & (a > 0) & (b > 0)
        for cf in (chain_conf, pass_conf):
            if cf is not None and mask.any():
                cf = cf.float().to(a.device)
                mask &= cf >= torch.quantile(cf[mask], 0.3)
        if mask.sum() >= 100:
            logs.append(torch.log(a[mask]) - torch.log(b[mask]))
    if not logs:
        return float("nan"), float("inf")
    r = torch.cat(logs)
    center = r.median()
    spread = float(1.4826 * (r - center).abs().median())
    return float(torch.exp(center)), spread
