"""Relative poses of retrieved keyframes from feed-forward two-view passes (DA3 or VGGT-Omega).

Each retrieved keyframe is paired with the current image in its own (batched) two-view pass; a joint pass over all
references lets a single non-overlapping reference corrupt the others. The reference's stored metric depth fixes the
scale of its relative pose, so a pose to a keyframe of a loaded map is expressed in that map's scale, independent of
the current session's learned-scale bias. A symmetric
covisibility score (predicted depth of one view reprojected into the other with consistent depth) replaces the PnP
inlier ratio as the verification and confidence. CROSS's hypothesis filter and delayed commitment are unchanged.
"""

import numpy as np
import torch
import torch.nn.functional as F


def _inverse(T):
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def _homogeneous(T):
    T = np.asarray(T, dtype=np.float64)
    if T.shape[-2:] == (4, 4):
        return T
    out = np.zeros((*T.shape[:-2], 4, 4))
    out[..., :3, :4] = T
    out[..., 3, 3] = 1.
    return out


class _DA3:
    """Batched two-view passes through the raw DA3 network (ImageNet-normalized, sides multiple of 14)."""

    def __init__(self, checkpoint, device):
        from depth_anything_3.api import DepthAnything3
        self.model = DepthAnything3.from_pretrained(checkpoint).to(device).eval()
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]

    def _prepare(self, rgb, resolution):
        import cv2
        h, w = rgb.shape[:2]
        ratio = resolution / max(h, w)
        size = (max(14, round(w * ratio / 14) * 14), max(14, round(h * ratio / 14) * 14))
        image = torch.as_tensor(cv2.resize(rgb, size, interpolation=cv2.INTER_AREA), device=self.device)
        return (image.permute(2, 0, 1).float() / 255. - self.mean) / self.std

    @torch.inference_mode()
    def __call__(self, pairs, resolution):
        """pairs: list of (reference_rgb, current_rgb); returns per-pair (c2w[2], K[2], depth[2], conf[2])."""
        batch = torch.stack([torch.stack([self._prepare(a, resolution), self._prepare(b, resolution)]) for a, b in pairs])
        result = self.model(batch, ref_view_strategy="first", export_feat_layers=[])
        depth = result["depth"].float()
        depth = depth[..., 0] if depth.dim() == 5 else depth
        conf = result["depth_conf"].float() if "depth_conf" in result else None
        extrinsics = result["extrinsics"].float().cpu().numpy()
        intrinsics = result["intrinsics"].float().cpu().numpy().astype(np.float64)
        return [(np.stack([_inverse(T) for T in _homogeneous(extrinsics[i])]), intrinsics[i], depth[i],
                 None if conf is None else conf[i]) for i in range(len(pairs))]


class _VGGTOmega:
    def __init__(self, checkpoint, device):
        from vggt_omega.models import VGGTOmega
        from vggt_omega.utils.pose_enc import encoding_to_camera
        self.decode = encoding_to_camera
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and isinstance(state.get("model"), dict):
            state = state["model"]
        state = {k.removeprefix("module."): v for k, v in state.items()
                 if not k.removeprefix("module.").startswith(("aggregator.gray_embed", "scale_head."))}
        self.model = VGGTOmega().eval()
        self.model.load_state_dict(state)
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model.aggregator.to(self.dtype)
        self.model = self.model.to(device)
        self.device = device

    @torch.inference_mode()
    def __call__(self, pairs, resolution):
        import cv2
        h, w = pairs[0][0].shape[:2]
        width = resolution
        height = max(16, int(round(h * resolution / w / 16)) * 16)
        prepare = lambda x: torch.from_numpy(cv2.resize(x, (width, height), interpolation=cv2.INTER_AREA)).permute(2, 0, 1).float() / 255.
        images = torch.stack([torch.stack([prepare(a), prepare(b)]) for a, b in pairs]).to(self.device)
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            pred = self.model(images)
        extrinsics, intrinsics = self.decode(pred["pose_enc"], (height, width))
        depth = pred["depth"].float()
        depth = depth[..., 0] if depth.dim() == 5 else depth
        conf = pred.get("depth_conf")
        if conf is not None:
            conf = conf.float()
            conf = conf[..., 0] if conf.dim() == 5 else conf
        extrinsics = extrinsics.float().cpu().numpy()
        intrinsics = intrinsics.float().cpu().numpy().astype(np.float64)
        return [(np.stack([_inverse(T) for T in _homogeneous(extrinsics[i])]), intrinsics[i], depth[i],
                 None if conf is None else conf[i]) for i in range(len(pairs))]


def covisibility(c2w, K, depth, conf, source, target, grid=48, tolerance=0.15, conf_quantile=0.3):
    """Fraction of confident source pixels that land in the target view with a consistent predicted depth."""
    _, H, W = depth.shape
    device = depth.device
    ys = torch.linspace(0, H - 1, grid, device=device)
    xs = torch.linspace(0, W - 1, grid, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    gy, gx = gy.reshape(-1), gx.reshape(-1)
    norm = torch.stack([gx / (W - 1) * 2 - 1, gy / (H - 1) * 2 - 1], -1).view(1, 1, -1, 2)
    d = F.grid_sample(depth[source][None, None], norm, align_corners=True).view(-1)
    valid = torch.isfinite(d) & (d > 1e-6)
    if conf is not None:
        c = F.grid_sample(conf[source][None, None], norm, align_corners=True).view(-1)
        valid &= c >= torch.quantile(conf[source].flatten()[::max(1, H * W // 20000)], conf_quantile)
    if int(valid.sum()) < 16:
        return 0.
    Kt = torch.as_tensor(K, device=device, dtype=torch.float32)
    x = (gx - Kt[source, 0, 2]) / Kt[source, 0, 0] * d
    y = (gy - Kt[source, 1, 2]) / Kt[source, 1, 1] * d
    points = torch.stack([x, y, d, torch.ones_like(d)], -1)
    T = torch.as_tensor(_inverse(c2w[target]) @ c2w[source], device=device, dtype=torch.float32)
    p = (T @ points.T).T[:, :3]
    z = p[:, 2]
    u = Kt[target, 0, 0] * p[:, 0] / z.clamp(min=1e-6) + Kt[target, 0, 2]
    v = Kt[target, 1, 1] * p[:, 1] / z.clamp(min=1e-6) + Kt[target, 1, 2]
    inside = (z > 1e-6) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
    uv = torch.stack([u / (W - 1) * 2 - 1, v / (H - 1) * 2 - 1], -1).view(1, 1, -1, 2)
    zt = F.grid_sample(depth[target][None, None], uv, align_corners=True).view(-1)
    consistent = inside & ((z - zt).abs() <= tolerance * zt.clamp(min=1e-6))
    return float((consistent & valid).sum()) / float(valid.sum())


def depth_scale(stored, predicted, conf=None, conf_quantile=0.3, min_pixels=200):
    """Median log ratio stored/predicted depth at confident pixels, and its median absolute deviation."""
    import cv2
    h, w = predicted.shape
    stored = cv2.resize(np.asarray(stored, dtype=np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    mask = np.isfinite(stored) & (stored > 1e-3) & np.isfinite(predicted) & (predicted > 1e-6)
    if conf is not None:
        mask &= conf >= np.quantile(conf, conf_quantile)
    if mask.sum() < min_pixels:
        return None, None
    ratio = np.log(stored[mask] / predicted[mask])
    center = float(np.median(ratio))
    return center, float(np.median(np.abs(ratio - center)))


class FeedForwardRelativePose:
    """CROSS pose-estimator contract: T_ref_current for each retrieved keyframe, validity mask, confidences."""

    def __init__(self, backend="da3", checkpoint="depth-anything/DA3-LARGE-1.1", device="cuda", resolution=504,
                 min_covisibility=0.3, max_distance=10., max_log_scale_mad=0.25):
        self.backend_name, self.checkpoint, self.device = backend, checkpoint, device
        self.resolution, self.min_covisibility = resolution, min_covisibility
        self.max_distance, self.max_log_scale_mad = max_distance, max_log_scale_mad
        self.model = _VGGTOmega(checkpoint, device) if backend == "vggt_omega" else _DA3(checkpoint, device)
        self.last_stds = None
        self.last_pair_audit = []

    @staticmethod
    def rgb(image):
        x = image.detach().cpu()
        if x.dtype != torch.uint8:
            x = (x.float().clamp(0, 1) * 255).round().to(torch.uint8)
        x = x.numpy()
        return x.transpose(1, 2, 0) if x.shape[0] == 3 else x

    def estimate_pose(self, ref_image, ref_depth, curr_image, curr_depth, **kwargs):
        import cv2
        import pypose as pp
        if ref_depth is None:
            raise ValueError("Feed-forward retrieval needs the stored metric depth of every keyframe")
        current = self.rgb(curr_image)
        size = (current.shape[1], current.shape[0])
        # Independent (reference, current) passes, batched: a non-overlapping reference in a joint
        # multi-view pass corrupts the poses of the overlapping ones.
        pairs = [(cv2.resize(self.rgb(image), size, interpolation=cv2.INTER_AREA), current) for image in ref_image]
        predictions = self.model(pairs, self.resolution) if pairs else []
        poses, confidences, stds = [], [], []
        valid = np.zeros(len(ref_image), dtype=bool)
        self.last_pair_audit = []
        for i, (c2w, K, depth, conf) in enumerate(predictions):
            covis = min(covisibility(c2w, K, depth, conf, 0, 1), covisibility(c2w, K, depth, conf, 1, 0))
            log_scale, mad = depth_scale(ref_depth[i].detach().cpu().float().numpy().squeeze(), depth[0].cpu().numpy(),
                                         None if conf is None else conf[0].cpu().numpy())
            audit = dict(proposal_method="feed_forward", backend=self.backend_name, covisibility=covis,
                         log_scale=log_scale, log_scale_mad=mad, accepted=False)
            self.last_pair_audit.append(audit)
            if log_scale is None or mad > self.max_log_scale_mad:
                audit["reason"] = "inconsistent_depth_scale"
                continue
            if covis < self.min_covisibility:
                audit["reason"] = "low_covisibility"
                continue
            pose = _inverse(c2w[0]) @ c2w[1]        # T_ref_current
            pose[:3, 3] *= np.exp(log_scale)
            distance = float(np.linalg.norm(pose[:3, 3]))
            if not np.isfinite(pose).all() or distance > self.max_distance:
                audit["reason"] = "implausible_distance"
                continue
            audit.update(reason="accepted", accepted=True, distance_m=distance)
            poses.append(pose)
            confidences.append(float(np.clip(covis, .01, 1.)))
            # translation: base + range + depth-scale scatter; rotation grows as covisibility drops
            # (inflated 1.5x over the office probe's median errors, so a fused learned pose does not outweigh later
            # matcher-based corrections)
            sigma_t = 1.5 * (0.05 + 0.05 * distance + distance * mad)
            sigma_r = 1.5 * (0.02 + 0.04 * (1. - covis))
            stds.append([sigma_t] * 3 + [sigma_r] * 3)
            valid[i] = True
        self.last_stds = torch.as_tensor(np.array(stds, dtype=np.float32).reshape(-1, 6))
        if not poses:
            return pp.identity_SE3(0, device=self.device), valid, torch.empty(0)
        matrices = torch.as_tensor(np.array(poses), dtype=torch.float32, device=self.device)
        return pp.from_matrix(matrices, pp.SE3_type), valid, torch.tensor(confidences)


class FallbackFeedForwardRelativePose:
    """Keep a matcher-based estimator's accepted poses; ask the feed-forward model only about rejected references.

    Matching succeeds on most same-session and easy revisit pairs; the learned two-view model is reserved for the
    references it rejects (large viewpoint or appearance change), which bounds its GPU load in steady state.
    """

    def __init__(self, primary, feed_forward, scope="all"):
        self.primary, self.feed_forward = primary, feed_forward
        # which rejected references the model sees: "all", "map" (loaded-map references), or "relocalization"
        # (loaded-map references while the session is not yet joined to the map; bounds the model's load to the
        # relocalization window)
        if scope not in {"all", "map", "relocalization"}:
            raise ValueError("Feed-forward scope must be all, map or relocalization")
        self.scope = scope
        self.last_stds = None
        self.last_pair_audit = []

    def estimate_pose(self, ref_image, ref_depth, curr_image, curr_depth, **kwargs):
        import pypose as pp
        poses, valid, confidences = self.primary.estimate_pose(ref_image, ref_depth, curr_image, curr_depth, **kwargs)
        audits = list(getattr(self.primary, "last_pair_audit", None) or [dict() for _ in range(len(ref_image))])
        stds = self.primary.last_stds
        by_index = {int(i): (poses[k], confidences[k], stds[k]) for k, i in enumerate(np.flatnonzero(valid))}
        candidates = ~valid
        if self.scope != "all":
            candidates &= np.asarray(kwargs.get("ref_loaded", [False] * len(ref_image)), dtype=bool)
        if self.scope == "relocalization":
            candidates &= bool(kwargs.get("session_unanchored", False))
        rejected = np.flatnonzero(candidates)
        if len(rejected):
            index = torch.as_tensor(rejected, device=ref_image.device if torch.is_tensor(ref_image) else "cpu")
            ff_poses, ff_valid, ff_conf = self.feed_forward.estimate_pose(
                ref_image[index], ref_depth[index.to(ref_depth.device)], curr_image, curr_depth)
            for k, (i, audit) in enumerate(zip(rejected, self.feed_forward.last_pair_audit)):
                audits[i] = dict(audits[i], feed_forward=audit, reason=audit["reason"] if audit["accepted"] else audits[i].get("reason"))
            for k, i in enumerate(rejected[ff_valid]):
                by_index[int(i)] = (ff_poses[k], ff_conf[k], self.feed_forward.last_stds[k])
                valid[i] = True
        self.last_pair_audit = audits
        order = sorted(by_index)
        if not order:
            self.last_stds = torch.empty((0, 6))
            return pp.identity_SE3(0, device=self.feed_forward.device), valid, torch.empty(0)
        device = self.feed_forward.device
        self.last_stds = torch.stack([torch.as_tensor(by_index[i][2]).float().cpu() for i in order])
        stacked = torch.stack([by_index[i][0].tensor().to(device) for i in order])
        return pp.SE3(stacked), valid, torch.stack([torch.as_tensor(by_index[i][1]).float().cpu() for i in order])
