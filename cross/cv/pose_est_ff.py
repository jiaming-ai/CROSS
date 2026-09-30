"""Feed-forward relative pose estimation with stereo scale anchors.

`PoseEstFeedForward` replaces the classical detect/match/PnP relative pose module of
CROSS with one multi-view forward pass of a learned geometry model (VGGT-Omega or
Depth Anything 3).  All retrieved reference keyframes and the current frame are
placed in one context so that they share a single similarity gauge; the current
stereo pair (and optionally stored right images of the references, and the previous
frame connected by odometry) fixes the metric scale of that gauge.

Outputs follow the CROSS `PoseEst` contract: for each reference keyframe, the pose of
the current camera expressed in the reference camera frame (T_ref_cam), a validity
mask and a scalar confidence.  The confidence is a geometric covisibility score
computed from the predicted depth maps (fraction of reference pixels that reproject
into the current view with a consistent depth), which plays the role of the PnP
inlier count.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pypose as pp
import torch
import torch.nn.functional as F
from loguru import logger

from cross.core.config import FeedForwardConfig, FFBackend
from cross.cv.stereo_scale import ScaleAnchor, ScaleEstimate, estimate_scale, invert_poses, scale_camera_centers
from cross.utils.profile import timeit


@dataclass
class FFPrediction:
    """Raw output of one forward pass (all views)."""
    c2w: np.ndarray                     # (S, 4, 4) camera-to-world, model gauge
    K: np.ndarray                       # (S, 3, 3) intrinsics at processed resolution
    depth: torch.Tensor                 # (S, H, W) model-gauge depth
    depth_conf: torch.Tensor            # (S, H, W) or None
    hw: tuple


class _Backend:
    def infer(self, images: torch.Tensor) -> FFPrediction:
        raise NotImplementedError


class _VGGTOmegaBackend(_Backend):
    def __init__(self, checkpoint: str, device: str, half_weights: bool = True):
        from vggt_omega.models import VGGTOmega
        from vggt_omega.utils.pose_enc import encoding_to_camera
        self._decode = encoding_to_camera
        self.device = device
        ckpt = Path(checkpoint)
        if not ckpt.is_file():
            raise FileNotFoundError(f"VGGT-Omega checkpoint not found: {ckpt}")
        self.model = VGGTOmega().eval()
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if half_weights:
            # the aggregator already runs under autocast; storing its weights in bf16 halves memory
            self.model.aggregator.to(self.dtype)
        self.model = self.model.to(device)

    @torch.inference_mode()
    def infer(self, images: torch.Tensor) -> FFPrediction:
        images = images.to(self.device)
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            pred = self.model(images)
        hw = tuple(images.shape[-2:])
        extr, intr = self._decode(pred["pose_enc"], hw)          # (1,S,3,4) w2c, (1,S,3,3)
        w2c = extr[0].float().cpu().numpy().astype(np.float64)
        c2w = invert_poses(_to_h(w2c))
        depth = pred["depth"][0].float()                         # (S,H,W,1) or (S,H,W)
        if depth.dim() == 4:
            depth = depth[..., 0]
        conf = pred.get("depth_conf")
        if conf is not None:
            conf = conf[0].float()
            if conf.dim() == 4:
                conf = conf[..., 0]
        return FFPrediction(c2w=c2w, K=intr[0].float().cpu().numpy(), depth=depth, depth_conf=conf, hw=hw)


class _DA3Backend(_Backend):
    def __init__(self, checkpoint: str, device: str, process_res: int = 504):
        from depth_anything_3.api import DepthAnything3
        ckpt = Path(checkpoint)
        if not (ckpt / "model.safetensors").is_file():
            raise FileNotFoundError(f"DA3 checkpoint not found under {ckpt}")
        self.device = device
        self.model = DepthAnything3.from_pretrained(str(ckpt)).to(device).eval()
        self.process_res = process_res

    @torch.inference_mode()
    def infer(self, images: torch.Tensor) -> FFPrediction:
        # DA3 preprocesses from uint8 arrays; keep the first view as the reference.
        arr = (images.clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        pred = self.model.inference(
            list(arr), process_res=self.process_res, process_res_method="upper_bound_resize",
            ref_view_strategy="first",
        )
        w2c = np.asarray(pred.extrinsics, dtype=np.float64)
        c2w = invert_poses(_to_h(w2c))
        depth = torch.from_numpy(np.asarray(pred.depth, dtype=np.float32)).to(self.device)
        conf = None if pred.conf is None else torch.from_numpy(np.asarray(pred.conf, dtype=np.float32)).to(self.device)
        return FFPrediction(c2w=c2w, K=np.asarray(pred.intrinsics, dtype=np.float64), depth=depth,
                            depth_conf=conf, hw=tuple(depth.shape[-2:]))


def _to_h(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    if T.shape[-2:] == (4, 4):
        return T
    out = np.zeros((*T.shape[:-2], 4, 4), dtype=np.float64)
    out[..., :3, :4] = T
    out[..., 3, 3] = 1.0
    return out


def covisibility_scores(
    pred: FFPrediction,
    src_indices: List[int],
    dst_index: int,
    grid: int = 48,
    rel_depth_tol: float = 0.15,
    min_conf_quantile: float = 0.3,
) -> np.ndarray:
    """Geometric consistency between each source view and the destination view.

    For every source view, a regular grid of pixels is unprojected with the predicted
    depth, transformed into the destination camera with the predicted relative pose,
    and projected with the destination intrinsics.  A pixel is an "inlier" when it
    lands inside the destination image and its transformed depth agrees with the
    destination depth map within `rel_depth_tol`.  The score is the inlier fraction
    over confident source pixels, i.e. a learned-geometry analogue of the PnP inlier
    ratio; it is low for wrongly retrieved (non-overlapping) references and for badly
    registered views.
    """
    device = pred.depth.device
    S, H, W = pred.depth.shape
    ys = torch.linspace(0, H - 1, grid, device=device)
    xs = torch.linspace(0, W - 1, grid, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    gy, gx = gy.reshape(-1), gx.reshape(-1)
    norm = torch.stack([gx / (W - 1) * 2 - 1, gy / (H - 1) * 2 - 1], dim=-1).view(1, 1, -1, 2)

    c2w = torch.from_numpy(pred.c2w).to(device=device, dtype=torch.float32)
    K = torch.from_numpy(pred.K).to(device=device, dtype=torch.float32)
    dst_depth = pred.depth[dst_index][None, None]
    dst_conf = pred.depth_conf[dst_index][None, None] if pred.depth_conf is not None else None
    scores = []
    for s in src_indices:
        d = F.grid_sample(pred.depth[s][None, None], norm, align_corners=True).view(-1)
        valid = torch.isfinite(d) & (d > 1e-6)
        if pred.depth_conf is not None:
            c = F.grid_sample(pred.depth_conf[s][None, None], norm, align_corners=True).view(-1)
            thr = torch.quantile(pred.depth_conf[s].flatten()[:: max(1, (H * W) // 20000)], min_conf_quantile)
            valid &= c >= thr
        if valid.sum() < 16:
            scores.append(0.0)
            continue
        Ks = K[s]
        x = (gx - Ks[0, 2]) / Ks[0, 0] * d
        y = (gy - Ks[1, 2]) / Ks[1, 1] * d
        P = torch.stack([x, y, d, torch.ones_like(d)], dim=-1)          # (N,4) in src cam
        T = torch.linalg.inv(c2w[dst_index]) @ c2w[s]                    # src cam -> dst cam
        Pd = (T @ P.T).T[:, :3]
        z = Pd[:, 2]
        Kd = K[dst_index]
        u = Kd[0, 0] * Pd[:, 0] / z.clamp(min=1e-6) + Kd[0, 2]
        v = Kd[1, 1] * Pd[:, 1] / z.clamp(min=1e-6) + Kd[1, 2]
        inside = (z > 1e-6) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
        un = torch.stack([u / (W - 1) * 2 - 1, v / (H - 1) * 2 - 1], dim=-1).view(1, 1, -1, 2)
        zd = F.grid_sample(dst_depth, un, align_corners=True).view(-1)
        consistent = inside & ((z - zd).abs() <= rel_depth_tol * zd.clamp(min=1e-6))
        scores.append(float((consistent & valid).sum()) / float(valid.sum()))
    return np.asarray(scores, dtype=np.float32)


class PoseEstFeedForward:
    """Relative pose estimation with a feed-forward geometry model and stereo scale anchors."""

    def __init__(self, device: str, config: FeedForwardConfig, T_right_in_left: Optional[np.ndarray] = None):
        self.device = device
        self.config = config
        self.T_right_in_left = None if T_right_in_left is None else np.asarray(T_right_in_left, dtype=np.float64)
        t0 = time.perf_counter()
        if config.backend == FFBackend.VGGT_OMEGA:
            self.backend = _VGGTOmegaBackend(config.checkpoint, device, half_weights=config.half_precision_weights)
        elif config.backend == FFBackend.DA3:
            self.backend = _DA3Backend(config.checkpoint, device, process_res=config.da3_process_res)
        else:
            raise ValueError(f"Unknown feed-forward backend {config.backend}")
        logger.info(f"Loaded {config.backend.value} from {config.checkpoint} in {time.perf_counter() - t0:.1f}s")
        self.last_info: dict = {}
        self.last_scale: Optional[ScaleEstimate] = None

    def set_stereo_calibration(self, T_right_in_left: np.ndarray):
        self.T_right_in_left = np.asarray(T_right_in_left, dtype=np.float64)

    # ------------------------------------------------------------------ #
    @timeit
    @torch.inference_mode()
    def estimate_pose(
        self,
        ref_image: torch.Tensor,
        ref_depth: Optional[torch.Tensor],
        curr_image: torch.Tensor,
        curr_depth: Optional[torch.Tensor],
        curr_image_right: Optional[torch.Tensor] = None,
        ref_images_right: Optional[List[Optional[torch.Tensor]]] = None,
        odom_anchor: Optional[dict] = None,
        ref_rel_poses: Optional[List[Tuple[int, int, np.ndarray]]] = None,
        **kwargs,
    ):
        """Estimate T_ref_cam for every reference image in one forward pass.

        Args:
            ref_image: (B, 3, H, W) reference (left) images in [0, 1]
            ref_depth: unused (kept for interface compatibility)
            curr_image: (3, H, W) current left image
            curr_depth: unused
            curr_image_right: (3, H, W) current right image (stereo anchor) or None
            ref_images_right: list of B entries, each (3, H, W) right image or None;
                at most `config.n_ref_anchors` of them are used as extra anchors
            odom_anchor: {"image": (3,H,W) previous left image, "T_prev_curr": (4,4) metric}
                optional temporal anchor from odometry
        Returns:
            poses: pp.SE3 (B_valid, 7) T_ref_cam (metric), valid_masks (B,) np.bool, confidences (B_valid,)
        """
        cfg = self.config
        t_start = time.perf_counter()
        B = int(ref_image.shape[0])
        views = [curr_image] + [ref_image[i] for i in range(B)]
        view_tags = ["curr_L"] + [f"ref{i}_L" for i in range(B)]
        anchors: List[ScaleAnchor] = []

        # temporal (odometry) anchor
        if odom_anchor is not None and cfg.use_odom_anchor:
            T_pc = np.asarray(odom_anchor["T_prev_curr"], dtype=np.float64)
            if np.linalg.norm(T_pc[:3, 3]) >= cfg.odom_anchor_min_translation:
                views.append(odom_anchor["image"])
                view_tags.append("prev_L")
                anchors.append(ScaleAnchor(idx_a=len(views) - 1, idx_b=0, T_ab=T_pc, kind="odom",
                                           weight=cfg.odom_anchor_weight))

        # stereo anchors: current pair first, then stored right images of the best references
        if self.T_right_in_left is not None:
            if curr_image_right is not None and cfg.use_curr_anchor:
                views.append(curr_image_right)
                view_tags.append("curr_R")
                anchors.append(ScaleAnchor(idx_a=0, idx_b=len(views) - 1, T_ab=self.T_right_in_left, kind="stereo"))
            if ref_images_right is not None and cfg.n_ref_anchors > 0:
                n_added = 0
                for i, img_r in enumerate(ref_images_right):
                    if img_r is None:
                        continue
                    views.append(img_r)
                    view_tags.append(f"ref{i}_R")
                    anchors.append(ScaleAnchor(idx_a=1 + i, idx_b=len(views) - 1, T_ab=self.T_right_in_left, kind="stereo"))
                    n_added += 1
                    if n_added >= cfg.n_ref_anchors:
                        break

        # map anchors: known metric relative poses between pairs of references (from the map)
        if ref_rel_poses and cfg.use_map_anchors:
            for i, j, T_ij in ref_rel_poses:
                if 0 <= i < B and 0 <= j < B and i != j:
                    anchors.append(ScaleAnchor(idx_a=1 + i, idx_b=1 + j, T_ab=np.asarray(T_ij, dtype=np.float64),
                                               kind="map", weight=cfg.map_anchor_weight))

        images = torch.stack([v.to(self.device) for v in views], dim=0).float()
        if images.max() > 1.5:
            images = images / 255.0

        t_model = time.perf_counter()
        pred = self.backend.infer(images)
        torch.cuda.synchronize()
        t_model = time.perf_counter() - t_model

        # ---- metric scale ----
        scale_est = estimate_scale(
            pred.c2w, anchors, method=cfg.scale_method,
            max_rot_err_deg=cfg.anchor_max_rot_err_deg, min_dir_cos=cfg.anchor_min_dir_cos,
            weight_by_baseline=cfg.anchor_weight_by_baseline,
        )
        self.last_scale = scale_est
        if not scale_est.valid:
            logger.debug(f"FF pose est: no valid scale anchor ({scale_est.n_anchors} anchors)")
            self.last_info = {"valid": False, "scale": scale_est.to_dict(), "t_model": t_model,
                              "n_views": len(views), "view_tags": view_tags}
            return torch.empty((0, 7)), np.zeros(B, dtype=bool), torch.empty(0)

        # calibrated metric-scale correction (the estimator's translations divided by their measured/true ratio)
        c2w_metric = scale_camera_centers(pred.c2w, scale_est.scale / float(getattr(self, "metric_scale_correction", 1.0) or 1.0), origin_index=0)

        # ---- per-reference confidence (covisibility / geometric consistency) ----
        ref_idx = list(range(1, 1 + B))
        covis = covisibility_scores(pred, ref_idx, 0, grid=cfg.covis_grid, rel_depth_tol=cfg.covis_depth_tol)
        if cfg.covis_symmetric:
            covis_back = np.asarray([covisibility_scores(pred, [0], r, grid=cfg.covis_grid,
                                                         rel_depth_tol=cfg.covis_depth_tol)[0] for r in ref_idx])
            covis = np.minimum(covis, covis_back)

        # ---- relative poses T_ref_cam = X_ref^-1 X_curr ----
        poses, confs, valid = [], [], []
        for i in range(B):
            T_ref_cam = invert_poses(c2w_metric[1 + i]) @ c2w_metric[0]
            dist = float(np.linalg.norm(T_ref_cam[:3, 3]))
            ok = (covis[i] >= cfg.min_covis) and (dist <= cfg.max_rel_distance)
            valid.append(bool(ok))
            if ok:
                poses.append(pp.from_matrix(torch.from_numpy(T_ref_cam).float(), ltype=pp.SE3_type))
                confs.append(float(covis[i]))
        valid = np.asarray(valid, dtype=bool)
        if valid.sum() > 0:
            poses = torch.stack([p.tensor() for p in poses], dim=0)
            poses = pp.SE3(poses)
            confs = torch.tensor(confs, dtype=torch.float32)
        else:
            poses, confs = torch.empty((0, 7)), torch.empty(0)

        self.last_info = {
            "valid": True,
            "scale": scale_est.to_dict(),
            "covis": covis.tolist(),
            "valid_masks": valid.tolist(),
            "t_model": t_model,
            "t_total": time.perf_counter() - t_start,
            "n_views": len(views),
            "view_tags": view_tags,
            "c2w_metric": c2w_metric,
        }
        logger.debug(
            f"FF pose est: {len(views)} views, scale={scale_est.scale:.3f} "
            f"({scale_est.n_used}/{scale_est.n_anchors} anchors, logstd={scale_est.log_std:.3f}), "
            f"covis={np.round(covis, 2).tolist()}, model {t_model * 1e3:.0f} ms"
        )
        return poses, valid, confs

    @property
    def scale_rel_std(self) -> float:
        """Relative (multiplicative) std of the last scale estimate; used to inflate translation std."""
        if self.last_scale is None or not self.last_scale.valid:
            return 1.0
        return float(self.last_scale.log_std)
