from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

from cross.core.system import System


@dataclass
class _SemanticObservation:
    kf_id: int
    kf_pose: List[float]  # [tx,ty,tz,qx,qy,qz,qw]
    box_xyxy: List[float]  # [x0,y0,x1,y1]
    score: float
    mask: Optional[Dict[str, Any]]
    depth_median: Optional[float]
    pos_cam: Optional[List[float]]
    pos_world: Optional[List[float]]
    pos_sigma: Optional[float]
    track_id: Optional[int] = None


@dataclass
class _SemanticTrack:
    track_id: int
    center_world: np.ndarray  # (3,)
    radius: float
    fused_score: float
    obs_count: int
    kf_ids: List[int]

    def update(self, obs_center: np.ndarray, obs_score: float, obs_radius: float, kf_id: int):
        w_prev = max(self.fused_score, 1e-6)
        w_new = max(obs_score, 1e-6)
        alpha = w_new / (w_prev + w_new)
        self.center_world = (1.0 - alpha) * self.center_world + alpha * obs_center
        self.radius = max(self.radius, float(obs_radius))
        self.fused_score = 1.0 - (1.0 - float(self.fused_score)) * (1.0 - float(obs_score))
        self.obs_count += 1
        if not self.kf_ids or self.kf_ids[-1] != kf_id:
            self.kf_ids.append(kf_id)


def _to_pil_rgb(img: torch.Tensor) -> Image.Image:
    if not isinstance(img, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor image, got {type(img)}")
    if img.ndim != 3 or img.shape[0] != 3:
        raise ValueError(f"Expected image shaped (3,H,W), got {tuple(img.shape)}")
    img_cpu = img.detach().float().cpu().clamp(0.0, 1.0)
    arr = (img_cpu.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a: (N,4), b: (M,4)
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), dtype=torch.float32)
    a = a.float()
    b = b.float()
    ax0, ay0, ax1, ay1 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx0, by0, bx1, by1 = b[:, 0].unsqueeze(0), b[:, 1].unsqueeze(0), b[:, 2].unsqueeze(0), b[:, 3].unsqueeze(0)
    ix0 = torch.maximum(ax0, bx0)
    iy0 = torch.maximum(ay0, by0)
    ix1 = torch.minimum(ax1, bx1)
    iy1 = torch.minimum(ay1, by1)
    iw = torch.clamp(ix1 - ix0, min=0.0)
    ih = torch.clamp(iy1 - iy0, min=0.0)
    inter = iw * ih
    area_a = torch.clamp(ax1 - ax0, min=0.0) * torch.clamp(ay1 - ay0, min=0.0)
    area_b = torch.clamp(bx1 - bx0, min=0.0) * torch.clamp(by1 - by0, min=0.0)
    union = area_a + area_b - inter + 1e-6
    return inter / union


def _nms_xyxy(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float) -> List[int]:
    if boxes.numel() == 0:
        return []
    order = torch.argsort(scores, descending=True)
    keep: List[int] = []
    while order.numel() > 0:
        i = int(order[0].item())
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        ious = _box_iou_xyxy(boxes[i].unsqueeze(0), boxes[rest]).squeeze(0)
        order = rest[ious <= iou_thresh]
    return keep


def _estimate_object_center_cam(
    depth_1hw: torch.Tensor,
    camera: Any,
    box_xyxy: torch.Tensor,
    mask_hw: Optional[torch.Tensor],
    *,
    max_points: int = 1500,
    min_points: int = 80,
    max_depth: float = 30.0,
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[float]]:
    if depth_1hw is None:
        return None, None, None
    if depth_1hw.ndim == 3:
        depth_hw = depth_1hw[0]
    else:
        depth_hw = depth_1hw

    depth_hw = depth_hw.detach().float().cpu()
    H, W = int(depth_hw.shape[-2]), int(depth_hw.shape[-1])

    x0, y0, x1, y1 = box_xyxy.round().long().tolist()
    x0 = int(max(0, min(W - 1, x0)))
    x1 = int(max(0, min(W, x1)))
    y0 = int(max(0, min(H - 1, y0)))
    y1 = int(max(0, min(H, y1)))
    if x1 <= x0 + 1 or y1 <= y0 + 1:
        return None, None, None

    if mask_hw is None:
        mask_roi = torch.zeros((H, W), dtype=torch.bool)
        mask_roi[y0:y1, x0:x1] = True
    else:
        mask_t = mask_hw.detach().cpu()
        if mask_t.ndim == 3:
            mask_t = mask_t.squeeze(0)
        mask = (mask_t > 0.5)
        if mask.shape != (H, W):
            # Fallback: box-only if mask size mismatches (shouldn't happen with correct preprocessing).
            mask_roi = torch.zeros((H, W), dtype=torch.bool)
            mask_roi[y0:y1, x0:x1] = True
        else:
            # Restrict to box for speed/stability.
            mask_roi = torch.zeros((H, W), dtype=torch.bool)
            mask_roi[y0:y1, x0:x1] = mask[y0:y1, x0:x1]

    ij = torch.nonzero(mask_roi, as_tuple=False)
    if ij.shape[0] < min_points:
        return None, None, None

    depths = depth_hw[ij[:, 0], ij[:, 1]]
    valid = torch.isfinite(depths) & (depths > 0.01) & (depths < float(max_depth))
    if valid.sum().item() < min_points:
        return None, None, None

    ij = ij[valid]
    depths = depths[valid]

    # Sample points deterministically (stride) for speed.
    if depths.numel() > max_points:
        step = int(max(1, depths.numel() // max_points))
        ij = ij[::step]
        depths = depths[::step]

    z_med = torch.median(depths).item()
    z_abs_dev = torch.abs(depths - z_med)
    z_mad = torch.median(z_abs_dev).item()
    z_sigma = float(1.4826 * z_mad + 1e-6)

    fx = float(getattr(camera, "fx"))
    fy = float(getattr(camera, "fy"))
    px = float(getattr(camera, "px"))
    py = float(getattr(camera, "py"))
    us = ij[:, 1].float()
    vs = ij[:, 0].float()
    xs = (us - px) * depths / fx
    ys = (vs - py) * depths / fy

    x_med = torch.median(xs).item()
    y_med = torch.median(ys).item()
    p_cam = np.array([x_med, y_med, z_med], dtype=np.float32)
    return p_cam, z_med, z_sigma


def _transform_point_world(T_world_cam: torch.Tensor, p_cam: np.ndarray) -> np.ndarray:
    R = T_world_cam[:3, :3].detach().cpu().numpy()
    t = T_world_cam[:3, 3].detach().cpu().numpy()
    return (R @ p_cam.astype(np.float32)) + t


def _clip_box_xyxy_int(box_xyxy: torch.Tensor, width: int, height: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = box_xyxy.round().long().tolist()
    x0 = int(max(0, min(width - 1, x0)))
    x1 = int(max(0, min(width, x1)))
    y0 = int(max(0, min(height - 1, y0)))
    y1 = int(max(0, min(height, y1)))
    return x0, y0, x1, y1


def _encode_mask_roi(
    mask_hw: Optional[torch.Tensor],
    *,
    box_xyxy_int: Tuple[int, int, int, int],
    height: int,
    width: int,
    thresh: float = 0.5,
) -> Optional[Dict[str, Any]]:
    if mask_hw is None:
        return None
    m = mask_hw.detach().cpu()
    if m.ndim == 3:
        m = m.squeeze(0)
    if tuple(m.shape) != (height, width):
        return None
    x0, y0, x1, y1 = box_xyxy_int
    if x1 <= x0 or y1 <= y0:
        return None
    roi = (m[y0:y1, x0:x1] > float(thresh)).numpy().astype(np.uint8, copy=False)
    if roi.size == 0:
        return None
    pack = np.packbits(roi.reshape(-1))
    return {
        "box_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "roi_hw": [int(y1 - y0), int(x1 - x0)],
        "packbits": pack,
    }


def _decode_mask_roi(mask_info: Dict[str, Any], *, height: int, width: int) -> np.ndarray:
    x0, y0, x1, y1 = [int(v) for v in mask_info["box_xyxy"]]
    roi_h, roi_w = [int(v) for v in mask_info["roi_hw"]]
    packed = np.asarray(mask_info["packbits"], dtype=np.uint8)
    bits = np.unpackbits(packed)[: roi_h * roi_w]
    roi = bits.reshape((roi_h, roi_w)).astype(bool, copy=False)
    full = np.zeros((height, width), dtype=bool)
    full[y0:y1, x0:x1] = roi
    return full


def _color_for_id(idx: int) -> Tuple[int, int, int]:
    x = int(idx) * 2654435761  # Knuth
    r = 64 + (x & 0x7F)
    g = 64 + ((x >> 8) & 0x7F)
    b = 64 + ((x >> 16) & 0x7F)
    return int(r), int(g), int(b)


class SemanticMemoryManager:
    def __init__(self, system: System):
        self.system = system
        self._detector = None

    def _get_detector(self):
        if self._detector is not None:
            return self._detector
        try:
            from cross.cv.detection.owl_sam import OwlSam
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "OWL-SAM detector is unavailable. Ensure dependencies for "
                "`cross/cv/detection/owl_sam.py` (transformers, sam2, etc.) are installed."
            ) from e

        dev = getattr(self.system, "device", "cuda")
        if dev == "cuda":
            dev = "cuda:0"
        self._detector = OwlSam(device=dev)
        return self._detector

    def search_semantic_memory(self, query: str, *, top_k: Optional[int] = None) -> Dict[str, Any]:
        detector = self._get_detector()
        camera = getattr(self.system, "camera", None)
        if camera is None:
            raise RuntimeError("System has no `camera` attribute; set `System.camera` during init.")

        # Permanent keyframes only (stable + faster).
        keyframes = [kf for kf in self.system.get_all_keyframes() if not getattr(kf, "temporary", False)]
        if not keyframes:
            return {"query": query, "results": [], "tracks": []}

        min_det_score = 0.15
        nms_iou = 0.7
        merge_cell = 0.5  # meters
        base_gate = 0.35  # meters
        max_depth = 30.0

        observations: List[_SemanticObservation] = []

        for kf in keyframes:
            pil = _to_pil_rgb(kf.raw_rgb_image)
            res = detector.predict(pil, [[query]])
            boxes = torch.as_tensor(res.get("boxes", []))
            scores = torch.as_tensor(res.get("scores", []), dtype=torch.float32)
            masks = res.get("masks", [])
            if isinstance(masks, torch.Tensor):
                masks_t = masks
            elif isinstance(masks, (list, tuple)) and len(masks) > 0:
                masks_t = torch.as_tensor(masks)
            else:
                masks_t = torch.empty((0,))

            if boxes.numel() == 0 or scores.numel() == 0:
                continue

            keep_score = scores >= float(min_det_score)
            boxes = boxes[keep_score]
            scores = scores[keep_score]
            if masks_t.ndim >= 3 and masks_t.shape[0] == keep_score.shape[0]:
                masks_t = masks_t[keep_score]
            else:
                masks_t = torch.empty((0,))
            if boxes.numel() == 0:
                continue

            keep = _nms_xyxy(boxes, scores, iou_thresh=nms_iou)
            if not keep:
                continue

            kf_pose = kf.pose_mu[0].detach().cpu().tensor().tolist()
            T_world_cam = kf.pose_mu[0].matrix()
            pose_std_t = None
            try:
                pose_std_t = float(torch.max(kf.pose_std[0].tensor()[:3]).detach().cpu().item())
            except Exception:
                pose_std_t = None

            width, height = int(pil.size[0]), int(pil.size[1])

            for idx in keep:
                box = boxes[idx].to(dtype=torch.float32)
                mask = masks_t[idx] if masks_t.ndim == 3 and idx < masks_t.shape[0] else None

                box_int = _clip_box_xyxy_int(box, width, height)
                mask_enc = _encode_mask_roi(mask, box_xyxy_int=box_int, height=height, width=width)

                p_cam, z_med, z_sigma = _estimate_object_center_cam(
                    kf.depth_image,
                    camera,
                    box,
                    mask,
                    max_depth=max_depth,
                )

                p_world = None
                pos_sigma = None
                if p_cam is not None:
                    p_world = _transform_point_world(T_world_cam, p_cam)
                    pos_sigma = float(max((float(z_sigma) if z_sigma is not None else 0.1) * 0.5, 0.05))
                    if pose_std_t is not None:
                        pos_sigma = float(max(pos_sigma, 2.0 * pose_std_t))

                observations.append(
                    _SemanticObservation(
                        kf_id=int(kf.id),
                        kf_pose=kf_pose,
                        box_xyxy=[float(v) for v in box_int],
                        score=float(scores[idx].item()),
                        mask=mask_enc,
                        depth_median=float(z_med) if z_med is not None else None,
                        pos_cam=p_cam.tolist() if p_cam is not None else None,
                        pos_world=p_world.tolist() if p_world is not None else None,
                        pos_sigma=float(pos_sigma) if pos_sigma is not None else None,
                    )
                )

        if not observations:
            return {"query": query, "results": [], "tracks": []}

        # If depth is missing everywhere, fall back to per-keyframe max detection score.
        has_3d = any(o.pos_world is not None for o in observations)
        tracks: List[_SemanticTrack] = []
        cell_to_track_ids: Dict[Tuple[int, int, int], List[int]] = {}
        next_track_id = 0

        if has_3d:
            for obs in observations:
                if obs.pos_world is None:
                    continue
                c = np.asarray(obs.pos_world, dtype=np.float32)
                cell = tuple(np.floor(c / float(merge_cell)).astype(np.int32).tolist())
                cand_ids: List[int] = []
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dz in (-1, 0, 1):
                            cand_ids.extend(cell_to_track_ids.get((cell[0] + dx, cell[1] + dy, cell[2] + dz), []))

                best_tid = None
                best_dist = float("inf")
                gate = float(base_gate + (obs.pos_sigma or 0.1))
                for tid in cand_ids:
                    tr = tracks[tid]
                    dist = float(np.linalg.norm(tr.center_world - c))
                    local_gate = float(gate + tr.radius)
                    if dist < local_gate and dist < best_dist:
                        best_dist = dist
                        best_tid = tid

                if best_tid is None:
                    tid = next_track_id
                    next_track_id += 1
                    radius = float(max(obs.pos_sigma or 0.1, 0.1))
                    tr = _SemanticTrack(
                        track_id=tid,
                        center_world=c.copy(),
                        radius=radius,
                        fused_score=float(obs.score),
                        obs_count=1,
                        kf_ids=[obs.kf_id],
                    )
                    tracks.append(tr)
                    cell_to_track_ids.setdefault(cell, []).append(tid)
                    obs.track_id = tid
                else:
                    tr = tracks[best_tid]
                    tr.update(c, float(obs.score), float(max(obs.pos_sigma or 0.1, 0.1)), obs.kf_id)
                    obs.track_id = best_tid

        track_score: Dict[int, float] = {}
        if has_3d:
            for tr in tracks:
                support = 1.0 if tr.obs_count >= 2 else 0.5
                track_score[tr.track_id] = float(tr.fused_score * support)

        # Keep only best detection per keyframe for efficient downstream visualization.
        kf_best_score: Dict[int, float] = {}
        kf_best_obs: Dict[int, _SemanticObservation] = {}
        for obs in observations:
            if not has_3d:
                s = float(obs.score)
            else:
                if obs.track_id is None:
                    continue
                s = float(track_score.get(obs.track_id, 0.0))
            if s <= 0.0:
                continue
            prev = float(kf_best_score.get(obs.kf_id, -1.0))
            if s > prev:
                kf_best_score[obs.kf_id] = float(s)
                kf_best_obs[obs.kf_id] = obs

        ranked = sorted(kf_best_score.items(), key=lambda kv: kv[1], reverse=True)
        if top_k is not None:
            ranked = ranked[: int(max(0, top_k))]

        results: List[Dict[str, Any]] = []
        for kf_id, score in ranked:
            obs = kf_best_obs[kf_id]
            det = {
                "box_xyxy": [float(v) for v in obs.box_xyxy],
                "det_score": float(obs.score),
                "track_id": int(obs.track_id) if obs.track_id is not None else None,
                "track_score": float(track_score.get(obs.track_id, obs.score)) if has_3d else float(obs.score),
                "depth_median": obs.depth_median,
                "pos_cam": obs.pos_cam,
                "pos_world": obs.pos_world,
                "mask": obs.mask,
            }
            results.append(
                {
                    "kf_id": int(kf_id),
                    "score": float(score),
                    "kf_pose": obs.kf_pose,
                    "det": det,
                }
            )

        out_tracks: List[Dict[str, Any]] = []
        if has_3d:
            for tr in tracks:
                out_tracks.append(
                    {
                        "track_id": int(tr.track_id),
                        "center_world": tr.center_world.tolist(),
                        "radius": float(tr.radius),
                        "fused_score": float(tr.fused_score),
                        "obs_count": int(tr.obs_count),
                        "kf_ids": [int(x) for x in tr.kf_ids],
                    }
                )

        return {"query": query, "results": results, "tracks": out_tracks}

    def visualize(
        self,
        search_results: Dict[str, Any],
        *,
        k: int = 5,
        save_path: Optional[str] = None,
        show: bool = False,
        mask_alpha: float = 0.45,
    ):
        results = list(search_results.get("results") or [])
        if not results:
            return None

        k = int(max(0, k))
        results = results[: min(k, len(results))]

        kf_by_id = {int(kf.id): kf for kf in self.system.get_all_keyframes()}

        import matplotlib.pyplot as plt

        if not show:
            plt.switch_backend("Agg")

        n = len(results)
        fig, axes = plt.subplots(n, 1, figsize=(10, 4.5 * n))
        if n == 1:
            axes = [axes]

        for ax, item in zip(axes, results):
            kf_id = int(item.get("kf_id", -1))
            kf = kf_by_id.get(kf_id)
            if kf is None or kf.raw_rgb_image is None:
                ax.axis("off")
                ax.set_title(f"kf_id={kf_id} (missing image)")
                continue

            rgb = kf.raw_rgb_image.detach().float().cpu().clamp(0.0, 1.0)
            img = (rgb.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
            h, w = int(img.shape[0]), int(img.shape[1])

            det = item.get("det") or {}
            box = det.get("box_xyxy") or [0, 0, 0, 0]
            x0, y0, x1, y1 = [int(round(v)) for v in box]
            x0 = max(0, min(w - 1, x0))
            x1 = max(0, min(w, x1))
            y0 = max(0, min(h - 1, y0))
            y1 = max(0, min(h, y1))

            mask_info = det.get("mask")
            if isinstance(mask_info, dict):
                try:
                    mask = _decode_mask_roi(mask_info, height=h, width=w)
                    track_id = det.get("track_id")
                    color = _color_for_id(int(track_id) if track_id is not None else kf_id)
                    overlay = img.astype(np.float32, copy=True)
                    m = mask.astype(bool, copy=False)
                    overlay[m, 0] = (1.0 - mask_alpha) * overlay[m, 0] + mask_alpha * float(color[0])
                    overlay[m, 1] = (1.0 - mask_alpha) * overlay[m, 1] + mask_alpha * float(color[1])
                    overlay[m, 2] = (1.0 - mask_alpha) * overlay[m, 2] + mask_alpha * float(color[2])
                    img = overlay.round().astype(np.uint8)
                except Exception:
                    pass

            pil = Image.fromarray(img, mode="RGB")
            draw = ImageDraw.Draw(pil)
            draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=3)
            img = np.asarray(pil)

            score = float(item.get("score", 0.0))
            det_score = float(det.get("det_score", 0.0))
            track_id = det.get("track_id", None)
            title = f"kf_id={kf_id} score={score:.3f} det={det_score:.3f}"
            if track_id is not None:
                title += f" track={int(track_id)}"

            ax.imshow(img)
            ax.axis("off")
            ax.set_title(title)

        fig.tight_layout()

        if save_path:
            import os

            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return fig
