"""Keyframe quality: is a frame worth storing as a permanent keyframe (and worth observing with)?

A frame whose view is mostly blocked by something close to the camera (a person stepping in front of the robot, a
hand, a box carried past, a wall corner at arm's length), or whose image carries little scene content (lens covered,
blank wall / floor / sky, dark or overexposed, motion blur), is useless as a map keyframe: the relative-pose estimators
fail on it.  It is also harmful: views of the same kind of occluder at different places look alike, so such keyframes
are retrieved at wrong places and seed false hypotheses and loop closures.  And a frame on which retrieval fails is
exactly the one the keyframe policy stores as a new permanent keyframe (a "novel" view).

One quantity decides: the *informative fraction* of the view, the share of image cells that show textured,
well-exposed, static scene content at a normal distance.  A cell is removed when it is
  - near:    most of its depth (sensor, stereo matching of the rectified pair, or the feed-forward pass's own depth of
             the current view) is closer than
             d_near = max(near_abs, near_rel * the session's typical depth);
  - person:  mostly inside a person detection (transient content, and the usual close occluder);
  - clipped: mostly under- or over-exposed pixels;
  - flat:    its gradient energy is below texture_rel * the session's typical textured-cell gradient (blank surface,
             blur, covered lens).
The session's typical values are running medians over the frames assessed so far (robust to a minority of junk
frames), so the test adapts to the camera, the exposure and the environment.  A frame is junk when its informative
fraction is below min(info_min, info_rel * the session's running median informative fraction).
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

CAUSES = ("near", "person", "clipped", "flat")


@dataclass
class FrameQuality:
    info: float                       # informative fraction of the view
    threshold: float                  # junk below this
    junk: bool
    reason: Optional[str]             # the cause that removed most cells (None: not junk)
    fractions: dict                   # fraction of cells removed by each cause (cells can have several causes)
    has_depth: bool = False
    n_person: int = 0
    stage: str = "image"              # "image" (pre-observation cues) or "pass" (refined with the pass depth)
    ms: float = 0.0
    cells: dict = field(default_factory=dict, repr=False)   # per-cell masks (torch bool), for refinement / figures

    def summary(self) -> dict:
        return {"info": round(self.info, 3), "thr": round(self.threshold, 3), "junk": bool(self.junk),
                "reason": self.reason, "stage": self.stage, "ms": round(self.ms, 2),
                **{k: round(float(v), 3) for k, v in self.fractions.items()}}


class _RunningMedian:
    def __init__(self, n: int):
        self.v = deque(maxlen=max(int(n), 1))

    def push(self, x):
        if x is not None and np.isfinite(x):
            self.v.append(float(x))

    def get(self, default=None):
        return float(np.median(self.v)) if self.v else default

    def __len__(self):
        return len(self.v)


class KeyframeQuality:
    """Per-frame informative-fraction test (see the module docstring).  `assess` uses the image (and sensor depth);
    `refine` adds the near-field cells from a depth map available only after the observation (feed-forward pass)."""

    def __init__(self, cfg, device="cuda"):
        self.cfg = cfg
        self.device = device
        n = int(getattr(cfg, "window", 200))
        self.grad_ref = _RunningMedian(n)       # typical gradient of a textured cell
        self.depth_ref = _RunningMedian(n)      # typical (median) scene depth
        self.info_ref = _RunningMedian(n)       # typical informative fraction
        self.n_assessed = 0
        self.stats = {"assessed": 0, "junk": 0, "rejected_permanent": 0, "skipped_observation": 0,
                      "by_reason": {c: 0 for c in CAUSES}, "ms_total": 0.0}
        self._detector = None
        self.fx = None              # focal length (px) of the step's images and the stereo baseline (m): SGBM near field
        self.baseline = None

    def set_stereo(self, K, T_right_in_left):
        """Calibration of the (transformed) stereo images: the near field from classical stereo matching."""
        if K is not None and T_right_in_left is not None:
            self.fx = float(np.asarray(K)[0, 0])
            self.baseline = float(abs(np.asarray(T_right_in_left)[0, 3]))

    # ------------------------------------------------------------------ cues
    def _person_mask(self, rgb: torch.Tensor, hw) -> tuple:
        """Union of person boxes (score >= person_score) rasterized at `hw`; (mask or None, count)."""
        if not getattr(self.cfg, "person", True):
            return None, 0
        if self._detector is None:
            from cross.mono.person_detector import make_person_detector
            self._detector = make_person_detector(rgb.device)
        with torch.inference_mode():
            res = self._detector([rgb.float().clamp(0, 1)])[0]
        keep = (res["labels"] == 1) & (res["scores"] >= float(getattr(self.cfg, "person_score", 0.5)))
        boxes = res["boxes"][keep]
        if boxes.numel() == 0:
            return None, 0
        H, W = rgb.shape[-2:]
        h, w = hw
        mask = torch.zeros((h, w), dtype=torch.bool, device=rgb.device)
        sx, sy = w / W, h / H
        for x0, y0, x1, y1 in boxes.tolist():
            mask[max(int(y0 * sy), 0):int(np.ceil(y1 * sy)), max(int(x0 * sx), 0):int(np.ceil(x1 * sx))] = True
        return mask, int(boxes.shape[0])

    def _stereo_depth(self, rgb: torch.Tensor, rgb_right: torch.Tensor) -> Optional[torch.Tensor]:
        """Metric depth of the left view from semi-global matching of the rectified pair at half resolution (an
        occluder near the camera has a large disparity).  The disparity range covers depths down to near_abs / 2."""
        import cv2
        if not self.fx or not self.baseline:
            return None
        def gray(t):
            t = t.float()
            t = t[0] if t.dim() == 4 else t
            y = (0.299 * t[0] + 0.587 * t[1] + 0.114 * t[2]) if t.shape[0] == 3 else t[0]
            return (y.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        L, R = gray(rgb), gray(rgb_right)
        W = L.shape[1]
        L, R = cv2.resize(L, (W // 2, L.shape[0] // 2), interpolation=cv2.INTER_AREA), \
            cv2.resize(R, (W // 2, R.shape[0] // 2), interpolation=cv2.INTER_AREA)
        fx = self.fx * L.shape[1] / W
        need = fx * self.baseline / (0.5 * float(getattr(self.cfg, "near_abs", 0.8)))
        nd = int(min(max(16 * int(np.ceil(need / 16)), 16), 128))
        sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=nd, blockSize=5, P1=8 * 25, P2=32 * 25,
                                     uniquenessRatio=10, speckleWindowSize=50, speckleRange=2,
                                     mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        disp = sgbm.compute(L, R).astype(np.float32) / 16.0
        z = np.where(disp > 0.5, fx * self.baseline / np.maximum(disp, 1e-3), 0.0).astype(np.float32)
        return torch.from_numpy(z)

    def _near_cells(self, depth: torch.Tensor, grid_hw, update: bool):
        """Cells whose valid depth is mostly closer than d_near; depth (1, H, W) or (H, W) metric, <= 0 invalid."""
        d = depth.float().reshape(1, 1, *depth.shape[-2:])
        valid = (d > 0) & torch.isfinite(d)
        if int(valid.sum()) < 0.05 * d.numel():
            return None
        med = float(d[valid].median())
        if update:
            self.depth_ref.push(med)
        ref = self.depth_ref.get(med)
        d_near = max(float(getattr(self.cfg, "near_abs", 0.8)), float(getattr(self.cfg, "near_rel", 0.25)) * ref)
        near = (valid & (d < d_near)).float()
        frac = F.adaptive_avg_pool2d(near, grid_hw)[0, 0]
        vfrac = F.adaptive_avg_pool2d(valid.float(), grid_hw)[0, 0]
        # a cell is near when most of its valid depth is near (and it has some valid depth)
        return (frac > 0.5 * vfrac.clamp_min(1e-6)) & (vfrac > 0.2)

    # ------------------------------------------------------------------ test
    @torch.inference_mode()
    def assess(self, rgb: torch.Tensor, depth: Optional[torch.Tensor] = None,
               rgb_right: Optional[torch.Tensor] = None) -> FrameQuality:
        """rgb (3, H, W) float in [0, 1] (the transformed image of the step); depth (1, H, W) metric or None; without
        depth, the rectified right image gives the near field by stereo matching (stereo_near)."""
        t0 = time.perf_counter()
        cfg = self.cfg
        x = rgb.float()
        if x.dim() == 4:
            x = x[0]
        H, W = x.shape[-2:]
        gw = int(getattr(cfg, "grid", 16))
        gh = max(int(round(gw * H / W)), 1)
        # luminance at <= 256 px width (the cues are coarse; this keeps the test ~1 ms)
        s = min(1.0, 256.0 / W)
        y = (0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2])[None, None]
        if s < 1.0:
            y = F.interpolate(y, scale_factor=s, mode="area")
        h, w = y.shape[-2:]
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=y.dtype, device=y.device).view(1, 1, 3, 3) / 8
        gx = F.conv2d(F.pad(y, (1, 1, 1, 1), mode="replicate"), kx)
        gy = F.conv2d(F.pad(y, (1, 1, 1, 1), mode="replicate"), kx.transpose(2, 3))
        g = torch.sqrt(gx * gx + gy * gy)
        cell_g = F.adaptive_avg_pool2d(g, (gh, gw))[0, 0]
        clip = ((y < float(getattr(cfg, "clip_dark", 0.04))) | (y > float(getattr(cfg, "clip_bright", 0.96)))).float()
        clipped = F.adaptive_avg_pool2d(clip, (gh, gw))[0, 0] > 0.5

        # typical textured-cell gradient: 75th percentile of the cells that are not clipped, as a running median
        g_ok = cell_g[~clipped]
        g75 = float(torch.quantile(g_ok, 0.75)) if g_ok.numel() >= 4 else None
        self.grad_ref.push(g75)
        g_ref = self.grad_ref.get(g75 if g75 is not None else 0.0)
        flat = cell_g < float(getattr(cfg, "texture_rel", 0.25)) * g_ref

        cells = {"clipped": clipped, "flat": flat & ~clipped}
        if depth is None and rgb_right is not None and getattr(cfg, "stereo_near", True):
            depth = self._stereo_depth(x, rgb_right)
        near = self._near_cells(depth.to(x.device), (gh, gw), update=True) if depth is not None else None
        if near is not None:
            cells["near"] = near
        pmask, n_person = self._person_mask(x, (h, w))
        if pmask is not None:
            cells["person"] = F.adaptive_avg_pool2d(pmask.float()[None, None], (gh, gw))[0, 0] > 0.5
        q = self._decide(cells, stage="image", has_depth=near is not None, n_person=n_person, update=True)
        q.ms = (time.perf_counter() - t0) * 1e3
        self.stats["assessed"] += 1
        self.stats["ms_total"] += q.ms
        return q

    @torch.inference_mode()
    def refine(self, q: FrameQuality, depth: torch.Tensor) -> FrameQuality:
        """Add the near-field cells of a metric depth map of the current view (the feed-forward pass's depth)."""
        if q is None or depth is None:
            return q
        t0 = time.perf_counter()
        some = next(iter(q.cells.values()))
        near = self._near_cells(depth.to(some.device), tuple(some.shape), update=not q.has_depth)
        if near is None:
            return q
        cells = dict(q.cells)
        cells["near"] = near if "near" not in cells else (cells["near"] | near)
        was_junk = q.junk
        r = self._decide(cells, stage="pass", has_depth=True, n_person=q.n_person, update=False, threshold=q.threshold)
        r.ms = q.ms + (time.perf_counter() - t0) * 1e3
        self.stats["ms_total"] += r.ms - q.ms
        if r.junk and not was_junk:
            self.stats["junk"] += 1
            self.stats["by_reason"][r.reason] += 1
        return r

    def _decide(self, cells: dict, stage: str, has_depth: bool, n_person: int, update: bool,
                threshold: Optional[float] = None) -> FrameQuality:
        cfg = self.cfg
        some = next(iter(cells.values()))
        removed = torch.zeros_like(some)
        for c in CAUSES:
            if c in cells:
                removed |= cells[c]
        info = 1.0 - float(removed.float().mean())
        fractions = {c: float(cells[c].float().mean()) if c in cells else 0.0 for c in CAUSES}
        if threshold is None:
            ref = self.info_ref.get(info)
            threshold = min(float(getattr(cfg, "info_min", 0.35)), float(getattr(cfg, "info_rel", 0.5)) * ref)
        warm = self.n_assessed < int(getattr(cfg, "warmup", 5))
        junk = (not warm) and info < threshold
        if update:
            self.info_ref.push(info)
            self.n_assessed += 1
        reason = None
        if junk:
            # the cause that alone removes the most cells (causes listed in order of precedence on ties)
            reason = max(CAUSES, key=lambda c: fractions[c])
            if stage == "image":
                self.stats["junk"] += 1
                self.stats["by_reason"][reason] += 1
        return FrameQuality(info=info, threshold=float(threshold), junk=bool(junk), reason=reason, fractions=fractions,
                            has_depth=has_depth, n_person=n_person, stage=stage, cells=cells)

    def summary(self) -> dict:
        s = dict(self.stats)
        s["by_reason"] = dict(self.stats["by_reason"])
        s["ms_mean"] = self.stats["ms_total"] / max(self.stats["assessed"], 1)
        s["refs"] = {"grad": self.grad_ref.get(), "depth": self.depth_ref.get(), "info": self.info_ref.get()}
        return s
