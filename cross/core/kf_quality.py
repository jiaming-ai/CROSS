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
fraction is below min(info_min, info_rel * the session's running median informative fraction).  When the cells are
mostly removed for being flat, the view must also be empty: no structure above the image's own noise level (a
low-contrast white wall still has structure and may be the only view of a place; tested: rejecting such views cost
relocalization in a home scene).
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

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
    person_checked: bool = False      # the person detector has run on this frame
    snr: float = float("inf")         # 90th-percentile cell gradient / the image's noise level (structure above noise)
    ms: float = 0.0
    cells: dict = field(default_factory=dict, repr=False)   # per-cell masks (numpy bool), for refinement / figures
    pending: Optional[tuple] = field(default=None, repr=False)   # deferred stereo near field (y_left, rgb_right, W)

    def summary(self) -> dict:
        return {"info": round(self.info, 3), "thr": round(self.threshold, 3), "junk": bool(self.junk), "snr": round(self.snr, 2),
                "reason": self.reason, "stage": self.stage, "ms": round(self.ms, 2), "person_checked": self.person_checked,
                **{k: round(float(v), 3) for k, v in self.fractions.items()}}


class _RunningMedian:
    """Median of the last `n` samples that fall within the last `n` ticks (assessed frames).  When every tick pushes a
    sample, both bounds coincide; samples pushed only on some ticks (a cue computed for keyframe candidates only) still
    describe the same recent stretch of the session, so the median adapts as fast after a change of environment."""

    def __init__(self, n: int):
        self.n = max(int(n), 1)
        self.v = deque(maxlen=self.n)
        self.t = deque(maxlen=self.n)
        self.now = 0

    def tick(self):
        self.now += 1

    def push(self, x):
        if x is not None and np.isfinite(x):
            self.v.append(float(x))
            self.t.append(self.now)

    def _expire(self):
        while self.t and self.t[0] <= self.now - self.n:
            self.t.popleft()
            self.v.popleft()

    def get(self, default=None):
        self._expire()
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
        self._seen = 0
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
    @staticmethod
    def _pool(a: np.ndarray, gh: int, gw: int) -> np.ndarray:
        """Mean of `a` (h, w) over a gh x gw grid of cells (cell edges floor(i * h / gh))."""
        h, w = a.shape
        ys = (np.arange(gh) * h) // gh
        xs = (np.arange(gw) * w) // gw
        s = np.add.reduceat(np.add.reduceat(a.astype(np.float64), ys, axis=0), xs, axis=1)
        n = np.diff(np.append(ys, h))[:, None] * np.diff(np.append(xs, w))[None, :]
        return s / np.maximum(n, 1)

    def _person_cells(self, rgb: torch.Tensor, gh: int, gw: int) -> tuple:
        """Cells mostly inside a person box (score >= person_score); (cells or None, count)."""
        if not getattr(self.cfg, "person", True) or self._detector is False:
            return None, 0
        if self._detector is None:
            try:
                from cross.mono.person_detector import GraphedPersonDetector, make_person_detector
                det = make_person_detector(self._detector_device(rgb))
                # CUDA-graph replay of the backbone and heads: the same detections, ~10x less time in a busy process
                self._detector = GraphedPersonDetector(det) if getattr(self.cfg, "person_cuda_graph", True) else det
            except Exception as err:          # noqa: BLE001  e.g. no weights offline: the other cues still work
                logger.warning(f"keyframe quality: person detector unavailable ({type(err).__name__}: {err}); "
                               "continuing without the person cue")
                self._detector = False
                return None, 0
        dev = self._detector_device(rgb)
        thr = float(getattr(self.cfg, "person_score", 0.5))
        with torch.inference_mode():
            # the step's image lives in host RAM: the detector runs on the GPU (SSDLite on the CPU took ~13 ms a call)
            x = rgb.to(dev, non_blocking=True).float().clamp(0, 1)
            if hasattr(self._detector, "persons"):
                boxes = self._detector.persons(x, thr).cpu().numpy()
            else:
                res = self._detector([x])[0]
                boxes = res["boxes"][(res["labels"] == 1) & (res["scores"] >= thr)].cpu().numpy()
        if len(boxes) == 0:
            return None, 0
        H, W = rgb.shape[-2:]
        h, w = 4 * gh, 4 * gw                   # boxes rasterized at 4 x 4 samples per cell
        mask = np.zeros((h, w), bool)
        for x0, y0, x1, y1 in boxes:
            mask[max(int(y0 * h / H), 0):int(np.ceil(y1 * h / H)), max(int(x0 * w / W), 0):int(np.ceil(x1 * w / W))] = True
        return self._pool(mask, gh, gw) > 0.5, int(len(boxes))

    def _detector_device(self, rgb: torch.Tensor):
        """The person detector's device: the system's compute device when it is a GPU, else the image's."""
        dev = torch.device(self.device) if self.device is not None else rgb.device
        return dev if (dev.type == "cuda" and torch.cuda.is_available()) else rgb.device

    def _stereo_depth(self, y_left: np.ndarray, rgb_right: torch.Tensor, W: int,
                      y_right: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Metric depth of the left view from semi-global matching of the rectified pair at <= 256 px width (an
        occluder near the camera has a large disparity).  The disparity range covers depths down to near_abs / 2."""
        import cv2
        if not self.fx or not self.baseline:
            return None
        L = (np.clip(y_left, 0, 1) * 255).astype(np.uint8)
        R = (np.clip(y_right if y_right is not None else self._luminance(rgb_right, L.shape[1]), 0, 1) * 255).astype(np.uint8)
        fx = self.fx * L.shape[1] / W
        need = fx * self.baseline / (0.5 * float(getattr(self.cfg, "near_abs", 0.8)))
        nd = int(min(max(16 * int(np.ceil(need / 16)), 16), 128))
        sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=nd, blockSize=5, P1=8 * 25, P2=32 * 25,
                                     uniquenessRatio=10, speckleWindowSize=50, speckleRange=2,
                                     mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        disp = sgbm.compute(L, R).astype(np.float32) / 16.0
        return np.where(disp > 0.5, fx * self.baseline / np.maximum(disp, 1e-3), 0.0).astype(np.float32)

    @staticmethod
    def _luminance_dev(t: torch.Tensor, width: int) -> torch.Tensor:
        """Luminance (h, width) float in [0, 1] of a (3, H, W) image in [0, 1], on the image's device."""
        t = t.float()
        t = t[0] if t.dim() == 4 else t
        y = (0.299 * t[0] + 0.587 * t[1] + 0.114 * t[2]) if t.shape[0] == 3 else t[0]
        H, W = y.shape
        if width < W:
            y = F.interpolate(y[None, None], size=(max(int(round(H * width / W)), 1), width), mode="area")[0, 0]
        return y

    @staticmethod
    def _luminance(t: torch.Tensor, width: int) -> np.ndarray:
        """Luminance (h, width) float in [0, 1] on the CPU: one small transfer of a (3, H, W) image in [0, 1]."""
        return KeyframeQuality._luminance_dev(t, width).cpu().numpy()

    def _near_cells(self, depth, gh: int, gw: int, update: bool):
        """Cells whose valid depth is mostly closer than d_near; depth (1, H, W) / (H, W) metric (tensor or array),
        <= 0 invalid."""
        st = max(1, depth.shape[-1] // 128)          # <= 128 px across: the cells are 8 px wide there
        if torch.is_tensor(depth):                   # subsample first: the same values, a fraction of the copy
            d = depth.reshape(depth.shape[-2:])[::st, ::st].float().cpu().numpy()
        else:
            d = np.asarray(depth, np.float32)
            d = d.reshape(d.shape[-2:])[::st, ::st]
        valid = (d > 0) & np.isfinite(d)
        if valid.sum() < 0.05 * d.size:
            return None
        med = float(np.median(d[valid]))
        if update:
            self.depth_ref.push(med)
        ref = self.depth_ref.get(med)
        d_near = max(float(getattr(self.cfg, "near_abs", 0.8)), float(getattr(self.cfg, "near_rel", 0.25)) * ref)
        frac = self._pool(valid & (d < d_near), gh, gw)
        vfrac = self._pool(valid, gh, gw)
        # a cell is near when most of its valid depth is near (and it has some valid depth)
        return (frac > 0.5 * np.maximum(vfrac, 1e-6)) & (vfrac > 0.2)

    # ------------------------------------------------------------------ test
    @torch.inference_mode()
    def assess(self, rgb: torch.Tensor, depth: Optional[torch.Tensor] = None,
               rgb_right: Optional[torch.Tensor] = None, person: Optional[bool] = None,
               _full: bool = False) -> FrameQuality:
        """rgb (3, H, W) float in [0, 1] (the transformed image of the step); depth (1, H, W) metric or None; without
        depth, the rectified right image gives the near field by stereo matching (stereo_near).  The cues run on the
        CPU on a <= 256 px luminance image (one transfer).  The person detector (the costly cue, on the image's
        device) runs when `person` (default: config) is true; otherwise `add_person` adds it later, for the frames
        that are about to become permanent keyframes."""
        import cv2
        t0 = time.perf_counter()
        cfg = self.cfg
        # assess_every > 1: the full cues run on every k-th observed frame (the running medians) and on the candidates
        # for a permanent keyframe (completed in add_person); the other frames return a deferred assessment
        every = int(getattr(cfg, "assess_every", 1))
        now = bool(getattr(cfg, "person", True)) if person is None else bool(person)
        self._seen += 1
        if not _full and every > 1 and not now and self.n_assessed >= int(getattr(cfg, "warmup", 5)) and self._seen % every:
            q = FrameQuality(info=1.0, threshold=0.0, junk=False, reason=None, fractions={c: 0.0 for c in CAUSES},
                             stage="deferred")
            q.pending = ("deferred", rgb, depth, rgb_right)
            return q
        x = rgb[0] if rgb.dim() == 4 else rgb
        for ref in (self.grad_ref, self.depth_ref, self.info_ref):
            ref.tick()
        H, W = x.shape[-2:]
        gw = int(getattr(cfg, "grid", 16))
        gh = max(int(round(gw * H / W)), 1)
        y_right = None
        run_person = bool(getattr(cfg, "person", True)) if person is None else (person and bool(getattr(cfg, "person", True)))
        stereo = depth is None and rgb_right is not None and getattr(cfg, "stereo_near", True) and bool(self.fx and self.baseline)
        # the stereo near field (SGBM, the costly cue of a frame) is needed only for the frames whose decision is used:
        # the candidates for a permanent keyframe (complete()), or every frame when junk frames are not observed
        lazy = stereo and bool(getattr(cfg, "lazy_stereo_near", True)) and not run_person
        if stereo and not lazy:
            yl = self._luminance_dev(x, min(W, 256))
            yr = self._luminance_dev(rgb_right, yl.shape[1])
            y, y_right = torch.stack([yl, yr]).cpu().numpy() if yr.shape == yl.shape else (yl.cpu().numpy(), yr.cpu().numpy())
        else:
            y = self._luminance(x, min(W, 256))
        gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE) / 8
        gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE) / 8
        cell_g = self._pool(np.sqrt(gx * gx + gy * gy), gh, gw)
        clip = (y < float(getattr(cfg, "clip_dark", 0.04))) | (y > float(getattr(cfg, "clip_bright", 0.96)))
        clipped = self._pool(clip, gh, gw) > 0.5

        # typical textured-cell gradient: 75th percentile of the cells that are not clipped, as a running median
        g_ok = cell_g[~clipped]
        g75 = float(np.quantile(g_ok, 0.75)) if g_ok.size >= 4 else None
        self.grad_ref.push(g75)
        g_ref = self.grad_ref.get(g75 if g75 is not None else 0.0)
        flat = cell_g < float(getattr(cfg, "texture_rel", 0.25)) * g_ref

        # structure above the sensor noise: the 90th-percentile cell gradient over the image's noise level (Immerkaer's
        # estimator; pure noise gives ~0.6).  A view that is textureless for the session (a white wall) still has
        # structure; an empty one (blank surface, covered lens) has none
        lap = cv2.filter2D(y, -1, np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32))[1:-1, 1:-1]
        noise = np.sqrt(np.pi / 2) / 6 * float(np.abs(lap).mean())
        snr = float(np.quantile(g_ok, 0.9)) / max(noise, 1e-6) if g_ok.size >= 4 else 0.0

        cells = {"clipped": clipped, "flat": flat & ~clipped}
        if stereo and not lazy:
            depth = self._stereo_depth(y, rgb_right, W, y_right=y_right)
        near = self._near_cells(depth, gh, gw, update=True) if depth is not None else None
        if near is not None:
            cells["near"] = near
        n_person = 0
        if run_person:
            pc, n_person = self._person_cells(x, gh, gw)
            if pc is not None:
                cells["person"] = pc
        q = self._decide(cells, stage="image", has_depth=near is not None, n_person=n_person, update=True, snr=snr)
        q.person_checked = run_person
        if lazy:
            q.pending = (y, rgb_right, W)
        q.ms = (time.perf_counter() - t0) * 1e3
        self.stats["assessed"] += 1
        self.stats["ms_total"] += q.ms
        return q

    @torch.inference_mode()
    def add_person(self, q: FrameQuality, rgb: torch.Tensor) -> FrameQuality:
        """Complete the assessment of a candidate for a permanent keyframe (same threshold as its first assessment):
        the deferred stereo near field, then the person cells."""
        if q is not None and q.pending is not None and isinstance(q.pending[0], str):      # deferred assessment
            _, rgb_d, depth_d, right_d = q.pending
            q = self.assess(rgb_d, depth_d, rgb_right=right_d, person=False, _full=True)
        if q is not None and q.pending is not None:
            q = self._complete_stereo(q)
        if q is None or q.person_checked or q.junk or not getattr(self.cfg, "person", True):
            return q
        t0 = time.perf_counter()
        gh, gw = next(iter(q.cells.values())).shape
        pc, n_person = self._person_cells(rgb[0] if rgb.dim() == 4 else rgb, gh, gw)
        q.person_checked = True
        if pc is None:
            dt = (time.perf_counter() - t0) * 1e3
            q.ms += dt
            self.stats["ms_total"] += dt
            return q
        cells = dict(q.cells)
        cells["person"] = pc
        r = self._decide(cells, stage=q.stage, has_depth=q.has_depth, n_person=n_person, update=False, threshold=q.threshold,
                         snr=q.snr)
        r.person_checked = True
        r.ms = q.ms + (time.perf_counter() - t0) * 1e3
        self.stats["ms_total"] += r.ms - q.ms
        if r.junk:
            self.stats["junk"] += 1
            self.stats["by_reason"][r.reason] += 1
        return r

    def _complete_stereo(self, q: FrameQuality) -> FrameQuality:
        """The deferred stereo near field of `assess` (SGBM on the rectified pair), with the running depth median."""
        y, rgb_right, W = q.pending
        q.pending = None
        t0 = time.perf_counter()
        gh, gw = next(iter(q.cells.values())).shape
        depth = self._stereo_depth(y, rgb_right, W)
        near = self._near_cells(depth, gh, gw, update=True) if depth is not None else None
        dt = (time.perf_counter() - t0) * 1e3
        self.stats["ms_total"] += dt
        if near is None:
            q.ms += dt
            return q
        cells = dict(q.cells)
        cells["near"] = near
        r = self._decide(cells, stage=q.stage, has_depth=True, n_person=q.n_person, update=False, threshold=q.threshold,
                         snr=q.snr)
        r.person_checked = q.person_checked
        r.ms = q.ms + dt
        if r.junk and not q.junk:
            self.stats["junk"] += 1
            self.stats["by_reason"][r.reason] += 1
        return r

    @torch.inference_mode()
    def refine(self, q: FrameQuality, depth) -> FrameQuality:
        """Add the near-field cells of a metric depth map of the current view (the feed-forward pass's depth)."""
        if q is None or depth is None:
            return q
        t0 = time.perf_counter()
        gh, gw = next(iter(q.cells.values())).shape
        near = self._near_cells(depth, gh, gw, update=not q.has_depth)
        if near is None:
            return q
        cells = dict(q.cells)
        cells["near"] = near if "near" not in cells else (cells["near"] | near)
        was_junk = q.junk
        r = self._decide(cells, stage="pass", has_depth=True, n_person=q.n_person, update=False, threshold=q.threshold,
                         snr=q.snr)
        r.person_checked = q.person_checked
        r.ms = q.ms + (time.perf_counter() - t0) * 1e3
        self.stats["ms_total"] += r.ms - q.ms
        if r.junk and not was_junk:
            self.stats["junk"] += 1
            self.stats["by_reason"][r.reason] += 1
        return r

    def _decide(self, cells: dict, stage: str, has_depth: bool, n_person: int, update: bool,
                threshold: Optional[float] = None, snr: float = float("inf")) -> FrameQuality:
        cfg = self.cfg
        removed = np.zeros_like(next(iter(cells.values())), dtype=bool)
        for c in CAUSES:
            if c in cells:
                removed |= cells[c]
        info = 1.0 - float(removed.mean())
        fractions = {c: float(cells[c].mean()) if c in cells else 0.0 for c in CAUSES}
        if threshold is None:
            ref = self.info_ref.get(info)
            threshold = min(float(getattr(cfg, "info_min", 0.35)), float(getattr(cfg, "info_rel", 0.5)) * ref)
        warm = self.n_assessed < int(getattr(cfg, "warmup", 5))
        # the cause that alone removes the most cells (causes listed in order of precedence on ties)
        main = max(CAUSES, key=lambda c: fractions[c])
        # a view that is only textureless for the session (a white wall, a floor) still has structure and is sometimes
        # the only view of a place: it is junk only when it shows no structure above the sensor noise (a blank surface
        # filling the view, a covered lens); a view blocked by something close, a person or clipping is junk below the
        # threshold
        junk = (not warm) and info < threshold
        if main == "flat":
            junk = junk and snr < float(getattr(cfg, "empty_snr", 1.5))
        if update:
            self.info_ref.push(info)
            self.n_assessed += 1
        reason = None
        if junk:
            reason = main
            if update:
                self.stats["junk"] += 1
                self.stats["by_reason"][reason] += 1
        return FrameQuality(info=info, threshold=float(threshold), junk=bool(junk), reason=reason, fractions=fractions,
                            has_depth=has_depth, n_person=n_person, stage=stage, cells=cells, snr=float(snr))

    def summary(self) -> dict:
        s = dict(self.stats)
        s["by_reason"] = dict(self.stats["by_reason"])
        s["ms_mean"] = self.stats["ms_total"] / max(self.stats["assessed"], 1)
        s["refs"] = {"grad": self.grad_ref.get(), "depth": self.depth_ref.get(), "info": self.info_ref.get()}
        return s
