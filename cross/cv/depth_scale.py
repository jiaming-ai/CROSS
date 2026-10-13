"""Metric scale of a feed-forward pass from measured depth (RGB-D mode with the feed-forward estimator).

A feed-forward geometry model (VGGT-Omega) predicts the cameras and depth maps of all views of a pass in one
similarity gauge.  Where a view has measured depth (an RGB-D sensor, dense or sparse, e.g. LiDAR), the ratio of the
measured to the predicted depth over pixels where both are trusted is the metric scale of that gauge: the predicted
camera translations scaled by it are metric.  One robust estimate per pass: the median log ratio per view, then the
median over the views used.  No stereo rig and no odometry are needed, and depth and image of one sensor share the frame
time (no pairing error).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from cross.cv.stereo_scale import ScaleEstimate


def view_log_ratios(pred_depth: torch.Tensor, meas_depth: torch.Tensor, conf: Optional[torch.Tensor] = None, *,
                    min_depth: float = 0.2, max_depth: float = 10.0, conf_quantile: float = 0.5,
                    max_pixels: int = 20000) -> torch.Tensor:
    """log(measured / predicted) depth over the trusted pixels of one view (1-D tensor, possibly empty).

    pred_depth: (H, W) model-gauge depth; meas_depth: (H, W) or (1, H, W) metric depth, 0 = missing (resized to the
    prediction's grid, nearest, when the sizes differ); conf: (H, W) model confidence or None.  Trusted: measured depth in
    [min_depth, max_depth], predicted depth > 0, and model confidence at or above its conf_quantile among those pixels.
    At most max_pixels samples (a regular subsample) are kept."""
    pred = pred_depth.float()
    meas = meas_depth.float().to(pred.device)
    if meas.dim() == 3:
        meas = meas[0]
    if meas.shape != pred.shape:
        meas = F.interpolate(meas[None, None], size=tuple(pred.shape), mode="nearest")[0, 0]
    ok = torch.isfinite(meas) & (meas >= min_depth) & (meas <= max_depth) & torch.isfinite(pred) & (pred > 1e-6)
    if conf is not None and conf_quantile > 0 and bool(ok.any()):
        c = conf.float().to(pred.device)
        thr = torch.quantile(c[ok][:: max(1, int(ok.sum()) // 100000)], float(conf_quantile))
        ok = ok & (c >= thr)
    r = torch.log(meas[ok]) - torch.log(pred[ok])
    if r.numel() > max_pixels:
        r = r[:: (r.numel() + max_pixels - 1) // max_pixels]
    return r


def depth_scale(pred_depths: Sequence[torch.Tensor], meas_depths: Sequence[Optional[torch.Tensor]],
                confs: Optional[Sequence[Optional[torch.Tensor]]] = None, *, min_depth: float = 0.2,
                max_depth: float = 10.0, conf_quantile: float = 0.5, min_pixels: int = 200,
                std_floor: float = 0.02) -> ScaleEstimate:
    """Metric scale of a pass's gauge from the views with measured depth (ScaleEstimate, method "depth").

    Per view with at least min_pixels trusted pixels: the median log ratio (view_log_ratios); the pass's scale is
    exp(median over those views).  log_std (the relative scale std used to inflate translation noise): the spread of
    the per-view medians (MAD) with several views, the pixel MAD / sqrt(pixels) with one, at least std_floor.  ratios:
    the per-view scales; n_used views of n_anchors with depth.  Invalid when no view has enough trusted pixels."""
    medians, mads, counts = [], [], []
    n_with_depth = 0
    for i, (d, m) in enumerate(zip(pred_depths, meas_depths)):
        if m is None or d is None:
            continue
        n_with_depth += 1
        c = None if confs is None else confs[i]
        r = view_log_ratios(d, m, c, min_depth=min_depth, max_depth=max_depth, conf_quantile=conf_quantile)
        if r.numel() < min_pixels:
            continue
        med = torch.median(r)
        medians.append(float(med))
        mads.append(float(torch.median(torch.abs(r - med))))
        counts.append(int(r.numel()))
    if not medians:
        return ScaleEstimate(scale=1.0, valid=False, n_anchors=n_with_depth, n_used=0, method="depth")
    med = np.asarray(medians)
    log_s = float(np.median(med))
    if len(med) >= 3:
        log_std = 1.4826 * float(np.median(np.abs(med - log_s)))
    elif len(med) == 2:
        log_std = 0.5 * abs(float(med[0] - med[1]))
    else:
        log_std = 1.4826 * mads[0] / np.sqrt(counts[0])
    return ScaleEstimate(scale=float(np.exp(log_s)), valid=True, n_anchors=n_with_depth, n_used=len(med),
                         ratios=np.exp(med), weights=np.asarray(counts, dtype=np.float64),
                         kinds=["depth"] * len(med), log_std=max(float(std_floor), float(log_std)), method="depth")
