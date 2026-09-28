"""Robust scale observations and a scalar log-scale filter.

Learned metric depth supplies a *prior*, not an observable metric measurement.
Spatial blocks and an irreducible variance floor prevent pixel count from
creating spurious certainty about a model's common scale bias.
"""

from dataclasses import dataclass

import numpy as np

from .config import ScaleConfig


@dataclass
class ScaleObservation:
    log_scale: float = 0.0
    variance: float = float("inf")
    inlier_fraction: float = 0.0
    log_mad: float = float("inf")
    pixels: int = 0
    tiles: int = 0
    accepted: bool = False
    reason: str = "no_valid_depth"


def observe_scale(target_depth, source_depth, confidence=None, config=None):
    """Estimate target/source using corresponding pixels of the SAME image."""
    cfg = config or ScaleConfig()
    target = np.asarray(target_depth, dtype=np.float64)
    source = np.asarray(source_depth, dtype=np.float64)
    if target.shape != source.shape or target.ndim != 2:
        raise ValueError("Depth maps must be equally sized HxW arrays")
    valid = np.isfinite(target) & np.isfinite(source) & (target > 1e-5) & (source > 1e-5)
    if confidence is not None:
        confidence = np.asarray(confidence)
        if confidence.shape != source.shape:
            raise ValueError("Confidence and depth shapes differ")
        finite = valid & np.isfinite(confidence)
        if finite.any():
            valid &= finite & (confidence >= np.quantile(confidence[finite], 0.5))
        else:
            valid[:] = False
    count = int(valid.sum())
    obs = ScaleObservation(pixels=count)
    if count < cfg.min_pixels:
        return obs
    ratios = np.full(source.shape, np.nan)
    ratios[valid] = np.log(target[valid]) - np.log(source[valid])
    center = float(np.median(ratios[valid]))
    mad = float(1.4826 * np.median(np.abs(ratios[valid] - center)))
    inliers = valid & (np.abs(ratios - center) < max(2.5 * mad, 0.08))
    block_medians = []
    for rows in np.array_split(np.arange(source.shape[0]), 8):
        for cols in np.array_split(np.arange(source.shape[1]), 8):
            block = ratios[np.ix_(rows, cols)]
            mask = inliers[np.ix_(rows, cols)]
            if mask.sum() >= max(4, block.size // 10):
                block_medians.append(float(np.median(block[mask])))
    obs.inlier_fraction = float(inliers.sum() / count)
    obs.log_mad = mad
    obs.tiles = len(block_medians)
    if len(block_medians) < cfg.min_tiles:
        obs.reason = "insufficient_spatial_support"
        return obs
    obs.log_scale = float(np.median(block_medians))
    block_scatter = 1.4826 * np.median(np.abs(np.asarray(block_medians) - obs.log_scale))
    obs.variance = float(max(cfg.observation_std_floor**2, block_scatter**2))
    obs.accepted = mad <= cfg.max_log_mad and obs.inlier_fraction >= cfg.min_inlier_fraction
    obs.reason = "accepted" if obs.accepted else "inconsistent_shape"
    return obs


def observe_sparse_scale(target_depth, source_depth, pixels, image_shape, config=None):
    """Metric/VO depth ratios at mature tracked patches, with spatial support.

    Each patch is a sparse correspondence in one image. This estimator uses
    the same systematic uncertainty floor as the dense estimator, without
    pretending the missing pixels were observed.
    """
    cfg = config or ScaleConfig()
    target, source = np.asarray(target_depth), np.asarray(source_depth)
    pixels = np.asarray(pixels)
    if target.shape != source.shape or target.ndim != 1 or pixels.shape != (len(target), 2):
        raise ValueError("Expected N depth pairs and Nx2 pixels")
    valid = np.isfinite(target) & np.isfinite(source) & (target > 0) & (source > 0)
    obs = ScaleObservation(pixels=int(valid.sum()))
    if valid.sum() < 20:
        return obs
    ratios = np.log(target[valid]) - np.log(source[valid])
    center = np.median(ratios)
    mad = float(1.4826 * np.median(np.abs(ratios - center)))
    good = np.abs(ratios - center) <= max(2.5 * mad, 0.08)
    tile_ids = np.floor(pixels[valid] / [image_shape[1], image_shape[0]] * 4).astype(int)
    tile_ids = np.clip(tile_ids, 0, 3)
    blocks = []
    for tile in np.unique(tile_ids, axis=0):
        mask = good & (tile_ids == tile).all(1)
        if mask.sum() >= 2:
            blocks.append(np.median(ratios[mask]))
    obs.tiles = len(blocks)
    obs.log_mad = mad
    obs.inlier_fraction = float(good.mean())
    if len(blocks) < min(cfg.min_tiles, 6):
        obs.reason = "insufficient_spatial_support"
        return obs
    obs.log_scale = float(np.median(blocks))
    obs.variance = float(max(cfg.observation_std_floor**2,
                             (1.4826 * np.median(np.abs(np.array(blocks) - obs.log_scale)))**2))
    obs.accepted = mad <= cfg.max_log_mad and obs.inlier_fraction >= cfg.min_inlier_fraction
    obs.reason = "accepted" if obs.accepted else "inconsistent_shape"
    return obs


class LogScaleFilter:
    def __init__(self, config=None):
        self.config = config or ScaleConfig()
        self.mean = 0.0
        self.variance = 1.0
        self.initialized = self.config.mode == "relative"
        self.accepted = 0
        self.rejected = 0

    @property
    def scale(self):
        return float(np.exp(self.mean))

    def predict(self, frames=1):
        self.variance += frames * self.config.process_std_per_frame**2

    def update(self, observation):
        if not observation.accepted:
            self.rejected += 1
            return False
        if self.config.mode == "relative" or (self.initialized and self.config.mode == "initial"):
            return False
        if not self.initialized or self.config.mode == "direct":
            self.mean, self.variance = observation.log_scale, observation.variance
            self.initialized = True
        else:
            residual = observation.log_scale - self.mean
            innovation = self.variance + observation.variance
            if abs(residual) > self.config.innovation_gate * np.sqrt(innovation):
                self.rejected += 1
                observation.accepted = False
                observation.reason = "innovation_gate"
                return False
            gain = self.variance / innovation
            self.mean += gain * residual
            self.variance *= 1.0 - gain
        self.variance = max(self.variance, self.config.posterior_std_floor**2)
        self.accepted += 1
        return True
