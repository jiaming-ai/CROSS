"""Per-frame geometry and appearance transforms of training windows.

Every crop is centred on the principal point (as in DUSt3R): the model's camera encoding is a field of view with the
principal point at the image centre, so the GT camera stays representable and the depth-derived point loss can use
the predicted (centred) intrinsics.  A crop smaller than the largest centred one zooms in (narrower field of view).
"""
from __future__ import annotations

import cv2
import numpy as np

cv2.setNumThreads(1)


def target_shape(aspect_hw: float, area: int = 512 * 512, patch: int = 16) -> tuple[int, int]:
    """(H, W) multiples of `patch` with H / W ~ aspect_hw and H * W ~ area (VGGT-Omega's 'balanced' mode)."""
    n = area / patch ** 2
    w = max(1, int(round(np.sqrt(n / aspect_hw))))
    h = max(1, int(round(n / w)))
    return h * patch, w * patch


def centred_crop_resize(img, dep, K, out_hw, zoom: float = 1.0):
    """Crop around the principal point with the aspect of out_hw (the largest such crop, shrunk by `zoom` <= 1), then
    resize.  Returns img (H,W,3) uint8, depth (H,W) f32, K (3x3)."""
    H, W = dep.shape
    oh, ow = out_hw
    cx, cy = K[0, 2], K[1, 2]
    hw = min(cx + 0.5, W - 0.5 - cx)           # half extents available around the principal point (pixel edges)
    hh = min(cy + 0.5, H - 0.5 - cy)
    a = ow / oh
    if hw / hh > a:
        hw = hh * a
    else:
        hh = hw / a
    hw, hh = hw * zoom, hh * zoom
    # integer crop (the principal point ends up within half a source pixel of the centre)
    ix0 = int(np.clip(round(cx + 0.5 - hw), 0, W - 1))
    iy0 = int(np.clip(round(cy + 0.5 - hh), 0, H - 1))
    ix1 = int(np.clip(ix0 + max(1, round(2 * hw)), 1, W))
    iy1 = int(np.clip(iy0 + max(1, round(2 * hh)), 1, H))
    img = img[iy0:iy1, ix0:ix1]
    dep = dep[iy0:iy1, ix0:ix1]
    sx, sy = ow / (ix1 - ix0), oh / (iy1 - iy0)
    img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_AREA if sx < 1 else cv2.INTER_LINEAR)
    dep = cv2.resize(dep, (ow, oh), interpolation=cv2.INTER_NEAREST)
    K2 = K.copy()
    K2[0, 0] *= sx
    K2[1, 1] *= sy
    K2[0, 2] = (cx - ix0 + 0.5) * sx - 0.5
    K2[1, 2] = (cy - iy0 + 0.5) * sy - 0.5
    return img, dep, K2


def color_jitter(img: np.ndarray, rng: np.random.Generator, strength: float = 1.0) -> np.ndarray:
    """img uint8 RGB -> float32 [0,1] with random brightness / contrast / saturation / hue / gamma."""
    x = img.astype(np.float32) / 255.0
    if strength <= 0:
        return x
    b = 1 + rng.uniform(-0.3, 0.3) * strength
    c = 1 + rng.uniform(-0.3, 0.3) * strength
    s = 1 + rng.uniform(-0.3, 0.3) * strength
    x = x * b
    m = x.mean()
    x = (x - m) * c + m
    g = x @ np.array([0.299, 0.587, 0.114], np.float32)
    x = (x - g[..., None]) * s + g[..., None]
    if rng.random() < 0.3 * strength:
        h = rng.uniform(-0.05, 0.05) * strength
        hsv = cv2.cvtColor(np.clip(x, 0, 1), cv2.COLOR_RGB2HSV)
        hsv[..., 0] = (hsv[..., 0] + h * 360) % 360
        x = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    if rng.random() < 0.3 * strength:
        x = np.clip(x, 0, 1) ** rng.uniform(0.7, 1.4)
    return np.clip(x, 0, 1)


def augment_frame(img_u8, rng, aug: dict | None):
    """Appearance augmentation of one frame; returns float32 [0,1] HxWx3."""
    if not aug:
        return img_u8.astype(np.float32) / 255.0
    x = color_jitter(img_u8, rng, aug.get("color", 1.0)) if rng.random() < aug.get("p_color", 0.8) \
        else img_u8.astype(np.float32) / 255.0
    if rng.random() < aug.get("p_gray", 0.05):
        g = x @ np.array([0.299, 0.587, 0.114], np.float32)
        x = np.repeat(g[..., None], 3, -1)
    if rng.random() < aug.get("p_blur", 0.05):
        k = int(rng.choice([3, 5]))
        x = cv2.GaussianBlur(x, (k, k), 0)
    if rng.random() < aug.get("p_noise", 0.05):
        x = np.clip(x + rng.normal(0, rng.uniform(0.005, 0.03), x.shape).astype(np.float32), 0, 1)
    return x
