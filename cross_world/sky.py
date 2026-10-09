"""The sky: segmentation masks of the views and a sky texture behind the Gaussians.

Outdoors the sky has no depth (stereo finds no match, a depth sensor no return), and the camera sees it only in a
narrow band of directions (KITTI: 29 degrees vertically).  Without a sky model the photometric loss paints it with
Gaussians at whatever distance they happen to sit: right from the training views, they float as white and dark patches
in the air as soon as the viewpoint leaves the path.  As street-scene splatting does (Street Gaussians, OmniRe, PVG),
the sky here is a texture over directions, an equirectangular map in a gravity-aligned frame, composited behind the
Gaussians (C = C_gs + (1 - alpha) C_sky(d)); a binary cross-entropy on the accumulated opacity keeps the Gaussians off
the pixels a segmentation network labels sky, and on the others.

The masks come from OneFormer (ADE20K, Swin-T; MIT licence, needs `transformers`).  The published weights are a
pickle, which current `transformers` refuses to load with torch < 2.6: convert them once to model.safetensors (see
docs/WORLD.md) and point CROSS_SKY_MODEL (or BuildConfig.sky_model) at that folder.
"""
from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_MODEL = "shi-labs/oneformer_ade20k_swin_tiny"


class SkySegmenter:
    """Sky masks of RGB images (ADE20K class "sky")."""

    def __init__(self, model: str = "", device="cuda"):
        from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
        path = model or os.environ.get("CROSS_SKY_MODEL") or DEFAULT_MODEL
        self.proc = OneFormerProcessor.from_pretrained(path)
        self.net = OneFormerForUniversalSegmentation.from_pretrained(path).to(device).eval()
        self.ids = [int(i) for i, n in self.net.config.id2label.items() if n == "sky"]
        self.device = device

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """(H, W, 3) uint8 RGB -> (H, W) bool."""
        x = self.proc(images=rgb, task_inputs=["semantic"], return_tensors="pt").to(self.device)
        seg = self.proc.post_process_semantic_segmentation(self.net(**x), target_sizes=[rgb.shape[:2]])[0]
        return torch.isin(seg, torch.tensor(self.ids, device=seg.device)).cpu().numpy()


def sky_frame(up: np.ndarray) -> np.ndarray:
    """Rotation from the map frame to the sky frame (z = up; rows are the sky axes in map coordinates)."""
    z = np.asarray(up, np.float64)
    z = z / np.linalg.norm(z)
    a = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = a - (a @ z) * z
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z])


def pixel_dirs(T_wc: torch.Tensor, Ks: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """(C, H, W, 3) unit viewing directions (map frame) of the pixel centres."""
    dev = T_wc.device
    ys, xs = torch.meshgrid(torch.arange(height, device=dev, dtype=torch.float32) + 0.5,
                            torch.arange(width, device=dev, dtype=torch.float32) + 0.5, indexing="ij")
    d = torch.stack([(xs[None] - Ks[:, 0, 2, None, None]) / Ks[:, 0, 0, None, None],
                     (ys[None] - Ks[:, 1, 2, None, None]) / Ks[:, 1, 1, None, None],
                     torch.ones_like(xs)[None].expand(len(Ks), -1, -1)], -1)
    return F.normalize(torch.einsum("cij,chwj->chwi", T_wc[:, :3, :3], d), dim=-1)


def _grid(dirs: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Directions -> grid_sample coordinates of the equirectangular map (x: longitude, y: -latitude)."""
    d = dirs.reshape(-1, 3) @ R.T
    lon = torch.atan2(d[:, 1], d[:, 0]) / math.pi
    lat = torch.asin(d[:, 2].clamp(-1, 1)) / (math.pi / 2)
    return torch.stack([lon, -lat], -1)


class SkyModel(torch.nn.Module):
    """Sky colour by direction: a (3, res, 2 res) equirectangular texture (logits) in the sky frame."""

    def __init__(self, R: np.ndarray, res: int = 256, init_rgb=(0.7, 0.8, 0.9), tex: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer("R", torch.as_tensor(np.asarray(R), dtype=torch.float32))
        if tex is None:
            c = torch.logit(torch.tensor(init_rgb, dtype=torch.float32).clamp(0.02, 0.98))
            tex = c[:, None, None].repeat(1, res, 2 * res)
        else:
            tex = torch.logit(tex.float().clamp(1e-3, 1 - 1e-3))
        self.tex = torch.nn.Parameter(tex[None].contiguous())
        self.register_buffer("seen", torch.zeros(tex.shape[1:]))

    def forward(self, dirs: torch.Tensor, frozen: bool = False) -> torch.Tensor:
        """Colours along `dirs` (..., 3); `frozen`: no gradient to the texture (it still flows to the directions)."""
        g = _grid(dirs, self.R)
        tex = self.tex.detach() if frozen else self.tex
        c = F.grid_sample(tex, g.view(1, -1, 1, 2), mode="bilinear", padding_mode="border", align_corners=False)
        return torch.sigmoid(c.view(3, -1).T).view(*dirs.shape[:-1], 3)

    @torch.no_grad()
    def mark_seen(self, dirs: torch.Tensor, mask: torch.Tensor, stride: int = 4) -> None:
        """Count the sky pixels (every stride-th) that fell into each texel."""
        d = dirs[:, ::stride, ::stride][mask[:, ::stride, ::stride]]
        if len(d) == 0:
            return
        g = _grid(d, self.R)
        h, w = self.seen.shape
        ix = ((g[:, 0] + 1) / 2 * w).long().clamp(0, w - 1)
        iy = ((g[:, 1] + 1) / 2 * h).long().clamp(0, h - 1)
        self.seen.view(-1).index_add_(0, iy * w + ix, torch.ones(len(ix), device=ix.device))

    @torch.no_grad()
    def colours(self) -> torch.Tensor:
        """(3, H, W) texture in [0, 1], with the texels no sky pixel reached filled from the seen ones (push-pull:
        the zenith and the ground side of a camera that never looked there get a smooth continuation)."""
        tex = torch.sigmoid(self.tex[0])
        w = (self.seen > 0).float()
        if w.sum() == 0:
            return tex
        return push_pull(tex, w)


def push_pull(img: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Fill the zero-weight pixels of (C, H, W) `img` from coarser weighted averages."""
    pyr = [(img * w, w)]
    while min(pyr[-1][1].shape) > 1:
        a, b = pyr[-1]
        pyr.append((F.avg_pool2d(a[None], 2, ceil_mode=True)[0], F.avg_pool2d(b[None, None], 2, ceil_mode=True)[0, 0]))
    out = pyr[-1][0] / pyr[-1][1].clamp(min=1e-9)
    for a, b in reversed(pyr[:-1]):
        up = F.interpolate(out[None], size=b.shape, mode="bilinear", align_corners=False)[0]
        own = a / b.clamp(min=1e-9)
        k = b.clamp(max=1.0)[None]
        out = k * own + (1 - k) * up
    return out


def composite(rgb: torch.Tensor, alpha: torch.Tensor, sky: Optional[SkyModel], T_wc: torch.Tensor, Ks: torch.Tensor,
              width: int, height: int) -> torch.Tensor:
    """Gaussians over the sky: rgb (C, H, W, 3) premultiplied, alpha (C, H, W, 1)."""
    if sky is None:
        return rgb
    return rgb + (1 - alpha) * sky(pixel_dirs(T_wc, Ks, width, height))


def sky_from_state(state: Optional[dict], device="cuda") -> Optional[SkyModel]:
    """A frozen SkyModel from a saved chunk sky ({"tex": (3, H, W) colours, "R": (3, 3)})."""
    if not state:
        return None
    m = SkyModel(state["R"].numpy() if torch.is_tensor(state["R"]) else state["R"], tex=state["tex"].float())
    m.seen.fill_(1)
    return m.to(device).requires_grad_(False)


def sky_splats(state: dict, center: np.ndarray, radius: float, res: int = 96, sh_coeffs: int = 0) -> dict:
    """The sky texture as a shell of flat Gaussians around `center` (for viewers that draw splats only)."""
    from cross_world.gaussians import matrix_to_quat, rgb_to_sh
    tex = state["tex"].float()
    R = torch.as_tensor(np.asarray(state["R"]), dtype=torch.float32)
    img = F.interpolate(tex[None], size=(res, 2 * res), mode="area")[0]               # (3, res, 2 res)
    lat = (0.5 - (torch.arange(res) + 0.5) / res) * math.pi
    lon = ((torch.arange(2 * res) + 0.5) / (2 * res) * 2 - 1) * math.pi
    LA, LO = torch.meshgrid(lat, lon, indexing="ij")
    d_sky = torch.stack([LA.cos() * LO.cos(), LA.cos() * LO.sin(), LA.sin()], -1).view(-1, 3)
    east = torch.stack([-LO.sin(), LO.cos(), torch.zeros_like(LO)], -1).view(-1, 3)
    north = torch.cross(d_sky, east, dim=-1)
    # sky-frame vectors -> map frame (R maps map -> sky, so its transpose maps back)
    to_map = lambda v: v @ R
    d, e, n = to_map(d_sky), to_map(east), to_map(north)
    step = math.pi / res
    s_lat = torch.full((len(d),), 0.8 * radius * step)
    s_lon = (0.8 * radius * step * LA.cos().view(-1)).clamp(min=0.05 * radius * step)
    rot = torch.stack([e, n, d], -1)                                                   # columns: local axes
    cols = img.permute(1, 2, 0).reshape(-1, 3)
    n_pts = len(d)
    return {"means": torch.from_numpy(np.asarray(center, np.float32))[None] + radius * d,
            "scales": torch.log(torch.stack([s_lon, s_lat, torch.full_like(s_lat, 1e-3 * radius)], -1)),
            "quats": matrix_to_quat(rot),
            "opacities": torch.full((n_pts,), 6.0),
            "sh0": rgb_to_sh(cols)[:, None, :],
            "shN": torch.zeros(n_pts, sh_coeffs, 3)}
