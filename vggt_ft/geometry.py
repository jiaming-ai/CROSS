"""Scene normalisation, unprojection and ground-truth covisibility for training windows.

Conventions: extrinsics are camera-from-world 4x4 (OpenCV axes), intrinsics 3x3 at the window's image resolution,
depth is z-depth (0 = invalid).  The model's gauge is VGGT's: frame 0's camera is the world origin and the scene is
scaled to unit mean distance of the valid points from that origin.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def unproject(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """depth (B,S,H,W), K (B,S,3,3) -> camera-frame points (B,S,H,W,3)."""
    B, S, H, W = depth.shape
    ys, xs = torch.meshgrid(torch.arange(H, device=depth.device, dtype=depth.dtype),
                            torch.arange(W, device=depth.device, dtype=depth.dtype), indexing="ij")
    fx, fy = K[..., 0, 0][..., None, None], K[..., 1, 1][..., None, None]
    cx, cy = K[..., 0, 2][..., None, None], K[..., 1, 2][..., None, None]
    return torch.stack([(xs - cx) / fx * depth, (ys - cy) / fy * depth, depth], -1)


def cam_to_ref(points: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """points (B,S,H,W,3) in each camera, E (B,S,4,4) camera-from-reference -> points in the reference frame."""
    T = torch.linalg.inv(E)
    R, t = T[..., :3, :3], T[..., :3, 3]
    return torch.einsum("bsij,bshwj->bshwi", R, points) + t[:, :, None, None, :]


def normalize_window(extrinsics, depths, masks, intrinsics, scale_frames=None):
    """GT in the model's gauge.  Returns E_norm (B,S,4,4) camera-from-frame-0 with unit-scale translation, depths_norm,
    world points in frame 0 (normalised) and scale (B,) = mean distance (metres, or SfM units) of the valid points of
    `scale_frames` (B,S bool; frames connected to frame 0 -- an unrelated negative must not set the scale)."""
    E0_inv = torch.linalg.inv(extrinsics[:, 0])
    E_rel = extrinsics @ E0_inv[:, None]                                    # cam_s from cam_0
    pts = cam_to_ref(unproject(depths, intrinsics), E_rel)
    dist = pts.norm(dim=-1)
    m = masks if scale_frames is None else masks & scale_frames[:, :, None, None]
    m = m.float()
    scale = ((dist * m).sum((1, 2, 3)) / m.sum((1, 2, 3)).clamp(min=1.0)).clamp(1e-4, 1e5)
    E_n = E_rel.clone()
    E_n[:, :, :3, 3] = E_n[:, :, :3, 3] / scale[:, None, None]
    return E_n, depths / scale[:, None, None, None], pts / scale[:, None, None, None, None], scale


@torch.no_grad()
def gt_covisibility(depths, masks, extrinsics, intrinsics, world_id=None, grid: int = 32, rel_tol: float = 0.15,
                    symmetric: bool = True) -> torch.Tensor:
    """Overlap fraction of every frame pair (B,S,S) in [0,1], diagonal 1.

    A grid of valid pixels of frame i is unprojected with the GT depth, moved into frame j and projected; it counts when
    it lands inside frame j in front of the camera and agrees with frame j's depth within rel_tol (or frame j has no
    valid depth there).  covis[i, j] = inlier fraction of frame i's valid pixels; symmetrised with min (both views must
    see the shared area).  Frames with different world ids share no world frame (0).  Same definition as CROSS's
    geometric gate (cross/cv/pose_est_ff.py) and the underwater fine-tune, so thresholds keep their meaning."""
    B, S, H, W = depths.shape
    dev = depths.device
    gy, gx = torch.meshgrid(torch.linspace(0, H - 1, grid, device=dev), torch.linspace(0, W - 1, grid, device=dev),
                            indexing="ij")
    gy, gx = gy.reshape(-1), gx.reshape(-1)
    iy, ix = gy.round().long(), gx.round().long()
    d = depths[:, :, iy, ix]
    v = masks[:, :, iy, ix] & (d > 0)
    fx, fy = intrinsics[..., 0, 0, None], intrinsics[..., 1, 1, None]
    cx, cy = intrinsics[..., 0, 2, None], intrinsics[..., 1, 2, None]
    P = torch.stack([(gx - cx) / fx * d, (gy - cy) / fy * d, d, torch.ones_like(d)], -1)   # (B,S,N,4)
    E = extrinsics.double()
    T = (E[:, None] @ torch.linalg.inv(E)[:, :, None]).float()                            # (B,i,j): cam_i -> cam_j
    Pj = torch.einsum("bijkl,binl->bijnk", T, P)[..., :3]
    z = Pj[..., 2]
    Kj = intrinsics[:, None]
    u = Kj[..., 0, 0, None] * Pj[..., 0] / z.clamp(min=1e-6) + Kj[..., 0, 2, None]
    w = Kj[..., 1, 1, None] * Pj[..., 1] / z.clamp(min=1e-6) + Kj[..., 1, 2, None]
    inside = (z > 1e-6) & (u >= 0) & (u <= W - 1) & (w >= 0) & (w <= H - 1)
    un = torch.stack([u / (W - 1) * 2 - 1, w / (H - 1) * 2 - 1], -1)
    N = un.shape[3]
    un = un.permute(0, 2, 1, 3, 4).reshape(B * S, 1, S * N, 2)
    zd = F.grid_sample(depths.reshape(B * S, 1, H, W).float(), un.float(), mode="nearest", align_corners=True)
    zd = zd.view(B, S, S, N).transpose(1, 2)
    agree = (zd <= 0) | ((z - zd).abs() <= rel_tol * zd.clamp(min=1e-6))
    ok = inside & agree & v[:, :, None]
    cov = ok.float().sum(-1) / v[:, :, None].float().sum(-1).clamp(min=1)
    if symmetric:
        cov = torch.minimum(cov, cov.transpose(1, 2))
    if world_id is not None:
        cov = cov * (world_id[:, :, None] == world_id[:, None, :]).float()
    eye = torch.eye(S, device=dev, dtype=torch.bool)[None]
    return torch.where(eye, torch.ones_like(cov), cov)


def connected_to_ref(covis: torch.Tensor, thr: float = 0.05) -> torch.Tensor:
    """(B,S) bool: frames joined to frame 0 by a chain of pairs with covis > thr (their pose is observable)."""
    B, S, _ = covis.shape
    adj = covis > thr
    reach = torch.zeros(B, S, dtype=torch.bool, device=covis.device)
    reach[:, 0] = True
    for _ in range(S):
        new = reach | (adj & reach[:, :, None]).any(1)
        if torch.equal(new, reach):
            break
        reach = new
    return reach
