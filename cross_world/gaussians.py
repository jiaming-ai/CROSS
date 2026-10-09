"""3D Gaussians of one chunk: initialisation from depth, training with gsplat, rendering and image metrics.

Training follows current practice for posed captures with depth (gsplat; MCMC densification with a Gaussian budget;
dense initialisation from back-projected depth, which needs little densification; L1 + D-SSIM photometric loss; L1 on
inverse depth; per-view pose refinement, since map poses carry centimetre-level errors that blur a radiance field; a
per-view affine colour transform for exposure / white-balance changes, which test views take from the training view
nearest in time).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

C0 = 0.28209479177387814          # SH band-0 constant


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / C0


def sh_to_rgb(sh: torch.Tensor) -> torch.Tensor:
    return sh * C0 + 0.5


@dataclass
class TrainConfig:
    steps_per_view: int = 40            # training steps per training view of the chunk ...
    min_steps: int = 3000               # ... within these bounds
    max_steps: int = 15000
    batch_size: int = 1
    sh_degree: int = 3
    sh_degree_interval: int = 1000      # one SH band more every this many steps
    # MCMC budget: Gaussians per megapixel of training images (61k: 8000 per 512x256 view) within these bounds (and
    # at least 1.2x the initial points); per pixel, so that full-resolution views get a proportional budget
    cap_per_mpix: int = 61_000
    cap_min: int = 300_000
    cap_max: int = 3_000_000
    init_stride: int = 2                # back-project every n-th pixel ...
    init_voxel_px: float = 2.0          # ... and keep one point per voxel of this many pixels' footprint at its depth
    init_opacity: float = 0.5
    far_points: int = 0                 # points on rays without depth, at far_depth (outdoor sky / distant scenery)
    far_depth: float = 0.0              # 0: 3x the 98th percentile of the depth
    ssim_lambda: float = 0.2
    depth_lambda: float = 0.2           # L1 on inverse depth (scaled by the median depth), decays 10x over training
    opacity_reg: float = 0.01           # MCMC regularisers
    scale_reg: float = 0.01
    pose_opt: bool = True
    pose_lr: float = 2e-4               # lazy Adam (only the batch's views): ~lr per time a view is sampled
    pose_reg: float = 1e-4
    app_opt: bool = True                # per-view affine colour
    app_lr: float = 1e-3
    app_reg: float = 1e-2
    means_lr: float = 1.6e-4            # x scene scale
    scales_lr: float = 5e-3
    quats_lr: float = 1e-3
    opacities_lr: float = 5e-2
    sh0_lr: float = 2.5e-3
    shN_lr: float = 2.5e-3 / 20
    strategy: str = "mcmc"              # mcmc | default (adaptive density control with absgrad)
    consistency_tol: float = 0.0        # > 0: multi-view depth consistency filter (relative tolerance)
    log_every: int = 500


# ---------------------------------------------------------------------------------------------------------- geometry
def rot6d_to_matrix(r6: torch.Tensor) -> torch.Tensor:
    a1, a2 = r6[..., :3], r6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)


def delta_transform(d: torch.Tensor) -> torch.Tensor:
    """(N, 9) pose residual (translation, 6-D rotation offset from identity) -> (N, 4, 4)."""
    eye6 = torch.tensor([1.0, 0, 0, 0, 1, 0], device=d.device, dtype=d.dtype)
    T = torch.eye(4, device=d.device, dtype=d.dtype).repeat(len(d), 1, 1)
    T[:, :3, :3] = rot6d_to_matrix(d[:, 3:] + eye6)
    T[:, :3, 3] = d[:, :3]
    return T


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of wxyz quaternions."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def matrix_to_quat(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 4) wxyz."""
    m = R
    t = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    q = torch.stack([1 + t, 1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2], 1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2],
                     1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2]], -1)
    k = q.argmax(-1)
    w = torch.empty(R.shape[:-2] + (4,), dtype=R.dtype, device=R.device)
    s = torch.sqrt(q.clamp(min=1e-12)) * 2
    # four branches, chosen per element by the largest diagonal term (numerically stable)
    br = [
        torch.stack([s[..., 0] / 4, (m[..., 2, 1] - m[..., 1, 2]) / s[..., 0], (m[..., 0, 2] - m[..., 2, 0]) / s[..., 0], (m[..., 1, 0] - m[..., 0, 1]) / s[..., 0]], -1),
        torch.stack([(m[..., 2, 1] - m[..., 1, 2]) / s[..., 1], s[..., 1] / 4, (m[..., 0, 1] + m[..., 1, 0]) / s[..., 1], (m[..., 0, 2] + m[..., 2, 0]) / s[..., 1]], -1),
        torch.stack([(m[..., 0, 2] - m[..., 2, 0]) / s[..., 2], (m[..., 0, 1] + m[..., 1, 0]) / s[..., 2], s[..., 2] / 4, (m[..., 1, 2] + m[..., 2, 1]) / s[..., 2]], -1),
        torch.stack([(m[..., 1, 0] - m[..., 0, 1]) / s[..., 3], (m[..., 0, 2] + m[..., 2, 0]) / s[..., 3], (m[..., 1, 2] + m[..., 2, 1]) / s[..., 3], s[..., 3] / 4], -1),
    ]
    for i in range(4):
        sel = k == i
        w[sel] = br[i][sel]
    w = w * torch.where(w[..., :1] < 0, -1.0, 1.0)
    return F.normalize(w, dim=-1)


# ---------------------------------------------------------------------------------------------------------- data
@dataclass
class ViewBatch:
    """The training / test views of a chunk on the GPU."""
    ids: List[int]
    images: torch.Tensor                 # (N, H, W, 3) uint8
    depths: Optional[torch.Tensor]       # (N, H, W) float16, 0 = none
    T_wc: torch.Tensor                   # (N, 4, 4) float32
    Ks: torch.Tensor                     # (N, 3, 3)
    times: torch.Tensor                  # (N,) timestamps (or index)
    width: int
    height: int


def backproject(depth: torch.Tensor, K: torch.Tensor, T_wc: torch.Tensor, stride: int = 1, mask=None):
    """World points and pixel coordinates of the valid depth pixels (every `stride`-th)."""
    d = depth[::stride, ::stride].float()
    h, w = d.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=d.device) * stride, torch.arange(w, device=d.device) * stride, indexing="ij")
    ok = d > 0
    if mask is not None:
        ok &= mask[::stride, ::stride]
    z = d[ok]
    x = (xs[ok].float() + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (ys[ok].float() + 0.5 - K[1, 2]) / K[1, 1] * z
    pc = torch.stack([x, y, z], -1)
    pw = pc @ T_wc[:3, :3].T + T_wc[:3, 3]
    return pw, ys[ok], xs[ok], z


def gaussian_budget(cfg: TrainConfig, n_views: int, width: int, height: int) -> int:
    return int(np.clip(cfg.cap_per_mpix * n_views * width * height / 1e6, cfg.cap_min, cfg.cap_max))


def init_budget(cfg: TrainConfig, n_views: int, width: int, height: int) -> int:
    """Initial Gaussians at most: 70 % of the MCMC budget, so that densification has room."""
    return int(0.7 * gaussian_budget(cfg, n_views, width, height))


@torch.no_grad()
def init_gaussians(vb: ViewBatch, cfg: TrainConfig, masks: Optional[List[torch.Tensor]] = None,
                   keep_fn=None, seed: int = 0) -> Dict[str, torch.Tensor]:
    """Gaussians from the back-projected depth of the views, one per voxel of an adaptive grid whose cell is
    `init_voxel_px` pixels' footprint at the point's depth (so near and far surfaces get similar screen density).
    Colour from the pixel, isotropic scale from the footprint.  `keep_fn(points) -> bool mask` drops points the
    chunk does not need."""
    dev = vb.images.device
    g = torch.Generator(device=dev).manual_seed(seed)
    pts, cols, foot = [], [], []
    far_rays = []
    fx_med = float(vb.Ks[:, 0, 0].median())
    for i in range(len(vb.ids)):
        if vb.depths is None:
            break
        m = masks[i] if masks is not None else None
        p, ys, xs, z = backproject(vb.depths[i], vb.Ks[i], vb.T_wc[i], cfg.init_stride, m)
        if keep_fn is not None and len(p):
            k = keep_fn(p)
            p, ys, xs, z = p[k], ys[k], xs[k], z[k]
        pts.append(p)
        cols.append(vb.images[i, ys, xs].float() / 255.0)
        foot.append(z / vb.Ks[i, 0, 0])
        if cfg.far_points > 0:
            nd = (vb.depths[i] <= 0)
            yy, xx = torch.nonzero(nd[::4, ::4], as_tuple=True)
            if len(yy):
                far_rays.append((i, yy * 4, xx * 4))
    if pts:
        P = torch.cat(pts)
        Cc = torch.cat(cols)
        Fp = torch.cat(foot)
    else:
        P = torch.zeros(0, 3, device=dev)
        Cc = torch.zeros(0, 3, device=dev)
        Fp = torch.zeros(0, device=dev)
    # adaptive voxel grid: level L = floor(log2(footprint / f0)), cell = f0 * 2^L * init_voxel_px
    if len(P):
        f0 = float(Fp.median())
        L = torch.floor(torch.log2((Fp / f0).clamp(min=1e-6))).clamp(-8, 8)
        budget = init_budget(cfg, len(vb.ids), vb.width, vb.height) - (cfg.far_points if cfg.far_points > 0 else 0)
        vox = cfg.init_voxel_px
        while True:              # coarser voxels until the initial points fit the budget
            cell = f0 * vox * torch.pow(2.0, L)
            key = torch.cat([L[:, None], torch.floor(P / cell[:, None])], 1).long()
            uniq, inv = torch.unique(key, dim=0, return_inverse=True)
            n = len(uniq)
            if n <= max(budget, 1000) or vox > 64:
                break
            vox *= 1.25
        cnt = torch.zeros(n, device=dev).index_add_(0, inv, torch.ones(len(P), device=dev))
        P = torch.zeros(n, 3, device=dev).index_add_(0, inv, P) / cnt[:, None]
        Cc = torch.zeros(n, 3, device=dev).index_add_(0, inv, Cc) / cnt[:, None]
        S = torch.zeros(n, device=dev).index_add_(0, inv, cell) / cnt
    else:
        S = torch.zeros(0, device=dev)
    # far points: rays without depth, at far_depth (sky, distant scenery), spread over the views
    if cfg.far_points > 0 and far_rays:
        dmax = float(torch.quantile(vb.depths[vb.depths > 0].float()[:1_000_000], 0.98)) if vb.depths is not None else 50.0
        fd = cfg.far_depth or 3.0 * dmax
        per = max(1, cfg.far_points // len(far_rays))
        fp, fc = [], []
        for i, yy, xx in far_rays:
            sel = torch.randperm(len(yy), generator=g, device=dev)[:per]
            yy, xx = yy[sel], xx[sel]
            z = torch.full((len(yy),), fd, device=dev) * (1 + 0.2 * torch.rand(len(yy), generator=g, device=dev))
            K = vb.Ks[i]
            pc = torch.stack([(xx.float() + 0.5 - K[0, 2]) / K[0, 0] * z, (yy.float() + 0.5 - K[1, 2]) / K[1, 1] * z, z], -1)
            fp.append(pc @ vb.T_wc[i, :3, :3].T + vb.T_wc[i, :3, 3])
            fc.append(vb.images[i, yy, xx].float() / 255.0)
        fp, fc = torch.cat(fp), torch.cat(fc)
        P = torch.cat([P, fp])
        Cc = torch.cat([Cc, fc])
        S = torch.cat([S, torch.full((len(fp),), fd * 4.0 / fx_med, device=dev)])
    n = len(P)
    quats = torch.zeros(n, 4, device=dev)
    quats[:, 0] = 1
    K_sh = (cfg.sh_degree + 1) ** 2
    return {
        "means": P.contiguous(),
        "scales": torch.log(S.clamp(min=1e-6))[:, None].repeat(1, 3),
        "quats": quats,
        "opacities": torch.logit(torch.full((n,), cfg.init_opacity, device=dev)),
        "sh0": rgb_to_sh(Cc)[:, None, :],
        "shN": torch.zeros(n, K_sh - 1, 3, device=dev),
    }


# ---------------------------------------------------------------------------------------------------------- render
def render(splats: Dict[str, torch.Tensor], T_wc: torch.Tensor, Ks: torch.Tensor, width: int, height: int,
           sh_degree: int, render_mode: str = "RGB+ED", backgrounds=None, **kw):
    """Render (C, H, W, 3) colour, (C, H, W, 1) expected depth, alpha and the rasterizer info."""
    from gsplat import rasterization
    viewmats = torch.linalg.inv(T_wc)
    colors = torch.cat([splats["sh0"], splats["shN"]], 1)
    out, alpha, info = rasterization(
        splats["means"], F.normalize(splats["quats"], dim=-1), torch.exp(splats["scales"]),
        torch.sigmoid(splats["opacities"]), colors, viewmats, Ks, width, height,
        sh_degree=min(sh_degree, int(math.isqrt(colors.shape[1])) - 1), render_mode=render_mode,
        packed=False, backgrounds=backgrounds, **kw)
    if render_mode == "RGB+ED":
        return out[..., :3], out[..., 3:4], alpha, info
    return out, None, alpha, info


def apply_appearance(rgb: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """rgb (C, H, W, 3), A (C, 3, 4): per-view affine colour."""
    return torch.einsum("chwj,cij->chwi", rgb, A[:, :, :3]) + A[:, None, None, :, 3]


# ---------------------------------------------------------------------------------------------------------- metrics
def _gauss_window(size=11, sigma=1.5, device="cpu"):
    x = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean SSIM of (C, H, W, 3) images in [0, 1] (11x11 Gaussian window)."""
    a = a.permute(0, 3, 1, 2)
    b = b.permute(0, 3, 1, 2)
    w = _gauss_window(device=a.device).repeat(3, 1, 1, 1)
    mu_a = F.conv2d(a, w, groups=3)
    mu_b = F.conv2d(b, w, groups=3)
    s_aa = F.conv2d(a * a, w, groups=3) - mu_a ** 2
    s_bb = F.conv2d(b * b, w, groups=3) - mu_b ** 2
    s_ab = F.conv2d(a * b, w, groups=3) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (s_aa + s_bb + c2))
    return m.mean()


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(-10 * torch.log10(F.mse_loss(a, b).clamp(min=1e-10)))


_LPIPS = {}


def lpips(a: torch.Tensor, b: torch.Tensor) -> Optional[float]:
    """LPIPS (AlexNet) of (C, H, W, 3) images in [0, 1]; None when the lpips package is missing."""
    try:
        import lpips as _lp
    except ImportError:
        return None
    dev = a.device
    if dev not in _LPIPS:
        _LPIPS[dev] = _lp.LPIPS(net="alex", verbose=False).to(dev).eval()
    with torch.no_grad():
        return float(_LPIPS[dev](a.permute(0, 3, 1, 2) * 2 - 1, b.permute(0, 3, 1, 2) * 2 - 1).mean())


# ---------------------------------------------------------------------------------------------------------- training
@dataclass
class TrainResult:
    splats: Dict[str, torch.Tensor]
    pose_delta: Dict[int, np.ndarray]          # view id -> 4x4 correction (T_wc_refined = T_wc @ delta)
    appearance: Dict[int, np.ndarray]          # view id -> 3x4 affine colour
    stats: dict = field(default_factory=dict)


def train_chunk(vb: ViewBatch, splats: Dict[str, torch.Tensor], cfg: TrainConfig, scene_scale: float,
                log=print, seed: int = 0) -> TrainResult:
    from gsplat.strategy import DefaultStrategy, MCMCStrategy
    torch.manual_seed(seed)
    dev = vb.images.device
    n_views = len(vb.ids)
    steps = int(np.clip(cfg.steps_per_view * n_views, cfg.min_steps, cfg.max_steps))
    bs = max(1, cfg.batch_size)
    steps = max(1, steps // bs)
    lr_scale = math.sqrt(bs)
    params = torch.nn.ParameterDict({k: torch.nn.Parameter(v.contiguous()) for k, v in splats.items()})
    lrs = {"means": cfg.means_lr * scene_scale, "scales": cfg.scales_lr, "quats": cfg.quats_lr,
           "opacities": cfg.opacities_lr, "sh0": cfg.sh0_lr, "shN": cfg.shN_lr}
    optimizers = {k: torch.optim.Adam([{"params": params[k], "lr": lrs[k] * lr_scale, "name": k}], eps=1e-15 / lr_scale,
                                      betas=(1 - bs * (1 - 0.9), 1 - bs * (1 - 0.999)))
                  for k in params}
    sched = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / steps))
    n_init = len(params["means"])
    if cfg.strategy == "mcmc":
        cap = max(gaussian_budget(cfg, n_views, vb.width, vb.height), int(1.2 * n_init))
        strategy = MCMCStrategy(cap_max=cap, refine_start_iter=min(500, steps // 10),
                                refine_stop_iter=int(steps * 0.85), refine_every=100)
        state = strategy.initialize_state()
    else:
        cap = None
        strategy = DefaultStrategy(absgrad=True, grow_grad2d=0.0008, refine_start_iter=min(500, steps // 10),
                                   refine_stop_iter=int(steps * 0.6), reset_every=max(3000, steps // 4),
                                   refine_every=100)
        state = strategy.initialize_state(scene_scale=scene_scale)
    strategy.check_sanity(params, optimizers)

    # per-view pose residual and affine colour as sparse embeddings with lazy Adam: only the views of the batch move.
    # With dense Adam every view kept moving for ~30 steps after each time it was sampled, by ~30 x lr whatever its
    # gradient, a random walk of about a degree over a run: harmless with 150 views per chunk, ruinous with 1000
    # (the runs with captured frames collapsed)
    pose_emb = torch.nn.Embedding(n_views, 9, sparse=True).to(dev)
    torch.nn.init.zeros_(pose_emb.weight)
    app_emb = torch.nn.Embedding(n_views, 12, sparse=True).to(dev)
    with torch.no_grad():
        app_emb.weight.copy_(torch.eye(3, 4, device=dev).reshape(1, 12).repeat(n_views, 1))
    opt_pose = torch.optim.SparseAdam(list(pose_emb.parameters()), lr=cfg.pose_lr * lr_scale) if cfg.pose_opt else None
    opt_app = torch.optim.SparseAdam(list(app_emb.parameters()), lr=cfg.app_lr * lr_scale) if cfg.app_opt else None

    depth_ok = vb.depths is not None and bool((vb.depths > 0).any())
    zmed = float(vb.depths[vb.depths > 0].float().median()) if depth_ok else 1.0
    eye34 = torch.eye(3, 4, device=dev)
    t0 = time.time()
    g = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(n_views, generator=g)
    pos = 0
    hist = []
    for step in range(steps):
        if pos + bs > n_views:
            order = torch.randperm(n_views, generator=g)
            pos = 0
        idx = order[pos:pos + bs].to(dev)
        pos += bs
        T = vb.T_wc[idx]
        if cfg.pose_opt:
            T = T @ delta_transform(pose_emb(idx))
        sh_deg = min(cfg.sh_degree, step // max(1, cfg.sh_degree_interval))
        rgb, ed, alpha, info = render(params, T, vb.Ks[idx], vb.width, vb.height, sh_deg)
        if cfg.app_opt:
            A_b = app_emb(idx).view(-1, 3, 4)
            rgb = apply_appearance(rgb, A_b)
        gt = vb.images[idx].float() / 255.0
        strategy.step_pre_backward(params, optimizers, state, step, info)
        l1 = (rgb - gt).abs().mean()
        lssim = 1 - ssim(rgb, gt)
        loss = (1 - cfg.ssim_lambda) * l1 + cfg.ssim_lambda * lssim
        ld = torch.zeros((), device=dev)
        if depth_ok and cfg.depth_lambda > 0:
            d = vb.depths[idx].float()[..., None]
            # where the Gaussians cover the pixel (the expected depth of a nearly empty pixel is noise), residuals
            # in inverse depth (relative to the median depth) clamped at 1: one floater against the camera must not
            # dominate the batch
            m = (d > 0) & (alpha.detach() > 0.5)
            if m.any():
                inv_r = zmed / ed.clamp(min=1e-2 * zmed)
                inv_g = zmed / d.clamp(min=1e-2 * zmed)
                ld = ((inv_r - inv_g).abs().clamp(max=1.0) * m).sum() / m.sum()
                lam = cfg.depth_lambda * (0.1 ** (step / steps))
                loss = loss + lam * ld
        if cfg.strategy == "mcmc":
            loss = loss + cfg.opacity_reg * torch.sigmoid(params["opacities"]).mean() \
                + cfg.scale_reg * torch.exp(params["scales"]).mean()
        if cfg.app_opt:
            loss = loss + cfg.app_reg * ((A_b - eye34) ** 2).mean()
        if cfg.pose_opt and cfg.pose_reg > 0:
            loss = loss + cfg.pose_reg * (pose_emb(idx) ** 2).sum()
        if not torch.isfinite(loss):             # a degenerate batch: skip it rather than poison the optimiser state
            for o in optimizers.values():
                o.zero_grad(set_to_none=True)
            for o in (opt_pose, opt_app):
                if o is not None:
                    o.zero_grad(set_to_none=True)
            continue
        loss.backward()
        for o in optimizers.values():
            o.step()
            o.zero_grad(set_to_none=True)
        for o in (opt_pose, opt_app):
            if o is not None:
                o.step()
                o.zero_grad(set_to_none=True)
        if cfg.strategy == "mcmc":
            strategy.step_post_backward(params, optimizers, state, step, info, lr=sched.get_last_lr()[0])
        else:
            strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
        sched.step()
        if (step + 1) % cfg.log_every == 0 or step == steps - 1:
            with torch.no_grad():
                p = psnr(rgb.detach(), gt)
            hist.append({"step": step + 1, "loss": float(loss), "psnr": p, "depth_l1": float(ld),
                         "n": len(params["means"]), "s": round(time.time() - t0, 1)})
            log(f"  step {step + 1}/{steps}  loss {float(loss):.4f}  psnr {p:.2f}  depth {float(ld):.4f}  "
                f"#G {len(params['means'])}  {time.time() - t0:.0f}s")
    torch.cuda.synchronize()
    dt = time.time() - t0
    with torch.no_grad():
        D = delta_transform(pose_emb.weight).cpu().numpy() if cfg.pose_opt else np.tile(np.eye(4), (n_views, 1, 1))
        A = app_emb.weight.detach().view(-1, 3, 4).cpu().numpy()
    out = {k: v.detach() for k, v in params.items()}
    return TrainResult(out, {vid: D[i] for i, vid in enumerate(vb.ids)}, {vid: A[i] for i, vid in enumerate(vb.ids)},
                       {"steps": steps, "batch": bs, "train_s": round(dt, 1), "n_init": n_init, "n_final": len(out["means"]),
                        "cap": cap, "n_views": n_views, "history": hist, "zmed": zmed})


def align_pose(splats, T_wc: torch.Tensor, K: torch.Tensor, width: int, height: int, gt: torch.Tensor, sh_degree: int,
               A: Optional[torch.Tensor] = None, iters: int = 100, lr: float = 2e-3) -> torch.Tensor:
    """Test-time pose alignment of one view: the pose refined photometrically against the frozen Gaussians (the
    usual protocol when test poses come from a different estimate than the reconstruction's own)."""
    d = torch.zeros(1, 9, device=T_wc.device, requires_grad=True)
    opt = torch.optim.Adam([d], lr=lr)
    frozen = {k: v.detach() for k, v in splats.items()}
    best, best_loss = T_wc[None].clone(), float("inf")
    for it in range(iters):
        T = T_wc[None] @ delta_transform(d)
        rgb, _, _, _ = render(frozen, T, K[None], width, height, sh_degree, render_mode="RGB")
        if A is not None:
            rgb = apply_appearance(rgb, A[None])
        loss = (rgb - gt[None]).abs().mean()
        if float(loss) < best_loss:
            best_loss, best = float(loss), T.detach().clone()
        opt.zero_grad()
        loss.backward()
        opt.step()
        for gq in opt.param_groups:
            gq["lr"] = lr * (0.1 ** ((it + 1) / iters))
    return best[0]
