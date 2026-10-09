"""BEV head on the frozen VGGT-Omega backbone.

Input: one VGGT pass (frame 0 = the current image, the other frames = references / stereo partner / previous frame), its
multi-layer tokens (DINO patch features + aggregator layers 4, 11, 17, 23, as the scale head), its own predicted depth
and cameras, the pass's metric scale (log s: metric = s x predicted depth; from the stereo baseline, odometry, IMU or
the scale head) and gravity in frame 0's camera (IMU, or the gravity head below).  Output: occupancy logits of an
N x N BEV grid in frame 0's BEV frame (cross_nav/bev.py).

1. Patch features: per frame, the five layers' patch tokens are projected and summed (D channels).
2. Lifting: every patch is lifted at 2 x 2 points with VGGT's predicted depth and cameras, scaled to metres and rotated
   into the BEV frame; each point carries its patch feature plus an embedding of its height, depth, confidence and
   whether it is frame 0's.  Points are averaged into a coarse grid (N / 4); a dense height histogram of all depth
   pixels (confidence-weighted counts per height bin) is rasterised at full resolution (exact geometry).
3. BEV encoder: convolutional stem on the histogram (N -> N / 4) + the lifted features -> tokens at N / 8, transformer
   blocks (self-attention over the BEV grid with learned positions, cross-attention to per-frame context tokens:
   camera / register tokens and pooled patch tokens of the layers), which fill in what no point covers (behind objects,
   outside the view) from scene context.
4. Decoder: upsampling back to N with skips from the stem, 1 logit per cell (occupied vs free).

GravityHead: up vector of frame 0's camera from its camera / register tokens of several layers.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bev import bev_axes


def fourier(x: torch.Tensor, n: int, scale: float) -> torch.Tensor:
    """x (...,) -> (..., 2n) sin / cos features at frequencies 2^k / scale."""
    f = (2.0 ** torch.arange(n, device=x.device, dtype=x.dtype)) * (math.pi / scale)
    a = x[..., None] * f
    return torch.cat([a.sin(), a.cos()], -1)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        g = 8 if cout % 8 == 0 else 1
        self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.GroupNorm(g, cout), nn.GELU(),
                                 nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.GroupNorm(g, cout))
        self.skip = (nn.Identity() if cin == cout and stride == 1 else
                     nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.GroupNorm(g, cout)))

    def forward(self, x):
        return F.gelu(self.net(x) + self.skip(x))


class BEVBlock(nn.Module):
    """Pre-norm transformer block: self-attention over BEV tokens, cross-attention to context tokens, MLP."""

    def __init__(self, dim, heads, mlp=4.0, dropout=0.0):
        super().__init__()
        self.n1, self.n2, self.n3, self.nc = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.sa = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ca = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp)), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(int(dim * mlp), dim))

    def forward(self, x, ctx, ctx_mask=None):
        h = self.n1(x)
        x = x + self.sa(h, h, h, need_weights=False)[0]
        c = self.nc(ctx)
        x = x + self.ca(self.n2(x), c, c, key_padding_mask=ctx_mask, need_weights=False)[0]
        return x + self.mlp(self.n3(x))


class BEVHead(nn.Module):
    def __init__(self, size: int = 400, res: float = 0.05, layers=(-1, 4, 11, 17, 23), dim_agg: int = 2048,
                 dim_dino: int = 1024, feat: int = 256, dim: int = 384, depth: int = 8, heads: int = 8,
                 hbins=(-2.6, 1.0, 24), ctx_grid: int = 4, ctx_layers=(11, 23), num_special: int = 17,
                 dropout: float = 0.0, max_frames: int = 32):
        super().__init__()
        assert size % 8 == 0
        self.size, self.res, self.layers = size, res, list(layers)
        self.hlo, self.hhi, self.nh = hbins
        self.ctx_layers, self.ctx_grid, self.num_special = list(ctx_layers), ctx_grid, num_special
        # 1. patch features
        self.proj = nn.ModuleDict({str(l): nn.Sequential(nn.LayerNorm(dim_dino if l < 0 else dim_agg),
                                                         nn.Linear(dim_dino if l < 0 else dim_agg, feat))
                                   for l in self.layers})
        # 2. point embedding: height (fourier), log depth, confidence, frame-0 flag, sub-patch position
        self.pt_mlp = nn.Sequential(nn.Linear(16 + 2 + 1 + 1, feat), nn.GELU(), nn.Linear(feat, feat))
        self.sub = nn.Parameter(torch.zeros(4, feat))
        self.lift_norm = nn.LayerNorm(feat)
        # 3. stem on the height histogram (N -> N/2 -> N/4), fusion with the lifted features, down to N/8
        c1, c2 = 48, 96
        self.stem0 = ConvBlock(self.nh + 1, 32)                                  # N
        self.stem1 = ConvBlock(32, c1, stride=2)                                 # N/2
        self.stem2 = ConvBlock(c1, c2, stride=2)                                 # N/4
        self.fuse = ConvBlock(c2 + feat + 1, 192)                                # N/4 (+ log point count)
        self.down = ConvBlock(192, dim, stride=2)                                # N/8
        g = size // 8
        self.pos = nn.Parameter(torch.zeros(1, g * g, dim))
        nn.init.normal_(self.pos, std=0.02)
        # context tokens per frame: final camera + registers, ctx_grid^2 pooled patches of each ctx layer
        self.ctx_proj = nn.ModuleDict({str(l): nn.Sequential(nn.LayerNorm(dim_agg), nn.Linear(dim_agg, dim))
                                       for l in self.ctx_layers})
        self.ctx_special = nn.Sequential(nn.LayerNorm(dim_agg), nn.Linear(dim_agg, dim))
        n_ctx = num_special + ctx_grid * ctx_grid * len(self.ctx_layers)
        self.ctx_pos = nn.Parameter(torch.zeros(1, 1, n_ctx, dim))
        self.frame_emb = nn.Parameter(torch.zeros(2, dim))                     # frame 0 / others
        nn.init.normal_(self.ctx_pos, std=0.02)
        nn.init.normal_(self.frame_emb, std=0.02)
        # metric scale and camera pitch / roll are known inputs: also given to the tokens
        self.cond = nn.Sequential(nn.Linear(1 + 3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([BEVBlock(dim, heads, dropout=dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        # 4. decoder
        self.up1 = ConvBlock(dim + 192, 192)                                     # N/4
        self.up2 = ConvBlock(192 + c1, 96)                                       # N/2
        self.up3 = ConvBlock(96 + 32, 48)                                        # N
        self.out = nn.Conv2d(48, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    # ------------------------------------------------------------------------------------------------ lifting
    def lift(self, depth, conf, extrinsics, intrinsics, log_scale, up_cam):
        """Points of every frame in frame 0's BEV frame, metric.  depth / conf (B,S,H,W); extrinsics (B,S,3,4)
        camera-from-frame-0; intrinsics (B,S,3,3); log_scale (B,); up_cam (B,3).  Returns pts (B,S,H,W,3)."""
        B, S, H, W = depth.shape
        dev = depth.device
        ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32) + 0.5,
                                torch.arange(W, device=dev, dtype=torch.float32) + 0.5, indexing="ij")
        fx, fy = intrinsics[..., 0, 0, None, None], intrinsics[..., 1, 1, None, None]
        cx, cy = intrinsics[..., 0, 2, None, None], intrinsics[..., 1, 2, None, None]
        Pc = torch.stack([(xs - cx) / fx * depth, (ys - cy) / fy * depth, depth], -1)        # (B,S,H,W,3)
        R, t = extrinsics[..., :3, :3], extrinsics[..., :3, 3]
        P0 = torch.einsum("bsji,bshwj->bshwi", R, Pc - t[:, :, None, None, :])                # R^T (P - t)
        P0 = P0 * log_scale.exp()[:, None, None, None, None]
        A = bev_axes(up_cam)                                                                 # (B,3,3)
        return torch.einsum("bij,bshwj->bshwi", A, P0)

    def splat(self, pts, w, feats, n_cells, res):
        """Weighted mean of point features per cell.  pts (B,M,3), w (B,M), feats (B,M,C) -> (B,C,n,n), counts (B,1,n,n)."""
        B, M, C = feats.shape
        col = torch.floor(pts[..., 0] / res + n_cells / 2).long()
        row = torch.floor(n_cells / 2 - pts[..., 1] / res).long()
        ok = (col >= 0) & (col < n_cells) & (row >= 0) & (row < n_cells) & (w > 0)
        idx = (row.clamp(0, n_cells - 1) * n_cells + col.clamp(0, n_cells - 1)) + \
            torch.arange(B, device=pts.device)[:, None] * n_cells * n_cells
        w = w * ok
        acc = feats.new_zeros(B * n_cells * n_cells, C)
        acc.index_add_(0, idx.reshape(-1), (feats * w[..., None]).reshape(-1, C))
        cnt = feats.new_zeros(B * n_cells * n_cells)
        cnt.index_add_(0, idx.reshape(-1), w.reshape(-1))
        acc = acc / cnt.clamp(min=1e-6)[:, None]
        return (acc.view(B, n_cells, n_cells, C).permute(0, 3, 1, 2),
                cnt.view(B, 1, n_cells, n_cells))

    def histogram(self, pts, w):
        """Height histogram per cell at full resolution: (B, nh + 1, N, N) = log(1 + weighted count) per height bin
        (heights relative to the camera, hbins) + all heights."""
        B = pts.shape[0]
        N = self.size
        hb = torch.floor((pts[..., 2] - self.hlo) / (self.hhi - self.hlo) * self.nh).long()
        col = torch.floor(pts[..., 0] / self.res + N / 2).long()
        row = torch.floor(N / 2 - pts[..., 1] / self.res).long()
        ok = (col >= 0) & (col < N) & (row >= 0) & (row < N) & (hb >= 0) & (hb < self.nh) & (w > 0)
        base = torch.arange(B, device=pts.device)[:, None] * (self.nh * N * N)
        idx = base + hb.clamp(0, self.nh - 1) * N * N + row.clamp(0, N - 1) * N + col.clamp(0, N - 1)
        h = pts.new_zeros(B * self.nh * N * N)
        h.index_add_(0, idx[ok], w[ok])
        h = h.view(B, self.nh, N, N)
        return torch.log1p(torch.cat([h, h.sum(1, keepdim=True)], 1))

    # ------------------------------------------------------------------------------------------------ forward
    def forward(self, tokens_list, patch_token_start, patch_hw, dino, depth, conf, extrinsics, intrinsics,
                log_scale, up_cam, frame_valid=None):
        """tokens_list: aggregator outputs (index = layer); dino (B,S,P,1024); depth / conf (B,S,H,W) predicted;
        extrinsics (B,S,3,4) / intrinsics (B,S,3,3) predicted (image resolution); log_scale (B,); up_cam (B,3) unit;
        frame_valid (B,S) bool (padding).  Returns logits (B,N,N)."""
        h, w = patch_hw
        B, S = depth.shape[:2]
        dt = torch.float32
        N, Nc = self.size, self.size // 4
        if frame_valid is None:
            frame_valid = torch.ones(B, S, dtype=torch.bool, device=depth.device)
        # 1. patch features (B*S, h*w, feat)
        x = 0
        for l in self.layers:
            t = dino if l < 0 else tokens_list[l][:, :, patch_token_start:]
            x = x + self.proj[str(l)](t.to(dt).reshape(B * S, h * w, -1))
        fpatch = x.view(B, S, h, w, -1)
        f0 = torch.zeros(B, S, device=depth.device, dtype=dt)
        f0[:, 0] = 1
        # 2. lifting at 2 x 2 points per patch; the geometry in fp32 also under autocast
        with torch.autocast(device_type="cuda", enabled=False):
            H, W = depth.shape[-2:]
            depth, extrinsics, intrinsics = depth.float(), extrinsics.float(), intrinsics.float()
            log_scale, up_cam = log_scale.float(), up_cam.float()
            conf_w = (conf.float() - 1).clamp(min=0, max=20) / 20 + 0.05       # VGGT conf >= 1
            d2 = F.adaptive_avg_pool2d(depth.reshape(B * S, 1, H, W), (2 * h, 2 * w)).view(B, S, 2 * h, 2 * w)
            c2 = F.adaptive_avg_pool2d(conf_w.reshape(B * S, 1, H, W), (2 * h, 2 * w)).view(B, S, 2 * h, 2 * w)
            K2 = intrinsics.clone()
            K2[..., 0, :] *= (2 * w) / W
            K2[..., 1, :] *= (2 * h) / H
            P = self.lift(d2, c2, extrinsics, K2, log_scale, up_cam)                       # (B,S,2h,2w,3)
            z = P[..., 2]
            rng = P.norm(dim=-1).clamp(min=1e-3)
            dmet = (d2 * log_scale.exp()[:, None, None, None]).clamp(min=1e-3)
            emb = torch.cat([fourier(z.clamp(-6, 6), 8, 8.0), torch.log(rng)[..., None] / 3,
                             torch.log(dmet)[..., None] / 3, c2[..., None],
                             f0[:, :, None, None, None].expand(-1, -1, 2 * h, 2 * w, 1)], -1)
            pf = fpatch.float().repeat_interleave(2, 2).repeat_interleave(2, 3)            # (B,S,2h,2w,feat)
            sub = self.sub.view(2, 2, -1).repeat(h, w, 1)                                  # (2h,2w,feat)
            pf = self.lift_norm(pf + sub + self.pt_mlp(emb))
            wv = c2 * frame_valid[:, :, None, None].to(dt)
            lifted, cnt = self.splat(P.reshape(B, -1, 3), wv.reshape(B, -1), pf.reshape(B, -1, pf.shape[-1]), Nc,
                                     self.res * 4)
            # dense height histogram from the depth (every second pixel)
            K1 = intrinsics.clone()
            K1[..., :2, :] /= 2
            Pd = self.lift(depth[..., ::2, ::2], None, extrinsics, K1, log_scale, up_cam)
            wd = (conf_w[..., ::2, ::2] * frame_valid[:, :, None, None].to(dt)).reshape(B, -1)
            hist = self.histogram(Pd.reshape(B, -1, 3), wd)
        # 3. encoder
        s0 = self.stem0(hist)
        s1 = self.stem1(s0)
        s2 = self.stem2(s1)
        f = self.fuse(torch.cat([s2, lifted, torch.log1p(cnt)], 1))
        g = self.down(f)                                                        # (B,dim,N/8,N/8)
        gs = g.shape[-1]
        tok = g.flatten(2).transpose(1, 2) + self.pos
        tok = tok + self.cond(torch.cat([log_scale.to(dt)[:, None], up_cam.to(dt)], -1))[:, None]
        # context tokens
        fin = tokens_list[-1]
        ctx = [self.ctx_special(fin[:, :, :patch_token_start].to(dt))]
        for l in self.ctx_layers:
            p = tokens_list[l][:, :, patch_token_start:].to(dt).reshape(B * S, h, w, -1).permute(0, 3, 1, 2)
            p = F.adaptive_avg_pool2d(p, self.ctx_grid).flatten(2).transpose(1, 2).reshape(B, S, -1, p.shape[1])
            ctx.append(self.ctx_proj[str(l)](p))
        ctx = torch.cat(ctx, 2) + self.ctx_pos
        fe = self.frame_emb[(f0 == 0).long()]                                  # (B,S,dim)
        ctx = (ctx + fe[:, :, None]).reshape(B, -1, ctx.shape[-1])
        cmask = (~frame_valid)[:, :, None].expand(-1, -1, ctx.shape[1] // S).reshape(B, -1)
        for blk in self.blocks:
            tok = blk(tok, ctx, cmask)
        g = self.norm(tok).transpose(1, 2).reshape(B, -1, gs, gs)
        # 4. decoder
        y = F.interpolate(g, scale_factor=2, mode="bilinear", align_corners=False)
        y = self.up1(torch.cat([y, f], 1))
        y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)
        y = self.up2(torch.cat([y, s1], 1))
        y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)
        y = self.up3(torch.cat([y, s0], 1))
        return self.out(y)[:, 0]


class GravityHead(nn.Module):
    """Up vector of frame 0's camera (camera coordinates) from its camera token and mean register token of several
    aggregator layers.  Robot / handheld cameras: the image content (floors, walls, vertical edges) fixes pitch / roll."""

    def __init__(self, layers=(11, 17, 23), dim_agg: int = 2048, dim: int = 512):
        super().__init__()
        self.layers = list(layers)
        self.net = nn.Sequential(nn.LayerNorm(2 * dim_agg * len(self.layers)),
                                 nn.Linear(2 * dim_agg * len(self.layers), dim), nn.GELU(),
                                 nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 3))
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor([0.0, -1.0, 0.0]))            # level camera: up = -y

    def forward(self, tokens_list, patch_token_start):
        x = torch.cat([torch.cat([tokens_list[l][:, 0, 0], tokens_list[l][:, 0, 1:patch_token_start].mean(1)], -1)
                       for l in self.layers], -1).float()
        return F.normalize(self.net(x), dim=-1)
