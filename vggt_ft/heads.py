"""Heads added on top of VGGT-Omega's final-layer tokens.

Both heads read only `aggregated_tokens_list[-1]` (camera token, 16 register tokens and patch tokens of every frame,
2 x 1024 channels), so an inference pipeline can run them on the output of the released model
(`VGGTOmega.forward` returns `camera_and_register_tokens`; the scale head also uses the mean patch token of each frame).
Depends on torch only.

ScaleHead : one number per window, log s, such that  metric depth = s * predicted depth  (and metric camera translation
            = s * predicted translation).  Attention pooling over all frames' camera / register / mean-patch tokens.
CovisHead : pairwise covisibility logits (S x S): the overlap fraction of two views, the quantity CROSS thresholds to
            accept a relative pose.  Same module as the underwater fine-tune (cross-uw `covis_head.py`).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class ScaleHead(nn.Module):
    def __init__(self, dim_in: int = 2048, dim: int = 512, depth: int = 2, num_heads: int = 8,
                 num_tokens: int = 18, init_log_scale: float = math.log(2.0), dropout: float = 0.0,
                 canonical_hfov: float | None = None):
        super().__init__()
        # canonical_hfov (deg): predict the scale of a canonical camera with this horizontal field of view and convert
        # with the model's own focal length (Depth Anything 3's canonical-focal transform): log s = head + log(f / f_c)
        self.canonical_f = None if canonical_hfov is None else 0.5 / math.tan(math.radians(canonical_hfov) / 2)
        self.proj = nn.Sequential(nn.LayerNorm(dim_in), nn.Linear(dim_in, dim))
        self.type_embed = nn.Parameter(torch.zeros(1, 1, num_tokens, dim))   # camera, registers, mean patch
        self.ref_embed = nn.Parameter(torch.zeros(1, 1, 1, dim))             # added to the reference frame (frame 0)
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.type_embed, std=0.02)
        nn.init.normal_(self.ref_embed, std=0.02)
        nn.init.normal_(self.query, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, num_heads, 4 * dim, dropout=dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.constant_(self.out[-1].bias, init_log_scale)

    def forward(self, tokens: torch.Tensor, patch_token_start: int, fov_w: torch.Tensor | None = None) -> torch.Tensor:
        """tokens (B, S, T, C) final-layer tokens -> (B,) log scale.  fov_w (B,): predicted horizontal field of view
        (radians) of frame 0, used with canonical_hfov."""
        tokens = tokens.float()
        cam_reg = tokens[:, :, :patch_token_start]
        patch_mean = tokens[:, :, patch_token_start:].mean(2, keepdim=True)
        x = self.proj(torch.cat([cam_reg, patch_mean], 2)) + self.type_embed
        x = torch.cat([x[:, :1] + self.ref_embed, x[:, 1:]], 1)
        B, S, T, D = x.shape
        x = torch.cat([self.query.expand(B, -1, -1), x.reshape(B, S * T, D)], 1)
        x = self.blocks(x)
        log_s = self.out(x[:, 0]).squeeze(-1)
        if self.canonical_f is not None and fov_w is not None:
            f_norm = 0.5 / torch.tan(fov_w.float().clamp(0.05, 3.0) / 2)       # focal / image width
            log_s = log_s + torch.log(f_norm / self.canonical_f)
        return log_s


class MultiLayerScaleHead(nn.Module):
    """Scale head over several depths of the network: the DINO patch features (layer -1, before the aggregator) and
    the aggregator layers kept for the dense head (4, 11, 17, 23).  The last layers of a model trained on
    scale-normalised geometry are scale-invariant by design, while earlier layers keep semantics (what objects are and
    how big they usually are), the cue a monocular scale estimate needs.  Per frame and layer it reads the camera
    token, the 16 register tokens and a grid x grid average pooling of the patch tokens (spatial layout kept), projects
    each layer with its own LayerNorm + Linear, and pools all frames' tokens with a learned query.

    forward(tokens_list, patch_token_start, patch_hw, dino=None) -> (B,) log scale; tokens_list[l] is (B,S,T,2048) for
    the cached layers, dino (B,S,P,1024) the patch-embedding output."""

    def __init__(self, layers=(-1, 4, 11, 17, 23), dim_agg: int = 2048, dim_dino: int = 1024, dim: int = 512,
                 depth: int = 2, num_heads: int = 8, grid: int = 4, num_special: int = 17,
                 init_log_scale: float = math.log(2.0), dropout: float = 0.0, arch: str = "multilayer",
                 canonical_hfov: float | None = None):
        super().__init__()
        self.layers, self.grid, self.num_special = list(layers), grid, num_special
        # canonical_hfov (deg): the head predicts the scale a camera with this horizontal FoV would see and the window's
        # focal lengths convert it (log s += mean log(f / f_c)); f = the known (calibrated) focal when the caller passes
        # intrinsics, else the model's own predicted FoV.  Metric3D / Depth Anything 3 canonical-camera transform.
        self.canonical_f = None if canonical_hfov is None else 0.5 / math.tan(math.radians(canonical_hfov) / 2)
        self.proj = nn.ModuleDict({str(l): nn.Sequential(nn.LayerNorm(dim_dino if l < 0 else dim_agg),
                                                         nn.Linear(dim_dino if l < 0 else dim_agg, dim))
                                   for l in self.layers})
        n_tok = {l: grid * grid + (0 if l < 0 else num_special) for l in self.layers}
        self.pos = nn.ParameterDict({str(l): nn.Parameter(torch.zeros(1, 1, n_tok[l], dim)) for l in self.layers})
        for p in self.pos.values():
            nn.init.normal_(p, std=0.02)
        self.ref_embed = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.ref_embed, std=0.02)
        nn.init.normal_(self.query, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, num_heads, 4 * dim, dropout=dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.constant_(self.out[-1].bias, init_log_scale)

    def _grid(self, patches, patch_hw):
        B, S, P, C = patches.shape
        h, w = patch_hw
        x = patches.reshape(B * S, h, w, C).permute(0, 3, 1, 2)
        x = torch.nn.functional.adaptive_avg_pool2d(x, self.grid)
        return x.flatten(2).transpose(1, 2).reshape(B, S, self.grid * self.grid, C)

    def forward(self, tokens_list, patch_token_start: int, patch_hw, dino=None, focal=None) -> torch.Tensor:
        """focal (B, S): focal length / image width of every frame (used with canonical_hfov)."""
        parts = []
        for l in self.layers:
            if l < 0:
                x = self._grid(dino.float(), patch_hw)
            else:
                t = tokens_list[l].float()
                x = torch.cat([t[:, :, :patch_token_start], self._grid(t[:, :, patch_token_start:], patch_hw)], 2)
            parts.append(self.proj[str(l)](x) + self.pos[str(l)])
        x = torch.cat(parts, 2)                                  # (B, S, tokens per frame, dim)
        x = torch.cat([x[:, :1] + self.ref_embed, x[:, 1:]], 1)
        B, S, T, D = x.shape
        x = torch.cat([self.query.expand(B, -1, -1), x.reshape(B, S * T, D)], 1)
        log_s = self.out(self.blocks(x)[:, 0]).squeeze(-1)
        if self.canonical_f is not None and focal is not None:
            log_s = log_s + torch.log(focal.float().clamp(min=0.05) / self.canonical_f).mean(1)
        return log_s


class DenseScaleHead(nn.Module):
    """Dense scale votes: every patch of every frame predicts the log ratio r of metric to predicted depth at that
    patch plus a confidence logit w; the window's log scale is the softmax(w)-weighted mean of r over all patches of
    all frames.  Trained with a dense loss on r (one target per patch with GT depth, ~1000x more supervision per window
    than the single window scale) and the window loss on log s.  Per frame, the patch tokens of the DINO features and
    the aggregator layers 4, 11, 17, 23 are projected and summed at full patch resolution (no pooling: the apparent size
    of objects is the monocular scale cue), a depthwise 3x3 convolution adds position, and `depth` transformer blocks
    run over the patches plus the final layer's camera / register tokens (multi-view context).

    canonical_hfov: as MultiLayerScaleHead, applied per frame to r.
    forward(...) -> (log_s (B,), r (B,S,h,w), w (B,S,h,w))."""

    def __init__(self, layers=(-1, 4, 11, 17, 23), dim_agg: int = 2048, dim_dino: int = 1024, dim: int = 256,
                 depth: int = 2, num_heads: int = 4, num_special: int = 17, init_log_scale: float = math.log(2.0),
                 dropout: float = 0.0, arch: str = "dense", canonical_hfov: float | None = None,
                 enc_dim: int | None = None):
        super().__init__()
        self.layers, self.num_special = list(layers), num_special
        # enc_dim: patch tokens of a trainable scale encoder (scale_encoder.py) for the first frames, added per patch
        self.proj_enc = None if enc_dim is None else nn.Sequential(nn.LayerNorm(enc_dim), nn.Linear(enc_dim, dim))
        self.enc_flag = None if enc_dim is None else nn.Parameter(torch.zeros(dim))
        self.canonical_f = None if canonical_hfov is None else 0.5 / math.tan(math.radians(canonical_hfov) / 2)
        self.proj = nn.ModuleDict({str(l): nn.Sequential(nn.LayerNorm(dim_dino if l < 0 else dim_agg),
                                                         nn.Linear(dim_dino if l < 0 else dim_agg, dim))
                                   for l in self.layers})
        self.special = nn.Sequential(nn.LayerNorm(dim_agg), nn.Linear(dim_agg, dim))
        self.special_pos = nn.Parameter(torch.zeros(1, num_special, dim))
        nn.init.normal_(self.special_pos, std=0.02)
        self.pos_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        layer = nn.TransformerEncoderLayer(dim, num_heads, 4 * dim, dropout=dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2))
        nn.init.zeros_(self.out[-1].weight)
        with torch.no_grad():
            self.out[-1].bias.copy_(torch.tensor([init_log_scale, 0.0]))

    def forward(self, tokens_list, patch_token_start: int, patch_hw, dino=None, focal=None, enc=None):
        """enc (B, k, P, enc_dim): scale-encoder patch tokens of the first k frames."""
        h, w = patch_hw
        B, S = tokens_list[-1].shape[:2]
        x = 0
        for l in self.layers:
            t = dino if l < 0 else tokens_list[l][:, :, patch_token_start:]
            x = x + self.proj[str(l)](t.float().reshape(B * S, h * w, -1))
        if enc is not None and self.proj_enc is not None:
            k = enc.shape[1]
            x = x.reshape(B, S, h * w, -1)
            x = torch.cat([x[:, :k] + self.proj_enc(enc.float()) + self.enc_flag, x[:, k:]], 1).reshape(B * S, h * w, -1)
        g = x.transpose(1, 2).reshape(B * S, -1, h, w)
        x = x + self.pos_conv(g).flatten(2).transpose(1, 2)
        sp = self.special(tokens_list[-1][:, :, :patch_token_start].float().reshape(B * S, patch_token_start, -1))
        z = self.blocks(torch.cat([sp + self.special_pos, x], 1))[:, patch_token_start:]
        o = self.out(z).reshape(B, S, h, w, 2)
        r, wl = o[..., 0], o[..., 1]
        if self.canonical_f is not None and focal is not None:
            r = r + torch.log(focal.float().clamp(min=0.05) / self.canonical_f)[:, :, None, None]
        a = torch.softmax(wl.reshape(B, -1), 1)
        return (a * r.reshape(B, -1)).sum(1), r, wl


def build_scale_head(cfg: dict | None):
    cfg = dict(cfg or {})
    if cfg.get("arch") == "multilayer":
        return MultiLayerScaleHead(**cfg)
    if cfg.get("arch") == "dense":
        return DenseScaleHead(**cfg)
    cfg.pop("arch", None)
    return ScaleHead(**cfg)


class CovisHead(nn.Module):
    def __init__(self, dim_in: int = 2048, dim: int = 256):
        super().__init__()
        # per-frame embedding from [camera token, mean register token]
        self.frame = nn.Sequential(nn.LayerNorm(2 * dim_in), nn.Linear(2 * dim_in, 2 * dim), nn.GELU(),
                                   nn.Linear(2 * dim, dim))
        # symmetric pair features [a*b, |a-b|] -> logit
        self.pair = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, 1))
        nn.init.zeros_(self.pair[-1].weight)
        nn.init.constant_(self.pair[-1].bias, -1.0)

    def forward(self, cam_reg: torch.Tensor) -> torch.Tensor:
        """cam_reg (B,S,T,C) final-layer camera (index 0) + register tokens -> (B,S,S) logits, symmetric."""
        cam_reg = cam_reg.float()
        f = self.frame(torch.cat([cam_reg[:, :, 0], cam_reg[:, :, 1:].mean(2)], -1))    # (B,S,D)
        a, b = f[:, :, None], f[:, None]
        return self.pair(torch.cat([a * b, (a - b).abs()], -1)).squeeze(-1)
