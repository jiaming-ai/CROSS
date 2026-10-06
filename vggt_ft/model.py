"""VGGT-Omega with a metric-scale head and a covisibility head, for training.

`VGGTOmegaFT` subclasses the released `VGGTOmega`, so its state dict is the released one plus `scale_head.*` and
`covis_head.*`: a fine-tuned checkpoint loads into the plain `VGGTOmega` with `strict=False` (CROSS does this), and the
released checkpoint loads into `VGGTOmegaFT` with the new heads at their initialisation.

Training additions: gradient checkpointing of the transformer blocks (patched per block instance, so the module tree
and the parameter names stay those of the release), a frozen-patch-embedding fast path, and a dense head that can run
on a subset of the frames (long windows) and is checkpointed per chunk of frames.
"""
from __future__ import annotations

import fnmatch
import types

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from vggt_omega.models import VGGTOmega

from .heads import CovisHead, DenseScaleHead, MultiLayerScaleHead, build_scale_head
from .scale_encoder import ScaleEncoder


def _checkpointed(module: nn.Module):
    """Make `module(...)` run under activation checkpointing while training with gradients enabled."""
    plain = module.forward

    def forward(self, *args):
        if self.training and torch.is_grad_enabled():
            return checkpoint(plain, *args, use_reentrant=False)
        return plain(*args)

    module.forward = types.MethodType(forward, module)


class VGGTOmegaFT(VGGTOmega):
    NEW_PREFIXES = ("scale_head.", "covis_head.", "scale_encoder.")

    def __init__(self, scale_head: bool = True, covis_head: bool = True, scale_head_cfg: dict | None = None):
        super().__init__()
        cfg = dict(scale_head_cfg or {})
        enc_cfg = cfg.pop("encoder", None)        # scale_head_cfg.encoder: LoRA scale encoder (dense head only)
        if enc_cfg:
            cfg["enc_dim"] = self.aggregator.patch_embed.embed_dim
        self.scale_head = build_scale_head(cfg) if scale_head else None
        self.scale_encoder = ScaleEncoder(self.aggregator.patch_embed, **enc_cfg) if (scale_head and enc_cfg) else None
        self.covis_head = CovisHead() if covis_head else None
        self.dense_chunk = 4          # frames per checkpointed dense-head chunk while training

    # ------------------------------------------------------------------ weights
    def load_weights(self, path: str, verbose: bool = True):
        sd = torch.load(path, map_location="cpu", weights_only=False)   # no mmap: SIGBUS risk on a FUSE mount
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        # a head of another architecture in the checkpoint (e.g. a single-layer scale head) is not loaded
        own = self.state_dict()
        skipped = sorted({k.split(".")[0] for k, v in sd.items() if k in own and own[k].shape != v.shape})
        sd = {k: v for k, v in sd.items() if k not in own or own[k].shape == v.shape}
        if skipped and verbose:
            print(f"[VGGTOmegaFT] {path}: shapes differ, not loaded: {skipped}")
        missing, unexpected = self.load_state_dict(sd, strict=False)
        new = sorted({m.split(".")[0] for m in missing if m.startswith(self.NEW_PREFIXES)})
        missing = [m for m in missing if not m.startswith(self.NEW_PREFIXES)]
        unexpected = [u for u in unexpected if not u.startswith(("text_alignment_head.",) + self.NEW_PREFIXES)]
        if missing or unexpected:
            raise RuntimeError(f"{path}: missing {missing[:5]} ({len(missing)}), unexpected {unexpected[:5]}")
        if verbose and new:
            print(f"[VGGTOmegaFT] {path}: no weights for {new} (initialised)")
        self.sync_scale_encoder()
        return self

    def sync_scale_encoder(self):
        """The scale encoder's base weights = the (frozen) patch embedding's; only its adapters are its own."""
        if getattr(self, "scale_encoder", None) is not None:
            self.scale_encoder.base_state_from(self.aggregator.patch_embed)

    def set_gradient_checkpointing(self, patch_embed: bool = True):
        agg = self.aggregator
        for blk in list(agg.frame_blocks) + list(agg.inter_frame_blocks):
            _checkpointed(blk)
        if patch_embed:
            for blk in agg.patch_embed.blocks:
                _checkpointed(blk)
        for blk in self.camera_head.trunk:
            _checkpointed(blk)
        if self.scale_encoder is not None:
            for blk in self.scale_encoder.dino.blocks:
                _checkpointed(blk)

    def set_trainable(self, patterns: list[str]) -> int:
        """fnmatch patterns over parameter names ('all' = everything); returns the number of trainable parameters."""
        n = 0
        for name, p in self.named_parameters():
            p.requires_grad = "all" in patterns or any(fnmatch.fnmatch(name, pat) for pat in patterns)
            n += p.numel() if p.requires_grad else 0
        return n

    # ------------------------------------------------------------------ forward
    def forward(self, images: torch.Tensor, depth_frames: tuple[int, int] | None = None,
                need_depth: bool = True, intrinsics: torch.Tensor | None = None) -> dict:
        """images (B, S, 3, H, W) in [0, 1].  depth_frames = (start, end): run the dense head on these frames only.
        intrinsics (B, S, 3, 3) at image resolution, optional: the known focal lengths for a canonical-camera scale head
        (without them it uses the model's predicted FoV).
        Returns pose_enc (B,S,9), depth (B,S',H,W,1), depth_conf (B,S',H,W), log_scale (B,), covis_logits (B,S,S),
        camera_and_register_tokens (B,S,17,2048); a dense scale head adds scale_r / scale_w (B,S,h,w)."""
        if images.dim() == 4:
            images = images.unsqueeze(0)
        agg = self.aggregator
        pe_frozen = not any(p.requires_grad for p in agg.patch_embed.parameters())
        amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        multilayer = isinstance(self.scale_head, (MultiLayerScaleHead, DenseScaleHead))
        dino_out = {}
        hook = None
        if multilayer and any(l < 0 for l in self.scale_head.layers):
            # the patch-embedding output (DINO features) for the multi-layer scale head
            hook = agg.patch_embed.register_forward_hook(
                lambda m, i, o: dino_out.__setitem__("x", o["x_norm_patchtokens"] if isinstance(o, dict) else o))
        with torch.autocast(device_type="cuda", dtype=amp):
            if pe_frozen and self.training:
                # no activations of the frozen DINO embedding are kept: run it without grad, feed its tokens back
                B, S = images.shape[:2]
                with torch.no_grad():
                    x = (images - agg._resnet_mean) / agg._resnet_std
                    pt = agg.patch_embed(x.view(B * S, *images.shape[2:]))
                    pt = pt["x_norm_patchtokens"] if isinstance(pt, dict) else pt
                dino_out["x"] = pt
                orig = agg.patch_embed.forward
                agg.patch_embed.forward = lambda _x: pt
                try:
                    tokens_list, pts = agg(images)
                finally:
                    agg.patch_embed.forward = orig
            else:
                tokens_list, pts = agg(images)
        if hook is not None:
            hook.remove()
        final = tokens_list[-1]
        out = {"camera_and_register_tokens": final[:, :, :pts]}
        with torch.autocast(device_type="cuda", enabled=False):
            out["pose_enc"] = self.camera_head(tokens_list, patch_token_start=pts)
            if need_depth:
                S = images.shape[1]
                s0, s1 = depth_frames if depth_frames is not None else (0, S)
                chunk = self.dense_chunk if (self.training and torch.is_grad_enabled()) else S
                deps, confs = [], []
                for a in range(s0, s1, chunk):
                    b = min(a + chunk, s1)
                    if self.training and torch.is_grad_enabled():
                        d, c = checkpoint(self.dense_head._forward_impl, tokens_list, images, pts, a, b,
                                          use_reentrant=False)
                    else:
                        d, c = self.dense_head._forward_impl(tokens_list, images, pts, a, b)
                    deps.append(d)
                    confs.append(c)
                out["depth"] = torch.cat(deps, 1)
                out["depth_conf"] = torch.cat(confs, 1)
            if multilayer:
                B, S = images.shape[:2]
                patch_hw = (images.shape[-2] // agg.patch_size, images.shape[-1] // agg.patch_size)
                dino = dino_out["x"].reshape(B, S, -1, dino_out["x"].shape[-1]) if "x" in dino_out else None
                focal = None
                if self.scale_head.canonical_f is not None:
                    focal = (intrinsics[..., 0, 0] / images.shape[-1] if intrinsics is not None
                             else 0.5 / torch.tan(out["pose_enc"][..., 8].detach().float() / 2))
                kw = {}
                if self.scale_encoder is not None:
                    # the scale encoder sees the first k frames (at inference: frame 0, one extra DINO pass)
                    k = min(self.scale_encoder.frames, S)
                    xe = (images[:, :k] - agg._resnet_mean) / agg._resnet_std
                    with torch.autocast(device_type="cuda", dtype=amp):
                        enc = self.scale_encoder(xe.reshape(B * k, *images.shape[2:]))
                    kw["enc"] = enc.reshape(B, k, -1, enc.shape[-1])
                res = self.scale_head(tokens_list, pts, patch_hw, dino, focal, **kw)
                if isinstance(res, tuple):
                    out["log_scale"], out["scale_r"], out["scale_w"] = res
                else:
                    out["log_scale"] = res
            elif self.scale_head is not None:
                fov_w = out["pose_enc"][:, 0, 8].detach() if getattr(self.scale_head, "canonical_f", None) else None
                out["log_scale"] = self.scale_head(final, pts, fov_w)
            if self.covis_head is not None:
                out["covis_logits"] = self.covis_head(final[:, :, :pts])
        return out


def heads_state_dict(model: VGGTOmegaFT) -> dict:
    """Weights of the new heads only (small file to ship next to any VGGT-Omega checkpoint)."""
    return {k: v for k, v in model.state_dict().items() if k.startswith(VGGTOmegaFT.NEW_PREFIXES)}
