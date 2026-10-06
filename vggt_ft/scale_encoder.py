"""Scale encoder: a LoRA-adapted copy of VGGT-Omega's DINO patch embedding for the metric-scale branch only.

The pose backbone stays frozen (its DINO included); this copy starts from the same weights, gets low-rank adapters on
every attention / MLP linear layer, and runs on the first `frames` frames of a window.  Its patch tokens feed the dense
scale head.  Why: the frozen backbone's features were trained on scale-normalised geometry, and every strong metric
model trains its encoder for metric depth (MoGe-2, Depth Anything 3 metric; Depth Pro's FoV head gains most from its
own trainable encoder).  Inference cost: one extra DINO pass per window (frame 0), a few % of a VGGT-Omega window.
The base weights are copied from the patch embedding at initialisation (`VGGTOmegaFT.sync_scale_encoder`); with LoRA only
the adapters `scale_encoder.*lora_*` are new, with `full: true` the whole copy is fine-tuned and its weights are new.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scaling = alpha / rank
        # the wrapped modules read these off their linear layers (e.g. attention: C = self.qkv.in_features)
        self.in_features, self.out_features = base.in_features, base.out_features

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        return self.base(x) + (x @ self.lora_a.t().to(x.dtype)) @ self.lora_b.t().to(x.dtype) * self.scaling


def add_lora(module: nn.Module, rank: int, alpha: float, names=("qkv", "proj", "fc1", "fc2")) -> int:
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in names:
            setattr(module, name, LoRALinear(child, rank, alpha))
            n += 1
        else:
            n += add_lora(child, rank, alpha, names)
    return n


class ScaleEncoder(nn.Module):
    def __init__(self, patch_embed: nn.Module, rank: int = 32, alpha: float = 32.0, frames: int = 1, full: bool = False):
        super().__init__()
        self.dino = copy.deepcopy(patch_embed)
        # full: the whole copy is fine-tuned (no adapters; its own weights are stored in the checkpoint)
        self.full = full
        self.n_lora = 0 if full else add_lora(self.dino, rank, alpha)
        self.frames = frames

    def base_state_from(self, patch_embed: nn.Module):
        """Copy the (frozen) patch-embedding weights into the base layers of this copy."""
        src = patch_embed.state_dict()
        own = self.dino.state_dict()
        mapped = {}
        for k in own:
            if "lora_" in k:
                continue
            kb = k.replace(".base.", ".")
            if kb in src:
                mapped[k] = src[kb]
        missing = [k for k in own if "lora_" not in k and k not in mapped]
        if missing:
            raise RuntimeError(f"scale encoder: no source for {missing[:5]}")
        self.dino.load_state_dict(mapped, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (N, 3, H, W) normalised as the patch embedding expects -> patch tokens (N, P, C)."""
        out = self.dino(x)
        return out["x_norm_patchtokens"] if isinstance(out, dict) else out
