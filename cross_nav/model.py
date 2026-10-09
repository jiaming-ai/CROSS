"""Frozen VGGT-Omega (fine-tuned checkpoint with its scale head) + the trainable BEV and gravity heads.

The backbone runs exactly as in CROSS (aggregator in bf16 autocast, camera / dense / scale heads in fp32) and never
receives gradients; `backbone()` returns everything the BEV head reads: the cached aggregator layers, the DINO patch
features, the predicted cameras and depth, and the scale head's log scale.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from vggt_ft.heads import DenseScaleHead, MultiLayerScaleHead
from vggt_ft.model import VGGTOmegaFT
from vggt_omega.utils.pose_enc import encoding_to_camera

from .head import GravityHead, build_bev_head


class NavModel(nn.Module):
    def __init__(self, checkpoint: str, bev_cfg: dict | None = None, gravity_cfg: dict | None = None):
        super().__init__()
        sd = torch.load(checkpoint, map_location="cpu", weights_only=False)
        shc = sd.get("scale_head_cfg") if isinstance(sd, dict) else None
        self.backbone_model = VGGTOmegaFT(scale_head=shc is not None, covis_head=False, scale_head_cfg=shc)
        msd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
        missing, _ = self.backbone_model.load_state_dict(msd, strict=False)
        missing = [m for m in missing if not m.startswith(("covis_head.", "scale_encoder."))]
        if missing:
            raise RuntimeError(f"{checkpoint}: missing {missing[:5]} ({len(missing)})")
        self.backbone_model.requires_grad_(False).eval()
        self.backbone_model.aggregator.to(torch.bfloat16)
        self.bev_head = build_bev_head(bev_cfg)
        self.gravity_head = GravityHead(**(gravity_cfg or {}))

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone_model.eval()
        return self

    @torch.no_grad()
    def backbone(self, images: torch.Tensor) -> dict:
        """images (B,S,3,H,W) in [0,1]."""
        m = self.backbone_model
        agg = m.aggregator
        B, S, _, H, W = images.shape
        dino = {}
        hook = agg.patch_embed.register_forward_hook(
            lambda mod, i, o: dino.__setitem__("x", o["x_norm_patchtokens"] if isinstance(o, dict) else o))
        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                tokens_list, pts = agg(images.to(torch.bfloat16))
        finally:
            hook.remove()
        out = {"tokens_list": tokens_list, "pts": pts, "patch_hw": (H // agg.patch_size, W // agg.patch_size),
               "dino": dino["x"].reshape(B, S, -1, dino["x"].shape[-1])}
        with torch.autocast(device_type="cuda", enabled=False):
            tl32 = [t.float() if t is not None else None for t in tokens_list]
            pose_enc = m.camera_head(tl32, patch_token_start=pts)
            depth, conf = m.dense_head(tl32, images=images.float(), patch_token_start=pts)
            out["pose_enc"], out["depth"], out["depth_conf"] = pose_enc, depth[..., 0], conf
            ext, intr = encoding_to_camera(pose_enc, (H, W))
            out["extrinsics"], out["intrinsics"] = ext, intr
            if isinstance(m.scale_head, (MultiLayerScaleHead, DenseScaleHead)):
                focal = None
                if m.scale_head.canonical_f is not None:
                    focal = 0.5 / torch.tan(pose_enc[..., 8].float() / 2)
                res = m.scale_head(tl32, pts, out["patch_hw"], out["dino"].float(), focal)
                out["log_scale_head"] = res[0] if isinstance(res, tuple) else res
            elif m.scale_head is not None:
                out["log_scale_head"] = m.scale_head(tl32[-1], pts)
        return out

    def heads(self, bb: dict, log_scale: torch.Tensor, up_cam: torch.Tensor, frame_valid=None):
        logits = self.bev_head(bb["tokens_list"], bb["pts"], bb["patch_hw"], bb["dino"], bb["depth"],
                               bb["depth_conf"], bb["extrinsics"], bb["intrinsics"], log_scale, up_cam, frame_valid)
        return logits

    def gravity(self, bb: dict) -> torch.Tensor:
        return self.gravity_head(bb["tokens_list"], bb["pts"])
