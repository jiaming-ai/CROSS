"""BEV prediction from the tensors of one VGGT pass (CROSS's backend or any other caller).

    pred = BEVPredictor("heads_012000_ema.pt", "cuda")
    out = pred(tokens, pts, patch_hw, dino, depth, conf, pose_enc, hw, log_scale, up_cam=None)
    out["prob"]   (N, N) P(occupied) in frame 0's BEV frame (cross_nav/bev.py); out["up_cam"] gravity used
    out["uncertainty"] error probability or attenuation score (see uncertainty_kind), or None for an old checkpoint

tokens: {layer: (1, S, T, 2048)} of the cached aggregator layers (4, 11, 17, 23); dino (S, P, 1024) or (1, S, P, 1024);
depth / conf (k, H, W) of the first k views (k <= S: the others give context but are not lifted); pose_enc (1, S, 9);
log_scale: metric = exp(log_scale) x model-gauge depth (CROSS: log of the pass's solved metric scale); up_cam: unit up
vector in frame 0's (model) camera, e.g. from an IMU; None = the gravity head's estimate.
"""
from __future__ import annotations

import math

import torch

from .head import GravityHead, build_bev_head
from .uncertainty import uncertainty_score

LAYERS = (4, 11, 17, 23)


class BEVPredictor:
    def __init__(self, heads_path: str, device: str = "cuda"):
        ck = torch.load(heads_path, map_location="cpu", weights_only=False)
        self.device = device
        self.bev = build_bev_head(ck.get("bev_cfg"))
        self.bev.load_state_dict(ck["bev_head"])
        self.grav = GravityHead(**(ck.get("gravity_cfg") or {}))
        self.grav.load_state_dict(ck["gravity_head"])
        self.bev = self.bev.to(device).eval()
        self.grav = self.grav.to(device).eval()
        self.size, self.res = self.bev.size, self.bev.res

    @torch.inference_mode()
    def __call__(self, tokens: dict, pts: int, patch_hw, dino, depth, conf, pose_enc, hw, log_scale: float,
                 up_cam=None) -> dict:
        from vggt_omega.utils.pose_enc import encoding_to_camera
        S = pose_enc.shape[1]
        tl = [None] * 24
        for l, t in tokens.items():
            tl[l] = t
        if dino.dim() == 3:
            dino = dino[None]
        k = depth.shape[0]
        H, W = hw
        d = depth.new_zeros((1, S, H, W))
        c = depth.new_ones((1, S, H, W))
        d[0, :k] = depth.float()
        if conf is not None and conf.shape[0]:
            c[0, :k] = conf.float()
        lift = torch.zeros(1, S, dtype=torch.bool, device=self.device)
        lift[0, :k] = True
        ext, intr = encoding_to_camera(pose_enc.float(), (H, W))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            up = self.grav(tl, pts).float()
        up_head = up[0].clone()
        if up_cam is not None:
            up = torch.nn.functional.normalize(torch.as_tensor(up_cam, dtype=torch.float32, device=self.device)[None],
                                               dim=-1)
        ls = torch.tensor([float(log_scale)], device=self.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = self.bev(tl, pts, patch_hw, dino, d, c, ext, intr, ls, up, None, lift, return_aux=True)
            logits = pred["logits"].float()
        return {"prob": torch.sigmoid(logits[0]),
                "uncertainty": (uncertainty_score(pred["error_logits"][0], self.bev.uncertainty_mode)
                                if pred["error_logits"] is not None else None), "up_cam": up[0], "up_head": up_head, "log_scale": float(log_scale),
                "res": self.res, "size": self.size, "uncertainty_kind": self.bev.uncertainty_mode}

    @staticmethod
    def log_metric_scale(metric: float) -> float:
        return math.log(max(float(metric), 1e-9))
