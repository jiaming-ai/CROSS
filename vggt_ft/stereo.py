"""Metric depth from rectified stereo pairs with FoundationStereo (NVlabs, zero-shot stereo; NVIDIA non-commercial
licence): depth = fx * baseline / disparity, where the scale comes from the stereo baseline, not from a learned prior.

    fs = FoundationStereo(code_dir, ckpt_dir)                  # ckpt_dir holds cfg.yaml + model_best_bp2.pth
    disp, ok = fs.disparity(left, right)                       # (B,H,W) float32, (B,H,W) bool (left-right consistent)
    depth = disparity_to_depth(disp, ok, fx, baseline, min_disp=3.0)

Left-right consistency: the right image's disparity comes from matching the horizontally flipped (right, left) pair
(the same trick as cross-uw's uwvggt/data/stereo.py); a left pixel is kept when the right disparity at its match agrees
within `lr_thresh` px.  A disparity floor keeps far pixels, where a one-pixel error is a large depth error, out.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _stub(*names):
    """Placeholder modules for imports FoundationStereo's Utils.py makes but inference never uses (open3d, ...)."""
    class _Placeholder(types.ModuleType):
        def __getattr__(self, attr):
            if attr.startswith("__"):
                raise AttributeError(attr)
            return type(attr, (), {})

    import importlib.machinery
    for name in names:
        if importlib.util.find_spec(name) is None:
            m = _Placeholder(name)
            m.__spec__ = importlib.machinery.ModuleSpec(name, None)    # torch._dynamo calls find_spec on modules
            sys.modules.setdefault(name, m)


class FoundationStereo:
    def __init__(self, code_dir: str, ckpt_dir: str, valid_iters: int = 32, device: str = "cuda"):
        code_dir, ckpt_dir = Path(code_dir), Path(ckpt_dir)
        _stub("open3d", "joblib", "pandas")
        if str(code_dir) not in sys.path:
            sys.path.insert(0, str(code_dir))
        # FoundationStereo builds DINOv2 with torch.hub.load('facebookresearch/dinov2', ...) (GitHub); use its local copy
        hub_load = torch.hub.load

        def local_hub(repo, model, *args, **kwargs):
            if repo == "facebookresearch/dinov2":
                kwargs.pop("source", None)
                return hub_load(str(code_dir / "dinov2"), model, *args, source="local", **kwargs)
            return hub_load(repo, model, *args, **kwargs)

        torch.hub.load = local_hub
        # its EdgeNeXt feature net is created with pretrained=True (an ImageNet download); the checkpoint overwrites it
        import timm
        create = timm.create_model
        timm.create_model = lambda *a, **k: create(*a, **dict(k, pretrained=False))
        try:
            from omegaconf import OmegaConf
            from core.foundation_stereo import FoundationStereo as FS
            from core.utils.utils import InputPadder
            cfg = OmegaConf.load(ckpt_dir / "cfg.yaml")
            if "vit_size" not in cfg:
                cfg["vit_size"] = "vitl"
            cfg["valid_iters"] = valid_iters
            self.model = FS(cfg)
        finally:
            torch.hub.load = hub_load
            timm.create_model = create
        ckpt = torch.load(ckpt_dir / "model_best_bp2.pth", map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(device).eval()
        self._padder = InputPadder
        self.iters, self.device = valid_iters, device

    @torch.no_grad()
    def _match(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """a, b (B,3,H,W) float 0..255 -> left disparity (B,H,W)."""
        H, W = a.shape[-2:]
        pad = self._padder(a.shape, divis_by=32, force_square=False)
        a, b = pad.pad(a, b)
        with torch.autocast("cuda", dtype=torch.float16):
            d = self.model.forward(a, b, iters=self.iters, test_mode=True)
        return pad.unpad(d.float()).reshape(-1, H, W)

    @torch.no_grad()
    def disparity(self, left: np.ndarray, right: np.ndarray, lr_thresh: float = 1.5):
        """left, right (B,H,W,3) or (H,W,3) uint8 RGB, rectified.  Returns disparity of the left images (B,H,W) and
        the left-right consistency mask (B,H,W)."""
        single = left.ndim == 3
        if single:
            left, right = left[None], right[None]
        a = torch.as_tensor(left, device=self.device).permute(0, 3, 1, 2).float()
        b = torch.as_tensor(right, device=self.device).permute(0, 3, 1, 2).float()
        dl = self._match(a, b)
        dr = self._match(b.flip(-1), a.flip(-1)).flip(-1)          # disparity of the right images
        B, H, W = dl.shape
        xs = torch.arange(W, device=self.device, dtype=torch.float32).view(1, 1, W).expand(B, H, W)
        xr = xs - dl                                                 # matching column in the right image
        grid_x = 2 * xr / (W - 1) - 1
        grid_y = (2 * torch.arange(H, device=self.device, dtype=torch.float32) / (H - 1) - 1).view(1, H, 1).expand(B, H, W)
        dr_at = F.grid_sample(dr[:, None], torch.stack([grid_x, grid_y], -1), align_corners=True,
                              padding_mode="border")[:, 0]
        ok = (dl > 0) & (xr >= 0) & ((dl - dr_at).abs() < lr_thresh)
        dl, ok = dl.cpu().numpy(), ok.cpu().numpy()
        return (dl[0], ok[0]) if single else (dl, ok)


def disparity_to_depth(disp: np.ndarray, ok: np.ndarray, fx: float, baseline: float, min_disp: float = 3.0):
    """Metric z-depth (0 = invalid) of the left image."""
    valid = ok & np.isfinite(disp) & (disp >= min_disp)
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.where(valid, fx * baseline / disp, 0.0)
    return depth.astype(np.float32)
