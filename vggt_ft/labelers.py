"""Off-the-shelf monocular metric-depth models (pseudo-labelers of metric scale): Depth Anything 3 metric, UniDepth V2,
MoGe-2.  Each is called as  depth = labeler(image (3,H,W) float in [0, 1] on the GPU, K (3,3) at that resolution)  and
returns metric depth (H,W) with NaN where invalid.  Their packages (depth_anything_3, unidepth, moge) must be importable;
weights are read from local folders (DA3METRIC-LARGE/, unidepth-v2-vitl14/, moge-2-vitl-normal/).
"""
import math
from pathlib import Path

import torch
import torch.nn.functional as F

NAN = float("nan")


def _stub(*names):
    """Placeholder modules for imports the labelers make but inference never uses (DA3's export code imports moviepy,
    pycolmap and evo; UniDepth's package init imports wandb), when they are not installed.  Any attribute of a
    placeholder is a dummy class, so `from x import Y` succeeds."""
    import importlib.util
    import sys
    import types

    class _Placeholder(types.ModuleType):
        def __getattr__(self, attr):
            if attr.startswith("__"):
                raise AttributeError(attr)
            return type(attr, (), {})

    for name in names:
        if importlib.util.find_spec(name.split(".")[0]) is not None:
            continue
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            sys.modules.setdefault(".".join(parts[:i]), _Placeholder(".".join(parts[:i])))


class DA3Metric:
    """depth-anything/DA3METRIC-LARGE as CROSS's mono mode uses it (cross/mono/models.py DA3MetricDepth): longer side
    504, ImageNet normalisation, canonical focal 300 at network resolution, sky (> 0.3) invalid."""

    def __init__(self, wdir, device, resolution=504):
        _stub("moviepy.editor", "pycolmap", "evo.core.trajectory")
        from depth_anything_3.api import DepthAnything3
        self.model = DepthAnything3.from_pretrained(wdir).to(device).eval()
        self.res = resolution
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]

    def __call__(self, img, K):
        H, W = img.shape[-2:]
        r = self.res / max(H, W)
        h, w = max(14, round(H * r / 14) * 14), max(14, round(W * r / 14) * 14)
        x = F.interpolate(img[None], size=(h, w), mode="bicubic", align_corners=False)[0].clamp(0, 1)
        out = self.model(((x - self.mean) / self.std)[None, None], export_feat_layers=[])
        d = out["depth"][0, 0].float().squeeze()
        d = d * (0.5 * (K[0, 0] * w / W + K[1, 1] * h / H)) / 300.0
        if "sky" in out:
            d = torch.where(out["sky"][0, 0].float().squeeze() > 0.3, torch.full_like(d, NAN), d)
        return F.interpolate(d[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]


class UniDepth:
    def __init__(self, wdir, device):
        _stub("wandb")
        from unidepth.models import UniDepthV2
        self.model = UniDepthV2.from_pretrained(wdir).to(device).eval()

    def __call__(self, img, K):
        pred = self.model.infer((img * 255).round().to(torch.uint8), K[None].float())
        return pred["depth"][0, 0].float()


class MoGe2:
    def __init__(self, wdir, device):
        from moge.model.v2 import MoGeModel
        self.model = MoGeModel.from_pretrained(str(Path(wdir) / "model.pt")).to(device).eval()

    def __call__(self, img, K):
        fov_x = math.degrees(2 * math.atan(img.shape[-1] / (2 * float(K[0, 0]))))
        out = self.model.infer(img, fov_x=fov_x)
        d = out["depth"].float()
        return torch.where(torch.isfinite(d) & (d > 0), d, torch.full_like(d, NAN))


LABELERS = {"da3metric": (DA3Metric, "DA3METRIC-LARGE"), "unidepth": (UniDepth, "unidepth-v2-vitl14"),
            "moge2": (MoGe2, "moge-2-vitl-normal")}


def load_labelers(names, weights_dir, device="cuda"):
    return {n: LABELERS[n][0](str(Path(weights_dir) / LABELERS[n][1]), device) for n in names}
