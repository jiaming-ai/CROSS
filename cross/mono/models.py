"""Adapters for released Depth Anything 3 checkpoints (loaded lazily)."""

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .geometry import homogeneous


@dataclass
class GeometryPrediction:
    depth: np.ndarray
    confidence: np.ndarray
    extrinsics: np.ndarray
    intrinsics: np.ndarray


class DA3Geometry:
    def __init__(self, model_id="depth-anything/DA3-SMALL", device="cuda", resolution=336):
        from depth_anything_3.api import DepthAnything3

        self.model_id, self.device, self.resolution = model_id, device, resolution
        self.model = DepthAnything3.from_pretrained(model_id).to(device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]

    def prepare(self, rgb):
        h, w = rgb.shape[:2]
        ratio = self.resolution / max(h, w)
        size = (max(14, round(w * ratio / 14) * 14), max(14, round(h * ratio / 14) * 14))
        small = cv2.resize(rgb, size, interpolation=cv2.INTER_CUBIC)
        image = torch.as_tensor(small.copy(), device=self.device).permute(2, 0, 1).float() / 255.0
        return (image - self.mean) / self.std

    @torch.inference_mode()
    def predict(self, images):
        result = self.model(torch.stack(images)[None], ref_view_strategy="first")
        def array(key):
            return result[key][0].float().cpu().numpy()
        depth = array("depth")
        if depth.ndim == 4:
            depth = depth.squeeze(-1)
        conf = array("depth_conf") if "depth_conf" in result else np.ones_like(depth)
        return GeometryPrediction(depth, conf, homogeneous(array("extrinsics")), array("intrinsics"))


class DA3MetricDepth(DA3Geometry):
    """Canonical focal output is converted to metres exactly once.

    The released metric model uses focal=300 at NETWORK resolution. Known
    calibration is resized with the image; using original-resolution focal
    on resized output would silently bias all scale observations.
    """

    @torch.inference_mode()
    def predict_metric(self, rgb, K, output_shape):
        image = self.prepare(rgb)
        result = self.model(image[None, None])
        depth = result["depth"][0, 0].float().cpu().numpy().squeeze()
        h, w = depth.shape
        focal = 0.5 * (K[0, 0] * w / rgb.shape[1] + K[1, 1] * h / rgb.shape[0])
        depth = depth * (focal / 300.0)
        if "sky" in result:
            sky = result["sky"][0, 0].float().cpu().numpy().squeeze()
            depth[sky > 0.3] = np.nan
        return cv2.resize(depth, (output_shape[1], output_shape[0]), interpolation=cv2.INTER_LINEAR)
