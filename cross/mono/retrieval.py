"""Scale-aware learned pose proposals for CROSS's existing hypothesis filter."""

import cv2
import numpy as np
import torch

from .geometry import relative_pose
from .scale import observe_scale
from .verification import verify_pair


class DA3RelativePose:
    def __init__(self, geometry, device="cuda"):
        self.geometry, self.device = geometry, device
        self.last_stds = None
        self.candidates = 0
        self.accepted = 0

    @staticmethod
    def rgb(image):
        return (image.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255).round().astype(np.uint8)

    @torch.inference_mode()
    def estimate_pose(self, ref_image, ref_depth, curr_image, curr_depth, **kwargs):
        import pypose as pp

        current_rgb = self.rgb(curr_image)
        current = self.geometry.prepare(current_rgb)
        poses, confidences, stds = [], [], []
        valid = np.zeros(len(ref_image), dtype=bool)
        if ref_depth is None:
            raise ValueError("Monocular keyframes must retain their predicted metric depth")
        for i, reference in enumerate(ref_image):
            self.candidates += 1
            reference_rgb = self.rgb(reference)
            prediction = self.geometry.predict([self.geometry.prepare(reference_rgb), current])
            verified, match_fraction = verify_pair(reference_rgb, current_rgb, prediction)
            if not verified:
                continue
            h, w = prediction.depth[0].shape
            depth = cv2.resize(ref_depth[i].detach().cpu().numpy().squeeze(), (w, h))
            observation = observe_scale(depth, prediction.depth[0], prediction.confidence[0])
            if not observation.accepted:
                continue
            pose = relative_pose(prediction.extrinsics[0], prediction.extrinsics[1], np.exp(observation.log_scale))
            if not np.isfinite(pose).all():
                continue
            # A consistent depth ratio alone cannot establish a loop. These
            # proposals still pass CROSS's multi-frame hypothesis lifecycle.
            confidence = observation.inlier_fraction * np.exp(-2 * observation.log_mad) * np.sqrt(match_fraction)
            sigma_t = np.sqrt(0.02**2 + pose[:3, 3]**2 * observation.variance)
            poses.append(pose)
            confidences.append(float(np.clip(confidence, 0.01, 1.0)))
            stds.append(np.r_[sigma_t, [0.03] * 3])
            valid[i] = True
            self.accepted += 1
        self.last_stds = torch.as_tensor(np.array(stds).reshape(-1, 6), dtype=torch.float32)
        if not poses:
            return pp.identity_SE3(0, device=self.device), valid, torch.empty(0)
        matrices = torch.as_tensor(np.array(poses), dtype=torch.float32, device=self.device)
        return pp.from_matrix(matrices, pp.SE3_type), valid, torch.tensor(confidences)
