"""Scale-aware learned pose proposals for CROSS's existing hypothesis filter."""

import cv2
from collections import OrderedDict
import hashlib
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


class MetricRelativePose:
    """Bidirectionally verified metric PnP proposals; commitment stays in CROSS.

    Uses its own matcher/detector state. Reading old keyframes must never
    contaminate the causal frontend's cached person masks or feature state.
    """
    def __init__(self, K, device="cuda", mask_people=False):
        from .refinement import XFeatRefiner
        self.refiner = XFeatRefiner(K, device, mask_people=mask_people, mask_interval=1)
        self.device = device
        self.cache = OrderedDict()
        self.last_stds = None

    def features(self, rgb):
        key = hashlib.sha256(rgb.tobytes()).digest()
        if key not in self.cache:
            self.cache[key] = self.refiner.extract(rgb)
            if len(self.cache) > 256:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    @torch.inference_mode()
    def estimate_pose(self, ref_image, ref_depth, curr_image, curr_depth, **kwargs):
        import pypose as pp
        from scipy.spatial.transform import Rotation
        current = self.features(DA3RelativePose.rgb(curr_image))
        poses, confidences, stds = [], [], []
        valid = np.zeros(len(ref_image), dtype=bool)
        if ref_depth is None or curr_depth is None:
            raise ValueError("Metric PnP verification requires learned depths of both images")
        current_depth = curr_depth.detach().cpu().numpy().squeeze()
        self.last_pair_audit = []
        for i, image in enumerate(ref_image):
            audit = dict(reason="forward_pnp", accepted=False)
            self.last_pair_audit.append(audit)
            reference = self.features(DA3RelativePose.rgb(image))
            depth = ref_depth[i].detach().cpu().numpy().squeeze()
            forward = self.refiner.estimate(reference, current, depth)
            audit["forward"] = dict(self.refiner.last_match_audit)
            if forward is None:
                continue
            backward = self.refiner.estimate(current, reference, current_depth)
            audit.update(reason="backward_pnp", backward=dict(self.refiner.last_match_audit))
            if backward is None:
                continue
            pose, count, error = forward
            cycle = pose @ backward[0]
            cycle_rotation = Rotation.from_matrix(cycle[:3, :3]).magnitude()
            cycle_translation = np.linalg.norm(cycle[:3, 3])
            audit.update(reason="inconsistent_cycle", cycle_rotation_rad=float(cycle_rotation),
                         cycle_translation_m=float(cycle_translation))
            if (cycle_rotation > 0.1 or
                    cycle_translation > 0.15 + 0.2 * np.linalg.norm(pose[:3, 3])):
                continue
            confidence = min(1., min(count, backward[1]) / 80.) * np.exp(-max(error, backward[2]) / 3.)
            poses.append(pose)
            confidences.append(float(np.clip(confidence, 0.01, 1.)))
            stds.append(np.r_[np.sqrt(0.02**2 + (pose[:3, 3] * 0.12)**2), [0.03] * 3])
            valid[i] = True
            audit.update(reason="accepted", accepted=True, confidence=confidences[-1])
        self.last_stds = torch.as_tensor(np.array(stds).reshape(-1, 6), dtype=torch.float32)
        if not poses:
            return pp.identity_SE3(0, device=self.device), valid, torch.empty(0)
        matrices = torch.as_tensor(np.array(poses), dtype=torch.float32, device=self.device)
        return pp.from_matrix(matrices, pp.SE3_type), valid, torch.tensor(confidences)
