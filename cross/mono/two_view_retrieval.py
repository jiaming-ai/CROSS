"""Image-geometry fallback for the existing CROSS observation mixture."""
import hashlib
from collections import OrderedDict

import numpy as np
import torch

from .retrieval import DA3RelativePose, MetricRelativePose
from .two_view_geometry import estimate_metric_two_view


class MetricTwoViewRelativePose(MetricRelativePose):
    """Keep accepted metric PnP; otherwise propose a calibrated two-view pose.

    One reference contributes at most one relative component per call. The
    fallback neither commits a place nor adds another observation to the filter.
    Metric scale comes from the reference prediction, as in metric PnP.
    Geometry covariance uses the existing heuristic floor, not a calibrated
    essential-matrix covariance. This option is an experimental baseline.
    """

    def __init__(self, K, device="cuda", mask_people=False, matcher="superpoint_lightglue"):
        from .refinement import XFeatRefiner
        if matcher != "superpoint_lightglue":
            raise ValueError("Two-view fallback requires superpoint_lightglue")
        # Preserve the original verified geometry before trying a different
        # feature family. More accepted pairs need not mean more accurate ones.
        super().__init__(K, device, mask_people, "lighterglue")
        self.fallback_refiner = XFeatRefiner(K, device, mask_people=mask_people, mask_interval=1,
                                            matcher=matcher)
        self.fallback_cache = OrderedDict()

    def fallback_features(self, rgb):
        key = hashlib.sha256(rgb.tobytes()).digest()
        if key not in self.fallback_cache:
            self.fallback_cache[key] = self.fallback_refiner.extract(rgb)
            if len(self.fallback_cache) > 256:
                self.fallback_cache.popitem(last=False)
        self.fallback_cache.move_to_end(key)
        return self.fallback_cache[key]

    @torch.inference_mode()
    def estimate_pose(self, ref_image, ref_depth, curr_image, curr_depth, **kwargs):
        import pypose as pp

        if ref_depth is None or curr_depth is None:
            raise ValueError("Two-view metric verification requires both learned depths")
        poses, stds, confidences, audits = [], [], [], []
        valid = np.zeros(len(ref_image), dtype=bool)
        for i in range(len(ref_image)):
            base_pose, base_valid, base_confidence = super().estimate_pose(
                ref_image[i:i+1], ref_depth[i:i+1], curr_image, curr_depth, **kwargs)
            audit = self.last_pair_audit[0]
            audits.append(audit)
            if base_valid[0]:
                poses.append(base_pose[0].matrix().cpu().numpy())
                stds.append(self.last_stds[0].cpu().numpy())
                confidences.append(float(base_confidence[0]))
                valid[i] = True
                audit["proposal_method"] = "metric_pnp"
                continue
            reference = self.fallback_features(DA3RelativePose.rgb(ref_image[i]))
            current = self.fallback_features(DA3RelativePose.rgb(curr_image))
            if min(len(reference["keypoints"]), len(current["keypoints"])) < 20:
                continue
            x0, x1 = self.fallback_refiner.match(reference, current)
            pose, geometry = estimate_metric_two_view(
                x0, x1, ref_depth[i].cpu().numpy().squeeze(), curr_depth.cpu().numpy().squeeze(),
                self.fallback_refiner.K, reference["shape"])
            audit["pnp_rejection"] = audit["reason"]
            audit.update(reason="two_view_rejected", two_view=geometry)
            if pose is None:
                continue
            confidence = min(1., geometry["metric_inliers"] / 80.) * np.exp(
                -geometry["median_reprojection_px"] / 3.)
            std = np.r_[np.sqrt(.02**2 + (.12*pose[:3, 3])**2), [.03]*3]
            poses.append(pose)
            stds.append(std)
            confidences.append(float(np.clip(confidence, .01, 1.)))
            valid[i] = True
            audit.update(reason="accepted", accepted=True, proposal_method="metric_two_view",
                         confidence=confidences[-1])
        self.last_pair_audit = audits
        self.last_stds = torch.as_tensor(np.asarray(stds).reshape(-1, 6), dtype=torch.float32)
        if not poses:
            return pp.identity_SE3(0, device=self.device), valid, torch.empty(0)
        matrices = torch.as_tensor(np.asarray(poses), dtype=torch.float32, device=self.device)
        return pp.from_matrix(matrices, pp.SE3_type), valid, torch.tensor(confidences)
