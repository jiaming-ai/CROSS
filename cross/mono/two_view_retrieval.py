"""Experimental image-geometry fallback for the existing CROSS observation mixture."""
import hashlib

import numpy as np
import torch

from .retrieval import DA3RelativePose, MetricRelativePose
from .two_view_geometry import estimate_metric_two_view


class MetricTwoViewRelativePose(MetricRelativePose):
    """Keep accepted metric PnP; otherwise propose a calibrated two-view pose.

    One reference contributes at most one relative component per call. The
    fallback neither commits a place nor adds another observation to the filter.
    Source scale is conditional on the reference prediction, as in metric PnP.
    Geometry covariance uses the existing heuristic floor, not a calibrated
    essential-matrix covariance. This option is an experimental baseline.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.factor_policy += ":two-view-fallback-v1"

    @torch.inference_mode()
    def estimate_pose(self, ref_image, ref_depth, curr_image, curr_depth, **kwargs):
        import pypose as pp
        from .metric_sources import pnp_scale_response

        if ref_depth is None or curr_depth is None:
            raise ValueError("Two-view metric verification requires both learned depths")
        poses, stds, confidences, conditional, audits = [], [], [], [], []
        valid = np.zeros(len(ref_image), dtype=bool)
        sources = kwargs.get("ref_metric_sources")
        target = kwargs.get("curr_metric_source")
        for i in range(len(ref_image)):
            single = dict(kwargs)
            if sources is not None:
                single["ref_metric_sources"] = [sources[i]]
            base_pose, base_valid, base_confidence = super().estimate_pose(
                ref_image[i:i+1], ref_depth[i:i+1], curr_image, curr_depth, **single)
            audit = self.last_pair_audit[0]
            audits.append(audit)
            if base_valid[0]:
                poses.append(base_pose[0].matrix().cpu().numpy())
                stds.append(self.last_stds[0].cpu().numpy())
                confidences.append(float(base_confidence[0]))
                conditional.extend(self.last_conditional_poses)
                valid[i] = True
                audit["proposal_method"] = "metric_pnp"
                continue
            if audit["reason"] == "reused_geometric_factor":
                continue
            reference = self.features(DA3RelativePose.rgb(ref_image[i]))
            current = self.features(DA3RelativePose.rgb(curr_image))
            if min(len(reference["keypoints"]), len(current["keypoints"])) < 20:
                continue
            x0, x1 = self.refiner.match(reference, current)
            pose, geometry = estimate_metric_two_view(
                x0, x1, ref_depth[i].cpu().numpy().squeeze(), curr_depth.cpu().numpy().squeeze(),
                self.refiner.K, reference["shape"])
            audit["pnp_rejection"] = audit["reason"]
            audit.update(reason="two_view_rejected", two_view=geometry)
            if pose is None:
                continue
            confidence = min(1., geometry["metric_inliers"] / 80.) * np.exp(
                -geometry["median_reprojection_px"] / 3.)
            std = np.r_[np.sqrt(.02**2 + (.12*pose[:3, 3])**2), [.03]*3]
            if self.conditional_sources:
                from cross.core.conditional import SourceFactor
                from cross.core.conditional_pose import ConditionalPose
                source = sources[i]  # validated by the original adapter above
                identity = hashlib.sha256((self.factor_policy + source["source_id"] + target["source_id"]).encode()).hexdigest()
                factor = SourceFactor(("image:"+source["source_id"],),
                                      np.asarray(pnp_scale_response(pose))[:, None],
                                      np.array([self.source_log_std**2]), factor_id=identity, log_depth_scale=True)
                std = np.array([.02]*3 + [.03]*3)
                conditional.append(ConditionalPose(np.diag(std**2), factor))
            if sources is not None:
                audit["metric_sources"]["forward_right_tangent_response"] = pnp_scale_response(pose)
            poses.append(pose)
            stds.append(std)
            confidences.append(float(np.clip(confidence, .01, 1.)))
            valid[i] = True
            audit.update(reason="accepted", accepted=True, proposal_method="metric_two_view",
                         confidence=confidences[-1])
        self.last_pair_audit, self.last_conditional_poses = audits, conditional
        self.last_stds = torch.as_tensor(np.asarray(stds).reshape(-1, 6), dtype=torch.float32)
        if not poses:
            return pp.identity_SE3(0, device=self.device), valid, torch.empty(0)
        matrices = torch.as_tensor(np.asarray(poses), dtype=torch.float32, device=self.device)
        return pp.from_matrix(matrices, pp.SE3_type), valid, torch.tensor(confidences)
