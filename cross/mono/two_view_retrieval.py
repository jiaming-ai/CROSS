"""Experimental image-geometry fallback for the existing CROSS observation mixture."""
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
    Source scale is conditional on the reference prediction, as in metric PnP.
    Geometry covariance uses the existing heuristic floor, not a calibrated
    essential-matrix covariance. This option is an experimental baseline.
    """

    def __init__(self, K, device="cuda", mask_people=False, matcher="superpoint_lightglue",
                 conditional_sources=False, source_log_std=.12, rotation_geometry=None):
        from .refinement import XFeatRefiner
        if matcher != "superpoint_lightglue":
            raise ValueError("Two-view fallback requires superpoint_lightglue")
        # Preserve the original verified geometry before trying a different
        # feature family. More accepted pairs need not mean more accurate ones.
        super().__init__(K, device, mask_people, "lighterglue", conditional_sources, source_log_std)
        self.fallback_refiner = XFeatRefiner(K, device, mask_people=mask_people, mask_interval=1,
                                            matcher=matcher)
        self.fallback_cache = OrderedDict()
        self.factor_policy += ":superpoint-two-view-fallback-v2"
        self.rotation_geometry = rotation_geometry
        self.rotation_cache = OrderedDict()
        # A screen does not create new geometric evidence. Keep the physical
        # factor identity unchanged, including when a map already saw this pair.

    def _check_rotation(self, reference_rgb, current_rgb, pose):
        """Optional learned rotation agreement, with no extra source/evidence.

        The .1 rad limit is the existing metric-PnP cycle rotation threshold.
        Agreement is a heuristic screen, not an independent pose likelihood.
        """
        from scipy.spatial.transform import Rotation
        from .geometry import relative_pose

        geometry = self.rotation_geometry
        prepared = []
        for rgb in (reference_rgb, current_rgb):
            key = hashlib.sha256(rgb.tobytes()).digest()
            if key not in self.rotation_cache:
                self.rotation_cache[key] = geometry.prepare(rgb)
                if len(self.rotation_cache) > 128:
                    self.rotation_cache.popitem(last=False)
            self.rotation_cache.move_to_end(key)
            prepared.append(self.rotation_cache[key])
        prediction = geometry.predict(prepared)
        learned = relative_pose(prediction.extrinsics[0], prediction.extrinsics[1])
        audit = dict(model_id=geometry.model_id, resolution=geometry.resolution,
                     limit_rad=.1, accepted=False)
        if not np.isfinite(learned).all():
            audit['reason'] = 'nonfinite_prediction'
            return audit
        angle = float(Rotation.from_matrix(pose[:3, :3].T @ learned[:3, :3]).magnitude())
        audit.update(rotation_difference_rad=angle, accepted=angle <= .1,
                     reason='accepted' if angle <= .1 else 'rotation_disagreement')
        return audit

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
            if getattr(self, 'rotation_geometry', None) is not None:
                check = self._check_rotation(DA3RelativePose.rgb(ref_image[i]),
                                             DA3RelativePose.rgb(curr_image), pose)
                audit['rotation_check'] = check
                if not check['accepted']:
                    audit['reason'] = 'learned_rotation_rejected'
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
