"""Lightweight learned correspondences and robust pose refinement."""

import cv2
import numpy as np
import torch

from .geometry import inverse


class XFeatRefiner:
    def __init__(self, K, device="cuda", keypoints=1600, mask_people=False, mask_interval=1, rotation_selection=False,
                 subpixel=False):
        self.K = np.asarray(K)
        self.device = device
        self.extractor = torch.hub.load("verlab/accelerated_features", "XFeat", pretrained=True,
                                       top_k=keypoints, detection_threshold=0.03, trust_repo=True)
        self.extractor.net.to(device).eval()
        self.extractor.dev = torch.device(device)
        self.detector = None
        self.mask_interval = mask_interval
        self.frame_index = 0
        self.boxes = None
        self.rotation_selection = rotation_selection
        self.last_rotation_only = False
        self.subpixel = subpixel
        self.last_correspondences = 0
        if mask_people:
            from .person_detector import make_person_detector
            self.detector = make_person_detector(device)

    @torch.inference_mode()
    def extract(self, rgb):
        tensor = torch.as_tensor(rgb.copy(), device=self.device).permute(2, 0, 1).float()[None] / 255.0
        result = self.extractor.detectAndCompute(tensor)[0]
        if self.detector is not None:
            age = self.frame_index % self.mask_interval
            if self.boxes is None or age == 0:
                detected = self.detector([tensor[0]])[0]
                self.boxes = detected["boxes"][(detected["labels"] == 1) & (detected["scores"] >= 0.5)]
            boxes = self.boxes
            keep = torch.ones(len(result["keypoints"]), device=self.device, dtype=torch.bool)
            points = result["keypoints"]
            for box in boxes:
                padding = 8 + 4 * age  # expand stale boxes conservatively
                inside = ((points >= box[:2] - padding) & (points <= box[2:] + padding)).all(dim=1)
                keep &= ~inside
            result["masked_keypoints"] = int((~keep).sum())
            result["person_boxes"] = len(boxes)
            padding = 8 + 4 * age
            result["exclusion_boxes"] = (boxes + boxes.new_tensor([-padding, -padding, padding, padding])).cpu().numpy()
            for key in ("keypoints", "descriptors", "scores"):
                result[key] = result[key][keep]
        result["shape"] = rgb.shape[:2]
        if self.subpixel:
            result["gray"] = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        self.frame_index += 1
        return result

    @torch.inference_mode()
    def estimate(self, ref, current, ref_depth, rotation=None):
        self.last_rotation_only = False
        self.last_correspondences = 0
        if min(len(ref["keypoints"]), len(current["keypoints"])) < 20:
            return None
        similarity = ref["descriptors"] @ current["descriptors"].T
        scores, indices = similarity.topk(2, dim=1)
        best = indices[:, 0]
        reverse = similarity.argmax(dim=0)
        matched = (reverse[best] == torch.arange(len(best), device=best.device)) & (scores[:, 0] > 0.65)
        matched &= (1 - scores[:, 0]).clamp_min(0) < 0.81 * (1 - scores[:, 1]).clamp_min(0)
        xy_ref = ref["keypoints"][matched].cpu().numpy()
        xy_cur = current["keypoints"][best[matched]].cpu().numpy()
        if self.subpixel:
            from .photometric import refine_correspondences
            xy_ref, xy_cur = refine_correspondences(ref["gray"], current["gray"], xy_ref, xy_cur)
        self.last_correspondences = len(xy_ref)
        if len(xy_ref) < 20:
            return None
        height, width = ref["shape"]
        uv = np.round(xy_ref * [ref_depth.shape[1] / width, ref_depth.shape[0] / height]).astype(int)
        uv[:, 0] = np.clip(uv[:, 0], 0, ref_depth.shape[1] - 1)
        uv[:, 1] = np.clip(uv[:, 1], 0, ref_depth.shape[0] - 1)
        z = ref_depth[uv[:, 1], uv[:, 0]]
        valid = np.isfinite(z) & (z > 0.01)
        xyz = np.c_[xy_ref[valid], np.ones(valid.sum())] @ np.linalg.inv(self.K).T * z[valid, None]
        pixels = np.ascontiguousarray(xy_cur[valid], dtype=np.float64)
        if len(xyz) < 20:
            return None
        if rotation is not None:
            from .translation import translation_given_rotation
            solution = translation_given_rotation(xyz, pixels, self.K, rotation)
            if solution is None:
                return None
            translation, inliers, error = solution
            tiles = np.floor(pixels[inliers] / [width, height] * 4).astype(int)
            if len(np.unique(tiles, axis=0)) < 4:
                return None
            transform = np.eye(4)
            transform[:3, :3], transform[:3, 3] = rotation, translation
            return inverse(transform), len(inliers), error
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            np.ascontiguousarray(xyz, dtype=np.float64), pixels, self.K, None,
            iterationsCount=150, reprojectionError=3.0, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
        if not success or inliers is None or len(inliers) < 20 or len(inliers) / len(xyz) < 0.3:
            return None
        inliers = inliers.ravel()
        tiles = np.floor(pixels[inliers] / [width, height] * 4).astype(int)
        if len(np.unique(tiles, axis=0)) < 4:
            return None
        rvec, tvec = cv2.solvePnPRefineLM(xyz[inliers], pixels[inliers], self.K, None, rvec, tvec)
        if self.rotation_selection:
            from .observability import rotation_only_model
            rotation_only, selected = rotation_only_model(xyz[inliers], pixels[inliers], self.K,
                                                          cv2.Rodrigues(rvec)[0], tvec)
            if selected:
                rvec, tvec = cv2.Rodrigues(rotation_only)[0], np.zeros((3, 1))
                self.last_rotation_only = True
        T_current_ref = np.eye(4)
        T_current_ref[:3, :3] = cv2.Rodrigues(rvec)[0]
        T_current_ref[:3, 3] = tvec.ravel()
        projected = cv2.projectPoints(xyz[inliers], rvec, tvec, self.K, None)[0].reshape(-1, 2)
        error = np.median(np.linalg.norm(projected - pixels[inliers], axis=1))
        if error > 3.0 or not np.isfinite(T_current_ref).all():
            return None
        return inverse(T_current_ref), len(inliers), float(error)
