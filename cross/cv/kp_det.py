import numpy as np
from typing import Dict
import torch
from abc import ABC, abstractmethod
from loguru import logger
import kornia.feature as KF
from cross.utils.profile import timeit
from collections import namedtuple

DetectionResult = namedtuple("DetectionResult", ["keypoints", "descriptors"])

class KpDet(ABC):
    """Keypoint Detection and Description"""

    def __init__(
        self, 
        device: str, 
        config: Dict,
        ):
        self.device = torch.device(device)
        self.config = config

    @abstractmethod
    def detect(self, image: torch.Tensor):
        """
        Args:
            image: torch.Tensor, shape (1, C, H, W)
        Returns:
            keypoints: torch.Tensor, shape (N, 2)
            descriptors: torch.Tensor, shape (N, 256)
        """
        pass


class KpDetDisk(KpDet):
    """Keypoint Detection and Description using DISK"""

    def __init__(self, device: str, config: Dict):
        super().__init__(device, config)

        self.disk = KF.DISK.from_pretrained("depth").to(self.device)
        self.disk.eval()
    
    def detect(self, image: torch.Tensor):
        """

        Args:
            image (torch.Tensor): shape (B, 3, H, W)
        Returns:
            A list of length B containing the detected features for batched input,
            or a DetectionResult for single image input.
        """

        if len(image.shape) == 3:
            image = image.unsqueeze(0)
            batch = False
        elif len(image.shape) == 4:
            batch = True

        disk_features = self.disk(
            image,
            n=self.config["n_keypoints"],
            score_threshold=self.config["detection_threshold"],
            pad_if_not_divisible=True)
        if batch:
            return disk_features
        else:
            return disk_features[0]

class KpDetXfeat(KpDet):
    """Keypoint Detection and Description using Xfeat"""

    def __init__(self, device: str, config: Dict):
        super().__init__(device, config)

        self.xfeat = torch.hub.load(
            'verlab/accelerated_features', 
            'XFeat', 
            pretrained = True, 
            top_k = self.config["n_keypoints"],
            detection_threshold=self.config["detection_threshold"]
        )

        self.xfeat.net.to(self.device)
        self.xfeat.dev = self.device

    @timeit
    def detect(self, image: torch.Tensor):
        """
        Args:
            image (torch.Tensor): shape (B, 3, H, W)
        Returns:
            A list of length B containing the detected features for batched input,
            or a DetectionResult for single image input.
        """
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
            batch = False
        
        elif len(image.shape) == 4:
            batch = True

        res = self.xfeat.detectAndCompute(image.to(self.device))
        if batch:
            return [DetectionResult(f["keypoints"], f["descriptors"]) for f in res]
        else:
            return DetectionResult(res[0]["keypoints"], res[0]["descriptors"])
        

class KpDetDeDoDe(KpDet):
    def __init__(self, device: str, config: Dict):
        super().__init__(device, config)

        self.dedode = KF.DeDoDe.from_pretrained(detector_weights="L-C4-v2", descriptor_weights="B-upright")
        self.dedode.to(self.device)
        self.dedode.eval()
    def detect(self, image: torch.Tensor):
        """
        Args:
            image (torch.Tensor): shape (B, 3, H, W)
        Returns:
            A list of length B containing the detected features for batched input,
            or a DetectionResult for single image input.
        """
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
            batch = False
        
        elif len(image.shape) == 4:
            batch = True
        
        keypoints, scores, features = self.dedode(
            image,
            n=self.config["n_keypoints"],
        ) 

        # TODO: filter the keypoints based on the scores

        if batch:
            return [DetectionResult(keypoints[i], features[i]) for i in range(image.shape[0])]
        else:
            return DetectionResult(keypoints[0], features[0])


def get_kp_det(kp_det_type: str, device: str, config: Dict):
    if kp_det_type == "disk":
        return KpDetDisk(device, config)
    elif kp_det_type == "xfeat":
        return KpDetXfeat(device, config)
    elif kp_det_type == "dedode":
        return KpDetDeDoDe(device, config)
    else:
        raise ValueError(f"Invalid keypoint detector type: {kp_det_type}")