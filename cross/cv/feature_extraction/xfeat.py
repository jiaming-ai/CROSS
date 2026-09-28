
import torch
import matplotlib.pyplot as plt

from .extractor import FeatureExtractor

import cv2
import numpy as np
from cross.utils.profile import timeit



class XfeatExtractor(FeatureExtractor):
    def __init__(self, 
                 device="cuda", 
                 detection_threshold=0.05,
                 max_num_keypoints=2048,
                 match_threshold=0.7,
                 **kwargs):
            

        # self.xfeat = XFeat(top_k = max_num_keypoints, detection_threshold=detection_threshold)
        self.xfeat = torch.hub.load(
            'verlab/accelerated_features', 
            'XFeat', 
            pretrained = True, 
            top_k = max_num_keypoints,
            detection_threshold=detection_threshold
        )

        self.xfeat.net.to(device)
        self.xfeat.dev = torch.device(device)
        self.match_threshold = match_threshold

        super().__init__(**kwargs)
        
    @timeit
    def extract(self, image: torch.Tensor, to_cpu: bool = True):
        """
        Args:
            image: torch.Tensor, shape (1, C, H, W)
        Returns:
            keypoints: torch.Tensor, shape (N, 2)
            descriptors: torch.Tensor, shape (N, 256)
        """
        batch = False
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        elif len(image.shape) == 4:
            batch = True

        res = self.xfeat.detectAndCompute(image)
        if to_cpu:
            res = [{k: v.cpu() for k, v in r.items()} for r in res]
        if batch:
            return res
        else:
            return res[0]
    
