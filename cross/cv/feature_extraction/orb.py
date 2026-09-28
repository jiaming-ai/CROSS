
from .extractor import FeatureExtractor

import cv2
import numpy as np
import torch



class ORBExtractor(FeatureExtractor):
    def __init__(self, 
                 max_num_keypoints=2048,
                 **kwargs):
        self.orb = cv2.ORB_create(nfeatures=max_num_keypoints)
        
        super().__init__(**kwargs)
        
    def extract(self, image: torch.Tensor):
        """
        Args:
            image: torch.Tensor, shape (1, C, H, W)
        Returns:
            keypoints: torch.Tensor, shape (N, 2)
            descriptors: torch.Tensor, shape (N, 256)
        """
        img = image.cpu().numpy()[0].transpose(1, 2, 0) * 255
        # input image is converted to gray scale image 
        imagegray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY) 
        keypoints, descriptors = self.orb.detectAndCompute(imagegray, None) 
        ret = {
            "keypoints": torch.tensor([kp.pt for kp in keypoints]), 
            "descriptors": torch.tensor(descriptors),
        }
        return ret
    
