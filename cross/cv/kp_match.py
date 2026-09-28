import torch
from typing import Dict
import kornia.feature as KF
from abc import ABC, abstractmethod
from cross.utils.profile import timeit

default_conf_xfeat = {
    "name": "xfeat",  # just for interfacing
    "input_dim": 64,  # input descriptor dimension (autoselected from weights)
    "descriptor_dim": 96,
    "add_scale_ori": False,
    "add_laf": False,  # for KeyNetAffNetHardNet
    "scale_coef": 1.0,  # to compensate for the SIFT scale bigger than KeyNet
    "n_layers": 6,
    "num_heads": 1,
    "flash": True,  # enable FlashAttention if available.
    "mp": False,  # enable mixed precision
    "filter_threshold": 0.1,  # match threshold
    "weights": None,
}
allow_batch_inference = {
    "depth_confidence": -1,  # early stopping, disable with -1
    "width_confidence": -1,  # point pruning, disable with -1
}

class KpMatch(ABC):
    """Keypoint Matching"""

    def __init__(self, device: str, config: Dict):
        self.device = device
        self.config = config

    @abstractmethod
    def match(self, feats1: torch.Tensor, feats2: torch.Tensor):
        pass


class KpMatchLightGlue(KpMatch):
    """Keypoint Matching using LightGlue"""

    def __init__(
        self, 
        feature: str,
        device: str, 
        config: Dict,
    ):
        super().__init__(device, config)

        if config["allow_batch_inference"]:
            KF.LightGlue.default_conf.update(allow_batch_inference)

        if feature == "xfeat":
            KF.LightGlue.default_conf.update(default_conf_xfeat)
            self.net = KF.LightGlue(None)
            state_dict = torch.hub.load_state_dict_from_url("https://github.com/verlab/accelerated_features/raw/main/weights/xfeat-lighterglue.pt")

            # rename old state dict entries
            for i in range(self.net.conf.n_layers):
                pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
                pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
                state_dict = {k.replace('matcher.', ''): v for k, v in state_dict.items()}

            self.net.load_state_dict(state_dict, strict=False)
            
        elif feature in ["disk", "superpoint"]:
            self.net = KF.LightGlue(
                features=feature,
            )
        else:
            raise ValueError(f"Feature type {feature} not supported for LightGlue")

        self.net.eval()
        self.net.to(self.device)
        self.net.conf.filter_threshold = config["min_conf"]
        
    @timeit
    def match(self, data: Dict):
        """
        Args:
            image0: dict
            keypoints: [B x M x 2] descriptors: [B x M x D] image: [B x C x H x W] or image_size: [B x 2]

            image1: dict
            keypoints: [B x N x 2] descriptors: [B x N x D] image: [B x C x H x W] or image_size: [B x 2]


        Returns:
            log_assignment: [B x M+1 x N+1] 
            matches0: [B x M] 
            matching_scores0: [B x M] 
            matches1: [B x N] 
            matching_scores1: [B x N] 
            matches: List[[Si x 2]], 
            scores: List[[Si]]

        """
        result = self.net(data)
        return result

        
        


def get_kp_match(kp_match_type: str, feature_type: str, device: str, config: Dict):
    if kp_match_type == "lightglue":
        return KpMatchLightGlue(feature_type, device, config)
    else:
        raise ValueError(f"Invalid keypoint matcher type: {kp_match_type}")