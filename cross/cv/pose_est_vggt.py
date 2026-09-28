import numpy as np
import torch
from cross.utils.profile import timeit
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
import pypose as pp

class PoseEstVGGT:
    def __init__(self, device: str = "cuda", **kwargs):
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        self.model.eval()
        self.device = device
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    @timeit
    @torch.inference_mode()
    def estimate_pose(
        self, 
        ref_image: torch.Tensor, 
        ref_depth: torch.Tensor,
        curr_image: torch.Tensor,
        curr_depth: torch.Tensor,
        **kwargs,
    ):
        """Estimate relative pose between two images using PnP
        TODO: cache the keypoints and descriptors
        Args:
            ref_image: (B, 3, H, W)
            ref_depth: (B, H, W)
            curr_image: (3, H, W)
            curr_depth: (H, W)
            retrieval_scores: (B,) optional, for visualization only
            keyframes: (B,) optional, for visualization only
        Returns:
            valid_poses: SE3 (B, 7). The pose is T_ref_cam
            valid_covs: se3 (B, 6)
            valid_masks: (B,)
            confidences: (B,)
        """
        images = torch.cat([curr_image.unsqueeze(0), ref_image], dim=0).to(self.device)
        
        with torch.amp.autocast('cuda', dtype=self.dtype):
            predictions = self.model(images)

        T = predictions["pose_enc"][0, 1:, :7] # (B, 7)
        T = pp.SE3(T) # T_ref_cam

        valid_masks = np.ones(T.shape[0], dtype=bool)
        confidences = torch.ones(T.shape[0], dtype=torch.float)

        return T, valid_masks, confidences

