import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

class DepthPred:
    def __init__(self, device: str = 'cuda'):
        self.model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True).to(device)
        self.device = device
        
        # Model configuration
        self.input_size = (616, 1064)  # for vit model
        self.padding = [123.675, 116.28, 103.53]
        self.mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None].to(device)
        self.std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None].to(device)
        
        self.model.eval()

    @torch.inference_mode()
    def predict(
        self, 
        rgb: np.ndarray, 
        intrinsic: list = None,
    ):
        """
        Args:
            rgb: (H, W, 3) - RGB image in numpy array format
            intrinsic: [fx, fy, cx, cy] - camera intrinsic parameters (optional, defaults to canonical)
        
        Returns:
            pred_depth: torch.Tensor - predicted depth map in metric scale
            confidence: torch.Tensor - confidence map 
            output_dict: dict - additional outputs from model
        """
        rgb_origin = rgb.copy()
        h_orig, w_orig = rgb_origin.shape[:2]
        
        # Default intrinsic if not provided (canonical camera)
        if intrinsic is None:
            intrinsic = [1000.0, 1000.0, w_orig/2, h_orig/2]
        
        #### Adjust input size to fit pretrained model ####
        # Keep ratio resize
        h, w = rgb_origin.shape[:2]
        scale = min(self.input_size[0] / h, self.input_size[1] / w)
        rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        
        # Scale intrinsic accordingly
        scaled_intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, 
                           intrinsic[2] * scale, intrinsic[3] * scale]
        
        # Padding to input_size
        h, w = rgb.shape[:2]
        pad_h = self.input_size[0] - h
        pad_w = self.input_size[1] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, 
                                pad_w_half, pad_w - pad_w_half, 
                                cv2.BORDER_CONSTANT, value=self.padding)
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
        
        #### Normalize ####
        rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float().to(self.device)
        rgb = torch.div((rgb - self.mean), self.std)
        rgb = rgb[None, :, :, :]  # Add batch dimension
        
        #### Inference ####
        pred_depth, confidence, output_dict = self.model.inference({'input': rgb})
        
        #### Post-processing ####
        # Un-pad
        pred_depth = pred_depth.squeeze()
        pred_depth = pred_depth[pad_info[0] : pred_depth.shape[0] - pad_info[1], 
                               pad_info[2] : pred_depth.shape[1] - pad_info[3]]
        
        # Upsample to original size
        pred_depth = torch.nn.functional.interpolate(
            pred_depth[None, None, :, :], (h_orig, w_orig), mode='bilinear'
        ).squeeze()
        
        #### De-canonical transform ####
        canonical_to_real_scale = scaled_intrinsic[0] / 1000.0  # 1000.0 is canonical focal length
        pred_depth = pred_depth * canonical_to_real_scale  # Convert to metric scale
        pred_depth = torch.clamp(pred_depth, 0, 300)  # Clamp to reasonable depth range

        visualize = False
        if visualize:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(rgb_origin)
            plt.title('Original Image')
            plt.subplot(1, 3, 2)
            plt.imshow(pred_depth.cpu().numpy())
            plt.title('Predicted Depth')
            plt.subplot(1, 3, 3)
            plt.imshow(confidence.cpu().numpy()[0,0])
            plt.title('Confidence')
            plt.show()
        return pred_depth, confidence, output_dict
