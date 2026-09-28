import torch
import numpy as np
from typing import Optional, Tuple, Dict

class DepthPredUni:
    def __init__(
        self,
        device: str = 'cuda',
        model_name: str = 'unidepth-v2-vitl14',
        version: str = 'v2',
        use_autocast: bool = False
    ):
        """
        Initialize UniDepth depth prediction model.

        Args:
            device: Device to run inference on ('cuda' or 'cpu')
            model_name: Model name for UniDepth
                For V2: 'unidepth-v2-vits14', 'unidepth-v2-vitb14', 'unidepth-v2-vitl14'
                For V1: check unidepth documentation
            version: UniDepth version ('v1' or 'v2')
            use_autocast: Use automatic mixed precision with bfloat16 (default: True for efficiency)
        """
        from unidepth.models import UniDepthV1, UniDepthV2

        self.device = device
        self.model_name = model_name
        self.version = version
        self.use_autocast = use_autocast and device == 'cuda'  # Only use autocast on CUDA

        # Load the appropriate model
        if version == 'v2':
            self.model = UniDepthV2.from_pretrained(f"lpiccinelli/{model_name}")
        elif version == 'v1':
            self.model = UniDepthV1.from_pretrained(f"lpiccinelli/{model_name}")
        else:
            raise ValueError(f"Unsupported version: {version}. Use 'v1' or 'v2'")

        self.model = self.model.to(device)
        self.model.eval()
        self.model.resolution_level = 0

    @torch.inference_mode()
    def predict(
        self,
        rgb: np.ndarray,
        intrinsic: Optional[list] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Predict depth from RGB image.

        Args:
            rgb: (H, W, 3) - RGB image in numpy array format
            intrinsic: [fx, fy, cx, cy] - camera intrinsic parameters (optional)
                If provided, will use known intrinsics for prediction.
                If None, the model will estimate intrinsics.

        Returns:
            pred_depth: torch.Tensor (H, W) - predicted depth map in metric scale
            confidence: torch.Tensor (1, 1, H, W) - confidence map (dummy for compatibility)
            output_dict: dict - additional outputs from model including:
                - 'depth': depth prediction
                - 'points': 3D point cloud in camera coordinates
                - 'intrinsics': predicted or provided camera intrinsics
        """
        # Convert RGB to torch tensor and permute to (C, H, W)
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()  # (3, H, W)
        rgb_tensor = rgb_tensor.to(self.device)

        # Prepare camera intrinsics if provided
        camera = None
        if intrinsic is not None:
            # Build 3x3 intrinsic matrix from [fx, fy, cx, cy]
            fx, fy, cx, cy = intrinsic
            K = torch.tensor([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=torch.float32).to(self.device)

            # For V2, use Pinhole camera class
            if self.version == 'v2':
                from unidepth.utils.camera import Pinhole
                camera = Pinhole(K=K)
            else:
                # For V1, pass K directly (check unidepth v1 docs if needed)
                camera = K

        # Run inference with autocast for efficiency
        if self.use_autocast:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if camera is not None:
                    predictions = self.model.infer(rgb_tensor, camera)
                else:
                    predictions = self.model.infer(rgb_tensor)
        else:
            if camera is not None:
                predictions = self.model.infer(rgb_tensor, camera)
            else:
                predictions = self.model.infer(rgb_tensor)

        # Extract depth
        pred_depth = predictions["depth"]

        # Ensure depth is 2D (H, W)
        if pred_depth.dim() > 2:
            pred_depth = pred_depth.squeeze()

        # Ensure output is float32 for consistency
        pred_depth = pred_depth.float()

        # Create a dummy confidence map for API compatibility
        # UniDepth doesn't provide explicit confidence, so we create a uniform one
        confidence = torch.ones(1, 1, pred_depth.shape[0], pred_depth.shape[1],
                               device=self.device, dtype=torch.float32)

        # Prepare output dict
        output_dict = {
            'depth': predictions.get('depth'),
            'points': predictions.get('points'),
            'intrinsics': predictions.get('intrinsics'),
            'all_predictions': predictions  # Keep all predictions for advanced use
        }

        return pred_depth, confidence, output_dict
