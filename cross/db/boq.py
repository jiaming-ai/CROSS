import torch
import torchvision.transforms as T
import numpy as np
from collections import OrderedDict
import hashlib

class BoQ():
    def __init__(
        self, 
        backbone_name: str = "resnet50", 
        device: str = "cuda", 
        cache_size: int = 1000, 
        enable_cache: bool = True
    ):
        # ResNet50 + BoQ
        if backbone_name == "resnet50":
            self.vpr_model = torch.hub.load(
                "amaralibey/bag-of-queries", 
                "get_trained_boq", 
                backbone_name=backbone_name, 
                output_dim=16384,
            )
            self.im_size = (384, 384) # to be used with ResNet50 backbone
            self.output_dim = 16384
        elif backbone_name == "dinov2":
            self.vpr_model = torch.hub.load(
                "amaralibey/bag-of-queries", 
                "get_trained_boq", 
                backbone_name=backbone_name, 
                output_dim=12288, 
                )
            self.im_size = (322, 322) # to be used with DinoV2 backbone
            self.output_dim = 12288
        self.device = device
        self.vpr_model.eval()
        self.vpr_model.to(self.device)
        self.transform = T.Compose([
            T.Resize(self.im_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        # Initialize cache
        self.enable_cache = enable_cache
        self.cache_size = cache_size if enable_cache else 0
        self.cache = OrderedDict() if enable_cache else None

    def _get_cache_key(self, img: torch.Tensor) -> str:
        """Generate a unique key for the image tensor using hash.
        Args:
            img: Input image tensor
        Returns:
            str: A unique hash key for the image
        """
        # Convert tensor to bytes and compute hash
        img_bytes = img.cpu().numpy().tobytes()
        return hashlib.md5(img_bytes).hexdigest()

    def _update_cache(self, key: str, value: torch.Tensor):
        """Update the cache with a new key-value pair.
        Args:
            key: Cache key
            value: Cache value (embedding)
        """
        if not self.enable_cache:
            return

        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
        else:
            # Add new entry
            self.cache[key] = value
            # Remove oldest if cache is full
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)

    def get_embed_dim(self) -> int:
        """Get the dimension of the embedding vector.
        Returns:
            int: The dimension of the embedding vector
        """
        return self.output_dim

    @torch.inference_mode()
    def get_embedding(self, img: torch.Tensor):
        """Get the VPR embedding of the image.
        Args:
            img: Input image tensor. Can be:
                - Single image: (H, W, C) or (C, H, W)
                - Batch of images: (B, H, W, C) or (B, C, H, W)
        Returns:
            torch.Tensor: Embedding vector(s) of shape (D,) for single image or (B, D) for batch
        """
        # Handle different input formats
        if img.ndim == 3:  # Single image
            if img.shape[0] == 3:  # CHW format
                img = img.permute(1, 2, 0)  # Convert to HWC
            img = img.unsqueeze(0)  # Add batch dimension
        elif img.ndim == 4:  # Batch of images
            if img.shape[1] == 3:  # BCHW format
                img = img.permute(0, 2, 3, 1)  # Convert to BHWC
        
        # Check cache only for single images and if cache is enabled
        is_single_image = img.shape[0] == 1
        if is_single_image and self.enable_cache:
            cache_key = self._get_cache_key(img)
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        # Apply transformations
        torch_img = self.transform(img.permute(0, 3, 1, 2)).to(self.device)  # Convert to BCHW for transform
        
        # Get embeddings
        output, _ = self.vpr_model(torch_img)
        
        # Cache single image result if cache is enabled
        if is_single_image and self.enable_cache:
            self._update_cache(cache_key, output.squeeze(0))
            return output.squeeze(0)
        return output if not is_single_image else output.squeeze(0)

