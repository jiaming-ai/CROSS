from typing import Tuple
import numpy as np

from torchvision import transforms
from cross.core.types import Camera

def get_transforms_vggt(
    camera: Camera,
    target_size: int = 518,
):
    height, width = camera.frame_height, camera.frame_width
    new_height = round(height * (target_size / width) / 14) * 14
    # Create transform pipeline for crop mode
    transform_list = [
        transforms.Resize((new_height, target_size), interpolation=transforms.InterpolationMode.BICUBIC),
    ]

    # Add center crop if height is larger than target_size
    if new_height > target_size:
        transform_list.insert(-1, transforms.CenterCrop((target_size, target_size)))

    rgb_transform = transforms.Compose([transforms.ToTensor()] + transform_list)
    depth_transform = transforms.Compose(transform_list)

    # --- Update the camera parameters ---
    
    # Account for resizing
    scale_w = target_size / width
    scale_h = new_height / height

    camera.fx *= scale_w
    camera.fy *= scale_h
    camera.px *= scale_w
    camera.py *= scale_h

    # Account for center cropping if it occurs
    if new_height > target_size:
        # The crop removes pixels from the top and bottom.
        # The principal point's y-coordinate needs to be adjusted.
        crop_top = (new_height - target_size) / 2.0
        camera.py -= crop_top

    # Update the camera's K matrix and frame dimensions
    camera.K[0, 0] = camera.fx
    camera.K[1, 1] = camera.fy
    camera.K[0, 2] = camera.px
    camera.K[1, 2] = camera.py
    
    final_height = target_size if new_height > target_size else new_height
    camera.frame_width = int(target_size)
    camera.frame_height = int(final_height)


    return rgb_transform, depth_transform

def get_transforms_target_max(
    camera: Camera,
    resize_target_short_side = 400,
    max_overall_dimension = 512,
):

    rgb_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(
            resize_target_short_side, 
            interpolation=transforms.InterpolationMode.BILINEAR, 
            max_size=max_overall_dimension,
        ),
    ])
    depth_transform = transforms.Compose([
        transforms.Resize(
            resize_target_short_side, 
            interpolation=transforms.InterpolationMode.BILINEAR, 
            max_size=max_overall_dimension,
        ),
    ])

    # transform the camera parameters according to the image size
    original_width = camera.frame_width
    original_height = camera.frame_height

    fake_image = np.zeros((original_height, original_width, 3))
    fake_image = rgb_transform(fake_image)
    new_h, new_w = fake_image.shape[1], fake_image.shape[2] # (H, W)

    scale_w = new_w / original_width
    scale_h = new_h / original_height

    camera.fx *= scale_w
    camera.fy *= scale_h
    camera.px *= scale_w
    camera.py *= scale_h
    
    camera.K[0, 0] = camera.fx
    camera.K[1, 1] = camera.fy
    camera.K[0, 2] = camera.px
    camera.K[1, 2] = camera.py
    
    camera.frame_width = int(new_w)
    camera.frame_height = int(new_h)

    return rgb_transform, depth_transform
    
    
    
    
    
    
    

