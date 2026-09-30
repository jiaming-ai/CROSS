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
    
    
    
    
    
    
    


def get_transforms_ff(
    camera: Camera,
    image_resolution: int = 512,
    patch_size: int = 16,
    min_aspect: float = 0.5,
    max_aspect: float = 2.0,
):
    """Transforms for feed-forward geometry models (VGGT-Omega / DA3).

    Mirrors `vggt_omega.utils.load_fn.load_and_preprocess_images(mode="max_size")`:
    center-crop extreme aspect ratios into [min_aspect, max_aspect], then resize the
    longest side to `image_resolution` with both sides rounded to a multiple of
    `patch_size`.  The camera intrinsics are updated in place.
    """
    width, height = camera.frame_width, camera.frame_height
    aspect = height / max(width, 1)
    crop_w, crop_h = width, height
    if aspect < min_aspect:
        crop_w = min(width, max(1, int(round(height / min_aspect))))
    elif aspect > max_aspect:
        crop_h = min(height, max(1, int(round(width * max_aspect))))
    aspect = crop_h / crop_w

    def round_patch(v):
        return max(patch_size, int(round(float(v) / patch_size)) * patch_size)

    if aspect >= 1.0:
        new_h, new_w = image_resolution, round_patch(image_resolution / aspect)
    else:
        new_w, new_h = image_resolution, round_patch(image_resolution * aspect)

    transform_list = []
    if (crop_w, crop_h) != (width, height):
        transform_list.append(transforms.CenterCrop((crop_h, crop_w)))
    transform_list.append(transforms.Resize((new_h, new_w), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True))
    rgb_transform = transforms.Compose([transforms.ToTensor()] + transform_list)
    depth_transform = transforms.Compose(
        ([transforms.CenterCrop((crop_h, crop_w))] if (crop_w, crop_h) != (width, height) else [])
        + [transforms.Resize((new_h, new_w), interpolation=transforms.InterpolationMode.NEAREST)]
    )

    # intrinsics: crop shifts the principal point, resize scales
    crop_left = (width - crop_w) / 2.0
    crop_top = (height - crop_h) / 2.0
    scale_w = new_w / crop_w
    scale_h = new_h / crop_h
    camera.fx *= scale_w
    camera.fy *= scale_h
    camera.px = (camera.px - crop_left) * scale_w
    camera.py = (camera.py - crop_top) * scale_h
    camera.K = camera.K.astype(np.float64).copy()
    camera.K[0, 0], camera.K[1, 1] = camera.fx, camera.fy
    camera.K[0, 2], camera.K[1, 2] = camera.px, camera.py
    camera.frame_width, camera.frame_height = int(new_w), int(new_h)
    return rgb_transform, depth_transform
