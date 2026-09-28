"""Use a pretrained person prior to select DPVO's source patches."""

import numpy as np
import torch


def background_indices(pixels, boxes, count):
    valid = torch.ones(len(pixels), dtype=torch.bool, device=pixels.device)
    for box in boxes:
        valid &= ~((pixels >= box[:2]) & (pixels <= box[2:])).all(dim=1)
    candidates = valid.nonzero().flatten()
    support = len(candidates)
    if not support:
        return torch.arange(count, device=pixels.device), 0
    # Uniform candidates were already sampled by the native patchifier.
    # Repetition is explicit when almost the entire view is excluded.
    return candidates[torch.arange(count, device=pixels.device) % support], min(count, support)


class BackgroundPatchifier(torch.nn.Module):
    """Oversample once, then retain native patches outside person regions.

    The released network's feature extraction, patch descriptors and optimizer
    are unchanged. This wrapper does not modify the DPVO checkout or weights.
    """
    def __init__(self, original, device="cuda", interval=3):
        super().__init__()
        from .person_detector import make_person_detector
        self.original = original
        self.detector = make_person_detector(device)
        self.interval, self.frame_index, self.boxes, self.valid_patches = interval, 0, None, 0
        self.current_boxes = None

    @torch.inference_mode()
    def observe(self, rgb):
        age = self.frame_index % self.interval
        if self.boxes is None or age == 0:
            image = torch.as_tensor(np.array(rgb), device=next(self.detector.parameters()).device).permute(2, 0, 1).float()/255.
            result = self.detector([image])[0]
            self.boxes = result['boxes'][(result['labels']==1)&(result['scores']>=.5)]
        padding = 8+4*age
        self.current_boxes = self.boxes + self.boxes.new_tensor([-padding,-padding,padding,padding])
        self.frame_index += 1

    def forward(self, images, patches_per_image=80, **kwargs):
        if images.shape[:2] != (1, 1):
            raise ValueError("The online background sampler expects one current image")
        output = self.original(images, patches_per_image=8*patches_per_image, **kwargs)
        pixels = output[3][0, :, :2, 1, 1] * 4
        indices, self.valid_patches = background_indices(pixels, self.current_boxes, patches_per_image)
        values = [output[0], output[1][:, indices], output[2][:, indices], output[3][:, indices], output[4][indices]]
        if len(output) == 6:
            values.append(output[5][:, indices])
        return tuple(values)
