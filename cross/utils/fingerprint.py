"""Content keys of image tensors, computed on their device (for caches of per-image network outputs)."""
from __future__ import annotations

import torch

_WEIGHTS: dict = {}


def content_keys(images: torch.Tensor) -> list:
    """One key per image of a batch (N, ...): two random integer projections of the raw bits of each image, summed
    with 64-bit wrap-around.  Integer sums do not depend on the reduction order, so an image gets the same key in any
    batch; different images collide with probability ~2^-128.  One device synchronisation for the whole batch."""
    if images.dtype == torch.float64:
        images = images.float()
    flat = images.contiguous().reshape(len(images), -1)
    if flat.dtype in (torch.float32, torch.int32):
        bits = flat.view(torch.int32).to(torch.int64)
    elif flat.dtype in (torch.float16, torch.bfloat16, torch.int16):
        bits = flat.view(torch.int16).to(torch.int64)
    else:
        bits = flat.to(torch.int64)
    key = (flat.shape[1], str(flat.device))
    w = _WEIGHTS.get(key)
    if w is None:
        g = torch.Generator(device="cpu").manual_seed(0)
        w = _WEIGHTS[key] = torch.randint(1, 2 ** 40, (2, flat.shape[1]), generator=g, dtype=torch.int64).to(flat.device)
    sums = torch.stack([(bits * wi).sum(dim=1) for wi in w], dim=1).cpu().tolist()
    shape = tuple(images.shape[1:])
    return [(shape, str(images.dtype), a, b) for a, b in sums]
