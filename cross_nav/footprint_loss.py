"""Optional supervision of the same disk-pooled occupancy used by the planner."""
import cv2
import torch
import torch.nn.functional as F


def footprint_bce(logits, labels, radius_cells=4):
    """BCE on max occupancy in a disk, only when its entire GT footprint is known.

    This supplements cell BCE; it is not a calibrated collision probability.
    Outside-grid labels are unknown and therefore excluded from supervision.
    """
    if radius_cells < 0 or int(radius_cells) != radius_cells:
        raise ValueError("radius_cells must be a nonnegative integer")
    if logits.shape != labels.shape or logits.ndim != 3:
        raise ValueError("expected matching B,H,W logits and labels")
    r = int(radius_cells)
    h, w = logits.shape[-2:]
    lp = F.pad(logits.float(), (r, r, r, r), value=-torch.inf)
    yp = F.pad(labels, (r, r, r, r), value=0)
    pieces = []
    known = torch.ones_like(labels, dtype=torch.bool)
    occupied = torch.zeros_like(known)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if not kernel[dy + r, dx + r]:
                continue
            ys, xs = slice(r + dy, r + dy + h), slice(r + dx, r + dx + w)
            pieces.append(lp[:, ys, xs])
            y = yp[:, ys, xs]
            known = known & ((y == 1) | (y == 2))
            occupied = occupied | (y == 2)
    pooled = torch.stack(pieces).amax(0)
    loss = F.binary_cross_entropy_with_logits(pooled, occupied.float(), reduction="none")
    count = known.sum((1, 2))
    per_sample = (loss * known).sum((1, 2)) / count.clamp_min(1)
    return per_sample.mean(), known.float().mean().detach()
