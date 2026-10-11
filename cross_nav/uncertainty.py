"""Learned occupancy-decision error, distinct from Bernoulli entropy.

Unknown GT cells carry no error target. Unseen *labelled* cells do: the head
can learn when completion is reliable without declaring every unseen cell wrong.
Calibration must be checked on held-out scenes before interpreting this as risk.
"""
import torch
import torch.nn.functional as F

from .bev import FREE, OCC


def error_loss(error_logits, occupancy_logits, labels):
    valid = (labels == FREE) | (labels == OCC)
    target = ((occupancy_logits.detach() > 0) != (labels == OCC)).float()
    loss = F.binary_cross_entropy_with_logits(error_logits.float(), target, reduction="none")
    return (loss * valid).sum() / valid.sum().clamp(min=1)


def error_metrics(prob, error_prob, labels, seen):
    """Numpy calibration and selective error metrics, separately seen/unseen."""
    import numpy as np
    out = {}
    for name, selection in (("all", np.ones_like(seen)), ("seen", seen), ("unseen", ~seen)):
        m = selection & ((labels == FREE) | (labels == OCC))
        u = error_prob[m].astype(float)
        e = ((prob[m] > .5) != (labels[m] == OCC)).astype(float)
        if not len(u):
            out[name] = {"n": 0}
            continue
        bins = np.minimum((u * 10).astype(int), 9)
        ece = sum(abs(u[bins == b].sum() - e[bins == b].sum()) for b in range(10)) / len(u)
        order = np.argsort(u, kind="stable")
        out[name] = {"n": len(u), "mean_uncertainty": float(u.mean()), "error": float(e.mean()),
                     "brier": float(np.mean((u - e) ** 2)), "ece": float(ece),
                     "error_at_50pct_coverage": float(e[order[:max(1, len(u)//2)]].mean())}
    return out
