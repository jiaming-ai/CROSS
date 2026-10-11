"""Learned occupancy-decision error, distinct from Bernoulli entropy.

Unknown GT cells carry no error target. Unseen *labelled* cells do: the head
can learn when completion is reliable without declaring every unseen cell wrong.
Calibration must be checked on held-out scenes before interpreting this as risk.
"""
import torch
import torch.nn.functional as F

from .bev import FREE, OCC


def uncertainty_score(raw, mode="error"):
    """Error probability in error mode; 1 - loss weight in attenuated mode.

    Both equal sigmoid(raw), since exp(-softplus(raw)) = sigmoid(-raw),
    but only the error mode is trained against decision-error targets.
    """
    if mode not in ("error", "attenuated"):
        raise ValueError(f"unknown uncertainty mode: {mode}")
    return raw.float().sigmoid()


def attenuated_bce(logits, raw_uncertainty, labels, beta=0.3, pos_weight=1.0):
    """Regularized learned attenuation, not an exact Bernoulli likelihood.

    s=softplus(raw)>=0, w=exp(-s); loss = w*BCE + beta*s.
    The nonnegative s prevents loss -> -infinity at zero residual. beta*s
    prevents the all-uncertain solution. For fixed BCE=l, optimal
    s=max(0, log(l/beta)). Average labelled cells per window, then windows.
    """
    if beta <= 0:
        raise ValueError("attenuation beta must be positive")
    valid = ((labels == FREE) | (labels == OCC)).float()
    bce = F.binary_cross_entropy_with_logits(
        logits.float(), (labels == OCC).float(), reduction="none",
        pos_weight=logits.new_tensor(pos_weight, dtype=torch.float32))
    s = F.softplus(raw_uncertainty.float())
    weight = torch.exp(-s)
    count = valid.sum((1, 2)).clamp(min=1)

    def average(x):
        return ((x * valid).sum((1, 2)) / count).mean()

    data_term, regularizer = average(weight * bce), beta * average(s)
    parts = {"loss_att_data": data_term.detach(), "loss_att_reg": regularizer.detach(),
             "attenuation_weight": average(weight).detach(), "uncertainty_s": average(s).detach()}
    for name, mask in (("occ", valid * (labels == OCC)), ("free", valid * (labels == FREE))):
        parts[f"attenuation_weight_{name}"] = ((weight * mask).sum() / mask.sum().clamp(min=1)).detach()
    return data_term + regularizer, parts


def error_loss(error_logits, occupancy_logits, labels):
    valid = (labels == FREE) | (labels == OCC)
    target = ((occupancy_logits.detach() > 0) != (labels == OCC)).float()
    loss = F.binary_cross_entropy_with_logits(error_logits.float(), target, reduction="none")
    return (loss * valid).sum() / valid.sum().clamp(min=1)


def error_metrics(prob, error_prob, labels, seen, kind="error"):
    """Calibration for error probabilities; ranking only for attenuation scores."""
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
    if kind == "attenuated":
        for metrics in out.values():
            metrics.pop("brier", None)
            metrics.pop("ece", None)
    return out
