"""Training losses (VGGT-Omega paper, sec. 3, plus the two new heads).

camera : L1 on the 9-D pose encoding (translation, quaternion, fov_h, fov_w) of the normalised GT; translation and
         rotation only for frames whose pose is observable (pose_mask); the quaternion term is sign-invariant (the
         released checkpoint emits quaternions with negative real part).
depth  : sum c * (1 + 1/D) * |e| + c * |grad e| - alpha * log c,   e = D_pred - D  (normalised depth), trimmed at a
         quantile against bad GT.
point  : the same with e = unproject(D_pred, predicted camera) - P (points in frame 0's camera, normalised).
scale  : |log s_pred - log s*|, s* = the metric scale of the model's gauge (see scale_target), metric windows only.
covis  : BCE of the pairwise logits against the GT overlap fraction (soft targets), pairs i < j.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from vggt_omega.utils.pose_enc import encoding_to_camera, extri_intri_to_pose_encoding

from .geometry import cam_to_ref, unproject


def camera_loss(pose_enc, E_norm, K, hw, pose_mask=None, w_t=1.0, w_r=1.0, w_f=0.5):
    gt = extri_intri_to_pose_encoding(E_norm[:, :, :3], K, hw)
    w = torch.ones(pose_enc.shape[:2], device=pose_enc.device) if pose_mask is None else pose_mask.float()
    w = w / (w > 0).float().sum().clamp(min=1)       # by the number of supervised frames: float weights stay weights
    lt =((pose_enc[..., :3] - gt[..., :3]).abs().clamp(max=100).mean(-1) * w).sum()
    q, qg = pose_enc[..., 3:7], gt[..., 3:7]
    lr = (torch.minimum((q - qg).abs().sum(-1), (q + qg).abs().sum(-1)) * w).sum() / 4
    lf = (pose_enc[..., 7:] - gt[..., 7:]).abs().mean()
    return {"loss_T": lt, "loss_R": lr, "loss_FL": lf, "loss_camera": w_t * lt + w_r * lr + w_f * lf}


def _grad_terms(e, conf, mask, scales=4):
    """Multi-scale finite differences of the residual map e (N,H,W[,C]) where both pixels are valid, weighted by c."""
    tot, n = e.new_zeros(()), 0
    for s in range(scales):
        k = 2 ** s
        es, cs, ms = e[:, ::k, ::k], conf[:, ::k, ::k], mask[:, ::k, ::k]
        for dim in (1, 2):
            a = es.narrow(dim, 1, es.shape[dim] - 1) - es.narrow(dim, 0, es.shape[dim] - 1)
            m = ms.narrow(dim, 1, ms.shape[dim] - 1) & ms.narrow(dim, 0, ms.shape[dim] - 1)
            c = cs.narrow(dim, 0, cs.shape[dim] - 1)
            if a.dim() == 4:
                a = a.norm(dim=-1)
            else:
                a = a.abs()
            if m.any():
                tot = tot + (c * a)[m].mean()
                n += 1
    return tot / max(n, 1)


def conf_regression(e, conf, gt_depth, mask, alpha=0.2, trim=0.98, w_grad=1.0):
    """e residual (N,H,W) or (N,H,W,3), conf (N,H,W), gt_depth (N,H,W) normalised, mask (N,H,W) bool."""
    if mask.sum() < 100:
        z = (e.sum() + conf.sum()) * 0
        return z, z, z
    err = e.norm(dim=-1) if e.dim() == 4 else e.abs()
    wd = 1.0 + 1.0 / gt_depth.clamp(min=0.05)          # (1 + 1/D) of the paper, capped for points very near the camera
    reg = (conf * wd * err)[mask]
    if 0 < trim < 1:
        q = torch.quantile(reg.detach().float()[:4_000_000], trim)
        keep = reg <= q
        reg = reg[keep]
    loss_reg = reg.mean()
    loss_conf = -alpha * torch.log(conf[mask]).mean()
    loss_grad = _grad_terms(e, conf, mask) * w_grad
    return loss_reg, loss_grad, loss_conf


def depth_loss(pred_depth, pred_conf, gt_depth, mask, alpha=0.2, trim=0.98, w_grad=1.0):
    pred = pred_depth[..., 0]
    B, S, H, W = gt_depth.shape
    e = (pred - gt_depth).reshape(B * S, H, W)
    r, g, c = conf_regression(e, pred_conf.reshape(B * S, H, W), gt_depth.reshape(B * S, H, W),
                              mask.reshape(B * S, H, W), alpha, trim, w_grad)
    return {"loss_reg_depth": r, "loss_grad_depth": g, "loss_conf_depth": c, "loss_depth": r + g + c}


def point_loss(pred_depth, pred_conf, pose_enc, gt_points, gt_depth, mask, hw, alpha=0.2, trim=0.98, w_grad=1.0):
    """Points from the predicted depth and the predicted cameras (frame 0 = predicted world) vs GT points in frame 0.
    pred_depth / mask cover the frames of `gt_points` (pose_enc must be sliced to them by the caller)."""
    E, K = encoding_to_camera(pose_enc, hw)
    E4 = torch.eye(4, device=E.device).expand(*E.shape[:2], 4, 4).clone()
    E4[..., :3, :] = E
    pts = cam_to_ref(unproject(pred_depth[..., 0], K), E4)
    B, S, H, W = gt_depth.shape
    e = (pts - gt_points).reshape(B * S, H, W, 3)
    r, g, c = conf_regression(e, pred_conf.reshape(B * S, H, W), gt_depth.reshape(B * S, H, W),
                              mask.reshape(B * S, H, W), alpha, trim, w_grad)
    return {"loss_reg_point": r, "loss_grad_point": g, "loss_point": r + g + c}


@torch.no_grad()
def scale_target(pred_depth, gt_depth_metric, mask, gt_scale, mode="pred"):
    """log s* per window.  mode 'pred': median log ratio of metric GT depth to the predicted depth (the factor that
    makes the model's own output metric); 'gt': the GT normalisation scale (mean point distance)."""
    if mode == "gt":
        return torch.log(gt_scale)
    lr = torch.log(gt_depth_metric.clamp(min=1e-6)) - torch.log(pred_depth[..., 0].float().clamp(min=1e-6))
    out = []
    for b in range(lr.shape[0]):
        v = lr[b][mask[b]]
        out.append(v.median() if v.numel() > 50 else torch.log(gt_scale[b]))
    return torch.stack(out)


@torch.no_grad()
def dense_scale_target(pred_depth, gt_depth_metric, mask, patch_hw):
    """Per-patch targets of a dense scale head: mean log(metric GT / predicted depth) over each patch's valid pixels,
    (B,S,h,w), and the fraction of valid pixels per patch."""
    B, S, H, W = gt_depth_metric.shape
    h, w = patch_hw
    p = H // h
    lr = torch.log(gt_depth_metric.clamp(min=1e-6)) - torch.log(pred_depth[..., 0].float().clamp(min=1e-6))
    v = mask.float()
    lr = torch.where(mask, lr, torch.zeros_like(lr))
    num = F.avg_pool2d((lr * v)[..., :h * p, :w * p].reshape(B * S, 1, h * p, w * p), p).reshape(B, S, h, w)
    den = F.avg_pool2d(v[..., :h * p, :w * p].reshape(B * S, 1, h * p, w * p), p).reshape(B, S, h, w)
    return num / den.clamp(min=1e-6), den


def covis_loss(logits, target, window_w=None):
    """window_w (B,): weight of each window (0 = no covisibility labels, e.g. windows without poses)."""
    S = logits.shape[1]
    iu = torch.triu_indices(S, S, 1, device=logits.device)
    lg, tg = logits[:, iu[0], iu[1]], target[:, iu[0], iu[1]]
    if window_w is None:
        out = {"loss_covis": F.binary_cross_entropy_with_logits(lg, tg)}
    else:
        bce = F.binary_cross_entropy_with_logits(lg, tg, reduction="none").mean(1)
        out = {"loss_covis": (bce * window_w).sum() / window_w.sum().clamp(min=1)}
    with torch.no_grad():
        p = torch.sigmoid(lg)
        neg, pos = tg < 0.02, tg > 0.3
        nan = torch.full((), float("nan"), device=p.device)
        out["covis_p_neg"] = p[neg].mean() if neg.any() else nan
        out["covis_p_pos"] = p[pos].mean() if pos.any() else nan
        out["covis_frac_neg"] = neg.float().mean()
    return out
