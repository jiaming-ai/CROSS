"""Train the BEV head (and the gravity head) on the frozen VGGT-Omega backbone.

    torchrun --nproc_per_node=4 -m cross_nav.train --config configs/cross_nav/<run>.yaml [key.sub=value ...]

Per step: the backbone runs without gradients on the batch's windows; the metric scale given to the head is the GT
scale of the pass (median GT / predicted depth of frame 0) with a random log-normal error, or with probability
`p_scale_head` the scale head's own estimate (a monocular pass without odometry); gravity is the GT up vector with a
small random tilt, or with probability `p_gravity_pred` the gravity head's estimate (no IMU).  Losses: binary
cross-entropy over the labelled cells (occupied vs free) + soft Dice of the occupied class, gravity 1 - cos.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from vggt_ft.train import apply_overrides, lr_factor

from .bev import FREE, OCC
from .data import NavStream
from .model import NavModel
from .uncertainty import error_loss


def gt_log_scale(depth_pred: torch.Tensor, depth_gt: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """(B,) log of the factor that makes the predicted depth metric: median GT / predicted over frame 0's valid pixels
    (all frames' if frame 0 has fewer than 500)."""
    out = []
    for b in range(depth_pred.shape[0]):
        m = masks[b, 0] & (depth_pred[b, 0] > 1e-4)
        if m.sum() >= 500:
            r = depth_gt[b, 0][m] / depth_pred[b, 0][m]
        else:
            mm = masks[b] & (depth_pred[b] > 1e-4)
            r = depth_gt[b][mm] / depth_pred[b][mm] if mm.sum() >= 500 else depth_pred.new_ones(1)
        out.append(torch.log(r.float().median().clamp(min=1e-4)))
    return torch.stack(out)


def tilt(up: torch.Tensor, deg: torch.Tensor) -> torch.Tensor:
    """Rotate unit vectors (B,3) by `deg` (B,) about random axes perpendicular to them."""
    a = F.normalize(torch.cross(up, torch.randn_like(up), dim=-1), dim=-1)
    th = torch.deg2rad(deg)[:, None]
    return F.normalize(up * th.cos() + torch.cross(a, up, dim=-1) * th.sin(), dim=-1)


def bev_losses(logits, bev, w_dice=0.5, pos_weight=1.0):
    lab = (bev == FREE) | (bev == OCC)
    tgt = (bev == OCC).float()
    m = lab.float()
    bce = F.binary_cross_entropy_with_logits(logits.float(), tgt, reduction="none",
                                             pos_weight=torch.tensor(pos_weight, device=logits.device))
    n = m.sum((1, 2)).clamp(min=1)
    l_bce = ((bce * m).sum((1, 2)) / n).mean()
    p = torch.sigmoid(logits.float())
    inter = (p * tgt * m).sum((1, 2))
    den = ((p + tgt) * m).sum((1, 2))
    dice = 1 - (2 * inter + 1) / (den + 1)
    has = (m.sum((1, 2)) > 0).float()
    l_dice = (dice * has).sum() / has.sum().clamp(min=1)
    return l_bce + w_dice * l_dice, {"loss_bce": l_bce.detach(), "loss_dice": l_dice.detach()}


@torch.no_grad()
def batch_metrics(logits, bev, vis):
    """IoU of occupied / free and accuracy over labelled cells, in cells seen by the window (vis bit 1) and unseen."""
    pred_occ = logits > 0
    out = {}
    lab = (bev == FREE) | (bev == OCC)
    occ = bev == OCC
    for name, sel in (("seen", (vis & 2) > 0), ("unseen", (vis & 2) == 0)):
        m = lab & sel
        if m.sum() == 0:
            continue
        tp = (pred_occ & occ & m).sum().float()
        fp = (pred_occ & ~occ & m).sum().float()
        fn = (~pred_occ & occ & m).sum().float()
        tn = (~pred_occ & ~occ & m).sum().float()
        out[f"iou_occ_{name}"] = tp / (tp + fp + fn).clamp(min=1)
        out[f"iou_free_{name}"] = tn / (tn + fp + fn).clamp(min=1)
        out[f"acc_{name}"] = (tp + tn) / m.sum()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = apply_overrides(yaml.safe_load(open(args.config)), args.overrides)
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=60))
    main_rank = rank == 0
    seed = cfg.get("seed", 0)
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    out_dir = Path(cfg["out_dir"]) / cfg["exp_name"]
    tc, lc, dc = cfg["train"], cfg["loss"], cfg["data"]
    if main_rank:
        out_dir.mkdir(parents=True, exist_ok=True)
        yaml.safe_dump(dict(cfg, world_size=world), open(out_dir / "config.yaml", "w"))
        log_f = open(out_dir / "log.jsonl", "a")
    else:
        globals()["print"] = lambda *a, **k: None

    torch.manual_seed(seed)                # identical head initialisation on every rank
    model = NavModel(cfg["backbone"], cfg.get("bev_head"), cfg.get("gravity_head")).cuda()
    torch.manual_seed(seed + rank)
    if tc.get("error_only", False):
        if not model.bev_head.uncertainty:
            raise ValueError("error_only requires bev_head.uncertainty")
        model.bev_head.requires_grad_(False)
        model.gravity_head.requires_grad_(False)
        model.bev_head.error_head.requires_grad_(True)
    trainable = [p for n, p in model.named_parameters() if p.requires_grad]
    n_bev = sum(p.numel() for p in model.bev_head.parameters())
    n_g = sum(p.numel() for p in model.gravity_head.parameters())
    print(f"BEV head {n_bev / 1e6:.1f}M, gravity head {n_g / 1e6:.2f}M parameters")
    decay = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim >= 2 and "pos" not in n.split(".")[-1]
             and "emb" not in n.split(".")[-1] and n.split(".")[-1] != "sub"]
    dset = {id(p) for p in decay}
    no_decay = [p for p in trainable if id(p) not in dset]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": tc.get("weight_decay", 0.05)},
                             {"params": no_decay, "weight_decay": 0.0}], lr=tc["lr"], betas=(0.9, 0.95), fused=True)
    start_step = 0
    ck_path = out_dir / "ckpt_last.pt"
    if cfg.get("resume", "auto") == "auto" and ck_path.exists():
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model.bev_head.load_state_dict(ck["bev_head"])
        model.gravity_head.load_state_dict(ck["gravity_head"])
        opt.load_state_dict(ck["optim"])
        start_step = ck["step"]
        print(f"resumed at step {start_step}")
        del ck
    elif cfg.get("init"):
        ck = torch.load(cfg["init"], map_location="cpu", weights_only=False)
        missing, unexpected = model.bev_head.load_state_dict(ck["bev_head"], strict=False)
        if unexpected or any(not k.startswith("error_head.") for k in missing):
            raise RuntimeError(f"incompatible warm start: {missing=}, {unexpected=}")
        model.gravity_head.load_state_dict(ck["gravity_head"])
    ema_decay = tc.get("ema", 0.0)
    ema = None
    if ema_decay and main_rank:
        ema = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        if start_step and (out_dir / "ema_last.pt").exists():
            ema.update({k: v.cuda() for k, v in torch.load(out_dir / "ema_last.pt", map_location="cpu").items()})

    stream = NavStream(dc, rank=rank, seed=seed + 1000 * start_step)
    stream._build()
    loader = DataLoader(stream, batch_size=None, num_workers=dc.get("num_workers", 8), pin_memory=True,
                        prefetch_factor=dc.get("prefetch", 2), persistent_workers=True)
    steps, warmup = tc["steps"], tc.get("warmup", 500)
    step = start_step
    model.train()
    agg = defaultdict(list)
    t0, t_data = time.time(), 0.0
    it = iter(loader)
    while step < steps:
        td = time.time()
        batch = next(it)
        t_data += time.time() - td
        f = lr_factor(step, steps, warmup, tc.get("min_lr_ratio", 0.05))
        for g in opt.param_groups:
            g["lr"] = tc["lr"] * f
        meta = {k: batch.pop(k) for k in ("dataset", "kind")}
        batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
        bb = model.backbone(batch["images"])
        B = batch["images"].shape[0]
        # metric scale input
        ls_gt = gt_log_scale(bb["depth"], batch["depths"], batch["masks"])
        sig = torch.rand(B, device=ls_gt.device) * lc.get("scale_noise", 0.1)
        ls_in = ls_gt + sig * torch.randn_like(ls_gt)
        if "log_scale_head" in bb and lc.get("p_scale_head", 0.0) > 0:
            use = torch.rand(B, device=ls_gt.device) < lc["p_scale_head"]
            ls_in = torch.where(use, bb["log_scale_head"].float(), ls_in)
        # gravity input
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            up_pred = model.gravity(bb).float()
        up_gt = batch["up_cam"]
        up_in = tilt(up_gt, torch.rand(B, device=up_gt.device) * lc.get("gravity_noise_deg", 2.0))
        use_g = torch.rand(B, device=up_gt.device) < lc.get("p_gravity_pred", 0.3)
        up_in = torch.where(use_g[:, None], up_pred.detach(), up_in)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model.heads(bb, ls_in, up_in, return_aux=True)
            logits = pred["logits"]
        l_bev, parts = bev_losses(logits, batch["bev"], lc.get("w_dice", 0.5), lc.get("pos_weight", 1.0))
        if pred["error_logits"] is not None:
            l_err = error_loss(pred["error_logits"], logits, batch["bev"])
            l_bev = l_bev + lc.get("w_error", 1.0) * l_err
            parts["loss_error"] = l_err.detach()
        l_g = (1 - (up_pred * up_gt).sum(-1)).mean()
        loss = l_bev + lc.get("w_gravity", 1.0) * l_g
        if not torch.isfinite(loss):
            print(f"step {step}: non-finite loss, skipped", flush=True)
            loss = loss.nan_to_num(0.0) * 0.0
        loss.backward()
        if world > 1:
            # mean gradient over ranks, one flat buffer (the heads are small)
            grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in trainable]
            flat = torch.cat([g.reshape(-1) for g in grads])
            dist.all_reduce(flat)
            flat /= world
            o = 0
            for p in trainable:
                p.grad = flat[o:o + p.numel()].view_as(p).clone()
                o += p.numel()
        gn = torch.nn.utils.clip_grad_norm_(trainable, tc.get("clip", 1.0))
        if torch.isfinite(gn):
            opt.step()
            if ema is not None:
                d = min(ema_decay, (1 + step) / (10 + step))
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        if n in ema:
                            ema[n].mul_(d).add_(p.detach(), alpha=1 - d)
        opt.zero_grad(set_to_none=True)
        step += 1
        mets = batch_metrics(logits.detach().float(), batch["bev"], batch["vis"])
        ang = torch.rad2deg(torch.acos((up_pred * up_gt).sum(-1).clamp(-1, 1))).mean()
        for k, v in {"loss": loss, "loss_gravity": l_g, "gravity_deg": ang, "gn": gn,
                     "scale_head_err": ((bb["log_scale_head"].float() - ls_gt).abs().mean()
                                        if "log_scale_head" in bb else torch.zeros(())), **parts, **mets}.items():
            v = float(v.detach()) if torch.is_tensor(v) else float(v)
            if math.isfinite(v):
                agg[k].append(v)
        for ds in set(meta["dataset"]):
            agg[f"n_{ds}"].append(sum(d == ds for d in meta["dataset"]))
        if step % cfg.get("log_every", 20) == 0:
            el = time.time() - t0
            msg = {k: round(float(np.mean(v)), 4) for k, v in agg.items() if not k.startswith("n_")}
            msg.update({k: int(np.sum(v)) for k, v in agg.items() if k.startswith("n_")})
            msg.update(step=step, lr=opt.param_groups[0]["lr"], s_per_step=round(el / cfg.get("log_every", 20), 2),
                       data_frac=round(t_data / el, 3), S=int(batch["images"].shape[1]), B=B,
                       mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1))
            if main_rank:
                print(json.dumps(msg), flush=True)
                log_f.write(json.dumps({"train": msg}) + "\n")
                log_f.flush()
            agg.clear()
            t0, t_data = time.time(), 0.0
        if main_rank and (step % tc.get("save_every", 500) == 0 or step == steps):
            sdh = {"bev_head": model.bev_head.state_dict(), "gravity_head": model.gravity_head.state_dict()}
            torch.save({**sdh, "optim": opt.state_dict(), "step": step}, out_dir / "ckpt_last.tmp")
            os.replace(out_dir / "ckpt_last.tmp", ck_path)
            meta_ck = {"step": step, "bev_cfg": cfg.get("bev_head"), "gravity_cfg": cfg.get("gravity_head"),
                       "backbone": cfg["backbone"]}
            if step % tc.get("keep_every", 1000) == 0 or step == steps:
                torch.save({**sdh, **meta_ck}, out_dir / f"heads_{step:06d}.pt")
                if ema is not None:
                    e = {n: p.detach() for n, p in model.named_parameters()
                         if n.startswith(("bev_head.", "gravity_head."))}
                    e.update(ema)
                    torch.save({"bev_head": {k[len("bev_head."):]: v for k, v in e.items() if k.startswith("bev_head.")},
                                "gravity_head": {k[len("gravity_head."):]: v for k, v in e.items()
                                                 if k.startswith("gravity_head.")}, **meta_ck},
                               out_dir / f"heads_{step:06d}_ema.pt")
            if ema is not None:
                torch.save({k: v.cpu() for k, v in ema.items()}, out_dir / "ema_last.tmp")
                os.replace(out_dir / "ema_last.tmp", out_dir / "ema_last.pt")
            print(f"saved step {step}", flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
