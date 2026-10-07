"""Fine-tune VGGT-Omega with the metric-scale and covisibility heads.

    torchrun --nproc_per_node=8 -m vggt_ft.train --config configs/vggt_ft/<run>.yaml [key.sub=value ...]
    (multi-node: add --nnodes / --node_rank / --rdzv_endpoint; images_per_gpu is per GPU)

Per step: GT covisibility of every frame pair of each window (geometry.gt_covisibility); frames not connected to frame
0 through shared scene get no pose / depth / point loss when they come from elsewhere (injected negatives, other
sessions), since their relative pose is unobservable; the scene is normalised to VGGT's gauge with the connected frames
only; the scale head is supervised on metric windows only.
"""
from __future__ import annotations

import argparse
import contextlib
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
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .dataio.windows import TrainStream
from .geometry import connected_to_ref, gt_covisibility, normalize_window
from .losses import camera_loss, covis_loss, dense_scale_target, depth_loss, point_loss, scale_target
from .model import VGGTOmegaFT, heads_state_dict


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for ov in overrides:
        k, v = ov.split("=", 1)
        d = cfg
        keys = k.split(".")
        for kk in keys[:-1]:
            d = d.setdefault(kk, {})
        d[keys[-1]] = yaml.safe_load(v)
    return cfg


def param_group(name: str) -> str:
    for g in ("scale_head", "scale_encoder", "covis_head", "camera_head", "dense_head"):
        if name.startswith(g + "."):
            return g
    return "patch_embed" if name.startswith("aggregator.patch_embed") else "aggregator"


def build_optimizer(model, tc):
    groups = defaultdict(lambda: ([], []))
    for n, p in model.named_parameters():
        if p.requires_grad:
            no_wd = p.ndim < 2 or n.endswith(".bias") or "token" in n or "embed" in n.split(".")[-1]
            groups[param_group(n)][1 if no_wd else 0].append(p)
    mult = tc.get("lr_mult", {})
    pgs = []
    for g, (wd, nwd) in groups.items():
        lr = tc["lr"] * mult.get(g, 1.0)
        if wd:
            pgs.append({"params": wd, "lr": lr, "weight_decay": tc.get("weight_decay", 0.05), "name": g})
        if nwd:
            pgs.append({"params": nwd, "lr": lr, "weight_decay": 0.0, "name": g + "_nowd"})
    opt = torch.optim.AdamW(pgs, betas=(0.9, 0.95), fused=True)
    for g in opt.param_groups:
        g["base_lr"] = g["lr"]
    return opt


def lr_factor(step, total, warmup, min_ratio=0.05):
    if step < warmup:
        return (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))


def distill_losses(pred, teach, pose_w, masks, pose_mask):
    """Stay close to the released model: L1 to its pose encoding (observable frames, weighted by pose_w) and to its log
    depth, weighted towards pixels without GT depth (sky, MVS holes), where nothing else supervises the student."""
    lp = ((pred["pose_enc"] - teach["pose_enc"]).abs().mean(-1) * pose_w).sum() / pose_mask.float().sum().clamp(min=1)
    ld = (torch.log(pred["depth"][..., 0].clamp(min=1e-4)) - torch.log(teach["depth"][..., 0].clamp(min=1e-4))).abs()
    m = pose_mask[:, :, None, None].float()
    wd = (1.0 + (~masks).float()) * m
    ld = (ld * wd).sum() / wd.sum().clamp(min=1)
    return {"loss_distill_pose": lp, "loss_distill_depth": ld}


def teacher_pose_weights(meta, lc, B, device):
    """(B,) factors for the GT camera loss and for pose distillation: sequential windows of `teacher_pose_datasets`
    (real data with SfM / MVS poses, which the released model fits less tightly than synthetic GT) take their pose
    mostly from the released model; cross-session windows always keep the GT (the teacher is weak there)."""
    gt_w, di_w = torch.ones(B, device=device), torch.ones(B, device=device)
    names = set(lc.get("teacher_pose_datasets", []))
    if meta is None or not names:
        return gt_w, di_w
    for b, (ds, kind) in enumerate(zip(meta["dataset"], meta["kind"])):
        if ds in names and kind != "cross":
            gt_w[b] = lc.get("teacher_pose_gt_w", 0.2)
            di_w[b] = lc.get("teacher_pose_distill_w", 5.0)
    return gt_w, di_w


def scale_distill_target(steach, pred, mask=None):
    """(B,) log scale that makes the student's depth metric according to a teacher with its own scale head:
    metric = s_t * d_t, so  log s* = log s_t + median(log d_t - log d_s)  over the window's pixels (no GT needed)."""
    lr = torch.log(steach["depth"][..., 0].float().clamp(min=1e-4)) - torch.log(pred["depth"][..., 0].float().clamp(min=1e-4))
    med = torch.stack([(lr[b][mask[b]] if mask is not None and mask[b].sum() > 100 else lr[b].flatten()).median()
                       for b in range(lr.shape[0])])
    return steach["log_scale"].float() + med


def compute_losses(pred, batch, lc, hw, teach=None, meta=None, steach=None):
    """All losses of one batch (dict of scalars) and the total."""
    depths, masks = batch["depths"], batch["masks"]
    covis_gt = gt_covisibility(depths, masks, batch["extrinsics"], batch["intrinsics"], batch["world_id"])
    conn = connected_to_ref(covis_gt, lc.get("connect_thr", 0.05))
    mode = lc.get("mask_unobservable", "unconnected_other_session")
    if mode == "unconnected":
        pose_mask = conn
    elif mode == "unconnected_other_session":
        pose_mask = conn | (batch["same_session"] & ~batch["is_neg"])
    else:
        pose_mask = torch.ones_like(conn)
    posed = batch.get("posed")
    if posed is not None:
        # windows without reliable poses (depth-only sequences) train the scale head only
        pose_mask = pose_mask & posed[:, None]
    E_n, d_n, pts_n, scale = normalize_window(batch["extrinsics"], depths, masks, batch["intrinsics"],
                                              scale_frames=pose_mask)
    dm = masks & pose_mask[:, :, None, None]
    # a window without enough valid GT depth has no translation scale (normalize_window would divide by ~0, e.g. DL3DV
    # frames whose teacher labels were rejected): no camera / scale loss for it
    win_ok = dm.sum((1, 2, 3)) > 1000
    pose_mask = pose_mask & win_ok[:, None]
    out = {"frac_win_nodepth": 1 - win_ok.float().mean()}
    raw_pred = pred
    if lc.get("scale_align") == "invariant":
        # predictions divided by their window gauge (median predicted / GT depth, gradient through the median): every
        # loss below is in GT units and exactly invariant to a global rescale of the outputs, so shrinking all outputs
        # cannot lower it (the GT-side version below can drift without an anchor such as distillation; cross-uw v19)
        ratio = pred["depth"][..., 0].float() / d_n.clamp(min=1e-6)
        s_al = torch.stack([ratio[b][dm[b]].median() if dm[b].sum() > 100 else ratio.new_ones(())
                            for b in range(ratio.shape[0])]).clamp(0.2, 5.0)
        pred = dict(pred)
        pred["depth"] = pred["depth"] / s_al[:, None, None, None, None]
        # out of place: s_al carries a gradient, so the division's backward needs the unmodified translations
        pe = pred["pose_enc"]
        pred["pose_enc"] = torch.cat([pe[..., :3] / s_al[:, None, None], pe[..., 3:]], -1)
        out["gauge_ratio"] = s_al.detach().mean()
    elif lc.get("scale_align", False):
        # the released model does not use our normalisation (its gauge is 0.83-1.17x unit mean point distance depending
        # on the dataset, 2026-10-04): scale the GT of each window to the prediction's own gauge (median depth ratio, no
        # gradient) so the losses do not push the whole output to a new scale; metric scale is the scale head's job
        with torch.no_grad():
            ratio = pred["depth"][..., 0].float() / d_n.clamp(min=1e-6)
            s_al = torch.stack([ratio[b][dm[b]].median() if dm[b].sum() > 100 else ratio.new_ones(())
                                for b in range(ratio.shape[0])]).clamp(0.2, 5.0)
        E_n = E_n.clone()
        E_n[:, :, :3, 3] = E_n[:, :, :3, 3] * s_al[:, None, None]
        d_n = d_n * s_al[:, None, None, None]
        pts_n = pts_n * s_al[:, None, None, None, None]
        out["gauge_ratio"] = s_al.mean()
    gt_w, di_w = teacher_pose_weights(meta, lc, pose_mask.shape[0], pose_mask.device)
    cl = camera_loss(pred["pose_enc"], E_n, batch["intrinsics"], hw, pose_mask=pose_mask.float() * gt_w[:, None])
    out.update(cl)
    total = lc.get("w_camera", 5.0) * cl["loss_camera"]
    if lc.get("w_depth", 1.0) > 0:
        dl = depth_loss(pred["depth"], pred["depth_conf"], d_n, dm, alpha=lc.get("alpha", 0.2))
        out.update(dl)
        total = total + lc.get("w_depth", 1.0) * dl["loss_depth"]
    if lc.get("w_point", 1.0) > 0:
        pl = point_loss(pred["depth"], pred["depth_conf"], pred["pose_enc"], pts_n, d_n, dm, hw,
                        alpha=lc.get("alpha", 0.2))
        out.update(pl)
        total = total + lc.get("w_point", 1.0) * pl["loss_point"]
    if "log_scale" in pred and lc.get("w_scale", 1.0) > 0:
        # the scale target needs metric depth only: unposed windows use every frame's valid depth
        dm_s = dm if posed is None else masks & (pose_mask | ~posed[:, None])[:, :, None, None]
        ok_s = dm_s.sum((1, 2, 3)) > 1000
        tgt = scale_target(raw_pred["depth"], depths, dm_s, scale, lc.get("scale_target", "pred"))
        m = (batch["metric"] & ok_s).float()
        if "pseudo_metric" in batch:
            m = m * torch.where(batch["pseudo_metric"], lc.get("w_scale_pseudo", 1.0), 1.0)
        err = (pred["log_scale"] - tgt).abs()
        out["loss_scale"] = (err * m).sum() / m.sum().clamp(min=1)
        out["scale_err_metric"] = out["loss_scale"].detach()
        out["frac_metric"] = m.mean()
        total = total + lc.get("w_scale", 1.0) * out["loss_scale"]
        if "scale_r" in pred and lc.get("w_scale_dense", 1.0) > 0:
            # dense scale votes: every patch with enough GT depth predicts its own log(metric / predicted depth)
            tgt_p, frac = dense_scale_target(raw_pred["depth"], depths, dm_s, tuple(pred["scale_r"].shape[-2:]))
            wp = (frac > 0.25).float() * m[:, None, None, None]
            out["loss_scale_dense"] = ((pred["scale_r"] - tgt_p).abs() * wp).sum() / wp.sum().clamp(min=1)
            total = total + lc.get("w_scale_dense", 1.0) * out["loss_scale_dense"]
    if steach is not None and "log_scale" in pred:
        # scale-head distillation from a fine-tuned model's head (loss.scale_teacher), on every window
        with torch.no_grad():
            tgt_t = scale_distill_target(steach, raw_pred)
        out["loss_scale_distill"] = (pred["log_scale"] - tgt_t).abs().mean()
        total = total + lc.get("w_scale_distill", 1.0) * out["loss_scale_distill"]
    if "covis_logits" in pred and pred["covis_logits"].shape[1] < 2:
        # 1-frame windows have no pairs: keep the covisibility head in the graph with zero weight (DDP needs a gradient
        # for every trainable parameter on every rank)
        total = total + 0.0 * pred["covis_logits"].sum()
    elif "covis_logits" in pred and lc.get("w_covis", 1.0) > 0:
        cv = covis_loss(pred["covis_logits"], covis_gt, None if posed is None else posed.float())
        out.update(cv)
        total = total + lc.get("w_covis", 1.0) * cv["loss_covis"]
    if teach is not None:
        dl = distill_losses(raw_pred, teach, pose_mask.float() * di_w[:, None], masks, pose_mask)
        out.update(dl)
        total = total + lc.get("w_distill_pose", 1.0) * dl["loss_distill_pose"] \
            + lc.get("w_distill_depth", 0.5) * dl["loss_distill_depth"]
    out["frac_unconnected"] = 1 - conn.float().mean()
    out["gt_scale"] = scale.mean()
    out["loss"] = total
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

    model = VGGTOmegaFT(scale_head=cfg.get("scale_head", True), covis_head=cfg.get("covis_head", True),
                        scale_head_cfg=cfg.get("scale_head_cfg"))
    model.load_weights(cfg["pretrained"], verbose=main_rank)
    start_step = 0
    resume = cfg.get("resume")
    if resume == "auto" and (out_dir / "ckpt_last.pt").exists():
        resume = str(out_dir / "ckpt_last.pt")
    elif resume == "auto":
        resume = None
    ck = None
    if resume:
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=False)
        start_step = ck.get("step", 0)
        print(f"resumed {resume} at step {start_step}")
    elif cfg.get("init"):
        model.load_weights(cfg["init"], verbose=main_rank)
    model.set_trainable(tc.get("trainable", ["all"]))
    k_frozen = int(tc.get("freeze_blocks", 0))
    if k_frozen:
        # the first k blocks (frame + inter-frame) and the input tokens stay as released: nothing before block k needs
        # gradients, so their backward pass is skipped too
        agg = model.aggregator
        for i in range(k_frozen):
            agg.frame_blocks[i].requires_grad_(False)
            agg.inter_frame_blocks[i].requires_grad_(False)
        agg.camera_token.requires_grad_(False)
        agg.register_token.requires_grad_(False)
        agg.patch_embed.requires_grad_(False)
    model.set_gradient_checkpointing(patch_embed=any(p.requires_grad for p in model.aggregator.patch_embed.parameters()))
    model.dense_chunk = tc.get("dense_chunk", 4)
    model = model.cuda()
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable {n_tr / 1e6:.1f}M of {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    opt = build_optimizer(model, tc)
    if ck is not None and "optim" in ck and not cfg.get("reset_optim", False):
        try:
            opt.load_state_dict(ck["optim"])
            for g in opt.param_groups:
                g["base_lr"] = tc["lr"] * tc.get("lr_mult", {}).get(g["name"].replace("_nowd", ""), 1.0)
        except ValueError as e:
            print("optimizer state not restored:", e)
    del ck
    teacher = None
    if lc.get("distill", False):
        from vggt_omega.models import VGGTOmega
        teacher = VGGTOmega()
        sd_t = torch.load(cfg.get("teacher", cfg["pretrained"]), map_location="cpu", weights_only=False)
        sd_t = sd_t["model"] if isinstance(sd_t, dict) and isinstance(sd_t.get("model"), dict) else sd_t
        missing_t, _ = teacher.load_state_dict(sd_t, strict=False)
        if missing_t:      # a wrapped / mismatched checkpoint must not leave a randomly initialised teacher
            raise RuntimeError(f"distillation teacher lacks {len(missing_t)} weights, e.g. {missing_t[:3]}")
        del sd_t
        teacher = teacher.to(torch.bfloat16).cuda().eval().requires_grad_(False)
        teacher.camera_head.float()
        teacher.dense_head.float()
    scale_teacher = None
    if lc.get("scale_teacher"):
        # a fine-tuned checkpoint whose scale head the student's head learns to reproduce (heads-only runs on a blended
        # backbone, where the jointly trained head does not transfer)
        sd_t = torch.load(lc["scale_teacher"], map_location="cpu", weights_only=False)
        scale_teacher = VGGTOmegaFT(scale_head=True, covis_head=False, scale_head_cfg=sd_t.get("scale_head_cfg"))
        scale_teacher.load_state_dict(sd_t["model"], strict=False)
        del sd_t
        scale_teacher = scale_teacher.to(torch.bfloat16).cuda().eval().requires_grad_(False)
        for h in (scale_teacher.camera_head, scale_teacher.dense_head, scale_teacher.scale_head):
            h.float()
    ema_decay = tc.get("ema", 0.0)
    ema = None
    if ema_decay and main_rank:
        # exponential moving average of the trainable weights (fp32, on rank 0's GPU); saved as *_ema_bf16.pt
        ema = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        if resume and (out_dir / "ema_last.pt").exists():
            ema.update({k: v.cuda() for k, v in torch.load(out_dir / "ema_last.pt", map_location="cpu").items()
                        if k in ema})
    fwd = DDP(model, device_ids=[local_rank], find_unused_parameters=False, gradient_as_bucket_view=True,
              static_graph=False) if world > 1 else model

    stream = TrainStream(dc, rank=rank, seed=seed + 1000 * start_step)
    # build the dataset index once here: forked data workers inherit it instead of each re-reading every scene.json
    # of every dataset from the shared file system (minutes per worker; identical pools, per-worker RNG streams)
    stream._build()
    loader = DataLoader(stream, batch_size=None, num_workers=dc.get("num_workers", 8), pin_memory=True,
                        prefetch_factor=dc.get("prefetch", 2), persistent_workers=True)
    steps, warmup = tc["steps"], tc.get("warmup", 500)
    model.train()
    step = start_step
    t0, t_data = time.time(), 0.0
    agg = defaultdict(list)
    it = iter(loader)
    while step < steps:
        td = time.time()
        batch = next(it)
        t_data += time.time() - td
        f = lr_factor(step, steps, warmup, tc.get("min_lr_ratio", 0.05))
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * f
        meta = {k: batch.pop(k) for k in ("dataset", "kind")}
        batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
        hw = tuple(batch["images"].shape[-2:])
        teach = None
        if teacher is not None:
            with torch.no_grad():
                teach = teacher(batch["images"])
        steach = None
        if scale_teacher is not None:
            with torch.no_grad():
                steach = scale_teacher(batch["images"])
        # known intrinsics: a canonical-camera scale head converts with the calibrated focal (others ignore them)
        pred = fwd(batch["images"], intrinsics=batch["intrinsics"])
        losses = compute_losses(pred, batch, lc, hw, teach, meta, steach)
        loss = losses["loss"]
        if not torch.isfinite(loss):
            print(f"step {step}: non-finite loss, skipped", flush=True)
            loss = loss.nan_to_num(0.0) * 0.0
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], tc.get("clip", 1.0))
        if torch.isfinite(gn):
            opt.step()
            if ema is not None:
                d = min(ema_decay, (1 + step) / (10 + step))   # warm start (global step: a resume keeps the average)
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        if n in ema:
                            ema[n].mul_(d).add_(p.detach(), alpha=1 - d)
        opt.zero_grad(set_to_none=True)
        step += 1
        for k, v in losses.items():
            v = float(v.detach()) if torch.is_tensor(v) else float(v)
            if math.isfinite(v):
                agg[k].append(v)
        for ds in set(meta["dataset"]):
            agg[f"n_{ds}"].append(sum(d == ds for d in meta["dataset"]))
        agg["n_cross"].append(sum(k == "cross" for k in meta["kind"]))
        agg["gn"].append(float(gn))
        if step % cfg.get("log_every", 20) == 0:
            el = time.time() - t0
            msg = {k: round(float(np.mean(v)), 4) for k, v in agg.items() if not k.startswith("n_")}
            msg.update({k: int(np.sum(v)) for k, v in agg.items() if k.startswith("n_")})
            msg.update(step=step, lr=opt.param_groups[0]["lr"], s_per_step=round(el / cfg.get("log_every", 20), 2),
                       data_frac=round(t_data / el, 3), S=int(batch["images"].shape[1]), B=int(batch["images"].shape[0]),
                       mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1))
            if world > 1:          # mean of the main losses over ranks
                keys = [k for k in ("loss", "loss_camera", "loss_depth", "loss_point", "loss_scale", "loss_covis")
                        if k in msg]
                t = torch.tensor([msg[k] for k in keys], device="cuda")
                dist.all_reduce(t)
                for k, v in zip(keys, (t / world).tolist()):
                    msg[k] = round(v, 4)
            if main_rank:
                print(json.dumps(msg), flush=True)
                log_f.write(json.dumps({"train": msg}) + "\n")
                log_f.flush()
            agg.clear()
            t0, t_data = time.time(), 0.0
        if main_rank and (step % tc.get("save_every", 1000) == 0 or step == steps):
            sd = model.state_dict()
            torch.save({"model": sd, "optim": opt.state_dict(), "step": step}, out_dir / "ckpt_last.tmp")
            os.replace(out_dir / "ckpt_last.tmp", out_dir / "ckpt_last.pt")
            if step % tc.get("keep_every", 2000) == 0 or step == steps:
                torch.save({"model": {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in sd.items()},
                            "step": step, "scale_head_cfg": cfg.get("scale_head_cfg")}, out_dir / f"ckpt_{step:06d}_bf16.pt")
                torch.save(heads_state_dict(model), out_dir / f"heads_{step:06d}.pt")
                if ema is not None:
                    sd_ema = {k: (ema[k] if k in ema else v) for k, v in sd.items()}
                    torch.save({"model": {k: v.to(torch.bfloat16) if v.is_floating_point() else v
                                          for k, v in sd_ema.items()}, "step": step,
                                "scale_head_cfg": cfg.get("scale_head_cfg")},
                               out_dir / f"ckpt_{step:06d}_ema_bf16.pt")
            if ema is not None:
                torch.save({k: v.cpu() for k, v in ema.items()}, out_dir / "ema_last.tmp")
                os.replace(out_dir / "ema_last.tmp", out_dir / "ema_last.pt")
            print(f"saved step {step}", flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
