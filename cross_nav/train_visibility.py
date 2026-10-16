"""Train the BEV observability head: P(a cell is observed by the window's input views), a second BEV head next to the
occupancy head (cross_nav.infer.BEVPredictor visibility_path), on the frozen backbone.

    python -m cross_nav.train_visibility --config cross_nav/configs/train/vfin2_vis_ft.yaml --out DIR [--steps N]

Config: the occupancy training's (cross_nav.train) with `data.observation_targets: true` (targets from the input
views' depths, cross_nav.observation) plus
    init       occupancy heads file (cross_nav.train): the head is a copy of its BEV head, and its gravity head is used
               frozen (the checkpoint records its hash; BEVPredictor's visibility_gravity_path must be this file)
    vis_init   optional trained observability checkpoint to fine-tune (e.g. on another backbone); without it, the
               copied occupancy head starts with a zeroed output projection (the 2026-10-11 recipe)
    train      lr (2e-4), steps (2000), weight_decay (0.01), clip (1.0), inputs:
               head   the backbone's scale-head scale and the gravity head's estimate (2026-10-11 recipe)
               gtmix  the occupancy training's inputs: GT scale with noise (scale head with p_scale_head), GT gravity
                      tilted (gravity head with p_gravity_pred); loss section's scale_noise / gravity_noise_deg.
                      Needed with a backbone whose scale head is less accurate (vfin2): with `head` the head learned
                      to predict less observed ground (outputs/2026-10-16_vfin2_heads)
Loss: BCE on labelled (known) cells. Writes visibility_<step>.pt every 500 steps and visibility_final.pt
({'task': 'observability', 'visibility_head', 'config', 'step', 'frozen_gravity_hash'}), log.jsonl, receipt.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from .data import NavStream
from .model import NavModel
from .train import expand_env, gt_log_scale, tilt


def digest(module) -> str:
    """SHA-256 of a module's state dict (names and raw bytes)."""
    h = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def head_inputs(bb, batch, up_pred, loss_cfg, mode):
    """(log metric scale (B,), gravity up (B, 3)) given to the head: the backbone's estimates (mode 'head') or the
    occupancy training's mix of GT with noise and the estimates (mode 'gtmix')."""
    ls = bb["log_scale_head"].float()
    if mode == "head":
        return ls, up_pred
    B = ls.shape[0]
    lg = gt_log_scale(bb["depth"], batch["depths"], batch["masks"])
    lg = lg + torch.rand(B, device=lg.device) * loss_cfg.get("scale_noise", 0.1) * torch.randn_like(lg)
    ls = torch.where(torch.rand(B, device=lg.device) < loss_cfg.get("p_scale_head", 0.0), ls, lg)
    ug = tilt(batch["up_cam"].float(), torch.rand(B, device=lg.device) * loss_cfg.get("gravity_noise_deg", 2.0))
    up = torch.where((torch.rand(B, device=lg.device) < loss_cfg.get("p_gravity_pred", 0.3))[:, None], up_pred, ug)
    return ls, up


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, help="overrides train.steps")
    a = ap.parse_args()
    cfg = expand_env(yaml.safe_load(Path(a.config).read_text()))
    tc = {"lr": 2e-4, "steps": 2000, "weight_decay": 0.01, "clip": 1.0, "inputs": "head", **cfg.get("train", {})}
    steps = a.steps or int(tc["steps"])
    if not cfg["data"].get("observation_targets"):
        raise SystemExit("set data.observation_targets: true (the observability targets)")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "visibility_final.pt").exists():
        raise SystemExit(f"{out}/visibility_final.pt exists")
    torch.manual_seed(cfg.get("seed", 0))
    stream = NavStream(cfg["data"], seed=cfg.get("seed", 0))
    stream._build()
    model = NavModel(cfg["backbone"], cfg["bev_head"], cfg.get("gravity_head")).cuda()
    init = torch.load(cfg["init"], map_location="cpu", weights_only=False)
    model.bev_head.load_state_dict(init["bev_head"])
    model.gravity_head.load_state_dict(init["gravity_head"])
    del init
    model.requires_grad_(False).eval()
    gravity_hash = digest(model.gravity_head)
    model.bev_head.requires_grad_(True).train()
    model.bev_head.error_head.requires_grad_(False)
    if cfg.get("vis_init"):
        vi = torch.load(cfg["vis_init"], map_location="cpu", weights_only=False)
        if vi.get("task") != "observability":
            raise SystemExit(f"{cfg['vis_init']} is not an observability checkpoint")
        model.bev_head.load_state_dict(vi["visibility_head"])
        del vi
    else:
        torch.nn.init.zeros_(model.bev_head.out.weight)
        torch.nn.init.zeros_(model.bev_head.out.bias)
    params = [p for p in model.bev_head.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tc["lr"], weight_decay=tc["weight_decay"])
    loader = DataLoader(stream, batch_size=None, num_workers=cfg["data"].get("num_workers", 6), pin_memory=True,
                        prefetch_factor=cfg["data"].get("prefetch", 2), persistent_workers=True)
    it = iter(loader)
    receipt = {"task": "observability", "steps": steps, "train": tc, "init": cfg["init"],
               "vis_init": cfg.get("vis_init"), "backbone": cfg["backbone"], "frozen_gravity_hash": gravity_hash,
               "pools": {p.name: len(p.scenes) for p in stream.sampler.pools},
               "trainable_parameters": sum(p.numel() for p in params)}
    (out / "config.yaml").write_text(yaml.safe_dump(cfg))
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2))
    start, ck = time.time(), None
    with (out / "log.jsonl").open("w") as log:
        for step in range(1, steps + 1):
            batch = next(it)
            batch = {k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            known = batch["bev"] > 0
            target = batch["observed"].float()
            with torch.no_grad():
                bb = model.backbone(batch["images"])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    up_pred = model.gravity(bb).float()
                ls, up = head_inputs(bb, batch, up_pred, cfg.get("loss", {}), tc["inputs"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model.heads(bb, ls, up)
                loss = F.binary_cross_entropy_with_logits(logits.float()[known], target[known])
            if not torch.isfinite(loss):
                print(f"step {step}: non-finite loss, skipped", flush=True)
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(params, tc["clip"])
            opt.step()
            if step <= 2 or step % 20 == 0:
                pred = logits.detach() > 0
                tp = (pred & (target > 0) & known).sum().item()
                row = {"step": step, "loss": loss.item(), "grad": float(grad),
                       "precision": tp / max(1, (pred & known).sum().item()),
                       "recall": tp / max(1, target.sum().item()), "elapsed": time.time() - start}
                print(json.dumps(row), flush=True)
                log.write(json.dumps(row) + "\n")
                log.flush()
            if step % 500 == 0 or step == steps:
                ck = {"task": "observability", "visibility_head": model.bev_head.state_dict(), "config": cfg,
                      "step": step, "frozen_gravity_hash": gravity_hash}
                torch.save(ck, out / f"visibility_{step:06d}.pt")
    if digest(model.gravity_head) != gravity_hash:
        raise RuntimeError("the frozen gravity head changed")
    receipt.update(status="complete", elapsed=time.time() - start)
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2))
    torch.save(ck, out / "visibility_final.pt")


if __name__ == "__main__":
    main()
