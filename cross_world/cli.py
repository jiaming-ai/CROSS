"""Command line of CROSS World.

    python -m cross_world.cli build  --map runs/x/map.pkl [--source data/seq] --out worlds/x
    python -m cross_world.cli eval   --world worlds/x/world.pt --map runs/x/map.pkl [--source data/seq] [--novel 200]
    python -m cross_world.cli export --world worlds/x/world.pt --out worlds/x/web [--sh 1]
    python -m cross_world.cli repose --world worlds/x/world.pt --map runs/x_after_pgo/map.pkl --out worlds/x/world_reposed.pt

`--set train.steps_per_view=60 max_views=150 ...` overrides fields of BuildConfig / TrainConfig.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
import yaml


def _set(cfg, items):
    from cross_world.world import BuildConfig
    for it in items or []:
        key, val = it.split("=", 1)
        val = yaml.safe_load(val)
        obj = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        if not hasattr(obj, parts[-1]):
            raise KeyError(f"unknown setting {key}")
        setattr(obj, parts[-1], val)
    return cfg


def _commit() -> str:
    root = Path(__file__).resolve().parents[1]
    c = root / "COMMIT"
    if c.exists():
        return c.read_text().strip()
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _views(args):
    from cross_world.map_views import load_map_views
    return load_map_views(args.map, args.source, max_side=args.max_side)


def cmd_build(args):
    from cross_world.world import BuildConfig, build_world, evaluate, split_views
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "build.log", "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()
    cfg = _set(BuildConfig(), args.set)
    if args.config:
        for k, v in (yaml.safe_load(Path(args.config).read_text()) or {}).items():
            _set(cfg, [f"{k}={json.dumps(v)}"] if not isinstance(v, dict) else [f"{k}.{a}={json.dumps(b)}" for a, b in v.items()])
    log(f"cross_world build {_commit()} map {args.map} source {args.source} cfg {cfg}")
    mv = _views(args)
    only = [int(x) for x in args.chunks.split(",")] if args.chunks else None
    extra = None
    if args.capture:
        from cross_world.map_views import load_capture_views, source_test_views
        # evaluation frames (held-out keyframes, novel source frames) never train, also not as captured frames
        _, test = split_views(mv, cfg.test_every)
        ex = [v.timestamp for v in test if v.timestamp is not None]
        if args.novel and mv.source is not None:
            ex += [v.timestamp for v in source_test_views(mv, min_gap=args.novel_gap, max_side=args.max_side,
                                                          limit=args.novel) if v.timestamp is not None]
        extra = load_capture_views(mv, max_side=args.max_side, exclude_times=np.array(ex))
        log(f"captured frames: {len(extra)} training views (evaluation frames excluded)")
    world = build_world(mv, cfg, device=args.device, log=log, only_chunks=only, extra_views=extra)
    world.meta.update({"commit": _commit(), "source": args.source, "max_side": args.max_side})
    world.save(out / "world.pt")
    log(f"saved {out / 'world.pt'}: {world.count()} Gaussians, build {world.meta['build_s']} s")
    res = {"build": {"build_s": world.meta["build_s"], "depth_s": world.meta["depth_s"], "count": world.count(),
                     "chunks": [c.stats for c in world.chunks]}}
    if not args.no_eval:
        res.update(run_eval(world, mv, args, out, log))
    (out / "metrics.json").write_text(json.dumps(res, indent=1, default=float))


def run_eval(world, mv, args, out, log):
    from cross_world.map_views import source_test_views
    from cross_world.world import evaluate
    res = {}
    by = mv.by_id()
    kf_times = {v.id: (v.timestamp if v.timestamp is not None else float(v.id)) for v in mv.views}
    test = [by[i] for i in world.meta["test_ids"] if i in by]
    if test:
        log(f"evaluating {len(test)} held-out keyframes")
        res["heldout_keyframes"] = evaluate(world, test, args.device, align=True, save_dir=out / "renders" / "heldout",
                                            log=log, kf_times=kf_times)
    train = [by[i] for i in world.meta["train_ids"] if i in by]
    if train and args.eval_train:
        sel = train[:: max(1, len(train) // args.eval_train)]
        res["train_keyframes"] = evaluate(world, sel, args.device, align=False, save_dir=out / "renders" / "train",
                                          log=log, kf_times=kf_times)
    if args.novel and mv.source is not None:
        nv = source_test_views(mv, min_gap=args.novel_gap, max_side=args.max_side, limit=args.novel)
        if nv:
            log(f"evaluating {len(nv)} novel source frames (>= {args.novel_gap} frames from any keyframe)")
            res["novel_frames"] = evaluate(world, nv, args.device, align=True, save_dir=out / "renders" / "novel",
                                           log=log, kf_times=kf_times)
    return res


def cmd_eval(args):
    from cross_world.world import World
    out = Path(args.out or Path(args.world).parent)
    world = World.load(args.world)
    mv = _views(args)
    res = run_eval(world, mv, args, out, print)
    (out / "metrics_eval.json").write_text(json.dumps(res, indent=1, default=float))


def cmd_export(args):
    from cross_world.export import export_world
    from cross_world.world import World
    world = World.load(args.world)
    mv = _views(args) if args.map else None
    wdir = Path(args.world).parent
    metrics = json.loads((wdir / "metrics.json").read_text()) if (wdir / "metrics.json").exists() else None
    if metrics:                                   # summaries only (the per-view rows stay in metrics.json)
        metrics = {k: (v.get("summary", v) if isinstance(v, dict) else v) for k, v in metrics.items()}
    export_world(world, Path(args.out), sh_degree=args.sh, mv=mv, thumbs=args.thumbs, max_file_mb=args.max_file_mb,
                 ply=args.ply, metrics=metrics, renders_dir=wdir / "renders")


def cmd_clean(args):
    from cross_world.world import World, clean_world
    world = World.load(args.world)
    mv = _views(args)
    clean_world(world, mv.views, min_views=args.min_views, needle_ratio=args.needle_ratio, device=args.device)
    world.save(args.out)
    if args.eval:
        res = run_eval(world, mv, args, Path(args.out).parent / "eval_clean", print)
        (Path(args.out).parent / "metrics_clean.json").write_text(json.dumps(res, indent=1, default=float))


def cmd_repose(args):
    from cross_world.map_views import load_map_views
    from cross_world.world import World
    world = World.load(args.world)
    mv = load_map_views(args.map, args.source, max_side=args.max_side)
    info = world.repose({v.id: v.T_wc for v in mv.views})
    print(f"re-posed: {info}")
    world.save(args.out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cross_world")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--map", help="saved CROSS map (map.pkl)")
        p.add_argument("--source", default=None, help="prepared sequence folder of the mapping session (full-resolution frames)")
        p.add_argument("--max-side", type=int, default=None, help="downscale source frames to this longer side")
        p.add_argument("--device", default="cuda")
        p.add_argument("--novel", type=int, default=0, help="evaluate on up to N source frames that are not keyframes")
        p.add_argument("--novel-gap", type=int, default=2, help="... at least this many frames from any keyframe")
        p.add_argument("--eval-train", type=int, default=0, help="also evaluate ~N training keyframes")
    b = sub.add_parser("build")
    common(b)
    b.add_argument("--out", required=True)
    b.add_argument("--config", default=None, help="YAML of BuildConfig fields (train: {...} for TrainConfig)")
    b.add_argument("--set", nargs="*", default=[])
    b.add_argument("--chunks", default=None, help="train only these chunks (comma separated)")
    b.add_argument("--no-eval", action="store_true")
    b.add_argument("--capture", action="store_true", help="also train on the map's captured frames (mapping.world_capture)")
    b.set_defaults(fn=cmd_build)
    e = sub.add_parser("eval")
    common(e)
    e.add_argument("--world", required=True)
    e.add_argument("--out", default=None)
    e.set_defaults(fn=cmd_eval)
    x = sub.add_parser("export")
    common(x)
    x.add_argument("--world", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--sh", type=int, default=1, help="SH degree of the exported splats")
    x.add_argument("--thumbs", type=int, default=640, help="keyframe thumbnail width in the viewer data (0: none)")
    x.add_argument("--max-file-mb", type=float, default=14.0)
    x.add_argument("--ply", action="store_true", help="also write standard 3DGS PLY files per chunk")
    x.set_defaults(fn=cmd_export)
    cl = sub.add_parser("clean", help="remove Gaussians no training view constrains (see world.clean_world)")
    common(cl)
    cl.add_argument("--world", required=True)
    cl.add_argument("--out", required=True)
    cl.add_argument("--min-views", type=int, default=2)
    cl.add_argument("--needle-ratio", type=float, default=30.0)
    cl.add_argument("--eval", action="store_true")
    cl.set_defaults(fn=cmd_clean)
    r = sub.add_parser("repose")
    common(r)
    r.add_argument("--world", required=True)
    r.add_argument("--out", required=True)
    r.set_defaults(fn=cmd_repose)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
