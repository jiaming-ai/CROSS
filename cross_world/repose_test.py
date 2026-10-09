"""Does the reconstruction follow a changed map without retraining?

The map's keyframe poses are bent smoothly, as a later loop closure or merged session bends a graph: a rotation about
the vertical that grows linearly along the trajectory up to `--deg` degrees at its end (about the first keyframe),
which moves the far end of the map by metres but its neighbouring keyframes by about the same amount.  The held-out
keyframes are evaluated at their bent poses with

    stale      the reconstruction as trained (it no longer matches the map)
    rigid      each chunk moved rigidly with the correction of its central keyframe
    anchored   every Gaussian moved with the blend of its four anchor keyframes' corrections (World.repose)

and the unbent reconstruction at the original poses as the reference.

    python -m cross_world.repose_test --world worlds/x/world.pt --map runs/x/map.pkl [--source data/seq] --deg 10
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from cross_world.map_views import View, load_map_views
from cross_world.world import World, evaluate


def bend(mv, deg: float):
    """{keyframe id: bent pose} and the bend of every keyframe."""
    vs = sorted(mv.views, key=lambda v: (v.timestamp if v.timestamp is not None else v.id))
    c = np.array([v.center for v in vs])
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
    s = s / max(s[-1], 1e-9)
    up = mv.up()
    p0 = c[0]
    out = {}
    for v, si in zip(vs, s):
        a = np.radians(deg * si)
        K = np.array([[0, -up[2], up[1]], [up[2], 0, -up[0]], [-up[1], up[0], 0]])
        R = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
        D = np.eye(4)
        D[:3, :3] = R
        D[:3, 3] = p0 - R @ p0
        out[v.id] = D @ v.T_wc
    return out


def rigid_chunks(world: World, new_poses: dict) -> World:
    """Each chunk moved by the correction of the core keyframe nearest its centre (one anchor per chunk)."""
    w = copy.deepcopy(world)
    part = w.partition
    for c in w.chunks:
        core = part.chunks[c.index].core_ids
        cen = np.mean([w.kf_poses[k][:3, 3] for k in core], axis=0)
        k0 = min(core, key=lambda k: np.linalg.norm(w.kf_poses[k][:3, 3] - cen))
        for l in c.layers:
            n = len(c.layers[l]["means"])
            c.anchor_ids[l] = torch.full((n, 1), int(k0), dtype=torch.int64)
            c.anchor_w[l] = torch.ones(n, 1)
    w.repose(new_poses)
    return w


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--source", default=None)
    ap.add_argument("--max-side", type=int, default=None)
    ap.add_argument("--deg", type=float, default=10.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    world = World.load(args.world)
    mv = load_map_views(args.map, args.source, max_side=args.max_side)
    by = mv.by_id()
    test = [by[i] for i in world.meta["test_ids"] if i in by]
    kf_times = {v.id: (v.timestamp if v.timestamp is not None else float(v.id)) for v in mv.views}
    new = bend(mv, args.deg)
    moved = np.array([np.linalg.norm(new[k][:3, 3] - by[k].T_wc[:3, 3]) for k in new])
    bent_test = [View(id=v.id, T_wc=new[v.id], K=v.K, width=v.width, height=v.height, timestamp=v.timestamp,
                      source_index=v.source_index, _image=v._image, _depth=v._depth) for v in test]
    res = {"deg": args.deg, "keyframe_shift_m": {"median": float(np.median(moved)), "max": float(moved.max())}}
    print(f"bend {args.deg} deg: keyframes move {np.median(moved):.2f} m median, {moved.max():.2f} m max")
    res["reference"] = evaluate(world, test, args.device, align=True, kf_times=kf_times)["summary"]
    res["stale"] = evaluate(world, bent_test, args.device, align=True, kf_times=kf_times)["summary"]
    res["rigid"] = evaluate(rigid_chunks(world, new), bent_test, args.device, align=True, kf_times=kf_times)["summary"]
    w = copy.deepcopy(world)
    info = w.repose(new)
    res["anchored"] = evaluate(w, bent_test, args.device, align=True, kf_times=kf_times)["summary"]
    res["anchored"]["max_move_m"] = info["max_move_m"]
    out = Path(args.out or Path(args.world).parent / f"repose_{args.deg:g}deg.json")
    out.write_text(json.dumps(res, indent=1))
    for k in ("reference", "stale", "rigid", "anchored"):
        s = res[k]
        print(f"{k:9s}  as posed PSNR {s['psnr_raw']:.2f} SSIM {s['ssim_raw']:.3f}   aligned PSNR {s['psnr_aligned']:.2f} "
              f"SSIM {s['ssim_aligned']:.3f} LPIPS {s.get('lpips_aligned', float('nan')):.3f}")


if __name__ == "__main__":
    main()
