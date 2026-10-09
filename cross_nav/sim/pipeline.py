"""Per-scene pipeline of the simulated CROSS Nav data and its job queue (shared folder, survives disconnects).

    # CPU stage (Blender scene, voxelised geometry, planning BEV, trajectories), any number of workers / hosts:
    python -m cross_nav.sim.pipeline worker --stage prep --scenes scenes.txt
    # GPU stage (Cycles renders into the scene cache, final mesh BEV with camera heights), one worker per GPU:
    CUDA_VISIBLE_DEVICES=3 python -m cross_nav.sim.pipeline worker --stage render --scenes scenes.txt --gpu 3
    # one scene, in-process (debugging):
    python -m cross_nav.sim.pipeline run --stage prep --scene 102344022

Environment: NAVSIM_ROOT (default /mnt/datasets-livsyn/jz/navsim: blend/, work/<scene>/, state/, logs/), HSSD_RAW
(downloaded HSSD, default /mnt/datasets-livsyn/jz/hssd/raw), NAV_CACHE (scene cache root, default
/mnt/datasets-livsyn/jz/cache), NAV_BEV (label root, default /mnt/datasets-livsyn/jz/nav/bev), BPY_PYTHON (python of
the bpy environment), SIMCHANGE_ROOT (SimChange repo copy), NAVSIM_TMP (local scratch for frames).
Claims: state/<stage>/<scene>/ is created atomically (mkdir); `done` / `fail` files inside mark the result.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("NAVSIM_ROOT", "/mnt/datasets-livsyn/jz/navsim"))
RAW = Path(os.environ.get("HSSD_RAW", "/mnt/datasets-livsyn/jz/hssd/raw"))
CACHE = Path(os.environ.get("NAV_CACHE", "/mnt/datasets-livsyn/jz/cache"))
BEV = Path(os.environ.get("NAV_BEV", "/mnt/datasets-livsyn/jz/nav/bev"))
BPY = os.environ.get("BPY_PYTHON", "/mnt/datasets-livsyn/jz/envs/bpy/bin/python")
SIMCHANGE = os.environ.get("SIMCHANGE_ROOT", "/mnt/datasets-livsyn/jz/code/SimChange")
DATASET = os.environ.get("NAV_DATASET", "hssd_nav")


def sh(cmd, log: Path, env=None, timeout=None):
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(map(str, cmd))}\n")
        f.flush()
        r = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, env=env, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[1] if len(cmd) > 1 else cmd[0]} failed (exit {r.returncode}), see {log}")


def child_env(**extra):
    env = dict(os.environ)
    code = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = code + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["SIMCHANGE_ROOT"] = SIMCHANGE
    env.setdefault("NUMBA_CACHE_DIR", "/data0/jz/numba_cache")
    env.update({k: str(v) for k, v in extra.items()})
    return env


def layout_regions(blend: Path):
    p = Path(str(blend) + ".layout.json")
    return json.loads(p.read_text()).get("regions", []) if p.exists() else []


# ---------------------------------------------------------------------------------------------------------- stages
def stage_prep(scene: str, log: Path, n_traj=None, seed=None):
    from cross_nav.sim.mesh_bev import build_mesh_bev, preview
    from cross_nav.sim.trajgen import make_trajectories
    blend = ROOT / "blend" / f"{scene}.blend"
    work = ROOT / "work" / scene
    work.mkdir(parents=True, exist_ok=True)
    blend.parent.mkdir(parents=True, exist_ok=True)
    if not blend.exists():
        tmp = blend.with_name(f"{scene}.tmp.blend")
        sh([BPY, f"{SIMCHANGE}/simchange/hssd_to_blend.py", "--root", RAW, "--scene", scene, "--out", tmp], log,
           env=child_env(), timeout=4 * 3600)
        os.replace(str(tmp) + ".layout.json", str(blend) + ".layout.json")
        os.replace(tmp, blend)
    geom_p = work / "geom.npz"
    if not geom_p.exists():
        sh([BPY, "-m", "cross_nav.sim.blend_geom", blend, work / "geom.tmp.npz"], log, env=child_env(), timeout=3 * 3600)
        os.replace(work / "geom.tmp.npz", geom_p)
    geom = dict(np.load(geom_p))
    regions = layout_regions(blend)
    out, stats, aux = build_mesh_bev(geom, regions=regions)
    free_m2 = stats["free"] * float(out["res"]) ** 2
    if n_traj is None:
        n_traj = 2 if free_m2 < 40 else (3 if free_m2 < 120 else 4)
    if seed is None:                                             # deterministic per scene
        import zlib
        seed = zlib.crc32(scene.encode()) & 0x7FFFFFFF
    trajs = make_trajectories(out, aux, n_traj, seed)
    if not trajs:
        raise RuntimeError(f"no trajectory (free area {free_m2:.0f} m2)")
    azim = float(np.random.default_rng(seed + 1).uniform(0, 360))
    meta = {"scene": scene, "source": "hssd", "seed": seed, "plan_stats": stats, "frame": [list(aux["frame"][0]), aux["frame"][1], aux["frame"][2]],
            "trajs": [{"name": t["name"], "params": t["params"], "preset": t["preset"], "K": t["K"].tolist(),
                       "width": t["width"], "height": t["height"], "azimuth": (azim + 47.0 * i) % 360,
                       "n": len(t["c2w"]), "length_m": t["length_m"]} for i, t in enumerate(trajs)]}
    arrs = {f"c2w/{t['name']}": t["c2w"] for t in trajs}
    arrs.update({f"h/{t['name']}": t["h"] for t in trajs})
    np.savez(work / "traj.tmp.npz", meta=json.dumps(meta), **arrs)
    os.replace(work / "traj.tmp.npz", work / "traj.npz")
    preview(out, work / "plan_bev.png", [t["c2w"][:, :2, 3] for t in trajs])
    return {"plan": stats, "free_m2": round(free_m2, 1), "trajs": [(t["name"], len(t["c2w"])) for t in trajs]}


def final_label(scene: str, dataset: str = DATASET, cache: Path = CACHE, bev_root: Path = BEV):
    """Mesh BEV seeded at the rendered cameras, with per-frame camera heights; written next to the accumulated labels."""
    from cross_nav.sim.mesh_bev import build_mesh_bev, camera_heights, preview
    from vggt_ft.dataio.scene import load_scene
    work = ROOT / "work" / scene
    z = np.load(work / "traj.npz")
    meta = json.loads(str(z["meta"]))
    geom = dict(np.load(work / "geom.npz"))
    sc = load_scene(cache / dataset / scene)
    seeds, hs, Es = [], [], {}
    for q in sc.sequences:
        sq = sc.seq(q)
        Es[q] = sq.E
        h = z[f"h/{q}"]
        C = sq.centres
        seeds.append(C)
        hs.append(h[: len(C)] if len(h) >= len(C) else np.full(len(C), np.nan))
    seeds, hs = np.concatenate(seeds), np.concatenate(hs)
    ok = np.isfinite(hs)
    fr = meta["frame"]
    out, stats, aux = build_mesh_bev(geom, seeds_xyz=seeds[ok], seeds_h=hs[ok], regions=layout_regions(ROOT / "blend" / f"{scene}.blend"),
                                     frame=(tuple(fr[0]), fr[1], fr[2]))
    for q, E in Es.items():
        out[f"cam_h/{scene}/{q}"] = camera_heights(E, aux["g_int"], out["origin"], float(out["res"]))
    allh = np.concatenate([out[f"cam_h/{scene}/{q}"] for q in Es])
    stats.update(frames=int(len(allh)), cam_h_ok=float(np.isfinite(allh).mean()),
                 cam_h_med=float(np.nanmedian(allh)) if np.isfinite(allh).any() else None, source="mesh")
    out["stats"] = json.dumps(stats)
    dst = bev_root / dataset / f"{scene}.npz"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".tmp.npz")
    np.savez_compressed(tmp, **out)
    os.replace(tmp, dst)
    preview(out, dst.with_suffix(".png"), [(-np.einsum("nji,nj->ni", E[:, :3, :3], E[:, :3, 3]))[:, :2] for E in Es.values()])
    return stats


def stage_render(scene: str, log: Path, samples: int = 32):
    work = ROOT / "work" / scene
    blend = ROOT / "blend" / f"{scene}.blend"
    tmp = os.environ.get("NAVSIM_TMP", "/data0/jz/navsim/tmp")
    Path(tmp).mkdir(parents=True, exist_ok=True)
    sh([BPY, "-m", "cross_nav.sim.render_seq", "--blend", blend, "--traj", work / "traj.npz", "--cache", CACHE,
        "--dataset", DATASET, "--scene", scene, "--samples", samples, "--tmp", tmp, "--info", work / "render.json"],
       log, env=child_env(), timeout=6 * 3600)
    return final_label(scene)


# ---------------------------------------------------------------------------------------------------------- queue
def gpu_free_mib(gpu: int) -> int:
    try:
        r = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=60)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:                                           # noqa: BLE001
        return 0


def worker(a):
    scenes = [l.split()[0] for l in open(a.scenes) if l.strip() and not l.startswith("#")]
    st = ROOT / "state" / a.stage
    st.mkdir(parents=True, exist_ok=True)
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    host = socket.gethostname()
    idle = 0
    while True:
        did = False
        for s in scenes:
            if a.stage == "render":
                if not (ROOT / "state" / "prep" / s / "done").exists():
                    continue
                while a.gpu is not None and gpu_free_mib(a.gpu) < a.min_free_mib:
                    print(f"{time.strftime('%H:%M:%S')} GPU {a.gpu}: < {a.min_free_mib} MiB free, waiting", flush=True)
                    time.sleep(120)
            d = st / s
            try:
                d.mkdir()
            except FileExistsError:
                continue
            (d / "claim").write_text(f"{host} pid {os.getpid()} gpu {a.gpu} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log = logs / f"{a.stage}_{s}.log"
            t0 = time.time()
            print(f"{time.strftime('%H:%M:%S')} {a.stage} {s} start", flush=True)
            try:
                res = stage_prep(s, log) if a.stage == "prep" else stage_render(s, log, a.samples)
                res["sec"] = round(time.time() - t0)
                (d / "done").write_text(json.dumps(res) + "\n")
                print(f"{time.strftime('%H:%M:%S')} {a.stage} {s} done {json.dumps(res)}", flush=True)
            except Exception as e:                              # noqa: BLE001
                (d / "fail").write_text(f"{e!r}\n{traceback.format_exc()}")
                print(f"{time.strftime('%H:%M:%S')} {a.stage} {s} FAILED {e!r}", flush=True)
            did = True
            if a.max_jobs and a.max_jobs <= 1:
                return
            if a.max_jobs:
                a.max_jobs -= 1
        if not did:
            idle += 1
            if a.stage == "prep" or idle > a.idle_rounds:
                print("no more work", flush=True)
                return
            time.sleep(120)
        else:
            idle = 0


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("worker")
    p.add_argument("--stage", choices=["prep", "render"], required=True)
    p.add_argument("--scenes", required=True)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--min-free-mib", type=int, default=16000)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--max-jobs", type=int, default=0)
    p.add_argument("--idle-rounds", type=int, default=90, help="render workers wait this many 2-min rounds for prep")
    p = sp.add_parser("run")
    p.add_argument("--stage", choices=["prep", "render", "label"], required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--samples", type=int, default=32)
    a = ap.parse_args()
    if a.cmd == "worker":
        worker(a)
    else:
        log = ROOT / "logs" / f"{a.stage}_{a.scene}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        r = {"prep": lambda: stage_prep(a.scene, log), "render": lambda: stage_render(a.scene, log, a.samples),
             "label": lambda: final_label(a.scene)}[a.stage]()
        print(json.dumps(r, default=str))


if __name__ == "__main__":
    main()
