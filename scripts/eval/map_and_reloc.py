#!/usr/bin/env python3
"""Mapping and relocalization benchmark on posed RGB-D sequences (cross/dataloader/posed_rgbd.py).

Phase 1 (mapping): run CROSS on the map sequence with simulated noisy odometry, save the map and the ground-truth pose
of every permanent keyframe; the map frame is aligned to ground truth with those keyframes (SE(3) Umeyama) and the
keyframe ATE is reported.
Phase 2 (relocalization): the query sequence (the same place under a change: lighting, rearranged objects, another
path) is split into independent trials (CROSS protocol: 100-frame trials; a new system loads the stored map for every
trial and starts without knowing its pose).  A trial succeeds when the final estimate of hypothesis 0 is within r_D of
the pose the map implies for the query frame (map-relative error, scripts/eval/reloc_metrics.py).

Usage:
  python scripts/eval/map_and_reloc.py --map data/sim/hssd_house/map --query data/sim/hssd_house/light_night \
      --out outputs/hssd_house/light_night --snr 10 --seed 0 --trial-len 100 --trial-stride 50
"""

from __future__ import annotations

import argparse
import atexit
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
# one log file per run (cross.core.system may read CROSS_LOG_FILE at import time)
if "--out" in sys.argv and "CROSS_LOG_FILE" not in os.environ:
    _out = Path(sys.argv[sys.argv.index("--out") + 1])
    _out.mkdir(parents=True, exist_ok=True)
    os.environ["CROSS_LOG_FILE"] = str(_out / "system.log")

import torch  # noqa: E402
from loguru import logger  # noqa: E402

from cross.core.config import SystemConfig, load_config  # noqa: E402
from cross.core.system import System  # noqa: E402
from cross.core.types import Camera  # noqa: E402
from cross.dataloader.posed_rgbd import PosedRGBDLoader  # noqa: E402
from reloc_metrics import build_trials, map_relative_errors, summarize_errors, summarize_trials  # noqa: E402


def umeyama_se3(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Rigid transform T with dst ~= T @ src (points as (N, 3))."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    H = (src - mu_s).T @ (dst - mu_d)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = mu_d - R @ mu_s
    return T


def invert(T: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def rotation_angle_deg(R: np.ndarray) -> float:
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64))
    R = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def pose_to_mat(p) -> np.ndarray:
    """pypose SE3 (tx ty tz qx qy qz qw) -> 4x4, quaternion normalised (float32 compositions shrink |q|)."""
    from scipy.spatial.transform import Rotation
    v = p.tensor().detach().cpu().double().numpy().reshape(-1)[:7]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(v[3:7] / np.linalg.norm(v[3:7])).as_matrix()
    T[:3, 3] = v[:3]
    return T


def make_config(args) -> SystemConfig:
    cfg = load_config(*args.config) if args.config else SystemConfig()
    cfg.async_update = False
    cfg.retrieval.top_k = args.top_k
    for kv in args.set or []:            # generic overrides: section.sub.key=value (YAML-parsed value)
        import yaml
        key, val = kv.split("=", 1)
        obj = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        cur = getattr(obj, parts[-1])
        v = yaml.safe_load(val)
        if hasattr(cur, "value") and not isinstance(v, type(cur)):   # enums
            v = type(cur)(v)
        setattr(obj, parts[-1], v)
    return cfg


def make_loader(path, args, seed):
    return PosedRGBDLoader(path, snr=args.snr, seed=seed, odom_scale_bias=args.odom_scale_bias,
                           odom_yaw_drift_deg_per_m=args.odom_yaw_drift)


def new_system(args, ds) -> System:
    camera = Camera(K=ds.rgb_K.copy(), frame_width=ds.rgb_width, frame_height=ds.rgb_height)
    return System(visualize=False, debug=False, camera=camera, config=make_config(args))


def release(system) -> None:
    system.shutdown()
    try:                     # System registers its shutdown with atexit, which would keep every system alive
        atexit.unregister(system.shutdown)
    except Exception:
        pass
    for name in ("pose_est", "hypothesis_manager"):
        if hasattr(system, name):
            setattr(system, name, None)
    if getattr(system, "db", None) is not None:
        system.db.vpr_model = None
        system.db = None
    gc.collect()
    torch.cuda.empty_cache()


def _timing_summary():
    from cross.utils.profile import timing_registry
    out = {}
    for name, rec in list(timing_registry.items()):
        n = int(rec.get("call_count", 0))
        if n > 0:
            out[name] = {"n": n, "total_s": float(rec["total_time"]), "mean_ms": float(rec["total_time"]) / n * 1e3}
    return out


def run_mapping(args, out: Path) -> dict:
    ds = make_loader(args.map, args, args.seed)
    system = new_system(args, ds)
    kf_gt, n, last_kf = {}, 0, None
    t0 = time.time()
    for idx, d in enumerate(ds.replay_data(start_idx=args.map_start, end_idx=args.map_end, stride=args.stride)):
        if idx == 0:
            d["delta_pose"] = None
        system.step(obs=d, data=d)
        n += 1
        if system.last_added_kf_id is not None and system.last_added_kf_id != last_kf:
            last_kf = system.last_added_kf_id
            kf_gt[int(last_kf)] = d["world_pose"].tolist()
    elapsed = time.time() - t0
    nodes = system.hypothesis_manager.nodes
    n_perm = len([k for k in nodes.values() if not k.temporary])
    logger.info(f"Mapping done: {n} frames in {elapsed:.1f}s ({n / elapsed:.2f} FPS), {len(nodes)} keyframes ({n_perm} permanent)")
    map_file = out / "map.pkl"
    system.save_map(str(map_file))
    ids, src, dst, kf_est = [], [], [], {}
    for kid, kf in nodes.items():
        if kf.temporary or kid not in kf_gt:
            continue
        M = pose_to_mat(kf.pose_mu[0])
        kf_est[int(kid)] = M.reshape(-1).tolist()
        src.append(M[:3, 3])
        dst.append(np.asarray(kf_gt[kid])[:3, 3])
    src, dst = np.asarray(src), np.asarray(dst)
    T = umeyama_se3(src, dst)
    ate = float(np.sqrt(np.mean(np.sum(((T[:3, :3] @ src.T).T + T[:3, 3] - dst) ** 2, 1))))
    meta = {"kf_gt": kf_gt, "kf_est": kf_est, "T_gt_from_map": T.tolist(), "map_ate_rmse": ate, "n_frames": n,
            "elapsed": elapsed, "n_keyframes": len(nodes), "n_permanent": n_perm, "timing": _timing_summary(),
            "map_file_bytes": map_file.stat().st_size}
    (out / "map_meta.json").write_text(json.dumps(meta, indent=1))
    logger.info(f"Map ATE vs GT (permanent kfs): {ate:.3f} m, map file {meta['map_file_bytes'] / 2**20:.1f} MB")
    release(system)
    return meta


def run_reloc(args, out: Path, meta: dict) -> dict:
    ds = make_loader(args.query, args, None if args.seed is None else args.seed + 1)
    T_gt_from_map = np.asarray(meta["T_gt_from_map"])
    q_start, q_end = args.query_start, args.query_end or len(ds)
    trials = build_trials(q_end - q_start, args.trial_len, args.trial_stride)
    rows, n_obs = [], 0
    t0 = time.time()
    t_steps = 0.0
    for ti, (ts_, te_) in enumerate(trials):
        system = new_system(args, ds)          # every trial is an independent relocalization session
        system.load_map(str(out / "map.pkl"))
        for idx, d in enumerate(ds.replay_data(start_idx=q_start + ts_, end_idx=q_start + te_, stride=args.stride)):
            if idx == 0:
                d["delta_pose"] = None
            ts = time.perf_counter()
            system.step(obs=d, data=d)
            dt = time.perf_counter() - ts
            t_steps += dt
            mu, sigma, w = system.hypothesis_manager.dist
            w = w.detach().cpu().numpy()
            gt = np.asarray(d["world_pose"])
            row = {"frame": int(d["frame_idx"]), "step": idx, "trial": ti, "dt": dt, "w0": float(w[0]),
                   "gt_pose": gt.reshape(-1).tolist()}
            for name, k in (("c0", 0), ("best", int(np.argmax(w)))):
                T_map = pose_to_mat(mu[k])
                row[f"{name}_pose"] = T_map.reshape(-1).tolist()
                err = invert(gt) @ (T_gt_from_map @ T_map)
                row[f"{name}_t_err"] = float(np.linalg.norm(err[:3, 3]))
                row[f"{name}_r_err"] = rotation_angle_deg(err[:3, :3])
            row["best_k"] = int(np.argmax(w))
            rows.append(row)
        logger.info(f"trial {ti}/{len(trials)}: final c0 err {rows[-1]['c0_t_err']:.2f} m / {rows[-1]['c0_r_err']:.1f} deg")
        release(system)
    elapsed = time.time() - t0
    rel = map_relative_errors(rows, meta)
    for r, e in zip(rows, rel):
        r.update(e)
    ts_c0 = summarize_trials(rows, "c0_rel", r_d=args.r_d)
    ts_best = summarize_trials(rows, "best_rel", r_d=args.r_d)
    fail = [t["final_t_err"] for t in ts_c0["trials"] if not t["success_rd"]]
    summary = {
        "n_frames": len(rows), "n_trials": len(trials), "elapsed": elapsed, "fps_steps": len(rows) / max(t_steps, 1e-9),
        "RS": ts_c0["RS"], "RS_1m_5deg": ts_c0["RS_1m_5deg"], "RS_0.5m_5deg": ts_c0["RS_0.5m_5deg"],
        "RS_best": ts_best["RS"], "failure_final_err_median": float(np.median(fail)) if fail else None,
        "trials": ts_c0, "trials_best": ts_best, "trial_len": args.trial_len,
        "map_relative": summarize_errors(rows, "c0_rel"),
        "step_time_median_s": float(np.median([r["dt"] for r in rows])),
        "step_time_mean_s": float(np.mean([r["dt"] for r in rows])),
        "timing": _timing_summary(),
    }
    (out / "reloc_rows.json").write_text(json.dumps(rows))
    (out / "reloc_summary.json").write_text(json.dumps(summary, indent=1))
    logger.info(f"RS {summary['RS']:.3f} (1 m / 5 deg {summary['RS_1m_5deg']:.3f}) over {len(trials)} trials, "
                f"{summary['fps_steps']:.1f} steps/s")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True, help="posed RGB-D folder of the mapping sequence")
    ap.add_argument("--query", required=True, help="posed RGB-D folder of the query sequence")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", nargs="*", default=[])
    ap.add_argument("--set", nargs="*", default=[], help="config overrides section.sub.key=value (YAML-parsed values)")
    ap.add_argument("--snr", type=float, default=10.0, help="odometry noise SNR (<= 0: perfect odometry)")
    ap.add_argument("--odom-scale-bias", type=float, default=0.0, help="systematic odometry scale error (0.02 = 2 %%)")
    ap.add_argument("--odom-yaw-drift", type=float, default=0.0, help="systematic heading drift (deg per metre)")
    ap.add_argument("--seed", type=int, default=0, help="seed of the odometry noise (query sessions use seed + 1)")
    ap.add_argument("--trial-len", type=int, default=100)
    ap.add_argument("--trial-stride", type=int, default=50)
    ap.add_argument("--r-d", type=float, default=2.0, help="relocalization-success radius (CROSS: 2 m indoor, 5 m outdoor)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--map-start", type=int, default=0)
    ap.add_argument("--map-end", type=int, default=None)
    ap.add_argument("--query-start", type=int, default=0)
    ap.add_argument("--query-end", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--skip-map", action="store_true", help="reuse map.pkl / map_meta.json in --out")
    ap.add_argument("--skip-reloc", action="store_true")
    args = ap.parse_args()
    if args.snr is not None and args.snr <= 0:
        args.snr = None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    meta = json.loads((out / "map_meta.json").read_text()) if args.skip_map else run_mapping(args, out)
    if not args.skip_reloc:
        run_reloc(args, out, meta)


if __name__ == "__main__":
    main()
