#!/usr/bin/env python3
"""System-level test: build a map on one stereo sequence, then relocalize on another.

Phase 1 (mapping): run CROSS on the map sequence with simulated noisy odometry, save the
map and the ground-truth pose of every permanent keyframe.
Phase 2 (relocalization): load the map, run CROSS on the query sequence (typically the
same trajectory under a different condition, or a different trajectory of the same
scene) and record, for every processed frame, the tracked pose of component 0 and of
the best hypothesis.  The map frame is aligned to ground truth with the permanent
keyframes of the map (SE(3) Umeyama), so query errors are measured in metres/degrees.

Usage:
  python scripts/map_and_reloc.py --map data/vkitti2/Scene01/clone --query data/vkitti2/Scene01/sunset \
      --estimator ff --out outputs/reloc/vk01_sunset_ff
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))
# one log file per run (cross.core.system reads CROSS_LOG_FILE at import time)
if "--out" in sys.argv and "CROSS_LOG_FILE" not in os.environ:
    _out = Path(sys.argv[sys.argv.index("--out") + 1])
    _out.mkdir(parents=True, exist_ok=True)
    os.environ["CROSS_LOG_FILE"] = str(_out / "system.log")

from cross.core.config import FFBackend, PoseEstType, SystemConfig, load_config
from cross.core.system import System
from cross.core.types import Camera
from cross.cv.stereo_scale import invert_poses, rotation_angle_deg
from cross.dataloader.stereo_loader import StereoSequenceLoader
from cross.pipeline import FAST_STEREO_PRESET, add_session_args, session_factory


def umeyama_se3(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Rigid transform T with dst ≈ T @ src (points as (N,3))."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    H = (src - mu_s).T @ (dst - mu_d)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = mu_d - R @ mu_s
    return T


def make_config(args) -> SystemConfig:
    configs = list(args.config or [])
    if getattr(args, "fast", False) and args.estimator == "ff":
        configs.append(FAST_STEREO_PRESET)                   # explicit --max-refs / --n-ref-anchors still win
    cfg = load_config(*configs) if configs else SystemConfig()
    cfg.async_update = False
    cfg.mapping.loop_closure.intra_enabled = not getattr(args, "no_intra_lc", False)
    lc = cfg.mapping.loop_closure
    lc.mode = getattr(args, "lc_mode", None) or lc.mode
    if getattr(args, "lc_confidence", None) is not None:
        lc.confidence = args.lc_confidence
    if getattr(args, "noise_config", None):
        lc.noise_file = args.noise_config
    if args.estimator == "ff":
        cfg.pose_est.type = PoseEstType.FF
        ff = cfg.pose_est.ff
        ff.backend = FFBackend(args.backend)
        ff.checkpoint = args.checkpoint or (
            "models/VGGT-Omega/vggt_omega_1b_512.pt" if args.backend == "vggt_omega" else "models/DA3-LARGE-1.1")
        if args.max_refs is not None:
            ff.max_refs = args.max_refs
        if args.n_ref_anchors is not None:
            ff.n_ref_anchors = args.n_ref_anchors
        ff.use_curr_anchor = not args.no_curr_anchor
        ff.use_odom_anchor = args.odom_anchor
        cfg.pose_est.obs_min_translation = args.obs_min_translation   # observation gating (generic option, set for the stereo estimator as before)
        cfg.pose_est.obs_min_rotation = args.obs_min_rotation
        cfg.pose_est.obs_max_interval_steps = args.obs_max_interval
        ff.scale_method = args.scale_method
        if getattr(args, "ff_meas_std", None):
            ff.base_measurement_std = list(args.ff_meas_std)
    else:
        cfg.pose_est.type = PoseEstType.PNP
    cfg.retrieval.top_k = args.top_k
    for kv in getattr(args, "set", None) or []:      # generic overrides: section.sub.key=value (YAML-parsed value)
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


def odom_kwargs(args) -> dict:
    """Systematic odometry error of the simulated odometry (cross/dataloader/dataloader.py): scale bias, heading drift."""
    return {"odom_scale_bias": args.odom_scale_bias, "odom_yaw_drift_deg_per_m": args.odom_yaw_drift,
            "odom_file": getattr(args, "odom_file", None)}


def pose_to_mat(p) -> np.ndarray:
    from cross.utils.lie_tensor import normalize_SE3
    return normalize_SE3(p).matrix().detach().cpu().numpy().astype(np.float64)


def depth_source(args) -> str:
    """Depth the loader computes: PnP depth (RGB-D*), none for the stereo / mono modes, stereo depth (SGBM) as the
    metric scale of visual odometry in the stereo mode."""
    if args.mode == "mono":
        return "none"
    if args.estimator == "pnp":
        return args.pnp_depth
    if args.odometry == "visual" and args.depth_source == "none":
        return "sgbm"
    return args.depth_source


def new_session(args, ds, seed):
    """A CROSS session (cross/pipeline.py) for --mode / --odometry; with external odometry in the stereo / RGB-D*
    modes it passes every frame to System.step unchanged."""
    camera = Camera(K=ds.rgb_K.copy(), frame_width=ds.rgb_width, frame_height=ds.rgb_height)
    return session_factory(args, camera, make_config(args), T_right_in_left=ds.T_right_in_left,
                           seed=0 if seed is None else seed)()


def step_time_stats(dts) -> dict:
    """Per-frame processing time: whether the system keeps up with the input rate (10 Hz: 0.1 s per frame)."""
    dts = np.asarray(dts, dtype=float)
    if len(dts) == 0:
        return {}
    return {"step_time_median_s": float(np.median(dts)), "step_time_mean_s": float(np.mean(dts)),
            "step_time_p95_s": float(np.percentile(dts, 95)), "step_time_max_s": float(np.max(dts)),
            "step_frac_over_100ms": float(np.mean(dts > 0.1))}


def run_mapping(args, out: Path):
    ds = StereoSequenceLoader(args.map, depth_source=depth_source(args),
                              snr=args.snr, baseline=args.baseline, seed=args.seed, **odom_kwargs(args))
    system = new_session(args, ds, args.seed)
    recorder = None
    if getattr(args, "dump_obs", False):          # every observation of the mapping run (offline back-end studies)
        from lc.obs_recorder import ObsRecorder
        recorder = ObsRecorder(system.mapper, out / "obs.jsonl")
    kf_gt = {}
    t0 = time.time()
    n = 0
    last_kf = None
    step_times = []
    for idx, d in enumerate(ds.replay_data(start_idx=args.map_start, end_idx=args.map_end, stride=args.stride)):
        if idx == 0:
            d["delta_pose"] = None
        ts = time.perf_counter()
        system.process(d)
        step_times.append(time.perf_counter() - ts)
        n += 1
        if system.last_added_kf_id != last_kf and system.last_added_kf_id is not None:
            last_kf = system.last_added_kf_id
            kf_gt[int(last_kf)] = d["world_pose"].tolist()
    elapsed = time.time() - t0
    n_perm = len([k for k in system.hypothesis_manager.nodes.values() if not k.temporary])
    logger.info(f"Mapping done: {n} frames in {elapsed:.1f}s ({n / elapsed:.2f} FPS), "
                f"{len(system.hypothesis_manager.nodes)} keyframes ({n_perm} permanent)")
    map_file = out / "map.pkl"
    system.save_map(map_file)
    if recorder is not None:
        recorder.close()
    # map -> GT alignment from permanent keyframes (component 0 mean)
    ids, src, dst = [], [], []
    kf_est = {}
    for kid, kf in system.hypothesis_manager.nodes.items():
        if kf.temporary or kid not in kf_gt:
            continue
        ids.append(kid)
        M = pose_to_mat(kf.pose_mu[0])
        kf_est[int(kid)] = M.reshape(-1).tolist()
        src.append(M[:3, 3])
        dst.append(np.asarray(kf_gt[kid])[:3, 3])
    src, dst = np.asarray(src), np.asarray(dst)
    T_gt_from_map = umeyama_se3(src, dst)
    map_ate = float(np.sqrt(np.mean(np.sum(((T_gt_from_map[:3, :3] @ src.T).T + T_gt_from_map[:3, 3] - dst) ** 2, 1))))
    meta = {
        "kf_gt": kf_gt, "kf_est": kf_est, "T_gt_from_map": T_gt_from_map.tolist(), "map_ate_rmse": map_ate,
        "n_frames": n, "elapsed": elapsed, **step_time_stats(step_times), "n_keyframes": len(system.hypothesis_manager.nodes), "n_permanent": n_perm,
        "timing": _timing_summary(),
        "map_file_bytes": map_file.stat().st_size,
    }
    (out / "map_meta.json").write_text(json.dumps(meta, indent=1))
    logger.info(f"Map ATE vs GT (permanent kfs): {map_ate:.3f} m")
    system.release()          # shuts down and releases the GPU models (`atexit` keeps a reference to the system)
    del system
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"GPU memory after mapping cleanup: {torch.cuda.memory_allocated() / 2**30:.2f} GB")
    return meta


def _timing_summary():
    from cross.utils.profile import timing_registry
    out = {}
    for name, rec in list(timing_registry.items()):
        n = int(rec.get("call_count", 0))
        if n > 0:
            out[name] = {"n": n, "total_s": float(rec["total_time"]), "mean_ms": float(rec["total_time"]) / n * 1e3}
    return out


def run_reloc(args, out: Path, meta: dict):
    ds = StereoSequenceLoader(args.query, depth_source=depth_source(args),
                              snr=args.snr, baseline=args.baseline, seed=None if args.seed is None else args.seed + 1,
                              **odom_kwargs(args))
    system = new_session(args, ds, args.seed)
    system.load_map(out / "map.pkl")
    n_map_kfs = len(system.hypothesis_manager.nodes)
    if "kf_est" not in meta:   # maps saved before the map-relative metric: recover keyframe estimates from the loaded map
        meta["kf_est"] = {str(kid): pose_to_mat(kf.pose_mu[0]).reshape(-1).tolist()
                          for kid, kf in system.hypothesis_manager.nodes.items() if not kf.temporary and str(kid) in meta["kf_gt"]}
        (out / "map_meta.json").write_text(json.dumps(meta, indent=1))
    T_gt_from_map = np.asarray(meta["T_gt_from_map"])
    rows = []
    t0 = time.time()
    n_obs = 0
    from reloc_metrics import build_trials
    q_start = args.query_start
    q_end = args.query_end or len(ds)
    trials = build_trials(q_end - q_start, args.trial_len, args.trial_stride)
    for ti, (ts_, te_) in enumerate(trials):
        if ti > 0:                                    # every trial is an independent relocalization session
            try:
                system.load_map(out / "map.pkl")
            except RuntimeError:                      # the mono mode loads a map only into a fresh session
                if hasattr(system, "shutdown"):
                    system.shutdown()
                system = new_session(args, ds, args.seed)
                system.load_map(out / "map.pkl")
        for idx, d in enumerate(ds.replay_data(start_idx=q_start + ts_, end_idx=q_start + te_, stride=args.stride)):
            if idx == 0:
                d["delta_pose"] = None
            ts = time.perf_counter()
            system.process(d)
            dt = time.perf_counter() - ts
            T_c0, T_best, w = system.belief(pose_to_mat)
            gt = np.asarray(d["world_pose"])
            row = {"frame": int(d["frame_idx"]), "step": idx, "trial": ti, "dt": dt, "w0": float(w[0]),
                   "observed": system.mapped_now and not getattr(system.mapper, "_steps_since_obs", 0)}
            n_obs += int(row["observed"])
            row["gt_pose"] = gt.reshape(-1).tolist()
            for name, k, T_map in (("c0", 0, T_c0), ("best", int(np.argmax(w)), T_best)):
                row[f"{name}_pose"] = T_map.reshape(-1).tolist()
                T = T_gt_from_map @ T_map
                err = invert_poses(gt) @ T
                row[f"{name}_t_err"] = float(np.linalg.norm(err[:3, 3]))
                row[f"{name}_r_err"] = float(rotation_angle_deg(err[:3, :3]))
                row[f"{name}_w"] = float(w[k])
            row["best_k"] = int(np.argmax(w))
            row["n_active"] = int((w > 1e-3).sum())
            rows.append(row)
            if idx % 50 == 0:
                logger.info(f"trial {ti} step {idx}: c0 err {row['c0_t_err']:.2f} m / {row['c0_r_err']:.1f} deg, "
                            f"best(k={row['best_k']}) {row['best_t_err']:.2f} m, w0={row['w0']:.2f}")
    elapsed = time.time() - t0
    n_new = len(system.hypothesis_manager.nodes) - n_map_kfs
    system.release()

    from reloc_metrics import map_relative_errors, summarize_errors, summarize_trials
    rel = map_relative_errors(rows, meta)          # errors w.r.t. the map (primary metric)
    for r, e in zip(rows, rel):
        r.update(e)
    trial_summary = summarize_trials(rows, "c0_rel", r_d=args.r_d)
    trial_summary_best = summarize_trials(rows, "best_rel", r_d=args.r_d)
    c0 = np.array([[r["c0_t_err"], r["c0_r_err"]] for r in rows])
    best = np.array([[r["best_t_err"], r["best_r_err"]] for r in rows])

    def recall(e, t, r):
        return float(np.mean((e[:, 0] < t) & (e[:, 1] < r)))

    def first_correct(e, t, r, hold=5):
        ok = (e[:, 0] < t) & (e[:, 1] < r)
        for i in range(len(ok) - hold + 1):
            if ok[i:i + hold].all():
                return int(i)
        return None

    def converged_error(e, t, r, hold=5):
        i = first_correct(e, t, r, hold)
        if i is None:
            return None
        return float(np.median(e[i:, 0])), float(np.median(e[i:, 1]))

    summary = {
        "n_frames": len(rows), "elapsed": elapsed, "fps": len(rows) / elapsed, "n_observations": n_obs,
        "n_new_keyframes": n_new,
        "c0_recall_1m_5deg": recall(c0, 1.0, 5.0), "c0_recall_2m_10deg": recall(c0, 2.0, 10.0),
        "c0_recall_0.5m_5deg": recall(c0, 0.5, 5.0),
        "best_recall_1m_5deg": recall(best, 1.0, 5.0), "best_recall_2m_10deg": recall(best, 2.0, 10.0),
        "best_recall_0.5m_5deg": recall(best, 0.5, 5.0),
        "c0_first_correct_step_1m": first_correct(c0, 1.0, 5.0), "best_first_correct_step_1m": first_correct(best, 1.0, 5.0),
        "c0_first_correct_step_2m": first_correct(c0, 2.0, 10.0), "best_first_correct_step_2m": first_correct(best, 2.0, 10.0),
        "c0_median_after_converge_2m": converged_error(c0, 2.0, 10.0),
        "c0_t_err_median": float(np.median(c0[:, 0])), "c0_r_err_median": float(np.median(c0[:, 1])),
        "best_t_err_median": float(np.median(best[:, 0])), "best_r_err_median": float(np.median(best[:, 1])),
        "trials": trial_summary, "trials_best": trial_summary_best, "trial_len": args.trial_len,
        "RS": trial_summary["RS"], "RS_1m_5deg": trial_summary["RS_1m_5deg"], "RS_0.5m_5deg": trial_summary["RS_0.5m_5deg"],
        "map_relative": summarize_errors(rows, "c0_rel"),
        "map_relative_best": summarize_errors(rows, "best_rel"),
        **step_time_stats([r["dt"] for r in rows]),
        "timing": _timing_summary(),
    }
    (out / "reloc_rows.json").write_text(json.dumps(rows))
    (out / "reloc_summary.json").write_text(json.dumps(summary, indent=1))
    logger.info(json.dumps({k: v for k, v in summary.items() if k != "timing"}, indent=1))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--estimator", choices=["ff", "pnp"], default="ff")
    ap.add_argument("--backend", choices=["vggt_omega", "da3"], default="vggt_omega")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--config", nargs="*", default=[])
    ap.add_argument("--depth-source", default="none", help="depth for keyframes when using ff (none|sgbm|gt)")
    ap.add_argument("--pnp-depth", default="sgbm", help="depth source for the PnP baseline (sgbm|gt)")
    ap.add_argument("--baseline", type=float, default=None, help="stereo baseline to use for SimChange sequences with several rendered right cameras")
    ap.add_argument("--trial-len", type=int, default=0, help="split the query into independent relocalization trials of this many frames (0: one trial)")
    ap.add_argument("--trial-stride", type=int, default=None, help="start offset between trials (default: trial length)")
    ap.add_argument("--r-d", type=float, default=2.0, help="distance threshold of the relocalization-success metric (CROSS paper: 2 m indoor, 5 m outdoor)")
    ap.add_argument("--snr", type=float, default=None, help="odometry noise SNR (None = perfect odometry)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--map-start", type=int, default=0)
    ap.add_argument("--map-end", type=int, default=None)
    ap.add_argument("--query-start", type=int, default=0)
    ap.add_argument("--query-end", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--max-refs", type=int, default=None, help="references per forward pass (default: config, 6)")
    ap.add_argument("--n-ref-anchors", type=int, default=None, help="reference right images per pass (default: config, 2)")
    ap.add_argument("--odom-anchor", action="store_true")
    ap.add_argument("--no-curr-anchor", action="store_true", help="ablation: drop the current stereo pair as scale anchor")
    ap.add_argument("--scale-method", default="adaptive")
    ap.add_argument("--obs-min-translation", type=float, default=0.0)
    ap.add_argument("--obs-min-rotation", type=float, default=0.0)
    ap.add_argument("--obs-max-interval", type=int, default=1)
    ap.add_argument("--skip-map", action="store_true", help="reuse map.pkl / map_meta.json in --out")
    ap.add_argument("--skip-reloc", action="store_true")
    ap.add_argument("--no-intra-lc", action="store_true", help="ablation: disable the intra-hypothesis loop closure (PGO of hypothesis 0)")
    ap.add_argument("--seed", type=int, default=None, help="seed of the odometry noise (reproducible runs)")
    add_session_args(ap)
    ap.add_argument("--lc-mode", choices=["verified", "heuristic"], default=None, help="loop-closure mode (default: config, 'verified')")
    ap.add_argument("--lc-confidence", type=float, default=None, help="chi-square confidence of the verified loop closure (default 0.999)")
    ap.add_argument("--noise-config", default=None, help="YAML from scripts/lc/calibrate_noise.py (calibrated noise model)")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides section.sub.key=value (YAML-parsed values)")
    ap.add_argument("--dump-obs", action="store_true", help="write every observation of the mapping run to obs.jsonl (scripts/lc/obs_recorder.py)")
    ap.add_argument("--odom-scale-bias", type=float, default=0.0, help="systematic odometry scale error (e.g. 0.02 = 2 %%)")
    ap.add_argument("--odom-yaw-drift", type=float, default=0.0, help="systematic heading drift of the odometry (deg per metre)")
    ap.add_argument("--odom-file", default=None, help="odometry file of the prepared folders to use instead of "
                    "odom_left.txt when present (e.g. odom_vio.txt from benchmark/datasets/prepare_vio.py)")
    ap.add_argument("--ff-meas-std", type=float, nargs=6, default=None, help="base measurement std [tx ty tz rx ry rz] of the FF estimator")
    args = ap.parse_args()
    if args.snr is not None and args.snr <= 0:
        args.snr = None        # perfect odometry
    if args.mode is None:
        args.mode = "stereo" if args.estimator == "ff" else "rgbd"      # rgbd: RGB-D* (PnP on stereo depth)
    elif args.mode == "stereo" and args.estimator != "ff" or args.mode == "rgbd" and args.estimator != "pnp":
        ap.error("--mode stereo goes with --estimator ff, --mode rgbd with --estimator pnp")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    if args.skip_map:
        meta = json.loads((out / "map_meta.json").read_text())
    else:
        meta = run_mapping(args, out)
    if not args.skip_reloc:
        run_reloc(args, out, meta)


if __name__ == "__main__":
    main()
