#!/usr/bin/env python3
"""Record the construction of a CROSS-stereo map and subsequent relocalization sessions.

The recorder attaches non-invasive hooks to a `System` instance and writes, for every processed
frame, the belief (all GMM components), the retrieval / observation result, new keyframes and
edges, loop-closure events and the keyframe poses changed by pose-graph optimisation.  The result
(`trace.json` + `frames/`) is the single input of `build_trace_page.py` (interactive web page) and
`render_trace_video.py` (MP4).

Session 0 is the mapping run (map saved to `map.pkl`).  Every further session reloads the map and
runs one query sequence continuously (one relocalization session per scene variant).

Usage (parameters mirror scripts/run_sim_experiments.sh):
  python scripts/viz/record_trace.py --scene lonemonk --map map_loop \
      --variants light_night reverse offset_2.0 yaw_45 --out outputs/viz/lonemonk/trace \
      --baseline 0.3 --snr 10 --obs-min-translation 0.3 --obs-min-rotation 0.15 --obs-max-interval 3
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
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
if "--out" in sys.argv and "CROSS_LOG_FILE" not in os.environ:
    _out = Path(sys.argv[sys.argv.index("--out") + 1])
    _out.mkdir(parents=True, exist_ok=True)
    os.environ["CROSS_LOG_FILE"] = str(_out / "system.log")

from cross.core.config import FFBackend, PoseEstType, SystemConfig, load_config  # noqa: E402
from cross.core.system import System  # noqa: E402
from cross.core.types import Camera, EdgeType  # noqa: E402
from cross.dataloader.stereo_loader import StereoSequenceLoader  # noqa: E402
from map_and_reloc import umeyama_se3  # noqa: E402
from lc.graph_io import dump_graph  # noqa: E402


def _r(x, nd=4):
    return [round(float(v), nd) for v in np.asarray(x).reshape(-1)]


def _pose7(lie) -> list:
    """pypose SE3 -> [x, y, z, qx, qy, qz, qw] (unit quaternion)."""
    v = lie.tensor().detach().cpu().numpy().astype(np.float64).reshape(-1)
    v[3:7] /= max(np.linalg.norm(v[3:7]), 1e-12)
    return _r(v)


def _mat_to_pose7(T: np.ndarray) -> list:
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(T[:3, :3]).as_quat()  # x, y, z, w
    return _r(np.concatenate([T[:3, 3], q]))


def _pose7_to_mat(p) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    T = np.eye(4); q = np.asarray(p[3:7], dtype=np.float64); q /= max(np.linalg.norm(q), 1e-12)
    T[:3, :3] = Rotation.from_quat(q).as_matrix(); T[:3, 3] = p[:3]
    return T


class TraceRecorder:
    """Hooks into a System and accumulates the per-step trace."""

    def __init__(self, out: Path, img_width: int = 320, jpeg_quality: int = 82, write_frames: bool = True):
        self.out = out
        self.write_frames = write_frames
        self.frames_dir = out / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.img_width = img_width
        self.jpeg_quality = jpeg_quality
        self.sessions = []
        self.nodes = {}
        self.edges = []
        self.steps = []
        self._session = None
        self._step_events = None
        self._node_snapshot = {}
        self._hyp_snapshot = None
        self.system = None
        self._last_vk = {}
        self.kf_gt_session = {}      # keyframe id -> ground-truth c2w of the keyframes created in the current session

    # ------------------------------------------------------------------ hooks
    def attach(self, system: System):
        self.system = system
        hm = system.hypothesis_manager
        rec = self

        orig_obs = system._construct_observation_dist

        def obs_hook(*a, **k):
            ret = orig_obs(*a, **k)
            vk0 = ret.get("valid_keyframes", []); conf0 = ret.get("pose_est_conf", []); rw0 = ret.get("valid_retrieval_weights", [])
            rec._last_vk = {int(kf.id): (float(conf0[i]), float(rw0[i])) for i, kf in enumerate(vk0)}
            ev = rec._step_events
            if ev is not None:
                ev["obs"] = True
                # feed-forward pass: metric camera-to-world poses of the current view (index 0) and of every retrieved
                # reference in the same pass (in-pass consistency of the references against the map)
                info = getattr(getattr(system, "pose_est", None), "last_info", None) or {}
                refs = ret.get("retrieved_keyframes", [])
                if info.get("valid") and "c2w_metric" in info and refs:
                    c2w = np.asarray(info["c2w_metric"])[:1 + len(refs)]
                    ev["ffp"] = {"ids": [int(kf.id) for kf in refs], "c2w": [_mat_to_pose7(T) for T in c2w],
                                 "valid": [bool(v) for v in info.get("valid_masks", [])], "covis": [round(float(c), 3) for c in info.get("covis", [])]}
                vk = ret.get("valid_keyframes", [])
                conf = ret.get("pose_est_conf", [])
                rw = ret.get("valid_retrieval_weights", [])
                ev["vk"] = [[int(kf.id), round(float(conf[i]), 3), round(float(rw[i]), 3)] for i, kf in enumerate(vk)]
                lr = system._last_retrieved_results
                if lr is not None:
                    ev["ret"] = [[int(kf.id), round(float(sc), 3)] for sc, kf in zip(lr["scores"], lr["keyframes"])]
            return ret

        system._construct_observation_dist = obs_hook

        orig_add_node = hm.add_node

        def add_node_hook(keyframe):
            r = orig_add_node(keyframe)
            ev = rec._step_events
            sid = rec._session["id"]
            key = int(keyframe.id) if sid == 0 else f"{sid}:{int(keyframe.id)}"   # keyframe ids restart after every map load
            rec.nodes[key] = {
                "s": rec._session["id"], "step": len(rec.steps), "f": ev["f"] if ev else None,
                "perm": not keyframe.temporary, "p": _pose7(keyframe.pose_mu[0]),
            }
            if ev is not None:
                ev["nk"] = int(keyframe.id)
                ev["nk_perm"] = not keyframe.temporary
            return r

        hm.add_node = add_node_hook

        orig_add_edge = hm.add_edge

        def add_edge_hook(id1, id2, rel_pose_mean, rel_pose_std, type, from_comp_id=0, to_comp_id=0, meta=None):
            r = orig_add_edge(id1, id2, rel_pose_mean, rel_pose_std, type, from_comp_id, to_comp_id, meta=meta)
            if id1 in hm.nodes and id2 in hm.nodes:
                t = {EdgeType.ODOMETRY: "odom", EdgeType.VISUAL: "visual"}.get(type, str(type))
                if t == "visual" and to_comp_id not in hm.hypotheses:
                    return r
                e = {"a": int(id1), "b": int(id2), "t": t, "s": rec._session["id"], "step": len(rec.steps),
                     "fc": int(from_comp_id), "tc": int(to_comp_id)}
                if t == "visual":
                    e["rel"] = _r(rel_pose_mean.tensor().detach().cpu().numpy()[:3], 3)
                    e["rel7"] = _pose7(rel_pose_mean)
                    e["sd"] = _r(rel_pose_std.tensor().detach().cpu().numpy(), 4)
                    c = rec._last_vk.get(int(id1))
                    if c is not None:
                        e["c"], e["rw"] = round(c[0], 3), round(c[1], 3)
                    facs = hm.hypotheses[to_comp_id].visual_edges.get((id1, id2), [])
                    if facs:
                        fac = facs[-1]
                        fac.conf, fac.rw = (c if c is not None else (None, None))
                        fac.step, fac.session = len(rec.steps), rec._session["id"]
                self.edges.append(e)
                if rec._step_events is not None:
                    rec._step_events.setdefault("ne", []).append(len(self.edges) - 1)
            return r

        hm.add_edge = add_edge_hook

        orig_detect = hm.detect_loop_closure

        def detect_hook(ret):
            r = orig_detect(ret)
            if r.get("loop_closure") and rec._step_events is not None:
                rec._step_events["lc"] = int(r["loop_closure_hypo_id"])
            return r

        hm.detect_loop_closure = detect_hook

        if hasattr(system, "_maybe_intra_hypothesis_loop_closure"):
            orig_intra = system._maybe_intra_hypothesis_loop_closure

            def intra_hook(*a, **k):
                r = orig_intra(*a, **k)
                if r and rec._step_events is not None:
                    rec._step_events["lc"] = 0    # 0 = loop closure inside hypothesis 0 (no merge)
                return r

            system._maybe_intra_hypothesis_loop_closure = intra_hook

        orig_apply = hm.apply_pgo_result

        def apply_hook(pgo_result):
            other = int(pgo_result.get("other_hypothesis_id", 0))
            lc_edges = []
            if other != 0 and other in hm.hypotheses:
                lc_edges = [[int(a), int(b)] for (a, b) in hm.hypotheses[other].visual_edges.keys()]
            r = orig_apply(pgo_result)
            if rec._step_events is not None:
                ev = rec._step_events
                ev["pgo"] = {"hypo": other, "n_opt": len(pgo_result.get("optimized_poses", {})),
                             "cost": float(r.get("cost", float("nan"))) if r.get("cost") is not None else None,
                             "lc_edges": lc_edges}
            return r

        hm.apply_pgo_result = apply_hook

    # ------------------------------------------------------------------ sessions / steps
    def begin_session(self, name: str, kind: str, variant: str, seq_dir: str, n_frames: int, first_gt: np.ndarray):
        sid = len(self.sessions)
        self._session = {"id": sid, "name": name, "kind": kind, "variant": variant, "seq": seq_dir,
                         "n_frames": int(n_frames), "start_step": len(self.steps), "n_steps": 0,
                         "first_gt": _mat_to_pose7(first_gt)}
        self.sessions.append(self._session)
        (self.frames_dir / f"s{sid}").mkdir(exist_ok=True)
        self._node_snapshot = {}
        self._hyp_snapshot = None
        self.kf_gt_session = {}
        logger.info(f"[trace] session {sid}: {name}")

    def before_step(self, d: dict):
        self._step_events = {"s": self._session["id"], "i": self._session["n_steps"], "f": int(d["frame_idx"]),
                             "obs": False}

    def after_step(self, d: dict, dt: float):
        ev = self._step_events
        sys_ = self.system
        hm = sys_.hypothesis_manager
        mu, sigma, w = hm.dist
        ev["gt"] = _mat_to_pose7(np.asarray(d["world_pose"], dtype=np.float64))
        if ev.get("nk") is not None:
            self.kf_gt_session[int(ev["nk"])] = np.asarray(d["world_pose"], dtype=np.float64)
        ev["mu"] = [_pose7(mu[k]) for k in range(mu.shape[0])]
        ev["sd"] = [round(float(v), 3) for v in sigma.tensor()[:, :3].norm(dim=1).detach().cpu().tolist()]
        ev["w"] = _r(w.detach().cpu().numpy(), 4)
        ev["rl"] = [bool(v) for v in hm.realized.detach().cpu().tolist()]
        ev["dt"] = round(float(dt), 4)
        hyp = {int(c): int(h.start_idx) for c, h in hm.hypotheses.items()}
        if hyp != self._hyp_snapshot:
            ev["hyp"] = hyp
            self._hyp_snapshot = hyp
        # keyframe poses changed by PGO / local smoothing, and temporary keyframes removed
        cur = {}
        with hm.graph_lock:
            ids = list(hm.nodes.keys())
            if ids:
                P = torch.stack([hm.nodes[i].pose_mu[0].tensor() for i in ids]).detach().cpu().numpy()
                for i, p in zip(ids, P):
                    cur[int(i)] = p
        upd = {}
        for i, p in cur.items():
            q = self._node_snapshot.get(i)
            if q is None or np.abs(q - p).max() > 5e-4:
                upd[i] = _r(p)   # includes the keyframe created this step (PGO in the same step may have moved it)
        removed = [i for i in self._node_snapshot if i not in cur]
        if upd:
            ev["upd"] = upd
        if removed:
            ev["rm"] = removed
            for i in removed:
                key = i if (ev["s"] == 0 or i in self.nodes and self.nodes[i]["s"] == 0) else f"{ev['s']}:{i}"
                if key in self.nodes:
                    self.nodes[key]["rm"] = len(self.steps)
        self._node_snapshot = cur
        # observation image
        if self.write_frames:
            rgb = d["rgb"]
            h = int(round(rgb.shape[0] * self.img_width / rgb.shape[1]))
            im = Image.fromarray(np.asarray(rgb)).resize((self.img_width, h), Image.BILINEAR)
            rel = f"frames/s{ev['s']}/{ev['i']:05d}.jpg"
            im.save(self.out / rel, quality=self.jpeg_quality)
            ev["img"] = rel
        self.steps.append(ev)
        self._session["n_steps"] += 1
        self._step_events = None

    def snapshot_nodes(self, tag: str):
        """Final keyframe poses of the current graph (used for the mapping session)."""
        hm = self.system.hypothesis_manager
        return {int(i): _pose7(kf.pose_mu[0]) for i, kf in hm.nodes.items()}

    def save(self, extra: dict):
        data = {"sessions": self.sessions, "nodes": self.nodes, "edges": self.edges, "steps": self.steps}
        data.update(extra)
        (self.out / "trace.json").write_text(json.dumps(data, separators=(",", ":")))
        logger.info(f"[trace] saved {len(self.steps)} steps, {len(self.nodes)} nodes, {len(self.edges)} edges -> {self.out / 'trace.json'}")


# ---------------------------------------------------------------------- run
def make_config(args) -> SystemConfig:
    cfg = load_config(*args.config) if getattr(args, "config", None) else SystemConfig()
    cfg.async_update = False
    cfg.mapping.loop_closure.intra_enabled = not getattr(args, "no_intra_lc", False)
    lc = cfg.mapping.loop_closure
    lc.mode = getattr(args, "lc_mode", None) or lc.mode
    if getattr(args, "lc_confidence", None) is not None:
        lc.confidence = args.lc_confidence
    if getattr(args, "noise_config", None):
        lc.noise_file = args.noise_config
    for name in ("inpass", "prior", "posterior"):
        if getattr(args, f"no_{name}", False):
            setattr(lc, f"use_{name}", False)
    if getattr(args, "recent_window", None) is not None:
        cfg.retrieval.recent_window_steps = args.recent_window
    if getattr(args, "legacy_lc", False):
        h = cfg.mapping.hypothesis
        h.detect_min_frames, h.detect_min_weight, h.verify_max_outlier_frac = 0, 0.0, 1.01
    if args.estimator == "pnp":
        cfg.pose_est.type = PoseEstType.PNP
        cfg.retrieval.top_k = args.top_k
        return cfg
    cfg.pose_est.type = PoseEstType.FF
    ff = cfg.pose_est.ff
    ff.backend = FFBackend(args.backend)
    ff.checkpoint = args.checkpoint or ("models/VGGT-Omega/vggt_omega_1b_512.pt" if args.backend == "vggt_omega" else "models/DA3-LARGE-1.1")
    ff.max_refs = args.max_refs
    ff.n_ref_anchors = args.n_ref_anchors
    cfg.pose_est.obs_min_translation = args.obs_min_translation   # observation gating (generic option, set for the stereo estimator as before)
    cfg.pose_est.obs_min_rotation = args.obs_min_rotation
    cfg.pose_est.obs_max_interval_steps = args.obs_max_interval
    ff.scale_method = args.scale_method
    if getattr(args, "ff_meas_std", None):
        ff.base_measurement_std = list(args.ff_meas_std)
    cfg.retrieval.top_k = args.top_k
    return cfg


def new_system(args, ds):
    camera = Camera(K=ds.rgb_K.copy(), frame_width=ds.rgb_width, frame_height=ds.rgb_height)
    return System(visualize=False, debug=False, camera=camera, config=make_config(args), T_right_in_left=ds.T_right_in_left)


def release(system):
    system.shutdown()
    system.pose_est = None
    system.db.vpr_model = None
    system.db = None
    system.hypothesis_manager = None
    del system
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="lonemonk")
    ap.add_argument("--data-root", default="data/sim")
    ap.add_argument("--map", default="map_loop")
    ap.add_argument("--variants", nargs="*", default=["light_night", "reverse", "offset_2.0", "yaw_45"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--estimator", choices=["ff", "pnp"], default="ff")
    ap.add_argument("--pnp-depth", default="gt", help="depth source of the PnP baseline (gt|sgbm)")
    ap.add_argument("--method", default=None, help="method name stored in the trace (default from estimator)")
    ap.add_argument("--no-frames", action="store_true", help="do not write JPEG frames (the page builder takes them from the dataset)")
    ap.add_argument("--backend", default="vggt_omega")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--baseline", type=float, default=0.3)
    ap.add_argument("--snr", type=float, default=10.0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--map-end", type=int, default=None)
    ap.add_argument("--query-start", type=int, default=0)
    ap.add_argument("--query-end", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--max-refs", type=int, default=6)
    ap.add_argument("--n-ref-anchors", type=int, default=2)
    ap.add_argument("--scale-method", default="adaptive")
    ap.add_argument("--obs-min-translation", type=float, default=0.3)
    ap.add_argument("--obs-min-rotation", type=float, default=0.15)
    ap.add_argument("--obs-max-interval", type=int, default=3)
    ap.add_argument("--img-width", type=int, default=320)
    ap.add_argument("--skip-map", action="store_true", help="reuse map.pkl and trace_map.json in --out")
    ap.add_argument("--no-intra-lc", action="store_true", help="ablation: disable the intra-hypothesis loop closure (PGO of hypothesis 0)")
    ap.add_argument("--ff-meas-std", type=float, nargs=6, default=None, help="base measurement std [tx ty tz rx ry rz] of the FF estimator")
    ap.add_argument("--seed", type=int, default=None, help="seed of the odometry noise (reproducible runs)")
    ap.add_argument("--recent-window", type=int, default=None, help="RetrievalConfig.recent_window_steps override (0 = split off)")
    ap.add_argument("--legacy-lc", action="store_true", help="ablation: old loop-closure test (no evidence validity / weight gate / verification)")
    ap.add_argument("--config", nargs="*", default=[], help="YAML config file(s) applied before the command-line overrides (e.g. configs/outdoor.yaml)")
    ap.add_argument("--lc-mode", choices=["verified", "heuristic"], default=None, help="loop-closure mode (default: config, 'verified')")
    ap.add_argument("--lc-confidence", type=float, default=None, help="chi-square confidence of the verified loop closure (default 0.999)")
    ap.add_argument("--noise-config", default=None, help="YAML from scripts/lc/calibrate_noise.py (calibrated noise model)")
    ap.add_argument("--no-inpass", action="store_true", help="ablation: disable the in-pass consistency test")
    ap.add_argument("--no-prior", action="store_true", help="ablation: disable the prior consistency test")
    ap.add_argument("--no-posterior", action="store_true", help="ablation: disable the posterior consistency test")
    args = ap.parse_args()
    if args.snr is not None and args.snr <= 0:
        args.snr = None        # perfect odometry

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    root = Path(args.data_root) / args.scene
    rec = TraceRecorder(out, img_width=args.img_width, write_frames=not args.no_frames)
    map_dir = root / args.map
    depth_source = args.pnp_depth if args.estimator == "pnp" else "none"
    method = args.method or ({"ff": "cross_stereo", "pnp": "cross_pnp_gt" if args.pnp_depth == "gt" else "cross_pnp_sgbm"}[args.estimator])

    # ---------------- session 0: mapping
    map_file = out / "map.pkl"
    map_trace = out / "trace_map.json"
    if args.skip_map and map_file.exists() and map_trace.exists():
        saved = json.loads(map_trace.read_text())
        rec.sessions, rec.nodes, rec.edges, rec.steps = saved["sessions"], {int(k): v for k, v in saved["nodes"].items()}, saved["edges"], saved["steps"]
        extra = saved["extra"]
        logger.info("[trace] reusing mapping session from trace_map.json")
    else:
        ds = StereoSequenceLoader(str(map_dir), depth_source=depth_source, snr=args.snr, baseline=args.baseline, seed=args.seed)
        system = new_system(args, ds)
        rec.attach(system)
        n_frames = len(ds) if args.map_end is None else min(len(ds), args.map_end)
        rec.begin_session(f"mapping ({args.map})", "map", args.map, str(map_dir), n_frames, ds.left_c2w[0])
        kf_gt = {}
        last_kf = None
        t0 = time.time()
        for idx, d in enumerate(ds.replay_data(start_idx=0, end_idx=args.map_end, stride=args.stride)):
            if idx == 0:
                d["delta_pose"] = None
            rec.before_step(d)
            ts = time.perf_counter()
            system.step(obs=d, data=d)
            rec.after_step(d, time.perf_counter() - ts)
            if system.last_added_kf_id != last_kf and system.last_added_kf_id is not None:
                last_kf = system.last_added_kf_id
                kf_gt[int(last_kf)] = d["world_pose"].tolist()
            if idx % 100 == 0:
                logger.info(f"[trace] mapping step {idx}/{n_frames} ({(idx + 1) / (time.time() - t0):.1f} FPS), "
                            f"{len(system.hypothesis_manager.nodes)} keyframes")
        if getattr(system, "_lc_verifier", None) is not None:
            rec.sessions[-1]["lc_stats"] = dict(system._lc_verifier.stats)
        system.save_map(str(map_file))
        dump_graph(system, out / "graph_s0.json", {int(k): np.asarray(v) for k, v in kf_gt.items()}, 0,
                   meta={"scene": args.scene, "method": method, "variant": args.map, "seed": args.seed, "snr": args.snr, "baseline": args.baseline})
        # alignments map -> GT: (a) first keyframe, (b) Umeyama over permanent keyframes (final poses)
        final_nodes = rec.snapshot_nodes("map_final")
        src, dst = [], []
        for kid, kf in system.hypothesis_manager.nodes.items():
            if kf.temporary or int(kid) not in kf_gt:
                continue
            src.append(kf.pose_mu[0].matrix().detach().cpu().numpy()[:3, 3])
            dst.append(np.asarray(kf_gt[int(kid)])[:3, 3])
        src, dst = np.asarray(src), np.asarray(dst)
        T_umeyama = umeyama_se3(src, dst)
        ate = float(np.sqrt(np.mean(np.sum(((T_umeyama[:3, :3] @ src.T).T + T_umeyama[:3, 3] - dst) ** 2, 1))))
        kf0 = min(system.hypothesis_manager.nodes.keys())
        M0 = system.hypothesis_manager.nodes[kf0].pose_mu[0].matrix().detach().cpu().numpy().astype(np.float64)
        G0 = np.asarray(kf_gt[int(kf0)], dtype=np.float64)
        T_first = G0 @ np.linalg.inv(M0)
        extra = {
            "scene": args.scene, "method": method, "estimator": args.estimator, "map_dir": str(map_dir), "baseline": args.baseline, "snr": args.snr,
            "T_gt_from_map_first": T_first.tolist(), "T_gt_from_map_umeyama": T_umeyama.tolist(), "map_ate_rmse": ate,
            "kf_gt": {str(k): _mat_to_pose7(np.asarray(v)) for k, v in kf_gt.items()},
            "map_final_nodes": final_nodes,
            "n_components": int(system.kf_gmm_n_components),
            "obs_cadence": {"min_translation": args.obs_min_translation, "min_rotation": args.obs_min_rotation,
                            "max_interval": args.obs_max_interval},
        }
        logger.info(f"[trace] map ATE (Umeyama, permanent kfs) {ate:.3f} m; {len(final_nodes)} keyframes")
        release(system)
        map_trace.write_text(json.dumps({"sessions": rec.sessions, "nodes": rec.nodes, "edges": rec.edges,
                                         "steps": rec.steps, "extra": extra}, separators=(",", ":")))
        rec.save(extra)

    # ---------------- relocalization sessions
    if args.variants:
        ds0 = StereoSequenceLoader(str(root / args.variants[0]), depth_source=depth_source, snr=args.snr, baseline=args.baseline)
        system = new_system(args, ds0)
        rec.attach(system)
        for vi, v in enumerate(args.variants):
            ds = StereoSequenceLoader(str(root / v), depth_source=depth_source, snr=args.snr, baseline=args.baseline,
                                      seed=None if args.seed is None else args.seed + 1 + vi)
            system.load_map(str(map_file))   # keeps the hypothesis-manager object, hooks stay attached
            q_end = len(ds) if args.query_end is None else min(len(ds), args.query_end)
            rec.begin_session(f"relocalization: {v}", "reloc", v, str(root / v), q_end - args.query_start,
                              ds.left_c2w[args.query_start])
            t0 = time.time()
            for idx, d in enumerate(ds.replay_data(start_idx=args.query_start, end_idx=q_end, stride=args.stride)):
                if idx == 0:
                    d["delta_pose"] = None
                rec.before_step(d)
                ts = time.perf_counter()
                system.step(obs=d, data=d)
                rec.after_step(d, time.perf_counter() - ts)
                if idx % 100 == 0:
                    logger.info(f"[trace] {v} step {idx} ({(idx + 1) / (time.time() - t0):.1f} FPS)")
            if getattr(system, "_lc_verifier", None) is not None:
                rec.sessions[-1]["lc_stats"] = dict(system._lc_verifier.stats)
            rec.save(extra)
            kf_gt_map = {int(k): _pose7_to_mat(p) for k, p in extra["kf_gt"].items()}
            dump_graph(system, out / f"graph_s{rec._session['id']}.json", {**kf_gt_map, **rec.kf_gt_session}, rec._session["id"],
                       meta={"scene": args.scene, "method": method, "variant": v, "seed": args.seed, "snr": args.snr, "baseline": args.baseline})
        release(system)
    rec.save(extra)
    logger.info("[trace] done")


if __name__ == "__main__":
    main()
