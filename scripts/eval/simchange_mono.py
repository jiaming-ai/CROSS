#!/usr/bin/env python3
"""SimChange map-and-relocalize protocol for monocular CROSS (same trials and metrics as CROSS-stereo).

Phase 1 maps the scene's `map` sequence from RGB only and saves the map with the pose of every permanent keyframe.
Phase 2 splits each query variant into independent relocalization trials (a fresh session that loads the map) and
records the hypothesis-0 pose of every frame. Scoring uses scripts/eval/reloc_metrics.py unchanged: map-relative
error (the map's own estimate of the nearest keyframe transported by ground truth), trial success when the final
pose is within r_d (and within 1 m / 5 deg for the strict rate). Ground truth is read only for scoring.

The SimChange sequences are sparse renders (10 cm steps, up to 10 deg per frame, some instantaneous turns), not
video; frames are fed at a nominal --fps with synthetic timestamps. --motion odom replaces the visual frontend's
motion with the dataset's simulated wheel odometry (odom_snr10.0.txt, as used by CROSS-stereo) as a diagnostic.

  python scripts/eval/simchange_mono.py --scene data/sim/classroom --out outputs/sim/classroom_mono \\
      --dpvo-checkpoint models/dpvo.pth --mono-args "$(profile arguments)"
"""

from __future__ import annotations

import argparse
import copy
import json
import shlex
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reloc_metrics import build_trials, map_relative_errors, summarize_trials  # noqa: E402


class Sequence:
    def __init__(self, path, fps):
        self.path = Path(path)
        calib = json.loads((self.path / "calib.json").read_text())
        self.K = np.asarray(calib["K"], dtype=np.float64)
        self.size = (int(calib["width"]), int(calib["height"]))
        images = self.path / "left" if (self.path / "left").is_dir() else self.path / "rgb"   # SimChange | posed TUM
        self.images = sorted(images.glob("*.png"))
        self.gt = np.loadtxt(self.path / "poses_left.txt").reshape(-1, 4, 4)
        odom = self.path / "odom_snr10.0.txt"
        self.odom = np.loadtxt(odom).reshape(-1, 4, 4) if odom.exists() else None
        if len(self.images) != len(self.gt):
            raise ValueError(f"{self.path}: {len(self.images)} images but {len(self.gt)} poses")
        self.fps = fps

    def __len__(self):
        return len(self.images)

    def rgb(self, i):
        bgr = cv2.imread(str(self.images[i]))
        if (bgr.shape[1], bgr.shape[0]) != self.size:
            raise ValueError("Image size differs from calibration")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def timestamp(self, i):
        return i / self.fps


def cached(factory):
    """Memoize a heavy model constructor by its arguments (models are reused across fresh sessions)."""
    store = {}

    def make(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in store:
            store[key] = factory(*args, **kwargs)
        return store[key]
    return make


class OdometryFrontend:
    """Diagnostic motion source: the dataset's simulated wheel odometry, learned metric depth for mapping."""

    def __init__(self, sequence, config, metric, start):
        self.sequence, self.config, self.metric = sequence, config, metric
        self.index, self.frame = 0, start
        self.pose = np.eye(4)

    def step(self, rgb, timestamp):
        from cross.mono.frontend import MonoEstimate
        odom = self.sequence.odom
        delta = np.eye(4) if self.index == 0 else np.linalg.inv(odom[self.frame - 1]) @ odom[self.frame]
        self.pose = self.pose @ delta
        t = float(np.linalg.norm(delta[:3, 3]))
        std_t, std_r = 0.005 + 0.1 * t, 0.002 + 0.1 * np.linalg.norm(cv2.Rodrigues(delta[:3, :3])[0])
        covariance = np.diag([std_t ** 2] * 3 + [std_r ** 2] * 3)
        depth = None
        if self.index % self.config.mapping_interval == 0:
            depth = self.metric.predict_metric(rgb, self.sequence.K, rgb.shape[:2])
        self.index += 1
        self.frame += 1
        return MonoEstimate(timestamp, self.pose.copy(), delta, covariance,
                            depth if depth is not None else np.ones(rgb.shape[:2], np.float32),
                            dict(valid=True, frame=self.index - 1))


class Runner:
    def __init__(self, mono_args, dpvo_checkpoint, motion, device="cuda"):
        import cross.db.db as db_module
        import cross.mono.dpvo_frontend as dpvo_module
        from cross.mono.run import build_parser, config_from_args
        from cross.mono.models import DA3Geometry, DA3MetricDepth
        db_module.BoQ = cached(db_module.BoQ)
        dpvo_module.DA3MetricDepth = cached(DA3MetricDepth)
        args = build_parser().parse_args(["unused", "--output", "unused", "--dpvo-checkpoint", str(dpvo_checkpoint),
                                          *mono_args])
        self.config = config_from_args(args)
        if self.config.frontend != "dpvo":
            raise ValueError("The SimChange runner uses the synchronous DPVO frontend (--frontend dpvo)")
        self.device, self.motion = device, motion
        self.geometry = DA3Geometry(self.config.pose_model, device, self.config.resolution)
        self.metric = dpvo_module.DA3MetricDepth(self.config.metric_model, device, self.config.metric_resolution)
        self.mono = None
        self.pristine = None

    def session(self, sequence, start, load_map=None):
        """A fresh relocalization session: new frontend, new CROSS mapper, optionally a loaded map."""
        from cross.core.system import System
        from cross.core.types import Camera
        from cross.mono.dpvo_frontend import DPVOFrontend
        from cross.mono.system import MonocularSystem
        if self.motion == "odom":
            frontend = OdometryFrontend(sequence, self.config, self.metric, start)
            frontend.geometry = self.geometry
        else:
            frontend = DPVOFrontend(sequence.K, self.config, self.device, geometry_model=self.geometry)
        if self.mono is not None:
            # release the previous session (DPVO buffers, graph tensors) before allocating the next one
            import gc
            import torch
            self.mono.mapper.shutdown()
            self.mono.frontend = self.mono.mapper = None
            gc.collect()
            torch.cuda.empty_cache()
            self.sessions = getattr(self, "sessions", 0) + 1
            if self.sessions % 10 == 0:
                print(json.dumps(dict(sessions=self.sessions, allocated_gb=torch.cuda.memory_allocated() / 1e9,
                                      reserved_gb=torch.cuda.memory_reserved() / 1e9)), flush=True)
        if self.mono is None:
            self.mono = MonocularSystem(sequence.K, sequence.size, self.config, device=self.device, frontend=frontend)
            self.pristine = copy.deepcopy(self.mono.mapper.config)
            self.pose_estimator = self.mono.mapper.pose_est
        else:
            from cross.mono.system import attach_pair_motion
            attach_pair_motion(frontend, self.pose_estimator, self.config, sequence.K)
            self.mono.frontend = frontend
            self.mono.mapper = System(device=self.device, visualize=False,
                                      camera=Camera(sequence.K.copy(), *sequence.size),
                                      config=copy.deepcopy(self.pristine), pose_estimator=self.pose_estimator)
            self.mono.initialized = False
            self.mono.map_alignment = np.eye(4)
        if load_map is not None:
            self.mono.load_map(load_map)
        return self.mono

    def run(self, sequence, start, end, load_map=None, record=None):
        mono = self.session(sequence, start, load_map)
        rows = []
        for i in range(start, end):
            estimate = mono.step(sequence.rgb(i), sequence.timestamp(i))
            rows.append(dict(frame=i, gt_pose=sequence.gt[i].reshape(-1).tolist(),
                             c0_pose=np.asarray(estimate.pose).reshape(-1).tolist(),
                             valid=bool(estimate.diagnostics.get("valid", True))))
        return mono, rows


def keyframe_poses(mono, sequence):
    """Permanent map keyframes: estimated pose (map frame) and ground truth, keyed by keyframe id."""
    est, gt = {}, {}
    for kid, kf in mono.mapper.hypothesis_manager.nodes.items():
        if kf.temporary or kf.timestamp is None:
            continue
        frame = int(round(float(kf.timestamp) * sequence.fps))
        est[str(kid)] = kf.pose_mu[0].matrix().detach().cpu().numpy().astype(float).reshape(-1).tolist()
        gt[str(kid)] = sequence.gt[frame].reshape(-1).tolist()
    return est, gt


def ate(est, gt):
    from cross.mono.evaluate import fit_alignment
    e = np.array([np.asarray(v).reshape(4, 4)[:3, 3] for v in est])
    g = np.array([np.asarray(v).reshape(4, 4)[:3, 3] for v in gt])
    out = {}
    for mode, scale in (("se3", False), ("sim3", True)):
        s, R, t = fit_alignment(e, g, scale)
        out[f"ate_{mode}_m"] = float(np.sqrt(np.mean(np.sum((s * e @ R.T + t - g) ** 2, axis=1))))
        if scale:
            out["sim3_scale"] = s
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--map-name", default="map", help="sequence used to build the map (Lone Monk: map_loop)")
    ap.add_argument("--variants", nargs="*", default=None, help="query variants (default: all with images)")
    ap.add_argument("--trial-len", type=int, default=40)
    ap.add_argument("--trial-stride", type=int, default=20)
    ap.add_argument("--r-d", type=float, default=2.0)
    ap.add_argument("--fps", type=float, default=5.0, help="nominal frame rate of the sparse renders")
    ap.add_argument("--motion", choices=["dpvo", "odom"], default="dpvo")
    ap.add_argument("--dpvo-checkpoint", required=True)
    ap.add_argument("--mono-args", default="", help="cross.mono.run arguments (quoted string)")
    ap.add_argument("--skip-map", action="store_true", help="reuse OUT/map.pkl and OUT/map_meta.json")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    runner = Runner(shlex.split(a.mono_args), a.dpvo_checkpoint, a.motion)
    map_seq = Sequence(a.scene / a.map_name, a.fps)
    if not a.skip_map:
        t0 = time.time()
        mono, rows = runner.run(map_seq, 0, len(map_seq))
        mono.save_map(a.out / "map.pkl")
        kf_est, kf_gt = keyframe_poses(mono, map_seq)
        meta = dict(kf_est=kf_est, kf_gt=kf_gt, n_frames=len(map_seq), seconds=time.time() - t0,
                    keyframe_ate=ate(list(kf_est.values()), [kf_gt[k] for k in kf_est]),
                    trajectory_ate=ate([r["c0_pose"] for r in rows], [r["gt_pose"] for r in rows]),
                    motion=a.motion, mono_args=a.mono_args, fps=a.fps)
        (a.out / "map_meta.json").write_text(json.dumps(meta))
        np.savetxt(a.out / "map_trajectory.txt", np.array([r["c0_pose"] for r in rows]))
        print(json.dumps(dict(map=str(a.scene), keyframe_ate=meta["keyframe_ate"],
                              trajectory_ate=meta["trajectory_ate"], keyframes=len(kf_est))), flush=True)
    meta = json.loads((a.out / "map_meta.json").read_text())
    variants = a.variants if a.variants is not None else sorted(
        p.name for p in a.scene.iterdir() if (p / "left").is_dir() and "__orig" not in p.name)
    for variant in variants:
        target = a.out / variant / "reloc_summary.json"
        if target.exists():
            continue
        seq = Sequence(a.scene / variant, a.fps)
        rows = []
        t0 = time.time()
        for ti, (s, e) in enumerate(build_trials(len(seq), a.trial_len, a.trial_stride)):
            _, trial_rows = runner.run(seq, s, e, load_map=a.out / "map.pkl")
            for r in trial_rows:
                r["trial"] = ti
            rows += trial_rows
        errors = map_relative_errors(rows, meta, pose_keys=("c0",))
        for r, err in zip(rows, errors):
            r.update(err)
        summary = summarize_trials(rows, "c0_rel", r_d=a.r_d)
        summary.update(variant=variant, seconds=time.time() - t0, n_frames=len(seq), trial_len=a.trial_len,
                       trial_stride=a.trial_stride, motion=a.motion)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary))
        (target.parent / "rows.json").write_text(json.dumps(
            [{k: v for k, v in r.items() if k not in ("gt_pose",)} for r in rows]))
        print(json.dumps(dict(variant=variant, RS=summary["RS"], RS_1m_5deg=summary["RS_1m_5deg"],
                              n_trials=summary["n_trials"], seconds=round(summary["seconds"], 1))), flush=True)


if __name__ == "__main__":
    main()
