#!/usr/bin/env python3
"""Run one job of the CROSS benchmark and write its result.json (benchmark/PROTOCOL.md).

Tasks
  map    map the scene's map sequence and keep the map (T1 result of that sequence; T2 / T3 reuse the map)
  t1     map another sequence for single-session accuracy only (the map is deleted afterwards)
  query  T2 (the whole query session against the stored map) and T3 (independent trials) for one query sequence;
         builds the map first when it does not exist yet (with a lock, so parallel workers wait for one another)

  python benchmark/run.py --dataset openloris --scene office --system cross_rgbd --setup rgbd --task map \\
      --data $BENCH_DATA --out $BENCH_RESULTS
  python benchmark/run.py ... --task query --query office1-2

Layout: <out>/<dataset>/<scene>/<system>/<setup>/s<seed>/{maps/<seq>, t1/<seq>, t2/<map>__<query>, t3/<map>__<query>}

Scenes with a partial map and / or a map extension (CROSS runners; the cross-camera datasets): `map_start` / `map_end`
map only those frames of the map sequence (map key <map>@<start>-<end>); `extend` (with optional `extend_start` /
`extend_end`) runs that sequence as a second session in the map (scripts/map_and_reloc.py --extend-map), and the queries
load the extended map (key <map key>+<extend variant>, its score in ext/<key>/result.json).  Such scenes (and datasets
with `xcam: true`) store per evaluated query frame the area it lies in (errors.npz `area`: 1 = within the larger
threshold of the first session's path, 2 = only of the extension's, 0 = not covered) and per T3 trial `covered` and
`area` of its last frame; covered = the union of both sessions' paths.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark" / "eval"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))
from metrics import (  # noqa: E402
    COMPLETENESS_RULE,
    ate,
    completeness,
    moving_fraction,
    moving_frames,
    moving_rule,
    multisession,
    wilson,
)

PY = sys.executable
# --odom-source: the odometry file the systems that take odometry read in each sequence folder (None = the dataset's
# own: odom_left.txt, else simulated from the ground truth with the dataset's SNR).  "vio": the stereo VIO odometry of
# benchmark/datasets/prepare_vio.py, falling back to odom_left.txt (wheel odometry) where the folder has none.
ODOM_FILES = {"external": None, "vio": "odom_vio.txt"}


def load_cfg():
    ds = yaml.safe_load((ROOT / "benchmark/configs/datasets.yaml").read_text())
    sy = yaml.safe_load((ROOT / "benchmark/configs/systems.yaml").read_text())
    return ds, sy


def env_info():
    info = {"host": socket.gethostname(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        info["gpu"] = "cpu"
    else:
      try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i",
                              os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        info["gpu"] = gpu
      except Exception:
        info["gpu"] = None
    info["commit"] = code_version()
    return info


def code_version():
    """Commit of the code that runs: the COMMIT file of a copied snapshot (its commit hash), else the git checkout's
    short hash, with "+dirty" when tracked files differ from it (the results page groups runs by this label)."""
    commit = ROOT / "COMMIT"
    if commit.is_file():
        return commit.read_text().strip() or None
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=20)
        if h.returncode != 0 or not h.stdout.strip():
            return None
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True,
                               text=True, timeout=20).stdout.strip()
        return h.stdout.strip() + ("+dirty" if dirty else "")
    except Exception:       # noqa: BLE001
        return None


def sh(cmd, log: Path, timeout=None, alloc_conf=False):
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(map(str, cmd))}\n")
        f.flush()
        try:
            # PYTHONPATH: a shared venv may hold an editable install of another CROSS checkout
            # CROSS_EXTRA_PYTHONPATH: packages outside the venv (DPVO, Depth Anything 3, LightGlue for the mono mode and
            # visual odometry)
            extra = os.environ.get("CROSS_EXTRA_PYTHONPATH")
            env = dict(os.environ, PYTHONPATH=str(ROOT) + (os.pathsep + extra if extra else ""), MPLBACKEND="Agg")
            if alloc_conf:     # CROSS only: expandable segments break CUDA IPC between processes (MASt3R-SLAM back end)
                env["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            rc = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT),
                                timeout=timeout, env=env).returncode
        except subprocess.TimeoutExpired:
            f.write(f"\nTIMEOUT after {timeout} s\n")
            rc = -9
    return rc, time.time() - t0


def gt_poses(seq: Path) -> np.ndarray:
    return np.loadtxt(seq / "poses_left.txt").reshape(-1, 4, 4)


def cross_kf_frames(meta: dict, gt: np.ndarray) -> list:
    """Frame indices of every keyframe a CROSS session created (permanent and temporary): `kf_frame` of map_meta.json,
    or, for maps written before it existed, the frames whose ground-truth position equals the keyframe's (`kf_gt`
    holds the ground truth of every keyframe the session created)."""
    if meta.get("kf_frame"):
        return sorted(int(v) for v in meta["kf_frame"].values())
    from scipy.spatial import cKDTree
    tree = cKDTree(gt[:, :3, 3])
    frames = []
    for g in meta["kf_gt"].values():
        d, j = tree.query(np.asarray(g, dtype=float).reshape(4, 4)[:3, 3])
        if d < 1e-3:
            frames.append(int(j))
    return sorted(frames)


def downsample(a, n=800):
    a = np.asarray(a)
    if len(a) <= n:
        return a
    return a[np.linspace(0, len(a) - 1, n).astype(int)]


def write_result(path: Path, res: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(res), indent=1))


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (np.floating, np.integer)):
        return _clean(x.item())
    return x


def map_run_ok(out: Path) -> bool:
    """The baseline's mapping run finished (map_time.json with rc 0); a run killed at BASELINE_TIMEOUT leaves a
    truncated map behind."""
    try:
        return int(json.loads((out / "map_time.json").read_text()).get("rc", -1)) == 0
    except (OSError, ValueError):
        return False


class Job:
    def __init__(self, a):
        self.a = a
        self.dcfg_all, self.scfg_all = load_cfg()
        self.dcfg = self.dcfg_all[a.dataset]
        self.scfg = self.scfg_all[a.system]
        self.scene = self.dcfg["scenes"][a.scene]
        self.setup_dir = self.dcfg["setups"][a.setup]
        self.data = Path(a.data) / self.dcfg.get("data", a.dataset)      # dev splits read another dataset's folders
        if a.odom_source != "external" and not self.scfg.get("uses_odometry"):
            sys.exit(f"--odom-source {a.odom_source}: {a.system} does not take external odometry")
        self.odom_file = ODOM_FILES[a.odom_source]
        system_dir = a.system + (f"+{a.odom_source}" if a.odom_source != "external" else "") + \
            (f"@{a.variant}" if a.variant else "")
        self.run_root = Path(a.out) / a.dataset / a.scene / system_dir / a.setup / f"s{a.seed}"
        # --map-from: the query sessions load the map of another system's run (same dataset, scene, setup, seed and
        # odometry source; <system>[@<variant>]), e.g. a mono query in a map built in the stereo mode
        self.map_root = None
        if getattr(a, "map_from", ""):
            ms, _, mv = a.map_from.partition("@")
            if ms not in self.scfg_all:
                sys.exit(f"--map-from: unknown system {ms}")
            self.map_root = Path(a.out) / a.dataset / a.scene / (
                ms + (f"+{a.odom_source}" if a.odom_source != "external" else "") + (f"@{mv}" if mv else "")) / a.setup / f"s{a.seed}"
        # partial map / map extension (cross-camera datasets): the key of the map the queries load
        sc = self.scene
        self.map_range = (int(sc.get("map_start", 0)), sc.get("map_end"))
        self.base_key = sc.get("map", "") + (f"@{self.map_range[0]}-{self.map_range[1] if self.map_range[1] is not None else ''}"
                                             if self.map_range != (0, None) else "")
        self.map_key = self.base_key + (f"+{Path(sc['extend']).name}" if sc.get("extend") else "")
        self.area_split = bool(self.dcfg.get("xcam") or sc.get("extend") or self.map_range != (0, None))
        self.outdoor = self.dcfg["environment"] == "outdoor"
        # T1 completeness: a frame is covered within 1 s or within this distance of travelled path of a pose
        self.path_m = float(self.dcfg.get("completeness_path_m", 2.0 if self.outdoor else 1.0))
        self.snr = self.dcfg.get("snr")
        self.base = {"dataset": a.dataset, "scene": a.scene, "system": a.system, "setup": a.setup, "seed": a.seed,
                     "label": self.scfg["label"], "uses_odometry": self.scfg.get("uses_odometry", False)}
        if a.variant:
            self.base["variant"] = a.variant
        if a.odom_source != "external":
            self.base["odom"] = a.odom_source
            self.base["label"] = self.base["label"].replace("external odometry", "stereo VIO odometry")

    def seq(self, name) -> Path:
        return self.data / name / self.setup_dir

    # ------------------------------------------------------------------ CROSS
    def cross_cmd(self, map_seq, query_seq, out, extra):
        a = self.a
        sysc = self.scfg
        cfgs = list(sysc.get("outdoor_config", [])) if self.outdoor else []
        stereo_folder = (self.seq(map_seq) / "left").is_dir()
        if sysc["runner"] in ("cross_rgbd", "cross_mono") and not stereo_folder:
            cmd = [PY, "scripts/map_and_reloc_rgbd.py", "--seed", a.seed]
            cmd += [str(v) for v in sysc.get("args", [])]
            if self.snr:
                cmd += ["--snr", self.snr]
        else:
            cmd = [PY, "scripts/map_and_reloc.py", "--seed", a.seed]
            if sysc["runner"] == "cross_rgbd":           # RGB-D*: PnP on stereo-matched depth
                cmd += ["--mode", "rgbd", "--pnp-depth", "sgbm"]
            cmd += [str(v) for v in sysc.get("args", [])]
            if self.dcfg.get("baseline"):
                cmd += ["--baseline", self.dcfg["baseline"]]
            if self.snr:
                cmd += ["--snr", self.snr]
        if self.outdoor:
            cmd += [str(v) for v in sysc.get("outdoor_args", [])]
        cmd += ["--map", self.seq(map_seq), "--query", self.seq(query_seq), "--out", out]
        if cfgs:
            cmd += ["--config"] + [str(ROOT / c) for c in cfgs]
        if self.odom_file:
            cmd += ["--odom-file", self.odom_file]
        if self.a.odom_source == "vio":
            # the Basalt VIO file is the shipped *_basalt mode (its configuration, e.g. the VIO odometry noise), not the
            # *_wheel file the system's own arguments select
            cmd += ["--odometry", "basalt"]
        # experiments: synchronous PGO for reproducibility; the shipped default is the background optimisation
        # (configs/base/common.yaml), whose results land at machine-dependent times (PROTOCOL.md, DEV.md)
        cmd += ["--reproducible"]
        return cmd + shlex.split(a.args) + extra

    def cross_map(self, map_seq, out: Path, extra=()):
        rc, dt = sh(self.cross_cmd(map_seq, map_seq, out, ["--skip-reloc"] + [str(e) for e in extra]), out / "bench.log",
                    self.a.timeout, alloc_conf=True)
        return rc, dt

    def cross_extend(self, base_dir: Path, out: Path):
        """The extension session: the scene's `extend` sequence in the map of base_dir, saved to out."""
        sc = self.scene
        extra = ["--extend-map", base_dir.resolve(), "--map-start", int(sc.get("extend_start", 0))]
        if sc.get("extend_end") is not None:
            extra += ["--map-end", int(sc["extend_end"])]
        return self.cross_map(sc["extend"], out, extra)

    def cross_ext_result(self, out: Path, dt):
        """Score of the extension session (map_meta.json "extension", written by scripts/map_and_reloc.py), with the
        protocol's completeness over the keyframes it created on its own sequence."""
        meta = json.loads((out / "map_meta.json").read_text())
        e = meta["extension"]
        sc = self.scene
        s0, s1 = int(sc.get("extend_start", 0)), sc.get("extend_end")
        seq_gt = gt_poses(self.seq(sc["extend"]))[s0:s1]
        last = max(int(v) for v in meta["kf_session"].values())
        frames = sorted(int(meta["kf_frame"][k]) - s0 for k, v in meta["kf_session"].items()
                        if int(v) == last and k in meta["kf_frame"])
        fps = json.loads((self.seq(sc["extend"]) / "calib.json").read_text()).get("fps", 10.0)
        return {**self.base, "track": "ext", "map": self.base_key, "extend": sc["extend"], "map_key": self.map_key, **e,
                "completeness": completeness(frames, seq_gt, fps=fps, path_m=self.path_m) if len(seq_gt) else None,
                "completeness_rule": COMPLETENESS_RULE, "map_ate_rmse": meta.get("map_ate_rmse"),
                "map_ate_rmse_all": meta.get("map_ate_rmse_all"), "n_keyframes": meta.get("n_permanent"),
                "fps": (meta["n_frames"] / meta["elapsed"]) if meta.get("n_frames") and meta.get("elapsed") else None,
                "wall_s": dt}

    def cross_t1_result(self, seq_name, out: Path, dt):
        meta = json.loads((out / "map_meta.json").read_text())
        ids = [k for k in meta["kf_est"] if str(k) in meta["kf_gt"]]
        est = {i: np.asarray(meta["kf_est"][k]).reshape(4, 4) for i, k in enumerate(ids)}
        gt = np.stack([np.asarray(meta["kf_gt"][str(k)]).reshape(4, 4) for k in ids])
        r = ate(est, gt, sim3=False)                     # ATE over the permanent map keyframes
        # completeness with the protocol's rule over every keyframe the session created: the temporary keyframes of
        # revisits hold poses too, they are only not kept in the map
        seq_gt = gt_poses(self.seq(seq_name))
        fps = json.loads((self.seq(seq_name) / "calib.json").read_text()).get("fps", 10.0)
        frames = cross_kf_frames(meta, seq_gt)
        try:                                 # a partial map (--map-start / --map-end): its frames only
            run_args = json.loads((out / "args.json").read_text())
        except (OSError, ValueError):
            run_args = {}
        s0, s1 = int(run_args.get("map_start") or 0), run_args.get("map_end")
        if s0 or s1 is not None:
            seq_gt = seq_gt[s0:s1]
            frames = [f - s0 for f in frames if 0 <= f - s0 < len(seq_gt)]
        r["completeness"] = completeness(frames, seq_gt, fps=fps, path_m=self.path_m)
        r["failed"] = bool(r["completeness"] < 0.8)
        r["completeness_rule"] = COMPLETENESS_RULE
        n = meta.get("n_frames")
        T = np.asarray(r["T_gt_from_est"])
        pos_est = (T[:3, :3] @ np.array([est[i][:3, 3] for i in est]).T).T + T[:3, 3]
        if s0 or s1 is not None:
            r["frames"] = [s0, s1]
        return {**self.base, "track": "t1", "sequence": seq_name, **{k: v for k, v in r.items() if k != "T_gt_from_est"},
                "fps": (n / meta["elapsed"]) if n and meta.get("elapsed") else None, "n_frames": n,
                "n_keyframes": meta.get("n_permanent"), "n_keyframes_all": len(frames), "map_bytes": meta.get("map_file_bytes"),
                "wall_s": dt, "traj_est": downsample(pos_est[:, [0, 1, 2]]).round(3), "traj_gt": downsample(gt[:, :3, 3]).round(3)}

    def cross_query(self, map_dir: Path, q: str, t2: Path, t3: Path):
        a = self.a
        out = {}
        for track, d, extra in (("t2", t2, ["--trial-len", "0"]),
                                # (a dataset without trials, e.g. dev_kitti, has T2 only: --only t2)
                                ("t3", t3, ["--trial-len", self.dcfg.get("trial_len"), "--trial-stride", self.dcfg.get("trial_stride"),
                                            "--r-d", self.scene.get("r_d", self.dcfg["r_d"])])):
            if a.only and track not in a.only:
                continue
            if (d / "result.json").is_file() and not a.force and track not in a.redo and not self.stale(d, track):
                continue
            if track in a.redo or self.stale(d, track):
                shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)
            for f in ("map.pkl", "map_meta.json"):
                if not (d / f).exists():
                    os.symlink((map_dir / f).resolve(), d / f)
            rc, dt = sh(self.cross_cmd(self.scene["map"], q, d, ["--skip-map"] + [str(e) for e in extra]), d / "bench.log", a.timeout,
                        alloc_conf=True)
            if rc != 0 or not (d / "reloc_summary.json").is_file():
                write_result(d / "result.json", {**self.base, "track": track, "map": self.scene["map"], "query": q,
                                                 "status": "failed", "rc": rc, "wall_s": dt, **env_info()})
                continue
            rows = json.loads((d / "reloc_rows.json").read_text())
            summ = json.loads((d / "reloc_summary.json").read_text())
            if track == "t2":
                res = self.t2_from_rows(rows, q, "c0_t_err", [r.get("c0_rel_t_err", np.inf) for r in rows], summ, d)
            else:
                res = self.t3_from_summary(summ, q)
            res.update({"wall_s": dt, "status": "ok", **env_info()})
            write_result(d / "result.json", res)
            for f in ("reloc_rows.json",):
                if (d / f).is_file() and not a.keep_rows:
                    (d / f).unlink()
            out[track] = res
        return out

    def stale(self, d: Path, track: str) -> bool:
        """A stored T2 result made before recall was kept at the whole threshold grid (it has to be re-run)."""
        if track != "t2" or not (d / "result.json").is_file():
            return False
        r = json.loads((d / "result.json").read_text())
        return r.get("status") == "ok" and (any(f"lr@{x:g}" not in r for x in self.dcfg["thresholds"]) or "covered" not in r)

    # ------------------------------------------------------------------ results shared by all systems
    LR_GRID = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)      # T2 recall is stored at every threshold of this grid

    def covered(self, q):
        """Query frames whose ground-truth position lies within the larger threshold of the map session's ground-truth path
        (only these are evaluated: elsewhere the map has nothing to relocalize against); with a partial map / an
        extension session, of the mapped frames and of the extension's path."""
        return self.areas(q) > 0

    def areas(self, q):
        """Per query frame: 1 = within the larger threshold of the (first) map session's path (its mapped frames), 2 = only
        of the extension session's path, 0 = of neither."""
        from scipy.spatial import cKDTree
        sc, r = self.scene, self.dcfg["thresholds"][1]
        gq = gt_poses(self.seq(q))[:, :3, 3]
        gm = gt_poses(self.seq(sc["map"]))[self.map_range[0]:self.map_range[1], :3, 3]
        area = np.where(cKDTree(gm).query(gq)[0] < r, 1, 0)
        if sc.get("extend"):
            ge = gt_poses(self.seq(sc["extend"]))[int(sc.get("extend_start", 0)):sc.get("extend_end"), :3, 3]
            area[(area == 0) & (cKDTree(ge).query(gq)[0] < r)] = 2
        return area

    def moving(self, q):
        """Query frames at which the robot moves (PROTOCOL.md, T2 / T3: moving robot), from the ground truth."""
        fps = json.loads((self.seq(q) / "calib.json").read_text()).get("fps", 10.0)
        return moving_frames(gt_poses(self.seq(q)), fps, **moving_rule(self.dcfg))

    def t2_from_rows(self, rows, q, key, rel_errors, summ, out_dir=None):
        thr = self.dcfg["thresholds"]
        area_q = self.areas(q)
        cov = area_q > 0
        mov = self.moving(q)
        if "t2" not in moving_rule(self.dcfg)["tracks"]:
            mov = np.ones(len(mov), bool)
        frames = [int(r.get("frame", i)) for i, r in enumerate(rows)]
        covered = np.array([cov[f] if 0 <= f < len(cov) else True for f in frames], dtype=bool)
        moving = np.array([mov[f] if 0 <= f < len(mov) else True for f in frames], dtype=bool)
        keep = covered & moving              # only covered frames at which the robot moves are evaluated
        e_all = np.array([r.get(key, np.inf) if r.get(key) is not None else np.inf for r in rows], dtype=float)
        rel_all = np.array([np.inf if v is None else v for v in rel_errors], dtype=float)
        rows = [r for r, k in zip(rows, keep) if k]
        rel_errors = list(rel_all[keep])
        lr_all = multisession(e_all, thresholds=self.LR_GRID)
        e = np.array([r.get(key, np.inf) if r.get(key) is not None else np.inf for r in rows], dtype=float)
        m = multisession(e, thresholds=self.LR_GRID, loc_threshold=thr[0])
        rel = np.array([np.inf if v is None else v for v in rel_errors], dtype=float)
        m_rel = multisession(rel, thresholds=self.LR_GRID)
        extra = {}
        if self.area_split:              # per area (1: first session's, 2: extension only): evaluated frames' recall
            area = np.array([area_q[f] if 0 <= f < len(area_q) else 1 for f in frames], dtype=np.int8)
            extra["area"] = area
            split = {}
            for a_, name in ((1, "first"), (2, "extension")):
                sel = keep & (area == a_)
                if sel.any():
                    split[name] = {**{k: v for k, v in multisession(e_all[sel], thresholds=self.LR_GRID).items()
                                      if k.startswith("lr@") or k in ("n_frames", "ms_ate", "ms_median", "est_frac")},
                                   "err_median": float(np.median(e_all[sel])),
                                   "map_relative": {k: v for k, v in multisession(rel_all[sel], thresholds=self.LR_GRID).items()
                                                    if k.startswith("lr@")}}
            extra_res = {"area_split": split, "map_key": self.map_key}
        else:
            extra_res = {}
        if out_dir is not None:          # per-frame errors: later threshold changes need no rerun
            np.savez_compressed(Path(out_dir) / "errors.npz", err=e_all.astype(np.float32), err_map_relative=rel_all.astype(np.float32),
                                covered=covered, moving=moving, frame=np.asarray(frames), **extra)
        return {**self.base, "track": "t2", "map": self.scene["map"], "query": q, **m, **extra_res,
                "covered": float(covered.mean()) if len(covered) else None,
                "moving": float(moving.mean()) if len(moving) else None, "evaluated": float(keep.mean()) if len(keep) else None,
                "n_frames_all": int(len(keep)),
                "all_frames": {k: v for k, v in lr_all.items() if k.startswith("lr@") or k == "ms_ate"},
                "map_relative": {k: v for k, v in m_rel.items() if k.startswith("lr@") or k == "ms_ate"},
                "err_curve": downsample(np.where(np.isfinite(e), e, -1.0)).round(3),
                "fps": summ.get("fps") or summ.get("fps_steps")}

    def t3_from_summary(self, summ, q):
        tr = summ["trials"]
        n = tr["n_trials"]
        k = int(round((tr["RS"] or 0) * n))
        mov = self.moving(q)                 # per trial: the fraction of its frames at which the robot moves
        tl = int(self.dcfg["trial_len"])
        trials = [{"start": t.get("start"), "success": t["success_rd"], "success_strict": t["success_1m_5deg"],
                   "final_err": t["final_t_err"],
                   "moving": round(moving_fraction(int(t.get("start") or 0), int(t.get("n") or tl), moving=mov), 3)}
                  for t in tr["trials"]]
        fails = [t["final_err"] for t in trials if not t["success"] and t["final_err"] is not None and np.isfinite(t["final_err"])]
        extra = {}
        if self.area_split:              # coverage (union of the sessions' paths) and area of each trial's last frame
            area_q = self.areas(q)
            for t, t0 in zip(trials, tr["trials"]):
                last = int(t0.get("start") or 0) + int(t0.get("n") or tl) - 1
                t["area"] = int(area_q[min(max(last, 0), len(area_q) - 1)])
                t["covered"] = t["area"] > 0
            extra = {"map_key": self.map_key}
        return {**self.base, "track": "t3", "map": self.scene["map"], "query": q, "n_trials": n, "n_success": k, **extra,
                "rs": tr["RS"], "rs_ci95": wilson(k, n),
                "fail_err_median": float(np.median(fails)) if fails else None, "trials": trials, "r_d": self.scene.get("r_d", self.dcfg["r_d"]),
                "trial_len": self.dcfg["trial_len"]}

    # ------------------------------------------------------------------ external baselines
    def baseline_cmd(self, map_seq, query_seq, out, extra):
        system = self.a.system
        cmd = [PY, "scripts/baselines/run_baselines.py", "--map", self.seq(map_seq), "--query", self.seq(query_seq),
               "--out", out, "--snr", self.snr or 10]
        if self.a.setup == "stereo" and self.dcfg.get("baseline"):
            cmd += ["--baseline", self.dcfg["baseline"]]
        if system in ("orbslam3", "orbslam3_imu"):
            cmd += ["--system", "orbslam3", "--orb-sensor", ("imu_" if system == "orbslam3_imu" else "") + self.a.setup]
        elif system in ("rtabmap", "rtabmap_vo", "rtabmap_vi"):
            cmd += ["--system", "rtabmap_stereo" if self.a.setup == "stereo" else "rtabmap"]
            if system in ("rtabmap_vo", "rtabmap_vi"):
                cmd += ["--rtab-vo"]
            if system == "rtabmap_vi":
                cmd += ["--rtab-imu"]
        cmd += [str(v) for v in self.scfg.get("args", [])]
        if self.odom_file:
            cmd += ["--odom-file", self.odom_file]
        return cmd + extra

    def baseline_map(self, map_seq, out: Path):
        """Map with up to two retries: ORB-SLAM3 occasionally crashes while saving its atlas at shutdown."""
        need = {"orbslam3": ["atlas.osa", "map_poses.txt"], "orbslam3_imu": ["atlas.osa", "map_poses.txt"],
                "rtabmap": ["map.db", "map_poses.txt"], "rtabmap_vo": ["map.db", "map_poses.txt"],
                "rtabmap_vi": ["map.db", "map_poses.txt"]}[self.a.system]
        total = 0.0
        for attempt in range(3):
            for f in need + ["map_time.json", "map_poses.txt.final"]:
                if (out / f).exists() and attempt > 0:
                    (out / f).unlink()
            rc, dt = sh(self.baseline_cmd(map_seq, map_seq, out, ["--map-only"]), out / "bench.log", self.a.timeout)
            total += dt
            if rc == 0 and all((out / f).is_file() and (out / f).stat().st_size > 0 for f in need) and map_run_ok(out):
                return 0, total
            if (out / "map_time.json").is_file() and json.loads((out / "map_time.json").read_text()).get("rc") == -9:
                break                                       # killed at BASELINE_TIMEOUT: a retry would time out too
        return (rc if rc != 0 else 1), total

    def baseline_t1_result(self, seq_name, out: Path, dt):
        from eval_traj import load_poses
        from run_baselines import apply_final_trajectory
        seq = self.seq(seq_name)
        gt = gt_poses(seq)
        fps = json.loads((seq / "calib.json").read_text()).get("fps", 10.0)
        mp = out / "map_poses.txt"
        poses = {}
        if mp.is_file() and mp.stat().st_size > 0:
            poses = load_poses(mp)
            fin = Path(str(mp) + ".final")
            if fin.is_file():
                if self.a.system in ("orbslam3", "orbslam3_imu"):
                    poses = {i: v for i, v in apply_final_trajectory({}, mp, fps).items()}
                else:
                    poses = load_poses(fin)
        est = {i: T for i, (st, T) in poses.items() if st == 2}
        sim3 = self.a.setup in self.scfg.get("sim3_setups", [])
        r = ate(est, gt, sim3=sim3, fps=fps, path_m=self.path_m)
        res = {**self.base, "track": "t1", "sequence": seq_name, **{k: v for k, v in r.items() if k != "T_gt_from_est"},
               "wall_s": dt}
        mt = out / "map_time.json"
        if mt.is_file():
            j = json.loads(mt.read_text())
            res["fps"] = len(gt) / j["seconds"] if j.get("seconds") else None
            res["rc"] = j.get("rc")
        if r.get("ate_rmse") is not None:
            T = np.asarray(r["T_gt_from_est"])
            ids = sorted(est)
            P = (T[:3, :3] @ np.array([est[i][:3, 3] for i in ids]).T).T + T[:3, 3]
            res["traj_est"] = downsample(P).round(3)
            res["traj_gt"] = downsample(gt[:, :3, 3]).round(3)
        return res

    def baseline_query(self, map_dir: Path, q: str, t2: Path, t3: Path):
        a = self.a
        out = {}
        for track, d, extra in (("t2", t2, ["--trial-len", "0"]),
                                # (a dataset without trials, e.g. dev_kitti, has T2 only: --only t2)
                                ("t3", t3, ["--trial-len", self.dcfg.get("trial_len"), "--trial-stride", self.dcfg.get("trial_stride"),
                                            "--r-d", self.scene.get("r_d", self.dcfg["r_d"])])):
            if a.reeval:              # re-score stored pose files with the current metrics
                if not (d / "query_poses_t0.txt").is_file():
                    continue
                old = json.loads((d / "result.json").read_text()) if (d / "result.json").is_file() else {}
                rc, dt = sh(self.baseline_cmd(self.scene["map"], q, d, [str(e) for e in extra] + ["--eval-only"]),
                            d / "bench.log", a.timeout)
                if (d / "reloc_summary.json").is_file() and rc == 0:
                    rows = json.loads((d / "reloc_rows.json").read_text())
                    summ = json.loads((d / "reloc_summary.json").read_text())
                    res = self.t2_from_rows(rows, q, "t_err", [r.get("c0_rel_t_err", np.inf) for r in rows], summ, d) if track == "t2" \
                        else self.t3_from_summary(summ, q)
                    keep = {k: old[k] for k in ("wall_s", "host", "gpu", "commit", "time", "rc") if k in old}
                    res.update({"status": "ok", **keep, "rescored": time.strftime("%Y-%m-%d %H:%M:%S")})
                    write_result(d / "result.json", res)
                continue
            if a.only and track not in a.only:
                continue
            if (d / "result.json").is_file() and not a.force and track not in a.redo:
                continue
            d.mkdir(parents=True, exist_ok=True)
            if (map_dir / "atlas.osa").exists() and not (d / "atlas.osa").exists():
                os.symlink((map_dir / "atlas.osa").resolve(), d / "atlas.osa")     # loaded read-only
            for f in ("map_poses.txt", "map_poses.txt.final", "map_poses.txt.times", "map_time.json"):
                if (map_dir / f).exists() and not (d / f).exists():
                    shutil.copy(map_dir / f, d / f)
            if (map_dir / "map.db").exists() and not (d / "map.db").exists():
                shutil.copy(map_dir / "map.db", d / "map.db")      # localization mode writes to the database
            rc, dt = sh(self.baseline_cmd(self.scene["map"], q, d, [str(e) for e in extra] + ["--require-map"]), d / "bench.log", a.timeout)
            if (d / "map.db").exists() and not a.keep_maps:
                (d / "map.db").unlink()
            if not (d / "reloc_summary.json").is_file():
                write_result(d / "result.json", {**self.base, "track": track, "map": self.scene["map"], "query": q,
                                                 "status": "failed", "rc": rc, "wall_s": dt, **env_info()})
                continue
            rows = json.loads((d / "reloc_rows.json").read_text())
            summ = json.loads((d / "reloc_summary.json").read_text())
            if track == "t2":
                res = self.t2_from_rows(rows, q, "t_err", [r.get("c0_rel_t_err", np.inf) for r in rows], summ, d)
            else:
                res = self.t3_from_summary(summ, q)
            res.update({"wall_s": dt, "status": "ok", "rc": rc, **env_info()})
            write_result(d / "result.json", res)
            out[track] = res
        return out

    # ------------------------------------------------------------------ systems without map persistence
    # script and flags; "{setup}" is replaced by the job's setup
    CONCAT = {"mast3r_slam": ["scripts/baselines/run_mast3r_slam.py", "--sim3"],
              "vggt_slam": ["scripts/baselines/run_vggt_slam.py"],
              "amb3r_slam": ["scripts/baselines/run_amb3r_slam.py", "--sensor", "{setup}"]}

    def concat_cmd(self, map_seq, query_seq, out, extra):
        script, *flags = self.CONCAT[self.a.system]
        flags = [f.replace("{setup}", self.a.setup) for f in flags]
        return [PY, script, "--map", self.seq(map_seq), "--query", self.seq(query_seq), "--out", out] + flags + extra

    def concat_map(self, map_seq, out: Path):
        return sh(self.concat_cmd(map_seq, map_seq, out, ["--map-only", "--timeout", self.a.concat_timeout]), out / "bench.log",
                  self.a.concat_timeout + 600)

    def concat_query(self, map_dir: Path, q: str, t2: Path, t3: Path):
        """Map + query (T2) or map + trial (T3) in one stream; T3 is capped at --concat-max-trials evenly spaced trials
        per query, because every trial re-runs the whole map session."""
        a = self.a
        out = {}
        for track, d, extra in (("t2", t2, ["--trial-len", "0"]),
                                # (a dataset without trials, e.g. dev_kitti, has T2 only: --only t2)
                                ("t3", t3, ["--trial-len", self.dcfg.get("trial_len"), "--trial-stride", self.dcfg.get("trial_stride"),
                                            "--r-d", self.scene.get("r_d", self.dcfg["r_d"]), "--max-trials", a.concat_max_trials])):
            if a.reeval:              # re-score from the stored per-frame rows (no system run)
                if (d / "reloc_rows.json").is_file() and (d / "reloc_summary.json").is_file() and (d / "result.json").is_file():
                    old = json.loads((d / "result.json").read_text())
                    rows = json.loads((d / "reloc_rows.json").read_text())
                    summ = json.loads((d / "reloc_summary.json").read_text())
                    res = self.t2_from_rows(rows, q, "t_err", [r.get("c0_rel_t_err", np.inf) for r in rows], summ, d) \
                        if track == "t2" else self.t3_from_summary(summ, q)
                    keep = {k: old[k] for k in ("wall_s", "host", "gpu", "commit", "time", "rc") if k in old}
                    res.update({"status": "ok", "concat": True, **keep, "rescored": time.strftime("%Y-%m-%d %H:%M:%S")})
                    write_result(d / "result.json", res)
                continue
            if a.only and track not in a.only:
                continue
            if (d / "result.json").is_file() and not a.force and track not in a.redo:
                continue
            d.mkdir(parents=True, exist_ok=True)
            rc, dt = sh(self.concat_cmd(self.scene["map"], q, d, [str(e) for e in extra] + ["--timeout", a.concat_timeout]),
                        d / "bench.log", a.concat_timeout * (a.concat_max_trials + 1))
            if not (d / "reloc_summary.json").is_file():
                write_result(d / "result.json", {**self.base, "track": track, "map": self.scene["map"], "query": q,
                                                 "status": "failed", "rc": rc, "wall_s": dt, **env_info()})
                continue
            rows = json.loads((d / "reloc_rows.json").read_text())
            summ = json.loads((d / "reloc_summary.json").read_text())
            crashed = [t for t in summ.get("times", []) if t.get("rc", 0) != 0]
            if crashed:               # out of memory / timeout of the system: not a localization result, re-run later
                write_result(d / "result.json", {**self.base, "track": track, "map": self.scene["map"], "query": q, "status": "failed",
                                                 "error": f"{len(crashed)} of {len(summ.get('times', []))} runs crashed or timed out",
                                                 "rc": crashed[0].get("rc"), "wall_s": dt, "concat": True, **env_info()})
                continue
            res = self.t2_from_rows(rows, q, "t_err", [r.get("c0_rel_t_err", np.inf) for r in rows], summ, d) if track == "t2" \
                else self.t3_from_summary(summ, q)
            res.update({"wall_s": dt, "status": "ok", "rc": rc, "concat": True, **env_info()})
            write_result(d / "result.json", res)
            out[track] = res
        return out

    # ------------------------------------------------------------------ tasks
    def runner(self):
        r = self.scfg["runner"]
        if r in ("cross_rgbd", "cross_ff", "cross_mono"):
            return self.cross_map, self.cross_t1_result, self.cross_query
        if r == "baseline":
            return self.baseline_map, self.baseline_t1_result, self.baseline_query
        if r == "concat":
            return self.concat_map, self.baseline_t1_result, self.concat_query
        raise NotImplementedError(f"runner {r} ({self.a.system})")

    def ensure_map(self, map_seq) -> Path:
        do_map, t1_result, _ = self.runner()
        return self._ensure(map_seq, lambda d: do_map(map_seq, d), lambda d, dt: t1_result(map_seq, d, dt),
                            self.run_root / "t1" / map_seq / "result.json", {"track": "t1", "sequence": map_seq})

    def scene_map(self) -> Path:
        """The map the scene's queries load: the map sequence's map, or (partial map / extension) the map of its frames
        and the extension session's map built on it."""
        if self.map_key == self.scene["map"]:
            return self.ensure_map(self.scene["map"])
        if self.scfg["runner"] not in ("cross_rgbd", "cross_ff", "cross_mono"):
            sys.exit(f"map_start / map_end / extend: CROSS runners only ({self.a.system})")
        m, (s0, s1) = self.scene["map"], self.map_range
        extra = ["--map-start", s0] + (["--map-end", s1] if s1 is not None else [])
        base = self._ensure(self.base_key, lambda d: self.cross_map(m, d, extra), lambda d, dt: self.cross_t1_result(m, d, dt),
                            self.run_root / "t1" / self.base_key / "result.json", {"track": "t1", "sequence": m})
        if not self.scene.get("extend"):
            return base

        def extend(d):
            rc = json.loads((base / "MAP_DONE").read_text()).get("rc", 0)
            if rc != 0:                         # no map to extend
                return rc, 0.0
            return self.cross_extend(base, d)
        return self._ensure(self.map_key, extend, self.cross_ext_result, self.run_root / "ext" / self.map_key / "result.json",
                            {"track": "ext", "map": self.base_key, "extend": self.scene["extend"]})

    def _ensure(self, map_key, build, result, res_file: Path, fields: dict) -> Path:
        """Build the map maps/<map_key> once (build(dir) -> (rc, seconds); result(dir, seconds) -> its result, written to
        res_file), with a lock so parallel workers share it."""
        d = self.run_root / "maps" / map_key
        done = d / "MAP_DONE"
        lock = self.run_root / "maps" / f".{map_key.replace('/', '_')}.lock"
        while not done.is_file():
            d.parent.mkdir(parents=True, exist_ok=True)
            try:
                lock.mkdir()
            except FileExistsError:
                if time.time() - lock.stat().st_mtime > self.a.timeout + 600:   # stale lock of a killed worker
                    shutil.rmtree(lock, ignore_errors=True)
                    continue
                if not self.a.wait_for_map:   # another worker builds this map: release the job, it is retried later
                    print(f"map {map_key} is being built by another worker", file=sys.stderr)
                    sys.exit(3)
                time.sleep(30)
                continue
            try:
                if not done.is_file():
                    d.mkdir(parents=True, exist_ok=True)
                    rc, dt = build(d)
                    try:
                        res = result(d, dt)
                        res.update({"status": "ok" if rc == 0 else "failed", "rc": rc, **env_info()})
                    except Exception as e:      # noqa: BLE001
                        res = {**self.base, **fields, "status": "failed", "rc": rc,
                               "error": repr(e), "wall_s": dt, **env_info()}
                    write_result(res_file, res)
                    done.write_text(json.dumps({"rc": rc, "wall_s": dt}))
            finally:
                shutil.rmtree(lock, ignore_errors=True)
        return d

    def reeval_t1(self, seq):
        """Re-score a stored T1 run with the current metrics (no system run): the map session's files in maps/<seq>,
        another session's in t1/<seq>/native.  Keeps the run's own fields (time, host, commit, ...)."""
        _, t1_result, _ = self.runner()
        res_file = self.run_root / "t1" / seq / "result.json"
        d = self.run_root / "maps" / seq if seq == self.scene.get("map") else self.run_root / "t1" / seq / "native"
        if not res_file.is_file() or not d.is_dir():
            print(f"nothing to re-score: {res_file}", file=sys.stderr)
            return
        old = json.loads(res_file.read_text())
        try:
            res = t1_result(seq, d, old.get("wall_s"))
        except Exception as e:        # noqa: BLE001
            print(f"cannot re-score {d}: {e!r}", file=sys.stderr)
            return
        keep = {k: old[k] for k in ("status", "rc", "wall_s", "host", "gpu", "commit", "time", "fps", "error") if k in old}
        res.update({**keep, "rescored": time.strftime("%Y-%m-%d %H:%M:%S")})
        write_result(res_file, res)
        print(f"{seq}: completeness {old.get('completeness')} -> {res.get('completeness')}, failed {old.get('failed')} -> "
              f"{res.get('failed')}, ATE {old.get('ate_rmse')} -> {res.get('ate_rmse')}")

    def ready(self, names) -> bool:
        return all((self.seq(n) / "calib.json").is_file() for n in names)

    def run(self):
        a = self.a
        do_map, t1_result, do_query = self.runner()
        need = [self.scene["map"]] + ([self.scene["extend"]] if self.scene.get("extend") else []) + \
            ([a.seq] if a.task == "t1" else []) + ([a.query] if a.task == "query" else [])
        if not self.ready(need):
            print(f"data not ready: {[str(self.seq(n)) for n in need]}", file=sys.stderr)
            sys.exit(3)                        # the worker releases the job for a later pass
        if self.map_root is not None and a.task != "query":
            sys.exit("--map-from: query tasks only (the map and its T1 belong to the other system's run)")
        if a.task == "map" and not a.reeval:
            md = self.scene_map()
            if not self.scene.get("queries") and not a.keep_maps:     # single-session scene: the map is not reused
                for f in ("map.pkl", "atlas.osa", "map.db"):
                    if (md / f).exists():
                        (md / f).unlink()
                shutil.rmtree(md / "views_map", ignore_errors=True)
                shutil.rmtree(md / "map.pkl.store", ignore_errors=True)       # CROSS map data (format v2)
        elif a.task in ("map", "t1") and a.reeval:
            self.reeval_t1(self.scene["map"] if a.task == "map" else a.seq)
        elif a.task == "t1":
            seq = a.seq
            if seq == self.scene.get("map"):
                self.ensure_map(seq)
                return
            d = self.run_root / "t1" / seq
            if (d / "result.json").is_file() and not a.force:
                return
            nat = d / "native"
            nat.mkdir(parents=True, exist_ok=True)
            rc, dt = do_map(seq, nat)
            try:
                res = t1_result(seq, nat, dt)
                res.update({"status": "ok" if rc == 0 else "failed", "rc": rc, **env_info()})
            except Exception as e:      # noqa: BLE001
                res = {**self.base, "track": "t1", "sequence": seq, "status": "failed", "rc": rc, "error": repr(e),
                       "wall_s": dt, **env_info()}
            write_result(d / "result.json", res)
            if not a.keep_maps:
                for f in ("map.pkl", "atlas.osa", "map.db"):
                    if (nat / f).exists():
                        (nat / f).unlink()
                shutil.rmtree(nat / "views_map", ignore_errors=True)
                shutil.rmtree(nat / "map.pkl.store", ignore_errors=True)
        elif a.task == "query":
            m = self.map_key                   # the map sequence, or the key of a partial / extended map
            if a.reeval:
                tag = f"{m}__{a.query}".replace("/", "_")
                do_query((self.map_root or self.run_root) / "maps" / m, a.query, self.run_root / "t2" / tag, self.run_root / "t3" / tag)
                return
            if self.map_root is not None:         # another system's map: wait for it (exit 3: the worker retries later)
                md = self.map_root / "maps" / m
                while not (md / "MAP_DONE").is_file():
                    if not a.wait_for_map:
                        print(f"map {md} not built yet", file=sys.stderr)
                        sys.exit(3)
                    time.sleep(30)
            else:
                md = self.scene_map()
            tag = f"{m}__{a.query}".replace("/", "_")
            if json.loads((md / "MAP_DONE").read_text()).get("rc", 0) != 0:     # no map to localize in
                for track in ("t2", "t3"):
                    write_result(self.run_root / track / tag / "result.json",
                                 {**self.base, "track": track, "map": m, "query": a.query, "status": "failed",
                                  "error": "mapping run failed", **env_info()})
                return
            do_query(md, a.query, self.run_root / "t2" / tag, self.run_root / "t3" / tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--setup", required=True)
    ap.add_argument("--task", choices=["map", "t1", "query"], required=True)
    ap.add_argument("--seq", help="sequence of a t1 task")
    ap.add_argument("--query", help="query sequence of a query task")
    ap.add_argument("--data", default=os.environ.get("BENCH_DATA"))
    ap.add_argument("--out", default=os.environ.get("BENCH_RESULTS"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=21600.0, help="seconds per harness run (a T3 run holds all trials)")
    ap.add_argument("--concat-timeout", type=float, default=5400.0,
                    help="seconds per run of a system without map persistence (map + trial stream)")
    ap.add_argument("--concat-max-trials", type=int, default=5,
                    help="T3 trials per query for systems without map persistence (evenly spaced)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--wait-for-map", action="store_true", help="wait while another worker builds the map (default: exit 3)")
    ap.add_argument("--reeval", action="store_true", help="re-score existing runs from their stored files (query tasks: "
                                                          "baselines' pose files; map / t1 tasks: the T1 result)")
    ap.add_argument("--redo", nargs="*", default=[], choices=["t2", "t3"], help="query task: re-run these tracks although results exist")
    ap.add_argument("--only", nargs="*", default=[], choices=["t2", "t3"], help="query task: run only these tracks")
    ap.add_argument("--keep-maps", action="store_true")
    ap.add_argument("--keep-rows", action="store_true", help="keep the per-frame rows of the query runs")
    ap.add_argument("--odom-source", choices=sorted(ODOM_FILES), default="external",
                    help="odometry of the systems that take odometry: the dataset's (external) or the stereo VIO "
                         "(vio: odom_vio.txt, wheel odometry where there is none); results go to <system>+vio")
    ap.add_argument("--map-from", default="", metavar="SYSTEM[@VARIANT]",
                    help="query tasks: relocalize in the map of this system's run (same dataset / scene / setup / seed / "
                         "odometry source) instead of one built by --system, e.g. a mono query (--system "
                         "cross_mono_ff_odom --setup stereo) in a stereo map (--map-from cross_stereo)")
    ap.add_argument("--variant", default="", help="label of an experiment: results go to <system>@<variant>")
    ap.add_argument("--args", default="", help="extra arguments of the CROSS command (e.g. \"--max-refs 4\"); a --set or "
                                               "--config here replaces one given by systems.yaml")
    a = ap.parse_args()
    Job(a).run()


if __name__ == "__main__":
    main()
