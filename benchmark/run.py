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
"""
from __future__ import annotations

import argparse
import json
import os
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
from metrics import ate, multisession, wilson  # noqa: E402

PY = sys.executable


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
    commit = ROOT / "COMMIT"
    info["commit"] = commit.read_text().strip() if commit.is_file() else None
    return info


def sh(cmd, log: Path, timeout=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(map(str, cmd))}\n")
        f.flush()
        try:
            # PYTHONPATH: a shared venv may hold an editable install of another CROSS checkout
            env = dict(os.environ, PYTHONPATH=str(ROOT), MPLBACKEND="Agg",
                       PYTORCH_CUDA_ALLOC_CONF=os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"))
            rc = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT),
                                timeout=timeout, env=env).returncode
        except subprocess.TimeoutExpired:
            f.write(f"\nTIMEOUT after {timeout} s\n")
            rc = -9
    return rc, time.time() - t0


def gt_poses(seq: Path) -> np.ndarray:
    return np.loadtxt(seq / "poses_left.txt").reshape(-1, 4, 4)


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


class Job:
    def __init__(self, a):
        self.a = a
        self.dcfg_all, self.scfg_all = load_cfg()
        self.dcfg = self.dcfg_all[a.dataset]
        self.scfg = self.scfg_all[a.system]
        self.scene = self.dcfg["scenes"][a.scene]
        self.setup_dir = self.dcfg["setups"][a.setup]
        self.data = Path(a.data) / a.dataset
        self.run_root = Path(a.out) / a.dataset / a.scene / a.system / a.setup / f"s{a.seed}"
        self.outdoor = self.dcfg["environment"] == "outdoor"
        self.snr = self.dcfg.get("snr")
        self.base = {"dataset": a.dataset, "scene": a.scene, "system": a.system, "setup": a.setup, "seed": a.seed,
                     "label": self.scfg["label"], "uses_odometry": self.scfg.get("uses_odometry", False)}

    def seq(self, name) -> Path:
        return self.data / name / self.setup_dir

    # ------------------------------------------------------------------ CROSS
    def cross_cmd(self, map_seq, query_seq, out, extra):
        a = self.a
        sysc = self.scfg
        cfgs = list(sysc.get("outdoor_config", [])) if self.outdoor else []
        stereo_folder = (self.seq(map_seq) / "left").is_dir()
        if sysc["runner"] == "cross_rgbd" and not stereo_folder:
            cmd = [PY, "scripts/map_and_reloc_rgbd.py", "--seed", a.seed]
            if self.snr:
                cmd += ["--snr", self.snr]
        else:
            cmd = [PY, "scripts/map_and_reloc.py", "--seed", a.seed]
            if sysc["runner"] == "cross_rgbd":           # RGB-D*: PnP on stereo-matched depth
                cmd += ["--estimator", "pnp", "--pnp-depth", "sgbm"]
            cmd += [str(v) for v in sysc.get("args", [])]
            if self.dcfg.get("baseline"):
                cmd += ["--baseline", self.dcfg["baseline"]]
            if self.snr:
                cmd += ["--snr", self.snr]
        cmd += ["--map", self.seq(map_seq), "--query", self.seq(query_seq), "--out", out]
        if cfgs:
            cmd += ["--config"] + [str(ROOT / c) for c in cfgs]
        return cmd + extra

    def cross_map(self, map_seq, out: Path):
        rc, dt = sh(self.cross_cmd(map_seq, map_seq, out, ["--skip-reloc"]), out / "bench.log", self.a.timeout)
        return rc, dt

    def cross_t1_result(self, seq_name, out: Path, dt):
        meta = json.loads((out / "map_meta.json").read_text())
        ids = [k for k in meta["kf_est"] if str(k) in meta["kf_gt"]]
        est = {i: np.asarray(meta["kf_est"][k]).reshape(4, 4) for i, k in enumerate(ids)}
        gt = np.stack([np.asarray(meta["kf_gt"][str(k)]).reshape(4, 4) for k in ids])
        r = ate(est, gt, sim3=False)
        r["completeness"] = 1.0              # CROSS reports a pose for every frame; keyframes span the whole run
        r["failed"] = False
        n = meta.get("n_frames")
        T = np.asarray(r["T_gt_from_est"])
        pos_est = (T[:3, :3] @ np.array([est[i][:3, 3] for i in est]).T).T + T[:3, 3]
        return {**self.base, "track": "t1", "sequence": seq_name, **{k: v for k, v in r.items() if k != "T_gt_from_est"},
                "fps": (n / meta["elapsed"]) if n and meta.get("elapsed") else None, "n_frames": n,
                "n_keyframes": meta.get("n_permanent"), "map_bytes": meta.get("map_file_bytes"), "wall_s": dt,
                "traj_est": downsample(pos_est[:, [0, 1, 2]]).round(3), "traj_gt": downsample(gt[:, :3, 3]).round(3)}

    def cross_query(self, map_dir: Path, q: str, t2: Path, t3: Path):
        a = self.a
        out = {}
        for track, d, extra in (("t2", t2, ["--trial-len", "0"]),
                                ("t3", t3, ["--trial-len", self.dcfg["trial_len"], "--trial-stride", self.dcfg["trial_stride"],
                                            "--r-d", self.dcfg["r_d"]])):
            if (d / "result.json").is_file() and not a.force:
                continue
            d.mkdir(parents=True, exist_ok=True)
            for f in ("map.pkl", "map_meta.json"):
                if not (d / f).exists():
                    os.symlink((map_dir / f).resolve(), d / f)
            rc, dt = sh(self.cross_cmd(self.scene["map"], q, d, ["--skip-map"] + [str(e) for e in extra]), d / "bench.log", a.timeout)
            if rc != 0 or not (d / "reloc_summary.json").is_file():
                write_result(d / "result.json", {**self.base, "track": track, "map": self.scene["map"], "query": q,
                                                 "status": "failed", "rc": rc, "wall_s": dt, **env_info()})
                continue
            rows = json.loads((d / "reloc_rows.json").read_text())
            summ = json.loads((d / "reloc_summary.json").read_text())
            if track == "t2":
                res = self.t2_from_rows(rows, q, "c0_t_err", [r.get("c0_rel_t_err", np.inf) for r in rows], summ)
            else:
                res = self.t3_from_summary(summ, q)
            res.update({"wall_s": dt, "status": "ok", **env_info()})
            write_result(d / "result.json", res)
            for f in ("reloc_rows.json",):
                if (d / f).is_file() and not a.keep_rows:
                    (d / f).unlink()
            out[track] = res
        return out

    # ------------------------------------------------------------------ results shared by all systems
    def t2_from_rows(self, rows, q, key, rel_errors, summ):
        thr = self.dcfg.get("lr_thresholds", [0.5, 1.0])
        e = np.array([r.get(key, np.inf) if r.get(key) is not None else np.inf for r in rows], dtype=float)
        m = multisession(e, thresholds=thr, loc_threshold=self.dcfg["r_d"] if self.outdoor else 1.0)
        rel = np.array([np.inf if v is None else v for v in rel_errors], dtype=float)
        m_rel = multisession(rel, thresholds=thr)
        return {**self.base, "track": "t2", "map": self.scene["map"], "query": q, **m,
                "map_relative": {k: v for k, v in m_rel.items() if k.startswith("lr@") or k == "ms_ate"},
                "err_curve": downsample(np.where(np.isfinite(e), e, -1.0)).round(3),
                "fps": summ.get("fps") or summ.get("fps_steps")}

    def t3_from_summary(self, summ, q):
        tr = summ["trials"]
        n = tr["n_trials"]
        k = int(round((tr["RS"] or 0) * n))
        trials = [{"start": t.get("start"), "success": t["success_rd"], "success_strict": t["success_1m_5deg"],
                   "final_err": t["final_t_err"]} for t in tr["trials"]]
        fails = [t["final_err"] for t in trials if not t["success"] and t["final_err"] is not None and np.isfinite(t["final_err"])]
        return {**self.base, "track": "t3", "map": self.scene["map"], "query": q, "n_trials": n, "n_success": k,
                "rs": tr["RS"], "rs_strict": tr["RS_1m_5deg"], "rs_ci95": wilson(k, n),
                "fail_err_median": float(np.median(fails)) if fails else None, "trials": trials, "r_d": self.dcfg["r_d"],
                "trial_len": self.dcfg["trial_len"]}

    # ------------------------------------------------------------------ external baselines
    def baseline_cmd(self, map_seq, query_seq, out, extra):
        system = self.a.system
        cmd = [PY, "scripts/baselines/run_baselines.py", "--map", self.seq(map_seq), "--query", self.seq(query_seq),
               "--out", out, "--snr", self.snr or 10]
        if self.a.setup == "stereo" and self.dcfg.get("baseline"):
            cmd += ["--baseline", self.dcfg["baseline"]]
        if system == "orbslam3":
            cmd += ["--system", "orbslam3", "--orb-sensor", self.a.setup]
        elif system == "rtabmap":
            cmd += ["--system", "rtabmap_stereo" if self.a.setup == "stereo" else "rtabmap"]
        return cmd + extra

    def baseline_map(self, map_seq, out: Path):
        """Map with up to two retries: ORB-SLAM3 occasionally crashes while saving its atlas at shutdown."""
        need = {"orbslam3": ["atlas.osa", "map_poses.txt"], "rtabmap": ["map.db", "map_poses.txt"]}[self.a.system]
        total = 0.0
        for attempt in range(3):
            for f in need + ["map_time.json", "map_poses.txt.final"]:
                if (out / f).exists() and attempt > 0:
                    (out / f).unlink()
            rc, dt = sh(self.baseline_cmd(map_seq, map_seq, out, ["--map-only"]), out / "bench.log", self.a.timeout)
            total += dt
            if rc == 0 and all((out / f).is_file() and (out / f).stat().st_size > 0 for f in need):
                return 0, total
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
                if self.a.system == "orbslam3":
                    poses = {i: v for i, v in apply_final_trajectory({}, mp, fps).items()}
                else:
                    poses = load_poses(fin)
        est = {i: T for i, (st, T) in poses.items() if st == 2}
        sim3 = self.a.setup in self.scfg.get("sim3_setups", [])
        r = ate(est, gt, sim3=sim3, fps=fps)
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
                                ("t3", t3, ["--trial-len", self.dcfg["trial_len"], "--trial-stride", self.dcfg["trial_stride"],
                                            "--r-d", self.dcfg["r_d"]])):
            if (d / "result.json").is_file() and not a.force:
                continue
            d.mkdir(parents=True, exist_ok=True)
            if (map_dir / "atlas.osa").exists() and not (d / "atlas.osa").exists():
                os.symlink((map_dir / "atlas.osa").resolve(), d / "atlas.osa")     # loaded read-only
            for f in ("map_poses.txt", "map_poses.txt.final", "map_time.json"):
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
                res = self.t2_from_rows(rows, q, "t_err", [r.get("c0_rel_t_err", np.inf) for r in rows], summ)
            else:
                res = self.t3_from_summary(summ, q)
            res.update({"wall_s": dt, "status": "ok", "rc": rc, **env_info()})
            write_result(d / "result.json", res)
            out[track] = res
        return out

    # ------------------------------------------------------------------ systems without map persistence
    CONCAT = {"mast3r_slam": ["scripts/baselines/run_mast3r_slam.py", "--sim3"],
              "vggt_slam": ["scripts/baselines/run_vggt_slam.py"]}

    def concat_cmd(self, map_seq, query_seq, out, extra):
        script, *flags = self.CONCAT[self.a.system]
        return [PY, script, "--map", self.seq(map_seq), "--query", self.seq(query_seq), "--out", out] + flags + extra

    def concat_map(self, map_seq, out: Path):
        return sh(self.concat_cmd(map_seq, map_seq, out, ["--map-only", "--timeout", self.a.timeout]), out / "bench.log",
                  self.a.timeout + 600)

    def concat_query(self, map_dir: Path, q: str, t2: Path, t3: Path):
        """Map + query (T2) or map + trial (T3) in one stream; T3 is capped at --concat-max-trials evenly spaced trials
        per query, because every trial re-runs the whole map session."""
        a = self.a
        out = {}
        for track, d, extra in (("t2", t2, ["--trial-len", "0"]),
                                ("t3", t3, ["--trial-len", self.dcfg["trial_len"], "--trial-stride", self.dcfg["trial_stride"],
                                            "--r-d", self.dcfg["r_d"], "--max-trials", a.concat_max_trials])):
            if (d / "result.json").is_file() and not a.force:
                continue
            d.mkdir(parents=True, exist_ok=True)
            rc, dt = sh(self.concat_cmd(self.scene["map"], q, d, [str(e) for e in extra] + ["--timeout", a.timeout]),
                        d / "bench.log", a.timeout * (a.concat_max_trials + 1))
            if not (d / "reloc_summary.json").is_file():
                write_result(d / "result.json", {**self.base, "track": track, "map": self.scene["map"], "query": q,
                                                 "status": "failed", "rc": rc, "wall_s": dt, **env_info()})
                continue
            rows = json.loads((d / "reloc_rows.json").read_text())
            summ = json.loads((d / "reloc_summary.json").read_text())
            res = self.t2_from_rows(rows, q, "t_err", [r.get("c0_rel_t_err", np.inf) for r in rows], summ) if track == "t2" \
                else self.t3_from_summary(summ, q)
            res.update({"wall_s": dt, "status": "ok", "rc": rc, "concat": True, **env_info()})
            write_result(d / "result.json", res)
            out[track] = res
        return out

    # ------------------------------------------------------------------ tasks
    def runner(self):
        r = self.scfg["runner"]
        if r in ("cross_rgbd", "cross_ff"):
            return self.cross_map, self.cross_t1_result, self.cross_query
        if r == "baseline":
            return self.baseline_map, self.baseline_t1_result, self.baseline_query
        if r == "concat":
            return self.concat_map, self.baseline_t1_result, self.concat_query
        raise NotImplementedError(f"runner {r} ({self.a.system})")

    def ensure_map(self, map_seq) -> Path:
        do_map, t1_result, _ = self.runner()
        d = self.run_root / "maps" / map_seq
        done = d / "MAP_DONE"
        lock = self.run_root / "maps" / f".{map_seq.replace('/', '_')}.lock"
        while not done.is_file():
            d.parent.mkdir(parents=True, exist_ok=True)
            try:
                lock.mkdir()
            except FileExistsError:
                if time.time() - lock.stat().st_mtime > self.a.timeout + 600:   # stale lock of a killed worker
                    shutil.rmtree(lock, ignore_errors=True)
                time.sleep(30)
                continue
            try:
                if not done.is_file():
                    d.mkdir(parents=True, exist_ok=True)
                    rc, dt = do_map(map_seq, d)
                    try:
                        res = t1_result(map_seq, d, dt)
                        res.update({"status": "ok" if rc == 0 else "failed", "rc": rc, **env_info()})
                    except Exception as e:      # noqa: BLE001
                        res = {**self.base, "track": "t1", "sequence": map_seq, "status": "failed", "rc": rc,
                               "error": repr(e), "wall_s": dt, **env_info()}
                    write_result(self.run_root / "t1" / map_seq / "result.json", res)
                    done.write_text(json.dumps({"rc": rc, "wall_s": dt}))
            finally:
                shutil.rmtree(lock, ignore_errors=True)
        return d

    def ready(self, names) -> bool:
        return all((self.seq(n) / "calib.json").is_file() for n in names)

    def run(self):
        a = self.a
        do_map, t1_result, do_query = self.runner()
        need = [self.scene["map"]] + ([a.seq] if a.task == "t1" else []) + ([a.query] if a.task == "query" else [])
        if not self.ready(need):
            print(f"data not ready: {[str(self.seq(n)) for n in need]}", file=sys.stderr)
            sys.exit(3)                        # the worker releases the job for a later pass
        if a.task == "map":
            md = self.ensure_map(self.scene["map"])
            if not self.scene.get("queries") and not a.keep_maps:     # single-session scene: the map is not reused
                for f in ("map.pkl", "atlas.osa", "map.db"):
                    if (md / f).exists():
                        (md / f).unlink()
                shutil.rmtree(md / "views_map", ignore_errors=True)
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
        elif a.task == "query":
            m = self.scene["map"]
            md = self.ensure_map(m)
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
    ap.add_argument("--concat-max-trials", type=int, default=20,
                    help="T3 trials per query for systems without map persistence (evenly spaced)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep-maps", action="store_true")
    ap.add_argument("--keep-rows", action="store_true", help="keep the per-frame rows of the query runs")
    a = ap.parse_args()
    Job(a).run()


if __name__ == "__main__":
    main()
