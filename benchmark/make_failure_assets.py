#!/usr/bin/env python3
"""Media and per-frame poses of the failure cases of the results page (benchmark/configs/failure_cases.yaml).

For every case it writes
  benchmark/site/assets/failures/<id>.mp4   the camera frames of the case (T3: next to the map session's frame nearest
                                            in ground truth; one row per camera of the session)
  benchmark/results/failure_cases.json      per frame: ground truth, the map frame shown, and every listed system's pose
                                            in the ground-truth frame (top-down) with its error
and checks each system's final error against benchmark/results/results.json (a mismatch means the run folder found is
not the published run).

  python benchmark/make_failure_assets.py --data $BENCH_DATA --runs <run root> [<run root> ...]

Run roots are searched in order for <root>/<dataset>/<scene>/<system>/<setup>/s0/{t3/<map>__<query>, maps/<seq>}
(a T3 folder <map>__<query>@<start> holds a single trial run with --query-start / --query-end).
The poses come from the stored run outputs: CROSS T3 needs the per-frame rows (run.py --keep-rows), the baselines'
pose files are kept by default; T1 cases read the mapping run of the session (maps/<seq>).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))
from eval_traj import evaluate, hold_latest, load_poses, umeyama  # noqa: E402
from reloc_metrics import build_trials  # noqa: E402

CAM_LABEL = {"stereo": "stereo camera (left)", "rgbd": "colour camera", ".": "camera"}
W = 320                     # width of one image tile in the video


# ---------------------------------------------------------------------------------------------------------- sequences
class Seq:
    """A prepared sequence folder: ground truth, frame times, image files."""

    def __init__(self, path: Path):
        self.path = path
        self.G = np.loadtxt(path / "poses_left.txt").reshape(-1, 4, 4)
        sub = path / "left" if (path / "left").is_dir() else path / "rgb"
        self.files = sorted(sub.glob("*.png")) or sorted(sub.glob("*.jpg"))
        t = path / "times.txt"
        self.t = np.loadtxt(t).reshape(-1)[:len(self.G)] if t.is_file() else np.arange(len(self.G)) / 10.0
        self.fps = json.loads((path / "calib.json").read_text()).get("fps", 10.0) if (path / "calib.json").is_file() else 10.0

    def at_time(self, t):
        return int(np.clip(np.argmin(np.abs(self.t - t)), 0, len(self.G) - 1))

    def image(self, i, label=None, sub=None, width=W):
        im = cv2.imread(str(self.files[min(i, len(self.files) - 1)]), cv2.IMREAD_COLOR)
        im = cv2.resize(im, (width, int(round(width * im.shape[0] / im.shape[1]))), interpolation=cv2.INTER_AREA)
        if label:
            put(im, label, 0)
        if sub:
            put(im, sub, 1)
        return im


def put(im, text, line):
    y = 14 + 15 * line
    cv2.putText(im, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(im, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)


def seq_dir(data: Path, ds_cfg, dataset, seq, setup):
    if dataset == "simchange":
        return data / dataset / seq
    return data / ds_cfg[dataset].get("data", dataset) / seq / ds_cfg[dataset]["setups"][setup]


class Plane:
    """Top-down view: the two world axes orthogonal to the mean up direction of the cameras (image y points down)."""

    def __init__(self, G):
        up = -G[:, :3, 1].mean(0)
        k = int(np.argmax(np.abs(up)))
        self.ax = [a for a in range(3) if a != k]
        if np.sign(up[k]) < 0:                 # keep the view right-handed when seen from above
            self.ax = self.ax[::-1]

    def xy(self, P):
        return P[..., self.ax]

    def pose(self, T):
        p, z = self.xy(T[:3, 3]), self.xy(T[:3, 2])
        return [round(float(p[0]), 3), round(float(p[1]), 3), round(float(np.degrees(np.arctan2(z[1], z[0]))), 1)]


def map_frame(Gm, T, near):
    """The map frame shown next to a query frame: within `near` of its position the one with the most similar viewing
    direction, else the nearest in position."""
    d = np.linalg.norm(Gm[:, :3, 3] - T[:3, 3], axis=1)
    ang = np.degrees(np.arccos(np.clip(Gm[:, :3, 2] @ T[:3, 2], -1, 1)))
    c = np.where(d < near)[0]
    j = int(c[np.argmin(ang[c])]) if len(c) else int(np.argmin(d))
    return j, float(d[j]), float(ang[j])


# ---------------------------------------------------------------------------------------------------------- systems
def find_dir(roots, rel, start=None):
    """Run folders of a cell in the roots, in order; a T3 folder may also hold a single re-run trial (<tag>@<start>)."""
    for r in roots:
        for d in ([Path(r) / f"{rel}@{start}"] if start is not None else []) + [Path(r) / rel]:
            if d.is_dir():
                yield d


def cross_t3(d: Path, trial: int):
    """{frame: (T in the ground-truth frame, map-relative error)} from the per-frame rows of a CROSS T3 run."""
    rows_f = d / "reloc_rows.json"
    if not rows_f.is_file():
        return None
    meta = json.loads((d / "map_meta.json").read_text())
    A = np.asarray(meta["T_gt_from_map"])
    out = {}
    for r in json.loads(rows_f.read_text()):
        if r.get("trial") != trial:
            continue
        T = A @ np.asarray(r["c0_pose"]).reshape(4, 4)
        out[int(r["frame"])] = (T, r.get("c0_rel_t_err"))
    return out


def baseline_t3(d: Path, system, trial, a, b, gm: Seq, gq: Seq, sim3):
    from run_baselines import final_map_poses
    qf = d / f"query_poses_t{trial}.txt"
    if not (d / "map_poses.txt").is_file() or not qf.is_file():
        return None
    mp = final_map_poses("orbslam3" if system.startswith("orbslam3") else "rtabmap", d / "map_poses.txt", gm.fps)
    try:
        q = {i + a: v for i, v in load_poses(qf).items()}
    except Exception:          # noqa: BLE001  (empty file: the system never output a pose)
        q = {}
    return _to_gt(mp, q, gm, gq, a, b, sim3)


def vggt_t3(d: Path, trial, a, b, gm: Seq, gq: Seq):
    from run_vggt_slam import read_traj, sim3_to_gt
    f = d / f"vggt_t{trial}_traj.txt"
    if not f.is_file():
        return None
    n_map = len(gm.files)
    poses, _ = sim3_to_gt(read_traj(f), gm.G, n_map)
    mp = {i: (2, T) for i, T in poses.items() if i < n_map}
    q = {i - n_map + a: (2, T) for i, T in poses.items() if i >= n_map}
    return _to_gt(mp, q, gm, gq, a, b, True)


def _to_gt(mp, q, gm: Seq, gq: Seq, a, b, sim3):
    """Query poses in the ground-truth frame (the map's alignment, as eval_traj.evaluate scores them), with the
    map-relative error of every frame; frames hold the latest pose of the last second."""
    rows, _ = evaluate(mp, q, gm.G, gq.G, sim3=sim3, frame_range=(a, b))
    ids = [i for i, (st, _) in mp.items() if st == 2 and i < len(gm.G)]
    if rows is None or len(ids) < 3:
        return {}
    s, R, t = umeyama(np.array([mp[i][1][:3, 3] for i in ids]), np.array([gm.G[i][:3, 3] for i in ids]), with_scale=sim3)
    held = hold_latest(q, (2,), (a, b), 10)
    out = {}
    for r in rows:
        i = r["frame"]
        if i in held and held[i][0] == 2:
            T = held[i][1]
            A = np.eye(4)
            A[:3, :3] = R @ T[:3, :3]
            A[:3, 3] = s * R @ T[:3, 3] + t
            out[i] = (A, r.get("c0_rel_t_err"))
    return out


def trial_index(d: Path, start, n_frames, tl, stride):
    """Index of the trial starting at `start` in the run's trial list (systems without map persistence ran a subset)."""
    s = d / "reloc_summary.json"
    if s.is_file():
        for t in json.loads(s.read_text()).get("trials", {}).get("trials", []):
            if int(t.get("start", -1)) == start:
                return int(t["trial"])
        return None
    for k, (a, _) in enumerate(build_trials(n_frames, tl, stride)):
        if a == start:
            return k
    return None


def _kf_frames(meta, gs: Seq):
    """{keyframe id: frame} from the keyframes' ground-truth poses (they are the frames' own), in keyframe order."""
    out, prev = {}, 0
    P = gs.G[:, :3, 3]
    for k in sorted(meta["kf_gt"], key=int):
        p = np.asarray(meta["kf_gt"][k]).reshape(4, 4)[:3, 3]
        d = np.linalg.norm(P[prev:] - p, axis=1)
        exact = np.where(d < 1e-4)[0]
        prev = prev + int(exact[0] if len(exact) else np.argmin(d))
        out[str(k)] = prev
    return out


def t1_poses(d: Path, system, gs: Seq, sim3):
    """{frame: T in the ground-truth frame} of a session's mapping run (the trajectory T1 scores) and the frames the
    system tracked online (None when it does not report it)."""
    if system.startswith("cross"):
        meta = json.loads((d / "map_meta.json").read_text())
        A = np.asarray(meta["T_gt_from_map"])
        kf = meta.get("kf_frame") or _kf_frames(meta, gs)     # maps written before kf_frame: by their ground truth
        out = {}
        for k, v in meta["kf_est"].items():
            if str(k) in kf:
                out[int(kf[str(k)])] = A @ np.asarray(v).reshape(4, 4)
        return out, None
    if system == "vggt_slam":
        mp = load_poses(d / "map_poses.txt")
    else:
        from run_baselines import final_map_poses
        mp = final_map_poses("orbslam3" if system.startswith("orbslam3") else "rtabmap", d / "map_poses.txt", gs.fps)
    online = {i for i, (st, _) in load_poses(d / "map_poses.txt").items() if st == 2}
    ids = sorted(i for i, (st, _) in mp.items() if st == 2 and i < len(gs.G))
    if len(ids) < 3:
        return {}, online
    s, R, t = umeyama(np.array([mp[i][1][:3, 3] for i in ids]), np.array([gs.G[i][:3, 3] for i in ids]), with_scale=sim3)
    out = {}
    for i in ids:
        T = mp[i][1]
        A = np.eye(4)
        A[:3, :3] = R @ T[:3, :3]
        A[:3, 3] = s * R @ T[:3, 3] + t
        out[i] = A
    return out, (online if system != "vggt_slam" else None)


# ---------------------------------------------------------------------------------------------------------- video
def write_video(frames, path: Path, fps):
    tmp = Path(tempfile.mkdtemp())
    try:
        for k, im in enumerate(frames):
            cv2.imwrite(str(tmp / f"{k:05d}.png"), im)
        h, w = frames[0].shape[:2]
        vf = f"scale={w - w % 2}:{h - h % 2}"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(tmp / "%05d.png"), "-vf", vf,
                        "-c:v", "libx264", "-preset", "slow", "-crf", "30", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        str(path)], check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def sep(h):
    return np.full((h, 4, 3), 255, np.uint8)


# ---------------------------------------------------------------------------------------------------------- cases
def run_case(c, a, ds_cfg, sy_cfg, published, out_dir: Path):
    data = Path(a.data)
    d = c["dataset"]
    cfg = ds_cfg[d]
    near = 1.0 if cfg["environment"] == "indoor" else 3.0
    cams = c.get("cameras") or (["."] if d == "simchange" else ["stereo"])
    cam_setup = {"stereo": "stereo", "rgbd": "rgbd", ".": "stereo"}
    t3 = c["track"] == "t3"
    seq = c["query"] if t3 else c["sequence"]
    ref = Seq(seq_dir(data, ds_cfg, d, seq, cam_setup[cams[0]]))
    a0 = int(c["start"])
    n = int(c.get("len", cfg.get("trial_len", 100)))
    b0 = min(a0 + n, len(ref.G))
    frames = list(range(a0, b0))
    gmap = Seq(seq_dir(data, ds_cfg, d, cfg["scenes"][c["scene"]]["map"], cam_setup[cams[0]])) if t3 else ref
    plane = Plane(gmap.G)
    rec = {k: c[k] for k in ("id", "category", "track", "dataset", "scene", "note") if k in c}
    rec.update(sequence=seq, map=cfg["scenes"][c["scene"]]["map"] if t3 else None, start=a0, n=len(frames), fps=ref.fps,
               thresholds=cfg["thresholds"], cameras=[CAM_LABEL[x] for x in cams], video=f"assets/failures/{c['id']}.mp4")
    rec["gt"] = [plane.pose(ref.G[i]) for i in frames]
    stepm = max(1, len(gmap.G) // 1500)
    rec["map_path"] = np.round(plane.xy(gmap.G[::stepm, :3, 3]), 2).tolist()
    # ---- video: per camera a row [query | map frame]
    rows_per_frame, mf_ref = [], []
    cam_seqs = [(cam, ref if k == 0 else Seq(seq_dir(data, ds_cfg, d, seq, cam_setup[cam]))) for k, cam in enumerate(cams)]
    cam_maps = [(cam, gmap if k == 0 else Seq(seq_dir(data, ds_cfg, d, rec["map"], cam_setup[cam]))) for k, cam in enumerate(cams)] if t3 else []
    for i in frames:
        rows = []
        j, dist, ang = map_frame(gmap.G, ref.G[i], near) if t3 else (None, None, None)
        mf_ref.append([j, round(dist, 2), round(ang, 1)] if t3 else None)
        for k, (cam, s) in enumerate(cam_seqs):
            qi = s.at_time(ref.t[i]) if k else i
            short = seq.split("/")[-1].replace("campus_large_", "")
            lab = f"{'query' if t3 else 'session'} {short}  {(i - a0) / ref.fps:4.1f} s" if k == 0 else None
            width = W if t3 else (640 if len(cams) == 1 else 480)     # mapping stretches: no map tile, larger frames
            tiles = [s.image(qi, lab, CAM_LABEL[cam] if len(cams) > 1 else None, width)]
            if t3:
                ms = cam_maps[k][1]
                mj = ms.at_time(gmap.t[j]) if k else j
                tiles += [sep(tiles[0].shape[0]), ms.image(mj, "map session, same place" if k == 0 else None,
                                                             f"{dist:.1f} m and {ang:.0f} deg away" if k == 0 else None)]
            rows.append(np.hstack(tiles))
        if not t3 and len(rows) > 1:          # mapping stretches: the cameras side by side
            hmax = max(r.shape[0] for r in rows)
            rows = [np.pad(r, ((0, hmax - r.shape[0]), (0, 0), (0, 0)), constant_values=255) for r in rows]
            im = rows[0]
            for r in rows[1:]:
                im = np.hstack([im, sep(hmax), r])
            rows_per_frame.append(im)
            continue
        wmax = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
        im = rows[0]
        for r in rows[1:]:
            im = np.vstack([im, np.full((4, wmax, 3), 255, np.uint8), r])
        rows_per_frame.append(im)
    write_video(rows_per_frame, out_dir / f"{c['id']}.mp4", ref.fps)
    if t3:
        rec["map_frame"] = mf_ref
        rec["map_gt"] = [plane.pose(gmap.G[m[0]]) for m in mf_ref]
    # ---- systems
    rec["systems"] = {}
    for key in c.get("systems", a.systems):
        system, setup = key.split("|")
        if setup not in cfg["setups"] or system not in sy_cfg:
            continue
        sdir = seq_dir(data, ds_cfg, d, seq, setup)
        own = Seq(sdir) if sdir != ref.path else ref
        pub = published.get((system, setup, d, c["track"], seq))
        ent = {"label": sy_cfg[system]["label"], "setup": setup}
        if t3 and pub is not None and pub.get("status") == "ok":
            tr = next((t for t in pub.get("trials", []) if int(t.get("start", -1)) == a0), None)
            if tr is not None:
                ent.update(final_err=None if tr.get("final_err") is None else round(tr["final_err"], 3))
        elif t3 and pub is not None:
            ent["run_failed"] = True
        if not t3 and pub is not None:
            ent.update(ate=pub.get("ate_rmse"), completeness=pub.get("completeness"), failed=pub.get("failed") or pub.get("status") != "ok")
        sim3 = setup in sy_cfg[system].get("sim3_setups", [])
        tag = f"{rec['map']}__{seq}".replace("/", "_") if t3 else str(seq)
        rel = Path(d) / c["scene"] / system / setup / "s0" / ("t3" if t3 else "maps") / tag
        poses, online, src = None, None, None
        for rd in find_dir(a.runs, rel, a0 if t3 else None):
            try:
                if t3:
                    gm = Seq(seq_dir(data, ds_cfg, d, rec["map"], setup))
                    if system.startswith("cross"):
                        ti = trial_index(rd, a0, len(own.G), cfg["trial_len"], cfg["trial_stride"])
                        p = cross_t3(rd, ti) if ti is not None else None
                    elif system in ("vggt_slam", "mast3r_slam"):
                        ti = trial_index(rd, a0, len(own.G), cfg["trial_len"], cfg["trial_stride"])
                        p = vggt_t3(rd, ti, a0, min(a0 + n, len(own.G)), gm, own) if ti is not None and system == "vggt_slam" else None
                    else:
                        ti = trial_index(rd, a0, len(own.G), cfg["trial_len"], cfg["trial_stride"])
                        p = baseline_t3(rd, system, ti, a0, min(a0 + n, len(own.G)), gm, own, sim3) if ti is not None else None
                    if p is None:
                        continue
                    # the published run: its final map-relative error must match
                    b = min(a0 + n, len(own.G))      # the final pose: the latest within 1 s of the trial's end (reloc_metrics)
                    last = max((k for k in p if k < b and p[k][1] is not None and np.isfinite(p[k][1])), default=None)
                    fe = p[last][1] if last is not None and b - 1 - last <= 10 else None
                    fe = None if fe is None or not np.isfinite(fe) else fe
                    want = ent.get("final_err")
                    ok = (fe is None and want is None) or (fe is not None and want is not None and abs(fe - want) < 0.05 + 0.02 * want)
                    if poses is None or ok:
                        poses, src = p, str(rd)
                        ent["match"] = bool(ok)
                    if ok:
                        break
                else:
                    p, on = t1_poses(rd, system, own, sim3)
                    if p:
                        poses, online, src = p, on, str(rd)
                        break
            except Exception as e:      # noqa: BLE001
                print(f"  {c['id']} {key} {rd}: {e!r}")
        ent["source"] = src
        if poses is not None:
            pos, err, trk = [], [], []
            keys = sorted(poses)
            for i in frames:
                j = own.at_time(ref.t[i]) if own is not ref else i
                if t3:
                    v = poses.get(j)
                    T, e = (v if v is not None else (None, None))
                else:              # mapping trajectory: the latest pose of the last second (keyframe systems), its error
                    k = np.searchsorted(keys, j, side="right") - 1      # against the ground truth of its own frame
                    T = poses[keys[k]] if k >= 0 and j - keys[k] <= own.fps else None
                    e = float(np.linalg.norm(T[:3, 3] - own.G[keys[k]][:3, 3])) if T is not None else None
                pos.append(plane.pose(T) if T is not None else None)
                err.append(None if e is None or not np.isfinite(e) else round(float(e), 3))
                if online is not None:
                    trk.append(int(j in online))
            ent.update(pos=pos, err=err)
            if online is not None:
                ent["tracked"] = trk
        rec["systems"][key] = ent
        print(f"  {c['id']:28s} {key:26s} {'poses' if poses else 'no poses':9s} final {ent.get('final_err')} match {ent.get('match')} {src or ''}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--cases", default=str(ROOT / "benchmark/configs/failure_cases.yaml"))
    ap.add_argument("--results", default=str(ROOT / "benchmark/results/results.json"))
    ap.add_argument("--out", default=str(ROOT / "benchmark/results/failure_cases.json"))
    ap.add_argument("--only", nargs="*", default=[], help="case ids to (re)build; the others are kept from --out")
    a = ap.parse_args()
    ds_cfg = yaml.safe_load((ROOT / "benchmark/configs/datasets.yaml").read_text())
    sy_cfg = yaml.safe_load((ROOT / "benchmark/configs/systems.yaml").read_text())
    spec = yaml.safe_load(Path(a.cases).read_text())
    a.systems = spec["systems"]
    published = {}
    for r in json.loads(Path(a.results).read_text())["results"]:
        if r.get("odom") or r.get("seed", 0) != 0:
            continue
        seq = r.get("query") if r["track"] == "t3" else r.get("sequence")
        published[(r["system"], r["setup"], r["dataset"], r["track"], seq)] = r
    out_dir = ROOT / "benchmark/site/assets/failures"
    out_dir.mkdir(parents=True, exist_ok=True)
    old = {}
    if Path(a.out).is_file():
        old = {c["id"]: c for c in json.loads(Path(a.out).read_text()).get("cases", [])}
    cases = []
    for c in spec["cases"]:
        if a.only and c["id"] not in a.only:
            if c["id"] in old:
                cases.append(old[c["id"]])
            continue
        print(c["id"])
        cases.append(run_case(c, a, ds_cfg, sy_cfg, published, out_dir))
    Path(a.out).write_text(json.dumps({"cases": cases}, separators=(",", ":")))
    print(f"wrote {a.out} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
