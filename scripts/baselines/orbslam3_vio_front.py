#!/usr/bin/env python3
"""ORB-SLAM3 monocular-inertial (or monocular) as a front end on one sequence: one session, no map reuse, metric
trajectory vs ground truth.  The same metrics as the CROSS mono front-end check (SE(3) ATE, the scale a Sim(3)
alignment would apply to the estimate, the median length ratio of 5 s windows), on the per-frame (online) poses of tracked frames, plus
completeness, lost events and the number of maps (resets).

  python scripts/baselines/orbslam3_vio_front.py <seq_dir> <out_dir> [--sensor imu_mono|mono] [--max-frames N]
      [--imu-time-offset S] [--imu-file imu.txt] [--imu-noise-scale k] [--bin <bundle dir>] [--voc ORBvoc.txt]

<bundle dir>: bin/orbslam3_reloc (+ lib/ and ld-linux-x86-64.so.2 for hosts with other system libraries).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

import run_baselines as rb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq")
    ap.add_argument("out")
    ap.add_argument("--sensor", default="imu_mono", choices=["imu_mono", "mono"])
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--imu-time-offset", type=float, default=0.0, help="camera clock = IMU clock + S")
    ap.add_argument("--imu-file", default=None, help="imu.txt (default: <seq>/imu.txt)")
    ap.add_argument("--imu-noise-scale", type=float, default=1.0)
    ap.add_argument("--bin", default=str(rb.BIN))
    ap.add_argument("--voc", default=str(rb.ORB_VOC))
    ap.add_argument("--imu-upsample", type=float, default=None,
                    help="linearly interpolate the IMU to this rate (Hz) first (KITTI's OXTS runs at the camera's 10 Hz)")
    ap.add_argument("--timeout", type=float, default=7200)
    ap.add_argument("--eval-only", action="store_true", help="evaluate the poses of an earlier run in <out_dir>")
    ap.add_argument("--eval-frames", type=int, default=None, help="evaluate the first N frames only")
    a = ap.parse_args()
    if a.eval_only:
        out = Path(a.out).resolve()
        res = json.loads((out / "result.json").read_text())
        gt = np.loadtxt(Path(a.seq) / "poses_left.txt").reshape(-1, 4, 4)
        fps = float(json.loads((Path(a.seq) / "calib.json").read_text()).get("fps", 10.0))
        ev = evaluate(out / "poses.txt", gt, fps, res["sensor"] == "imu_mono", a.eval_frames)
        print(json.dumps({"frames": a.eval_frames, **ev}, indent=1))
        return

    seq, out = Path(a.seq).resolve(), Path(a.out).resolve()      # the binary runs in out/
    out.mkdir(parents=True, exist_ok=True)
    calib = json.loads((seq / "calib.json").read_text())
    fps = float(calib.get("fps", 10.0))
    n = rb.n_frames(seq) if a.max_frames is None else min(a.max_frames, rb.n_frames(seq))
    view = rb.make_chunk(seq, out, 0, n)
    imu_cal = json.loads((seq / "imu.json").read_text()) if (seq / "imu.json").is_file() else {}
    tf = seq / imu_cal.get("frame_times", "times.txt")
    times = np.loadtxt(tf).reshape(-1)[:n] if tf.is_file() else np.arange(n) / fps   # no times file: frame i at i / fps
    np.savetxt(view / "frame_times.txt", times, fmt="%.6f")
    rb.orb_yaml(seq, out / "settings.yaml", fps=fps, imu=a.sensor == "imu_mono", imu_noise_scale=a.imu_noise_scale)

    b = Path(a.bin)
    exe = b / "bin" / "orbslam3_reloc" if (b / "bin" / "orbslam3_reloc").is_file() else b / "orbslam3_reloc"
    cmd = [str(exe)]
    if (b / "ld-linux-x86-64.so.2").is_file():
        cmd = [str(b / "ld-linux-x86-64.so.2"), "--library-path", str(b / "lib"), str(exe)]
    cmd += [a.voc, str(out / "settings.yaml"), str(view), str(out / "poses.txt"), "--sensor", a.sensor,
            "--fps", str(fps), "--times", str(view / "frame_times.txt")]
    if a.sensor == "imu_mono":
        imu_file = Path(a.imu_file or seq / "imu.txt")
        if a.imu_upsample:
            d = np.loadtxt(imu_file)
            tu = np.arange(d[0, 0], d[-1, 0], 1.0 / a.imu_upsample)
            up = np.column_stack([tu] + [np.interp(tu, d[:, 0], d[:, j]) for j in range(1, 7)])
            imu_file = out / "imu_upsampled.txt"
            np.savetxt(imu_file, up, fmt="%.6f")
        cmd += ["--imu", str(imu_file), "--imu-time-offset", str(a.imu_time_offset)]
    t0 = time.time()
    with open(out / "orbslam3.log", "w") as log:
        try:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=a.timeout, cwd=str(out)).returncode
        except subprocess.TimeoutExpired:
            rc = -9
    seconds = time.time() - t0

    res = {"seq": str(seq), "sensor": a.sensor, "frames": n, "rc": rc, "seconds": seconds,
           "imu_time_offset": a.imu_time_offset, "imu_noise_scale": a.imu_noise_scale, "imu_upsample": a.imu_upsample}
    res.update(evaluate(out / "poses.txt", np.loadtxt(seq / "poses_left.txt").reshape(-1, 4, 4)[:n], fps,
                        a.sensor == "imu_mono"))
    nm = out / "poses.txt.final_nmaps"
    res["maps"] = int(nm.read_text().split()[0]) if nm.is_file() else None
    log_text = (out / "orbslam3.log").read_text(errors="replace")
    res["resets"] = log_text.count("Reseting active map") + log_text.count("Resetting active map")
    res["viba1"], res["viba2"] = log_text.count("end VIBA 1"), log_text.count("end VIBA 2")
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def _align(src, dst):
    """Umeyama: R, t (SE(3)) and the scale a Sim(3) alignment applies to src."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    H = (src - mu_s).T @ (dst - mu_d) / len(src)
    U, S, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    var_s = ((src - mu_s) ** 2).sum() / len(src)
    scale = float(np.trace(np.diag(S) @ D) / var_s) if var_s > 0 else float("nan")
    return R, mu_d - R @ mu_s, scale, mu_d - scale * R @ mu_s


def evaluate(poses_file: Path, gt: np.ndarray, fps: float, inertial: bool, n_eval=None) -> dict:
    """Online per-frame poses vs ground truth.  A metric frame is tracked (state 2 = OK) and, with an IMU, in a map
    whose IMU is initialized.  The trajectory is cut into segments at (re)initializations and new maps (each its own
    frame); each segment with >= 10 metric frames is aligned to the ground truth on its own (SE(3), and Sim(3))."""
    if not poses_file.is_file():
        return {"tracked": 0}
    rows = np.loadtxt(poses_file, ndmin=2)
    if n_eval is not None:
        rows = rows[rows[:, 0] < n_eval]
    n = len(rows)
    idx, state = rows[:, 0].astype(int), rows[:, 1].astype(int)
    T = rows[:, 2:18].reshape(-1, 4, 4)
    nmaps = rows[:, 18].astype(int) if rows.shape[1] > 18 else np.ones(n, int)
    imu_ok = rows[:, 19] > 0 if rows.shape[1] > 19 else np.zeros(n, bool)
    map_id = rows[:, 20].astype(int) if rows.shape[1] > 20 else np.zeros(n, int)
    ok = state == 2
    metric = ok & imu_ok if inertial else ok
    seg_id = np.cumsum(np.r_[0, ((state[1:] <= 1) & (state[:-1] > 1)) | (nmaps[1:] != nmaps[:-1])
                             | (map_id[1:] != map_id[:-1])])
    out = {"tracked": float(ok.mean()), "metric": float(metric.mean()),
           "lost_events": int(np.sum((state[1:] >= 3) & (state[1:] <= 4) & (state[:-1] == 2))),
           "first_metric": int(idx[metric][0]) if metric.any() else None}
    w = max(int(round(5 * fps)), 2)
    segs, ratios = [], []
    for sid in np.unique(seg_id[metric]):
        m = metric & (seg_id == sid)
        if m.sum() < 10:
            continue
        src, dst = T[m, :3, 3], gt[idx[m], :3, 3]
        R, t, scale, t2 = _align(src, dst)
        segs.append({"frames": int(m.sum()), "first": int(idx[m][0]),
                     "ate_se3": float(np.sqrt(((src @ R.T + t - dst) ** 2).sum(1).mean())),
                     "sim3_scale": scale,
                     "ate_sim3": float(np.sqrt(((scale * src @ R.T + t2 - dst) ** 2).sum(1).mean()))})
        f = idx[m]
        for k in range(0, len(f) - w + 1, w // 2):        # length ratio (estimate / truth) of 5 s windows
            if f[k + w - 1] - f[k] == w - 1:
                lg = np.linalg.norm(np.diff(dst[k:k + w], axis=0), axis=1).sum()
                if lg > 0.5:
                    ratios.append(np.linalg.norm(np.diff(src[k:k + w], axis=0), axis=1).sum() / lg)
    out["segments"] = len(segs)
    if segs:
        big = max(segs, key=lambda x: x["frames"])
        nf = sum(x["frames"] for x in segs)
        out.update(largest_frames=big["frames"], largest_ate_se3=big["ate_se3"], largest_sim3_scale=big["sim3_scale"],
                   largest_ate_sim3=big["ate_sim3"],
                   ate_se3_weighted=sum(x["frames"] * x["ate_se3"] for x in segs) / nf,
                   segment_scale_median=float(np.median(ratios)) if ratios else None,
                   segment_scale_p10_p90=[float(np.percentile(ratios, 10)), float(np.percentile(ratios, 90))] if ratios else None,
                   segment_list=segs)
    out["path_m"] = float(np.linalg.norm(np.diff(gt[idx, :3, 3], axis=0), axis=1).sum())
    return out


if __name__ == "__main__":
    main()
