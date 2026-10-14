"""Stereo depth scale of a prepared two-iPhone stereo folder against phone B's LiDAR: tests the rig calibration
(baseline length, rotation) and the pairing of scripts/rig/prepare_iphone_stereo.py.

For sampled prepared frames, phone B's LiDAR depth (high-confidence pixels, 0.3 m to --max-range) is back-projected
with that frame's ARKit intrinsics, rotated into the rectified left camera (the rotation between B's ARKit pose and
the prepared left pose, so the upright turn and the pair's rectification are both included) and projected with the
folder's K; the stereo (SGBM) depth of the pair is read at the same pixels.  ratio = stereo / LiDAR: its median is the
scale error of the stereo depth; a fit of the measured against the LiDAR disparity gives a disparity scale and offset
(an offset in px is a rotation error about the vertical axis, a scale a baseline error).

The B frame of each prepared pair comes from pairs.txt (written by the converter) or, for folders without it, from the
converter's pairing redone with calib.json's clock offset.

  python scripts/rig/check_stereo_scale.py --prepared data/walk1 --b rec_B [--a rec_A] [--n 40] [--json out.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cross.dataloader.stereo_loader import StereoSequenceLoader  # noqa: E402
from cross.rig.recording import Recording  # noqa: E402
from cross.rig.selfcalib import pair_frames  # noqa: E402


def b_frames(prepared: Path, recB: Recording, a_spec):
    """Index (into recB) of the B frame of every prepared pair."""
    p = prepared / "pairs.txt"
    if p.is_file():
        ids = np.loadtxt(p, ndmin=2)[:, 1].astype(int)
        pos = {int(f): i for i, f in enumerate(recB.frames.frame_id.to_numpy())}
        return np.array([pos[f] for f in ids])
    if a_spec is None:
        sys.exit(f"{prepared} has no pairs.txt: pass --a to redo the pairing")
    cal = json.loads((prepared / "calib.json").read_text())
    iB, _, _ = pair_frames(Recording.load(a_spec), recB, cal["clock_offset_ms"] / 1000.0, max_dt=0.06)
    t = np.loadtxt(prepared / "times.txt")
    if len(iB) != len(t) or not np.allclose(recB.t[iB] - recB.t[iB[0]], t, atol=1e-5):
        sys.exit(f"{prepared}: the redone pairing does not match times.txt (other --max-dt?)")
    return iB


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--b", required=True, help="recording of phone B (left), as given to the converter")
    ap.add_argument("--a", help="recording of phone A: only for folders without pairs.txt")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max-range", type=float, default=5.0)
    ap.add_argument("--json", type=Path, help="write the per-frame numbers")
    a = ap.parse_args()
    recB = Recording.load(a.b)
    iB = b_frames(a.prepared, recB, a.a)
    poses = np.loadtxt(a.prepared / "poses_left.txt").reshape(-1, 4, 4)
    ds = StereoSequenceLoader(str(a.prepared), depth_source="sgbm", snr=None, depth_cache=False)
    assert len(ds) == len(iB) == len(poses)
    K = np.asarray(json.loads((a.prepared / "calib.json").read_text())["K"])
    fB = K[0, 0] * ds.baseline
    out = []
    for k in np.linspace(0, len(ds) - 1, a.n).astype(int):
        i = int(iB[k])
        dl = recB.depth(i)
        if dl is None:
            continue
        d, (sx, sy) = dl
        Kb = recB.K(i)
        v, u = np.mgrid[0:d.shape[0], 0:d.shape[1]]
        m = np.isfinite(d) & (d > 0.3) & (d < a.max_range)
        uu, vv, zz = (u[m] + 0.5) * sx - 0.5, (v[m] + 0.5) * sy - 0.5, d[m].astype(np.float64)
        X = np.stack([(uu - Kb[0, 2]) / Kb[0, 0] * zz, (vv - Kb[1, 2]) / Kb[1, 1] * zz, zz], 0)  # raw B camera
        M = np.linalg.inv(recB.T[i]) @ poses[k]                    # rectified left camera in raw B camera
        Xr = M[:3, :3].T @ (X - M[:3, 3:4])
        p = K @ Xr
        pu, pv = p[0] / p[2], p[1] / p[2]
        sd = ds[int(k)]["depth"]
        H, W = sd.shape
        ok = (Xr[2] > 0) & (pu >= 0) & (pu < W - 1) & (pv >= 0) & (pv < H - 1)
        s = sd[np.round(pv[ok]).astype(int), np.round(pu[ok]).astype(int)].astype(np.float64)
        zl = Xr[2][ok]
        good = np.isfinite(s) & (s > 0.3)
        if good.sum() < 200:
            continue
        ratio = s[good] / zl[good]
        disp_meas, disp_true = fB / s[good], fB / zl[good]
        (sc, off), *_ = np.linalg.lstsq(np.c_[disp_true, np.ones_like(disp_true)], disp_meas, rcond=None)
        out.append({"frame": int(k), "n": int(good.sum()), "ratio_median": float(np.median(ratio)),
                    "abs_rel_err_median": float(np.median(np.abs(ratio - 1))), "disp_scale": float(sc),
                    "disp_offset_px": float(off)})
    if not out:
        sys.exit(f"{a.prepared}: no frame with enough LiDAR / stereo overlap")
    rs = np.array([o["ratio_median"] for o in out])
    q = lambda key, p: round(float(np.percentile([o[key] for o in out], p)), 4)  # noqa: E731
    summ = {"session": a.prepared.name, "baseline_m": round(ds.baseline, 4), "frames": len(out),
            "stereo_over_lidar_median": round(float(np.median(rs)), 4), "p10_p90": [q("ratio_median", 10), q("ratio_median", 90)],
            "abs_rel_err_median": q("abs_rel_err_median", 50),
            "disp_scale_median": q("disp_scale", 50), "disp_offset_px_median": q("disp_offset_px", 50),
            "disp_offset_px_p10_p90": [q("disp_offset_px", 10), q("disp_offset_px", 90)]}
    print(json.dumps(summ))
    if a.json:
        a.json.write_text(json.dumps({"summary": summ, "frames": out}, indent=1))


if __name__ == "__main__":
    main()
