"""Compass heading on NCLT (prepared sessions: imu.txt with the magnetometer, gnss.txt, gt_body.txt): error of the
tilt-compensated magnetometer heading against the ground-truth yaw, outdoors (a GPS fix within 1 s) and indoors (no
fix), for cross.geo.compass.Compass with / without disturbance detection and hard-iron calibration.  The heading
offset (declination + mounting) is calibrated online as in the system: against the reference heading while GPS is
good (here the ground-truth yaw stands in for the GNSS-anchored map's heading).

  python scripts/geo/compass_study.py --prepared /data/nclt/prepared --sessions 2012-01-08 --out outputs/compass
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cross.geo.compass import Compass, CompassConfig, wrap  # noqa: E402


def quat_to_R(q):
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - z * w); R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w); R[:, 2, 1] = 2 * (y * z + x * w); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def run(d: Path, variant: str, step: int):
    imu = np.loadtxt(d / "imu.txt")[::step]
    gt = np.loadtxt(d / "gt_body.txt")
    g = np.loadtxt(d / "gnss.txt")
    t = imu[:, 0]
    ok = (t > gt[0, 0]) & (t < gt[-1, 0])
    imu, t = imu[ok], t[ok]
    j = np.clip(np.searchsorted(gt[:, 0], t), 0, len(gt) - 1)
    R = quat_to_R(gt[j, 4:8])                          # body (x fwd, y right, z down) -> NED
    yaw_ned = np.arctan2(R[:, 1, 0], R[:, 0, 0])       # heading from north, clockwise
    yaw_enu = wrap(np.pi / 2 - yaw_ned)
    k = np.clip(np.searchsorted(g[:, 0], t), 1, len(g) - 1)
    gap = np.minimum(np.abs(g[k, 0] - t), np.abs(g[k - 1, 0] - t))
    fix = gap <= 1.0
    c = Compass(CompassConfig(min_offset_samples=30, hard_iron=(variant == "full")))
    if variant == "raw":
        c.disturbed = lambda *a, **kw: False
    if variant in ("raw", "no_hard_iron"):
        c._add_hard_iron_sample = lambda *a, **kw: None
    err = np.full(len(t), np.nan)
    flagged = np.zeros(len(t), bool)
    for i in range(len(t)):
        h = c.heading({"mag": imu[i, 1:4], "accel": imu[i, 4:7]}, outdoor_ok=bool(fix[i]))
        if h is None:
            flagged[i] = True
            continue
        if fix[i]:
            c.add_offset_sample(h[0], yaw_enu[i])
        cam = c.camera_yaw(h[0])
        if cam is not None:
            err[i] = abs(float(wrap(cam - yaw_enu[i])))
    out = {"n": int(len(t)), "frac_fix": float(fix.mean()), "offset_deg": None if c.offset is None else math.degrees(c.offset),
           "hard_iron": None if c.hard_iron is None else [float(x) for x in c.hard_iron]}
    for lab, m in (("fix", fix), ("no_fix", ~fix)):
        e = np.degrees(err[m & np.isfinite(err)])
        out[lab] = {"frac_flagged": float(flagged[m].mean()) if m.sum() else None,
                    "p50": float(np.percentile(e, 50)) if len(e) else None,
                    "p90": float(np.percentile(e, 90)) if len(e) else None, "n": int(len(e))}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--step", type=int, default=5, help="use every step-th IMU sample (~10 Hz)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for s in a.sessions:
        res = {v: run(a.prepared / s, v, a.step) for v in ("raw", "no_hard_iron", "full")}
        (a.out / f"compass_{s}.json").write_text(json.dumps(res, indent=1))
        print(s, " | ".join(f"{v}: fix p50/p90 {r['fix']['p50']:.1f}/{r['fix']['p90']:.1f} flag {r['fix']['frac_flagged']:.2f}; "
                            f"no-fix p50/p90 {(r['no_fix']['p50'] or float('nan')):.1f}/{(r['no_fix']['p90'] or float('nan')):.1f} "
                            f"flag {(r['no_fix']['frac_flagged'] or 0):.2f}" for v, r in res.items()), flush=True)


if __name__ == "__main__":
    main()
