"""Run scripts/map_and_reloc.py with the cross_nav BEV head on and record, per pass, where the BEV's gravity came from
(IMU frontend = "given", else the gravity head) and the angle between the IMU's and the head's up vectors.

    python scripts/nav_gravity_check.py <stats.json> -- <map_and_reloc.py arguments>
"""
import json
import runpy
import sys

import numpy as np

from cross.cv import pose_est_ff

out_path, args = sys.argv[1], sys.argv[sys.argv.index("--") + 1:]
stats = []
orig = pose_est_ff.PoseEstFeedForward._predict_bev


def wrapped(self, metric):
    r = orig(self, metric)
    if r is not None:
        stats.append({"source": r["gravity_source"], "head_deg": r["gravity_head_deg"],
                      "log_scale": r["log_scale"], "occ_frac": float((r["prob"] > 0.5).mean())})
    return r


pose_est_ff.PoseEstFeedForward._predict_bev = wrapped
sys.argv = ["scripts/map_and_reloc.py"] + args
try:
    runpy.run_path("scripts/map_and_reloc.py", run_name="__main__")
finally:
    g = [s["head_deg"] for s in stats if s["source"] == "given"]
    summary = {"passes": len(stats), "given": len(g), "head": len(stats) - len(g),
               "imu_vs_head_deg_median": float(np.median(g)) if g else None,
               "imu_vs_head_deg_p90": float(np.percentile(g, 90)) if g else None}
    json.dump({"summary": summary, "passes": stats}, open(out_path, "w"))
    print("NAV_GRAVITY", json.dumps(summary))
