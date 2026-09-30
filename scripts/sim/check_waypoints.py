#!/usr/bin/env python3
"""Check trajectory waypoints against an occupancy grid: clearance, connectivity (within annotated regions) and a
suggested nearby point with >= min clearance.   python scripts/sim/check_waypoints.py --occ occ.npz --layout layout.json --radius 0.3 --wp "x:y,x:y,..."
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--occ", required=True); ap.add_argument("--layout", default=None); ap.add_argument("--radius", type=float, default=0.3)
    ap.add_argument("--min-clear", type=float, default=0.45); ap.add_argument("--wp", required=True)
    a = ap.parse_args()
    occ = np.load(a.occ); grid = occ["grid"]; x0, y0, res = float(occ["x0"]), float(occ["y0"]), float(occ["res"])
    dist = cv2.distanceTransform((grid == 0).astype(np.uint8), cv2.DIST_L2, 5) * res
    free = dist >= a.radius
    if a.layout:
        lay = json.loads(Path(a.layout).read_text()); m = np.zeros(grid.shape, np.uint8)
        for r in lay.get("regions", []):
            if r.get("poly") and len(r["poly"]) >= 3:
                pts = np.array([[(p[0] - x0) / res, (p[1] - y0) / res] for p in r["poly"]], np.int32).reshape(-1, 1, 2); cv2.fillPoly(m, [pts], 1)
        k = int(round(0.4 / res)) * 2 + 1; free &= cv2.dilate(m, np.ones((k, k), np.uint8)) > 0
    n, lab = cv2.connectedComponents(free.astype(np.uint8), connectivity=8)
    WP = [tuple(map(float, w.split(":"))) for w in a.wp.split(",")]
    good = (dist >= a.min_clear) & free; ys, xs = np.nonzero(good)
    labels = []
    for (x, y) in WP:
        ix, iy = int(round((x - x0) / res)), int(round((y - y0) / res))
        kk = np.argmin((ys - iy) ** 2 + (xs - ix) ** 2); sx, sy = x0 + xs[kk] * res, y0 + ys[kk] * res
        l = int(lab[iy, ix]) if free[iy, ix] else int(lab[ys[kk], xs[kk]])
        labels.append(l)
        print(f"({x:6.1f},{y:6.1f}) clear {dist[iy, ix]:.2f} comp {l:>3}  -> nearest with >= {a.min_clear} m: ({sx:6.2f},{sy:6.2f}) d={np.hypot(sx - x, sy - y):.2f}")
    print("components used:", sorted(set(labels)), "(all equal = connected)")
    print("suggested:", ",".join(f"{x0 + xs[np.argmin((ys - int(round((y - y0) / res))) ** 2 + (xs - int(round((x - x0) / res))) ** 2)] * res:.2f}:{y0 + ys[np.argmin((ys - int(round((y - y0) / res))) ** 2 + (xs - int(round((x - x0) / res))) ** 2)] * res:.2f}" for (x, y) in WP))


if __name__ == "__main__":
    main()
