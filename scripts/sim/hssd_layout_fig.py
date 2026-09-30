#!/usr/bin/env python3
"""Top-down layout figure of an HSSD scene: object footprints coloured by mobility class, room regions, optional
trajectory and rearrangement plan (arrows from old to new position, crosses for removed objects).

    python scripts/sim/hssd_layout_fig.py --layout data/sim/assets/hssd_103997940.blend.layout.json --out fig.png \
        [--occ temp/occ_hssd.npz] [--path poses.npy|poses_left.txt] [--plan plan.json] [--topdown topdown.png]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hssd_classes import mobility_class, CLASS_COLOR  # noqa: E402


def load_path(p):
    p = Path(p)
    if p.suffix == ".npy":
        return np.load(p)
    P = np.loadtxt(p)
    if P.ndim == 2 and P.shape[1] == 16:
        return P.reshape(-1, 4, 4)[:, :2, 3]
    return P[:, :2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--occ", default=None)
    ap.add_argument("--path", default=None)
    ap.add_argument("--plan", default=None)
    ap.add_argument("--topdown", default=None, help="rendered top-down png to use as background (same extent as stage bbox)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()
    lay = json.loads(Path(args.layout).read_text())
    (x0, y0, _), (x1, y1, _) = lay["stage_bbox"]
    fig, ax = plt.subplots(figsize=(14, 14 * (y1 - y0) / max(x1 - x0, 1e-6)))
    if args.topdown:
        img = plt.imread(args.topdown)
        s = max(x1 - x0, y1 - y0) * 1.04
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ax.imshow(img, extent=[cx - s / 2, cx + s / 2, cy - s / 2, cy + s / 2], alpha=0.55, zorder=0)
    if args.occ:
        occ = np.load(args.occ)
        g = occ["grid"]; res = float(occ["res"]); ox, oy = float(occ["x0"]), float(occ["y0"])
        ax.imshow(np.where(g > 0, 0.25, np.nan), cmap="gray", vmin=0, vmax=1, origin="lower",
                  extent=[ox, ox + g.shape[1] * res, oy, oy + g.shape[0] * res], zorder=1, alpha=0.8)
    for r in lay.get("regions", []):
        poly = np.array(r["poly"])[:, :2] if r["poly"] else None
        if poly is None or len(poly) < 3:
            continue
        ax.add_patch(Polygon(poly, closed=True, fill=False, ec="#4a4a4a", lw=0.8, ls="--", zorder=2))
        c = poly.mean(0)
        ax.text(c[0], c[1], r["name"], fontsize=6, color="#333", ha="center", va="center", zorder=6)
    counts = {}
    for o in lay["objects"]:
        cls = mobility_class(o["category"], o.get("maxdim", -1), o.get("super", ""), o["bbox_min"][2])
        counts[cls] = counts.get(cls, 0) + 1
        mn, mx = o["bbox_min"], o["bbox_max"]
        if mn[2] > 2.6:            # ceiling-mounted: outline only
            continue
        ax.add_patch(Rectangle((mn[0], mn[1]), mx[0] - mn[0], mx[1] - mn[1], fc=CLASS_COLOR[cls], ec="none",
                               alpha=0.35 if cls == "structural" else 0.75, zorder=3))
    if args.path:
        P = load_path(args.path)
        ax.plot(P[:, 0], P[:, 1], "-", color="#111", lw=1.6, zorder=7)
        ax.plot(P[0, 0], P[0, 1], "o", color="#1b9e77", ms=8, zorder=8)
        ax.plot(P[-1, 0], P[-1, 1], "s", color="#d95f02", ms=7, zorder=8)
        for k in range(0, len(P), 100):
            ax.text(P[k, 0], P[k, 1], str(k), fontsize=6, color="#111", zorder=9)
    if args.plan:
        plan = json.loads(Path(args.plan).read_text())
        for ch in plan["changes"]:
            a = ch["from_xy"]
            if ch["op"] in ("remove", "follow_remove"):
                ax.plot(a[0], a[1], "x", color="k", ms=6, mew=1.5, zorder=9)
            elif ch["op"] in ("relocate", "jitter", "swap", "follow"):
                b = ch["to_xy"]
                col = {"relocate": "k", "jitter": "#1b9e77", "swap": "#e7298a", "follow": "#666"}[ch["op"]]
                ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color=col, lw=1.0), zorder=9)
    ax.set_xlim(x0 - 1, x1 + 1); ax.set_ylim(y0 - 1, y1 + 1); ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    handles = [Rectangle((0, 0), 1, 1, fc=CLASS_COLOR[c], alpha=0.75) for c in CLASS_COLOR]
    ax.legend(handles, [f"{c} ({counts.get(c, 0)})" for c in CLASS_COLOR], loc="upper right", fontsize=8, title="mobility class")
    ax.set_title(args.title or f"{lay['scene']}: {len(lay['objects'])} objects", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi)
    print("saved", args.out, counts)


if __name__ == "__main__":
    main()
