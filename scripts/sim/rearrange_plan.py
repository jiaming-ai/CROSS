#!/usr/bin/env python3
"""Plan a quantified object rearrangement of an HSSD scene.

Pool = objects of the non-structural mobility classes that lie in rooms visited by the trajectory (or within
`near` metres of it).  A level L in [0, 1] changes round(L * |pool|) objects, sampled without replacement with
class weights (clutter > light > decor > heavy).  Every changed object gets one operation:
    relocate  new free spot in the same room (floor-standing) or on the same support surface (supported), falling back
              to a neighbouring visited room for floor-standing objects
    jitter    small push (0.1-0.5 m) and rotation (<= 20 deg) in place
    remove    object taken away
    swap      exchange position and heading with another object of the same class and support type (different model)
Objects standing on a moved / removed object follow it ("follow" changes).  Objects that rest on a surface of the
static stage (kitchen counters, built-in shelves) can only be pushed slightly, removed or swapped.  The trajectory
corridor (`clearance` m) is kept free.  Output: plan JSON consumed by gen_simchange.py and hssd_layout_fig.py.

    python scripts/sim/rearrange_plan.py --layout <blend>.layout.json --occ temp/occ_hssd.npz --path path_xy.npy \
        --level 0.5 --seed 0 --out plans/rearr_50_s0.json
"""
from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path
import numpy as np


class Poly:
    """Minimal polygon (ray casting) so that the planner runs inside Blender's Python without matplotlib."""

    def __init__(self, vertices):
        self.vertices = np.asarray(vertices, dtype=np.float64)

    def contains_point(self, p):
        x, y = float(p[0]), float(p[1])
        v = self.vertices
        inside = False
        j = len(v) - 1
        for i in range(len(v)):
            xi, yi = v[i]; xj, yj = v[j]
            if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi:
                inside = not inside
            j = i
        return inside

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hssd_classes import mobility_class, CLASS_WEIGHT, CLASS_OPS  # noqa: E402

YAW_CHOICES = [0.0, math.pi / 2, math.pi, -math.pi / 2]


class Occ:
    def __init__(self, npz):
        d = np.load(npz)
        self.grid = d["grid"] > 0
        self.x0, self.y0, self.res = float(d["x0"]), float(d["y0"]), float(d["res"])
        self.H, self.W = self.grid.shape

    def cells(self, xmin, ymin, xmax, ymax):
        ix0 = int(math.floor((xmin - self.x0) / self.res)); ix1 = int(math.ceil((xmax - self.x0) / self.res))
        iy0 = int(math.floor((ymin - self.y0) / self.res)); iy1 = int(math.ceil((ymax - self.y0) / self.res))
        inside = ix0 >= 0 and iy0 >= 0 and ix1 < self.W and iy1 < self.H
        return max(ix0, 0), max(iy0, 0), min(ix1, self.W - 1), min(iy1, self.H - 1), inside


def rect_rotated_extent(rect, cxy, dyaw):
    """Axis-aligned extent of a rectangle rotated by dyaw about its centre and moved to cxy."""
    w, h = rect[2] - rect[0], rect[3] - rect[1]
    c, s = abs(math.cos(dyaw)), abs(math.sin(dyaw))
    W, H = w * c + h * s, w * s + h * c
    return np.array([cxy[0] - W / 2, cxy[1] - H / 2, cxy[0] + W / 2, cxy[1] + H / 2])


def overlap(a, b, margin=0.0):
    return not (a[2] + margin <= b[0] or b[2] + margin <= a[0] or a[3] + margin <= b[1] or b[3] + margin <= a[1])


def overlap_area(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def penetration(rect, px, py):
    """Deepest intrusion (m) of the occupied cells (px, py) into the rectangle (0 if none inside)."""
    if len(px) == 0:
        return 0.0
    d = np.minimum.reduce([px - rect[0], rect[2] - px, py - rect[1], rect[3] - py])
    return float(max(0.0, d.max()))


def rigid_xy(p, pivot, dyaw, d):
    """Apply rotation dyaw about pivot followed by translation d to point p (all xy)."""
    c, s = math.cos(dyaw), math.sin(dyaw)
    x, y = p[0] - pivot[0], p[1] - pivot[1]
    return np.array([pivot[0] + c * x - s * y + d[0], pivot[1] + s * x + c * y + d[1]])


UNDER_TABLE = {"chair", "stool", "bench", "ottoman", "armchair", "bar_stool", "counter_stool"}
TABLES = {"table", "desk", "dining_table", "coffee_table", "side_table", "end_table", "nightstand", "counter", "kitchen_island", "dining_area"}


def make_plan(layout: dict, occ_npz: str, path_xy, level: float, seed: int = 0, near: float = 3.0, clearance: float = 0.45,
              max_height: float = 2.3, ops: str | None = None, occ_low_npz: str | None = None) -> dict:
    rng = np.random.default_rng(seed)
    lay = json.loads(json.dumps(layout))
    occ = Occ(occ_low_npz) if occ_low_npz else Occ(occ_npz)      # leg-level slice: chairs may slide under table tops
    P = np.asarray(path_xy, dtype=np.float64)[:, :2]
    objs = lay["objects"]
    for o in objs:
        o["cls"] = mobility_class(o["category"], o.get("maxdim", -1), o.get("super", ""), o["bbox_min"][2])
        mn, mx = o["bbox_min"], o["bbox_max"]
        o["rect"] = np.array([mn[0], mn[1], mx[0], mx[1]], dtype=np.float64)
        o["cxy"] = np.array([(mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2])
        o["supported"] = mn[2] > 0.15
        o["z0"] = mn[2]
    regions = [(r["name"], Poly(np.array(r["poly"])[:, :2])) for r in lay.get("regions", []) if r.get("poly") and len(r["poly"]) >= 3]
    room_centroid = {n: mp.vertices.mean(0) for n, mp in regions}

    def region_of(xy):
        for name, mp in regions:
            if mp.contains_point(xy):
                return name
        return None

    visited = set(filter(None, (region_of(p) for p in P[::5])))
    for o in objs:
        o["room"] = region_of(o["cxy"])
        o["dpath"] = float(np.min(np.linalg.norm(P - o["cxy"][None, :], axis=1)))
    pool = [o for o in objs if o["cls"] != "structural" and o["z0"] < max_height and (o["room"] in visited or o["dpath"] <= near)]
    pool_idx = {o["index"] for o in pool}
    # support relations (parent = object whose top matches the child's bottom and whose footprint contains the child's centre)
    tops = [o for o in objs if o["cls"] in ("heavy", "light", "structural") and not o["supported"]]
    for o in objs:
        o["support"] = None
    for o in pool:
        if not o["supported"]:
            continue
        best = None
        for t in tops:
            if t is o or abs(t["bbox_max"][2] - o["z0"]) > 0.08:
                continue
            r = t["rect"]
            if r[0] - 0.03 <= o["cxy"][0] <= r[2] + 0.03 and r[1] - 0.03 <= o["cxy"][1] <= r[3] + 0.03:
                area = (r[2] - r[0]) * (r[3] - r[1])
                if best is None or area < best[0]:
                    best = (area, t)
        o["support"] = best[1] if best else None
        o["floating"] = best is None            # rests on the static stage (counter, built-in shelf) or unknown support
    children = {}
    for o in pool:
        if o.get("support") is not None:
            children.setdefault(o["support"]["index"], []).append(o)
    by_index = {o["index"]: o for o in objs}

    n_target = int(round(level * len(pool)))
    w = np.array([CLASS_WEIGHT[o["cls"]] for o in pool])
    keys = np.log(np.maximum(w, 1e-9)) + rng.gumbel(size=len(pool))
    chosen = [pool[i] for i in np.argsort(-keys)[:n_target]]
    # parents first so that children follow before they are (possibly) changed themselves
    chosen.sort(key=lambda o: (o["supported"], o["cls"] != "heavy"))

    floor_rects = {o["index"]: o["rect"].copy() for o in objs if not o["supported"] and o["z0"] < max_height}
    for o in objs:
        o["orig_rect"] = o["rect"].copy()
        r = o["rect"]
        ix0, iy0, ix1, iy1, _ = occ.cells(r[0] - 0.05, r[1] - 0.05, r[2] + 0.05, r[3] + 0.05)
        sub = occ.grid[iy0:iy1 + 1, ix0:ix1 + 1]
        if sub.any():
            ys, xs = np.nonzero(sub)
            px = occ.x0 + (xs + ix0) * occ.res; py = occ.y0 + (ys + iy0) * occ.res
            inside_own = (px >= r[0] - 0.06) & (px <= r[2] + 0.06) & (py >= r[1] - 0.06) & (py <= r[3] + 0.06)
            # own legs are inside the own rect too; the original penetration therefore counts every occupied cell inside
            o["orig_penetration"] = penetration(r, px, py)
        else:
            o["orig_penetration"] = 0.0
    removed, done = set(), set()

    def path_clear(rect):
        cx = np.clip(P[:, 0], rect[0], rect[2]); cy = np.clip(P[:, 1], rect[1], rect[3])
        return bool(np.all(np.hypot(P[:, 0] - cx, P[:, 1] - cy) >= clearance))

    def floor_free(rect, self_idx, margin=0.05):
        ix0, iy0, ix1, iy1, inside = occ.cells(rect[0] - margin, rect[1] - margin, rect[2] + margin, rect[3] + margin)
        if not inside:
            return False
        me = by_index[self_idx]
        for j, r in floor_rects.items():
            if j == self_idx or j in removed or not overlap(rect, r, margin):
                continue
            if me["category"] in UNDER_TABLE and by_index[j]["category"] in TABLES:
                continue                                    # seats may tuck under tables (legs are checked by the grid)
            if overlap_area(rect, r) <= overlap_area(me["orig_rect"], by_index[j]["orig_rect"]) + 1e-3:
                continue                                    # no new overlap beyond what already existed (touching furniture)
            return False
        sub = occ.grid[iy0:iy1 + 1, ix0:ix1 + 1]
        if sub.any():
            ys, xs = np.nonzero(sub)
            px = occ.x0 + (xs + ix0) * occ.res; py = occ.y0 + (ys + iy0) * occ.res
            # cells of the object's own old footprint and of removed / moved movable objects are free
            own = [me["orig_rect"]] + [by_index[j]["orig_rect"] for j in removed if j in floor_rects]
            ok = np.zeros(len(px), bool)
            for r in own:
                ok |= (px >= r[0] - 0.06) & (px <= r[2] + 0.06) & (py >= r[1] - 0.06) & (py <= r[3] + 0.06)
            if not ok.all():
                # remaining occupied cells (walls, fixed furniture): allowed only if the object does not penetrate deeper
                # than it already did in its original place (sliding along a wall is fine, pushing into it is not)
                pen = penetration(rect, px[~ok], py[~ok])
                if pen > me["orig_penetration"] + 0.01:
                    return False
        return path_clear(rect)

    def support_free(rect, sup, self_idx, margin=0.02):
        r = sup["rect"]
        me = by_index[self_idx]
        o0 = me["orig_rect"]
        over = [max(0.0, r[0] - o0[0]), max(0.0, r[1] - o0[1]), max(0.0, o0[2] - r[2]), max(0.0, o0[3] - r[3])]   # original overhang
        if not (rect[0] >= r[0] - over[0] - 0.05 and rect[1] >= r[1] - over[1] - 0.05 and rect[2] <= r[2] + over[2] + 0.05 and rect[3] <= r[3] + over[3] + 0.05):
            return False
        for other in children.get(sup["index"], []):
            if other["index"] == self_idx or other["index"] in removed or not overlap(rect, other["rect"], margin):
                continue
            if overlap_area(rect, other["rect"]) <= overlap_area(o0, other["orig_rect"]) + 1e-3:
                continue
            return False
        return True

    op_override = {k: float(v) for k, v in (kv.split(":") for kv in ops.split(","))} if ops else None

    def sample_op(cls):
        mix = op_override or CLASS_OPS[cls]
        ks = list(mix); ps = np.array([mix[k] for k in ks]); ps = ps / ps.sum()
        return ks[rng.choice(len(ks), p=ps)]

    def candidate_rooms(o):
        """Same room first; a move to one of the two nearest visited rooms only with probability 0.2 (real revisits are
        dominated by within-room changes)."""
        rooms = [o["room"]] if o["room"] else []
        if rng.random() < 0.2:
            others = [n for n in visited if n != o["room"] and n in room_centroid]
            if o["room"] in room_centroid:
                others.sort(key=lambda n: np.linalg.norm(room_centroid[n] - room_centroid[o["room"]]))
            rooms += others[:2]
        return rooms

    def try_relocate(o):
        if o["supported"]:
            if o.get("floating"):
                return None
            sup, r = o["support"], o["support"]["rect"]
            w, h = o["rect"][2] - o["rect"][0], o["rect"][3] - o["rect"][1]
            for _ in range(150):
                dyaw = rng.choice(YAW_CHOICES) + math.radians(rng.uniform(-15, 15))
                if r[2] - r[0] <= w or r[3] - r[1] <= h:
                    break
                c = np.array([rng.uniform(r[0] + w / 2, r[2] - w / 2), rng.uniform(r[1] + h / 2, r[3] - h / 2)])
                rect = rect_rotated_extent(o["rect"], c, dyaw)
                if support_free(rect, sup, o["index"]) and np.linalg.norm(c - o["cxy"]) > 0.15:
                    return c, dyaw, rect
            return None
        for room_i, room in enumerate(candidate_rooms(o) or [None]):
            mp = next((m for n, m in regions if n == room), None)
            lo, hi = (mp.vertices.min(0), mp.vertices.max(0)) if mp is not None else (o["cxy"] - 3, o["cxy"] + 3)
            for _ in range(300 if room_i == 0 else 60):
                dyaw = rng.choice(YAW_CHOICES) + math.radians(rng.uniform(-15, 15))
                c = np.array([rng.uniform(lo[0], hi[0]), rng.uniform(lo[1], hi[1])])
                if mp is not None and not mp.contains_point(c):
                    continue
                if np.linalg.norm(c - o["cxy"]) < 0.4:
                    continue
                rect = rect_rotated_extent(o["rect"], c, dyaw)
                if floor_free(rect, o["index"]):
                    return c, dyaw, rect
        return None

    def try_jitter(o):
        amp = (0.05, 0.15) if o.get("floating") else ((0.1, 0.3) if o["cls"] == "heavy" else (0.1, 0.5))
        for _ in range(100):
            ang = rng.uniform(0, 2 * math.pi); rad = rng.uniform(*amp)
            c = o["cxy"] + rad * np.array([math.cos(ang), math.sin(ang)])
            dyaw = math.radians(rng.uniform(-20, 20)) if not o.get("floating") else math.radians(rng.uniform(-10, 10))
            rect = rect_rotated_extent(o["rect"], c, dyaw)
            if o.get("floating"):
                ok = path_clear(rect)
            elif o["supported"]:
                ok = support_free(rect, o["support"], o["index"])
            else:
                ok = floor_free(rect, o["index"])
            if ok:
                return c, dyaw, rect
        return None

    def try_swap(o):
        cands = [p for p in pool if p is not o and p["cls"] == o["cls"] and p["supported"] == o["supported"]
                 and p["template"] != o["template"] and p["index"] not in done and p["index"] not in removed]
        if o["supported"]:
            cands = [p for p in cands if bool(p.get("floating")) == bool(o.get("floating")) and abs(p["z0"] - o["z0"]) < 0.3]
        same_room = [p for p in cands if p["room"] == o["room"]]
        nearby = [p for p in cands if np.linalg.norm(p["cxy"] - o["cxy"]) < 6.0]
        cands = same_room or nearby
        rng.shuffle(cands)
        for p in cands[:25]:
            ra = rect_rotated_extent(o["rect"], p["cxy"], 0.0); rb = rect_rotated_extent(p["rect"], o["cxy"], 0.0)
            if o["supported"]:
                if o.get("floating"):
                    ok = path_clear(ra) and path_clear(rb)
                else:
                    ok = support_free(ra, p["support"], p["index"]) and support_free(rb, o["support"], o["index"])
            else:
                others = [r for j, r in floor_rects.items() if j not in (o["index"], p["index"]) and j not in removed]
                ok = path_clear(ra) and path_clear(rb) and all(not overlap(ra, r, 0.03) for r in others) and all(not overlap(rb, r, 0.03) for r in others)
            if ok:
                return p, ra, rb
        return None

    changes, stats_ops, fallbacks = [], {}, {}

    def record_move(o, op, c, dyaw, rect, dz=0.0, partner=None):
        pivot = o["cxy"].copy()
        d = c - pivot
        changes.append({"index": o["index"], "name": o["name"], "class": o["cls"], "op": op, "partner": partner,
                        "from_xy": pivot.tolist(), "to_xy": [float(c[0]), float(c[1])], "pivot_xy": pivot.tolist(),
                        "dyaw": float(dyaw), "dz": float(dz), "displacement": float(np.linalg.norm(d))})
        if o["index"] in floor_rects:
            floor_rects[o["index"]] = rect
        o["rect"] = rect; o["cxy"] = np.array(c)
        done.add(o["index"])
        stats_ops[op] = stats_ops.get(op, 0) + 1
        for ch in children.get(o["index"], []):          # objects standing on it follow
            if ch["index"] in removed:
                continue
            nc = rigid_xy(ch["cxy"], pivot, dyaw, d)
            changes.append({"index": ch["index"], "name": ch["name"], "class": ch["cls"], "op": "follow", "partner": o["index"],
                            "from_xy": ch["cxy"].tolist(), "to_xy": nc.tolist(), "pivot_xy": pivot.tolist(),
                            "dyaw": float(dyaw), "dz": float(dz), "displacement": float(np.linalg.norm(nc - ch["cxy"]))})
            ch["rect"] = rect_rotated_extent(ch["rect"], nc, dyaw); ch["cxy"] = nc
            done.add(ch["index"])
            stats_ops["follow"] = stats_ops.get("follow", 0) + 1

    def record_remove(o, op="remove"):
        removed.add(o["index"]); done.add(o["index"])
        changes.append({"index": o["index"], "name": o["name"], "class": o["cls"], "op": op, "partner": None,
                        "from_xy": o["cxy"].tolist(), "to_xy": None, "pivot_xy": None, "dyaw": 0.0, "dz": 0.0, "displacement": 0.0})
        stats_ops[op] = stats_ops.get(op, 0) + 1
        for ch in children.get(o["index"], []):
            if ch["index"] not in removed:
                record_remove(ch, "follow_remove")

    for o in chosen:
        if o["index"] in done or o["index"] in removed:
            continue
        op = sample_op(o["cls"])
        if op == "swap":
            sw = try_swap(o)
            if sw is not None:
                p, ra, rb = sw
                ca, cb = o["cxy"].copy(), p["cxy"].copy()
                record_move(o, "swap", cb, 0.0, ra, dz=p["z0"] - o["z0"], partner=p["index"])
                record_move(p, "swap", ca, 0.0, rb, dz=o["z0"] - p["z0"], partner=o["index"])
                continue
            fallbacks["swap->relocate"] = fallbacks.get("swap->relocate", 0) + 1
            op = "relocate"
        res = None
        if op == "relocate":
            res = try_relocate(o)
            if res is None:
                fallbacks["relocate->jitter"] = fallbacks.get("relocate->jitter", 0) + 1
                op = "jitter"
        if op == "jitter":
            res = try_jitter(o)
            if res is None:
                fallbacks["jitter->remove"] = fallbacks.get("jitter->remove", 0) + 1
                op = "remove"
        if op == "remove":
            record_remove(o)
        else:
            c, dyaw, rect = res
            record_move(o, op, c, dyaw, rect)

    by_class, pool_by_class = {}, {}
    for ch in changes:
        by_class[ch["class"]] = by_class.get(ch["class"], 0) + 1
    for o in pool:
        pool_by_class[o["cls"]] = pool_by_class.get(o["cls"], 0) + 1
    moved = [c["displacement"] for c in changes if c["op"] in ("relocate", "jitter", "swap", "follow")]
    plan = {"scene": lay["scene"], "level": level, "seed": seed, "pool_size": len(pool), "pool_by_class": pool_by_class,
            "n_changed": len(changes), "changed_by_class": by_class, "changed_by_op": stats_ops, "fallbacks": fallbacks,
            "mean_displacement": float(np.mean(moved)) if moved else 0.0, "median_displacement": float(np.median(moved)) if moved else 0.0,
            "visited_rooms": sorted(visited), "pool_indices": sorted(pool_idx), "changes": changes}
    print(f"level {level} seed {seed}: pool {len(pool)} {pool_by_class} -> {len(changes)} changes {stats_ops}; fallbacks {fallbacks}; "
          f"by class {by_class}; mean displacement {plan['mean_displacement']:.2f} m; visited rooms {len(visited)}", flush=True)
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", required=True)
    ap.add_argument("--occ", required=True)
    ap.add_argument("--path", required=True, help="npy (N,2) xy of the map trajectory")
    ap.add_argument("--level", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--near", type=float, default=3.0)
    ap.add_argument("--clearance", type=float, default=0.45)
    ap.add_argument("--max-height", type=float, default=2.3)
    ap.add_argument("--ops", default=None, help="override op mix for all classes, e.g. relocate:1.0 or remove:1.0")
    ap.add_argument("--occ-low", default=None, help="leg-level occupancy (z 0.03-0.45 m) used for placement checks")
    args = ap.parse_args()
    layout = json.loads(Path(args.layout).read_text())
    plan = make_plan(layout, args.occ, np.load(args.path), args.level, args.seed, args.near, args.clearance, args.max_height, args.ops, args.occ_low)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(plan, indent=1))


if __name__ == "__main__":
    main()
