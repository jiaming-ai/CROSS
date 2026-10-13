"""Test set of CROSS Nav: the SimChange scenes (never trained on).

    # scene cache of the `map` traversals (dataset simchange_test; one scene per SimChange scene, sequence `map`)
    python -m cross_nav.sim.simchange_test cache --bench <bench_data>/simchange --cache <root> [--scenes ...]
    # mesh BEV labels in the SimChange world frame (geometry from blend_geom.py on the SimChange blend files)
    python -m cross_nav.sim.simchange_test label --bench ... --geom-dir <dir> --cache <root> --out <bev>/simchange_test
    # alignment check: benchmark depth of a few frames projected onto the label
    python -m cross_nav.sim.simchange_test check --bench ... --out <bev>/simchange_test

SimChange worlds are z up (the Blender frame of the scene); poses_left.txt holds camera-to-world matrices in OpenCV
axes; depth/*.png is uint16 millimetres.  The cache stores K with the principal point at (W/2 - 0.5, H/2 - 0.5)
(pixel-centre convention of the scene cache; calib.json's W/2 is the continuous image centre of Blender).
"""
from __future__ import annotations

import os
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

SCENES = {   # name -> (geometry file stem, camera height of the SimChange route (m), kind, layout json)
    "hssd_house": ("hssd_104348010", 1.3, "indoor", "hssd_104348010.blend.layout.json"),
    "hssd_restaurant": ("hssd_103997718", 1.3, "indoor", "hssd_103997718.blend.layout.json"),
    "classroom": ("classroom", 1.2, "indoor", None),
    "lonemonk": ("lone_monk", 1.4, "outdoor", None),
}


def load_trav(d: Path):
    P = np.loadtxt(d / "poses_left.txt").reshape(-1, 4, 4)
    calib = json.loads((d / "calib.json").read_text())
    K = np.array(calib["K"], np.float64)
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    return P, K, calib


def cmd_cache(a):
    from vggt_ft.prep.common import SceneWriter
    for name in a.scenes:
        d = Path(a.bench) / name / "map"
        P, K, calib = load_trav(d)
        sw = SceneWriter(a.cache, a.dataset, name, world=name, metric=True, synthetic=True, kind=SCENES[name][2],
                         extra={"source": "simchange", "traversal": "map"})
        q = sw.sequence("map", session="map")
        for k in range(len(P)):
            img = cv2.imread(str(d / "left" / f"{k:06d}.png"), cv2.IMREAD_COLOR)
            dep = cv2.imread(str(d / "depth" / f"{k:06d}.png"), cv2.IMREAD_UNCHANGED)
            if img is None or dep is None:
                print(name, "missing frame", k, flush=True)
                continue
            q.add(f"{k:06d}", cv2.cvtColor(img, cv2.COLOR_BGR2RGB), dep.astype(np.float32) / 1000.0, K,
                  np.linalg.inv(P[k]), t=k / float(calib.get("fps", 10.0)))
        print(name, sw.close(), "frames", flush=True)


def cmd_label(a):
    from cross_nav.sim.mesh_bev import build_mesh_bev, camera_heights, preview
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in a.scenes:
        stem, h, _, lay = SCENES[name]
        gp = Path(a.geom_dir) / f"{stem}.npz"
        if not gp.exists():
            print(name, "no geometry", gp, flush=True)
            continue
        geom = dict(np.load(gp))
        regions = json.loads((Path(a.geom_dir) / lay).read_text()).get("regions", []) if lay and (Path(a.geom_dir) / lay).exists() else None
        d = Path(a.bench) / name / "map"
        P, K, _ = load_trav(d)
        seeds = P[:, :3, 3]
        out, stats, aux = build_mesh_bev(geom, seeds_xyz=seeds, seeds_h=np.full(len(seeds), h), regions=regions)
        E = np.linalg.inv(P)
        out[f"cam_h/{name}/map"] = camera_heights(E, aux["g_int"], out["origin"], float(out["res"]))
        ch = out[f"cam_h/{name}/map"]
        stats.update(frames=len(P), cam_h_ok=float(np.isfinite(ch).mean()), cam_h_med=float(np.nanmedian(ch)), source="mesh",
                     scene=name)
        out["stats"] = json.dumps(stats)
        np.savez_compressed(out_dir / f"{name}.npz", **out)
        preview(out, out_dir / f"{name}.png", [seeds[:, :2]])
        print(name, json.dumps(stats), flush=True)


def cmd_check(a):
    """Benchmark depth of `n` frames per scene: points between ground + 0.15 and + 1.7 m should fall on occupied cells,
    floor points (|z - ground| < 3 cm, away from obstacles) on free cells.  Also draws them on the label."""
    from cross_nav.labels import frame_points
    res_all = {}
    for name in a.scenes:
        p = Path(a.out) / f"{name}.npz"
        if not p.exists():
            continue
        L = dict(np.load(p))
        bev, o, r = L["bev"], L["origin"], float(L["res"])
        g = L["ground"].astype(np.float32)
        d = Path(a.bench) / name / "map"
        P, K, _ = load_trav(d)
        idx = np.linspace(0, len(P) - 1, a.n).round().astype(int)
        H, W = bev.shape
        img = np.full(bev.shape + (3,), 128, np.uint8)
        img[bev == 1] = 255
        img[bev == 2] = 0
        n_obs = n_obs_occ = n_fl = n_fl_free = 0
        from scipy import ndimage
        reached = np.isfinite(g)
        dist, (ny, nx) = ndimage.distance_transform_edt(~reached, return_indices=True)
        gi = g[ny, nx]
        for k in idx:
            dep = cv2.imread(str(d / "depth" / f"{k:06d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
            Pw, _ = frame_points(dep, K, np.linalg.inv(P[k]), 4, 8.0)
            ix = np.floor((Pw[:, 0] - o[0]) / r).astype(int)
            iy = np.floor((Pw[:, 1] - o[1]) / r).astype(int)
            ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
            ix, iy, z = ix[ok], iy[ok], Pw[ok, 2]
            dz = z - gi[iy, ix]
            obs = (dz > 0.15) & (dz < 1.7)
            fl = (np.abs(dz) < 0.03) & reached[iy, ix]
            n_obs += obs.sum()
            n_obs_occ += (bev[iy[obs], ix[obs]] == 2).sum()
            n_fl += fl.sum()
            n_fl_free += (bev[iy[fl], ix[fl]] == 1).sum()
            img[iy[obs], ix[obs]] = (0, 0, 255)
            img[iy[fl], ix[fl]] = (0, 200, 0)
        res_all[name] = dict(frames=len(idx), obstacle_pts=int(n_obs), obstacle_on_occ=round(n_obs_occ / max(n_obs, 1), 4),
                             floor_pts=int(n_fl), floor_on_free=round(n_fl_free / max(n_fl, 1), 4))
        cv2.imwrite(str(Path(a.out) / f"{name}_depthcheck.png"), cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)[::-1])
        print(name, json.dumps(res_all[name]), flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(res_all, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cache", "label", "check"])
    ap.add_argument("--bench", default=os.path.join(os.environ.get("CROSS_NAV_ROOT", "."), "bench_data/simchange"))
    ap.add_argument("--cache", default=os.path.join(os.environ.get("CROSS_NAV_ROOT", "."), "cache"))
    ap.add_argument("--dataset", default="simchange_test")
    ap.add_argument("--geom-dir", default=os.path.join(os.environ.get("CROSS_NAV_ROOT", "."), "navsim/simchange_geom"))
    ap.add_argument("--out", default=os.path.join(os.environ.get("CROSS_NAV_ROOT", "."), "nav/bev/simchange_test"))
    ap.add_argument("--scenes", nargs="*", default=list(SCENES))
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    {"cache": cmd_cache, "label": cmd_label, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
