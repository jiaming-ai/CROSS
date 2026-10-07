"""Geo-referenced accuracy of a CROSS map built with GNSS anchoring (`scripts/map_and_reloc*.py --gnss`) on a posed
folder whose ground truth is in a local frame tied to WGS84.

Reads <run>/map.pkl (the map's geo state: keyframe latitude / longitude) and <run>/map_meta.json (ground-truth camera
pose of each keyframe), converts the keyframes' latitude / longitude into the ground-truth frame and reports the
absolute (no alignment) and shape (best 2-D rigid alignment) errors, overall and split by GNSS availability (no fix
for > 2 s = indoors), plus a GeoJSON of the map and the ground truth.  Ground-truth frames: `nclt` (NCLT local frame,
north-east-down, linearised at lat 42.293227, lon -83.709657, alt 270 m; prepare_nclt.to_local).

  python scripts/geo/eval_geo_map.py --run outputs/nclt_map_gnss --gnss-file <posed>/gnss.txt --times <posed>/times.txt
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cross.geo.geodesy import geojson_linestring  # noqa: E402
from cross.geo.gnss import fit_rigid_2d  # noqa: E402

LAT0, LON0, ALT0 = 42.293227, -83.709657, 270.0
R_EQ, R_POL = 6378135.0, 6356750.0


def _radii(lat0):
    d = (R_EQ * np.cos(lat0)) ** 2 + (R_POL * np.sin(lat0)) ** 2
    return (R_EQ * R_POL) ** 2 / d ** 1.5, R_EQ ** 2 / np.sqrt(d)


def nclt_to_local(lat, lon, alt):
    lat0, lon0 = np.radians(LAT0), np.radians(LON0)
    rns, rew = _radii(lat0)
    lat, lon = np.radians(lat), np.radians(lon)
    return np.stack([np.sin(lat - lat0) * rns, np.sin(lon - lon0) * rew * np.cos(lat0), ALT0 - np.asarray(alt)], -1)


def nclt_from_local(xyz):
    lat0, lon0 = np.radians(LAT0), np.radians(LON0)
    rns, rew = _radii(lat0)
    xyz = np.asarray(xyz, float)
    return np.stack([np.degrees(np.arcsin(xyz[:, 0] / rns) + lat0),
                     np.degrees(np.arcsin(xyz[:, 1] / (rew * np.cos(lat0))) + lon0), ALT0 - xyz[:, 2]], -1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--gnss-file", type=Path, default=None, help="the folder's gnss.txt (indoor / outdoor split)")
    ap.add_argument("--times", type=Path, default=None, help="the folder's times.txt (frame -> time)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    meta = json.loads((a.run / "map_meta.json").read_text())
    with open(a.run / "map.pkl", "rb") as f:
        m = pickle.load(f)
    geo = m.get("geo") or {}
    lla = {int(k): v for k, v in (geo.get("keyframe_lla") or {}).items()}
    kf_gt = {int(k): np.asarray(v) for k, v in meta["kf_gt"].items()}
    kf_frame = {int(k): int(v) for k, v in meta.get("kf_frame", {}).items()}
    ids = sorted(k for k in lla if k in kf_gt)
    res = {"n_keyframes_geo": len(ids), "anchored": bool(geo.get("anchor", {}).get("R") is not None)}
    if ids:
        P = nclt_to_local(*np.array([lla[k] for k in ids]).T)
        G = np.array([kf_gt[k][:3, 3] for k in ids])
        e = np.linalg.norm(P[:, :2] - G[:, :2], axis=1)
        R, t = fit_rigid_2d(P[:, :2], G[:, :2], np.ones(len(ids)))
        es = np.linalg.norm(P[:, :2] @ R.T + t - G[:, :2], axis=1)
        res.update({"geo_rmse": float(np.sqrt((e ** 2).mean())), "geo_median": float(np.median(e)),
                    "geo_p95": float(np.percentile(e, 95)), "geo_max": float(e.max()),
                    "shape_rmse": float(np.sqrt((es ** 2).mean()))})
        if a.gnss_file is not None and a.times is not None and kf_frame:
            g = np.loadtxt(a.gnss_file)
            gt_ = np.unique(g[:, 0])
            times = np.loadtxt(a.times).reshape(-1)
            tk = np.array([times[min(kf_frame.get(k, 0), len(times) - 1)] for k in ids])
            j = np.clip(np.searchsorted(gt_, tk), 1, len(gt_) - 1)
            gap = np.minimum(np.abs(gt_[j] - tk), np.abs(gt_[j - 1] - tk))
            indoor = gap > 2.0
            res["indoor_frac"] = float(indoor.mean())
            if indoor.any():
                res["indoor_rmse"] = float(np.sqrt((e[indoor] ** 2).mean()))
            res["outdoor_rmse"] = float(np.sqrt((e[~indoor] ** 2).mean())) if (~indoor).any() else None
        fc = {"type": "FeatureCollection", "features": [
            geojson_linestring(np.array([lla[k] for k in ids]), {"name": "map (GNSS-anchored)"}),
            geojson_linestring(nclt_from_local(G), {"name": "ground truth"})]}
        (a.out or a.run).mkdir(parents=True, exist_ok=True)
        ((a.out or a.run) / "geo_eval.geojson").write_text(json.dumps(fc))
    res["geo_state"] = {k: geo.get(k) for k in ("noise", "err", "gate", "stats")}
    res["map_ate_rmse_aligned"] = meta.get("map_ate_rmse")
    (a.out or a.run).mkdir(parents=True, exist_ok=True)
    ((a.out or a.run) / "geo_eval.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "geo_state"}, indent=1, default=float))


if __name__ == "__main__":
    main()
