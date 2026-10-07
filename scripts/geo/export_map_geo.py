"""Export the keyframes of a GNSS-anchored CROSS map (map.pkl saved with geo.enabled) as GeoJSON and KML: the
keyframe track, the keyframes' latitude / longitude / altitude, and the anchor (ENU origin, heading, uncertainty).

  python scripts/geo/export_map_geo.py outputs/run/map.pkl --out outputs/run/map_geo
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cross.geo.geodesy import geojson_linestring, geojson_points, kml_linestring  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="output prefix (writes <out>.geojson, <out>.kml, <out>.json)")
    a = ap.parse_args()
    with open(a.map, "rb") as f:
        m = pickle.load(f)
    geo = m.get("geo")
    if not geo or not geo.get("keyframe_lla"):
        sys.exit("the map has no geo anchor (built without --gnss, or never anchored)")
    lla = {int(k): v for k, v in geo["keyframe_lla"].items()}
    ids = sorted(lla)
    arr = np.array([lla[k] for k in ids])
    perm = {d["id"] for d in m["db_data"]["keyframes"]}
    feats = [geojson_linestring(arr, {"name": "keyframes", "n": len(ids)})]
    feats += geojson_points(arr[[i for i, k in enumerate(ids) if k in perm]],
                            [{"id": k, "permanent": True} for k in ids if k in perm])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(a.out) + ".geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    Path(str(a.out) + ".kml").write_text(kml_linestring(arr, name=a.map.parent.name))
    anchor = geo.get("anchor", {})
    info = {"origin": geo.get("frame"), "anchor_yaw_deg": None if anchor.get("yaw") is None else float(np.degrees(anchor["yaw"])),
            "anchor_cov": anchor.get("cov"), "n_keyframes": len(ids), "n_permanent": len(perm & set(ids)),
            "noise": geo.get("noise"), "correlation_time_s": (geo.get("err") or {}).get("tau"), "gate": geo.get("gate"),
            "compass": {k: (geo.get("compass") or {}).get(k) for k in ("offset", "offset_std", "hard_iron")}}
    Path(str(a.out) + ".json").write_text(json.dumps(info, indent=1, default=float))
    print(json.dumps(info, indent=1, default=float))


if __name__ == "__main__":
    main()
