#!/usr/bin/env python3
"""Representative images for the Datasets tab of the results page: what each dataset looks like and, for the
multi-session datasets, what changes between the sessions.

  python benchmark/make_dataset_assets.py --data $BENCH_DATA

Reads the prepared sequences ($BENCH_DATA/<dataset>/<sequence>/<setup dir>) and benchmark/configs/dataset_samples.yaml.
Writes benchmark/site/assets/datasets/<dataset>/... (JPEG tiles) and benchmark/results/dataset_samples.json, which
build_site.py copies into the page data.  Run where the dataset folders are; needs numpy, Pillow and PyYAML.

A multi-session scene is shown as places: a place is a frame of the map session and every query session contributes the
frame whose ground-truth pose is nearest to it, so the tiles of a place differ only by what changed between the sessions.
For SimChange variants that are rendered from the map's own poses and change objects, a second image marks the pixels
that differ from the map frame.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "benchmark/site/assets/datasets"

# ------------------------------------------------------------------------------------------------ session captions
SIM_FAMILIES = [   # (key, title, what changes)
    ("map", "Map session", "The reference traversal that the map is built from."),
    ("light", "Lighting", "Other light sources (sun position and colour, sky, lamps; not a dimmed copy). Geometry and objects stay the same."),
    ("objects", "Object rearrangement", "A share of the movable objects (relocated, pushed, swapped or removed) differs from the map; "
                                         "the percentage is the share of the movable objects in the visited rooms that change."),
    ("background", "Background materials", "Wall, floor and furniture materials are swapped; the shapes stay the same."),
    ("view", "Viewpoint", "The robot drives a parallel path (offset), turns its camera (yaw) or sits lower (height)."),
    ("route", "Traversal", "The route is driven in the opposite direction (reverse) or only its second half is driven (half)."),
    ("combo", "Combined changes", "Several factors at once."),
]
LIGHT_NAMES = {"morning": "morning", "noon": "noon", "afternoon": "afternoon", "evening": "evening", "night": "night",
               "overcast": "overcast"}


def sim_variant(v: str):
    """(family key, short label) of a SimChange variant name, e.g. 'rearr_50_s1' -> ('objects', '50 % rearranged, seed 1')."""
    if v == "map":
        return "map", "map"
    if "+" in v:
        return "combo", " + ".join(sim_variant(p)[1] for p in v.split("+"))
    m = re.fullmatch(r"light_(\w+)", v)
    if m:
        return "light", f"{LIGHT_NAMES.get(m[1], m[1])} light"
    m = re.fullmatch(r"rearr_(\d+)(?:_s(\d+))?(?:_(remove|relocate|jitter))?", v)
    if m:
        lab = {None: "rearranged", "remove": "removed", "relocate": "relocated", "jitter": "pushed"}[m[3]]
        return "objects", f"{m[1]} % {lab}" + (f", seed {m[2]}" if m[2] else "")
    m = re.fullmatch(r"(move|remove)_(\d+)", v)
    if m:
        return "objects", f"{m[2]} % {'moved' if m[1] == 'move' else 'removed'}"
    if v == "background":
        return "background", "other materials"
    m = re.fullmatch(r"offset_([\d.]+)", v)
    if m:
        return "view", f"{m[1]} m sideways"
    m = re.fullmatch(r"yaw_([\d.]+)", v)
    if m:
        return "view", f"turned {m[1]}°"
    m = re.fullmatch(r"height_([\d.]+)", v)
    if m:
        return "view", f"camera at {m[1]} m"
    if v == "reverse":
        return "route", "opposite direction"
    if v == "half":
        return "route", "second half only"
    if v == "combo":
        return "combo", "evening + 50 % moved + reverse"
    return "combo", v


def rover_label(seq: str):
    m = re.fullmatch(r"campus_large_(.+?)_(\d{4}-\d{2}-\d{2})(?:_\d)?", seq)
    tag, date = (m[1], m[2]) if m else (seq, "")
    note = {"night-light": "night, with extra light (the grass in front of the robot is lit)", "night": "night", "dusk": "dusk"}.get(tag, "")
    return tag, f"{date}" + (f", {note}" if note else "")


# ------------------------------------------------------------------------------------------------ data access
def seq_folder(root: Path, dataset: str, seq: str):
    base = root / dataset / seq
    for s in ("rgbd", "stereo", "."):
        f = base if s == "." else base / s
        d = f / "left" if (f / "left").is_dir() else f / "rgb"
        if d.is_dir() and any(d.iterdir()):        # skip sequences that are not prepared yet (empty folders)
            return f
    return None


def frame_files(f: Path):
    d = f / ("left" if (f / "left").is_dir() else "rgb")
    return sorted(p for p in d.iterdir() if p.suffix in (".png", ".jpg"))


def load_poses(f: Path):
    p = f / "poses_left.txt"
    return np.loadtxt(p).reshape(-1, 4, 4) if p.is_file() else None


def nearest(Pm, i, Pq, any_heading=False):
    """Frame of the query poses nearest to map frame i: (index, distance in m, viewing-direction difference in degrees)."""
    d = np.linalg.norm(Pq[:, :3, 3] - Pm[i, :3, 3], axis=1)
    ang = np.degrees(np.arccos(np.clip(Pq[:, :3, 2] @ Pm[i, :3, 2], -1, 1)))
    j = int(np.argmin(d if any_heading else d + ang / 20))
    return j, float(d[j]), float(ang[j])


def save(im: Image.Image, path: Path, width: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    im.convert("RGB").resize((width, round(width * im.height / im.width)), Image.LANCZOS).save(path, quality=82)


def change_overlay(a: Image.Image, b: Image.Image):
    """b with the pixels that differ from a tinted red (same camera pose, same lighting: object changes only)."""
    A = np.asarray(a.convert("RGB").filter(ImageFilter.GaussianBlur(1.5)), np.float32)
    B = np.asarray(b.convert("RGB").filter(ImageFilter.GaussianBlur(1.5)), np.float32)
    diff = np.abs(A - B).max(2)
    mask = Image.fromarray(((diff > 28) * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(1.5))
    m = (np.asarray(mask, np.float32) / 255.0)[..., None]
    out = np.asarray(b.convert("RGB"), np.float32) * (1 - 0.55 * m) + np.array([255, 30, 30], np.float32) * 0.55 * m
    return Image.fromarray(out.clip(0, 255).astype(np.uint8)), float((diff > 28).mean())


# ------------------------------------------------------------------------------------------------ datasets
def multi_session(root, dataset, ds_cfg, smp, out):
    res = {}
    W = smp["width"]
    for scene, sc in ds_cfg["scenes"].items():
        maps = seq_folder(root, dataset, sc["map"])
        if maps is None:
            continue
        queries = {q: seq_folder(root, dataset, q) for q in sc.get("queries", [])}
        queries = {q: f for q, f in queries.items() if f}
        Pm = load_poses(maps)
        files = {sc["map"]: frame_files(maps)}
        entry = {"places": []}
        if scene in smp.get("unaligned", {}):         # no ground truth: same fraction of every session, not the same place
            for frac in smp["unaligned"][scene]:
                tiles = []
                for s, f in [(sc["map"], maps)] + list(queries.items()):
                    fl = files.get(s) or frame_files(f)
                    j = int(frac * (len(fl) - 1))
                    p = out / dataset / scene / f"u{int(frac * 100)}" / f"{s}.jpg"
                    save(Image.open(fl[j]), p, W)
                    tiles.append({"session": s, "role": "map" if s == sc["map"] else "query", "frame": j, "img": str(p.relative_to(ROOT / "benchmark/site"))})
                entry["places"].append({"aligned": False, "tiles": tiles})
            res[scene] = entry
            continue
        for i in smp["places"].get(scene, []):
            tiles = []
            mp = out / dataset / scene / f"f{i}" / f"{sc['map']}.jpg".replace("/", "_")
            mim = Image.open(files[sc["map"]][i]).convert("RGB")
            save(mim, mp, W)
            tiles.append({"session": sc["map"], "role": "map", "frame": i, "img": str(mp.relative_to(ROOT / "benchmark/site"))})
            for q, f in queries.items():
                fam = sim_variant(q.split("/")[-1])[0] if dataset == "simchange" else None
                j, d, a = nearest(Pm, i, load_poses(f), any_heading=fam == "route" and q.endswith("reverse"))
                if d > smp["max_dist"] * (2.5 if fam == "route" else 1) or (a > smp["max_angle"] and not (fam == "route" and q.endswith("reverse"))):
                    continue           # the session does not pass this place (in this direction)
                fl = files.get(q) or frame_files(f)
                files[q] = fl
                im = Image.open(fl[j]).convert("RGB")
                p = out / dataset / scene / f"f{i}" / f"{q}.jpg".replace("/", "_")
                save(im, p, W)
                t = {"session": q, "role": "query", "frame": j, "dist": round(d, 2), "angle": round(a, 1), "img": str(p.relative_to(ROOT / "benchmark/site"))}
                if dataset == "simchange" and fam == "objects" and "+" not in q and d < smp.get("diff_dist", 0.05) and a < 2:
                    ov, frac = change_overlay(mim, im)
                    t["changed_px"] = round(frac, 3)
                    if frac >= 0.005:          # nothing visible from this place otherwise
                        pd = p.with_name(p.stem + "_diff.jpg")
                        save(ov, pd, W)
                        t["diff"] = str(pd.relative_to(ROOT / "benchmark/site"))
                tiles.append(t)
            entry["places"].append({"aligned": True, "frame": i, "tiles": tiles})
        res[scene] = entry
    return res


def kitti(root, ds_cfg, smp, out):
    res = []
    for seq in ds_cfg["scenes"]:
        f = seq_folder(root, "kitti", seq)
        if f is None:
            continue
        fl = frame_files(f)
        j = int(smp["fraction"] * (len(fl) - 1))
        p = out / "kitti" / f"{seq}.jpg"
        save(Image.open(fl[j]), p, smp["width"])
        res.append({"session": seq, "frame": j, "img": str(p.relative_to(ROOT / "benchmark/site"))})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="$BENCH_DATA: folder with openloris/, kitti/, rover/, simchange/")
    ap.add_argument("--datasets", nargs="*", default=["openloris", "rover", "simchange", "kitti"])
    a = ap.parse_args()
    root = Path(a.data)
    cfg = yaml.safe_load((ROOT / "benchmark/configs/datasets.yaml").read_text())
    smp = yaml.safe_load((ROOT / "benchmark/configs/dataset_samples.yaml").read_text())
    dst = ROOT / "benchmark/results/dataset_samples.json"
    res = json.loads(dst.read_text()) if dst.is_file() else {}
    for d in a.datasets:
        res[d] = kitti(root, cfg[d], smp[d], ASSETS) if d == "kitti" else multi_session(root, d, cfg[d], smp[d], ASSETS)
        n = sum(len(p["tiles"]) for s in res[d].values() for p in s["places"]) if d != "kitti" else len(res[d])
        print(f"{d}: {n} images")
    if "simchange" in res:
        res["simchange_families"] = [list(f) for f in SIM_FAMILIES]
    for sc in res.get("simchange", {}).values():
        for pl in sc["places"]:
            for t in pl["tiles"]:
                t["family"], t["label"] = sim_variant(t["session"].split("/")[-1])
    for sc in res.get("rover", {}).values():
        for pl in sc["places"]:
            for t in pl["tiles"]:
                t["label"], t["note"] = ("day", "2024-09-25, map") if t["role"] == "map" else rover_label(t["session"])
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(res, indent=1) + "\n")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
