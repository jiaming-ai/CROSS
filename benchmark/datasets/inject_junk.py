#!/usr/bin/env python3
"""Inject junk views into a prepared sequence folder: the test set of the keyframe quality filter.

Junk views are what a robot's camera sees when something close blocks it or the image carries no scene: a person
stepping in front of the robot, an object carried past the lens, a wall or a box at arm's length, the lights going off,
a fast turn (motion blur), glare.  They come in bursts (a person walks past in 0.5-2 s).  The same occluders (the same
people, the same objects) appear in the map and in the query sessions at unrelated places, which is what makes them
harmful: their views look alike, so they are retrieved at the wrong place.

  # once: occluder bank (person cutouts with masks from frames with people; close-up textures from other scenes)
  python benchmark/datasets/inject_junk.py bank --person-frames <dir>... --texture-frames <dir>... --out bank/
  # per sequence: a copy of a prepared folder (symlinks, only the corrupted frames written) and junk_manifest.json
  python benchmark/datasets/inject_junk.py inject --src $BENCH_DATA/openloris/office1-1/stereo \
      --dst $BENCH_DATA/openloris/office1-1_occ/stereo --bank bank/ --seed 0

Folder layouts: stereo (`left/`, `right/` with calib.json `baseline` and `K`: the occluder is pasted into the right
image with the disparity of its depth) and posed RGB-D (`rgb/`, `depth/` as .npy metres or .png millimetres: the
occluder's pixels get its depth).  Deterministic for a given seed and source folder name.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

KINDS = {"person": 0.35, "object": 0.2, "blank": 0.15, "dark": 0.1, "blur": 0.1, "glare": 0.1}


# ------------------------------------------------------------------------------------------------------------- bank
def build_bank(a):
    import torch
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    (out / "person").mkdir(parents=True, exist_ok=True)
    (out / "texture").mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1).to(dev).eval()
    frames = []
    for d in a.person_frames:
        fs = sorted(p for p in Path(d).iterdir() if p.suffix.lower() in (".png", ".jpg"))
        frames += fs[:: max(len(fs) // a.frames_per_dir, 1)]
    rng.shuffle(frames)
    n = 0
    meta = []
    for f in frames:
        if n >= a.n_person:
            break
        img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        with torch.inference_mode():
            r = model([torch.from_numpy(img).permute(2, 0, 1).float().div(255).to(dev)])[0]
        for box, lab, sc, m in zip(r["boxes"].cpu().numpy(), r["labels"].cpu().numpy(), r["scores"].cpu().numpy(),
                                   r["masks"][:, 0].cpu().numpy()):
            if lab != 1 or sc < 0.9:
                continue
            x0, y0, x1, y1 = [int(round(v)) for v in box]
            if (y1 - y0) < 0.25 * H or x0 <= 1 or x1 >= W - 2 or y0 <= 1:   # big enough, not cut at the sides / top
                continue
            alpha = (m[y0:y1, x0:x1] > 0.5).astype(np.uint8) * 255
            if alpha.mean() / 255 < 0.3:
                continue
            rgba = np.dstack([img[y0:y1, x0:x1], alpha])
            p = out / "person" / f"p{n:03d}.png"
            cv2.imwrite(str(p), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
            meta.append({"file": p.name, "source": str(f), "score": float(sc)})
            n += 1
            break
    tex = []
    for d in a.texture_frames:
        fs = sorted(p for p in Path(d).iterdir() if p.suffix.lower() in (".png", ".jpg"))
        tex += fs
    for i in range(a.n_texture):
        f = tex[int(rng.integers(len(tex)))]
        img = cv2.imread(str(f))
        H, W = img.shape[:2]
        w, h = int(W * 0.15), int(H * 0.15)
        x, y = int(rng.integers(0, W - w)), int(rng.integers(0, H - h))
        cv2.imwrite(str(out / "texture" / f"t{i:03d}.png"), img[y:y + h, x:x + w])
    (out / "bank.json").write_text(json.dumps({"person": meta, "n_texture": a.n_texture, "seed": a.seed}, indent=1))
    print(f"bank: {n} person cutouts, {a.n_texture} textures -> {out}")


# ------------------------------------------------------------------------------------------------------------- inject
def _seed(seed: int, name: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "little")


def _bursts(n: int, frac: float, rng, min_len=5, max_len=20, gap=10):
    """Non-overlapping bursts [start, end) covering about `frac` of the n frames, >= `gap` clean frames apart; the
    first 10 frames stay clean (session start)."""
    target, used, out = int(frac * n), 0, []
    tries = 0
    while used < target and tries < 1000:
        tries += 1
        L = int(rng.integers(min_len, max_len + 1))
        s = int(rng.integers(10, max(n - L, 11)))
        e = min(s + L, n)
        if any(s < b + gap and e + gap > a_ for a_, b in out) or (out and used + e - s > 1.25 * target):
            continue
        out.append((s, e))
        used += e - s
    return sorted(out)


def _paste(dst, src_rgb, src_alpha, cx, cy):
    """Alpha-blend src (h, w, 3 float) centred at (cx, cy) into dst (H, W, 3 float) in place; returns the coverage mask."""
    H, W = dst.shape[:2]
    h, w = src_alpha.shape
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    mask = np.zeros((H, W), np.float32)
    if xa >= xb or ya >= yb:
        return mask
    a = src_alpha[ya - y0:yb - y0, xa - x0:xb - x0, None]
    dst[ya:yb, xa:xb] = dst[ya:yb, xa:xb] * (1 - a) + src_rgb[ya - y0:yb - y0, xa - x0:xb - x0] * a
    mask[ya:yb, xa:xb] = a[..., 0]
    return mask


class Occluder:
    """One burst's occluder: kind, appearance, depth, motion across the burst."""

    def __init__(self, kind, bank, rng, W, H):
        self.kind, self.W, self.H = kind, W, H
        self.z = float(rng.uniform(0.35, 0.9) if kind == "person" else rng.uniform(0.2, 0.6))
        self.cov = float(rng.uniform(0.3, 0.9))
        self.id = None
        if kind == "person":
            p = bank["person"][int(rng.integers(len(bank["person"])))]
            self.id = p.name
            rgba = cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA).astype(np.float32) / 255
            self.rgb0, self.a0 = rgba[..., :3], rgba[..., 3]
            self.blur = float(rng.uniform(1.0, 2.5)) * W / 640
        elif kind == "object":
            t = bank["texture"][int(rng.integers(len(bank["texture"])))]
            self.id = t.name
            tex = cv2.cvtColor(cv2.imread(str(t)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            # a rounded blob of the texture (a box / bag / hand at arm's length)
            h, w = tex.shape[:2]
            a = np.zeros((h, w), np.float32)
            cv2.ellipse(a, (w // 2, h // 2), (int(w * 0.48), int(h * 0.48)), 0, 0, 360, 1.0, -1)
            self.rgb0, self.a0 = tex, cv2.GaussianBlur(a, (0, 0), max(min(w, h) * 0.03, 1))
            self.blur = float(rng.uniform(2.0, 4.0)) * W / 640
        elif kind == "blank":
            g = float(rng.uniform(0.3, 0.8))
            tint = rng.uniform(-0.05, 0.05, 3)
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            ramp = (rng.uniform(-0.1, 0.1) * xx / W + rng.uniform(-0.1, 0.1) * yy / H)
            self.surface = np.clip(g + tint[None, None] + ramp[..., None], 0, 1).astype(np.float32)
            self.cov = 1.0
        elif kind == "dark":
            self.gain = float(rng.uniform(0.03, 0.1))
        elif kind == "glare":
            self.gain, self.offset = float(rng.uniform(2.0, 3.0)), float(rng.uniform(0.3, 0.5))
        elif kind == "blur":
            L = int(rng.uniform(25, 50) * W / 640)
            k = np.zeros((L, L), np.float32)
            k[L // 2, :] = 1.0
            M = cv2.getRotationMatrix2D((L / 2 - 0.5, L / 2 - 0.5), float(rng.uniform(-30, 30)), 1.0)
            k = cv2.warpAffine(k, M, (L, L))
            self.kernel = k / k.sum()
        if kind in ("person", "object"):
            # scale so that the cutout covers `cov` of the frame when inside it; centre path across the burst
            area = float(self.a0.mean()) * self.a0.size
            self.scale = np.sqrt(self.cov * W * H / max(area, 1.0))
            self.c0 = np.array([rng.uniform(0.3, 0.7) * W, rng.uniform(0.45, 0.75) * H])
            self.v = np.array([rng.uniform(-0.3, 0.3) * W, rng.uniform(-0.05, 0.05) * H])

    def apply(self, img, t, disparity=0.0):
        """img (H, W, 3) float [0, 1]; t in [0, 1] position in the burst; disparity: horizontal shift (right image).
        Returns (image, occluder mask)."""
        H, W = img.shape[:2]
        if self.kind in ("person", "object"):
            s = self.scale * (1.0 + 0.1 * np.sin(np.pi * t))
            rgb = cv2.resize(self.rgb0, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            a = cv2.resize(self.a0, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            if rgb.shape[0] * rgb.shape[1] > 16 * W * H:            # far beyond the frame: crop around the centre
                ch, cw = min(rgb.shape[0], 4 * H), min(rgb.shape[1], 4 * W)
                y0, x0 = (rgb.shape[0] - ch) // 2, (rgb.shape[1] - cw) // 2
                rgb, a = rgb[y0:y0 + ch, x0:x0 + cw], a[y0:y0 + ch, x0:x0 + cw]
            rgb = cv2.GaussianBlur(rgb, (0, 0), self.blur)
            a = cv2.GaussianBlur(a, (0, 0), self.blur * 0.5)
            c = self.c0 + self.v * (t - 0.5)
            out = img.copy()
            m = _paste(out, rgb, a, c[0] - disparity, c[1])
            return out, m
        if self.kind == "blank":
            return self.surface + np.random.default_rng(int(t * 1e6)).normal(0, 0.01, img.shape).astype(np.float32), np.ones((H, W), np.float32)
        if self.kind == "dark":
            return np.clip(img * self.gain + np.random.default_rng(int(t * 1e6)).normal(0, 0.008, img.shape).astype(np.float32), 0, 1), np.zeros((H, W), np.float32)
        if self.kind == "glare":
            return np.clip(img * self.gain + self.offset, 0, 1), np.zeros((H, W), np.float32)
        if self.kind == "blur":
            return cv2.filter2D(img, -1, self.kernel, borderType=cv2.BORDER_REFLECT), np.zeros((H, W), np.float32)
        raise ValueError(self.kind)


def _read(p: Path):
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    gray = im.ndim == 2
    rgb = np.repeat(im[..., None], 3, 2) if gray else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0, gray


def _write(p: Path, img, gray):
    x = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) if gray else cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(p), x)


def inject(a):
    src, dst = Path(a.src).resolve(), Path(a.dst)
    bank = {"person": sorted((Path(a.bank) / "person").glob("*.png")),
            "texture": sorted((Path(a.bank) / "texture").glob("*.png"))}
    stereo = (src / "left").is_dir()
    img_dir = "left" if stereo else ("rgb" if (src / "rgb").is_dir() else "left")
    frames = sorted(p for p in (src / img_dir).iterdir() if p.suffix.lower() in (".png", ".jpg"))
    n = len(frames)
    name = src.parent.name + "/" + src.name
    rng = np.random.default_rng(_seed(a.seed, name))
    bursts = _bursts(n, a.frac, rng, a.min_len, a.max_len)
    kinds = [k for k in a.kinds.split(",")]
    probs = np.array([KINDS[k] for k in kinds], float)
    probs /= probs.sum()
    calib = json.loads((src / "calib.json").read_text()) if (src / "calib.json").exists() else {}
    fx = float(np.asarray(calib.get("K", [[0]]))[0][0]) if calib.get("K") else 0.0
    baseline = float(calib.get("baseline", 0.0) or 0.0)

    # the copy: symlinks for everything, image folders filled per frame
    dst.mkdir(parents=True, exist_ok=True)
    img_dirs = [img_dir] + (["right"] if stereo and (src / "right").is_dir() else []) + \
               (["depth"] if (not stereo and (src / "depth").is_dir()) else [])
    for p in src.iterdir():
        q = dst / p.name
        if p.name in img_dirs or q.exists() or q.is_symlink():
            continue
        os.symlink(p, q)
    for d in img_dirs:
        (dst / d).mkdir(exist_ok=True)
        for p in (src / d).iterdir():
            q = dst / d / p.name
            if not (q.exists() or q.is_symlink()):
                os.symlink(p, q)
    depth_files = sorted((src / "depth").iterdir()) if "depth" in img_dirs else []
    right_files = sorted((src / "right").iterdir()) if "right" in img_dirs else []

    manifest = {"src": str(src), "seed": a.seed, "frac": a.frac, "n_frames": n, "bursts": [], "frames": {}}
    for bi, (s, e) in enumerate(bursts):
        kind = kinds[int(rng.choice(len(kinds), p=probs))]
        img0, _ = _read(frames[s])
        H, W = img0.shape[:2]
        occ = Occluder(kind, bank, rng, W, H)
        manifest["bursts"].append({"start": s, "end": e, "kind": kind, "occluder": occ.id, "z": occ.z})
        for f in range(s, e):
            t = (f - s) / max(e - s - 1, 1)
            img, gray = _read(frames[f])
            out, m = occ.apply(img, t)
            q = dst / img_dir / frames[f].name
            q.unlink()
            _write(q, out, gray)
            if right_files:
                imr, grayr = _read(right_files[f])
                disp = fx * baseline / occ.z if (fx > 0 and baseline > 0) else 0.0
                outr, _ = occ.apply(imr, t, disparity=disp)
                q = dst / "right" / right_files[f].name
                q.unlink()
                _write(q, outr, grayr)
            if depth_files and kind in ("person", "object", "blank"):
                pd = depth_files[f]
                q = dst / "depth" / pd.name
                if pd.suffix == ".npy":
                    d = np.load(pd).astype(np.float32)
                    mm = cv2.resize(m, (d.shape[1], d.shape[0])) > 0.5
                    d[mm] = occ.z
                    q.unlink()
                    np.save(q, d)
                else:
                    d = cv2.imread(str(pd), cv2.IMREAD_UNCHANGED)
                    mm = cv2.resize(m, (d.shape[1], d.shape[0])) > 0.5
                    d[mm] = int(round(occ.z * 1000))
                    q.unlink()
                    cv2.imwrite(str(q), d)
            manifest["frames"][str(f)] = {"kind": kind, "burst": bi, "coverage": round(float(m.mean()), 3)
                                          if kind in ("person", "object", "blank") else 1.0}
    (dst / "junk_manifest.json").write_text(json.dumps(manifest, indent=1))
    n_j = len(manifest["frames"])
    print(f"{name}: {len(bursts)} bursts, {n_j} of {n} frames ({n_j / n:.0%}) -> {dst}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bank")
    b.add_argument("--person-frames", nargs="+", required=True, help="image folders with people (cutouts)")
    b.add_argument("--texture-frames", nargs="+", required=True, help="image folders of other scenes (close-ups)")
    b.add_argument("--out", required=True)
    b.add_argument("--n-person", type=int, default=40)
    b.add_argument("--n-texture", type=int, default=40)
    b.add_argument("--frames-per-dir", type=int, default=200)
    b.add_argument("--seed", type=int, default=0)
    i = sub.add_parser("inject")
    i.add_argument("--src", required=True)
    i.add_argument("--dst", required=True)
    i.add_argument("--bank", required=True)
    i.add_argument("--seed", type=int, default=0)
    i.add_argument("--frac", type=float, default=0.2, help="fraction of the frames in junk bursts")
    i.add_argument("--min-len", type=int, default=5)
    i.add_argument("--max-len", type=int, default=20)
    i.add_argument("--kinds", default=",".join(KINDS))
    a = ap.parse_args()
    build_bank(a) if a.cmd == "bank" else inject(a)


if __name__ == "__main__":
    main()
