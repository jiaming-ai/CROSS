"""Export a World: standard 3DGS PLY (any tool) and SPZ files plus the data of the web viewer.

SPZ (Niantic's compressed splat format, gzip of 24-bit fixed-point centres and 8-bit attributes; ~10x smaller than
PLY) is written in version 2, the layout of the Spark viewer's own encoder (rust/spark-lib/src/spz.rs, Spark 2.2),
which its decoder reads (versions 1-3).  Positions are stored relative to the partition origin (24-bit fixed point
with 12 fractional bits covers +-2 km at 0.24 mm).

The viewer data (`world.json` + `*.spz` + `thumbs.bin`):
    chunks       cell bounds, near / far files (a far layer is drawn only while the camera is in its cell)
    keyframes    id, pose, intrinsics, test / train, thumbnail (offset / length in thumbs.bin)
    edges        covisibility graph (hypothesis 0's visual edges)
    path         [id, x, y, z] of every keyframe, permanent and temporary, in creation order (the session path)
    view         rotation that brings the map's vertical to +Y (three.js)
"""
from __future__ import annotations

import gzip
import json
import struct
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from cross_world.world import World


def _prune(sp: Dict[str, torch.Tensor], min_opacity: float) -> Dict[str, torch.Tensor]:
    keep = torch.sigmoid(sp["opacities"]) >= min_opacity
    return {k: v[keep] for k, v in sp.items()}


def spz_bytes(means: np.ndarray, quats_wxyz: np.ndarray, log_scales: np.ndarray, opacity: np.ndarray,
              sh0: np.ndarray, shN: Optional[np.ndarray], sh_degree: int, fractional_bits: int = 12) -> bytes:
    """SPZ v2 of N splats.  means (N,3), quats (N,4) wxyz, log_scales (N,3), opacity (N,) in [0,1], sh0 (N,3) DC
    coefficients, shN (N, K-1, 3) higher bands (coefficient-major, RGB interleaved, as gsplat stores them)."""
    n = len(means)
    head = struct.pack("<IIIBBBB", 0x5053474E, 2, n, sh_degree, fractional_bits, 0, 0)
    frac = float(1 << fractional_bits)
    ip = np.clip(np.round(means.astype(np.float64) * frac), -0x7FFFFF, 0x7FFFFF).astype(np.int32)
    pos = ip.astype("<i4").view(np.uint8).reshape(n, 3, 4)[:, :, :3].reshape(-1)
    alpha = np.clip(np.round(opacity * 255.0), 0, 255).astype(np.uint8)
    rgb = np.clip(np.round((sh0 * 0.15 + 0.5) * 255.0), 0, 255).astype(np.uint8).reshape(-1)
    sc = np.clip(np.round((log_scales + 10.0) * 16.0), 0, 255).astype(np.uint8).reshape(-1)
    q = quats_wxyz / np.linalg.norm(quats_wxyz, axis=1, keepdims=True)
    q = q * np.where(q[:, :1] < 0, -1.0, 1.0)
    rot = np.clip(np.round((q[:, 1:4] + 1.0) * 127.5), 0, 255).astype(np.uint8).reshape(-1)
    parts = [head, pos.tobytes(), alpha.tobytes(), rgb.tobytes(), sc.tobytes(), rot.tobytes()]
    if sh_degree > 0 and shN is not None:
        nb = (sh_degree + 1) ** 2 - 1
        s = shN[:, :nb, :].reshape(n, nb * 3)

        def quant(x, bits):
            v = np.round(x * 128.0) + 128.0
            b = float(1 << (8 - bits))
            v = np.floor((v + b / 2) / b) * b
            return np.clip(np.round(v), 0, 255).astype(np.uint8)
        q1 = quant(s[:, :9], 5)
        blocks = [q1]
        if sh_degree >= 2:
            blocks.append(quant(s[:, 9:], 4))
        parts.append(np.concatenate(blocks, 1).tobytes())
    return gzip.compress(b"".join(parts), compresslevel=6, mtime=0)


def write_ply(path, sp: Dict[str, torch.Tensor], offset: np.ndarray = np.zeros(3)) -> None:
    """The INRIA 3DGS PLY layout (f_rest channel-major), readable by SuperSplat, Spark, gsplat viewers."""
    n = len(sp["means"])
    shN = sp["shN"].numpy()
    rest = shN.transpose(0, 2, 1).reshape(n, -1)
    cols = [("x", sp["means"].numpy()[:, 0] - offset[0]), ("y", sp["means"].numpy()[:, 1] - offset[1]),
            ("z", sp["means"].numpy()[:, 2] - offset[2])]
    cols += [("nx", np.zeros(n)), ("ny", np.zeros(n)), ("nz", np.zeros(n))]
    cols += [(f"f_dc_{i}", sp["sh0"].numpy()[:, 0, i]) for i in range(3)]
    cols += [(f"f_rest_{i}", rest[:, i]) for i in range(rest.shape[1])]
    cols += [("opacity", sp["opacities"].numpy())]
    cols += [(f"scale_{i}", sp["scales"].numpy()[:, i]) for i in range(3)]
    cols += [(f"rot_{i}", sp["quats"].numpy()[:, i]) for i in range(4)]
    arr = np.stack([c[1].astype(np.float32) for c in cols], 1)
    header = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n
    header += "".join(f"property float {c[0]}\n" for c in cols) + "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(arr.astype("<f4").tobytes())


def view_rotation(up: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """Rotation (rows = viewer x, y, z in map coordinates): the map's vertical -> +Y, the main ground axis -> +X."""
    y = up / np.linalg.norm(up)
    x = axes[0] - (axes[0] @ y) * y
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    return np.stack([x, y, z])


def _split_by_size(sp, max_bytes: float, bytes_per: float):
    n = len(sp["means"])
    k = max(1, int(np.ceil(n * bytes_per / max_bytes)))
    if k == 1:
        return [sp]
    order = torch.argsort(sp["means"][:, 0])               # spatially coherent parts (along one axis)
    idx = torch.tensor_split(order, k)
    return [{kk: v[i] for kk, v in sp.items()} for i in idx]


def export_world(world: World, out: Path, sh_degree: int = 1, mv=None, thumbs: int = 640, max_file_mb: float = 14.0,
                 min_opacity: float = 0.02, ply: bool = False, metrics: Optional[dict] = None,
                 renders_dir: Optional[Path] = None) -> dict:
    import cv2
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    part = world.partition
    origin = part.origin
    sh_degree = min(sh_degree, world.sh_degree)
    chunks_meta, total = [], 0
    # indoors, the Gaussians above the cameras (upper walls, ceiling) go to an "above" layer that the viewer's overview
    # hides (a dollhouse cut: from above, the ceiling hides the rooms)
    cam_h = np.array([(T[:3, 3] - origin) @ part.up for T in world.kf_poses.values()])
    zmed = float(world.meta.get("zmed", 1.0))
    cut = float(cam_h.max() + max(0.3, 0.25 * zmed)) if zmed < 5.0 and len(cam_h) else None
    for c in world.chunks:
        files = {}
        layers = dict(c.layers)
        if cut is not None and "near" in layers:
            nsp = layers["near"]
            hh = (nsp["means"].float().numpy() - origin) @ part.up
            hi = torch.from_numpy(hh > cut)
            layers["near"] = {k: v[~hi] for k, v in nsp.items()}
            layers["above"] = {k: v[hi] for k, v in nsp.items()}
        for layer, sp in layers.items():
            sp = _prune({k: v.float() for k, v in sp.items()}, min_opacity)
            if len(sp["means"]) == 0:
                continue
            bper = 19 + 3 * ((sh_degree + 1) ** 2 - 1)
            names = []
            for pi, p in enumerate(_split_by_size(sp, max_file_mb * 1e6 / 0.8, bper)):
                b = spz_bytes((p["means"].numpy() - origin), p["quats"].numpy(), p["scales"].numpy(),
                              torch.sigmoid(p["opacities"]).numpy(), p["sh0"].numpy()[:, 0, :],
                              p["shN"].numpy() if sh_degree > 0 else None, sh_degree)
                name = f"c{c.index:03d}_{layer}_{pi}.spz"
                (out / name).write_bytes(b)
                names.append({"file": name, "n": len(p["means"]), "bytes": len(b)})
                total += len(b)
            if ply:
                write_ply(out / f"c{c.index:03d}_{layer}.ply", sp, origin)
            files[layer] = names
        chunks_meta.append({"index": c.index, "lo": part.chunks[c.index].lo.tolist() if c.index < len(part.chunks) else None,
                            "hi": part.chunks[c.index].hi.tolist() if c.index < len(part.chunks) else None,
                            "files": files, "stats": {k: v for k, v in c.stats.items() if k != "history"}})
    # keyframes, graph, thumbnails
    kfs, edges, traj = [], [], []
    blob = bytearray()
    test_ids = set(world.meta.get("test_ids", []))
    if mv is not None:
        for v in sorted(mv.views, key=lambda v: v.id):
            T = v.T_wc.copy()
            T[:3, 3] -= origin
            row = {"id": v.id, "T": np.round(T, 5).reshape(-1).tolist(), "K": np.round(v.K, 3).reshape(-1).tolist(),
                   "w": v.width, "h": v.height, "t": v.timestamp, "test": v.id in test_ids,
                   "chunk": int(world.chunk_of(v.center))}
            if thumbs:
                img = v.image()
                s = thumbs / img.shape[1]
                small = cv2.resize(img, (thumbs, int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
                ok, enc = cv2.imencode(".jpg", small[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 80])
                row["thumb"] = [len(blob), len(enc)]
                blob += enc.tobytes()
            kfs.append(row)
        edges = [[a, b, round(c, 3)] for a, b, c in mv.edges]
        allp = {**{k: np.asarray(T) for k, T in mv.temporary_poses.items()}, **{v.id: v.T_wc for v in mv.views}}
        traj = [[int(k)] + (allp[k][:3, 3] - origin).round(3).tolist() for k in sorted(allp)]
    comps = []
    if renders_dir is not None and Path(renders_dir).is_dir():
        for sub in ("heldout", "novel"):          # photo above render, at most 8 of each, <= 640 px wide
            for p in sorted((Path(renders_dir) / sub).glob("view_*.jpg"))[:8]:
                im = cv2.imread(str(p))
                if im.shape[1] > 640:
                    im = cv2.resize(im, (640, int(round(im.shape[0] * 640 / im.shape[1]))), interpolation=cv2.INTER_AREA)
                ok, enc = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])
                comps.append({"set": sub, "view": p.stem, "img": [len(blob), len(enc)]})
                blob += enc.tobytes()
    if blob:
        (out / "thumbs.bin").write_bytes(bytes(blob))
    R = view_rotation(part.up, part.axes)
    manifest = {"version": 1, "origin": origin.tolist(), "view_rotation": R.tolist(), "up": part.up.tolist(),
                "axes": part.axes.tolist(), "region": [part.region_lo.tolist(), part.region_hi.tolist()],
                "sh_degree": sh_degree, "chunks": chunks_meta, "above_cut": cut, "keyframes": kfs, "edges": edges, "path": traj,
                "comparisons": comps, "meta": {k: v for k, v in world.meta.items() if k not in ("train_ids", "test_ids")},
                "metrics": metrics or {}, "splat_bytes": total}
    (out / "world.json").write_text(json.dumps(manifest, default=float))
    print(f"exported {sum(len(f) for c in chunks_meta for f in c['files'].values())} SPZ files, "
          f"{total / 1e6:.1f} MB, {len(kfs)} keyframes, thumbs {len(blob) / 1e6:.1f} MB -> {out}")
    return manifest
