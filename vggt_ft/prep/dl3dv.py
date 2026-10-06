"""DL3DV-10K (official DL3DV-ALL-960P zips) -> scene cache, with depth labels from the released VGGT-Omega.

DL3DV ships images and COLMAP poses (nerfstudio transforms.json: OpenGL camera-to-world, intrinsics at 3840 x 2160,
negligible distortion) but no depth.  Depth labels come from the released model (Depth Anything 3's teacher-label
idea): chunks of `--chunk` consecutive frames go through VGGT-Omega; a similarity transform aligns its predicted
camera centres to the COLMAP ones (Umeyama), and its scale converts the predicted depth to COLMAP units.  A chunk whose
predicted cameras disagree with COLMAP (median rotation residual > --max-rot deg, or centre residual > --max-res of
the trajectory extent) gets no depth (its frames still supervise pose); low-confidence pixels (below the chunk's
--conf-q quantile) are left invalid.  The labels are multi-view consistent and keep the fine-tune close to the
released model's geometry on static scenes.  Scale is arbitrary (metric: false, units normalised by units_for).

python -m vggt_ft.prep.dl3dv <raw_root containing <batch>/<hash>.zip> <cache_root> --ckpt vggt_omega_1b_512.pt
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .common import SceneWriter, units_for

GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def umeyama(src: np.ndarray, dst: np.ndarray):
    """Similarity (s, R, t) with dst ~ s R src + t (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (xs ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / max(var, 1e-12))
    return s, R, mu_d - s * R @ mu_s


def load_model(ckpt: str, device="cuda"):
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.pose_enc import encoding_to_camera
    m = VGGTOmega()
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=False)
    if device == "cuda":
        m.aggregator.to(torch.bfloat16)
    return m.to(device).eval(), encoding_to_camera


@torch.no_grad()
def label_chunk(model, decode, imgs: list[np.ndarray], c2w_gt: np.ndarray, hw=(384, 688), conf_q=0.3,
                max_rot=3.0, max_res=0.05, device="cuda"):
    """imgs: RGB uint8 frames; c2w_gt (S,4,4) OpenCV.  Returns depth maps in COLMAP units at the image resolution
    (None for a rejected chunk) and a diagnostic dict."""
    H0, W0 = imgs[0].shape[:2]
    x = torch.from_numpy(np.stack([cv2.resize(im, (hw[1], hw[0]), interpolation=cv2.INTER_AREA) for im in imgs]))
    x = x.to(device).permute(0, 3, 1, 2).float()[None] / 255.0
    pred = model(x)
    E, _ = decode(pred["pose_enc"].float(), hw)
    E4 = torch.eye(4, device=device).repeat(len(imgs), 1, 1)
    E4[:, :3] = E[0]
    c2w_p = torch.linalg.inv(E4).double().cpu().numpy()
    s, R, t = umeyama(c2w_p[:, :3, 3], c2w_gt[:, :3, 3])
    rot_err = []
    for a, b in zip(c2w_p, c2w_gt):
        Ra = R @ a[:3, :3]
        c = (np.trace(Ra.T @ b[:3, :3]) - 1) / 2
        rot_err.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
    extent = np.linalg.norm(c2w_gt[:, :3, 3] - c2w_gt[:, :3, 3].mean(0), axis=1).max() + 1e-9
    res = np.linalg.norm((s * (R @ c2w_p[:, :3, 3].T)).T + t - c2w_gt[:, :3, 3], axis=1).max() / extent
    info = {"scale": s, "rot_med": float(np.median(rot_err)), "res": float(res)}
    if info["rot_med"] > max_rot or res > max_res or extent < 1e-6:
        return None, info
    d = pred["depth"][0, ..., 0].float()
    c = pred["depth_conf"][0].float()
    thr = torch.quantile(c.flatten()[::7], conf_q)
    d = torch.where(c >= thr, d * s, torch.zeros_like(d))
    d = F.interpolate(d[:, None], size=(H0, W0), mode="nearest")[:, 0]
    return d.cpu().numpy(), info


def convert(zpath: Path, out_root: str, model, decode, every=2, chunk=12, device="cuda"):
    name = zpath.stem
    if (Path(out_root) / "dl3dv" / name / "scene.json").exists():
        return name, -1, {}
    z = zipfile.ZipFile(zpath)
    tf = [n for n in z.namelist() if n.endswith("transforms.json")][0]
    meta = json.loads(z.read(tf))
    root = tf.rsplit("/", 1)[0]
    frames = sorted(meta["frames"], key=lambda f: f["file_path"])[::every]
    files = set(z.namelist())
    img_dirs = sorted({n.split("/")[1] for n in files if n.count("/") == 2 and n.split("/")[1].startswith("images")})
    imgs, c2ws, names = [], [], []
    for f in frames:
        stem = Path(f["file_path"]).name
        p = next((f"{root}/{d}/{stem}" for d in img_dirs if f"{root}/{d}/{stem}" in files), None)
        if p is None:
            continue
        im = cv2.imdecode(np.frombuffer(z.read(p), np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
        imgs.append(np.ascontiguousarray(im))
        c2ws.append(np.asarray(f["transform_matrix"], np.float64) @ GL_TO_CV)
        names.append(Path(stem).stem)
    if len(imgs) < 8:
        return name, 0, {}
    H, W = imgs[0].shape[:2]
    sx, sy = W / meta["w"], H / meta["h"]
    K = np.array([[meta["fl_x"] * sx, 0, (meta["cx"] + 0.5) * sx - 0.5],
                  [0, meta["fl_y"] * sy, (meta["cy"] + 0.5) * sy - 0.5], [0, 0, 1.0]])
    c2ws = np.stack(c2ws)
    depths = [None] * len(imgs)
    infos, step = [], max(1, chunk - 4)
    for s0 in range(0, len(imgs), step):
        idx = list(range(s0, min(s0 + chunk, len(imgs))))
        if len(idx) < 4:
            idx = list(range(max(0, len(imgs) - chunk), len(imgs)))
        d, info = label_chunk(model, decode, [imgs[i] for i in idx], c2ws[idx], device=device)
        infos.append(info)
        if d is not None:
            centre = (idx[0] + idx[-1]) / 2
            for k, i in enumerate(idx):      # keep the label from the chunk where the frame is most central
                if depths[i] is None or abs(i - centre) < depths[i][1]:
                    depths[i] = (d[k], abs(i - centre))
    good = [d[0] for d in depths if d is not None]
    units = units_for(good) if good else 1.0
    sw = SceneWriter(out_root, "dl3dv", name, metric=False, kind="mixed", units=units,
                     extra={"depth_source": "vggt_omega_1b_512 teacher, Sim3-aligned to COLMAP"})
    q = sw.sequence("seq")
    for i in range(len(imgs)):
        dep = depths[i][0] if depths[i] is not None else np.zeros((H, W), np.float32)
        q.add(names[i], imgs[i], dep, K, np.linalg.inv(c2ws[i]))
    return name, sw.close(), {"chunks": len(infos), "kept": sum(1 for i in infos if i["rot_med"] <= 3.0 and i["res"] <= 0.05),
                              "rot_med": float(np.median([i["rot_med"] for i in infos]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    model, decode = load_model(a.ckpt, a.device)
    zips = sorted(p for p in Path(a.raw).glob("*/*.zip") if Path(str(p) + ".done").exists())
    if a.limit:
        zips = zips[:a.limit]
    for zp in zips:
        try:
            name, n, info = convert(zp, a.out, model, decode, a.every, device=a.device)
        except Exception as e:      # noqa: BLE001  a broken zip must not stop the batch
            print(zp.name, "error", e, flush=True)
            continue
        if n >= 0:
            print(name, n, info, flush=True)


if __name__ == "__main__":
    main()
