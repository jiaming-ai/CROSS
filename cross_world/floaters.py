"""Floaters: Gaussians in space that the views see as empty.

A training view's stereo depth says the space along each ray up to the surface is free.  A Gaussian that projects in
front of that surface (camera depth < (1 - tol) x the view's depth at its pixel) violates it.  `free_space_votes`
counts, per Gaussian, the training views it violates and the views that see it at the surface (within tol);
`diagnose` reports the floaters (>= min_views violations) of a world and renders views off the trajectory.

    python -m cross_world.floaters --world W --map M --source S --out DIR
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from cross_world.gaussians import render, sh_to_rgb


@torch.no_grad()
def free_space_votes(means: torch.Tensor, opac: torch.Tensor, Ts: List[np.ndarray], Ks: List[np.ndarray],
                     depths: List[torch.Tensor], tol: float = 0.15, min_opacity: float = 0.05):
    """(violations, agreements) per Gaussian over the views (torch int32, on means' device).  Only Gaussians with
    opacity >= min_opacity vote; pixels without depth (0) cast no vote."""
    dev = means.device
    n = len(means)
    viol = torch.zeros(n, dtype=torch.int32, device=dev)
    agree = torch.zeros(n, dtype=torch.int32, device=dev)
    act = opac >= min_opacity
    for T, K, d in zip(Ts, Ks, depths):
        T_cw = torch.linalg.inv(torch.as_tensor(T, dtype=torch.float32, device=dev))
        Kt = torch.as_tensor(K, dtype=torch.float32, device=dev)
        d = d.float()
        p = means @ T_cw[:3, :3].T + T_cw[:3, 3]
        z = p[:, 2]
        h, w = d.shape
        uv = (p @ Kt.T)[:, :2] / z.clamp(min=1e-6)[:, None]
        u, v = uv[:, 0].floor().long(), uv[:, 1].floor().long()
        ok = act & (z > 0.1) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        dz = torch.zeros_like(z)
        dz[ok] = d[v[ok], u[ok]]
        ok &= dz > 0
        viol += (ok & (z < (1 - tol) * dz)).int()
        agree += (ok & ((z - dz).abs() <= tol * dz)).int()
    return viol, agree


def _shift(T: np.ndarray, right: float = 0.0, up: float = 0.0, yaw_deg: float = 0.0) -> np.ndarray:
    """A camera moved in its own frame (OpenCV: x right, y down) and turned about its vertical axis."""
    a = np.radians(yaw_deg)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    D = np.eye(4)
    D[:3, :3] = R
    D[:3, 3] = [right, -up, 0.0]
    return T @ D


def main(argv=None):
    from cross_world.depth import StereoDepth, view_depth
    from cross_world.map_views import load_map_views
    from cross_world.world import World
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--source", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", type=int, default=40, help="training views that vote")
    ap.add_argument("--tol", type=float, default=0.15)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    world = World.load(args.world)
    mv = load_map_views(args.map, args.source)
    by = mv.by_id()
    train = [by[i] for i in world.meta["train_ids"] if i in by]
    train = [train[i] for i in np.linspace(0, len(train) - 1, min(args.views, len(train))).round().astype(int)]
    st = StereoDepth(train[0].width)
    depths, Ts, Ks = [], [], []
    for v in train:
        d = view_depth(v, mv, "auto", None, st)
        depths.append(torch.from_numpy(d).to(args.device))
        Ts.append(v.T_wc @ world.pose_delta.get(v.id, np.eye(4)))
        Ks.append(v.K)
    sp = world.splats_for(None, args.device)
    opac = torch.sigmoid(sp["opacities"])
    viol, agree = free_space_votes(sp["means"], opac, Ts, Ks, depths, args.tol)
    flo = viol >= 2
    rgb = sh_to_rgb(sp["sh0"][:, 0]).clamp(0, 1)
    lum = rgb.mean(1)
    scale = torch.exp(sp["scales"]).max(1).values
    zmed = float(world.meta.get("zmed", 1.0))
    rep = {"n": len(opac), "opaque": int((opac >= 0.05).sum()), "floaters": int(flo.sum()),
           "floaters_share_of_opaque": float(flo.sum()) / max(1, int((opac >= 0.05).sum())),
           "floater_lum_quantiles": [float(x) for x in torch.quantile(lum[flo][:100000], torch.tensor([0.1, 0.5, 0.9], device=lum.device))] if flo.any() else None,
           "all_lum_quantiles": [float(x) for x in torch.quantile(lum[:100000], torch.tensor([0.1, 0.5, 0.9], device=lum.device))],
           "floater_dark_share": float((lum[flo] < 0.15).float().mean()) if flo.any() else None,
           "floater_bright_share": float((lum[flo] > 0.85).float().mean()) if flo.any() else None,
           "floater_opacity_median": float(opac[flo].median()) if flo.any() else None,
           "floater_scale_median_rel": float(scale[flo].median() / zmed) if flo.any() else None,
           "all_scale_median_rel": float(scale.median() / zmed),
           "sh_rest_energy_floaters": float(sp["shN"][flo].abs().mean()) if flo.any() else None,
           "sh_rest_energy_all": float(sp["shN"].abs().mean())}
    print(json.dumps(rep, indent=1))
    (out / "floaters.json").write_text(json.dumps(rep, indent=1))
    # renders: held-out keyframes as posed and moved off the trajectory, full SH vs degree 1, with / without floaters
    test = [by[i] for i in world.meta["test_ids"] if i in by][::6][:6]
    for v in test:
        sp = world.splats_for(v.center, args.device)             # the viewer's layers for this camera
        o_ = torch.sigmoid(sp["opacities"])
        vi, _ = free_space_votes(sp["means"], o_, Ts, Ks, depths, args.tol)
        keep = {k: x[vi < 2] for k, x in sp.items()}
        K = torch.from_numpy(v.K).float().to(args.device)[None]
        rows = []
        for tag, T in (("posed", v.T_wc), ("right1.5m", _shift(v.T_wc, right=1.5)), ("up1m_yaw10", _shift(v.T_wc, up=1.0, yaw_deg=10))):
            Tt = torch.from_numpy(T).float().to(args.device)[None]
            ims = []
            for s_, deg in ((sp, world.sh_degree), (sp, 1), (keep, world.sh_degree)):
                im, _, _, _ = render(s_, Tt, K, v.width, v.height, deg, render_mode="RGB")
                ims.append((im[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
            rows.append(np.concatenate(ims, 1))
        cv2.imwrite(str(out / f"view_{v.id}.jpg"), np.concatenate(rows, 0)[..., ::-1])
    print("renders: rows posed / right 1.5 m / up 1 m + yaw 10 deg; columns SH full / SH 1 / floaters removed")


if __name__ == "__main__":
    main()
