"""Usage: python scripts/vggt_ft/gauge_check.py <ckpt> (from the repo root; reads configs/vggt_ft/eval_suite.yaml windows).
   Gauge of the released VGGT-Omega vs our GT normalisation (unit mean point distance of the window, frame 0 at the
origin): per window, median of predicted depth / normalised GT depth, and median of |t_pred| / |t_gt| over frames > 0."""
import json, sys
import numpy as np, torch, yaml
from pathlib import Path
from vggt_omega.utils.pose_enc import encoding_to_camera
from vggt_ft.dataio.scene import load_scene
from vggt_ft.dataio.windows import materialise
from vggt_ft.dataio.transforms import target_shape
from vggt_ft.geometry import normalize_window, gt_covisibility, connected_to_ref
from vggt_ft.evaluate import load_model

model = load_model(sys.argv[1])
suite = yaml.safe_load(open("configs/vggt_ft/eval_suite.yaml"))
wd = Path(suite["windows_dir"])
out = {}
for name in ["eth3d", "nrgbd", "dtu", "hiroom", "7scenes", "openloris_seq", "simchange_seq", "tum_dynamic"]:
    ts = next(t for t in suite["sets"] if t["name"] == name)
    wins = json.load(open(wd / f"{name}.json"))[:40]
    hw = target_shape(ts.get("aspect", 0.75))
    rd, rt = [], []
    cache = {}
    for spec in wins:
        frames = []
        for d, q, i in spec:
            sc = cache.setdefault(d, load_scene(Path(d)))
            frames.append((sc, sc.seq(q), i))
        b = materialise(frames, hw)
        with torch.no_grad():
            pred = model(b["images"][None].cuda())
        dep, msk = b["depths"][None].cuda(), b["masks"][None].cuda()
        E, K = b["extrinsics"][None].cuda(), b["intrinsics"][None].cuda()
        conn = connected_to_ref(gt_covisibility(dep, msk, E, K))
        E_n, d_n, _, s = normalize_window(E, dep, msk, K, conn)
        m = msk & conn[:, :, None, None]
        if m.sum() < 100:
            continue
        rd.append(float((pred["depth"][..., 0].float()[m] / d_n[m]).median()))
        Ep, _ = encoding_to_camera(pred["pose_enc"].float(), hw)
        tp = Ep[0, 1:, :3, 3].norm(dim=-1)
        tg = E_n[0, 1:, :3, 3].norm(dim=-1)
        ok = (tg > 0.05) & conn[0, 1:]
        if ok.any():
            rt.append(float((tp[ok] / tg[ok]).median()))
    out[name] = {"depth_ratio_med": float(np.median(rd)), "depth_ratio_iqr": [float(np.quantile(rd, .25)), float(np.quantile(rd, .75))],
                 "trans_ratio_med": float(np.median(rt)) if rt else None, "n": len(rd)}
    print(name, json.dumps(out[name]), flush=True)
