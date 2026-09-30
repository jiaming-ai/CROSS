#!/usr/bin/env python3
"""Quantify rearrangement variants from object-index passes rendered along the map trajectory.

For every station k (ids/<k>.npy of the map and of the variant, same camera):
    changed-object pixel ratio  CPR_obj    = |{p : id_map(p) in C or id_var(p) in C or id_map(p) != id_var(p)}| / N
    strict change ratio         CPR_strict = |{p : id_map(p) != id_var(p)}| / N
    object ratio                OBJ        = |{p : id_map(p) > 0}| / N            (pixels on any placed object)
    movable ratio               MOV        = |{p : id_map(p) in pool}| / N        (pixels on objects that could change)
where C is the set of pass indices (object index + 1) changed by the plan and pool the rearrangement pool.

    python scripts/sim/quantify_rearrangement.py --root temp/hssd_quant --map map --variants rearr_10,rearr_25,... --out outputs/hssd_quant
"""
import argparse, json
from pathlib import Path
import numpy as np


def load_ids(d):
    files = sorted((Path(d) / "ids").glob("*.npy"))
    idx = np.loadtxt(Path(d) / "frame_index.txt", dtype=int) if (Path(d) / "frame_index.txt").is_file() else np.arange(len(files))
    return {int(k): f for k, f in zip(np.atleast_1d(idx), files)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--map", default="map")
    ap.add_argument("--variants", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    map_ids = load_ids(root / args.map)
    results = {}
    for v in args.variants.split(","):
        vd = root / v
        if not (vd / "plan.json").is_file():
            print("no plan for", v); continue
        plan = json.loads((vd / "plan.json").read_text())
        changed = np.array(sorted({c["index"] + 1 for c in plan["changes"]}), dtype=np.int64)
        pool = np.array(sorted({i + 1 for i in plan.get("pool_indices", [])}), dtype=np.int64)
        var_ids = load_ids(vd)
        rows = []
        for k, fm in sorted(map_ids.items()):
            if k not in var_ids:
                continue
            a = np.load(fm).astype(np.int64); b = np.load(var_ids[k]).astype(np.int64)
            in_c = np.isin(a, changed) | np.isin(b, changed)
            diff = a != b
            rows.append((k, float((in_c | diff).mean()), float(diff.mean()), float((a > 0).mean()), float(np.isin(a, pool).mean()),
                         int(len(np.unique(a[in_c & (a > 0)])))))
        R = np.array(rows)
        results[v] = {"level": plan["level"], "seed": plan["seed"], "n_changed": plan["n_changed"], "pool_size": plan["pool_size"],
                      "changed_by_op": plan["changed_by_op"], "changed_by_class": plan["changed_by_class"],
                      "mean_displacement": plan["mean_displacement"],
                      "cpr_obj_mean": float(R[:, 1].mean()), "cpr_obj_median": float(np.median(R[:, 1])), "cpr_obj_p90": float(np.percentile(R[:, 1], 90)),
                      "cpr_strict_mean": float(R[:, 2].mean()), "obj_ratio_mean": float(R[:, 3].mean()), "movable_ratio_mean": float(R[:, 4].mean()),
                      "frac_frames_cpr_obj_gt_0.2": float((R[:, 1] > 0.2).mean()), "frac_frames_cpr_obj_gt_0.5": float((R[:, 1] > 0.5).mean()),
                      "visible_changed_objects_mean": float(R[:, 5].mean()),
                      "per_frame": {"station": R[:, 0].astype(int).tolist(), "cpr_obj": R[:, 1].round(4).tolist(), "cpr_strict": R[:, 2].round(4).tolist(),
                                    "obj_ratio": R[:, 3].round(4).tolist(), "movable_ratio": R[:, 4].round(4).tolist()}}
        print(f"{v:<18} level {plan['level']:.2f} seed {plan['seed']} changed {plan['n_changed']:>3}/{plan['pool_size']}  CPR_obj mean {R[:,1].mean():.3f} "
              f"median {np.median(R[:,1]):.3f} p90 {np.percentile(R[:,1],90):.3f} | strict {R[:,2].mean():.3f} | object px {R[:,3].mean():.3f} movable px {R[:,4].mean():.3f}")
    (out / "quantification.json").write_text(json.dumps(results, indent=1))
    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    levels = sorted({(r["level"], v) for v, r in results.items() if r["seed"] == 0 and "_remove" not in v and "_relocate" not in v and "_jitter" not in v})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    cmap = plt.get_cmap("viridis")
    for i, (lvl, v) in enumerate(levels):
        pf = results[v]["per_frame"]
        col = cmap(i / max(len(levels) - 1, 1))
        axes[0].plot(np.array(pf["station"]) * 0.1, pf["cpr_obj"], color=col, lw=1.2, label=f"{int(round(lvl*100))} % ({results[v]['n_changed']} objects)")
    if results:
        any_v = next(iter(results.values()))["per_frame"]
        axes[0].plot(np.array(any_v["station"]) * 0.1, any_v["obj_ratio"], color="k", lw=0.8, ls="--", label="all placed objects (ceiling)")
        axes[0].plot(np.array(any_v["station"]) * 0.1, any_v["movable_ratio"], color="gray", lw=0.8, ls=":", label="rearrangement pool")
    axes[0].set_xlabel("distance along the map trajectory [m]"); axes[0].set_ylabel("changed-object pixel ratio"); axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=7, ncol=2); axes[0].set_title("per-frame visual change")
    xs = [lvl for lvl, v in levels]; ys = [results[v]["cpr_obj_mean"] for _, v in levels]; p90 = [results[v]["cpr_obj_p90"] for _, v in levels]
    ys_s = [results[v]["cpr_strict_mean"] for _, v in levels]
    axes[1].plot(xs, ys, "o-", label="CPR (changed objects) mean"); axes[1].plot(xs, p90, "s--", label="CPR 90th percentile")
    axes[1].plot(xs, ys_s, "^-", label="strict (id differs) mean")
    seeds = [(r["level"], r["cpr_obj_mean"]) for v, r in results.items() if r["seed"] != 0]
    if seeds:
        axes[1].plot([s[0] for s in seeds], [s[1] for s in seeds], "x", color="k", label="other seeds")
    for key, mk in (("_remove", "v"), ("_relocate", "D"), ("_jitter", "P")):
        pts = [(r["level"], r["cpr_obj_mean"]) for v, r in results.items() if key in v]
        if pts:
            axes[1].plot([p[0] for p in pts], [p[1] for p in pts], mk, label=f"only {key[1:]}")
    axes[1].set_xlabel("rearrangement level (fraction of movable objects changed)"); axes[1].set_ylabel("mean pixel ratio"); axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=7); axes[1].set_title("level -> visual change")
    fig.tight_layout(); fig.savefig(out / "quantification.png", dpi=150)
    print("saved", out / "quantification.png")


if __name__ == "__main__":
    main()
