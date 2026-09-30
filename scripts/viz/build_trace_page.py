#!/usr/bin/env python3
"""Build the interactive map-construction viewer from recorded traces of several scenes and methods.

Input layout (written by record_trace.py / record_baseline_trace.py):
  <viz_root>/<scene>/trace_<method>/trace.json

Output layout:
  <out>/index.html                      the viewer
  <out>/manifest.js                     scenes, methods, variants, per-session summaries
  <out>/traces/<scene>__<method>.js     one trace per scene and method
  <out>/frames/<scene>/<variant>/NNNNNN.jpg   left-camera frames (from the dataset, shared by all methods)
  <out>/standalone_<scene>.html         self-contained page of one scene (subsampled embedded images) with --standalone

Per trace the builder precomputes the keyframe pose history (with a "misplaced" flag for relocalization
keyframes that the graph places far from where they were captured), edge visibility windows, the first
loop-closure merge of every session, and the per-step errors: vs ground truth under both map->GT
alignments and the map-relative metric of the CROSS protocol.

Usage:
  python scripts/viz/build_trace_page.py --viz-root outputs/viz --scenes lonemonk classroom archiviz \
      --out outputs/viz/page --standalone
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
METHOD_LABELS = {
    "cross_stereo_verified": "CROSS-stereo (ours, 2026-09-12: verified loop closure, calibrated noise, online scale)",
    "cross_stereo_fixed": "CROSS-stereo (ours, 2026-09-08: renormalized poses, verified loop closures, gated intra-hypothesis PGO)",
    "cross_stereo": "CROSS-stereo (ours, run of 2026-09-05)", "cross_pnp_gt": "CROSS-PnP (GT depth)", "cross_pnp_sgbm": "CROSS-PnP (SGBM stereo depth)",
    "orbslam3": "ORB-SLAM3 stereo", "rtabmap_stereo": "RTAB-Map stereo", "rtabmap": "RTAB-Map RGB-D", "mast3r_slam": "MASt3R-SLAM",
}
METHOD_ORDER = list(METHOD_LABELS)
MISPLACED_M = 3.0


def quat_to_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def pose7_to_T(p):
    T = np.eye(4)
    T[:3, :3] = quat_to_R(p[3:7])
    T[:3, 3] = p[:3]
    return T


def rot_deg(R):
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64))     # project onto SO(3) (float32 quaternion drift)
    R = U @ np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))]) @ Vt
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def build(trace: dict, trim: bool = False):
    """Precompute everything the page needs; returns the data dict (images are referenced by variant + frame)."""
    steps = trace["steps"]
    sessions = trace["sessions"]
    edges = trace["edges"]
    n_steps = len(steps)
    external = bool(trace.get("external"))

    # ---- session-unique keyframe keys (keyframe ids restart after every map load in CROSS)
    raw = trace["nodes"]
    map_ids = [int(k) for k, v in raw.items() if ":" not in k and v["s"] == 0]
    thr = (max(map_ids) + 1) if map_ids else 0

    def nk(s, i):
        i = int(i)
        return i if (s == 0 or i < thr) else s * 100000 + i

    nodes = {}
    for k, v in raw.items():
        if ":" in k:
            s_, i = k.split(":")
            key = nk(int(s_), int(i))
        else:
            key = nk(v["s"], int(k))
        nodes[key] = dict(v)
        nodes[key].pop("rm", None)
    for t, st in enumerate(steps):
        s_ = st["s"]
        if st.get("nk") is not None:
            key = nk(s_, st["nk"])
            if key not in nodes or nodes[key]["step"] != t:
                nodes[key] = {"s": s_, "step": t, "f": st["f"], "perm": bool(st.get("nk_perm", True)), "p": st["mu"][0]}
            st["nk"] = key
        if "rm" in st:
            st["rm"] = [nk(s_, i) for i in st["rm"]]
            for key in st["rm"]:
                if key in nodes:
                    nodes[key]["rm"] = t
        if "upd" in st:
            st["upd"] = {nk(s_, i): p for i, p in st["upd"].items()}
        if "vk" in st:
            st["vk"] = [[nk(s_, v[0]), *v[1:]] for v in st["vk"]]
        if "ret" in st:
            st["ret"] = [[nk(s_, v[0]), v[1]] for v in st["ret"]]
        if "pgo" in st:
            st["pgo"]["lc_edges"] = [[nk(s_, a), nk(s_, b)] for a, b in st["pgo"].get("lc_edges", [])]
    for e in edges:
        e["a"], e["b"] = nk(e["s"], e["a"]), nk(e["s"], e["b"])

    for s in sessions:
        s["end_step"] = s["start_step"] + s["n_steps"]
        pg = [t for t in range(s["start_step"], s["end_step"]) if "pgo" in steps[t]]
        s["merge_step"] = pg[0] if (pg and s["kind"] != "map") else None
        s["n_lc"] = len(pg)

    # ---- keyframe pose history
    for nid, nd in nodes.items():
        nd["hist"] = [[nd["step"], nd["p"]]]
    for t, st in enumerate(steps):
        for nid, p in st.get("upd", {}).items():
            if nid in nodes:
                h = nodes[nid]["hist"]
                if h[-1][0] == t:
                    h[-1][1] = p
                else:
                    h.append([t, p])
    for t, st in enumerate(steps):   # keyframe created in a PGO step: its post-PGO pose is the re-aligned belief
        if "pgo" in st and st.get("nk") is not None and st["nk"] in nodes:
            h = nodes[st["nk"]]["hist"]
            hit = [e for e in h if e[0] == t]
            if hit:
                hit[0][1] = st["mu"][0]
            else:
                h.append([t, st["mu"][0]])
                h.sort(key=lambda e: e[0])
    for nid, nd in nodes.items():
        s = sessions[nd["s"]]
        nd["end"] = nd.get("rm", s["end_step"] if s["kind"] != "map" else n_steps)
        nd.pop("p", None)

    # ---- map-relative reference (CROSS protocol): nearest permanent map keyframe by ground truth
    kf_gt = {int(k): pose7_to_T(v) for k, v in trace.get("kf_gt", {}).items()}
    fin = {int(k): pose7_to_T(v) for k, v in trace.get("map_final_nodes", {}).items()}
    ids = [k for k in fin if k in kf_gt and k in nodes and nodes[k]["perm"]]
    Gm = np.array([kf_gt[k] for k in ids]) if ids else np.zeros((0, 4, 4))
    Mm = np.array([fin[k] for k in ids]) if ids else np.zeros((0, 4, 4))
    Gm_inv = np.array([np.linalg.inv(g) for g in Gm]) if ids else Gm

    def map_pred(G):
        j = int(np.argmin(np.linalg.norm(Gm[:, :3, 3] - G[:3, 3], axis=1)))
        return Mm[j] @ Gm_inv[j] @ G

    # misplaced flag: relocalization keyframes whose graph position is far from the map-implied position of
    # the frame they were captured at (created under an arbitrary initial pose, or dragged by a re-merge)
    for nid, nd in nodes.items():
        if nd["s"] == 0 or not ids:
            continue
        G = pose7_to_T(steps[nd["step"]]["gt"])
        pred = map_pred(G)[:3, 3]
        for h in nd["hist"]:
            if float(np.linalg.norm(np.asarray(h[1][:3]) - pred)) > MISPLACED_M:
                h.append(1)

    # ---- hypothesis branch snapshots per step (comp -> start kf id)
    hyp_at = []
    cur = {}
    for st in steps:
        if "hyp" in st:
            cur = {int(k): v for k, v in st["hyp"].items()}
        hyp_at.append(cur)

    # ---- edge visibility windows
    lc_step = {}
    for t, st in enumerate(steps):
        if "pgo" in st:
            for a, b in st["pgo"].get("lc_edges", []):
                lc_step.setdefault((a, b), t)
    out_edges = []
    for e in edges:
        s = sessions[e["s"]]
        end = s["end_step"] if s["kind"] != "map" else n_steps
        if e["t"] == "visual" and e["tc"] != 0:
            branch = hyp_at[e["step"]].get(e["tc"])
            for t in range(e["step"] + 1, end):
                if hyp_at[t].get(e["tc"]) != branch:
                    end = t
                    break
        rec = {"a": e["a"], "b": e["b"], "t": e["t"], "s": e["s"], "st": e["step"], "en": end, "c": e["tc"]}
        if (e["a"], e["b"]) in lc_step:
            rec["lc"] = lc_step[(e["a"], e["b"])]
            rec["en"] = s["end_step"] if s["kind"] != "map" else n_steps
        out_edges.append(rec)
    for nid, nd in nodes.items():   # odometry chain bridged over removed temporary keyframes
        if "rm" not in nd:
            continue
        t_rm = nd["rm"]
        preds = [e for e in out_edges if e["t"] == "odom" and e["b"] == nid and e["st"] <= t_rm]
        succs = [e for e in out_edges if e["t"] == "odom" and e["a"] == nid and e["st"] <= t_rm]
        if preds and succs:
            a, b = preds[-1]["a"], succs[-1]["b"]
            out_edges.append({"a": a, "b": b, "t": "odom", "s": nd["s"], "st": t_rm, "en": (sessions[nd["s"]]["end_step"] if sessions[nd["s"]]["kind"] != "map" else n_steps), "c": 0, "bridge": 1})

    # ---- per-step errors [first-kf: t0 r0 tb rb | umeyama: t0 r0 tb rb | map-relative: t0 r0 tb rb]
    T_first = np.asarray(trace["T_gt_from_map_first"])
    T_ume = np.asarray(trace["T_gt_from_map_umeyama"])
    for st in steps:
        G = pose7_to_T(st["gt"])
        w = np.asarray(st["w"])
        k_best = int(np.argmax(w))
        err = []
        if st.get("lost"):
            err = [None] * 12
        else:
            for T_al in (T_first, T_ume):
                for k in (0, k_best):
                    M = T_al @ pose7_to_T(st["mu"][k])
                    E = np.linalg.inv(G) @ M
                    err.append(round(float(np.linalg.norm(E[:3, 3])), 3))
                    err.append(round(rot_deg(E[:3, :3]), 2))
            if sessions[st["s"]]["kind"] == "map" and not external:
                err.extend(err[0:4])     # mapping session: drift vs GT (first keyframe)
            elif ids:
                pred_inv = np.linalg.inv(map_pred(G))
                for k in (0, k_best):
                    E = pred_inv @ pose7_to_T(st["mu"][k])
                    err.append(round(float(np.linalg.norm(E[:3, 3])), 3))
                    err.append(round(rot_deg(E[:3, :3]), 2))
            else:
                err.extend(err[4:8])
        st["err"] = err
        st["kb"] = k_best
        st.pop("hyp", None)
        st.pop("upd", None)
        st.pop("img", None)
        if trim:
            st.pop("sd", None)
            st.pop("ret", None)
            # inactive hypotheses are never drawn: keep only component 0 and the active ones
            st["mu"] = [[round(v, 3) for v in m] if (k == 0 or w[k] >= 1e-3) else None for k, m in enumerate(st["mu"])]
            st["gt"] = [round(v, 3) for v in st["gt"]]
            st["err"] = [None if v is None else round(v, 2) for v in st["err"]]
            if "vk" in st:
                st["vk"] = [[v[0], round(v[1], 2)] for v in st["vk"]]

    # ---- per-session summary (map-relative, component 0)
    for s in sessions:
        e = [steps[t]["err"][8] for t in range(s["start_step"], s["end_step"])]
        ok = [(v is not None and v < 2.0) for v in e]
        first = None
        for i in range(len(ok) - 4):
            if all(ok[i:i + 5]):
                first = i
                break
        vals = [v for v in e if v is not None]
        s["summary"] = {"frac_2m": round(float(np.mean(ok)), 3) if ok else None,
                        "median_err": round(float(np.median(vals)), 3) if vals else None,
                        "first_correct": first, "n_lost": int(sum(1 for v in e if v is None))}

    return {
        "scene": trace["scene"], "method": trace.get("method", "cross_stereo"), "external": external,
        "map_dir": trace["map_dir"], "map_variant": Path(trace["map_dir"]).name, "baseline": trace.get("baseline"), "snr": trace.get("snr"),
        "obs_cadence": trace.get("obs_cadence"), "n_components": trace["n_components"], "map_ate_rmse": trace["map_ate_rmse"],
        "T_first": T_first.tolist(), "T_umeyama": T_ume.tolist(),
        "sessions": sessions, "nodes": nodes, "edges": out_edges, "steps": steps,
    }


def make_frames(data_root: Path, scene: str, variants, out: Path, width: int):
    from PIL import Image
    made = 0
    for v in variants:
        src = data_root / scene / v / "left"
        dst = out / "frames" / scene / v
        if not src.is_dir():
            print(f"  ! no frames for {scene}/{v} ({src})")
            continue
        dst.mkdir(parents=True, exist_ok=True)
        files = sorted(src.glob("*.png"))
        if len(list(dst.glob("*.jpg"))) >= len(files):
            continue
        for i, f in enumerate(files):
            o = dst / f"{i:06d}.jpg"
            if o.exists():
                continue
            im = Image.open(f).convert("RGB")
            h = int(round(im.height * width / im.width))
            im.resize((width, h), Image.BILINEAR).save(o, quality=82)
            made += 1
    return made


def embed_images(data_root: Path, scene: str, variants, every: int, width: int):
    from PIL import Image
    imgs = {}
    for v in variants:
        src = data_root / scene / v / "left"
        if not src.is_dir():
            continue
        d = {}
        for i, f in enumerate(sorted(src.glob("*.png"))):
            if i % every:
                continue
            im = Image.open(f).convert("RGB")
            h = int(round(im.height * width / im.width))
            buf = io.BytesIO()
            im.resize((width, h), Image.BILINEAR).save(buf, format="JPEG", quality=55, optimize=True)
            d[i] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        imgs[v] = d
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viz-root", default="outputs/viz")
    ap.add_argument("--data-root", default="data/sim")
    ap.add_argument("--scenes", nargs="+", default=["lonemonk", "classroom", "archiviz"])
    ap.add_argument("--out", default="outputs/viz/page")
    ap.add_argument("--frame-width", type=int, default=320)
    ap.add_argument("--standalone", action="store_true")
    ap.add_argument("--standalone-methods", nargs="*", default=None, help="methods embedded in the standalone pages (default: all)")
    ap.add_argument("--embed-every", type=int, default=5)
    ap.add_argument("--embed-width", type=int, default=144)
    ap.add_argument("--no-frames", action="store_true", help="skip writing frames/ (standalone-only builds)")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "traces").mkdir(parents=True, exist_ok=True)
    template = (HERE / "trace_page.html").read_text()
    manifest = {"scenes": {}, "labels": METHOD_LABELS}
    for scene in args.scenes:
        sdir = Path(args.viz_root) / scene
        traces = sorted(p for p in sdir.glob("trace_*/trace.json"))
        if not traces:
            print(f"no traces for {scene}")
            continue
        methods = {}
        variants = set()
        for tp in traces:
            method = tp.parent.name[len("trace_"):]
            raw = json.loads(tp.read_text())
            raw.setdefault("method", method)
            data = build(raw)
            key = f"{scene}__{method}"
            (out / "traces" / f"{key}.js").write_text(f"window.TRACES=window.TRACES||{{}};window.TRACES[{json.dumps(key)}]=" + json.dumps(data, separators=(",", ":")) + ";")
            methods[method] = {"file": f"traces/{key}.js", "label": METHOD_LABELS.get(method, method),
                               "sessions": [{"variant": s["variant"], "kind": s["kind"], "n_steps": s["n_steps"], **s["summary"]} for s in data["sessions"]],
                               "map_ate": data["map_ate_rmse"], "n_steps": len(data["steps"])}
            variants.update(s["variant"] for s in data["sessions"])
            print(f"{key}: {len(data['steps'])} steps, {len(data['nodes'])} keyframes, {len(data['edges'])} edges")
        methods = {m: methods[m] for m in sorted(methods, key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)}
        n = 0 if args.no_frames else make_frames(Path(args.data_root), scene, sorted(variants), out, args.frame_width)
        manifest["scenes"][scene] = {"methods": methods, "variants": sorted(variants)}
        print(f"{scene}: {len(methods)} methods, {len(variants)} variants, {n} frames written")
    (out / "manifest.js").write_text("window.MANIFEST=" + json.dumps(manifest, separators=(",", ":")) + ";")
    (out / "index.html").write_text(template.replace("<!--DATA-->", '<script src="manifest.js"></script>'))
    print(f"wrote {out / 'index.html'}")

    if args.standalone:
        for scene, sc in manifest["scenes"].items():
            wanted = [m for m in sc["methods"] if args.standalone_methods is None or m in args.standalone_methods]
            parts = ["window.TRACES={};"]
            for m in wanted:
                raw = json.loads((Path(args.viz_root) / scene / f"trace_{m}" / "trace.json").read_text())
                raw.setdefault("method", m)
                data = build(raw, trim=True)
                parts.append(f"window.TRACES[{json.dumps(f'{scene}__{m}')}]=" + json.dumps(data, separators=(",", ":")) + ";")
            imgs = embed_images(Path(args.data_root), scene, sc["variants"], args.embed_every, args.embed_width)
            sub = {"scenes": {scene: {"methods": {m: dict(sc["methods"][m], file=None) for m in wanted}, "variants": sc["variants"]}}, "labels": METHOD_LABELS, "standalone": True}
            html = template.replace("<!--DATA-->", "<script>window.MANIFEST=" + json.dumps(sub, separators=(",", ":")) + ";" + "".join(parts) +
                                    "window.IMGS=" + json.dumps(imgs, separators=(",", ":")) + ";</script>")
            html = html.replace("<title>SimChange Map Replay</title>", f"<title>{ {'lonemonk': 'Lone Monk', 'classroom': 'Classroom', 'archiviz': 'Archiviz', 'hssd_house': 'HSSD House', 'hssd_restaurant': 'HSSD Restaurant'}.get(scene, scene.title()) } Map Replay</title>")
            p = out / f"standalone_{scene}.html"
            p.write_text(html)
            print(f"wrote {p} ({len(html.encode()) / 2**20:.1f} MB, methods: {', '.join(wanted)})")


if __name__ == "__main__":
    main()
