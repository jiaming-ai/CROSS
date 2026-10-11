"""Pack the exported viz folder into the page's data files: data/manifest.json, data/<set>__<dataset>.json (data URIs),
data/maps.json.   python pack.py <viz dir> <page dir> <checkpoint label>"""
import base64
import json
import sys
from pathlib import Path

viz, page, ck = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
data = page / "data"
data.mkdir(parents=True, exist_ok=True)
import shutil
shutil.copyfile(Path(__file__).resolve().parents[1] / "cross_nav/viewer/index.html", page / "index.html")


def uri(p: Path):
    mime = "image/png" if p.suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()


man = json.load(open(viz / "win" / "manifest.json"))
packs = {}
for it in man["items"]:
    d = viz / "win" / it["key"]
    refs = sorted(d.glob("r*.jpg"), key=lambda p: int(p.stem[1:]))
    packs.setdefault(f"{it['set']}__{it['dataset']}", {})[it["key"]] = {
        "f0": uri(d / "f0.jpg"), "refs": [uri(r) for r in refs[:7]], "gt": uri(d / "gt.png"), "po": uri(d / "po.jpg"),
        "pm": uri(d / "pm.jpg"), "err": uri(d / "err.png"), "errm": uri(d / "errm.png"), "seen": uri(d / "seen.png"),
        "pt": uri(d / "pt.png"), "ptm": uri(d / "ptm.png"), "geo": uri(d / "geo.png"), "geom": uri(d / "geom.png")}
    for name in ("uo", "um", "uo_selective", "um_selective"):
        if (d / (name + ".png")).exists():
            packs[f"{it['set']}__{it['dataset']}"][it["key"]][name] = uri(d / (name + ".png"))
for k, v in packs.items():
    json.dump(v, open(data / f"{k}.json", "w"))
json.dump({"checkpoint": ck, "backbone": "VGGT-Omega vdl_s1 (frozen)", "items": man["items"], "note": (viz / "note.txt").read_text() if (viz / "note.txt").exists() else ""}, open(data / "manifest.json", "w"))

# planning statistics over all windows (stats manifest of nav_viz_export.py --no-images)
st_path = viz / "stats" / "manifest.json"
if man["items"]:
    st = (json.load(open(st_path))["items"] if st_path.exists() else
          [{**it, "plan_status": {c: it["plan"][c]["status"] for c in ("oracle", "mono")},
            "plan": {c: it["plan"][c]["summary"] for c in ("oracle", "mono")}} for it in man["items"]])
    def agg(rows, cond, m):
        rs = [s for r in rows for s in r["plan_status"][cond][m]]
        n = len(rs)
        out = {k: (sum(x == k for x in rs) / n if n else 0.0) for k in ("success", "no_path", "collision", "detour")}
        # SPL: mean over goals of the per-window SPL weighted by goal count
        tot = sum(r["plan"][cond][m].get("spl", 0) * r["plan"][cond][m].get("n", 0) for r in rows)
        out["spl"] = tot / n if n else 0.0
        for metric in ("known_success", "unknown_path_rate"):
            if all(metric in r["plan"][cond][m] for r in rows if r["plan"][cond][m].get("n", 0)):
                out[metric] = sum(r["plan"][cond][m].get(metric, 0) * r["plan"][cond][m].get("n", 0)
                                  for r in rows) / n if n else 0.0
        out["n"] = n
        return out
    METHODS = ("head", "soft_risk", "soft_uncertainty", "head_t40", "head_t30", "geom", "geom_pess")
    by = {}
    for r in st:
        by.setdefault(f"{r['set']}/{r['dataset']}", []).append(r)
    ps = {"by_dataset": {k: {"windows": len(v), **{c: {m: agg(v, c, m) for m in METHODS if m in v[0]["plan_status"]["oracle"]} for c in ("oracle", "mono")}}
                         for k, v in by.items()},
          "total": {"windows": len(st), **{c: {m: agg(st, c, m) for m in METHODS if m in st[0]["plan_status"]["oracle"]} for c in ("oracle", "mono")}}}
    json.dump(ps, open(data / "plan_stats.json", "w"))
    for c in ("oracle", "mono"):
        print(c, {m: {k: round(v, 3) for k, v in ps["total"][c][m].items()} for m in ps["total"][c]})

MAPS = [("hssd_house", "SimChange house", "HSSD house, CROSS stereo map, 275 keyframes. Mesh ground truth: complete, walls filled."),
        ("hssd_restaurant", "SimChange restaurant", "HSSD restaurant, CROSS stereo map, 306 keyframes. Mesh ground truth."),
        ("home", "OpenLORIS home", "Real robot, T265 stereo map, 145 keyframes. Ground truth accumulated from all home sessions (RGB-D)."),
        ("cafe", "OpenLORIS cafe", "Real robot, T265 stereo map, 69 keyframes. Ground truth accumulated from all cafe sessions (RGB-D).")]
maps = []
m = viz / "maps"
for name, title, note in MAPS:
    if not (m / f"{name}_head.json").exists():
        continue
    jh, jg = json.load(open(m / f"{name}_head.json")), json.load(open(m / f"{name}_geom.json"))
    maps.append({"title": title, "note": note, "size": jh["size"], "traj": jh["traj"][::2],
                 "gt": uri(m / f"{name}_head_gt.png"), "head": uri(m / f"{name}_head_pred.jpg"),
                 "head_err": uri(m / f"{name}_head_err.png"), "geom": uri(m / f"{name}_geom_pred.jpg"),
                 "eval_head": jh["eval"], "eval_geom": jg["eval"], "plan_head": jh.get("plan"), "plan_geom": jg.get("plan")})
json.dump({"maps": maps}, open(data / "maps.json", "w"))
for p in sorted(data.iterdir()):
    print(p.name, round(p.stat().st_size / 1e6, 2), "MB")

# A local-file friendly version for sharing without a web server.
embedded = {"data/" + p.name: json.loads(p.read_text()) for p in data.glob("*.json")}
payload = json.dumps(embedded).replace("</", "<" + chr(92) + "/")
html = (page / "index.html").read_text()
html = html.replace("<script>", '<script type="application/json" id="embedded-data">' + payload + '</script><script>', 1)
(page / "standalone.html").write_text(html)
