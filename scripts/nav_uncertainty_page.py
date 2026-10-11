"""Build a standalone, CPU-only uncertainty inspector from saved prediction arrays.

Usage: python scripts/nav_uncertainty_page.py --windows windows.json --run 'Error=/path/win' \
    --run 'Attenuation=/other/win' --out page.html
The first three windows per scene are selected in manifest order, before inspecting predictions.
"""
import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image


def png(a):
    b = io.BytesIO()
    Image.fromarray(a).save(b, format="PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()



def add_supplement_scores(supplement, scores):
    """Add saved observed-prefix outcomes without changing example predictions."""
    if scores and not supplement:
        raise ValueError("Supplemental score updates require --supplement")
    for value in scores:
        label, path = value.split("=", 1)
        raw = json.loads(Path(path).read_text())
        if raw["protocol"] != "observed-prefix-v1" or raw["counts"]["local_opportunity"] != supplement["denominator"]:
            raise ValueError("Supplemental protocol or denominator mismatch")
        for method, result in raw["methods"].items():
            name = label + " / " + {"soft_risk": "occupancy risk", "soft_uncertainty": "risk + uncertainty"}.get(method, method)
            if sum(result["outcomes"].values()) != supplement["denominator"]:
                raise ValueError("Supplemental outcome counts do not sum to denominator")
            supplement["rows"] = [r for r in supplement["rows"] if r["label"] != name]
            supplement["rows"].append(dict(label=name, outcomes=result["outcomes"]))
        if path not in supplement["sources"]:
            supplement["sources"].append(path)
    return supplement


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--windows", type=Path, required=True)
    p.add_argument("--run", action="append", required=True, help="Label=export/win")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--kind", action="append", choices=("error", "attenuated", "unavailable"), help="Explicit uncertainty kind per run if export is still running")
    p.add_argument("--per-scene", type=int, default=3)
    p.add_argument("--note", default="")
    p.add_argument("--supplement-score", action="append", default=[], help="Label=observed_plan.json; append verified outcomes to --supplement")
    p.add_argument("--supplement", type=Path, help="JSON containing supplementary scores and embedded failure figures")
    p.add_argument("--candidate-run", type=Path, help="Current mono occupancy/uncertainty export, alongside historical comparisons")
    p.add_argument("--candidate-observability-root", type=Path, help="Current full-head observability predictions, INDEX.npz")
    p.add_argument("--observability-root", type=Path, help="Learned mono observability predictions, INDEX.npz / observed_probability")
    p.add_argument("--reserved-results", type=Path, help="Separately labelled frozen-candidate head-held-out summary JSON")
    p.add_argument("--local-stop-results", type=Path, help="JSON summary of explicitly labelled observed-local-stop-v2 trials")
    p.add_argument("--observed-root", type=Path, help="Directory of evaluation-only observed masks, named INDEX.npz")
    p.add_argument("--observed-score", action="append", default=[], help="Label=observed_plan.json")
    p.add_argument("--related-page", action="append", default=[], help="Label=relative.html; related input recipe")
    p.add_argument("--score", action="append", default=[], help="Label=planability.json; full benchmark summary, separate from shown examples")
    a = p.parse_args()
    windows = json.loads(a.windows.read_text())["windows"]
    if bool(a.candidate_run) != bool(a.candidate_observability_root):
        p.error("Supply both --candidate-run and --candidate-observability-root")
    candidate_manifest = json.loads((a.candidate_run / "manifest.json").read_text()) if a.candidate_run else None
    runs = [(label, Path(path)) for label, path in (v.split("=", 1) for v in a.run)]
    if a.kind and len(a.kind) != len(runs):
        p.error("Supply one --kind per --run")
    counts, rows = {}, []
    for i, w in enumerate(windows):
        scene, seq, frame = w["frames"][0]
        name = Path(scene).name
        if counts.get(name, 0) >= a.per_scene:
            continue
        files = [r / "arrays" / f"{a.windows.stem}_{w['dataset']}_{i}.npz" for _, r in runs]
        if not all(f.exists() for f in files):
            continue
        counts[name] = counts.get(name, 0) + 1
        predictions = []
        gt = None
        for (label, root), f in zip(runs, files):
            with np.load(f) as d:
                if gt is not None:
                    assert np.array_equal(gt, d["gt"]), "Paired GT mismatch"
                gt = d["gt"].copy()
                fields = {}
                for k in ("pm", "po", "um", "uo"):
                    fields[k] = png(np.rint(np.clip(d[k], 0, 1) * 255).astype(np.uint8)) if k in d else None
                predictions.append(dict(label=label, **fields))
        refs = []
        for sc, sq, fi in w["frames"]:
            folder = Path(sc) / sq / "rgb"
            candidates = [folder / f"{fi:06d}{ext}" for ext in (".jpg", ".png")]
            existing = next((x for x in candidates if x.exists()), None)
            if existing:
                im = Image.open(existing).convert("RGB"); im.thumbnail((320, 240))
                refs.append(png(np.asarray(im)))
        support = {}
        if a.observed_root:
            with np.load(a.observed_root / f"{i}.npz") as d:
                for key in ("observed", "obstacle_surface_seen"):
                    assert d[key].shape == gt.shape
                    support[key] = png(d[key].astype(np.uint8) * 255)
        if a.observability_root:
            with np.load(a.observability_root / f"{i}.npz") as d:
                probability = d["observed_probability"]
                assert probability.shape == gt.shape and np.isfinite(probability).all()
                support["observability"] = png(np.rint(np.clip(probability, 0, 1) * 255).astype(np.uint8))
        if a.candidate_run:
            with np.load(a.candidate_run / "arrays" / files[0].name) as c, np.load(a.candidate_observability_root / f"{i}.npz") as v, np.load(files[0]) as ref:
                for key in ("gt", "res", "seed"):
                    assert np.array_equal(c[key], ref[key]), (i, key, "candidate/reference mismatch")
                fields = {}
                for key, probability in (("pm", c["pm"]), ("um", c["um"]), ("observability", v["observed_probability"])):
                    assert probability.shape == gt.shape and np.isfinite(probability).all()
                    fields[key] = png(np.rint(np.clip(probability, 0, 1) * 255).astype(np.uint8))
                support["candidate"] = dict(label="Current frozen candidate", kind=candidate_manifest["uncertainty_kind"], **fields)
        rows.append(dict(**support, id=i, scene=name, frame=frame, views=len(w["frames"]), refs=refs,
                         gt=png(gt.astype(np.uint8)), predictions=predictions))
    assert rows, "No paired caches available"
    kinds = []
    for ri, (label, root) in enumerate(runs):
        m = json.loads((root / "manifest.json").read_text()) if (root / "manifest.json").exists() else {}
        kinds.append(dict(label=label, kind=m.get("uncertainty_kind", a.kind[ri] if a.kind else "unavailable"), checkpoint=m.get("heads", str(root))))
    scores = []
    for value in a.score:
        label, path = value.split("=", 1)
        raw = json.loads(Path(path).read_text())
        scores.append(dict(label=label, source=path, windows=len(raw["windows"]),
                           **{k: v for k, v in raw.items() if k != "windows"}))
    observed_scores = []
    for value in a.observed_score:
        label, path = value.split("=", 1)
        raw = json.loads(Path(path).read_text())
        observed_scores.append(dict(label=label, source=path, **{k: v for k, v in raw.items() if k != "records"}))
    payload = json.dumps(dict(candidate_provenance=dict(occupancy_head=candidate_manifest["heads"], observability_arrays=str(a.candidate_observability_root)) if candidate_manifest else None, reserved=json.loads(a.reserved_results.read_text()) if a.reserved_results else None, local_stop=json.loads(a.local_stop_results.read_text()) if a.local_stop_results else None, observability_root=str(a.observability_root) if a.observability_root else None, supplement=add_supplement_scores(json.loads(a.supplement.read_text()) if a.supplement else None, a.supplement_score), observed_scores=observed_scores, observed_root=str(a.observed_root) if a.observed_root else None, rows=rows, runs=kinds, scores=scores, note=a.note, source=str(a.windows), links=[dict(label=v.split("=", 1)[0], href=v.split("=", 1)[1]) for v in a.related_page])).replace("</", "<\\/")
    template = Path(__file__).resolve().parents[1] / "cross_nav/viewer/uncertainty.html"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(template.read_text().replace("__PAYLOAD__", payload))
    print(json.dumps({"output": str(a.out), "examples": len(rows), "scenes": counts, "bytes": a.out.stat().st_size}))


if __name__ == "__main__":
    main()
