#!/usr/bin/env python3
"""Download the subset of HSSD needed for given scenes.

Scene descriptions, semantics and stages come from hssd/hssd-hab; object render assets come from hssd/hssd-models
(uncompressed PNG/JPEG textures; the hssd-hab objects use KHR_texture_basisu, which Blender cannot import).
Decomposed parts (<hash>_part_N, only present in hssd-hab) are replaced by the whole source object <hash>.

    python scripts/sim/hssd_download.py --root data/sim/assets/hssd --scenes 106366323_174226647 103997940_171031257
"""
import argparse, json, os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

HAB = "hssd/hssd-hab"
MODELS = "hssd/hssd-models"


def object_repo_path(name: str) -> str:
    base = name.split("_part_")[0]
    if base.startswith("xxxx"):
        return f"objects/x/{base}.glb"
    if "-" in base and len(base) <= 8:
        return f"objects/openings/{base}.glb"
    return f"objects/{base[0]}/{base}.glb"


def fetch(repo, path, root, token):
    try:
        hf_hub_download(repo, path, repo_type="dataset", local_dir=str(root), token=token)
        return path, True, ""
    except Exception as e:  # noqa: BLE001
        return path, False, str(e)[:160]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or (Path.home() / ".cache/huggingface/token").read_text().strip()
    api = HfApi(token=token)
    for f in ["semantics/objects.csv", "semantics/hssd-hab_semantic_lexicon.json", "hssd-hab.scene_dataset_config.json", "scene_splits.yaml"]:
        fetch(HAB, f, root, token)
    try:
        for e in api.list_repo_tree(HAB, path_in_repo="metadata", repo_type="dataset"):
            fetch(HAB, e.path, root, token)
    except Exception as e:  # noqa: BLE001
        print("metadata listing failed:", e)
    for scene in args.scenes:
        print(f"== scene {scene}", flush=True)
        sj = f"scenes/{scene}.scene_instance.json"
        _, ok, err = fetch(HAB, sj, root, token)
        if not ok:
            print("cannot fetch scene json:", err); continue
        fetch(HAB, f"semantics/scenes/{scene}.semantic_config.json", root, token)
        d = json.loads((root / sj).read_text())
        stage = d["stage_instance"]["template_name"].split("/")[-1]
        names = sorted(set(o["template_name"] for o in d["object_instances"]))
        paths = sorted(set(object_repo_path(n) for n in names))
        parts = sorted(n for n in names if "_part_" in n)          # decomposed parts exist only in hssd-hab (KTX2 textures)
        print(f"   {len(d['object_instances'])} instances, {len(names)} templates -> {len(paths)} model files + {len(parts)} part files", flush=True)
        missing = []
        with ThreadPoolExecutor(args.workers) as ex:
            futs = {ex.submit(fetch, HAB, f"stages/{stage}.glb", root, token): "stage",
                    ex.submit(fetch, HAB, f"stages/{stage}.stage_config.json", root, token): "stage_cfg"}
            for p in paths:
                futs[ex.submit(fetch, MODELS, p, root, token)] = p
            for n in parts:
                base = n.split("_part_")[0]
                futs[ex.submit(fetch, HAB, f"objects/decomposed/{base}/{n}.glb", root, token)] = n
            done = 0
            for fu in as_completed(futs):
                path, ok, err = fu.result()
                done += 1
                if done % 100 == 0:
                    print(f"   {done}/{len(futs)}", flush=True)
                if not ok:
                    missing.append((path, err))
        print(f"   done; missing {len(missing)}", flush=True)
        for m in missing[:40]:
            print("   MISSING", m)
        (root / f"scenes/{scene}.missing.json").write_text(json.dumps(missing, indent=1))


if __name__ == "__main__":
    main()
