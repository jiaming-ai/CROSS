"""HSSD download for machines without access to huggingface.co (the H20s), in the layout of SimChange's
hssd_download.py (hssd-hab and hssd-models files under one root).

The machine with access (and the gated-dataset token) resolves the signed CDN URLs, the download machines fetch them:

    # where huggingface.co is reachable (HF_TOKEN set; listings from HfApi.list_repo_tree):
    python -m cross_nav.sim.hssd_dl plan --hab-list hab.json --models-list models.json --scene-dir <hab>/scenes \
        --out files.tsv [--exclude 104348010_171512832 103997718_171030855]
    python -m cross_nav.sim.hssd_dl resolve --files files.tsv --out-dir batches/ --batch 1500   # writes batch_*.json
    # on the download machines (any number; batches are claimed by renaming):
    python -m cross_nav.sim.hssd_dl fetch --queue <shared>/hssd/queue --root <shared>/hssd/raw --threads 32

Signed URLs are valid for about an hour, so batches are resolved just before they are fetched.  Decomposed object parts
(<hash>_part_N, only in hssd-hab with KTX2 textures) are not downloaded: hssd_to_blend.py then places the whole
source object.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HAB, MODELS = "hssd/hssd-hab", "hssd/hssd-models"
TEST_SCENES = ("104348010_171512832", "103997718_171030855")       # SimChange hssd_house / hssd_restaurant


def object_repo_path(name: str) -> str:
    """Same mapping as SimChange hssd_download.object_repo_path."""
    base = name.split("_part_")[0]
    if base.startswith("xxxx"):
        return f"objects/x/{base}.glb"
    if "-" in base and len(base) <= 8:
        return f"objects/openings/{base}.glb"
    return f"objects/{base[0]}/{base}.glb"


def plan(a):
    hab = dict(map(tuple, json.load(open(a.hab_list))))
    mod = dict(map(tuple, json.load(open(a.models_list))))
    files = []
    common = ["semantics/objects.csv", "semantics/hssd-hab_semantic_lexicon.json", "hssd-hab.scene_dataset_config.json",
              "scene_splits.yaml"] + [p for p in hab if p.startswith("metadata/")]
    files += [(HAB, p, hab[p]) for p in common if p in hab]
    models = set()
    scenes = sorted(p.name.split(".")[0] for p in Path(a.scene_dir).glob("*.scene_instance.json"))
    if a.only:
        scenes = [s for s in scenes if s in a.only]
    n_missing = 0
    for s in scenes:
        if s in a.exclude:
            continue
        d = json.load(open(Path(a.scene_dir) / f"{s}.scene_instance.json"))
        stage = d["stage_instance"]["template_name"].split("/")[-1]
        for p in (f"scenes/{s}.scene_instance.json", f"semantics/scenes/{s}.semantic_config.json",
                  f"stages/{stage}.glb", f"stages/{stage}.stage_config.json"):
            if p in hab:
                files.append((HAB, p, hab[p]))
        for o in d["object_instances"]:
            p = object_repo_path(o["template_name"])
            if p in mod:
                models.add(p)
            else:
                n_missing += 1
    files += [(MODELS, p, mod[p]) for p in sorted(models)]
    seen, out = set(), []
    for f in files:
        if f[:2] not in seen:
            seen.add(f[:2])
            out.append(f)
    with open(a.out, "w") as f:
        for r, p, s in out:
            f.write(f"{r}\t{p}\t{s}\n")
    print(f"{len(scenes)} scenes, {len(out)} files, {sum(s for *_, s in out) / 1e9:.1f} GB, "
          f"{n_missing} instances without a model file", flush=True)


def _resolve_one(r, token):
    import urllib.error
    import urllib.request

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *x, **k):
            return None

    opener = urllib.request.build_opener(NoRedirect)
    repo, path, size = r
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"
    req = urllib.request.Request(url, method="HEAD", headers={"Authorization": f"Bearer {token}"})
    for attempt in range(8):
        try:
            opener.open(req, timeout=30)
            return {"repo": repo, "path": path, "size": size, "url": None, "src": url}   # small git file: fetch here
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 307, 308):
                loc = e.headers["location"]
                if loc.startswith("/") or "://huggingface.co/" in loc:
                    # small files are served by huggingface.co itself (resolve-cache), not by the CDN: fetch here
                    return {"repo": repo, "path": path, "size": size, "url": None, "src": url}
                return {"repo": repo, "path": path, "size": int(e.headers.get("x-linked-size") or size), "url": loc}
            if e.code == 429:
                time.sleep(60 + 30 * attempt)
                continue
            print("HTTP", e.code, path, flush=True)
            return None
        except Exception as e:                       # noqa: BLE001
            print("err", e, path, flush=True)
            time.sleep(5)
    return None


def resolve(a):
    token = os.environ["HF_TOKEN"]
    rows = [l.rstrip("\n").split("\t") for l in open(a.files) if l.strip()]
    rows = [(r, p, int(s)) for r, p, s in rows]
    if a.skip_existing:
        have = set(l.strip() for l in open(a.skip_existing)) if Path(a.skip_existing).exists() else set()
        rows = [r for r in rows if r[1] not in have]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for b0 in range(0, len(rows), a.batch):
        part = rows[b0:b0 + a.batch]
        t0 = time.time()
        with ThreadPoolExecutor(a.threads) as ex:
            res = [x for x in ex.map(lambda r: _resolve_one(r, token), part) if x]
        local = [x for x in res if x["url"] is None]
        res = [x for x in res if x["url"] is not None]
        for x in local:                              # plain git files are not on the CDN: download them here
            import urllib.request
            dst = out / "local" / x["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            req = urllib.request.Request(x["src"], headers={"Authorization": f"Bearer {token}"})
            dst.write_bytes(urllib.request.urlopen(req, timeout=60).read())
        name = out / f"batch_{int(time.time())}_{b0:06d}.json"
        json.dump({"t": time.time(), "files": res}, open(name, "w"))
        print(f"{name.name}: {len(res)}/{len(part)} resolved in {time.time() - t0:.0f}s", flush=True)
        if a.push:
            os.system(a.push.replace("{}", str(name)))


def _fetch_one(f, root):
    import requests
    dst = Path(root) / f["path"]
    if "://huggingface.co/" in f["url"]:
        print("SKIP (not on the CDN)", f["path"], flush=True)
        return False, 0
    if dst.exists() and dst.stat().st_size == f["size"]:
        return True, 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + f".part{os.getpid()}")
    for attempt in range(6):
        try:
            with requests.get(f["url"], timeout=(20, 120), stream=True) as r:
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}")
                with open(tmp, "wb") as fh:
                    for blk in r.iter_content(1 << 20):
                        fh.write(blk)
            if tmp.stat().st_size != f["size"]:
                raise RuntimeError(f"size {tmp.stat().st_size} != {f['size']}")
            os.replace(tmp, dst)
            return True, f["size"]
        except Exception as e:                       # noqa: BLE001
            err = str(e)[:120]
            time.sleep(3 * (attempt + 1))
    try:
        tmp.unlink()
    except OSError:
        pass
    print("FAILED", f["path"], err, flush=True)
    return False, 0


def fetch(a):
    q = Path(a.queue)
    host = socket.gethostname()
    while True:
        cands = sorted(q.glob("batch_*.json"))
        if not cands:
            if a.once:
                return
            time.sleep(20)
            continue
        mine = q / f"{cands[0].stem}.{host}.work"
        try:
            os.rename(cands[0], mine)
        except OSError:
            continue
        b = json.load(open(mine))
        if time.time() - b["t"] > 3300:
            print(f"{mine.name}: URLs expired, returned for re-resolution", flush=True)
            os.rename(mine, q / f"{cands[0].stem}.expired")
            continue
        t0 = time.time()
        with ThreadPoolExecutor(a.threads) as ex:
            res = list(ex.map(lambda f: _fetch_one(f, a.root), b["files"]))
        ok = sum(r[0] for r in res)
        nb = sum(r[1] for r in res)
        failed = [f for f, r in zip(b["files"], res) if not r[0]]
        if failed:
            json.dump({"t": b["t"], "files": failed}, open(q / f"{cands[0].stem}.failed", "w"))
        try:
            os.rename(mine, q / f"{cands[0].stem}.{host}.done")
        except OSError:
            pass
        print(f"{time.strftime('%H:%M:%S')} {cands[0].name}: {ok}/{len(res)} ok, {nb / 1e9:.2f} GB in "
              f"{time.time() - t0:.0f}s ({nb / max(time.time() - t0, 1) / 1e6:.1f} MB/s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("plan")
    p.add_argument("--hab-list", required=True)
    p.add_argument("--models-list", required=True)
    p.add_argument("--scene-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--exclude", nargs="*", default=[])
    p = sp.add_parser("resolve")
    p.add_argument("--files", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--batch", type=int, default=1500)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--skip-existing", default=None, help="text file of repo paths already downloaded")
    p.add_argument("--push", default=None, help="shell command run per batch file, {} = its path")
    p = sp.add_parser("fetch")
    p.add_argument("--queue", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--once", action="store_true")
    a = ap.parse_args()
    {"plan": plan, "resolve": resolve, "fetch": fetch}[a.cmd](a)


if __name__ == "__main__":
    main()
