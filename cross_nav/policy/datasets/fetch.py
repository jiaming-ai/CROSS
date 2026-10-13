"""Download a list of public files (Hugging Face dataset repos, Google Cloud Storage buckets) and verify them.

    # write a manifest of the files to fetch
    python -m cross_nav.policy.datasets.fetch hf franciszzj/SCAND --include 'random_mdps/*Jackal*.bag' --manifest M.json
    python -m cross_nav.policy.datasets.fetch gcs gresearch robotics/berkeley_gnm_sac_son/0.1.0/ --manifest M.json
    # download (sequentially; each file with parallel HTTP range requests), verify size and checksum
    python -m cross_nav.policy.datasets.fetch get M.json --out RAW_DIR [--workers 8]

Manifest rows: {"url", "path" (relative to --out), "size", optional "sha256" (HF LFS oid) or "md5" (GCS, base64)}.
A verified file gets a "<file>.ok" marker, so reruns skip it. "--select every:K" / "--select first:N" take a
subset of the matched files (sorted by path) for datasets too large to fetch completely.
"""
import argparse, base64, fnmatch, hashlib, json, os, sys, time, urllib.parse, urllib.request
from pathlib import Path

from .parallel_download import download


def _json(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)


def hf_rows(repo, include, exclude=(), revision='main'):
    tree = _json(f'https://huggingface.co/api/datasets/{repo}/tree/{revision}?recursive=true')
    rows = []
    for x in tree:
        if x['type'] != 'file': continue
        p = x['path']
        if not any(fnmatch.fnmatch(p, g) for g in include) or any(fnmatch.fnmatch(p, g) for g in exclude): continue
        lfs = x.get('lfs') or {}
        rows.append(dict(url=f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{urllib.parse.quote(p)}',
                         path=p, size=int(lfs.get('size', x.get('size', 0))), sha256=lfs.get('oid')))
    return rows


def gcs_rows(bucket, prefix, include=('*',)):
    rows, token = [], None
    while True:
        q = dict(prefix=prefix, fields='items(name,size,md5Hash),nextPageToken')
        if token: q['pageToken'] = token
        d = _json(f'https://storage.googleapis.com/storage/v1/b/{bucket}/o?' + urllib.parse.urlencode(q))
        for x in d.get('items', []):
            if x['name'].endswith('$folder$') or not any(fnmatch.fnmatch(x['name'], g) for g in include): continue
            rows.append(dict(url=f'https://storage.googleapis.com/{bucket}/{urllib.parse.quote(x["name"])}',
                             path=x['name'], size=int(x['size']), md5=x.get('md5Hash')))
        token = d.get('nextPageToken')
        if not token: return rows


def select(rows, spec):
    rows = sorted(rows, key=lambda r: r['path'])
    if not spec: return rows
    kind, k = spec.split(':'); k = int(k)
    return rows[::k] if kind == 'every' else rows[:k]


def verify(path, row):
    if os.path.getsize(path) != row['size']: return False
    if row.get('sha256'):
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
        return h.hexdigest() == row['sha256']
    if row.get('md5'):
        h = hashlib.md5()
        with open(path, 'rb') as f:
            for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
        return base64.b64encode(h.digest()).decode() == row['md5']
    return True


def get(rows, out, workers=8):
    out = Path(out); t0 = time.time(); total = sum(r['size'] for r in rows); done = 0
    for r in rows:
        dest = out / r['path']; dest.parent.mkdir(parents=True, exist_ok=True)
        ok = Path(str(dest) + '.ok')
        if not ok.exists():
            for attempt in range(3):
                download(r['url'], str(dest), workers=workers, chunk=64 << 20)
                if verify(dest, r): break
                print('checksum mismatch, refetching', dest, flush=True); os.remove(dest)
            else:
                raise RuntimeError(f'cannot verify {dest}')
            ok.write_text(json.dumps(r))
        done += r['size']
        print(f'[fetch] {done/1e9:.1f}/{total/1e9:.1f} GB  {(time.time()-t0)/60:.0f} min  {r["path"]}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    h = sub.add_parser('hf'); h.add_argument('repo'); h.add_argument('--include', nargs='+', default=['*'])
    h.add_argument('--exclude', nargs='*', default=[]); h.add_argument('--revision', default='main')
    g = sub.add_parser('gcs'); g.add_argument('bucket'); g.add_argument('prefix'); g.add_argument('--include', nargs='+', default=['*'])
    for s in (h, g):
        s.add_argument('--manifest', required=True); s.add_argument('--select', default=None)
    d = sub.add_parser('get'); d.add_argument('manifest'); d.add_argument('--out', required=True); d.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    if a.cmd == 'get':
        get(json.loads(Path(a.manifest).read_text()), a.out, a.workers); return
    rows = hf_rows(a.repo, a.include, a.exclude, a.revision) if a.cmd == 'hf' else gcs_rows(a.bucket, a.prefix, a.include)
    rows = select(rows, a.select)
    Path(a.manifest).write_text(json.dumps(rows, indent=1))
    print(f'{len(rows)} files, {sum(r["size"] for r in rows)/1e9:.2f} GB -> {a.manifest}', file=sys.stderr)


if __name__ == '__main__':
    main()
