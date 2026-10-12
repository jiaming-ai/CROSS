"""Download large files (e.g. Hugging Face dataset files) with parallel HTTP range requests.

python -m cross_nav.policy.datasets.parallel_download URL OUT [--workers 16]
Resumable: finished chunks are recorded in OUT.parts.json.
"""
import argparse, json, os, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor


def _size(url):
    req = urllib.request.Request(url, method='HEAD')
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers['Content-Length'])


def _get(url, start, end, out, retries=8, offset=0):
    """Write remote bytes [offset+start, offset+end] to out[start:end+1]."""
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={'Range': f'bytes={offset+start}-{offset+end}'})
            with urllib.request.urlopen(req, timeout=120) as r, open(out, 'r+b') as f:
                f.seek(start); pos = start
                while True:
                    b = r.read(1 << 20)
                    if not b: break
                    f.write(b); pos += len(b)
            if pos == end + 1: return
            start = pos
        except Exception:  # network hiccup: retry from where we stopped
            time.sleep(2 + 3 * k)
    raise RuntimeError(f'chunk {start}-{end} failed')


def download(url, out, workers=16, chunk=256 << 20, start=0, end=None):
    """Download bytes [start, end) of url (whole file by default) into out."""
    total = _size(url)
    end = total if end is None else end
    n = end - start
    if os.path.exists(out) and os.path.getsize(out) == n and not os.path.exists(out + '.parts.json'):
        print('exists', out, n, flush=True); return out
    state = out + '.parts.json'
    done = set(json.load(open(state))) if os.path.exists(state) else set()
    if not os.path.exists(out):
        with open(out, 'wb') as f: f.truncate(n)
    lock = threading.Lock(); t0 = time.time()
    todo = [(s, min(s + chunk, n) - 1) for s in range(0, n, chunk) if s not in done]
    off = start

    def job(se):
        _get(url, se[0], se[1], out, offset=off)
        with lock:
            done.add(se[0]); json.dump(sorted(done), open(state, 'w'))
            got = len(done) * chunk
            print(f'{os.path.basename(out)} {min(got, n)/1e9:.1f}/{n/1e9:.1f} GB {got/1e6/max(time.time()-t0,1):.0f} MB/s', flush=True)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(job, todo))
    os.remove(state)
    print('done', out, n, flush=True)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('url'); ap.add_argument('out'); ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--start', type=int, default=0, help='first byte (inclusive)')
    ap.add_argument('--end', type=int, default=None, help='last byte (exclusive); default end of file')
    a = ap.parse_args()
    download(a.url, a.out, a.workers, start=a.start, end=a.end)
