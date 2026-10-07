#!/usr/bin/env python3
"""Parallel downloader for the NCLT dataset (University of Michigan North Campus Long-Term, Carlevaris-Bianco et al.,
IJRR 2016) from its public S3 mirror.

  python benchmark/datasets/nclt_download.py <raw_root> --kinds sensors gt cov calib [--sessions 2012-01-08 ...]
  python benchmark/datasets/nclt_download.py <raw_root> --kinds images --sessions 2012-01-08 --connections 16

One file is fetched with N parallel HTTP range requests written at their offsets into `<file>.part`, then renamed when
its size matches the server's Content-Length (a finished file is skipped, an interrupted one restarts its missing
ranges from a small `<file>.part.done` ledger).  The S3 mirror serves ~11 MB/s per connection.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import json
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

BASE = "https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu"
SESSIONS = [
    "2012-01-08", "2012-01-15", "2012-01-22", "2012-02-02", "2012-02-04", "2012-02-05", "2012-02-12", "2012-02-18",
    "2012-02-19", "2012-03-17", "2012-03-25", "2012-03-31", "2012-04-29", "2012-05-11", "2012-05-26", "2012-06-15",
    "2012-08-04", "2012-08-20", "2012-09-28", "2012-10-28", "2012-11-04", "2012-11-16", "2012-11-17", "2012-12-01",
    "2013-01-10", "2013-02-23", "2013-04-05",
]


def remote_path(kind: str, session: str) -> str:
    return {
        "sensors": f"sensor_data/{session}_sen.tar.gz",
        "gt": f"ground_truth/groundtruth_{session}.csv",
        "cov": f"covariance/cov_{session}.csv",
        "images": f"images/{session}_lb3.tar.gz",
        "velodyne": f"velodyne_data/{session}_vel.tar.gz",
    }[kind]


def content_length(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def _fetch_range(url: str, path: Path, start: int, end: int, retries: int = 20) -> int:
    """Bytes [start, end] of url written at their offset into path; resumes within the range after an error."""
    pos = start
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={pos}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r, open(path, "r+b") as f:
                f.seek(pos)
                while True:
                    buf = r.read(1 << 20)
                    if not buf:
                        break
                    f.write(buf)
                    pos += len(buf)
            if pos == end + 1:
                return end + 1 - start
        except Exception as ex:  # network hiccup: resume from pos
            print(f"  retry {attempt + 1} {path.name} [{pos}-{end}]: {ex}", file=sys.stderr, flush=True)
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"range {start}-{end} of {url} failed")


def download(url: str, dest: Path, connections: int = 8, chunk: int = 256 << 20) -> Path:
    if dest.exists():
        return dest
    size = content_length(url)
    part, ledger = dest.with_name(dest.name + ".part"), dest.with_name(dest.name + ".part.done")
    if not part.exists() or part.stat().st_size != size:
        with open(part, "wb") as f:
            f.truncate(size)
        ledger.write_text("")
    done = {int(x) for x in ledger.read_text().split()} if ledger.exists() else set()
    ranges = [(s, min(s + chunk, size) - 1) for s in range(0, size, chunk)]
    todo = [r for r in ranges if r[0] not in done]
    lock, t0, got = threading.Lock(), time.time(), [0]

    def work(r):
        n = _fetch_range(url, part, *r)
        with lock:
            got[0] += n
            with open(ledger, "a") as f:
                f.write(f"{r[0]}\n")
            el = time.time() - t0
            print(f"  {dest.name}: {got[0] / 1e9:.2f} / {sum(b - a + 1 for a, b in todo) / 1e9:.2f} GB "
                  f"({got[0] / max(el, 1e-3) / 1e6:.1f} MB/s)", flush=True)

    with cf.ThreadPoolExecutor(connections) as ex:
        list(ex.map(work, todo))
    assert part.stat().st_size == size
    part.rename(dest)
    ledger.unlink(missing_ok=True)
    return dest


class RangeStream(io.RawIOBase):
    """Read-only, sequential file object over a remote file, fetched with `connections` parallel range requests kept
    `ahead` chunks in front of the reader (memory ~ ahead x chunk).  Lets `tarfile.open(fileobj=..., mode="r|gz")`
    process a 100 GB archive at the full multi-connection bandwidth without storing it."""

    def __init__(self, url: str, connections: int = 8, chunk: int = 32 << 20, ahead: int = 24, start: int = 0):
        super().__init__()
        self.url, self.size = url, content_length(url)
        self.chunk, self.ahead = chunk, max(ahead, connections)
        self.pool = cf.ThreadPoolExecutor(connections)
        self.starts = list(range(start, self.size, chunk))
        self.futures = {}
        self.next_submit = 0
        self.buf, self.buf_pos, self.consumed = b"", 0, start
        self.t0 = time.time()
        self._fill()

    def _get(self, s: int) -> bytes:
        e = min(s + self.chunk, self.size) - 1
        for attempt in range(30):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={s}-{e}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                if len(data) == e - s + 1:
                    return data
                raise IOError(f"short read {len(data)} of {e - s + 1}")
            except Exception as ex:
                print(f"  retry {attempt + 1} range {s}: {ex}", file=sys.stderr, flush=True)
                time.sleep(min(60, 2 ** attempt))
        raise RuntimeError(f"range {s} of {self.url} failed")

    def _fill(self):
        while self.next_submit < len(self.starts) and len(self.futures) < self.ahead:
            s = self.starts[self.next_submit]
            self.futures[s] = self.pool.submit(self._get, s)
            self.next_submit += 1

    def readable(self):
        return True

    def readinto(self, b) -> int:
        if self.buf_pos >= len(self.buf):
            s = self.consumed
            if s >= self.size:
                return 0
            self.buf, self.buf_pos = self.futures.pop(s).result(), 0
            self.consumed += len(self.buf)
            self._fill()
        n = min(len(b), len(self.buf) - self.buf_pos)
        b[:n] = self.buf[self.buf_pos:self.buf_pos + n]
        self.buf_pos += n
        return n

    def rate(self) -> float:
        """MB/s fetched so far."""
        return self.consumed / max(time.time() - self.t0, 1e-3) / 1e6

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
        super().close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_root", type=Path)
    ap.add_argument("--kinds", nargs="+", default=["sensors", "gt", "cov", "calib"],
                    choices=["sensors", "gt", "cov", "calib", "images", "velodyne"])
    ap.add_argument("--sessions", nargs="+", default=SESSIONS)
    ap.add_argument("--connections", type=int, default=8)
    a = ap.parse_args()
    a.raw_root.mkdir(parents=True, exist_ok=True)
    if "calib" in a.kinds:
        download(f"{BASE}/ladybug3_calib/cam_params.zip", a.raw_root / "cam_params.zip", 1)
    for kind in [k for k in a.kinds if k != "calib"]:
        for s in a.sessions:
            rp = remote_path(kind, s)
            dest = a.raw_root / Path(rp).name
            t0 = time.time()
            download(f"{BASE}/{rp}", dest, a.connections if kind in ("images", "velodyne") else min(a.connections, 4))
            print(f"{dest.name}: {dest.stat().st_size / 1e6:.1f} MB in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
