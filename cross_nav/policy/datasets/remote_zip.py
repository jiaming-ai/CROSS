"""Read members of a large remote zip (HTTP range requests) without downloading the whole archive.

index(url, cd_path)  -> parse a locally downloaded central directory into arrays (name, offset, sizes, method)
fetch_span(url, entries) -> download one contiguous byte span covering the given entries and return their bytes
"""
import struct, time, urllib.request, zlib

import numpy as np


def eocd(url, size):
    """Return (n_entries, cd_size, cd_offset) from the (zip64) end-of-central-directory record."""
    tail = _range(url, size - 65536, size - 1)
    i = tail.rfind(b'PK\x06\x06')
    if i >= 0:
        _, _, _, _, _, _, _, ntot, cdsize, cdoff = struct.unpack('<IQHHIIQQQQ', tail[i:i + 56])
        return ntot, cdsize, cdoff
    i = tail.rfind(b'PK\x05\x06')
    _, _, _, _, ntot, cdsize, cdoff, _ = struct.unpack('<IHHHHIIH', tail[i:i + 22])
    return ntot, cdsize, cdoff


def _range(url, a, b, retries=8, size=None):
    if size is not None: b = min(b, size - 1)
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={'Range': f'bytes={a}-{b}'})
            with urllib.request.urlopen(req, timeout=300) as r: data = r.read()
            if len(data) == b - a + 1: return data
        except Exception:
            pass
        time.sleep(2 + 3 * k)
    raise RuntimeError(f'range {a}-{b} failed')


def parse_central_directory(buf):
    """Yield (name, local_header_offset, compressed_size, uncompressed_size, method) for each entry."""
    mv = memoryview(buf); p = 0; n = len(buf)
    while p + 46 <= n and mv[p:p + 4] == b'PK\x01\x02':
        (method, crc, csize, usize, nlen, xlen, clen, off) = struct.unpack_from('<H4xIIIHHH8xI', buf, p + 10)
        name = bytes(mv[p + 46:p + 46 + nlen]).decode('utf-8', 'replace')
        if 0xFFFFFFFF in (csize, usize, off):  # zip64 extra field
            x = p + 46 + nlen; end = x + xlen
            while x + 4 <= end:
                hid, hlen = struct.unpack_from('<HH', buf, x)
                if hid == 1:
                    q = x + 4
                    if usize == 0xFFFFFFFF: usize = struct.unpack_from('<Q', buf, q)[0]; q += 8
                    if csize == 0xFFFFFFFF: csize = struct.unpack_from('<Q', buf, q)[0]; q += 8
                    if off == 0xFFFFFFFF: off = struct.unpack_from('<Q', buf, q)[0]; q += 8
                    break
                x += 4 + hlen
        yield name, off, csize, usize, method
        p += 46 + nlen + xlen + clen


def member_bytes(span, span_start, offset, csize, method):
    """Extract one member from a downloaded span that starts at absolute offset span_start."""
    h = offset - span_start
    if span[h:h + 4] != b'PK\x03\x04': raise ValueError('bad local header')
    nlen, xlen = struct.unpack_from('<HH', span, h + 26)
    d = h + 30 + nlen + xlen
    raw = bytes(span[d:d + csize])
    if method == 0: return raw
    if method == 8: return zlib.decompress(raw, -15)
    raise ValueError(f'compression method {method}')


def fetch_span(url, offsets, csizes, size, slack=1 << 16):
    """Download the contiguous span covering members at offsets (+ local headers). Returns (bytes, start)."""
    a = int(min(offsets)); last = int(np.argmax(offsets))
    b = int(offsets[last] + csizes[last] + slack)
    return _range(url, a, b, size=size), a
