#!/usr/bin/env python3
"""Fetch the 100 Hz OXTS (oxts/) and the image timestamps of the KITTI raw *_extract drives of the benchmark (the synced
drives carry 10 Hz OXTS only) without the images: the zip's central directory via remotezip, then one HTTP range request
over the contiguous byte span of the wanted members, parsed locally (~10 MB per drive).

  uv run --no-project --with remotezip --with requests python benchmark/datasets/fetch_kitti_oxts.py <out_root>

Writes <out_root>/<date>/<drive>_extract/{oxts/data/*.txt, oxts/timestamps.txt, image_0{2,3}/timestamps.txt}.
"""
import os, struct, sys, zlib
import requests
from remotezip import RemoteZip
DRIVES = ["2011_10_03_drive_0027", "2011_10_03_drive_0042", "2011_10_03_drive_0034", "2011_09_30_drive_0016",
          "2011_09_30_drive_0018", "2011_09_30_drive_0020", "2011_09_30_drive_0027", "2011_09_30_drive_0028",
          "2011_09_30_drive_0033", "2011_09_30_drive_0034"]
out = sys.argv[1]


def fetch(url, infos):
    infos = sorted(infos, key=lambda i: i.header_offset)
    groups, cur = [], [infos[0]]
    for i in infos[1:]:                                   # split where the gap is > 1 MB (timestamps sit elsewhere)
        if i.header_offset - (cur[-1].header_offset + cur[-1].compress_size + 1024) > (1 << 20):
            groups.append(cur); cur = [i]
        else:
            cur.append(i)
    groups.append(cur)
    for g in groups:
        lo = g[0].header_offset
        hi = g[-1].header_offset + g[-1].compress_size + 30 + 4096
        buf = requests.get(url, headers={"Range": f"bytes={lo}-{hi}"}, timeout=600).content
        for i in g:
            o = i.header_offset - lo
            sig, = struct.unpack("<I", buf[o:o + 4]); assert sig == 0x04034b50, i.filename
            nlen, elen = struct.unpack("<HH", buf[o + 26:o + 30])
            data = buf[o + 30 + nlen + elen: o + 30 + nlen + elen + i.compress_size]
            if i.compress_type == 8:
                data = zlib.decompress(data, -15)
            dst = os.path.join(out, i.filename)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            open(dst, "wb").write(data)


for d in DRIVES:
    url = f"https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/{d}/{d}_extract.zip"
    with RemoteZip(url) as z:
        infos = [i for i in z.infolist() if not i.filename.endswith("/") and (
            "/oxts/" in i.filename or i.filename.endswith("image_02/timestamps.txt")
            or i.filename.endswith("image_03/timestamps.txt"))]
    fetch(url, infos)
    print(d, len(infos), flush=True)
