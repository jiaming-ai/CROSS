"""Fetch selected public TUM/Bonn sequences; retain RGB and metadata only.

Run on a NAS-mounted compute server. --root is deliberately required.
"""

import argparse
import concurrent.futures
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path


def download(name, root):
    if name.startswith("freiburg"):
        group = name.split("_")[0]
        dirname = f"rgbd_dataset_{name}"
        url = f"https://cvg.cit.tum.de/rgbd/dataset/{group}/{dirname}.tgz"
        suffix = ".tgz"
    elif name.startswith("bonn_"):
        dirname = f"rgbd_{name}"
        url = f"https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/{dirname}.zip"
        suffix = ".zip"
    else:
        raise ValueError(f"Unsupported sequence: {name}")
    archive = root / (dirname + suffix)
    dest = root / dirname
    if (dest / ".rgb_complete").exists():
        return str(dest)
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-fL", "--retry", "3", "--connect-timeout", "30", "-o", str(archive), url], check=True)
    def keep(path):
        # Explicit allowlist also prevents archive traversal and link extraction.
        parts = Path(path).parts
        return ".." not in parts and not Path(path).is_absolute() and (
            "rgb" in parts or Path(path).name in {"rgb.txt", "groundtruth.txt", "accelerometer.txt", "README.txt"}
        )
    if suffix == ".tgz":
        with tarfile.open(archive) as handle:
            for member in handle:
                if member.isfile() and keep(member.name):
                    target = root / member.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with handle.extractfile(member) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
    else:
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                if not member.is_dir() and keep(member.filename):
                    target = root / member.filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with handle.open(member) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
    (dest / ".rgb_complete").write_text(url + "\n")
    return str(dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+", default=["freiburg1_desk", "freiburg1_xyz", "bonn_balloon2"])
    args = parser.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(lambda name: download(name, args.root), args.sequences):
            print(result, flush=True)
