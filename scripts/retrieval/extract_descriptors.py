"""BoQ descriptors (the keyframe database's VPR model) of posed image folders, for index / projection studies.

    python scripts/retrieval/extract_descriptors.py --out desc.npz --stride 5 --label openloris \
        $BENCH_DATA/openloris/office1-1/stereo $BENCH_DATA/openloris/office1-2/stereo

A folder holds left/ or rgb/ images (sorted by name) and poses_left.txt (16 values per row, camera-to-world).  The
output .npz has desc (N, 16384) float16 (L2-normalized), pos (N, 3) camera positions, seq (N,) folder index, frame (N,)
image index, label, folders.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def list_images(folder):
    for sub in ("left", "rgb", "image_left", "images"):
        d = os.path.join(folder, sub)
        if os.path.isdir(d):
            names = sorted(f for f in os.listdir(d) if f.lower().endswith((".png", ".jpg", ".jpeg")))
            return [os.path.join(d, f) for f in names]
    raise FileNotFoundError(f"no image folder in {folder}")


def load_positions(folder, n):
    p = os.path.join(folder, "poses_left.txt")
    if not os.path.exists(p):
        return np.full((n, 3), np.nan)
    P = np.loadtxt(p).reshape(-1, 4, 4)
    out = np.full((n, 3), np.nan)
    out[: min(n, len(P))] = P[: min(n, len(P)), :3, 3]
    return out


class _Images(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        import cv2
        im = cv2.imread(self.paths[i], cv2.IMREAD_COLOR)[:, :, ::-1]
        im = cv2.resize(im, (384, 384), interpolation=cv2.INTER_AREA)   # BoQ's input size (it resizes anyway)
        return torch.from_numpy(np.ascontiguousarray(im)).float() / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--label", default="")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    from cross.db.boq import BoQ
    model = BoQ(backbone_name="resnet50", device="cuda", enable_cache=False)
    descs, poss, seqs, frames = [], [], [], []
    for si, folder in enumerate(a.folders):
        paths = list_images(folder)
        pos = load_positions(folder, len(paths))
        idx = np.arange(0, len(paths), a.stride)
        dl = torch.utils.data.DataLoader(_Images([paths[i] for i in idx]), batch_size=a.batch,
                                         num_workers=a.workers)
        out = []
        for x in dl:
            e = model.get_embedding(x.cuda())
            out.append(e.reshape(-1, e.shape[-1]).float().cpu().half())
        D = torch.cat(out).numpy()
        descs.append(D); poss.append(pos[idx]); seqs.append(np.full(len(idx), si)); frames.append(idx)
        print(f"{folder}: {len(idx)} images", flush=True)
    np.savez(a.out, desc=np.concatenate(descs), pos=np.concatenate(poss), seq=np.concatenate(seqs),
             frame=np.concatenate(frames), label=np.array(a.label), folders=np.array(a.folders))


if __name__ == "__main__":
    main()
