#!/usr/bin/env python3
"""KITTI raw drives: in-drive revisits and cross-drive overlap from the OXTS ground truth.
usage: python scripts/lc/kitti_overlap.py <date_root> <drive> [<drive> ...]"""
import glob
import sys

import numpy as np

from cross.dataloader.kitti_calib import oxts_to_imu_poses


def first_revisit(P, gap=300, r=5.0):
    for i in range(gap + 5, len(P)):
        if np.linalg.norm(P[:i - gap] - P[i], axis=1).min() < r:
            return i
    return None


def main():
    root = sys.argv[1]
    tr = {}
    for d in sys.argv[2:]:
        fs = sorted(glob.glob(f"{root}/{d}/oxts/data/*.txt"))
        if not fs:
            print(d, "not extracted yet")
            continue
        P = oxts_to_imu_poses(fs)[:, :3, 3]
        tr[d] = P
        L = float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))
        n_rev, n_rev_same = 0, 0
        for i in range(305, len(P), 5):
            dd = np.linalg.norm(P[:i - 300] - P[i], axis=1)
            if dd.min() < 5:
                n_rev += 1
                j = int(np.argmin(dd))
                # same travel direction?
                if i + 1 < len(P) and j + 1 < len(P):
                    va, vb = P[i + 1] - P[i], P[j + 1] - P[j]
                    if np.dot(va, vb) > 0:
                        n_rev_same += 1
        print(f"{d}: {len(P)} frames, {L:.0f} m, revisit frames (< 5 m, >= 300 frames apart): {n_rev * 5} (same direction {n_rev_same * 5}); first revisit frame {first_revisit(P)}")
    ks = list(tr)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            A, B = tr[ks[i]], tr[ks[j]]
            d = np.min(np.linalg.norm(A[:, None, :2] - B[None, :, :2], axis=2), axis=1)
            print(f"{ks[i]} vs {ks[j]}: frames of the first within 10 m of the second: {(d < 10).sum()} / {len(A)}; min distance {d.min():.1f} m")


if __name__ == "__main__":
    main()

