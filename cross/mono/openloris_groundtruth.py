"""Offline conversion from OpenLORIS base_link GT to D400 color camera poses.

Keep the dataset's common world coordinates across sessions. This module is
never imported by the image-only estimator. The published transform convention
is T_parent_child; camera GT is T_world_base @ T_base_color.
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .evaluate import read_trajectory


def camera_groundtruth(package):
    package = Path(package)
    handle = cv2.FileStorage(str(package/"trans_matrix.yaml"), cv2.FILE_STORAGE_READ)
    transforms = []
    try:
        entries = handle.getNode("trans_matrix")
        for i in range(entries.size()):
            entry = entries.at(i)
            if (entry.getNode("parent_frame").string() == "base_link" and
                    entry.getNode("child_frame").string() == "d400_color_optical_frame"):
                transforms.append(entry.getNode("matrix").mat())
    finally:
        handle.release()
    if len(transforms) != 1:
        raise ValueError("Expected one direct T_base_color calibration")
    transform = transforms[0]
    if (transform.shape != (4, 4) or not np.isfinite(transform).all() or
            not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8) or
            not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6) or
            not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-6)):
        raise ValueError("T_base_color must be a rigid transform")
    stamps, base = read_trajectory(package/"groundtruth.txt", sort_and_deduplicate=True)
    return stamps, base @ transform, transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    provenance = args.output.with_suffix(args.output.suffix + ".json")
    if args.output.exists() or provenance.exists():
        raise FileExistsError(args.output)
    stamps, poses, transform = camera_groundtruth(args.package)
    rows = np.column_stack((stamps, poses[:, :3, 3], Rotation.from_matrix(poses[:, :3, :3]).as_quat()))
    np.savetxt(args.output, rows, fmt="%.12f", header="T_world_color; shared dataset world retained")
    provenance.write_text(json.dumps(dict(
        frame="d400_color_optical_frame", world_alignment="none", T_base_color=transform.tolist(),
        groundtruth_normalization="stable timestamp sort; keep last duplicate",
        inputs={name: hashlib.sha256((args.package/name).read_bytes()).hexdigest()
                for name in ["groundtruth.txt", "trans_matrix.yaml"]}), indent=2) + "\n")


if __name__ == "__main__":
    main()
