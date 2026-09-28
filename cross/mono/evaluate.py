"""Offline-only TUM-format evaluator; never imported by the estimator."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .geometry import inverse


def read_trajectory(path):
    rows = np.loadtxt(path, comments="#", ndmin=2)
    if rows.shape[1] != 8 or len(rows) < 3 or not np.isfinite(rows).all():
        raise ValueError("Expected at least three finite timestamp tx ty tz qx qy qz qw poses")
    if np.any(np.diff(rows[:, 0]) <= 0):
        raise ValueError("Trajectory timestamps must be strictly increasing")
    poses = np.broadcast_to(np.eye(4), (len(rows), 4, 4)).copy()
    poses[:, :3, 3] = rows[:, 1:4]
    poses[:, :3, :3] = Rotation.from_quat(rows[:, 4:]).as_matrix()
    return rows[:, 0], poses


def fit_alignment(source, target, with_scale):
    """Umeyama alignment mapping source camera centres into target."""
    xm, ym = source.mean(0), target.mean(0)
    x, y = source - xm, target - ym
    u, singular, vh = np.linalg.svd(y.T @ x / len(x))
    correction = np.diag([1., 1., np.linalg.det(u @ vh)])
    rotation = u @ correction @ vh
    variance = np.mean(np.sum(x*x, axis=1))
    if with_scale and variance < 1e-12:
        raise ValueError("Degenerate estimated trajectory: similarity scale is undefined")
    scale = float(np.sum(singular * np.diag(correction)) / variance) if with_scale else 1.0
    translation = ym - scale * rotation @ xm
    return scale, rotation, translation


def bonn_camera_poses(poses):
    """Apply the dataset's published mocap/ROS-to-camera calibration.

    https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/
    Its T_m contains a ~1.059 uniform scale from point-cloud calibration.
    Use its polar rotation for camera orientation, and its stated lever arm;
    do not rescale the already-metric mocap translations. Raw TUM-format ATE
    remains available as the default for comparison with common protocols.
    """
    ros = np.eye(4)
    ros[:3, :3] = [[-1., 0, 0], [0, 0, 1.], [0, 1., 0]]
    marker = np.array([[1.0157, 0.1828, -0.2389, 0.0113],
                       [0.0009, -0.8431, -0.6413, -0.0098],
                       [-0.3009, 0.6147, -0.8085, 0.0111], [0, 0, 0, 1.]])
    u, _, vh = np.linalg.svd(marker[:3, :3])
    marker[:3, :3] = u @ vh
    return ros.T @ poses @ ros @ marker


def evaluate(estimate_path, groundtruth_path, max_gap=0.1, rpe_interval=1.0, groundtruth_frame="raw"):
    times, est = read_trajectory(estimate_path)
    gt_times, gt = read_trajectory(groundtruth_path)
    if groundtruth_frame == "bonn-camera":
        gt = bonn_camera_poses(gt)
    elif groundtruth_frame != "raw":
        raise ValueError("Unknown ground-truth coordinate convention")
    idx = np.searchsorted(gt_times, times)
    left = np.clip(idx - 1, 0, len(gt_times) - 1)
    right = np.clip(idx, 0, len(gt_times) - 1)
    valid = ((times >= gt_times[0]) & (times <= gt_times[-1]) &
             (gt_times[right] - gt_times[left] <= max_gap))
    query, poses = times[valid], est[valid]
    if len(query) < 3:
        raise ValueError("Fewer than three ground-truth associations")
    truth = np.broadcast_to(np.eye(4), poses.shape).copy()
    for axis in range(3):
        truth[:, axis, 3] = np.interp(query, gt_times, gt[:, axis, 3])
    truth[:, :3, :3] = Slerp(gt_times, Rotation.from_matrix(gt[:, :3, :3]))(query).as_matrix()
    result = {"estimated_frames": len(times), "associated_frames": len(query),
              "association_fraction": float(valid.mean()), "duration_seconds": float(query[-1] - query[0]),
              "gt_path_length_m": float(np.linalg.norm(np.diff(truth[:, :3, 3], axis=0), axis=1).sum()),
              "groundtruth_frame": groundtruth_frame,
              "association": f"translation interpolation and quaternion SLERP; maximum GT gap {max_gap:g} s"}
    for mode, with_scale in [("se3", False), ("sim3", True)]:
        scale, rotation, translation = fit_alignment(poses[:, :3, 3], truth[:, :3, 3], with_scale)
        aligned = scale * poses[:, :3, 3] @ rotation.T + translation
        errors = np.linalg.norm(aligned - truth[:, :3, 3], axis=1)
        result[f"ate_{mode}_rmse_m"] = float(np.sqrt(np.mean(errors**2)))
        result[f"ate_{mode}_median_m"] = float(np.median(errors))
        if with_scale:
            result["sim3_alignment_scale"] = scale
            result["metric_scale_error_percent"] = 100 * abs(1 / scale - 1)
    # RPE uses estimated metric increments without a fitted similarity scale.
    trans_errors, rot_errors = [], []
    for i, timestamp in enumerate(query):
        j = int(np.searchsorted(query, timestamp + rpe_interval))
        if j >= len(query) or abs(query[j] - timestamp - rpe_interval) > 0.1:
            continue
        error = inverse(inverse(truth[i]) @ truth[j]) @ (inverse(poses[i]) @ poses[j])
        trans_errors.append(np.linalg.norm(error[:3, 3]))
        rot_errors.append(np.degrees(Rotation.from_matrix(error[:3, :3]).magnitude()))
    result["rpe_interval_seconds"] = rpe_interval
    result["rpe_pairs"] = len(trans_errors)
    result["rpe_translation_rmse_m"] = float(np.sqrt(np.mean(np.square(trans_errors)))) if trans_errors else None
    result["rpe_rotation_rmse_degrees"] = float(np.sqrt(np.mean(np.square(rot_errors)))) if rot_errors else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("estimate", type=Path)
    parser.add_argument("groundtruth", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--groundtruth-frame", choices=["raw", "bonn-camera"], default="raw")
    args = parser.parse_args()
    result = evaluate(args.estimate, args.groundtruth, groundtruth_frame=args.groundtruth_frame)
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
