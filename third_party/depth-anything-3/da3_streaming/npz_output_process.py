import argparse
import os
from glob import glob
import numpy as np
from loop_utils.sim3utils import save_confident_pointcloud_batch

from da3_streaming import depth_to_point_cloud_vectorized


def read_camera_poses(pose_file):
    poses = []
    with open(pose_file) as f:
        for line in f:
            if line.strip():
                numbers = list(map(float, line.strip().split()))
                if len(numbers) == 16:
                    pose = np.array(numbers).reshape(4, 4)
                    poses.append(pose)
    return poses


def create_point_cloud(npz_folder, pose_file, output_ply, conf_threshold_coef, sample_ratio=1.0):

    poses = read_camera_poses(pose_file)
    print(f"Read {len(poses)} camera poses")

    npz_files = sorted(glob(os.path.join(npz_folder, "frame_*.npz")))
    npz_files = [f for f in npz_files if os.path.basename(f).startswith("frame_")]

    npz_files.sort(key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]))

    print(f"Found {len(npz_files)} .npz files")

    assert len(poses) == len(
        npz_files
    ), f"Pose file has {len(poses)} lines, but npz folder has {len(npz_files)} files"

    all_points = []
    all_colors = []
    all_confs = []

    for idx, (npz_path, c2w) in enumerate(zip(npz_files, poses)):
        if idx % 50 == 0:
            print(f"Processing frame {idx}/{len(npz_files)}...")

        data = np.load(npz_path)

        image = data["image"]  # [H, W, 3] uint8
        depth = data["depth"]  # [H, W] float32
        conf = data["conf"]  # [H, W] float32
        intrinsics = data["intrinsics"]  # [3, 3] float32

        depth_reshaped = depth[np.newaxis, :, :]  # [1, H, W]
        intrinsics_reshaped = intrinsics[np.newaxis, :, :]  # [1, 3, 3]

        w2c = np.linalg.inv(c2w)
        extrinsics = w2c[:3, :]  # [3, 4]
        extrinsics_reshaped = extrinsics[np.newaxis, :, :]  # [1, 3, 4]

        points_world = depth_to_point_cloud_vectorized(
            depth_reshaped, intrinsics_reshaped, extrinsics_reshaped
        )
        points_world = points_world[0]  # [H, W, 3]

        all_points.append(points_world)
        all_colors.append(image)
        all_confs.append(conf)

    if not all_points:
        print("No valid point cloud data found!")
        return

    points_combined = np.stack(all_points, axis=0)  # [b, H, W, 3]
    colors_combined = np.stack(all_colors, axis=0)  # [b, H, W, 3]
    confs_combined = np.stack(all_confs, axis=0)  # [b, H, W]

    conf_threshold = np.mean(confs_combined) * conf_threshold_coef

    print(
        f"Original point cloud contains \
            {points_combined.shape[0] * points_combined.shape[1] * points_combined.shape[2]} points"
    )

    save_confident_pointcloud_batch(
        points=points_combined,
        colors=colors_combined,
        confs=confs_combined,
        output_path=output_ply,
        conf_threshold=conf_threshold,
        sample_ratio=sample_ratio,
        batch_size=1000000,
    )

    print(f"Downsampled point cloud saved to: {output_ply}")


def main():
    parser = argparse.ArgumentParser(description="Create point cloud from DA3-Long output")
    parser.add_argument("--npz_folder", type=str, required=True, help="Path to npz folder")
    parser.add_argument("--pose_file", type=str, required=True, help="Path to pose file")
    parser.add_argument(
        "--output_file", type=str, default="output.ply", help="Path to output PLY file"
    )
    parser.add_argument(
        "--conf_threshold_coef", type=float, default=0.5, help="Confidence threshold coefficient"
    )
    parser.add_argument(
        "--sample_ratio", type=float, default=0.015, help="Sample ratio for downsampling"
    )

    args = parser.parse_args()

    npz_folder = args.npz_folder
    pose_file = args.pose_file
    output_file = args.output_file

    conf_threshold_coef = (
        args.conf_threshold_coef
    )  # conf_threshold = np.mean(confs) * conf_threshold_coef
    sample_ratio = args.sample_ratio

    if not os.path.exists(npz_folder):
        print(f"Error: Folder {npz_folder} does not exist")
        return

    if not os.path.exists(pose_file):
        print(f"Error: File {pose_file} does not exist")
        return

    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    create_point_cloud(
        npz_folder=npz_folder,
        pose_file=pose_file,
        output_ply=output_file,
        conf_threshold_coef=conf_threshold_coef,
        sample_ratio=sample_ratio,
    )


if __name__ == "__main__":
    main()
