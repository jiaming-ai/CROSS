#!/usr/bin/env bash
# KITTI raw synced+rectified drives of the odometry sequences 00-10 (03 has no raw release) and the odometry
# ground-truth poses.  Usage: download_kitti.sh <dest>
set -euo pipefail
DEST=${1:?dest dir}
S3=https://s3.eu-central-1.amazonaws.com/avg-kitti
mkdir -p "$DEST"; cd "$DEST"
[ -d poses ] || { wget -q -c $S3/data_odometry_poses.zip && unzip -q -o data_odometry_poses.zip && mv dataset/poses . && rm -rf dataset data_odometry_poses.zip; }
for date in 2011_09_30 2011_10_03; do
  [ -f $date/calib_cam_to_cam.txt ] || { wget -q -c $S3/raw_data/${date}_calib.zip && unzip -q -o ${date}_calib.zip && rm -f ${date}_calib.zip; }
done
# odometry sequence -> raw drive (KITTI odometry devkit readme)
for d in 2011_10_03_drive_0027 2011_10_03_drive_0042 2011_10_03_drive_0034 2011_09_30_drive_0016 2011_09_30_drive_0018 \
         2011_09_30_drive_0020 2011_09_30_drive_0027 2011_09_30_drive_0028 2011_09_30_drive_0033 2011_09_30_drive_0034; do
  [ -d ${d%%_drive*}/${d}_sync ] && continue
  echo "[$(date +%T)] $d"
  wget -q -c $S3/raw_data/$d/${d}_sync.zip && unzip -q -o ${d}_sync.zip && rm -f ${d}_sync.zip && echo "[$(date +%T)] done $d"
done
echo KITTI_DL_DONE
