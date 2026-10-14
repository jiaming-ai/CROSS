"""Self-calibrate a two-iPhone stereo rig from its recordings (no calibration target needed).

Each pair is one recording of both phones (A = right phone, B = left phone), as folders written by the capture app
(frames.csv, rgb/, depth/, confidence/) or folders inside a capture zip ("capture.zip::path/to/recording_..._B").
Recordings of the same mounting are combined into one calibration; the per-recording spread shows whether the rig
moved between them.  Output: a YAML in the layout of stereo_extrinsics.yaml (R, T: x_B = R x_A + T, OpenCV camera
axes; camera matrices; baseline) plus the clock offset of every recording and the quality numbers.

  python scripts/rig/calibrate_iphone_rig.py --pair rec_1_A rec_1_B --pair rec_2_A rec_2_B --out rig.yaml \
      [--compare stereo_extrinsics.yaml] [--frames 120] [--no-images] [--no-lidar]

Steps (cross/rig/selfcalib.py): clock offset from the phones' angular speed; extrinsic by hand-eye calibration of the
two ARKit tracks (metric); refinement of rotation, translation direction and clock offset on SIFT matches between the
phones' images with the motion between the two exposures compensated (ARKit); baseline length checked against LiDAR;
clock offset refined on matches while the rig turns (angular speed gives the initial value and fallback).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cross.rig.recording import Recording  # noqa: E402
from cross.rig.selfcalib import calibrate_recording, combine  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs=2, action="append", required=True, metavar=("A", "B"),
                    help="recording folders of phone A (right) and phone B (left); repeat for more recordings")
    ap.add_argument("--out", type=Path, required=True, help="calibration YAML to write")
    ap.add_argument("--compare", type=Path, help="an existing stereo_extrinsics.yaml to compare with")
    ap.add_argument("--frames", type=int, default=120, help="image pairs used for the refinement per recording")
    ap.add_argument("--no-images", action="store_true", help="motion only (no image refinement, no LiDAR check)")
    ap.add_argument("--no-lidar", action="store_true", help="keep the hand-eye baseline length")
    a = ap.parse_args()
    results, recs = {}, {}
    for pa, pb in a.pair:
        name = Path(str(pb).split("::")[-1].rstrip("/")).name
        print(f"{name}:", flush=True)
        recA, recB = Recording.load(pa), Recording.load(pb)
        recs[name] = (recA, recB)
        results[name] = calibrate_recording(recA, recB, images=not a.no_images, n_frames=a.frames,
                                            lidar=not a.no_lidar, log=lambda s: print(s, flush=True))
    cal = combine(results, recs, use_lidar_scale=not a.no_lidar)
    out = cal.to_dict()
    if a.compare:
        y = yaml.safe_load(a.compare.read_text())
        Ry, Ty = np.asarray(y["R"]), np.asarray(y["T"]).reshape(3)
        out["compared_with"] = {"file": str(a.compare), "baseline_m": float(np.linalg.norm(Ty)),
                                "rotation_difference_deg": float(np.degrees(np.linalg.norm(
                                    Rotation.from_matrix(cal.R @ Ry.T).as_rotvec()))),
                                "translation_direction_difference_deg": float(np.degrees(np.arccos(np.clip(
                                    cal.T @ Ty / (np.linalg.norm(cal.T) * np.linalg.norm(Ty)), -1, 1))))}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(yaml.safe_dump(json.loads(json.dumps(out, default=float)), sort_keys=False))
    spread = [v["rotation_from_combined_deg"] for v in cal.per_recording.values()]
    print(f"rig: baseline {cal.baseline:.4f} m; rotation spread over recordings max {max(spread):.3f} deg"
          f"{'; vs ' + str(a.compare) + ': ' + json.dumps(out['compared_with']) if a.compare else ''}")
    print(f"written {a.out}")


if __name__ == "__main__":
    main()
