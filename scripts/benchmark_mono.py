"""Run the fixed monocular profile on complete RGB sequences, then evaluate.

Use a remote GPU server. Choose its physical GPU using CUDA_VISIBLE_DEVICES.
This script is sequential; separate processes can use separate idle GPUs.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+", required=True, help="Directory basenames, e.g. rgbd_bonn_balloon2")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--frontend-only", action="store_true")
    parser.add_argument("--metric-interval", type=int, default=30)
    parser.add_argument("--mapping-interval", type=int, default=15)
    parser.add_argument("--mask-interval", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    failures = []
    for seed in args.seeds:
        for name in args.sequences:
            sequence = args.data_root / name
            output = args.output / f"{name}_s{seed}"
            if output.exists():
                raise FileExistsError(f"Refusing to overwrite {output}")
            command = [sys.executable, "-m", "cross.mono.run", str(sequence), "--output", str(output),
                       "--frontend", "metric_pnp", "--mask-people", "--seed", str(seed),
                       "--mask-interval", str(args.mask_interval), "--metric-interval", str(args.metric_interval),
                       "--mapping-interval", str(args.mapping_interval)]
            command.append("--frontend-only" if args.frontend_only else "--save-map")
            log_path = args.output / f"{name}_s{seed}.log"
            with log_path.open("x") as log:
                print("Running", name, "seed", seed, flush=True)
                status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                if status.returncode:
                    failures.append({"sequence": name, "seed": seed, "stage": "inference", "exit_code": status.returncode})
                    continue
                evaluate = [sys.executable, "-m", "cross.mono.evaluate", str(output / "trajectory.txt"),
                            str(sequence / "groundtruth.txt"), "--output", str(output / "metrics.json")]
                status = subprocess.run(evaluate, stdout=log, stderr=subprocess.STDOUT)
                if status.returncode:
                    failures.append({"sequence": name, "seed": seed, "stage": "evaluation", "exit_code": status.returncode})
                if "bonn" in name and status.returncode == 0:
                    calibrated = evaluate[:-1] + [str(output / "metrics_bonn_camera.json"), "--groundtruth-frame", "bonn-camera"]
                    subprocess.run(calibrated, stdout=log, stderr=subprocess.STDOUT, check=True)
    (args.output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    if failures:
        raise SystemExit(f"{len(failures)} failed runs; inspect failures.json and logs")


if __name__ == "__main__":
    main()
