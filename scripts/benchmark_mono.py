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
    parser.add_argument("--profile", type=Path, help="JSON with an arguments list for cross.mono.run")
    parser.add_argument("--dpvo-checkpoint", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Print the exact commands without creating outputs or loading models")
    parser.add_argument("--frontend-only", action="store_true")
    parser.add_argument("--metric-interval", type=int, help="Override profile cadence; baseline default 30")
    parser.add_argument("--mapping-interval", type=int, help="Override profile cadence; baseline default 15")
    parser.add_argument("--mask-interval", type=int, help="Override profile cadence; baseline default 3")
    args = parser.parse_args()
    if args.profile:
        profile = json.loads(args.profile.read_text())
        options = profile.get("arguments")
        reserved = {"--output", "--seed", "--dpvo-checkpoint", "--frontend-only", "--save-map"}
        if (not isinstance(options, list) or not all(isinstance(x, str) for x in options)
                or any(x.split("=", 1)[0] in reserved for x in options)):
            parser.error("Profile arguments must be a string list; output, seed, checkpoint and frontend-only are runner options")
        options = options.copy()
        if profile.get("requires_dpvo_checkpoint") and args.dpvo_checkpoint is None:
            parser.error("This profile requires --dpvo-checkpoint")
    else:
        options = ["--frontend", "metric_pnp", "--mask-people",
                   "--metric-interval", "30", "--mapping-interval", "15", "--mask-interval", "3"]
    for option in ("metric_interval", "mapping_interval", "mask_interval"):
        value = getattr(args, option)
        if value is not None:
            if value < 1:
                parser.error("Cadence overrides must be positive")
            options += ["--"+option.replace("_", "-"), str(value)]
    if args.dpvo_checkpoint:
        if not args.dry_run and not args.dpvo_checkpoint.is_file():
            parser.error("DPVO checkpoint does not exist")
        options += ["--dpvo-checkpoint", str(args.dpvo_checkpoint)]
    options.append("--frontend-only" if args.frontend_only else "--save-map")
    tasks = []
    for seed in args.seeds:
        for name in args.sequences:
            sequence = args.data_root / name
            output = args.output / f"{name}_s{seed}"
            command = [sys.executable, "-m", "cross.mono.run", str(sequence), "--output", str(output),
                       *options, "--seed", str(seed)]
            tasks.append(dict(sequence=name, seed=seed, output=str(output), command=command))
    plan = dict(profile=str(args.profile) if args.profile else None, tasks=tasks)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "benchmark_plan.json").open("x") as handle:
        handle.write(json.dumps(plan, indent=2) + "\n")
    failures = []
    for task in tasks:
        seed, name = task["seed"], task["sequence"]
        sequence = args.data_root / name
        output = Path(task["output"])
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        log_path = args.output / f"{name}_s{seed}.log"
        with log_path.open("x") as log:
            print("Running", name, "seed", seed, flush=True)
            status = subprocess.run(task["command"], stdout=log, stderr=subprocess.STDOUT)
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
                status = subprocess.run(calibrated, stdout=log, stderr=subprocess.STDOUT)
                if status.returncode:
                    failures.append({"sequence": name, "seed": seed, "stage": "camera_frame_evaluation", "exit_code": status.returncode})
    (args.output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    if failures:
        raise SystemExit(f"{len(failures)} failed runs; inspect failures.json and logs")


if __name__ == "__main__":
    main()
