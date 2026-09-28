"""Collect completed benchmark runs without modifying raw results."""

import argparse
import json
from pathlib import Path


def summarize(root):
    rows = []
    for path in sorted(root.glob("*/run.json")):
        run = json.loads(path.read_text())
        summary_path, metrics_path = path.parent / "summary.json", path.parent / "metrics.json"
        row = {"run": path.parent.name, "sequence": Path(run["command"]["sequence"]).name,
               "frontend": run["config"].get("frontend", "da3"), "scale_mode": run["config"]["scale"]["mode"],
               "stride": run["command"]["stride"], "source_commit": run["source_commit"],
               "completed": summary_path.exists() and metrics_path.exists()}
        if summary_path.exists():
            row.update(json.loads(summary_path.read_text()))
        if metrics_path.exists():
            row.update(json.loads(metrics_path.read_text()))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = summarize(args.root)
    args.output.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    print("| Run | Frames | Metric ATE (m) | Sim(3) ATE (m) | Fitted scale | Coverage | FPS |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        if not row["completed"]:
            print(f"| {row['run']} | incomplete | — | — | — | — | — |")
            continue
        print(f"| {row['run']} | {row['frames']} | {row['ate_se3_rmse_m']:.4f} | "
              f"{row['ate_sim3_rmse_m']:.4f} | {row['sim3_alignment_scale']:.4f} | "
              f"{100*row['tracking_coverage']:.1f}% | {row['fps_including_io']:.1f} |")


if __name__ == "__main__":
    main()
