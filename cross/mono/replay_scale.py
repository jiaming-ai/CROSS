"""Causal scale ablations on exactly the same recorded DPVO geometry.

No ground truth is read. These are controlled estimator replays, not new
inference runs, and must not be assigned the source run's runtime numbers.
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from .config import ScaleConfig
from .scale import LogScaleFilter, ScaleObservation
from .scaled_motion import ScaledTranslation


def replay(rows, diagnostics, config):
    """Recover unit-gauge increments and apply a different causal scale filter."""
    rows = np.asarray(rows)
    if rows.ndim != 2 or rows.shape[1] != 8 or len(rows) != len(diagnostics):
        raise ValueError("Expected one diagnostic per timestamped frontend pose")
    result = rows.copy()
    result[:, 1:4] = 0
    previous_source = np.zeros(3)
    current = np.zeros(3)
    filter_ = LogScaleFilter(config)
    startup_contract = [d.get('scale_application') == 'anchored_startup_v1' for d in diagnostics]
    if any(startup_contract) and not all(startup_contract):
        raise ValueError('Scale application contracts cannot change within a run')
    anchored = all(startup_contract) and len(diagnostics) > 0
    scaled_translation = ScaledTranslation()
    scales = []
    for i, (row, diagnostic) in enumerate(zip(rows, diagnostics)):
        original_scale = diagnostic["scale"]
        if not np.isfinite(original_scale) or original_scale <= 0:
            raise ValueError("Recorded scale must be finite and positive")
        filter_.predict()
        observations = ([diagnostic['scale_observation']] if 'scale_observation' in diagnostic else [])
        observations += [event['observation'] for event in diagnostic.get('metric_result_events', [])
                         if event['scale_update_requested']]
        for observation in observations:
            values = dict(observation)
            # Online innovation rejection depends on the old filter state;
            # repeat it under the policy being tested. Shape rejection stays.
            if values["reason"] == "innovation_gate":
                values.update(accepted=True, reason="accepted")
            for name in ("variance", "log_mad"):
                if values[name] is None:
                    values[name] = float("inf")
            filter_.update(ScaleObservation(**values))
        if anchored:
            if 'unit_translation' not in diagnostic:
                raise ValueError('Anchored startup replay requires recorded unit positions')
            unit_position = diagnostic['unit_translation']
            if unit_position is not None:
                current = scaled_translation.update(unit_position, filter_.scale if filter_.initialized else None)
        else:
            # Legacy recordings can contain unit-gauge motion before the first
            # metric prior. Preserve their old convention for exact replay.
            unit_increment = (row[1:4] - previous_source) / original_scale
            current = current + filter_.scale * unit_increment
        result[i, 1:4] = current
        previous_source = row[1:4]
        scales.append(filter_.scale)
    return result, np.asarray(scales)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["filtered", "initial", "direct", "relative"], required=True)
    parser.add_argument("--recovery-observations", type=int)
    args = parser.parse_args()
    metadata = json.loads((args.source / "run.json").read_text())
    if metadata["config"]["frontend"] not in {"dpvo", "streaming_dpvo"}:
        raise ValueError("Replay currently requires the DPVO increment convention")
    if metadata["config"]["scale"]["mode"] in {"initial", "relative"}:
        raise ValueError("Source must contain periodic metric observations")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite results: {args.output}")
    scale_config = dict(metadata["config"]["scale"])
    scale_config.setdefault("recovery_observations", 0)  # legacy run semantics
    if args.recovery_observations is not None:
        scale_config["recovery_observations"] = args.recovery_observations
    config = replace(ScaleConfig(**scale_config), mode=args.mode)
    source_paths = [args.source / "frontend_trajectory.txt", args.source / "diagnostics.jsonl"]
    rows = np.loadtxt(source_paths[0], ndmin=2)
    diagnostics = [json.loads(line) for line in source_paths[1].read_text().splitlines()]
    trajectory, scales = replay(rows, diagnostics, config)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output / "trajectory.txt", trajectory, fmt="%.9f")
    np.savetxt(args.output / "scale.txt", np.c_[rows[:, 0], scales], fmt="%.9f")
    provenance = {
        "kind": "causal_scale_replay", "source_run": str(args.source.resolve()),
        "source_commit": metadata["source_commit"], "mode": args.mode,
        "recovery_observations": config.recovery_observations,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        "ground_truth_used": False, "runtime_measured": False,
        "geometry_and_metric_observations_shared_across_policies": True,
    }
    (args.output / "replay.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
