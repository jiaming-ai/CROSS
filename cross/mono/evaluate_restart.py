"""Evaluate a saved-map query using a rigid alignment fitted ONLY on its reference.

Ground truth is an offline evaluator input. No query alignment or scale fitting
is performed. Pose correctness following a commitment is a diagnostic, not a
ground-truth place-identity label or proof that a topological merge was correct.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .evaluate import associate_trajectory, fit_alignment, read_trajectory


def evaluate_restart(reference, reference_truth, query, query_truth, *, diagnostics=None,
                     max_gap=0.1, translation_threshold=1.0, rotation_threshold=30.0):
    if not all(np.isfinite(x) and x > 0 for x in (translation_threshold, rotation_threshold)):
        raise ValueError("Correctness thresholds must be finite and positive")
    _, ref, ref_gt, _ = associate_trajectory(reference, reference_truth, max_gap)
    # A collinear trajectory cannot determine the reference rotation uniquely.
    for points in (ref[:, :3, 3], ref_gt[:, :3, 3]):
        singular = np.linalg.svd(points - points.mean(0), compute_uv=False)
        if singular[1] <= max(1e-8, singular[0] * 1e-6):
            raise ValueError("Reference motion is degenerate for rigid world alignment")
    _, rotation, translation = fit_alignment(ref[:, :3, 3], ref_gt[:, :3, 3], False)
    alignment = np.eye(4)
    alignment[:3, :3], alignment[:3, 3] = rotation, translation
    stamps, estimated, truth, valid = associate_trajectory(query, query_truth, max_gap)
    all_stamps, _ = read_trajectory(query)
    aligned = alignment @ estimated
    position_errors = np.linalg.norm(aligned[:, :3, 3] - truth[:, :3, 3], axis=1)
    angle_errors = np.degrees(Rotation.from_matrix(
        truth[:, :3, :3].transpose(0, 2, 1) @ aligned[:, :3, :3]).magnitude())
    correct = (position_errors <= translation_threshold) & (angle_errors <= rotation_threshold)
    correct_indices = np.flatnonzero(correct)
    result = dict(
        alignment_policy="Rigid SE(3), reference trajectory only; applied unchanged to query",
        reference_alignment=alignment.tolist(), reference_associated_frames=len(ref),
        reference_ate_rmse_m=float(np.sqrt(np.mean(np.sum(
            ((alignment @ ref)[:, :3, 3] - ref_gt[:, :3, 3])**2, axis=1)))),
        query_estimated_frames=len(valid), query_associated_frames=len(stamps),
        query_association_fraction=float(valid.mean()),
        query_ate_rmse_m=float(np.sqrt(np.mean(position_errors**2))),
        query_rotation_rmse_degrees=float(np.sqrt(np.mean(angle_errors**2))),
        query_correct_pose_fraction=float(correct.mean()),
        translation_threshold_m=translation_threshold, rotation_threshold_degrees=rotation_threshold,
        first_correct_pose_delay_seconds=float(stamps[correct_indices[0]]-all_stamps[0]) if len(correct_indices) else None,
        first_correct_pose_is_not_sustained_relocalization=True,
        applied_commitments=None, commitment_diagnostics_available=diagnostics is not None,
    )
    if diagnostics is not None:
        rows = [json.loads(line) for line in Path(diagnostics).read_text().splitlines()]
        if len(rows) != len(all_stamps):
            raise ValueError("Diagnostics must contain one row for every emitted query pose")
        # Use the output at which mapping was received, not the older source
        # image or an unconsumed result drained after the final output.
        associated = {int(i): j for j, i in enumerate(np.flatnonzero(valid))}
        events = []
        for i, row in enumerate(rows):
            updates = row.get("mapping_updates", [])
            if row.get("mapping_update") and "mapping_event" in row:
                updates = [dict(mapping_event=row["mapping_event"])]
            for update in updates:
                if not update.get("mapping_event", {}).get("loop_closure_applied", False):
                    continue
                j = associated.get(i)
                events.append(dict(output_index=i, timestamp=float(all_stamps[i]),
                                   delay_seconds=float(all_stamps[i]-all_stamps[0]),
                                   source_frame=update.get("source_frame"),
                                   output_position_error_m=float(position_errors[j]) if j is not None else None,
                                   output_rotation_error_degrees=float(angle_errors[j]) if j is not None else None,
                                   output_pose_correct=bool(correct[j]) if j is not None else None))
        result["applied_commitments"] = events
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "reference_truth", "query", "query_truth"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--max-gap", type=float, default=0.1)
    parser.add_argument("--translation-threshold", type=float, default=1.0)
    parser.add_argument("--rotation-threshold", type=float, default=30.0)
    parser.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args())
    output = args.pop("output")
    if output.exists():
        raise FileExistsError(output)
    text = json.dumps(evaluate_restart(**args), indent=2, allow_nan=False)
    output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
