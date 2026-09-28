import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cross.mono.evaluate import evaluate
from cross.mono.evaluate_restart import evaluate_restart


def write_poses(path, stamps, poses):
    rows = np.column_stack((stamps, poses[:, :3, 3], Rotation.from_matrix(poses[:, :3, :3]).as_quat()))
    np.savetxt(path, rows, fmt="%.12f")
    return path


def example(tmp_path):
    t = np.arange(12) * 0.05
    ref = np.broadcast_to(np.eye(4), (len(t), 4, 4)).copy()
    ref[:, :3, 3] = np.column_stack((np.cos(t*3), np.sin(t*3), t*t))
    ref[:, :3, :3] = Rotation.from_euler("z", t).as_matrix()
    alignment = np.eye(4)
    alignment[:3, :3] = Rotation.from_euler("xyz", [.2, -.1, .6]).as_matrix()
    alignment[:3, 3] = [2, 3, -1]
    files = [write_poses(tmp_path/"reference.txt", t, ref),
             write_poses(tmp_path/"reference_gt.txt", t, alignment @ ref),
             write_poses(tmp_path/"query.txt", t+100, ref),
             write_poses(tmp_path/"query_gt.txt", t+100, alignment @ ref)]
    return t, ref, alignment, files


def test_reference_only_alignment_exposes_failed_relocalization(tmp_path):
    t, ref, alignment, files = example(tmp_path)
    wrong = ref.copy()
    wrong[:, 0, 3] += 5
    write_poses(files[2], t+100, wrong)
    # Independent query alignment completely hides the wrong map origin.
    assert evaluate(files[2], files[3])["ate_se3_rmse_m"] < 1e-9
    result = evaluate_restart(*files)
    np.testing.assert_allclose(result["reference_alignment"], alignment, atol=1e-9)
    assert result["query_ate_rmse_m"] == pytest.approx(5)
    assert result["query_correct_pose_fraction"] == 0
    assert result["first_correct_pose_delay_seconds"] is None


def test_query_truth_cannot_change_alignment_and_commit_uses_received_output(tmp_path):
    t, ref, alignment, files = example(tmp_path)
    rows = [{} for _ in t]
    rows[7] = dict(mapping_updates=[dict(source_frame=2, mapping_event=dict(loop_closure_applied=True))])
    diagnostics = tmp_path/"diagnostics.jsonl"
    diagnostics.write_text("\n".join(json.dumps(row) for row in rows))
    correct = evaluate_restart(*files, diagnostics=diagnostics)
    assert correct["query_ate_rmse_m"] < 1e-9
    assert correct["query_rotation_rmse_degrees"] < 1e-7
    assert correct["applied_commitments"][0]["delay_seconds"] == pytest.approx(t[7])
    assert correct["applied_commitments"][0]["output_pose_correct"]
    moved_truth = alignment @ ref
    moved_truth[:, 1, 3] += 3
    write_poses(files[3], t+100, moved_truth)
    changed = evaluate_restart(*files, diagnostics=diagnostics)
    assert changed["reference_alignment"] == correct["reference_alignment"]
    assert not changed["applied_commitments"][0]["output_pose_correct"]
    assert changed["query_ate_rmse_m"] == pytest.approx(3)


def test_held_outputs_are_counted_and_degenerate_reference_rejected(tmp_path):
    t, ref, _, files = example(tmp_path)
    ref[1:5] = ref[0]
    write_poses(files[2], t+100, ref)
    result = evaluate_restart(*files)
    assert result["query_estimated_frames"] == len(t)
    assert result["query_associated_frames"] == len(t)
    assert result["query_ate_rmse_m"] > 0.1
    ref[:, :3, 3] = np.column_stack((t, t*0, t*0))
    write_poses(files[0], t, ref)
    with pytest.raises(ValueError, match="degenerate"):
        evaluate_restart(*files)


def test_openloris_gt_preserves_shared_world_and_camera_lever_arm(tmp_path):
    import cv2
    from cross.mono.openloris_groundtruth import camera_groundtruth
    base = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
    base[:, :3, 3] = [[10, 2, 0], [10, 3, 0], [11, 3, 0]]
    base[:, :3, :3] = Rotation.from_euler("z", [90, 0, -90], degrees=True).as_matrix()
    extrinsic = np.eye(4)
    extrinsic[0, 3] = 1.
    extrinsic[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()
    write_poses(tmp_path/"groundtruth.txt", np.arange(3), base)
    handle = cv2.FileStorage(str(tmp_path/"trans_matrix.yaml"), cv2.FILE_STORAGE_WRITE)
    try:
        handle.startWriteStruct("trans_matrix", cv2.FileNode_SEQ)
        handle.startWriteStruct("", cv2.FileNode_MAP)
        handle.write("parent_frame", "base_link")
        handle.write("child_frame", "d400_color_optical_frame")
        handle.write("matrix", extrinsic)
        handle.endWriteStruct()
        handle.endWriteStruct()
    finally:
        handle.release()
    stamps, camera, _ = camera_groundtruth(tmp_path)
    np.testing.assert_array_equal(stamps, np.arange(3))
    np.testing.assert_allclose(camera[:, :3, 3], [[10, 3, 0], [11, 3, 0], [11, 2, 0]], atol=1e-10)
    np.testing.assert_allclose(camera[:, :3, :3], base[:, :3, :3] @ extrinsic[:3, :3], atol=1e-10)
