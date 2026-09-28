from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")

from cross.mono.config import MonoConfig, ScaleConfig
from cross.mono.metric_sources import TranslationResponse, identify_prediction
from cross.mono.streaming import FrameSnapshot, StreamingMonocularSystem, StreamingPnPFrontend


def make_frontend(trace):
    frontend = StreamingPnPFrontend.__new__(StreamingPnPFrontend)
    frontend.config = MonoConfig(frontend="streaming_pnp", trace_metric_sources=trace,
                                 scale=ScaleConfig(interval=2), mapping_interval=2)
    frontend.K = np.eye(3)
    frontend.anchor_features = frontend.anchor_depth = None
    frontend.anchor_pose = frontend.metric_pose = np.eye(4)
    frontend.anchor_index, frontend.index, frontend.last_submitted = -1, 0, -1
    frontend.last_timestamp, frontend.last_valid = None, True
    frontend.world_std_prefix, frontend.provide_mapping_depth = np.zeros(6), True
    frontend.scale_filter = SimpleNamespace(uncertainty_variance=.01)
    frontend.pending_depths, frontend.ready_depths = [], []
    frontend.trace_metric_sources = trace
    frontend.metric_revision, frontend.metric_session_id = "weights", "acquisition"
    frontend.anchor_metric_source = None
    frontend.anchor_scale_response = frontend.pose_scale_prefix = TranslationResponse()
    frontend.pose_scale_tail = None
    def estimate(anchor, current, depth):
        if current["frame"] == 3:
            return None
        transform = np.eye(4)
        angle = .2*(current["frame"]-anchor["frame"])
        transform[:2,:2] = [[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
        transform[:3,3] = [current["frame"]-anchor["frame"], .2, .1]
        return transform, 100, .5
    frontend.refiner = SimpleNamespace(extract=lambda _: {"frame": frontend.index},
                                       estimate=estimate, last_correspondences=100)
    pending = []
    def poll():
        result = [((snapshot, np.ones((4,4))), {}) for snapshot in pending]
        pending.clear()
        return result
    frontend.depth_worker = SimpleNamespace(submit=pending.append, poll=poll, statistics=lambda: {})
    frontend._predict_snapshot = lambda snapshot: (snapshot, np.ones((4,4)))
    return frontend


def test_anchor_refresh_and_invalid_hold_preserve_exact_pose_and_source_lineage():
    traced, control = make_frontend(True), make_frontend(False)
    snapshots = {}
    sources = {}
    for i in range(6):
        image = np.full((4,4,3), i, dtype=np.uint8)
        actual, expected = traced.step(image, i*.05), control.step(image, i*.05)
        for field in ("pose", "delta_pose", "motion_covariance"):
            np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
        for (snapshot, _), _ in traced.take_depths():
            snapshots[snapshot.index] = snapshot
            sources[snapshot.index] = snapshot.metric_source.source_id
        if i == 3:
            assert traced.anchor_index == 2 and not actual.diagnostics["valid"]
            # The anchor changed, but the emitted pose was held. Its response
            # is still the old source's response, not the new anchor's tail.
            assert traced._capture_scale_response() == snapshots[2].scale_response
        if i == 4:
            expected_response = snapshots[2].scale_response.with_displacement(
                sources[2], actual.pose[:3,3]-snapshots[2].pose[:3,3])
            actual_response = traced._capture_scale_response().record()
            assert actual_response.keys() == expected_response.record().keys()
            for source, vector in expected_response.terms:
                np.testing.assert_allclose(actual_response[source], vector, rtol=0, atol=1e-14)
    assert set(snapshots[4].scale_response.record()) == {sources[0], sources[2]}
    assert sources[4] not in snapshots[4].scale_response.record()


def test_reverse_recovery_uses_new_depth_identity_without_rewriting_held_trace():
    frontend = make_frontend(True)
    frontend.step(np.zeros((4,4,3),dtype=np.uint8), 0.)
    held = frontend._capture_scale_response()
    source = identify_prediction(np.ones((4,4,3),dtype=np.uint8), frontend.K,
        model_id="teacher", revision="weights", resolution=504, session_id="acquisition")
    snapshot = FrameSnapshot(10, .5, np.ones((4,4,3),dtype=np.uint8), {}, np.eye(4),
                             np.zeros(6), False, metric_source=source, scale_response=held)
    frontend.depth_worker = SimpleNamespace(poll=lambda: [((snapshot, np.ones((4,4))), {})])
    frontend.index, frontend.last_valid = 12, False
    frontend.config.delayed_recovery = True
    reverse = np.eye(4)
    reverse[0,3] = -2.
    frontend.refiner.estimate = lambda *_: (reverse, 100, .1)
    renewed, received = frontend._receive()
    assert renewed
    recovered = received[0][0][0]
    assert recovered.scale_response.record() == {source.source_id: [-2.,0.,0.]}
    assert frontend.anchor_metric_source == source
    assert snapshot.scale_response == held and not snapshot.valid
    assert frontend._capture_scale_response() == held and frontend.metric_pose[0,3] == 0.


def test_mapping_drop_keeps_signed_interval_and_does_not_mislabel_node_posterior():
    import torch
    captured = []
    mapped_pose = SimpleNamespace(matrix=lambda: torch.eye(4))
    system = StreamingMonocularSystem.__new__(StreamingMonocularSystem)
    system.pool = system.map_stream = None
    system.mapper = SimpleNamespace(step=captured.append, get_current_pose=lambda: mapped_pose,
        db=SimpleNamespace(get_size=lambda: 1), hypothesis_manager=SimpleNamespace(nodes={},hypotheses={}))
    f = make_frontend(True)
    f.step(np.zeros((4,4,3),dtype=np.uint8),0.)
    first = f.take_depths()[0][0][0]
    one = TranslationResponse().with_displacement(first.metric_source.source_id, [1.,0.,0.])
    final = TranslationResponse().with_displacement(first.metric_source.source_id, [3.,0.,0.])
    previous_pose, final_pose = np.eye(4), np.eye(4)
    previous_pose[0,3], final_pose[0,3] = 1.,3.
    system.previous_snapshot = replace(first, index=20, pose=previous_pose, scale_response=one)
    current = replace(first, index=90, timestamp=4.5, pose=final_pose, scale_response=final)
    system._map_snapshot((current, np.ones((4,4))))
    observation = captured[0]
    assert observation["metric_source"] == first.metric_source.record()
    assert observation["metric_input"]["previous_frame"] == 20
    assert observation["metric_input"]["motion_right_tangent_response"] == {
        first.metric_source.source_id: [-2.,0.,0.,0.,0.,0.]}
    assert observation["delta_pose"][0,3] == 2.
