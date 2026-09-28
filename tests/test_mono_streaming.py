from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")

from cross.mono.streaming import FrameSnapshot, StreamingMonocularSystem, StreamingPnPFrontend, snapshot_motion


def test_replaced_mapping_images_preserve_motion_and_correlated_error():
    first = SimpleNamespace(index=0, pose=np.eye(4), world_std_prefix=np.ones(6)*.01)
    last_pose = np.eye(4)
    last_pose[0, 3] = 2.
    last = SimpleNamespace(index=99, pose=last_pose, world_std_prefix=np.ones(6))
    delta, covariance = snapshot_motion(first, last)
    assert delta[0, 3] == 2.
    np.testing.assert_allclose(np.diag(covariance), .99**2)
    with pytest.raises(ValueError, match="increase"):
        snapshot_motion(last, first)


def test_worker_uses_snapshot_rgb_depth_timestamp_and_native_mapper_message():
    class FakePose:
        def matrix(self):
            import torch
            result = torch.eye(4)
            result[1, 3] = 5.
            return result

    class FakeMapper:
        def __init__(self):
            self.received = []
            self.db = SimpleNamespace(get_size=lambda: 4)
            self.hypothesis_manager = SimpleNamespace(nodes={1: None}, hypotheses=[None])
            self.last_step_diagnostics = {"loop_closure_applied": False, "verified_keyframes": 3}
        def step(self, observation):
            self.received.append(observation)
        def get_current_pose(self):
            return FakePose()

    system = StreamingMonocularSystem.__new__(StreamingMonocularSystem)
    system.mapper, system.map_stream, system.previous_snapshot, system.pool = FakeMapper(), None, None, None
    pose = np.eye(4)
    pose[0, 3] = 2.
    snapshot = SimpleNamespace(index=9, timestamp=.3, pose=pose,
                               rgb=np.ones((4, 4, 3)), world_std_prefix=np.ones(6))
    depth = np.ones((4, 4))*7.
    alignment, event = system._map_snapshot((snapshot, depth))
    np.testing.assert_allclose((alignment@pose)[:3, 3], [0., 5., 0.])
    received = system.mapper.received[0]
    assert received["rgb"] is snapshot.rgb and received["depth"] is depth
    assert received["timestamp"] == snapshot.timestamp
    assert event["mapping_event"]["verified_keyframes"] == 3
    assert event["source_frame"] == 9


def test_delayed_depth_recovers_its_source_without_rewriting_emitted_pose():
    frontend = StreamingPnPFrontend.__new__(StreamingPnPFrontend)
    failed = FrameSnapshot(10, .3, np.zeros((4, 4, 3)), {"frame": 10}, np.eye(4), np.ones(6), False)
    depth = np.ones((4, 4))*2
    frontend.depth_worker = SimpleNamespace(poll=lambda: [((failed, depth), {})])
    frontend.ready_depths = []
    frontend.pending_depths, frontend.index = [], 12
    frontend.anchor_index, frontend.last_valid = 0, False
    frontend.anchor_pose = frontend.metric_pose = np.eye(4)
    frontend.anchor_features = {"frame": 0}
    frontend.config = SimpleNamespace(scale=SimpleNamespace(interval=30), delayed_recovery=True, teacher_lag_frames=0)
    def reverse(source, reference, received_depth):
        assert source["frame"] == 10 and reference["frame"] == 0
        assert received_depth is depth
        transform = np.eye(4)
        transform[0, 3] = -2.
        return transform, 30, .1
    frontend.refiner = SimpleNamespace(estimate=reverse)
    renewed, received = frontend._receive()
    assert renewed and frontend.anchor_pose[0, 3] == 2.
    assert received[0][1]["delayed_reverse_recovery"]
    assert received[0][0][0].valid
    assert failed.pose[0, 3] == 0.  # original capture record remains immutable
    assert frontend.metric_pose[0, 3] == 0.  # emitted state is not rewritten


def test_teacher_assimilation_schedule_holds_ready_depth_without_blocking_pose():
    frontend = StreamingPnPFrontend.__new__(StreamingPnPFrontend)
    source_pose = np.eye(4)
    source_pose[0, 3] = 2.
    snapshot = FrameSnapshot(30, 1., np.zeros((4, 4, 3)), {}, source_pose, np.ones(6), True)
    incoming = [((snapshot, np.ones((4, 4))), {})]
    def poll():
        result = incoming.copy()
        incoming.clear()
        return result
    frontend.depth_worker = SimpleNamespace(poll=poll)
    frontend.pending_depths, frontend.ready_depths, frontend.index = [], [], 31
    frontend.anchor_index, frontend.last_valid = 0, True
    frontend.anchor_pose = frontend.metric_pose = np.eye(4)
    frontend.config = SimpleNamespace(scale=SimpleNamespace(interval=30), delayed_recovery=False, teacher_lag_frames=3)
    assert frontend._receive() == (False, [])
    assert len(frontend.pending_depths) == 1 and frontend.anchor_index == 0
    frontend.index = 32
    assert frontend._receive() == (False, [])
    frontend.index = 33
    renewed, received = frontend._receive()
    assert renewed and frontend.anchor_index == 30
    assert received[0][1]["held_ready_frames"] == 2
    assert received[0][1]["late_delivery_frames"] == 0
    assert frontend.metric_pose[0, 3] == 0.  # no retrospective pose correction
    assert not frontend.pending_depths
    # A final ready result must still reach mapping if the input stops before
    # the configured assimilation age; no already emitted pose changes.
    from dataclasses import replace
    incoming.append(((replace(snapshot, index=60), np.ones((4, 4))), {}))
    frontend.index = 61
    assert frontend._receive() == (False, [])
    _, drained = frontend._receive(drain=True)
    assert drained[0][1]["final_drain"] and not frontend.pending_depths
    assert frontend.metric_pose[0, 3] == 0.


def test_teacher_failure_still_closes_the_mapping_worker():
    calls = []
    def failed_teacher():
        raise RuntimeError("teacher failed")
    system = StreamingMonocularSystem.__new__(StreamingMonocularSystem)
    system.finished = False
    system.frontend = SimpleNamespace(finish=failed_teacher)
    system.map_worker = SimpleNamespace(close=lambda: calls.append("map_closed"))
    system._receive_maps = lambda: calls.append("received")
    with pytest.raises(RuntimeError, match="teacher failed"):
        system.finish()
    assert calls == ["map_closed", "received"]


def test_proactive_anchor_keeps_the_verified_source_pose_and_rejects_stale_depth():
    frontend = StreamingPnPFrontend.__new__(StreamingPnPFrontend)
    source_pose = np.eye(4)
    source_pose[0, 3] = 2.
    snapshot = FrameSnapshot(5, .25, np.zeros((4, 4, 3)), {"source": 5},
                             source_pose, np.zeros(6), True, refresh_anchor=True)
    pending = [((snapshot, np.ones((4, 4))*3), {})]
    def poll():
        result = pending.copy()
        pending.clear()
        return result
    frontend.depth_worker = SimpleNamespace(poll=poll)
    frontend.pending_depths, frontend.ready_depths, frontend.index = [], [], 7
    frontend.anchor_index, frontend.last_valid = 0, True
    frontend.anchor_pose = frontend.metric_pose = np.eye(4)
    frontend.config = SimpleNamespace(scale=SimpleNamespace(interval=20), delayed_recovery=False, teacher_lag_frames=0)
    renewed, received = frontend._receive()
    assert renewed and frontend.anchor_index == 5  # before the fixed interval
    assert frontend.anchor_features["source"] == 5
    np.testing.assert_array_equal(frontend.anchor_pose, source_pose)
    assert received[0][0][0] is snapshot
    assert frontend.metric_pose[0, 3] == 0.  # no rewrite of emitted poses
    from dataclasses import replace
    pending.append(((replace(snapshot, index=4), np.ones((4, 4))*9), {}))
    renewed, received = frontend._receive()
    assert not renewed and not received
    assert frontend.anchor_index == 5 and frontend.anchor_depth[0, 0] == 3


@pytest.mark.parametrize("enabled,count,valid,index,requested", [
    (True, 60, True, 5, True),
    (True, 60, True, 4, False),
    (True, 100, True, 5, False),
    (False, 60, True, 5, False),
    (True, 0, False, 5, True),  # inherited failure request, not proactive
])
def test_proactive_requests_are_geometric_causal_and_budgeted(enabled, count, valid, index, requested):
    from cross.mono.config import MonoConfig, ScaleConfig
    frontend = StreamingPnPFrontend.__new__(StreamingPnPFrontend)
    frontend.config = MonoConfig(frontend="streaming_pnp", adaptive_anchor=enabled,
                                 scale=ScaleConfig(interval=20), mapping_interval=10)
    frontend.anchor_features, frontend.anchor_depth = {"source": 0}, np.ones((4, 4))
    frontend.anchor_pose = frontend.metric_pose = np.eye(4)
    frontend.anchor_index, frontend.index, frontend.last_submitted = 0, index, 0
    frontend.last_timestamp, frontend.last_valid = 0., True
    frontend.world_std_prefix, frontend.provide_mapping_depth = np.zeros(6), True
    frontend.scale_filter = SimpleNamespace(uncertainty_variance=.01)
    features = {"source": index}
    relative = np.eye(4)
    relative[0, 3] = 1.
    frontend.refiner = SimpleNamespace(extract=lambda rgb: features,
        estimate=lambda *args: (relative, count, .5) if valid else None, last_correspondences=count)
    submitted = []
    frontend.depth_worker = SimpleNamespace(submit=submitted.append, statistics=lambda: {})
    frontend._receive = lambda: (False, [])
    estimate = frontend.step(np.zeros((4, 4, 3)), .05*index)
    assert bool(submitted) is requested
    assert estimate.diagnostics["proactive_metric_request"] is (enabled and valid and count < 80 and requested)
    if submitted:
        assert submitted[0].index == index
        assert submitted[0].valid is valid
        assert submitted[0].refresh_anchor is (enabled and valid and count < 80)
        np.testing.assert_array_equal(submitted[0].pose, estimate.pose)
