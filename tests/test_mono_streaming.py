from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")

from cross.mono.streaming import StreamingMonocularSystem, snapshot_motion


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
    system.mapper, system.map_stream, system.previous_snapshot = FakeMapper(), None, None
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
