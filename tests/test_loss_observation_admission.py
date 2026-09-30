"""A tracking failure must not turn a global RGB observation into certain motion."""
from types import SimpleNamespace

import numpy as np
import pytest

from cross.mono.config import MonoConfig


def test_loss_observation_policy_requires_a_streaming_mapping_cadence():
    with pytest.raises(ValueError, match='streaming_pnp'):
        MonoConfig(retrieve_during_loss=True)
    with pytest.raises(ValueError, match='positive mapping interval'):
        MonoConfig(frontend='streaming_pnp', retrieve_during_loss=True, mapping_interval=0)


@pytest.mark.parametrize('enabled', [False, True])
def test_loss_observations_reach_cross_with_failed_motion_and_bounded_cadence(enabled):
    torch = pytest.importorskip('torch')
    from cross.mono.streaming import FrameSnapshot, StreamingMonocularSystem

    observed = []
    snapshots = [FrameSnapshot(i, i*.05, np.full((4, 4, 3), i, np.uint8),
                              {'frame': i}, np.eye(4), np.ones(6)*i, i in (0, 23))
                 for i in (0, 5, 10, 15, 20, 23, 25, 30, 33, 38)]
    ready = [((x, np.ones((4, 4))*(x.index+1)), {}) for x in snapshots]
    system = StreamingMonocularSystem.__new__(StreamingMonocularSystem)
    system.config = MonoConfig(frontend='streaming_pnp', retrieve_during_loss=enabled, mapping_interval=10)
    system.last_mapping_submission = -10
    system.pool = system.map_stream = system.previous_snapshot = None
    system.mapper = SimpleNamespace(step=observed.append,
        get_current_pose=lambda: SimpleNamespace(matrix=lambda: torch.eye(4)),
        db=SimpleNamespace(get_size=lambda: 1),
        hypothesis_manager=SimpleNamespace(nodes={}, hypotheses=[]),
        last_step_diagnostics={'loop_closure_applied': False})
    system.frontend = SimpleNamespace(take_depths=lambda: ready)
    submitted = []
    def submit(item):
        submitted.append(item)
        system._map_snapshot(item)
    system.map_worker = SimpleNamespace(submit=submit)
    system._submit_depths()
    expected = [0, 10, 20, 23, 33] if enabled else [0, 23]
    assert [x.index for x, _ in submitted] == expected
    for (snapshot, depth), observation in zip(submitted, observed):
        original = next(x for x in snapshots if x.index == snapshot.index)
        assert snapshot.valid == original.valid
        assert snapshot.features == {} and original.features == {'frame': original.index}
        assert observation['rgb'] is original.rgb and observation['depth'] is depth
        assert observation['timestamp'] == original.timestamp
        np.testing.assert_array_equal(snapshot.pose, original.pose)
        np.testing.assert_array_equal(snapshot.world_std_prefix, original.world_std_prefix)
        if not snapshot.valid:
            np.testing.assert_array_equal(observation['delta_pose'], np.eye(4))
            assert np.min(np.diag(observation['motion_covariance'])) >= 100.
    # A valid observation resets the admission cadence; a loss-time capture
    # five frames later cannot double the mapper's configured request rate.
    assert 25 not in expected and 30 not in expected
