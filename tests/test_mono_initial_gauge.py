"""Delayed map initialization must preserve the already emitted pose frame."""
from collections import deque
from threading import Lock
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

torch = pytest.importorskip('torch')
pp = pytest.importorskip('pypose')

from cross.core.config import HypothesisConfig, SystemConfig
from cross.core.conditional import SourceFactor
from cross.core.hypothesis import HypothesisManager
from cross.core.odom_accum import OdomAccumulator
from cross.core.system import System
from cross.core.types import Keyframe
from cross.mono.geometry import inverse
from cross.mono.metric_sources import TranslationResponse
from cross.mono.streaming import FrameSnapshot, StreamingMonocularSystem
from cross.mono.system import MonocularSystem


class Database:
    def __init__(self):
        self.nodes = []

    def get_all_atlases(self):
        return []

    def create_atlas(self):
        return None

    def get_size(self):
        return len(self.nodes)

    def insert(self, index, rgb, depth, **kw):
        node = Keyframe(kw['mu'], kw['sigma'], kw['weights'], pose_charts=kw['pose_charts'])
        self.nodes.append(node)
        return node


def mapper(conditional=False):
    value = System.__new__(System)
    value.config = SystemConfig()
    value.config.mapping.hypothesis.conditional_sources = conditional
    value.device = value.storage_device = value.state_device = 'cpu'
    value.topo_map = None
    value.kf_gmm_n_components = 2
    value._processed_frame_num = 0
    value.db = Database()
    value.hypothesis_manager = HypothesisManager(value, 2, HypothesisConfig(chart_aware=True, session_recovery=True))
    value.odom_accumulator = OdomAccumulator(device='cpu')
    value.use_odometry, value.async_update = True, False
    value.use_depth_pred = value.visualize = False
    value._cur_obs_queue, value._cur_obs_lock = deque(), Lock()
    value.rgb_transform = torch.as_tensor
    value.depth_transform = lambda x: x
    value.shutdown = lambda: None
    return value


def pose():
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec([.3, -.4, .1]).as_matrix()
    result[:3, 3] = [.2, -.7, .1]
    return result


@pytest.mark.parametrize('conditional', [False, True])
def test_delayed_source_initializes_map_without_a_receipt_time_pose_jump(conditional):
    system = StreamingMonocularSystem.__new__(StreamingMonocularSystem)
    system.mapper, system.pool, system.map_stream, system.previous_snapshot = mapper(conditional), None, None, None
    system.config = SimpleNamespace(conditional_sources=conditional)
    source = pose()
    snapshot = FrameSnapshot(16, .8, np.zeros((8, 8, 3)), {}, source.copy(), np.ones(6), True,
                             scale_response=TranslationResponse())
    alignment, event = system._map_snapshot((snapshot, np.ones((8, 8))))
    # The first snapshot predates the receipt. Initialization cannot change
    # either a later receiving pose or the stored source camera coordinates.
    receiving = source @ pose()
    np.testing.assert_allclose(alignment @ receiving, receiving, atol=3e-7)
    initial = system.mapper.db.nodes[0]
    np.testing.assert_allclose(initial.pose_mu[0].matrix(), source, atol=3e-7)
    assert not event['mapping_event']['loop_closure_applied']
    assert 'verified_keyframes' not in event['mapping_event']
    np.testing.assert_array_equal(initial.pose_weights, [1., 0.])
    assert tuple(initial.pose_charts.tolist()) == (0, -1)
    # An ordinary motion update composes from that camera, without counting
    # the accumulated pre-initialization displacement a second time.
    delta = pp.mat2SE3(torch.as_tensor(inverse(source) @ receiving, dtype=torch.float32))
    with torch.inference_mode():
        system.mapper.hypothesis_manager.motion_update(delta, pp.se3(torch.full((6,), .01)),
            source_factor=SourceFactor((), np.empty((6, 0)), np.empty(0)) if conditional else None)
    np.testing.assert_allclose(system.mapper.get_current_pose().matrix(), receiving, atol=5e-7)
    np.testing.assert_allclose(initial.pose_mu[0].matrix(), source, atol=3e-7)
    np.testing.assert_allclose(snapshot.pose, source, atol=0)


def test_synchronous_first_valid_frame_keeps_the_same_output_gauge():
    source = pose()
    estimate = SimpleNamespace(pose=source.copy(), depth=np.ones((8, 8)), delta_pose=source.copy(),
        motion_covariance=np.eye(6), diagnostics={'valid': True})
    system = MonocularSystem.__new__(MonocularSystem)
    system.mapper = mapper()
    system.config = SimpleNamespace(mapping_interval=10)
    system.frontend = SimpleNamespace(index=17, step=lambda rgb, timestamp: estimate)
    system.initialized, system.map_alignment = False, np.eye(4)
    result = system.step(np.zeros((8, 8, 3)), .8)
    np.testing.assert_allclose(result.pose, source, atol=3e-7)
    np.testing.assert_allclose(system.mapper.db.nodes[0].pose_mu[0].matrix(), source, atol=3e-7)
    np.testing.assert_allclose(system.mapper.odom_accumulator._accumulated_odom.matrix(), np.eye(4), atol=3e-7)


def test_loaded_map_keeps_its_independent_chart_and_default_initialization(monkeypatch):
    system = mapper()
    # Reuse the actual initialization to create a historical stored node.
    historical = system._init_system(torch.zeros(3, 8, 8), None)
    before = historical.pose_mu.clone()
    monkeypatch.setattr('cross.core.system.new_atlas_center', lambda system: [5., 6., 7.])
    system._processed_frame_num = 1
    current = system._init_system(torch.zeros(3, 8, 8), None, initial_chart_pose=pose())
    expected = np.eye(4)
    expected[:3, 3] = [5., 6., 7.]
    np.testing.assert_allclose(current.pose_mu[0].matrix(), expected, atol=1e-7)
    torch.testing.assert_close(historical.pose_mu, before, rtol=0, atol=0)
    assert historical.pose_charts[0] != current.pose_charts[0]
    assert len(system.hypothesis_manager.hypotheses) == 1


@pytest.mark.parametrize('bad', [np.eye(3), np.diag([2., 1., 1., 1.]),
    np.diag([-1., 1., 1., 1.]), np.diag([1., 1., 1., 2.]), np.full((4, 4), np.nan)])
def test_invalid_initial_chart_pose_is_rejected(bad):
    system = mapper()
    with pytest.raises(ValueError, match='Initial chart pose'):
        system._init_system(torch.zeros(3, 8, 8), None, initial_chart_pose=bad)
    assert system.db.get_size() == 0
