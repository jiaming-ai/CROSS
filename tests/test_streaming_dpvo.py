from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from cross.mono.config import MonoConfig, ScaleConfig
from cross.mono.frontend import MonoEstimate
from cross.mono.replay_scale import replay
from cross.mono.scale import LogScaleFilter, ScaleObservation
from cross.mono.scaled_motion import ScaledTranslation
from cross.mono.streaming import FrameSnapshot
from cross.mono.streaming_dpvo import ScaleRequest, StreamingDPVOFrontend


def bare_frontend():
    f = StreamingDPVOFrontend.__new__(StreamingDPVOFrontend)
    f.config = MonoConfig(frontend='streaming_dpvo', dpvo_checkpoint='weights', mapping_interval=10,
                          scale=ScaleConfig(interval=20))
    f.index, f.last_timestamp = 0, None
    f.last_received_source = f.last_request_source = -1
    f.last_request_frame = f.last_scale_request_frame = -20
    f.scale_filter, f.translation = LogScaleFilter(f.config.scale), ScaledTranslation()
    f.metric_pose, f.world_std_prefix = np.eye(4), np.zeros(6)
    f.gauge_origin, f.native_origin_inverse = np.eye(4), np.eye(4)
    f.native_frontend = SimpleNamespace(restarts=0, input_index=int, rgb_memory={})
    f.history, f.ready_depths, f.scale_events = {}, [], []
    f.provide_mapping_depth, f.finished = True, False
    f.depth_worker = SimpleNamespace(poll=lambda: [])
    return f


def snapshot(index, valid=True):
    pose = np.eye(4)
    pose[0, 3] = float(index)
    return FrameSnapshot(index, index * .05, np.zeros((8, 8, 3), dtype=np.uint8), {}, pose, np.ones(6), valid)


def test_scale_delivery_keeps_immutable_image_pose_and_accounts_for_delivery_time():
    f = bare_frontend()
    f.index = 11
    source = snapshot(4, valid=False)
    pose = source.pose.copy()
    depth = np.ones((8, 8)) * 2
    request = ScaleRequest(source, np.zeros((24, 2)), np.ones(24), True, 9)
    observation = ScaleObservation(log_scale=np.log(2.), variance=.0144, accepted=True)
    f.depth_worker.poll = lambda: [((request, depth, observation), {'service_seconds': .1})]
    events = f._receive()
    assert f.scale_filter.scale == 2. and events[0]['received_frame'] == 11
    assert events[0]['requested_frame'] == 9 and events[0]['source_frame'] == 4
    item = f.take_depths()[0][0]
    assert item[0] is source and item[1] is depth and not item[0].valid
    np.testing.assert_array_equal(source.pose, pose)
    with pytest.raises(ValueError, match='unique'):
        f._receive()  # A duplicated delivery must not become new information.


def test_future_metric_result_is_rejected():
    f = bare_frontend()
    f.index = 4
    request = ScaleRequest(snapshot(4), np.zeros((24, 2)), np.ones(24), True, 4)
    f.depth_worker.poll = lambda: [((request, np.ones((8, 8)), ScaleObservation()), {})]
    with pytest.raises(ValueError, match='past'):
        f._receive()


def test_mapping_only_depth_does_not_update_metric_filter():
    f = bare_frontend()
    f.index = 9
    request = ScaleRequest(snapshot(4), np.zeros((24, 2)), np.ones(24), False, 8)
    f.depth_worker.poll = lambda: [((request, np.ones((8, 8)),
                                    ScaleObservation(log_scale=2., variance=.1, accepted=True)), {})]
    f._receive()
    assert not f.scale_filter.initialized and len(f.take_depths()) == 1


def test_teacher_compares_frozen_sparse_depth_with_its_actual_image():
    f = bare_frontend()
    f.K, f.depth_stream = np.eye(3), None
    source = snapshot(4)
    f.metric = SimpleNamespace(predict_metric=lambda rgb, K, shape: np.ones(shape) * 2.)
    pixels = np.indices((8, 8))[::-1].reshape(2, -1).T
    request = ScaleRequest(source, pixels, np.ones(64), True, 9)
    returned, depth, observation = f._predict(request)
    assert returned is request and depth.shape == source.rgb.shape[:2]
    assert observation.accepted and observation.log_scale == np.log(2.)


def test_mature_request_copies_patches_and_does_not_repeat_source():
    f = bare_frontend()
    f.index = 9
    source = snapshot(4)
    f.history[4] = source
    patches = torch.ones((8, 24, 3, 3, 3))
    tracker = SimpleNamespace(is_initialized=True, n=8, RES=4,
                              pg=SimpleNamespace(tstamps_=np.arange(8), patches_=patches))
    f.native_frontend = SimpleNamespace(tracker=tracker, restarts=0, input_index=int)
    requests = []
    f.depth_worker.submit = requests.append
    assert f._submit_mature()
    assert requests[0].snapshot is source and requests[0].update_scale
    patches[:] = 19
    np.testing.assert_allclose(requests[0].unit_depth, 1.)
    np.testing.assert_allclose(requests[0].pixels, 4.)
    f.index = 30
    assert not f._submit_mature() and len(requests) == 1


def test_causal_scale_startup_and_bounded_snapshot_history():
    f = bare_frontend()
    f._submit_mature = lambda: False
    class Native:
        rgb_memory = {}
        def step(self, rgb, timestamp):
            self.rgb_memory = {f.index: rgb}
            pose = np.eye(4)
            pose[0, 3] = f.index + 1.
            return MonoEstimate(timestamp, pose, np.eye(4), np.eye(6), None,
                                dict(valid=True, initializing=False, total_seconds=.01))
    f.native_frontend = Native()
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    first = f.step(rgb, 0.)
    captured = f.history[0]
    assert not first.diagnostics['valid'] and np.all(first.pose[:3, 3] == 0.)
    f.scale_filter.update(ScaleObservation(log_scale=np.log(.5), variance=.0144, accepted=True))
    second = f.step(rgb, .05)
    assert second.diagnostics['valid'] and second.pose[0, 3] == 1.
    assert len(f.history) == 1 and 1 in f.history
    assert captured.pose[0, 3] == 0. and not captured.valid
    np.testing.assert_array_equal(first.pose[:3, 3], np.zeros(3))


def test_async_replay_consumes_only_scale_scheduled_results():
    rows = np.zeros((3, 8)); rows[:, 0] = np.arange(3); rows[:, 7] = 1.
    observations = [dict(scale_application='anchored_startup_v1', scale=1., unit_translation=[float(i), 0., 0.])
                    for i in range(3)]
    observation = dict(log_scale=float(np.log(2.)), variance=.01, log_mad=.01,
                       pixels=24, tiles=8, inlier_fraction=1., accepted=True, reason='accepted')
    observations[1]['metric_result_events'] = [dict(observation=observation, scale_update_requested=False)]
    observations[2]['metric_result_events'] = [dict(observation=observation, scale_update_requested=True)]
    result, _ = replay(rows, observations, ScaleConfig())
    np.testing.assert_allclose(result[:, 1], [0., 0., 4.])


@pytest.mark.parametrize('options', [dict(dpvo_checkpoint=None), dict(scale=ScaleConfig(mode='relative')),
                                   dict(delayed_recovery=True), dict(dpvo_metric_bootstrap=True),
                                   dict(conditional_sources=True, chart_aware=True, session_recovery=True)])
def test_incompatible_streaming_dpvo_options_are_explicit(options):
    values = dict(frontend='streaming_dpvo', dpvo_checkpoint='weights')
    values.update(options)
    with pytest.raises(ValueError):
        MonoConfig(**values)


def test_tracker_restart_holds_pose_resets_scale_and_continues_from_last_pose():
    f = bare_frontend()
    f._submit_mature = lambda: False
    script = {0: (1., False), 1: (2., False), 2: (None, True), 3: (0., False), 4: (4., False)}
    class Native:
        rgb_memory = {}
        restarts = 0
        pose = np.eye(4)
        def step(self, rgb, timestamp):
            x, restarted = script[f.index]
            self.rgb_memory = {f.index: rgb}
            if restarted:
                self.restarts += 1          # native pose is held at its last value
            else:
                self.pose = self.pose.copy() if f.index < 3 else Native.origin @ np.eye(4)
                if f.index < 3:
                    self.pose[0, 3] = x
                else:
                    self.pose = Native.origin.copy(); self.pose[0, 3] += x   # new gauge, unit scale 1 again
            return MonoEstimate(timestamp, self.pose.copy(), np.eye(4), np.eye(6), None,
                                dict(valid=not restarted, initializing=restarted, total_seconds=.01,
                                     tracker_restarted=restarted, tracker_restarts=self.restarts))
    native = Native()
    f.native_frontend = native
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    f.scale_filter.update(ScaleObservation(log_scale=np.log(.5), variance=.0144, accepted=True))
    f.step(rgb, 0.)
    held = f.step(rgb, .05).pose.copy()           # 2 units at 0.5 m/unit
    Native.origin = native.pose.copy()
    restart = f.step(rgb, .1)
    assert not restart.diagnostics['valid'] and np.allclose(restart.pose, held)
    assert not f.scale_filter.initialized          # the new gauge needs a new scale
    f.scale_filter.update(ScaleObservation(log_scale=np.log(.25), variance=.0144, accepted=True))
    f.step(rgb, .15)
    after = f.step(rgb, .2)
    assert held[0, 3] == pytest.approx(1.)
    assert after.pose[0, 3] == pytest.approx(held[0, 3] + 4 * .25)   # continues from the held pose in the new scale
