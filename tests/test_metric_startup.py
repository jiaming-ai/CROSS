"""Metric output must not depend on an arbitrary visual unit during startup."""

import numpy as np
import pytest

from cross.mono.config import ScaleConfig
from cross.mono.replay_scale import replay
from cross.mono.scaled_motion import ScaledTranslation


def test_unknown_scale_holds_translation_then_anchors_all_accumulated_motion():
    state = ScaledTranslation()
    np.testing.assert_array_equal(state.update([1., 2., 3.]), np.zeros(3))
    np.testing.assert_array_equal(state.update([2., 3., 4.]), np.zeros(3))
    np.testing.assert_allclose(state.update([3., 4., 5.], .2), [.6, .8, 1.])
    np.testing.assert_allclose(state.update([4., 5., 6.], .3), [.9, 1.1, 1.3])


def test_metric_trajectory_is_independent_of_visual_gauge_units():
    rng = np.random.default_rng(183)
    positions = rng.normal(size=(50, 3)).cumsum(axis=0)
    scale = np.exp(rng.normal(size=50) * .03)
    first, rescaled = ScaledTranslation(), ScaledTranslation()
    for i, (position, magnitude) in enumerate(zip(positions, scale)):
        x = first.update(position, None if i < 12 else magnitude)
        y = rescaled.update(position * 71., None if i < 12 else magnitude / 71.)
        np.testing.assert_allclose(x, y, atol=1e-12)


def test_relative_mode_and_immediate_metric_start_match_increment_integration():
    positions = np.array([[0., 0., 0.], [2., 1., 3.], [3., 4., 5.]])
    for scale in (1., .4):
        state = ScaledTranslation()
        np.testing.assert_allclose([state.update(p, scale) for p in positions], scale * positions)


@pytest.mark.parametrize('position,scale', [([np.nan, 0, 0], 1.), ([1., 2.], 1.),
                                          ([0., 0., 0.], 0.), ([0., 0., 0.], np.inf)])
def test_invalid_motion_or_scale_rejected(position, scale):
    with pytest.raises(ValueError):
        ScaledTranslation().update(position, scale)


def test_initialized_scale_cannot_disappear():
    state = ScaledTranslation()
    state.update([1., 0., 0.], 2.)
    with pytest.raises(ValueError):
        state.update([2., 0., 0.])


def test_recorded_unit_positions_allow_relative_replay_before_metric_startup():
    rows = np.zeros((4, 8))
    rows[:, 0] = np.arange(4)
    rows[:, 7] = 1.
    observations = [dict(scale_application='anchored_startup_v1', scale=1.,
                         unit_translation=None if i == 0 else [float(i), 0., 0.]) for i in range(4)]
    observations[2]['scale_observation'] = dict(log_scale=float(np.log(.4)), variance=.01,
        log_mad=.01, tiles=8, pixels=100, inlier_fraction=1., accepted=True, reason='accepted')
    for d in observations[2:]:
        d['scale'] = .4
    original, _ = replay(rows, observations, ScaleConfig(mode='filtered'))
    np.testing.assert_allclose(original[:, 1], [0., 0., .8, 1.2])
    relative, _ = replay(original, observations, ScaleConfig(mode='relative'))
    np.testing.assert_allclose(relative[:, 1], [0., 1., 2., 3.])
    repeated, _ = replay(original, observations, ScaleConfig(mode='filtered'))
    np.testing.assert_allclose(repeated, original)
