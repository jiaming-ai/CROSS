"""Wrong-floor gate per mode (configs/base/vertical_gate.yaml): on in the modes with gravity-aligned metric odometry that
passed the 2026-10-16 screen; off elsewhere, and off in code for monocular DPVO odometry (it reuses mono_wheel.yaml)."""
import pytest

from cross.core.config import load_config
from cross.pipeline import mode_config_files, vertical_gate_odometry_ok

ON = [("stereo", "external", None), ("stereo", "basalt", None), ("mono", "external", "ff"), ("mono", "vgio", "ff"),
      ("mono", "vio", "ff")]
OFF = [("stereo", "vgio", None), ("stereo", "visual", None), ("rgbd", "external", None), ("rgbd", "basalt", None),
       ("rgbd", "visual", None), ("mono", "visual", None), ("mono", "external", None)]


def gate(files):
    c = load_config(*files).mapping
    return c.loop_closure.anchor_needs_posterior, c.loop_closure.noise.gate_vertical_ratio, c.hypothesis.h0_innovation_gate


@pytest.mark.parametrize("mode,odo,est", ON)
def test_on_in_gravity_aligned_modes(mode, odo, est):
    assert gate(mode_config_files(mode, odo, est)) == (True, 0.25, 16.81)


@pytest.mark.parametrize("mode,odo,est", OFF)
def test_off_elsewhere(mode, odo, est):
    assert gate(mode_config_files(mode, odo, est)) == (False, 1.0, 0.0)


def test_monocular_dpvo_turned_off_in_code():
    assert not vertical_gate_odometry_ok("mono", "visual")
    assert vertical_gate_odometry_ok("mono", "external") and vertical_gate_odometry_ok("stereo", "visual")
