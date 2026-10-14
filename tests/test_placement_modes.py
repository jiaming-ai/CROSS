"""Placement rule per mode (configs/base/placement.yaml): on in the modes with metric, rigid odometry; off elsewhere, and
off in code for monocular DPVO odometry (it reuses mono_wheel.yaml).  The strict overlay only tightens it."""
import pytest

from cross.core.config import load_config
from cross.pipeline import mode_config_files, placement_odometry_ok

ON = [("stereo", "external", None), ("stereo", "basalt", None), ("mono", "external", "ff"), ("mono", "vgio", "ff"),
      ("mono", "vio", "ff")]
OFF = [("stereo", "vgio", None), ("stereo", "visual", None), ("rgbd", "external", None), ("rgbd", "basalt", None),
       ("rgbd", "visual", None), ("mono", "visual", None), ("mono", "external", None)]


def hyp(files):
    return load_config(*files).mapping.hypothesis


@pytest.mark.parametrize("mode,odo,est", ON)
def test_on_in_metric_odometry_modes(mode, odo, est):
    h = hyp(mode_config_files(mode, odo, est))
    assert h.reloc_placement and h.reloc_placement_min_steps == 6 and h.reloc_placement_rival_min_steps == 2
    assert h.reloc_placement_places_after_m == 20.0 and h.reloc_placement_min_spread_m == 0.0


@pytest.mark.parametrize("mode,odo,est", OFF)
def test_off_elsewhere(mode, odo, est):
    assert not hyp(mode_config_files(mode, odo, est)).reloc_placement


def test_strict_overlay_tightens_only():
    h = hyp(mode_config_files("stereo", "external") + ["configs/overlays/strict_merge.yaml"])
    assert h.reloc_placement and h.reloc_placement_min_spread_m == 1.5 and h.reloc_placement_min_steps == 3
    assert not hyp(mode_config_files("rgbd", "external") + ["configs/overlays/strict_merge.yaml"]).reloc_placement


def test_monocular_dpvo_turned_off_in_code():
    assert not placement_odometry_ok("mono", "visual")
    assert placement_odometry_ok("mono", "external") and placement_odometry_ok("stereo", "visual")
