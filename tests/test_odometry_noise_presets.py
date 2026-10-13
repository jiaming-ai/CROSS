"""Odometry noise presets: the VIO mode files include configs/base/odometry_vio.yaml, the wheel modes keep the generic
values, and configs/overlays/odometry_vio.yaml gives external VIO odometry the VIO values (CPU, config only)."""
import os

import pytest

from cross.core.config import NoiseModelConfig, load_config

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
VIO = ["stereo_basalt", "stereo_imu", "mono_imu", "mono_vio", "mono_basalt", "rgbd_basalt", "rgbd_ff_basalt"]
WHEEL = ["stereo_wheel", "mono_wheel", "rgbd_wheel", "rgbd_ff_wheel"]


def noise(*files):
    return load_config(*[os.path.join(ROOT, f) for f in files], vggt_finetuned=False).mapping.loop_closure.noise


@pytest.mark.parametrize("mode", VIO)
def test_vio_modes_use_the_vio_preset(mode):
    n = noise(f"modes/{mode}.yaml")
    assert (n.odom_k_t, n.odom_floor_t, n.odom_k_r, n.odom_floor_r) == (0.04, 0.0015, 0.03, 0.0005)


@pytest.mark.parametrize("mode", WHEEL)
def test_wheel_modes_keep_the_generic_values(mode):
    n, d = noise(f"modes/{mode}.yaml"), NoiseModelConfig()
    assert (n.odom_k_t, n.odom_k_r, n.odom_floor_r) == (d.odom_k_t, d.odom_k_r, d.odom_floor_r)


def test_overlay_gives_external_vio_the_preset():
    n = noise("modes/stereo_wheel.yaml", "overlays/odometry_vio.yaml")
    assert (n.odom_k_t, n.odom_k_r) == (0.04, 0.03)
