"""cross/pipeline.py: mode inputs, observation cadence, map-frame output between observations, sessions."""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pp = pytest.importorskip("pypose")

from cross.mono.frontend import MonoEstimate  # noqa: E402
from cross.pipeline import OdometryFrontend, Pipeline, restrict_inputs  # noqa: E402


def translation(x):
    T = np.eye(4)
    T[0, 3] = x
    return T


class FakeMapper:
    """Records the observations; its belief is the frontend pose shifted by `offset` (a fixed map alignment)."""

    def __init__(self, offset=10.0):
        self.calls, self.loaded, self.offset = [], [], offset
        self.pose = np.eye(4)
        self.last_added_kf_id = None
        self.hypothesis_manager = SimpleNamespace(dist=None)

    def step(self, obs, data=None):
        self.calls.append(obs)
        if obs.get("delta_pose") is not None:
            self.pose = self.pose @ obs["delta_pose"]
        if obs.get("rgb") is not None:
            mu = pp.mat2SE3(torch.tensor(translation(self.offset) @ self.pose, dtype=torch.float32)).unsqueeze(0)
            self.hypothesis_manager.dist = (mu, None, torch.ones(1))

    def get_current_pose(self):
        return self.hypothesis_manager.dist[0][0]

    def load_map(self, path):
        self.loaded.append(path)


class FakeFrontend:
    def __init__(self, valid_from=0):
        self.index, self.pose, self.valid_from = 0, np.eye(4), valid_from

    def track(self, frame):
        delta = translation(1.0) if self.index else np.eye(4)
        self.pose = self.pose @ delta
        self.index += 1
        return MonoEstimate(frame["timestamp"], self.pose.copy(), delta, np.eye(6) * 1e-4, None,
                            dict(valid=self.index - 1 >= self.valid_from))


def frame(i, **extra):
    return dict(rgb=np.zeros((4, 4, 3), np.uint8), timestamp=float(i), depth=np.ones((4, 4), np.float32),
                rgb_right=np.zeros((4, 4, 3), np.uint8), delta_pose=translation(0.5), **extra)


def to_mat(p):
    return p.matrix().detach().cpu().numpy().astype(np.float64)


def test_inputs_of_each_mode():
    f = frame(0)
    mono = restrict_inputs(f, "mono", "visual")
    assert mono["depth"] is None and mono["rgb_right"] is None and mono["delta_pose"] is None
    rgbd = restrict_inputs(f, "rgbd", "external")
    assert rgbd["depth"] is not None and rgbd["rgb_right"] is None and rgbd["delta_pose"] is not None
    stereo = restrict_inputs(f, "stereo", "visual")
    assert stereo["rgb_right"] is not None and stereo["delta_pose"] is None
    assert f["depth"] is not None          # the loader's frame is not modified


def test_external_odometry_passes_frames_unchanged():
    mapper = FakeMapper()
    session = Pipeline(mapper, None, mode="rgbd", odometry="external")
    f = frame(0)
    session.process(f)
    assert mapper.calls == [f] and session.mapped_now


def test_observation_cadence_and_map_frame_poses_between_observations():
    mapper = FakeMapper(offset=10.0)
    session = Pipeline(mapper, FakeFrontend(valid_from=2), mapping_interval=3, mode="stereo", odometry="visual")
    for i in range(9):
        session.process(frame(i))
        c0, best, _ = session.belief(to_mat)
        if session.initialized:
            # map frame = frontend pose + the mapper's fixed offset, observed or not
            assert np.allclose(c0[:3, 3], [10.0 + i, 0, 0], atol=1e-4)
            assert np.allclose(best, c0, atol=1e-4)
    observed = [i for i, c in enumerate(mapper.calls) if c["rgb"] is not None]
    assert observed == [2, 3, 6]            # first valid frame, then every third frame
    assert all(c["delta_pose"] is not None for c in mapper.calls)       # motion reaches the back end every frame
    assert mapper.calls[2]["initial_chart_pose"] is not None and mapper.calls[3]["initial_chart_pose"] is None
    assert mapper.calls[2]["rgb_right"] is not None and mapper.calls[4]["rgb_right"] is None
    assert np.allclose(mapper.calls[3]["depth"], 1.0)                  # the frame's own depth (stereo mode)


def test_mono_mode_predicts_depth_for_observations():
    calls = []

    class Depth:
        def predict_metric(self, rgb, K, shape):
            calls.append(shape)
            return np.full(shape, 2.0, np.float32)
    mapper = FakeMapper()
    session = Pipeline(mapper, FakeFrontend(), mapping_interval=2, mode="mono", odometry="visual",
                       depth_model=Depth(), K=np.eye(3))
    for i in range(4):
        session.process(frame(i))
    assert len(calls) == 2 and all(np.allclose(c["depth"], 2.0) for c in mapper.calls if c["rgb"] is not None)


def test_load_map_starts_a_fresh_frontend():
    mapper = FakeMapper()
    session = Pipeline(mapper, FakeFrontend(), 1, "rgbd", "visual", frontend_factory=FakeFrontend)
    session.process(frame(0))
    first = session.frontend
    session.load_map("map.pkl")
    assert session.frontend is not first and session.frontend.index == 0 and not session.initialized
    assert mapper.loaded == ["map.pkl"]


def test_odometry_frontend_composes_dataset_motion():
    frontend = OdometryFrontend()
    poses = [frontend.track(frame(i)).pose for i in range(3)]
    assert np.allclose([p[0, 3] for p in poses], [0.0, 0.5, 1.0])      # the first frame's motion is ignored


def test_mode_config_files_and_left_image():
    """The shipped configuration of each mode: the stereo mode keeps the pair and its cadence with every odometry, 6
    images per pass (stereo DPVO 10, the others 8); the opt-in low-bandwidth overlay observes on the left image (no stereo anchors) with
    6 references at half the rate; the fast overlays (Basalt, DPVO) keep the pair, at 416 px."""
    import os
    from cross.core.config import load_config
    from cross.pipeline import CONFIG_DIR, apply_right_image, mode_config_files
    lowband = os.path.join(CONFIG_DIR, "overlays", "stereo_lowband.yaml")
    rel = lambda files: [os.path.relpath(f, CONFIG_DIR) for f in files]          # noqa: E731
    assert rel(mode_config_files("rgbd", "external")) == ["local/rgbd_wheel.yaml"]
    assert rel(mode_config_files("stereo", "basalt", "cloud")) == ["cloud/stereo_basalt.yaml"]
    assert rel(mode_config_files("stereo", "external", "cloud", fast=True)) == ["cloud/stereo_wheel.yaml",
                                                                                "overlays/fast/stereo_wheel.yaml"]
    assert mode_config_files("stereo", "external", "local-fast") == mode_config_files("stereo", "external", fast=True)
    for odometry in ("external", "basalt", "vgio", "visual"):
        for fast in (False, True):
            pe = apply_right_image(load_config(*mode_config_files("stereo", odometry, fast=fast))).pose_est
            assert (pe.obs_min_translation, pe.obs_min_rotation, pe.obs_max_interval_steps) == (0.3, 0.15, 3)
            assert pe.ff.right_image == "pair" and pe.ff.use_curr_anchor
            # 8 images (pair + 6 references) except stereo DPVO: 10 (+ 2 reference right images)
            refs = (4, 0) if fast and odometry == "basalt" else (6, 2 if odometry == "visual" else 0)
            assert (pe.ff.max_refs, pe.ff.n_ref_anchors) == refs
            assert pe.ff.image_resolution == (416 if fast and odometry in ("basalt", "visual") else 512)
    pe = apply_right_image(load_config(*mode_config_files("stereo", "external"), lowband)).pose_est
    assert (pe.obs_min_translation, pe.obs_min_rotation, pe.obs_max_interval_steps) == (0.6, 0.3, 6)
    assert pe.ff.right_image == "left" and not pe.ff.use_curr_anchor and pe.ff.n_ref_anchors == 0
    assert pe.ff.use_odom_anchor and pe.ff.use_map_anchors and pe.ff.max_refs == 6


def test_profile_folders():
    """Every deployment folder and the fast overlay have one file per mode; the folders differ from each other only
    where they should (phase 1: nowhere), the fast overlay changes only the image size of the feed-forward modes (not
    RGB-D, not mono DPVO); the cloud keeps the stereo pair."""
    import dataclasses
    import os
    from cross.core.config import load_config, load_yaml_layers
    from cross.pipeline import CONFIG_DIR, PROFILES
    modes = sorted(f for f in os.listdir(os.path.join(CONFIG_DIR, "local")) if f.endswith(".yaml"))
    assert len(modes) == 12 and PROFILES == ("local", "cloud")
    for folder in PROFILES + ("overlays/fast",):
        assert sorted(f for f in os.listdir(os.path.join(CONFIG_DIR, folder)) if f.endswith(".yaml")) == modes
    for m in modes:
        local = dataclasses.asdict(load_config(os.path.join(CONFIG_DIR, "local", m)))
        cloud = load_config(os.path.join(CONFIG_DIR, "cloud", m))
        assert local["mapping"]["loop_closure"]["async_pgo"] and dataclasses.asdict(cloud) == local
        assert cloud.pose_est.ff.right_image == "pair"
        fast = dataclasses.asdict(load_config(os.path.join(CONFIG_DIR, "local", m), os.path.join(CONFIG_DIR, "overlays/fast", m)))
        # only stereo + Basalt (6 images at 416 px) and stereo DPVO (416 px) have a fast overlay
        if m in ("stereo_basalt.yaml", "stereo_dpvo.yaml"):
            assert fast["pose_est"]["ff"]["image_resolution"] == 416
            fast["pose_est"]["ff"]["image_resolution"] = 512
        if m == "stereo_basalt.yaml":
            assert (fast["pose_est"]["ff"]["max_refs"], fast["pose_est"]["ff"]["n_ref_anchors"]) == (4, 0)
            fast["pose_est"]["ff"]["max_refs"] = local["pose_est"]["ff"]["max_refs"]
        assert fast == local
    assert load_yaml_layers(os.path.join(CONFIG_DIR, "local", "stereo_imu.yaml"))["imu"] == {"vgio_align": True}
    assert load_yaml_layers(os.path.join(CONFIG_DIR, "cloud", "mono_imu.yaml"))["imu"] == {"vgio_align": False}
    assert not load_config(os.path.join(CONFIG_DIR, "local", "stereo_imu.yaml")).mapping.loop_closure.odom_guard_rescale


def test_named_odometry_and_profile_arguments():
    import argparse
    from cross.pipeline import BASALT_ODOM_FILE, add_session_args, finish_session_args, shipped_config_files
    ap = argparse.ArgumentParser()
    ap.add_argument("--odom-file", default=None)
    add_session_args(ap)
    args = finish_session_args(ap.parse_args(["--mode", "stereo", "--odometry", "basalt"]))
    assert args.odometry == "external" and args.odometry_name == "basalt" and args.odom_file == BASALT_ODOM_FILE
    assert shipped_config_files(args)[0].endswith("local/stereo_basalt.yaml")
    args = finish_session_args(ap.parse_args(["--mode", "rgbd", "--odometry", "basalt", "--odom-file", "x.txt"]))
    assert args.odom_file == "x.txt"
    args = finish_session_args(ap.parse_args(["--mode", "stereo", "--profile", "cloud", "--fast"]))
    files = shipped_config_files(args)
    assert files[0].endswith("cloud/stereo_wheel.yaml") and files[1].endswith("overlays/fast/stereo_wheel.yaml")
    args = finish_session_args(ap.parse_args(["--mode", "stereo", "--profile", "local-fast"]))   # deprecated alias
    assert args.profile == "local" and args.fast and shipped_config_files(args)[1].endswith("overlays/fast/stereo_wheel.yaml")
    args = finish_session_args(ap.parse_args(["--mode", "mono", "--mono-estimator", "ff", "--odometry", "vio"]))
    assert shipped_config_files(args)[0].endswith("local/mono_vio.yaml")


def test_mode_file_names_select_the_mode():
    import argparse
    import pytest
    from cross.pipeline import BASALT_ODOM_FILE, MODE_FILES, add_session_args, finish_session_args, shipped_config_files
    ap = argparse.ArgumentParser()
    ap.add_argument("--odom-file", default=None)
    add_session_args(ap)
    for name, (mode, odometry, estimator) in MODE_FILES.items():   # every name selects its own file
        args = finish_session_args(ap.parse_args(["--mode", name, "--profile", "cloud"]))
        assert args.mode == mode and args.odometry_name == odometry
        assert shipped_config_files(args)[0].endswith(f"cloud/{name}.yaml")
        if mode == "mono":
            assert args.mono_estimator == estimator
    args = finish_session_args(ap.parse_args(["--mode", "mono_imu"]))
    assert (args.mode, args.odometry, args.mono_estimator) == ("mono", "vgio", "ff")
    args = finish_session_args(ap.parse_args(["--mode", "stereo_basalt"]))
    assert args.odometry == "external" and args.odom_file == BASALT_ODOM_FILE
    args = finish_session_args(ap.parse_args(["--mode", "mono", "--odometry", "vgio"]))   # vgio implies VGGT-Omega
    assert args.mono_estimator == "ff" and shipped_config_files(args)[0].endswith("local/mono_imu.yaml")
    args = finish_session_args(ap.parse_args(["--mode", "mono"]))                          # sensor mode: defaults as before
    assert (args.odometry, args.mono_estimator) == ("external", "da3")
    assert finish_session_args(args) is args and args.odometry_name == "external"         # idempotent
    with pytest.raises(ValueError):
        finish_session_args(ap.parse_args(["--mode", "mono_imu", "--odometry", "vio"]))
    with pytest.raises(ValueError):
        finish_session_args(ap.parse_args(["--mode", "mono", "--odometry", "vgio", "--mono-estimator", "da3"]))


def test_left_image_session_drops_the_right_image():
    mapper = FakeMapper()
    mapper.config = SimpleNamespace(pose_est=SimpleNamespace(ff=SimpleNamespace(right_image="left")))
    session = Pipeline(mapper, None, mode="stereo", odometry="external")
    f = frame(0)
    session.process(f)
    assert mapper.calls[0]["rgb_right"] is None and mapper.calls[0]["rgb"] is f["rgb"]
    assert f["rgb_right"] is not None      # the loader's frame is not modified


def test_vgio_stereo_structural_settings():
    from cross.mono.config import MonoConfig
    from cross.pipeline import _vgio_config
    stereo = _vgio_config(MonoConfig(), "stereo", np.eye(4))
    assert not stereo.imu.depth_prior and not stereo.imu.vgio_depth_bias
    assert _vgio_config(MonoConfig(), "mono").imu.depth_prior


def test_mono_ff_shipped_defaults():
    import os
    from cross.core.config import load_config
    from cross.pipeline import mode_config_files
    files = mode_config_files("mono", "visual", mono_estimator="ff")
    assert [os.path.basename(f) for f in files] == ["mono_wheel.yaml"]
    assert [os.path.basename(f) for f in mode_config_files("mono", "visual", mono_estimator="da3")] == ["mono_dpvo.yaml"]
    assert [os.path.basename(f) for f in mode_config_files("stereo", "external")] == ["stereo_wheel.yaml"]
    ft, rel = ["pose_est.ff.checkpoint=finetuned"], ["pose_est.ff.checkpoint=released"]
    cfg = load_config(*files, sets=ft)
    assert cfg.pose_est.ff.map_anchor_min_pair_covis == 0.15
    ff = cfg.pose_est.ff         # the feed-forward back end without a stereo pair
    assert cfg.pose_est.type.value == "ff" and not ff.use_curr_anchor and ff.n_ref_anchors == 0
    assert ff.use_odom_anchor and ff.use_map_anchors and cfg.pose_est.obs_max_interval_steps == 3
    # the fine-tuned checkpoint: its settings (base/vggt_finetuned.yaml: the mapped covisibility head; geometric with
    # the VGGT-inertial odometry) and the outdoor gate
    assert ff.covis_source == "auto"
    vgio = mode_config_files("mono", "vgio", mono_estimator="ff")
    assert [os.path.basename(f) for f in vgio] == ["mono_imu.yaml"]
    assert load_config(*vgio, sets=ft).pose_est.ff.covis_source == "geometric"
    assert load_config(*mode_config_files("stereo", "vgio"), sets=ft).pose_est.ff.covis_source == "geometric"
    for odometry in ("external", "basalt", "visual"):
        assert load_config(*mode_config_files("stereo", odometry), sets=ft).pose_est.ff.covis_source == "auto"
    # DPVO + IMU (vio): the feed-forward back end's base (the head) with the old back end's hypothesis behaviour
    vio = mode_config_files("mono", "vio", mono_estimator="ff")
    assert [os.path.basename(f) for f in vio] == ["mono_vio.yaml"]
    hyp = load_config(*vio, sets=ft).mapping.hypothesis
    assert hyp.adopt_in_mapping_session and hyp.merge_in_mapping_session and not hyp.h0_informative_only
    assert load_config(*vio, sets=ft).pose_est.ff.covis_source == "auto"
    hyp = cfg.mapping.hypothesis                   # mono_wheel (VGGT + DPVO alone): the defaults
    assert not hyp.adopt_in_mapping_session and not hyp.merge_in_mapping_session and hyp.h0_informative_only
    assert [os.path.basename(f) for f in mode_config_files("mono", "vio", "cloud", "ff", fast=True)] == \
        ["mono_vio.yaml", "mono_vio.yaml"]
    outdoor = os.path.join(os.path.dirname(os.path.dirname(files[0])), "overlays", "outdoor.yaml")
    assert load_config(*files, outdoor, sets=ft).pose_est.ff.min_covis == 0.10
    assert load_config(*files, sets=ft).pose_est.ff.min_covis == 0.15
    assert load_config(*files, outdoor, sets=ft).mapping.cluster_eps == 3.0
    # the released checkpoint: the default settings
    for cfg in (load_config(*files, outdoor, sets=rel), load_config(*vgio, outdoor, sets=rel)):
        assert cfg.pose_est.ff.covis_source == "geometric" and cfg.pose_est.ff.min_covis == 0.15
        assert cfg.mapping.cluster_eps == 3.0 and cfg.pose_est.ff.map_anchor_min_pair_covis == 0.15
    assert load_config(*files, sets=rel).pose_est.ff.checkpoint == "models/VGGT-Omega/vggt_omega_1b_512.pt"
    # the DA3 backend never takes the fine-tuned VGGT-Omega's settings
    da3 = load_config(*files, outdoor, sets=["pose_est.ff.backend=da3"])
    assert da3.pose_est.ff.covis_source == "geometric" and da3.pose_est.ff.checkpoint == "models/DA3-LARGE-1.1"


def test_vggt_checkpoint_selection(tmp_path, monkeypatch):
    """pose_est.ff.checkpoint "auto" = the fine-tuned VGGT-Omega when installed, else the released one; a checkpoint is
    fine-tuned when it carries a covis_head_map, and then the files' vggt_finetuned sections apply."""
    import torch
    from cross.core.config import (FINETUNED_VGGT_CHECKPOINT, RELEASED_VGGT_CHECKPOINT, is_finetuned_vggt, load_config,
                                   resolve_ff_checkpoint)
    monkeypatch.chdir(tmp_path)
    assert resolve_ff_checkpoint("auto") == FINETUNED_VGGT_CHECKPOINT           # nothing installed: install.sh's default
    (tmp_path / RELEASED_VGGT_CHECKPOINT).parent.mkdir(parents=True)
    (tmp_path / RELEASED_VGGT_CHECKPOINT).touch()
    assert resolve_ff_checkpoint("auto") == RELEASED_VGGT_CHECKPOINT
    (tmp_path / FINETUNED_VGGT_CHECKPOINT).parent.mkdir(parents=True)
    (tmp_path / FINETUNED_VGGT_CHECKPOINT).touch()
    assert resolve_ff_checkpoint("auto") == FINETUNED_VGGT_CHECKPOINT
    assert resolve_ff_checkpoint("released") == RELEASED_VGGT_CHECKPOINT
    assert resolve_ff_checkpoint("finetuned") == FINETUNED_VGGT_CHECKPOINT
    assert resolve_ff_checkpoint("my/ckpt.pt") == "my/ckpt.pt"
    assert resolve_ff_checkpoint("auto", "da3") == "models/DA3-LARGE-1.1"
    # by content: a covis_head_map makes a checkpoint fine-tuned
    mapped, plain = tmp_path / "mapped.pt", tmp_path / "plain.pt"
    torch.save({"model": {"w": torch.zeros(2)}, "covis_head_map": {"x": [0.0, 1.0], "y": [0.0, 1.0]}}, mapped)
    torch.save({"model": {"w": torch.zeros(2)}}, plain)
    assert is_finetuned_vggt(str(mapped)) and not is_finetuned_vggt(str(plain))
    assert not is_finetuned_vggt(str(tmp_path / "missing.pt"))
    assert is_finetuned_vggt(FINETUNED_VGGT_CHECKPOINT) and not is_finetuned_vggt(RELEASED_VGGT_CHECKPOINT)
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("pose_est:\n  ff:\n    min_covis: 0.2\nvggt_finetuned:\n  pose_est:\n    ff:\n      min_covis: 0.1\n")
    assert load_config(cfg_file, sets=[f"pose_est.ff.checkpoint={mapped}"]).pose_est.ff.min_covis == 0.1
    assert load_config(cfg_file, sets=[f"pose_est.ff.checkpoint={plain}"]).pose_est.ff.min_covis == 0.2
    assert load_config(cfg_file, sets=[f"pose_est.ff.checkpoint={plain}"], vggt_finetuned=True).pose_est.ff.min_covis == 0.1
    assert load_config(cfg_file).pose_est.ff.min_covis == 0.1                      # auto: the installed fine-tuned one
    # a later file or --set overrides a vggt_finetuned section of an earlier file
    later = tmp_path / "later.yaml"
    later.write_text("pose_est:\n  ff:\n    min_covis: 0.3\n")
    assert load_config(cfg_file, later).pose_est.ff.min_covis == 0.3
    assert load_config(cfg_file, sets=["pose_est.ff.min_covis=0.4"]).pose_est.ff.min_covis == 0.4


def test_map_and_reloc_mono_loads_shipped_defaults():
    """scripts/map_and_reloc.py (the stereo-image datasets' runner, also in mono mode) loads the mono files for
    the mono mode's feed-forward estimator (it ignored it until 2026-10-09: KITTI / SimChange mono runs had no pair filter)."""
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "scripts" / "map_and_reloc.py"
    spec = importlib.util.spec_from_file_location("map_and_reloc_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    base = ["--map", "m", "--query", "q", "--out", "o"]
    cfg = mod.make_config(mod.build_parser().parse_args(base + ["--mode", "mono", "--mono-estimator", "ff", "--odometry", "vgio"]))
    assert cfg.pose_est.ff.map_anchor_min_pair_covis == 0.15
    cfg = mod.make_config(mod.build_parser().parse_args(base + ["--mode", "stereo"]))
    assert cfg.pose_est.ff.map_anchor_min_pair_covis == 0.0
    # explicit options of the runner win over the mode's file (they used to be overwritten by the mono presets)
    cfg = mod.make_config(mod.build_parser().parse_args(base + ["--mode", "mono", "--mono-estimator", "ff",
                                                                 "--obs-max-interval", "5"]))
    assert cfg.pose_est.obs_max_interval_steps == 5
    # the checkpoint decides the fine-tuned VGGT-Omega's settings; --set still wins
    from cross.core.config import resolve_ff_checkpoint
    cfg = mod.make_config(mod.build_parser().parse_args(base + ["--mode", "stereo", "--checkpoint", "finetuned"]))
    assert cfg.pose_est.ff.covis_source == "auto" and cfg.pose_est.ff.checkpoint.endswith("vggt_omega_cross_1b_512.pt")
    cfg = mod.make_config(mod.build_parser().parse_args(base + ["--mode", "stereo", "--checkpoint", "released"]))
    assert cfg.pose_est.ff.covis_source == "geometric" and cfg.pose_est.ff.checkpoint.endswith("vggt_omega_1b_512.pt")
    cfg = mod.make_config(mod.build_parser().parse_args(
        base + ["--mode", "stereo", "--set", "pose_est.ff.checkpoint=finetuned", "pose_est.ff.covis_source=min"]))
    assert cfg.pose_est.ff.covis_source == "min"
    assert resolve_ff_checkpoint(cfg.pose_est.ff.checkpoint).endswith("vggt_omega_cross_1b_512.pt")   # the estimator resolves it
    cfg = mod.make_config(mod.build_parser().parse_args(base + ["--mode", "stereo", "--backend", "da3"]))
    assert cfg.pose_est.ff.checkpoint == "models/DA3-LARGE-1.1" and cfg.pose_est.ff.covis_source == "geometric"


def test_config_risks_of_the_left_image():
    """The left-image overlay warns with odometry that can run away and with the fast passes, not with trusted
    external odometry (the low-bandwidth use) or with the stereo pair."""
    import os
    from cross.core.config import load_config
    from cross.pipeline import CONFIG_DIR, config_risks, mode_config_files
    lowband = os.path.join(CONFIG_DIR, "overlays", "stereo_lowband.yaml")
    left = load_config(*mode_config_files("stereo", "external"), lowband)
    assert config_risks(left, "stereo", "external") == []
    assert config_risks(left, "stereo", "external", "odom_left.txt") == []
    for odometry, odom_file in (("basalt", "odom_vio.txt"), ("visual", None), ("external", "odom_vio.txt")):
        (risk,) = config_risks(left, "stereo", odometry, odom_file)
        assert "run away" in risk and "38.2 m" in risk
    assert config_risks(left, "stereo", "vgio") == []          # valid: the frontend keeps the pair
    # the stereo + Basalt fast overlay (4 references, 416 px) under the left image: both caveats (+ the runaway one)
    fast_left = load_config(*mode_config_files("stereo", "basalt", fast=True), lowband)
    risks = config_risks(fast_left, "stereo", "external")
    assert len(risks) == 2 and "28 -> 24" in risks[0] and "416 px" in risks[1] and "not tested" in risks[1]
    assert len(config_risks(fast_left, "stereo", "basalt")) == 3
    few_refs = load_config(*mode_config_files("stereo", "external"), lowband)
    few_refs.pose_est.ff.max_refs = 4
    (risk,) = config_risks(few_refs, "stereo", "external")
    assert "4 references" in risk and "28 -> 24" in risk
    assert config_risks(load_config(*mode_config_files("stereo", "basalt", fast=True)), "stereo", "basalt") == []
    assert config_risks(load_config(*mode_config_files("mono", "basalt", mono_estimator="ff"), lowband), "mono", "basalt") == []


def test_confirm_config_risks(monkeypatch):
    """No terminal: abort unless --accept-config-risk; at a terminal: the answer decides; asked once per process."""
    import io
    import pytest
    import cross.pipeline as P
    monkeypatch.setattr(P, "_ACCEPTED_RISKS", set())
    P.confirm_config_risks([])                                 # nothing to confirm
    monkeypatch.setattr("sys.stdin", io.StringIO(""))          # not a terminal
    with pytest.raises(SystemExit):
        P.confirm_config_risks(["risk a"])
    P.confirm_config_risks(["risk a"], accept=True)
    P.confirm_config_risks(["risk a"])                         # accepted before: not asked again

    class Tty(io.StringIO):
        def isatty(self):
            return True
    monkeypatch.setattr("sys.stdin", Tty())
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    with pytest.raises(SystemExit):
        P.confirm_config_risks(["risk b"])
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    P.confirm_config_risks(["risk b"])
