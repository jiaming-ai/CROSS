"""The shipped configuration files hold only keys the configuration classes know (a typo would change nothing), every
overlay loads on top of every mode, a user's file with an unknown key gets a warning, and --config / --set are
repeatable in run.py and the scripts/ runners (a second --config used to replace the first)."""
import importlib.util
import logging
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cross.core.config import load_config   # noqa: E402

CONFIGS = ROOT / "configs"
MODES = sorted((CONFIGS / "modes").glob("*.yaml"))
OVERLAYS = sorted((CONFIGS / "overlays").glob("*.yaml"))


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.name == "cross.core.config" and r.levelno >= logging.WARNING]


@pytest.mark.parametrize("mode", MODES, ids=lambda p: p.stem)
def test_mode_files_and_overlays_have_only_known_keys(mode, caplog):
    caplog.set_level(logging.WARNING, logger="cross.core.config")
    for finetuned in (False, True):          # the vggt_finetuned sections too
        load_config(mode, vggt_finetuned=finetuned)
        load_config(mode, CONFIGS / "overlays" / "fast" / mode.name, vggt_finetuned=finetuned)
        for overlay in OVERLAYS:
            load_config(mode, overlay, vggt_finetuned=finetuned)
    assert _warnings(caplog) == []


def test_mono_and_imu_sections_have_only_known_keys():
    import dataclasses
    from cross.imu.scale_filter import ImuConfig
    from cross.mono.config import MonoConfig

    def check(cls, data, where):
        names = {f.name for f in dataclasses.fields(cls)}
        assert not set(data) - names, f"{where}: unknown keys {sorted(set(data) - names)}"
        for f in dataclasses.fields(cls):
            if f.name in data and isinstance(data[f.name], dict) and dataclasses.is_dataclass(f.default_factory()
                                                                                                   if callable(f.default_factory) else f.default):
                sub = f.default_factory() if callable(f.default_factory) else f.default
                check(type(sub), data[f.name], f"{where}.{f.name}")

    for path in sorted(CONFIGS.rglob("*.yaml")):
        data = yaml.safe_load(path.read_text()) or {}
        if "mono" in data:
            check(MonoConfig, data["mono"], f"{path.relative_to(ROOT)} mono")
        if "imu" in data:
            check(ImuConfig, data["imu"], f"{path.relative_to(ROOT)} imu")


def test_unknown_key_in_a_users_file_is_reported(tmp_path, caplog):
    caplog.set_level(logging.WARNING, logger="cross.core.config")
    mine = tmp_path / "my_robot.yaml"
    mine.write_text("pose_est:\n  obs_max_intervl_steps: 2\nmapping:\n  loop_closure:\n    confidence: 0.9995\n")
    cfg = load_config(CONFIGS / "modes" / "rgbd_wheel.yaml", mine)
    assert cfg.mapping.loop_closure.confidence == 0.9995               # the known key applies
    msgs = _warnings(caplog)
    assert len(msgs) == 1 and "obs_max_intervl_steps" in msgs[0] and "PoseEstConfig" in msgs[0]


def test_unknown_key_in_set_is_an_error():
    with pytest.raises(AttributeError):
        load_config(CONFIGS / "modes" / "rgbd_wheel.yaml", sets=["pose_est.obs_max_intervl_steps=2"])


def _run_py_parser():
    spec = importlib.util.spec_from_file_location("cross_run_cli", ROOT / "run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    return run.build_parser()


@pytest.mark.parametrize("runner", ["run.py", "map_and_reloc", "map_and_reloc_rgbd"])
def test_config_and_set_are_repeatable(runner):
    if runner == "run.py":
        ap, base = _run_py_parser(), ["seq"]
    else:
        ap, base = importlib.import_module(runner).build_parser(), ["--map", "m", "--query", "q", "--out", "o"]
    args = ap.parse_args(base + ["--config", "a.yaml", "b.yaml", "--set", "x.y=1", "--config", "c.yaml",
                                 "--set", "z.w=2"])
    assert args.config == ["a.yaml", "b.yaml", "c.yaml"]
    assert args.set == ["x.y=1", "z.w=2"]
