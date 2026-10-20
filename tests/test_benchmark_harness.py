"""benchmark/run.py command lines of the CROSS systems: the stereo VIO odometry source (--odom-source vio) runs the
shipped *_basalt mode file, not the *_wheel file the system's own arguments select; outdoor datasets add the outdoor
overlay; experiments are synchronous (--reproducible)."""
import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def bench_run():
    """benchmark/run.py under its own module name (the repository's top-level run.py is `run` too)."""
    spec = importlib.util.spec_from_file_location("bench_run", ROOT / "benchmark" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cmd(tmp_path, system, dataset="openloris", scene="office", setup="stereo", odom_source="external", args=""):
    run = bench_run()
    a = argparse.Namespace(dataset=dataset, scene=scene, system=system, setup=setup, seed=0, task="query", seq=None,
                           query=None, data=str(tmp_path / "data"), out=str(tmp_path / "out"), odom_source=odom_source,
                           map_from="", variant="", args=args, reeval=False, force=False, wait_for_map=False, redo=[],
                           only=[], keep_maps=False, keep_rows=False, timeout=60.0)
    job = run.Job(a)
    sc = job.scene
    return [str(c) for c in job.cross_cmd(sc["map"], sc["queries"][0], tmp_path / "out" / "run", [])]


def session_args(command):
    """The arguments the runner the harness calls (scripts/map_and_reloc*.py) parses from its command line."""
    import map_and_reloc
    import map_and_reloc_rgbd

    from cross.pipeline import finish_session_args
    rgbd_runner = command[1].endswith("map_and_reloc_rgbd.py")
    runner = map_and_reloc_rgbd if rgbd_runner else map_and_reloc
    args = finish_session_args(runner.build_parser().parse_args(command[2:]))
    args.mode = args.mode or ("rgbd" if rgbd_runner else "stereo")      # as the runners' main()
    return args


@pytest.mark.parametrize("system, mode_file", [("cross_stereo", "stereo_basalt.yaml"),
                                               ("cross_rgbd", "rgbd_basalt.yaml"),
                                               ("cross_mono_ff_odom", "mono_basalt.yaml")])
def test_vio_odometry_runs_the_basalt_mode_file(tmp_path, system, mode_file):
    from cross.pipeline import shipped_config_files
    setup = {"cross_stereo": "stereo", "cross_rgbd": "rgbd", "cross_mono_ff_odom": "mono"}[system]
    c = cmd(tmp_path, system, dataset="rover", scene="campus_large", setup=setup, odom_source="vio")
    assert c[c.index("--odom-file") + 1] == "odom_vio.txt"
    args = session_args(c)
    assert args.odometry_name == "basalt" and args.odom_file == "odom_vio.txt"
    assert shipped_config_files(args)[0].endswith(f"modes/{mode_file}")


def test_external_odometry_keeps_the_wheel_mode_file(tmp_path):
    from cross.pipeline import shipped_config_files
    c = cmd(tmp_path, "cross_stereo")
    assert "--odom-file" not in c and "basalt" not in c
    assert shipped_config_files(session_args(c))[0].endswith("modes/stereo_wheel.yaml")


def test_outdoor_overlay_and_reproducible(tmp_path):
    c = cmd(tmp_path, "cross_stereo", dataset="rover", scene="campus_large")
    assert c[c.index("--config") + 1].endswith("configs/overlays/outdoor.yaml")
    assert "--reproducible" in c
    assert "--config" not in cmd(tmp_path, "cross_stereo")          # indoor: no overlay


def test_a_variants_config_adds_to_the_outdoor_overlay(tmp_path):
    """--args "--config X" of a variant comes after the harness's overlay; both apply (--config extends)."""
    c = cmd(tmp_path, "cross_stereo", dataset="rover", scene="campus_large",
            args="--config configs/overlays/strict_merge.yaml")
    files = session_args(c).config
    assert files[0].endswith("configs/overlays/outdoor.yaml") and files[-1] == "configs/overlays/strict_merge.yaml"
