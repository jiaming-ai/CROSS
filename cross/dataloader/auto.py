"""Open a sequence by its path: the dataset format is detected from the folder layout (run.py, examples/)."""

from __future__ import annotations

from pathlib import Path

LOADERS = ("auto", "r3d", "rosbag", "loris", "posed", "stereo")


def detect_loader(path: str) -> str:
    """The loader of a sequence: a posed RGB-D folder (calib.json + rgb/), a stereo folder (calib.json + left/, or a
    KITTI raw / TartanAir V2 / Virtual KITTI 2 sequence), a Record3D `.r3d` file, or by name a ROS bag export or an
    OpenLORIS folder in its original layout.  (TUM RGB-D: convert with scripts/datasets/convert_tum.py.)"""
    p = Path(path)
    if p.is_dir() and (p / "calib.json").is_file():
        if (p / "rgb").is_dir():
            return "posed"
        if (p / "left").is_dir():
            return "stereo"
    if p.is_dir():
        from cross.dataloader.stereo_loader import StereoSequenceLoader
        try:
            StereoSequenceLoader._infer_type(p.resolve())
            return "stereo"
        except ValueError:
            pass
    if p.suffix == ".r3d":
        return "r3d"
    name = str(path)
    if "rosbag" in name or "topomap" in name:
        return "rosbag"
    if "loris" in name or "corridor" in name or "cafe" in name:
        return "loris"
    raise ValueError(f"Could not detect the format of {path}: pass the loader ({', '.join(LOADERS[1:])}); "
                     "docs/GUIDE.md describes the folder layouts")


def load_dataset(path: str, loader: str = "auto", **kwargs):
    """The dataset loader for `path` (`loader` "auto": detect_loader); keyword arguments go to the loader."""
    if loader == "auto":
        loader = detect_loader(path)
    if loader == "r3d":
        from cross.dataloader.r3d_loader import R3DDataset
        return R3DDataset(path, **kwargs)
    if loader == "rosbag":
        from cross.dataloader.rosbag_loader import RosbagLoader
        return RosbagLoader(path, **kwargs)
    if loader == "loris":
        from cross.dataloader.loris import OpenLorisLoader
        return OpenLorisLoader(path, **kwargs)
    if loader == "posed":
        from cross.dataloader.posed_rgbd import PosedRGBDLoader
        return PosedRGBDLoader(path, **kwargs)
    if loader == "stereo":
        from cross.dataloader.stereo_loader import StereoSequenceLoader
        return StereoSequenceLoader(path, **kwargs)
    raise ValueError(f"Unknown loader: {loader} (one of {', '.join(LOADERS)})")
