"""Optional process isolation of the unmodified CROSS mapping policy."""

import os
from types import SimpleNamespace

import cv2
import numpy as np
import torch

_system = None


def initialize(K, image_size, config, system_config, device):
    global _system
    from .models import DA3Geometry
    from .streaming import StreamingMonocularSystem
    from .system import MonocularSystem

    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.manual_seed(config.seed)
    cv2.setRNGSeed(config.seed)
    np.random.seed(config.seed)
    if os.environ.get("CROSS_TORCH_HUB"):
        torch.hub.set_dir(os.environ["CROSS_TORCH_HUB"])
    geometry = DA3Geometry(config.pose_model, device, config.resolution)
    wrapper = MonocularSystem(K, image_size, config, system_config, device, SimpleNamespace(geometry=geometry))
    # Reuse the exact mapping operation of the in-process path, with its
    # sole owner and prior snapshot in this process. No nested worker/pool.
    _system = StreamingMonocularSystem.__new__(StreamingMonocularSystem)
    _system.pool, _system.map_stream, _system.previous_snapshot = None, None, None
    _system.mapper, _system.geometry, _system.device = wrapper.mapper, geometry, device
    _system.config = config


def operation(name, value):
    if _system is None:
        raise RuntimeError("Mapping process was not initialized")
    if name == "step":
        alignment, event = _system._map_snapshot(value)
        if _system.device.startswith("cuda"):
            event["mapper_process_peak_gpu_allocated_gb"] = torch.cuda.max_memory_allocated()/1e9
        return alignment, event
    if name == "warmup":
        _system.warmup_mapping(value)
        image = _system.geometry.prepare(value)
        _system.geometry.predict([image, image])
        if _system.device.startswith("cuda"):
            torch.cuda.synchronize()
        return
    if name == "save":
        return _system.mapper.save_map(value)
    if name == "load":
        return _system.mapper.load_map(value)
    if name == "shutdown":
        return _system.mapper.shutdown()
    raise ValueError(f"Unknown mapping operation: {name}")
