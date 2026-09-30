#!/usr/bin/env python3
"""Run VGGT-SLAM's main.py headless (used by run_vggt_slam.py; runs inside the VGGT-SLAM venv).

    <venv>/bin/python vggt_slam_headless.py <VGGT-SLAM dir> [main.py arguments ...]

VGGT-SLAM always starts a viser server on port 8080 and pushes the whole map to it at the end, which collides
between parallel runs and can keep the process alive after main() returns.  The viewer is replaced by a no-op
stub (visualization only: tracking, mapping, loop closure and the pose log are untouched), main.py runs
unmodified, and the process exits hard once the pose log is written.  The peak GPU memory is printed as
`VGGT_SLAM_PEAK_GPU_GB allocated reserved`.
"""

import os
import runpy
import sys


class _NoViewer:
    """Absorbs any attribute access / call chain (viewer.server.scene.add_point_cloud(...), ...)."""

    def __init__(self, *a, **k):
        pass

    def __getattr__(self, name):
        return self

    def __call__(self, *a, **k):
        return self


def main():
    repo = os.path.abspath(sys.argv[1])
    sys.path.insert(0, repo)
    os.chdir(repo)
    import vggt_slam.solver as solver
    solver.Viewer = _NoViewer
    solver.Solver.update_all_submap_vis = lambda self: None
    solver.Solver.update_latest_submap_vis = lambda self: None
    sys.argv = [os.path.join(repo, "main.py")] + sys.argv[2:]
    rc = 0
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException:
        import traceback
        traceback.print_exc()
        rc = 1
    try:
        import torch
        if torch.cuda.is_available():
            print(f"VGGT_SLAM_PEAK_GPU_GB {torch.cuda.max_memory_allocated() / 2**30:.3f} "
                  f"{torch.cuda.max_memory_reserved() / 2**30:.3f}")
    except Exception:
        pass
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)


if __name__ == "__main__":
    main()
