"""Simulated training data for CROSS Nav: HSSD scenes rendered with Blender (via the SimChange code) into the scene
cache, with world BEV labels computed from the scene meshes (complete also where no camera looked).

Pipeline per scene (cross_nav/sim/pipeline.py): HSSD download (hssd_dl.py) -> Blender scene (SimChange
hssd_to_blend.py) -> voxelised mesh geometry (blend_geom.py, bpy) -> mesh BEV + camera trajectories (mesh_bev.py,
trajgen.py) -> Cycles RGB-D renders into the scene cache (render_seq.py, bpy) -> final BEV label with per-frame camera
heights (mesh_bev.py).
"""
