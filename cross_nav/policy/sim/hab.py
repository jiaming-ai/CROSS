"""Habitat-sim scenes for the ground-robot point-goal task (habitat environment, no torch).

Robot: navmesh cylinder radius 0.18 m, height 0.9 m, max climb 0.2 m (0.1 m splits HSSD floors into one island per
room); one forward RGB camera 512x384, 90 deg horizontal FoV, 0.8 m above the floor, level. Door objects (closed
solid slabs in HSSD) are removed, so doorways are open in both the navmesh and the renders. The navmesh treats static
furniture as obstacles and is cached per scene. HSSD navmeshes include outdoor yards (black sky): episodes use indoor
points only, by the scene's room polygons (semantics/scenes/<id>.semantic_config.json).
"""
from __future__ import annotations

import csv
import json
import math
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

RADIUS, HEIGHT, MAX_CLIMB, CELL, CAM_H, W, H, HFOV = .18, .9, .2, .05, .8, 512, 384, 90.
EXPERT_RADII = (.3, .24)    # expert clearance ladder: ~12 cm, else ~6 cm beyond the robot radius (narrow doorways)
OUTDOOR = {'outdoor', 'porch/terrace/deck/driveway', 'balcony'}


def scene_config(root):
    return str(Path(root) / 'hssd-hab.scene_dataset_config.json')


@lru_cache(maxsize=1)
def door_ids(root):
    op = Path(root) / 'objects/openings'
    with open(Path(root) / 'metadata/fpmodels-with-decomposed.csv') as f:
        return frozenset(r['id'] for r in csv.DictReader(f) if r['main_category'] == 'door' and (op / f"{r['id']}.glb").exists())


def open_doors(sim, root):
    rom, doors = sim.get_rigid_object_manager(), door_ids(root)
    removed = 0
    for h in list(rom.get_object_handles()):
        if h.split(':')[0].rstrip('_') in doors:
            rom.remove_object_by_handle(h); removed += 1
    return removed


def make_sim(root, scene, gpu=0, navmesh_dir=None, depth=False):
    import habitat_sim
    sc = habitat_sim.SimulatorConfiguration()
    sc.scene_dataset_config_file = scene_config(root)
    sc.scene_id = scene
    sc.gpu_device_id = gpu
    sc.enable_physics = False
    sensors = []
    for uuid, kind in [('rgb', habitat_sim.SensorType.COLOR)] + ([('depth', habitat_sim.SensorType.DEPTH)] if depth else []):
        s = habitat_sim.CameraSensorSpec()
        s.uuid, s.sensor_type, s.resolution = uuid, kind, [H, W]
        s.position, s.hfov = [0., CAM_H, 0.], HFOV
        sensors.append(s)
    agent = habitat_sim.agent.AgentConfiguration(height=HEIGHT, radius=RADIUS, sensor_specifications=sensors)
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sc, [agent]))
    open_doors(sim, root)
    navmesh(sim, sim.pathfinder, scene, RADIUS, navmesh_dir)
    sim.expert_pathfinders = [navmesh(sim, habitat_sim.nav.PathFinder(), scene, r, navmesh_dir) for r in EXPERT_RADII]
    return sim


def navmesh(sim, pf, scene, radius, navmesh_dir=None):
    import habitat_sim
    cached = Path(navmesh_dir) / f'{scene}_r{radius}_c{MAX_CLIMB}_opendoors.navmesh' if navmesh_dir else None
    if cached is not None and cached.exists():
        pf.load_nav_mesh(str(cached))
        return pf
    ns = habitat_sim.NavMeshSettings()
    ns.set_defaults()
    ns.agent_radius, ns.agent_height, ns.agent_max_climb, ns.cell_size = radius, HEIGHT, MAX_CLIMB, CELL
    ns.include_static_objects = True
    if not sim.recompute_navmesh(pf, ns):
        raise RuntimeError(f'navmesh failed: {scene} r={radius}')
    if cached is not None:
        cached.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(cached) + f'.{os.getpid()}.tmp'
        pf.save_nav_mesh(tmp)
        os.replace(tmp, cached)
    return pf


def expert_path(sim, pos, goal):
    """Shortest path with clearance, for the expert labels.

    Planned on the most inflated navmesh of EXPERT_RADII on which the robot (snapped at most 0.6 m) and the goal's
    nearest point (any distance up to 1 m; the last stretch to a goal beside a wall follows the robot-radius navmesh)
    are connected, without a detour longer than 2x the robot-radius geodesic + 3 m; the robot-radius path otherwise.
    Returns (geodesic distance on the robot navmesh, (N,3) path points or None)."""
    d, pts = geodesic(sim.pathfinder, pos, goal)
    if pts is None:
        return d, None
    for pf2 in getattr(sim, 'expert_pathfinders', []):
        a, b = np.asarray(pf2.snap_point(np.asarray(pos, np.float32))), np.asarray(pf2.snap_point(np.asarray(goal, np.float32)))
        if not (np.isfinite(a).all() and np.isfinite(b).all()) or np.linalg.norm((a - pos)[[0, 2]]) > .6 \
                or np.linalg.norm((b - goal)[[0, 2]]) > 1.:
            continue
        d2, inner = geodesic(pf2, a, b)
        if inner is None or d2 > 2. * d + 3.:
            continue
        _, tail = geodesic(sim.pathfinder, b, goal)
        if tail is None:
            continue
        return d, np.vstack(([pos], inner, tail[1:]))
    return d, pts


class Rooms:
    """Room polygons of an HSSD scene; indoor(p) for a 3D point p (x, y, z)."""
    def __init__(self, root, scene):
        cfg = json.loads((Path(root) / 'semantics/scenes' / f'{scene}.semantic_config.json').read_text())
        self.regions = [(r['label'], np.asarray(r['poly_loop'], np.float64)[:, [0, 2]], r['floor_height'], r['extrusion_height'])
                        for r in cfg['region_annotations']]

    @staticmethod
    def _inside(x, z, poly):
        x1, z1 = poly[:, 0], poly[:, 1]
        x2, z2 = np.roll(x1, -1), np.roll(z1, -1)
        cross = ((z1 > z) != (z2 > z)) & (x < (x2 - x1) * (z - z1) / np.where(z2 == z1, 1e-12, z2 - z1) + x1)
        return bool(cross.sum() % 2)

    def region(self, p):
        for label, poly, fh, eh in self.regions:
            if fh - .3 <= p[1] <= fh + eh and self._inside(p[0], p[2], poly):
                return label
        return None

    def indoor(self, p):
        r = self.region(p)
        return r is not None and r not in OUTDOOR


def indoor_island(pf, rooms, samples=3000):
    """Island with the largest indoor navigable area (area-uniform samples)."""
    area = {}
    for _ in range(samples):
        p = np.asarray(pf.get_random_navigable_point())
        if np.isfinite(p).all() and rooms.indoor(p):
            i = pf.get_island(p)
            area[i] = area.get(i, 0) + 1
    if not area:
        raise RuntimeError('no indoor navigable area')
    return max(area, key=area.get), max(area.values()) / samples * pf.navigable_area


def largest_island(pf):
    n = pf.num_islands
    return max(range(n), key=lambda i: pf.island_area(i))


def geodesic(pf, a, b):
    """(distance, (N,3) points) of the navmesh shortest path, (inf, None) when unreachable."""
    import habitat_sim
    sp = habitat_sim.ShortestPath()
    sp.requested_start, sp.requested_end = np.asarray(a, np.float32), np.asarray(b, np.float32)
    if not pf.find_path(sp) or not np.isfinite(sp.geodesic_distance) or len(sp.points) == 0:
        return math.inf, None
    return float(sp.geodesic_distance), np.asarray(sp.points, dtype=np.float64)


def set_hfov(sim, hfov):
    """Horizontal field of view of all cameras (degrees); the image size stays W x H."""
    import magnum as mn
    for sensor in sim._sensors.values():
        sensor._sensor_object.fov = mn.Deg(float(hfov))


def camera_sample(rng, mode):
    """Per-episode camera: nominal (0.8 m, 90 deg, level) or randomised height / FoV / pitch / roll for generality."""
    if mode == 'nominal':
        return {'height': CAM_H, 'hfov': HFOV, 'pitch': 0., 'roll': 0.}
    return {'height': float(rng.uniform(.35, 1.2)), 'hfov': float(rng.uniform(70., 110.)),
            'pitch': float(np.radians(rng.uniform(-15., 5.))), 'roll': float(np.radians(rng.normal(0., 1.5)))}


def render(sim, pos, yaw, sensor='rgb', camera=None):
    """Render at robot floor position `pos` and heading `yaw`; camera = {'height', 'pitch', 'roll'} (radians, pitch up +)."""
    import habitat_sim
    import magnum as mn
    from habitat_sim.utils.common import quat_from_angle_axis, quat_from_magnum
    cam = camera or {'height': CAM_H, 'pitch': 0., 'roll': 0.}
    st = habitat_sim.AgentState()
    st.position = np.asarray(pos, np.float32)
    st.rotation = quat_from_angle_axis(float(yaw), np.array([0., 1., 0.]))
    rot = (mn.Quaternion.rotation(mn.Rad(float(yaw)), mn.Vector3.y_axis())
           * mn.Quaternion.rotation(mn.Rad(cam['pitch']), mn.Vector3.x_axis())
           * mn.Quaternion.rotation(mn.Rad(cam['roll']), mn.Vector3.z_axis()))
    pose = habitat_sim.agent.SixDOFPose()
    pose.position = np.asarray(pos, np.float32) + np.array([0., cam['height'], 0.], np.float32)
    pose.rotation = quat_from_magnum(rot)
    st.sensor_states = {k: pose for k in sim._sensors}
    sim.get_agent(0).set_state(st, reset_sensors=False, infer_sensor_states=False)
    obs = sim.get_sensor_observations()
    return obs[sensor][..., :3] if sensor == 'rgb' else obs[sensor]
