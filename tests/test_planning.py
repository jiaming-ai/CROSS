"""Topological planning over the keyframe graph, without neural models: SimpleTopo (System.topo_map: odometry and
proximity edges, Dijkstra / A*) and the PlanningSystem of examples/demo.py (sparse backbone and shortcut edges)."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pp = pytest.importorskip("pypose")

from cross.core.config import HypothesisConfig, SystemConfig   # noqa: E402
from cross.core.hypothesis import HypothesisManager             # noqa: E402
from cross.core.types import Edge, EdgeType, Keyframe            # noqa: E402

# a U-shaped corridor on the ground plane (camera x, z): 10 m out along x, back 2 m to the side; ids 0..20
POSITIONS = [(float(x), 0.0) for x in range(11)] + [(float(x), 2.0) for x in range(10, 0, -1)]
CORRIDOR = 10.0 + 2.0 + 9.0          # length of the odometry chain from keyframe 0 to keyframe 20


def _graph(temporary=()):
    system = SimpleNamespace(device="cpu", topo_map=None, loaded_node_ids=frozenset(), config=SystemConfig())
    hm = HypothesisManager(system, 1, HypothesisConfig())
    hm.nodes, hm.odom_edges = {}, {}
    for i, (x, z) in enumerate(POSITIONS):
        pose = pp.identity_SE3(1)
        pose[0, 0], pose[0, 2] = x, z
        kf = Keyframe(pose, pp.se3(torch.ones(1, 6) * 0.1), torch.tensor([1.0]), temporary=i in temporary)
        kf.id = i
        hm.nodes[i] = kf
    for a in range(len(POSITIONS) - 1):
        rel = hm.nodes[a].pose_mu[0].Inv() @ hm.nodes[a + 1].pose_mu[0]
        hm.odom_edges[(a, a + 1)] = Edge(rel, pp.se3(torch.ones(6) * 0.1), EdgeType.ODOMETRY)
    hm.dist = (pp.identity_SE3(1), pp.se3(torch.ones(1, 6) * 0.1), torch.tensor([1.0]))
    system.hypothesis_manager = hm
    system.get_current_kf = lambda: {"closest_perm_node": hm.nodes[0]}
    return system, hm


def _topo(system, thresh):
    from cross.core.simple_topo import SimpleTopo, SimpleTopoConfig
    topo = SimpleTopo(system, SimpleTopoConfig(proximity_distance_thresh=thresh))
    topo.rebuild_graph()
    system.topo_map = topo
    return topo


@pytest.mark.parametrize("algorithm", ["dijkstra", "astar"])
def test_without_shortcuts_the_plan_follows_the_odometry_chain(algorithm):
    system, _ = _graph()
    topo = _topo(system, thresh=0.5)              # nothing closer than 1 m: no proximity edges
    assert not topo.proximity_edges
    plan = topo.plan_to_goal(20, algorithm=algorithm)
    assert plan["path"] == list(range(21))
    assert plan["cost"] == pytest.approx(CORRIDOR, abs=1e-4)


@pytest.mark.parametrize("algorithm", ["dijkstra", "astar"])
def test_a_proximity_edge_across_the_corridor_shortens_the_plan(algorithm):
    system, _ = _graph()
    topo = _topo(system, thresh=2.5)              # keyframes 0 (0, 0) and 20 (1, 2) are 2.24 m apart
    assert (0, 20) in topo.proximity_edges
    assert (0, 1) not in topo.proximity_edges     # never parallel to an odometry edge
    plan = topo.plan_to_goal(20, algorithm=algorithm)
    assert plan["path"] == [0, 20]
    assert plan["cost"] == pytest.approx(5 ** 0.5, abs=1e-4)


def test_temporary_keyframes_get_no_proximity_edges_and_unknown_goals_no_plan():
    system, _ = _graph(temporary={20})
    topo = _topo(system, thresh=2.5)
    assert all(20 not in key for key in topo.proximity_edges)
    plan = topo.plan_to_goal(20, algorithm="astar")   # reached along the chain, through the temporary keyframe
    assert plan["path"][0] == 0 and plan["path"][-1] == 20
    assert topo.plan(0, 999) is None


def test_next_waypoints_follow_the_plan():
    system, _ = _graph()
    topo = _topo(system, thresh=0.5)
    topo.plan_to_goal(20, algorithm="astar")
    nxt = topo.get_next_waypoint(next_n_waypoints=3)
    assert nxt is not None and not nxt["reached_goal"]
    assert nxt["waypoints"] == [1, 2, 3] and nxt["remaining_waypoints"] == 20
    assert len(nxt["waypoint_poses"]) == 3


@pytest.mark.parametrize("algorithm", ["dijkstra", "astar"])
@pytest.mark.parametrize("use_temporary", [True, False])
def test_planning_system_plans_over_backbone_and_shortcuts(algorithm, use_temporary):
    from cross.core.planning_system import PlanningSystem
    temporary = {3, 7, 15}
    system, _ = _graph(temporary=temporary)
    _topo(system, thresh=0.5)
    ps = PlanningSystem(system, config={"use_temporary_keyframes": use_temporary})   # builds the sparse graph
    assert ps.sparse_graph.permanent_kf_ids == [k for k in range(21) if k not in temporary]
    ps.set_goal(goal_kf_id=20)
    plan = ps.plan_to_goal(algorithm=algorithm)
    assert set(plan["sparse_path"]) <= set(ps.sparse_graph.permanent_kf_ids)      # permanent keyframes only
    path = plan["detailed_path"]
    assert path[0] == 0 and path[-1] == 20
    if use_temporary:                         # expanded along the odometry chain, temporary keyframes included
        assert path == list(range(21))
    else:                                     # expanded along the permanent backbone
        assert path == ps.sparse_graph.permanent_kf_ids
    assert plan["detailed_cost"] == pytest.approx(CORRIDOR, abs=1e-3)
