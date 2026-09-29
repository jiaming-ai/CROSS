"""Chart isolation, persistence and actual graph joins without neural models."""
from types import SimpleNamespace
import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pp = pytest.importorskip("pypose")

from cross.core.charts import cluster_by_chart, restore_node_charts
from cross.core.config import HypothesisConfig, SystemConfig
from cross.core.hypothesis import HypothesisManager
from cross.core.types import Keyframe, Edge, EdgeType, VisualEdge


def manager(components=3):
    system = SimpleNamespace(device="cpu", topo_map=None, loaded_node_ids=frozenset(), config=SystemConfig())
    hm = HypothesisManager(system, components, HypothesisConfig(chart_aware=True, session_recovery=True))
    hm.dist = (pp.identity_SE3(components), pp.se3(torch.ones(components, 6)*.1),
               torch.tensor([1.] + [0.]*(components-1)))
    hm.start_tracking_chart()
    return hm


def node(node_id, chart, x=0., components=3):
    poses = pp.identity_SE3(components)
    poses[0, 0] = x
    kf = Keyframe(poses, pp.se3(torch.ones(components, 6)*.1),
                  torch.tensor([1.] + [0.]*(components-1)),
                  pose_charts=torch.tensor([chart] + [-1]*(components-1)))
    kf.id = node_id
    return kf


def proposal(chart, x=0., keyframe=10):
    pose = pp.identity_SE3()
    pose[0] = x
    return dict(pose=pose, std=pp.se3(torch.ones(6)*.1), score=torch.tensor(.8),
                chart_id=chart, source_indices=[(keyframe, 0)])


def test_clustering_is_separate_before_representative_selection_and_origin_independent():
    poses = np.array([[0., 0., 0.], [.2, 0., 0.], [0., 0., 0.], [.2, 0., 0.]])
    charts = np.array([4, 4, 9, 9])
    before = cluster_by_chart(poses, charts, 1., 1)
    assert before.tolist() == [0, 0, 1, 1]
    poses[charts == 4, :2] += [100., -80.]
    np.testing.assert_array_equal(cluster_by_chart(poses, charts, 1., 1), before)
    with pytest.raises(ValueError, match="known coordinate chart"):
        cluster_by_chart(poses, [-1, 4, 9, 9], 1., 1)


def test_global_message_preserves_equal_coordinate_modes_in_two_charts():
    from cross.core.system import System
    system = System.__new__(System)
    system.device, system.config = "cpu", SystemConfig()
    system.hypothesis_manager = manager()
    frames = [node(10, 4), node(20, 9)]
    ret = dict(convolved_mus=torch.stack([k.pose_mu for k in frames]),
               convolved_stds=torch.stack([k.pose_std for k in frames]),
               pose_est_conf=torch.tensor([.7,.9]),
               valid_ref_component_weights=torch.stack([k.pose_weights for k in frames]),
               valid_keyframes=frames)
    *_, edges = system._merge_and_align_components(ret)
    assert edges == {10: (0,1), 20: (0,2)}
    hm = system.hypothesis_manager
    assert hm.component_charts.tolist() == [0,4,9]
    assert [a['nearest_prior_distance'] for a in hm.last_alignment_audit] == [None,None]


def test_alignment_requires_same_chart_and_recycled_slot_does_not_keep_identity():
    hm = manager(2)
    _, _, _, _, edges = hm.align_proposal_prior([proposal(8)])
    assert edges == {10: (0,1)}  # coordinate equality must not match active chart 0
    hm.dist[2][:] = .5
    hm.newborn[:] = False
    hm.align_proposal_prior([proposal(8)])
    assert hm.last_alignment_audit[0]['action'] == 'matched'
    generation = hm.component_generations[1]
    hm.log_c_hist[1] = 5.
    hm.align_proposal_prior([proposal(9, keyframe=20)])
    assert hm.component_charts.tolist() == [0,9]
    assert hm.component_generations[1] > generation
    assert hm.log_c_hist[1].count_nonzero() == 0


def test_one_chart_recovers_the_inherited_global_message_and_filter():
    from cross.core.system import System
    frames = [node(10,0,.2),node(20,0,.4)]
    ret = dict(convolved_mus=torch.stack([k.pose_mu for k in frames]),
               convolved_stds=torch.stack([k.pose_std for k in frames]),
               pose_est_conf=torch.tensor([.7,.9]),
               valid_ref_component_weights=torch.stack([k.pose_weights for k in frames]),valid_keyframes=frames)
    results=[]
    for enabled in (False,True):
        system=System.__new__(System)
        system.device,system.config='cpu',SystemConfig()
        system.hypothesis_manager=hm=manager()
        hm.chart_aware=enabled
        mu,std,weights,confidence,edges=system._merge_and_align_components(ret)
        hm.gmm_filtering(mu,std,weights,confidence)
        results.append((mu,std,weights,confidence,edges,*hm.dist))
    for original,augmented in zip(*results):
        if isinstance(original,dict):
            assert original==augmented
        else:
            torch.testing.assert_close(original,augmented,rtol=0,atol=0)


def test_realization_stores_new_branch_pose_and_chart_in_its_first_keyframe():
    hm = manager()
    hm.component_charts[1] = 7
    hm.dist[0][1,0] = 12.
    hm.dist[2][:] = torch.tensor([.1,.9,0.])
    hm.last_sum_pos[1], hm.last_hit_rate[1] = 2., .5
    hm.hist_valid[1, :2], hm.llr_hist[1, :2] = True, 1.  # two recorded supporting frames
    kf = node(20, 0)
    hm.add_node(kf)
    assert kf.pose_charts.tolist() == [0,7,-1]
    assert float(kf.pose_mu[1,0]) == 12.
    assert kf.pose_weights[1] == pytest.approx(.9)


def test_chart_recovery_keeps_temporal_support_after_an_earlier_join():
    hm = manager(2)
    hm.component_charts[1] = 3
    hm.dist[2][:] = .5
    hm.realized[:] = True
    hm.hist_valid[1] = True  # recorded (not empty) evidence frames
    hm.log_c_hist[1] = 2.
    hm.reference_support.start({10,11})
    hm.reference_support.mark_anchored()  # a prior join does not anchor every saved chart
    for i in range(4):
        hm.reference_support.observe(i,{10:(0,1),11:(0,1)})
    assert hm.detect_loop_closure({})['loop_closure']
    assert hm.last_loop_audit['candidates'][0]['distance_m'] is None
    hm.log_c_hist[1] = -5.
    assert not hm.detect_loop_closure({})['loop_closure']


def test_legacy_map_charts_follow_committed_graph_not_atlas_or_proximity():
    nodes = {i: node(i,0) for i in [2,3,8,9]}
    for k in nodes.values():
        k.pose_charts = None
        k.pose_weights[1] = .3  # dropped uncommitted state is not reintroduced
    visual = {(8,9): [VisualEdge(pp.identity_SE3(), pp.se3(torch.ones(6)), from_comp_id=0,to_comp_id=1)]}
    assert restore_node_charts(nodes,{(2,3):None},visual) == 3
    assert [int(k.pose_charts[0]) for k in nodes.values()] == [0,0,1,2]
    assert all(float(k.pose_weights[1]) == 0. for k in nodes.values())
    labels = [k.pose_charts.clone() for k in nodes.values()]
    restore_node_charts(nodes,{(2,3):None},visual)
    for k, saved in zip(nodes.values(),labels):
        torch.testing.assert_close(k.pose_charts,saved)
    with pytest.raises(ValueError, match="unmerged coordinate charts"):
        restore_node_charts(nodes,{(2,8):None},{})


def test_database_and_temporary_node_roundtrip_keep_disconnected_charts(monkeypatch):
    import cross.db.db as db_module
    monkeypatch.setattr(db_module, 'BoQ', lambda **_: SimpleNamespace(
        get_embed_dim=lambda: 1, get_embedding=lambda _: torch.ones(1)))
    db = db_module.KeyframeDatabase(None, device='cpu')
    atlas = db.create_atlas()
    kf = node(1,5)
    saved = db.insert(1,None,None,mu=kf.pose_mu,sigma=kf.pose_std,weights=kf.pose_weights,
                      atlas=atlas,pose_charts=kf.pose_charts,
                      metric_source=dict(source_id="image-a",session_id="session-a"))
    hm = manager()
    hm.nodes = {saved.id:saved,100:node(100,9)}
    hm.nodes[100].temporary = True
    hm.nodes[100].metric_source = dict(source_id="image-b",session_id="session-b")
    db_state, hm_state = copy.deepcopy(db.save_state()), copy.deepcopy(hm.save_state())
    db2 = db_module.KeyframeDatabase(None,device='cpu')
    loaded = db2.load_state(db_state,'cpu')
    hm2 = manager()
    hm2.load_state(hm_state,db2,'cpu','cpu',loaded)
    assert int(hm2.nodes[saved.id].pose_charts[0]) == 5
    assert int(hm2.nodes[100].pose_charts[0]) == 9
    assert hm2.nodes[saved.id].metric_source == saved.metric_source
    assert hm2.nodes[100].metric_source == hm.nodes[100].metric_source
    hm2.start_tracking_chart()
    assert int(hm2.component_charts[0]) == 10


def test_proximity_cannot_join_independent_charts_at_identical_origins():
    from cross.core.simple_topo import SimpleTopo
    hm = manager()
    hm.nodes = {1:node(1,0),2:node(2,1),3:node(3,0)}
    topo = SimpleTopo(SimpleNamespace(hypothesis_manager=hm))
    topo.rebuild_graph()
    assert set(topo.proximity_edges) == {(1,3)}


def graph_example(offset=20.):
    from cross.core.pgo import PoseGraph
    hm = manager(2)
    hm.nodes = {1:node(1,99,-100.,2)}  # unrelated old map, must not anchor the join
    for i in range(10,16):
        hm.nodes[i] = node(i,4,float(i-10),2)
    for i in range(30,36):
        hm.nodes[i] = node(i,8,float(i-30)+offset,2)
    hm.component_charts[:] = torch.tensor([8,4])
    hm.dist[2][:] = .5
    hm.dist[0][0] = hm.nodes[35].pose_mu[0]
    hm.dist[0][1,0] = 5.
    hm.create_hypothesis_branch(1,30)
    for i in range(30,36):
        kf=hm.nodes[i]
        kf.pose_mu[1] = hm.nodes[i-20].pose_mu[0]
        kf.pose_weights[1] = .5
        kf.pose_charts[1] = 4
    for ids in (range(10,16),range(30,36)):
        for a,b in zip(ids,list(ids)[1:]):
            hm.odom_edges[(a,b)] = Edge(hm.nodes[a].pose_mu[0].Inv() @ hm.nodes[b].pose_mu[0],
                                         pp.se3(torch.ones(6)*.1),EdgeType.ODOMETRY)
    for a,b in [(10,30),(15,35)]:
        hm.add_edge(a,b,pp.identity_SE3(),pp.se3(torch.ones(6)*.1),EdgeType.VISUAL,0,1)
    hm.system.last_added_kf_id = 35
    pg = PoseGraph(hm, depth=1000, device='cpu')
    pg.construct_for_loop_closure(35,1)
    return hm,pg


@pytest.mark.parametrize('offset',[0.,20.,-50.])
def test_pgo_join_transports_chart_and_excludes_unrelated_graphs(offset):
    hm,pg = graph_example(offset)
    assert 1 not in pg.vertex_map
    assert pg.preferred_fixed_node == 10
    assert len(pg.vertices) == len(set(v.id for v in pg.vertices))  # sparse node IDs
    pg.solve(set(pg.vertex_map)-{10},{10})
    # A node arriving after the optimization snapshot still needs transport.
    hm.nodes[40] = node(40,8,offset+6.,2)
    result=hm.apply_pgo_result(dict(pose_graph=pg, optimized_poses=pg.optimized_poses,
                                    other_hypothesis_id=1))
    assert result['success']
    assert int(hm.nodes[1].pose_charts[0]) == 99
    assert float(hm.nodes[1].pose_mu[0,0]) == -100.
    for i in range(30,36):
        assert int(hm.nodes[i].pose_charts[0]) == 4
        assert float(hm.nodes[i].pose_mu[0,0]) == pytest.approx(i-30,abs=1e-4)
    assert hm.component_charts.tolist() == [4,-1]
    assert float(hm.nodes[40].pose_mu[0,0]) == pytest.approx(6.,abs=1e-4)
    assert int(hm.nodes[40].pose_charts[0]) == 4
    # Reload does not infer any link to the unrelated chart.
    assert restore_node_charts(hm.nodes,hm.odom_edges,hm.hypotheses[0].visual_edges) == 100


def test_async_result_from_recycled_slot_is_rejected_without_mutation():
    hm,pg = graph_example()
    pg.output_chart = 4
    before = hm.nodes[30].pose_mu.clone()
    hm._reset_component_evidence(1)
    hm.component_charts[1] = 4  # even a new alternative in the same chart is a new identity
    result = hm.apply_pgo_result(dict(pose_graph=pg,optimized_poses={30:pp.identity_SE3()},other_hypothesis_id=1))
    assert not result['success']
    torch.testing.assert_close(hm.nodes[30].pose_mu,before)
