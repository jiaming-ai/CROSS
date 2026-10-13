"""cross.core.placement: a session placement confirmed from several separate places wins; look-alikes confirmed from one
spot, or two competing confirmed placements, do not."""
import math

import numpy as np

from cross.core.placement import PlacementTracker


def rot_y(a, t=(0, 0, 0)):
    c, s = math.cos(a), math.sin(a)
    T = np.eye(4)
    T[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    T[:3, 3] = t
    return T


def step_forward(tr, d=1.0):
    D = np.eye(4)
    D[2, 3] = d
    tr.odometry(D)


def measure(tr, A, step):
    """A measurement of the current frame consistent with placement A, to a reference 2 m behind it."""
    T_map_cur = A @ tr.T
    T_map_ref = T_map_cur @ np.linalg.inv(rot_y(0.0, (0, 0, 2.0)))
    tr.add(T_map_ref, rot_y(0.0, (0, 0, 2.0)), step)


def make():
    return PlacementTracker(places=3, sep_m=10.0, yaw_tol_deg=10.0, t_tol_m=3.0, up=[0, 1, 0])


def test_true_placement_confirmed_from_separate_places():
    tr = make()
    A_true, A_alias = rot_y(0.3, (5, 0, 7)), rot_y(-2.5, (40, 0, -12))
    step = 0
    for k in range(40):                      # 40 m walk, a true measurement every 2 m
        step_forward(tr)
        step += 1
        if k % 2 == 0:
            measure(tr, A_true, step)
        if 10 <= k < 14:                     # an alias seen from one spot only
            measure(tr, A_alias, step)
    conf = tr.confirmed(step)
    assert conf is not None and tr.find(A_true) is conf and len(conf.places) >= 3
    assert tr.find(A_alias) is not conf


def test_alias_from_one_spot_not_confirmed():
    tr = make()
    for k in range(30):
        step_forward(tr, 0.1)                # 3 m in total: one place
        measure(tr, rot_y(1.0, (3, 0, 3)), k)
    assert tr.confirmed(30) is None and tr.provisional() is not None


def test_two_confirmed_placements_are_ambiguous():
    tr = make()
    A1, A2 = rot_y(0.0, (0, 0, 0)), rot_y(math.pi, (30, 0, 30))
    for k in range(40):
        step_forward(tr)
        measure(tr, A1 if k % 2 else A2, k)
    assert tr.confirmed(40) is None


def test_drifting_placement_split_in_two_tracks_is_not_ambiguous():
    tr = make()
    for k in range(40):                      # the true placement drifts 12 deg / 5 m over the walk
        step_forward(tr)
        f = k / 39.0
        measure(tr, rot_y(math.radians(12) * f, (5 * f, 0, 0)), k)
    assert tr.confirmed(40) is not None


def test_decide_merges_a_dominant_placement_at_once():
    tr = make()
    A = rot_y(0.3, (5, 0, 7))
    for k in range(4):
        step_forward(tr, 0.2)
        measure(tr, A, k)
    ok, why, t = tr.decide(A)
    assert ok and why == "dominant"


def test_decide_waits_while_placements_compete_and_then_follows_the_joined_one():
    tr = make()
    A1, A2 = rot_y(0.0, (0, 0, 0)), rot_y(2.0, (25, 0, -15))
    for k in range(10):                      # two look-alike placements, equally supported
        step_forward(tr, 0.3)
        measure(tr, A1, 2 * k)
        measure(tr, A2, 2 * k + 1)
    assert not tr.decide(A1)[0] and tr.decide(A1)[1] == "competing placements"
    for k in range(10, 60):                  # the walk goes on: only A1 keeps being confirmed, from new places
        step_forward(tr, 1.0)
        measure(tr, A1, 100 + k)
    ok, why, t = tr.decide(A1)
    assert ok
    tr.commit(t)
    assert tr.decide(A2)[1] == "disagrees with the joined placement"
    assert tr.decide(A1)[0]


def test_short_walk_steps_decide_and_one_off_rivals_do_not_block():
    """A 4 m walk (no second place possible): as first built, any rival blocks (places clause); with places_after_m the
    candidate's 3x more supporting steps win; a rival seen at one step is ignored with rival_min_steps=2."""
    A, A_alias = rot_y(0.3, (5, 0, 7)), rot_y(-2.5, (40, 0, -12))

    def walk(tr, alias_steps):
        for k in range(12):
            step_forward(tr, 0.33)
            measure(tr, A, k)
            if k in alias_steps:
                measure(tr, A_alias, k)

    tr = make()
    walk(tr, (3, 4, 5, 6))
    assert tr.decide(A)[1] == "competing placements"
    tr = PlacementTracker(places=3, sep_m=10.0, yaw_tol_deg=10.0, t_tol_m=3.0, up=[0, 1, 0], places_after_m=20.0)
    walk(tr, (3, 4, 5, 6))
    assert tr.decide(A)[0]
    tr = PlacementTracker(places=3, sep_m=10.0, yaw_tol_deg=10.0, t_tol_m=3.0, up=[0, 1, 0], places_after_m=20.0,
                          dominance=4.0)
    walk(tr, (3, 4, 5, 6))
    assert tr.decide(A)[1] == "competing placements"
    tr = PlacementTracker(places=3, sep_m=10.0, yaw_tol_deg=10.0, t_tol_m=3.0, up=[0, 1, 0], places_after_m=20.0,
                          dominance=4.0, rival_min_steps=2)
    walk(tr, (3,))
    assert tr.decide(A)[0]


def test_places_clause_applies_after_a_long_walk():
    """After places_after_m of walk, a candidate seen from one place does not beat a rival with as many places."""
    tr = PlacementTracker(places=3, sep_m=10.0, yaw_tol_deg=10.0, t_tol_m=3.0, up=[0, 1, 0], places_after_m=20.0)
    A, A_alias = rot_y(0.3, (5, 0, 7)), rot_y(-2.5, (40, 0, -12))
    for k in range(30):                      # 30 m: the alias seen at two spots, the candidate many times at one
        step_forward(tr, 1.0)
        if k in (2, 3, 20, 21):
            measure(tr, A_alias, k)
    for k in range(10):
        step_forward(tr, 0.1)
        measure(tr, A, 100 + k)
    assert tr.walked > 20.0
    assert tr.decide(A)[1] == "competing placements"


def test_min_spread_needs_support_from_more_than_one_viewpoint():
    A = rot_y(0.3, (5, 0, 7))
    tr = PlacementTracker(places=3, sep_m=10.0, yaw_tol_deg=10.0, t_tol_m=3.0, up=[0, 1, 0], min_spread_m=1.5)
    for k in range(8):                       # standing still: 8 steps, no spread
        tr.odometry(np.eye(4))
        measure(tr, A, k)
    assert tr.decide(A)[1] == "too little support"
    for k in range(8, 12):                   # then moving 2 m
        step_forward(tr, 0.5)
        measure(tr, A, k)
    assert tr.decide(A)[0]
