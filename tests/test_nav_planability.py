import numpy as np

from cross_nav.planability import classify_goals, evaluate_goals, summarize_goals


def test_unreachable_and_unknown_are_distinguished():
    gt = np.ones((9, 9), np.uint8)
    gt[:, 4] = 2
    rows, _, _ = classify_goals(gt, (4, 1), [(4, 7), (4, 2)], 1, 0)
    assert [r['feasibility'] for r in rows] == ['unplanable', 'planable']
    gt[4, 4] = 0
    rows, _, _ = classify_goals(gt, (4, 1), [(4, 7)], 1, 0)
    assert rows[0]['feasibility'] == 'unknown'
    gt[4, 4] = 1
    rows, _, _ = classify_goals(gt, (4, 1), [(4, 7)], 1, 0)
    assert rows[0]['feasibility'] == 'planable'


def test_no_snapping_and_no_corner_cutting():
    gt = np.array([[1, 2], [2, 1]], np.uint8)
    rows, _, _ = classify_goals(gt, (0, 0), [(1, 1)], 1, 0)
    assert rows[0]['feasibility'] == 'unplanable'
    rows, _, _ = classify_goals(gt, (0, 1), [(0, 0)], 1, 0)
    assert rows[0]['feasibility'] == 'unplanable'
    assert rows[0]['start'] == (0, 1)


def test_unknown_shortcut_cannot_inflate_success():
    gt = np.ones((11, 11), np.uint8)
    gt[4:7, 4:7] = 0
    rows = evaluate_goals(gt, {'blind': np.zeros_like(gt), 'known': gt != 1},
                          (5, 1), [(5, 9)], 1, 0)
    assert rows[0]['feasibility'] == 'planable'
    assert summarize_goals(rows, 'blind')['outcomes']['unknown_path'] == 1
    assert summarize_goals(rows, 'blind')['success_rate'] == 0
    assert summarize_goals(rows, 'known')['success_rate'] == 1


def test_footprint_and_empty_denominator():
    gt = np.ones((11, 11), np.uint8)
    gt[:, 5] = 2
    gt[5, 5] = 1  # point can pass, radius-one robot cannot
    rows = evaluate_goals(gt, {'gt': gt == 2}, (5, 2), [(5, 8)], 1, 1)
    out = summarize_goals(rows, 'gt')
    assert out['unplanable'] == 1 and out['success_rate'] is None
    gt[4:7, 5] = 1
    rows = evaluate_goals(gt, {'gt': gt == 2}, (5, 2), [(5, 8)], 1, 1)
    assert summarize_goals(rows, 'gt')['success_rate'] == 1
