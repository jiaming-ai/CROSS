from cross.core.reference_support import ReferenceSupport


def test_reference_support_requires_temporal_and_reference_redundancy():
    support = ReferenceSupport(3, 8, .5, enabled=True)
    support.start({10, 11})
    for i in range(3):
        support.observe(i, {10: (0, 1), 11: (0, 1)})
    assert not support.audit(1)["eligible"]
    support.observe(2, {10: (0, 1), 11: (0, 1)})  # repeated delivery adds nothing
    assert support.audit(1)["supported_frames"] == 3
    support.observe(3, {10: (0, 1), 11: (0, 1)})
    assert support.audit(1)["eligible"]
    support.mark_anchored()
    assert not support.audit(1)["unanchored_reference_candidate"]


def test_single_alias_self_support_reused_slots_and_stale_evidence_are_rejected():
    support = ReferenceSupport(3, 8, .5, enabled=True)
    support.start({10, 11})
    for i in range(8):
        support.observe(i, {10: (0, 1), 99: (1, 1), 11: (1, 1)})
    assert not support.audit(1)["eligible"]  # only one direct historical pose
    assert support.audit(1)["reference_ids"] == [10]
    for i in range(8, 12):
        support.observe(i, {10: (0, 1), 11: (0, 1)})
    assert support.audit(1)["eligible"]
    support.observe(12, {11: (0, 1)}, newborns=[1])
    assert support.audit(1)["supported_frames"] == 1
    assert not support.audit(1)["eligible"]
    for i in range(13, 21):
        support.observe(i, {})
    assert support.audit(1)["supported_frames"] == 0
    assert support.audit(1)["unanchored_reference_candidate"]  # cannot revert to permissive distance gate


def test_legacy_default_and_empty_map_do_not_enable_recovery():
    for enabled, ids in [(False, {10, 11}), (True, set())]:
        support = ReferenceSupport(3, 8, .5, enabled=enabled)
        support.start(ids)
        for i in range(8):
            support.observe(i, {10: (0, 1), 11: (0, 1)})
        assert not support.audit(1)["unanchored_reference_candidate"]
