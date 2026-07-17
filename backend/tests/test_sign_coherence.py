"""Unit 4 — class-agnostic sign-coherence detector (pure, CONSUME module-weight-schema).

``detect_sign_split`` decides whether a group about to be treated as one coordinated program
actually splits in sign (per-member signed kME). No chemistry/structural-class ontology is used —
the split is decided purely from the sign of each member's weight (L15). When a split is detected the
group is partitioned into sign-coherent subgroups so synthesis never fuses a mixed-sign set.

Design decisions (resolving axis-A-dependent unknowns at implementation time):
  * ``members_kme`` maps member name -> signed kME (float) or ``None`` (magnitude absent for that row).
  * The magnitude ``floor`` affects only the split *decision* (near-zero members are excluded from
    the vote); partition membership assigns EVERY member with a defined sign to its subgroup.
  * ``minority_tol`` tolerates a small opposite-sign minority: split only when the smaller side's
    fraction of decision-survivors is strictly greater than the tolerance.
  * ``sign_only=True`` is the graceful-degradation path (magnitude absent for the group): the floor
    is ignored and every nonzero member is counted.
"""

from kestrel_backend.graph.sign_coherence import SplitResult, detect_sign_split


class TestDetectSignSplit:
    def test_all_positive_no_split(self):
        r = detect_sign_split({"a": 0.8, "b": 0.7, "c": 0.9})
        assert isinstance(r, SplitResult)
        assert r.is_split is False
        assert r.positive == ["a", "b", "c"]
        assert r.negative == []

    def test_clean_mix_splits_into_two_subgroups(self):
        r = detect_sign_split({"a": 0.8, "b": -0.7, "c": 0.6, "d": -0.9})
        assert r.is_split is True
        assert r.positive == ["a", "c"]
        assert r.negative == ["b", "d"]
        assert r.unassigned == []

    def test_near_zero_member_excluded_by_floor(self):
        # c is opposite-sign but below the floor → excluded from the split decision → no split.
        r = detect_sign_split({"a": 0.8, "b": 0.7, "c": -0.02}, floor=0.1)
        assert r.is_split is False
        # partition still assigns c (defined sign) to its subgroup, even though it didn't vote.
        assert r.negative == ["c"]

    def test_lone_minority_within_tolerance_no_split(self):
        # 1 of 4 survivors is minority-sign → fraction 0.25, not > tol 0.25 → no split.
        r = detect_sign_split(
            {"a": 0.8, "b": 0.7, "c": 0.9, "d": -0.8}, floor=0.1, minority_tol=0.25
        )
        assert r.is_split is False

    def test_minority_above_tolerance_splits(self):
        r = detect_sign_split(
            {"a": 0.8, "b": 0.7, "c": 0.9, "d": -0.8}, floor=0.1, minority_tol=0.2
        )
        assert r.is_split is True

    def test_none_magnitude_member_is_unassigned(self):
        r = detect_sign_split({"a": 0.8, "b": -0.7, "c": None})
        assert r.is_split is True
        assert r.unassigned == ["c"]
        assert "c" not in r.positive and "c" not in r.negative

    def test_zero_member_is_unassigned(self):
        r = detect_sign_split({"a": 0.8, "b": -0.7, "c": 0.0})
        assert r.unassigned == ["c"]

    def test_sign_only_fallback_ignores_floor(self):
        # sign-only: magnitudes are unit sentinels; floor would normally exclude nothing here, but
        # the flag guarantees no floor is applied and every nonzero member is counted.
        r = detect_sign_split(
            {"a": 1.0, "b": -1.0, "c": 1.0}, floor=0.5, minority_tol=0.0, sign_only=True
        )
        assert r.is_split is True
        assert r.sign_only is True
        assert r.positive == ["a", "c"]
        assert r.negative == ["b"]

    def test_empty_group_no_split(self):
        r = detect_sign_split({})
        assert r.is_split is False
        assert r.positive == [] and r.negative == [] and r.unassigned == []

    def test_single_member_no_split(self):
        r = detect_sign_split({"a": 0.9})
        assert r.is_split is False
