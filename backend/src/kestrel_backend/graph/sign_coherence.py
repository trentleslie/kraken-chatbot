"""Class-agnostic sign-coherence detector (Axis D, Guard 2 — CONSUME module-weight-schema).

For any group about to be treated as one coordinated program (e.g. "7 gamma-glutamyl dipeptides as
one GGT program"), read axis A's per-member signed ``kME`` and, if the group **splits in sign**,
partition it into sign-coherent subgroups rather than fusing — the mechanism that otherwise averages
an opposite-sign signal away. The split is decided **purely from the sign** of each member's weight;
no chemistry/structural-class ontology is consulted (L15).

This module is pure and has no dependency on axis A's concrete ``ModuleSpine`` type — the caller
(synthesis) owns the adapter that reads ``ModuleSpine.members{name -> kME}`` and hands this function a
plain ``dict[name -> float | None]``. That isolates the field-name coupling to one place and lets
Guard 2 be developed and tested before axis A lands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class SplitResult:
    """Outcome of a sign-coherence check on one group.

    ``is_split`` — whether the group splits in sign (both signs present beyond ``minority_tol``).
    ``positive`` / ``negative`` — sign-coherent partitions (every member with a defined sign, in
    input order, including members that were below the decision floor).
    ``unassigned`` — members with no usable sign (kME ``None`` or exactly 0.0).
    ``sign_only`` — the group was evaluated in the magnitude-absent graceful-degradation mode.
    """

    is_split: bool
    positive: list[str]
    negative: list[str]
    unassigned: list[str]
    sign_only: bool = False


def detect_sign_split(
    members_kme: Mapping[str, float | None],
    *,
    floor: float = 0.0,
    minority_tol: float = 0.0,
    sign_only: bool = False,
) -> SplitResult:
    """Decide whether ``members_kme`` splits in sign and partition it into sign-coherent subgroups.

    Args:
        members_kme: member name -> signed kME (float in [-1, 1]) or ``None`` when magnitude is
            absent for that row.
        floor: magnitude noise-floor; a member votes in the split decision only when
            ``abs(kME) >= floor``. Ignored when ``sign_only`` is set.
        minority_tol: tolerate a minority opposite-sign fraction; split only when the smaller side's
            share of decision-survivors is strictly greater than this.
        sign_only: graceful-degradation mode (L4) — magnitude absent for the group; the floor is not
            applied and every nonzero member is counted.

    Partition disposition: the floor affects only the split *decision*. ``positive``/``negative``
    assign every member with a defined (nonzero, non-None) sign — including floor-below members —
    while ``None``/zero members are listed as ``unassigned``.
    """
    effective_floor = 0.0 if sign_only else floor

    positive: list[str] = []
    negative: list[str] = []
    unassigned: list[str] = []
    pos_votes = 0
    neg_votes = 0

    for name, kme in members_kme.items():
        if kme is None or kme == 0.0:
            unassigned.append(name)
            continue
        if kme > 0:
            positive.append(name)
        else:
            negative.append(name)
        # Voting is floor-gated; partition membership above is not.
        if abs(kme) >= effective_floor:
            if kme > 0:
                pos_votes += 1
            else:
                neg_votes += 1

    total_votes = pos_votes + neg_votes
    minority = min(pos_votes, neg_votes)
    is_split = (
        pos_votes > 0
        and neg_votes > 0
        and total_votes > 0
        and (minority / total_votes) > minority_tol
    )

    return SplitResult(
        is_split=is_split,
        positive=positive,
        negative=negative,
        unassigned=unassigned,
        sign_only=sign_only,
    )
