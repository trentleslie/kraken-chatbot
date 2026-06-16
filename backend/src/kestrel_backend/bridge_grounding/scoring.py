"""U4 — v1 chain scoring: ratio + counts (Beta-Binomial CI deferred to v2 / U9).

Per leg: ``support_fraction = supports / total_labeled`` where ``total_labeled`` = supports +
refutes + neither (off_topic is EXCLUDED — it is not a parameter here; callers must not include
it). A neither-dominated leg therefore reads low (the denominator is inflated), which is intended.

Chain headline = the WEAKER leg's support_fraction (min-leg gating); the stronger leg is retained
as a secondary ranking key so a downstream ranker is not forced to sort on the lossy min scalar.
A binding leg below ``binding_leg_floor`` usable+relevant abstracts yields insufficient_literature
rather than a number. Nothing here raises.
"""

from dataclasses import dataclass

# v1 production binding-leg floor (BridgeGroundingConfig.binding_leg_floor_k default).
DEFAULT_BINDING_LEG_FLOOR = 2
# Headline at/above this is "grounded", else "ungrounded" (insufficient is separate).
DEFAULT_GROUNDED_THRESHOLD = 0.5

DECISION_GROUNDED = "grounded"
DECISION_UNGROUNDED = "ungrounded"
DECISION_INSUFFICIENT = "insufficient_literature"


@dataclass(frozen=True)
class LegScore:
    total_labeled: int
    support_fraction: float | None  # None when total_labeled == 0


@dataclass(frozen=True)
class ChainScore:
    support_fraction: float            # headline (weaker leg); 0.0 when insufficient
    strong_leg_fraction: float | None  # stronger leg's fraction (secondary ranking key)
    strong_leg_n: int | None           # total_labeled behind strong_leg_fraction
    decision: str
    weak_leg_index: int | None         # which leg set the headline (None when insufficient)


def score_leg(support: int, refute: int, neither: int) -> LegScore:
    """One leg's support_fraction over its labeled abstracts (off_topic already excluded)."""
    total = support + refute + neither
    if total <= 0:
        return LegScore(total_labeled=0, support_fraction=None)
    return LegScore(total_labeled=total, support_fraction=support / total)


def score_chain(
    legs: list[tuple[int, int, int]],
    *,
    binding_leg_floor: int = DEFAULT_BINDING_LEG_FLOOR,
    grounded_threshold: float = DEFAULT_GROUNDED_THRESHOLD,
) -> ChainScore:
    """Score a chain from its legs' (support, refute, neither) counts (off_topic excluded).

    Every binding leg must clear ``binding_leg_floor`` labeled abstracts AND have a defined
    fraction, else the chain is insufficient_literature. Otherwise the headline is the weaker
    leg's fraction; the stronger leg is reported separately.
    """
    leg_scores = [score_leg(s, r, n) for (s, r, n) in legs]
    usable = bool(leg_scores) and all(
        ls.support_fraction is not None and ls.total_labeled >= binding_leg_floor
        for ls in leg_scores
    )
    if not usable:
        return ChainScore(0.0, None, None, DECISION_INSUFFICIENT, None)

    # All non-None here (usable guard); the `or 0.0` is unreachable but keeps the type float.
    fracs: list[float] = [ls.support_fraction or 0.0 for ls in leg_scores]
    weak_idx = min(range(len(fracs)), key=lambda i: fracs[i])  # first-min on ties
    headline = fracs[weak_idx]
    # Stronger leg = the max fraction (ties → the other index); for a 1-leg chain, no strong leg.
    strong_idx = max(range(len(fracs)), key=lambda i: fracs[i])
    if len(leg_scores) > 1 and strong_idx != weak_idx:
        strong_fraction: float | None = fracs[strong_idx]
        strong_n: int | None = leg_scores[strong_idx].total_labeled
    else:
        strong_fraction, strong_n = None, None

    decision = DECISION_GROUNDED if headline >= grounded_threshold else DECISION_UNGROUNDED
    return ChainScore(headline, strong_fraction, strong_n, decision, weak_idx)
