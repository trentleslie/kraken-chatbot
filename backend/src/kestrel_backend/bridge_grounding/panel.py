"""U8 Tier A — calibration panel fixtures, PRE-REGISTERED thresholds, and the gate evaluator.

This is the build-precondition *tripwire* (falsification only — NOT calibration; Tier B is the
real calibration, deferred behind the temporal-eval port). The thresholds below are committed to
git BEFORE the paid run = pre-registration. The panel chains are injected as synthetic Bridge
fixtures so the gate tests the SCORER, not upstream bridge detection.

Reproducibility note: the Claude Agent SDK exposes no temperature; determinism comes from pinning
the model and snapshotting per-abstract labels to disk (the runner does this). Passing Tier A
licenses "the scorer is not broken/inverted", NOT "calibrated".
"""

from dataclasses import dataclass
from typing import Literal

from ..graph.state import Bridge

Polarity = Literal["positive", "negative", "hard_negative"]


@dataclass(frozen=True)
class TierAThresholds:
    """PRE-REGISTERED Tier A acceptance criteria (commit before running)."""

    margin_min: float = 0.30          # min(positives) - max(negatives) must be >= this
    positive_min: float = 0.60        # every positive's support_fraction must be >= this
    negative_max: float = 0.35        # every negative's support_fraction must be <= this
    off_topic_max: float = 0.50       # per binding leg: off_topic fraction must be <= this
    min_labeled_per_leg: int = 5      # each leg must have >= this many labeled abstracts to count


PRE_REGISTERED = TierAThresholds()


@dataclass(frozen=True)
class ChainSpec:
    """A known A→B→C chain to inject as a synthetic Bridge fixture."""

    name: str
    polarity: Polarity
    entities: tuple[str, str, str]
    entity_names: tuple[str, str, str]
    predicates: tuple[str, str]
    predicate_directions: tuple[bool, bool] = (True, True)


# The panel. Positives are well-established mechanisms; negatives are famously refuted; the hard
# negative has two individually-plausible legs that do NOT compose into a mechanism. Middle-node
# names are the O2 lever — tune from the measured off_topic rate during the run.
PANEL: list[ChainSpec] = [
    ChainSpec(
        "hpv_e7_cervical_cancer", "positive",
        ("NCBITaxon:10566", "PR:000007377", "MONDO:0002974"),
        ("Human papillomavirus", "E7 oncoprotein", "cervical carcinoma"),
        ("biolink:causes", "biolink:contributes_to"),
    ),
    ChainSpec(
        "hpylori_caga_peptic_ulcer", "positive",
        ("NCBITaxon:210", "PR:000022342", "MONDO:0004247"),
        ("Helicobacter pylori", "CagA protein", "peptic ulcer"),
        ("biolink:causes", "biolink:contributes_to"),
    ),
    ChainSpec(
        "mmr_inflammation_autism", "negative",
        ("VO:0000737", "HP:0002037", "MONDO:0005260"),
        ("MMR vaccine", "intestinal inflammation", "autism"),
        ("biolink:related_to", "biolink:related_to"),
    ),
    ChainSpec(
        "hrt_estrogen_chd", "negative",
        ("CHEBI:50113", "GO:0030520", "MONDO:0005010"),
        ("hormone replacement therapy", "estrogen receptor signaling", "coronary heart disease"),
        ("biolink:affects", "biolink:related_to"),
    ),
    ChainSpec(
        "coffee_caffeine_pancreatic_cancer", "hard_negative",
        ("CHEBI:27732", "CHEBI:27732", "MONDO:0005192"),
        ("coffee consumption", "caffeine metabolism", "pancreatic cancer"),
        ("biolink:related_to", "biolink:related_to"),
    ),
]


def build_bridge(spec: ChainSpec) -> Bridge:
    """Inject a ChainSpec as a synthetic 3-node Bridge fixture (tests the scorer, not detection)."""
    return Bridge(
        path_description=f"calibration: {spec.name}",
        entities=list(spec.entities),
        entity_names=list(spec.entity_names),
        predicates=list(spec.predicates),
        predicate_directions=list(spec.predicate_directions),
        tier=2,
        novelty="known",
        significance=f"Tier A calibration chain ({spec.polarity})",
    )


@dataclass
class ChainResult:
    """Scored result for one panel chain (fed to the gate evaluator)."""

    name: str
    polarity: Polarity
    decision: str
    support_fraction: float | None        # None when insufficient
    min_labeled_per_leg: int              # smallest total_labeled across the chain's legs
    max_off_topic_fraction: float         # largest per-leg off_topic fraction


def evaluate_tier_a(
    results: list[ChainResult], thresholds: TierAThresholds = PRE_REGISTERED
) -> dict:
    """Apply the pre-registered Tier A gate. Pure: no I/O, no SDK.

    Returns a verdict dict with ``passed`` (bool or None), per-criterion booleans, and the
    measured values. ``passed is None`` means the panel is NOT EVALUABLE (some chain is
    insufficient or under the labeled-n floor) — fix retrieval, do not rationalize a partial pass.
    """
    t = thresholds
    # Evaluability precondition: every chain must be scored and clear the labeled-n floor.
    not_evaluable = [
        r.name for r in results
        if r.decision == "insufficient_literature"
        or r.support_fraction is None
        or r.min_labeled_per_leg < t.min_labeled_per_leg
    ]
    if not_evaluable or not results:
        return {"passed": None, "reason": "not_evaluable", "not_evaluable": not_evaluable}

    positives = [r for r in results if r.polarity == "positive"]
    negatives = [r for r in results if r.polarity in ("negative", "hard_negative")]
    if not positives or not negatives:
        return {"passed": None, "reason": "panel_missing_polarity"}

    # Evaluability above guarantees support_fraction is non-None; the filter narrows the type.
    pos_fracs: list[float] = [r.support_fraction for r in positives if r.support_fraction is not None]
    neg_fracs: list[float] = [r.support_fraction for r in negatives if r.support_fraction is not None]
    margin = min(pos_fracs) - max(neg_fracs)

    checks = {
        "margin_ok": margin >= t.margin_min,
        "positives_above_floor": all(f >= t.positive_min for f in pos_fracs),
        "negatives_below_ceiling": all(f <= t.negative_max for f in neg_fracs),
        "pools_on_topic": all(r.max_off_topic_fraction <= t.off_topic_max for r in results),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "margin": margin,
        "positive_fractions": {r.name: r.support_fraction for r in positives},
        "negative_fractions": {r.name: r.support_fraction for r in negatives},
        "max_off_topic": {r.name: r.max_off_topic_fraction for r in results},
        "thresholds": t.__dict__,
    }
