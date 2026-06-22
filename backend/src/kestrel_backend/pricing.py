"""Per-model token pricing for the performance report's cost estimate.

Generalizes the hardcoded Sonnet-4.5 computation in ``agent.py:340-348``
(``TurnMetrics.cost_usd``) into a per-model rate map. Rates are USD per 1M
tokens. Cost is an **estimate** — the table is maintained by hand and can drift
from actual billing; ``LAST_VERIFIED`` records when it was last checked against
the published pricing, and unknown models return ``None`` (logged) rather than a
guessed number.

Source: the ``claude-api`` skill reference (Anthropic published pricing).
Cache-read ≈ 0.1× input; cache-write (5-minute TTL) ≈ 1.25× input.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Last time the rates below were checked against published Anthropic pricing.
LAST_VERIFIED = "2026-06-22"


@dataclass(frozen=True)
class Rate:
    """USD per 1M tokens for one model family."""

    input: float
    output: float
    cache_read: float
    cache_creation: float


# Keyed by a canonical family token (see _family_of). Rates per 1M tokens.
_RATES: dict[str, Rate] = {
    # Opus 4.5–4.8: $5 / $25 in/out.
    "opus-4-8": Rate(5.0, 25.0, 0.50, 6.25),
    "opus-4-7": Rate(5.0, 25.0, 0.50, 6.25),
    "opus-4-6": Rate(5.0, 25.0, 0.50, 6.25),
    "opus-4-5": Rate(5.0, 25.0, 0.50, 6.25),
    # Opus 4.0 / 4.1 (legacy): $15 / $75 in/out.
    "opus-4-1": Rate(15.0, 75.0, 1.50, 18.75),
    "opus-4-0": Rate(15.0, 75.0, 1.50, 18.75),
    # Sonnet 4.x (incl. claude-sonnet-4-20250514, the pipeline default): $3 / $15.
    "sonnet-4": Rate(3.0, 15.0, 0.30, 3.75),
    # Haiku 4.5: $1 / $5.
    "haiku-4-5": Rate(1.0, 5.0, 0.10, 1.25),
    # Fable 5 / Mythos 5: $10 / $50.
    "fable-5": Rate(10.0, 50.0, 1.00, 12.50),
    "mythos-5": Rate(10.0, 50.0, 1.00, 12.50),
}


def _family_of(model_name: str) -> str | None:
    """Map a free-form model id to a canonical rate-map family token.

    ``ModelUsageRecord.model_name`` is free-form (e.g. ``anthropic/claude-sonnet-4-20250514``,
    ``claude-opus-4-8``). Normalize by lowercasing and matching the most specific
    family key contained in the id. Returns ``None`` if no family matches.
    """
    if not model_name:
        return None
    name = model_name.lower()
    # Most specific keys first so "opus-4-8" wins over a hypothetical "opus-4".
    for family in sorted(_RATES, key=len, reverse=True):
        if family in name:
            return family
    return None


def rate_for(model_name: str) -> Rate | None:
    """Return the Rate for a model id, or None (with a warning) if unknown."""
    family = _family_of(model_name)
    if family is None:
        logger.warning("pricing: no rate for model %r — cost will be null", model_name)
        return None
    return _RATES[family]


def cost_of_record(
    *,
    model_name: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> float | None:
    """Estimated USD cost for one model-usage record, or None if the model is unknown.

    Records are per-call deltas (each ``ModelUsageRecord`` is one API call), so summing
    them is correct — there is no cumulative counter to double-count.
    """
    rate = rate_for(model_name)
    if rate is None:
        return None
    return (
        input_tokens * rate.input
        + output_tokens * rate.output
        + cache_read_tokens * rate.cache_read
        + cache_creation_tokens * rate.cache_creation
    ) / 1_000_000


def estimate_cost(records) -> float:
    """Total estimated USD across model-usage records.

    Each record is a ``ModelUsageRecord`` (or any object exposing the same token
    attributes). Records with an unknown model contribute 0 to the total (the
    unknown is logged by ``rate_for``); an empty list returns 0.0.
    """
    total = 0.0
    for r in records:
        cost = cost_of_record(
            model_name=getattr(r, "model_name", ""),
            input_tokens=getattr(r, "input_tokens", 0),
            output_tokens=getattr(r, "output_tokens", 0),
            cache_read_tokens=getattr(r, "cache_read_tokens", 0),
            cache_creation_tokens=getattr(r, "cache_creation_tokens", 0),
        )
        if cost is not None:
            total += cost
    return total
