"""Unit 2: per-model pricing module."""

from dataclasses import dataclass

from kestrel_backend import pricing


@dataclass
class FakeUsage:
    model_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


def test_known_model_hand_computed_cost():
    # Sonnet: 1M input + 1M output = $3 + $15 = $18.
    cost = pricing.cost_of_record(
        model_name="claude-sonnet-4-6", input_tokens=1_000_000, output_tokens=1_000_000
    )
    assert cost == 18.0


def test_cache_tokens_included():
    # Sonnet cache: 1M cache_read ($0.30) + 1M cache_creation ($3.75) = $4.05.
    cost = pricing.cost_of_record(
        model_name="claude-sonnet-4-6",
        cache_read_tokens=1_000_000,
        cache_creation_tokens=1_000_000,
    )
    assert round(cost, 6) == 4.05


def test_freeform_model_name_normalizes():
    # The pipeline's actual model_name shape.
    assert pricing.rate_for("anthropic/claude-sonnet-4-20250514") == pricing.rate_for(
        "claude-sonnet-4-6"
    )


def test_opus_distinct_from_sonnet():
    opus = pricing.cost_of_record(model_name="claude-opus-4-8", input_tokens=1_000_000)
    sonnet = pricing.cost_of_record(model_name="claude-sonnet-4-6", input_tokens=1_000_000)
    assert opus == 5.0
    assert sonnet == 3.0


def test_reproduces_legacy_sonnet_number():
    # Mirrors agent.py:340-348 TurnMetrics.cost_usd for a Sonnet-only call.
    cost = pricing.cost_of_record(
        model_name="claude-sonnet-4-20250514",
        input_tokens=10_000,
        output_tokens=2_000,
        cache_read_tokens=5_000,
        cache_creation_tokens=1_000,
    )
    expected = (10_000 * 3.0 + 2_000 * 15.0 + 5_000 * 0.30 + 1_000 * 3.75) / 1_000_000
    assert round(cost, 10) == round(expected, 10)


def test_unknown_model_returns_none(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        cost = pricing.cost_of_record(model_name="gpt-4o", input_tokens=1_000)
    assert cost is None
    assert any("no rate for model" in rec.message for rec in caplog.records)


def test_unknown_model_does_not_raise():
    # estimate_cost must tolerate unknown models (contribute 0), never raise.
    records = [
        FakeUsage("claude-sonnet-4-6", input_tokens=1_000_000),  # $3
        FakeUsage("some-unknown-model", input_tokens=1_000_000),  # 0
    ]
    assert pricing.estimate_cost(records) == 3.0


def test_empty_records_is_zero_not_none():
    assert pricing.estimate_cost([]) == 0.0


def test_estimate_sums_per_call_deltas():
    # Two separate calls sum (records are per-call deltas, safe to sum).
    records = [
        FakeUsage("claude-sonnet-4-6", input_tokens=1_000_000),
        FakeUsage("claude-sonnet-4-6", output_tokens=1_000_000),
    ]
    assert pricing.estimate_cost(records) == 18.0


def test_last_verified_present():
    assert pricing.LAST_VERIFIED  # documented "as-of" date for drift tracking
