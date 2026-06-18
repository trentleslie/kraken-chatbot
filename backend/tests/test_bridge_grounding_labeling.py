"""U8 harness — LLM label parsing + tally (pure; the SDK call itself is not unit-tested).

Run with: uv run python -m pytest tests/test_bridge_grounding_labeling.py -v
"""

from src.kestrel_backend.bridge_grounding.labeling import (
    json_output_instruction,
    parse_label_response,
    tally_labels,
)


def _resp(*labels):
    items = ", ".join(
        f'{{"pmid": "{p}", "evidence": "{e}", "label": "{lbl}"}}' for p, e, lbl in labels)
    return f'{{"labels": [{items}]}}'


def test_parse_plain_json():
    out = parse_label_response(_resp(("1", "q", "supports_leg"), ("2", "none", "off_topic")))
    assert out == [
        {"pmid": "1", "evidence": "q", "label": "supports_leg"},
        {"pmid": "2", "evidence": "none", "label": "off_topic"},
    ]


def test_parse_strips_markdown_fence():
    text = "```json\n" + _resp(("9", "x", "refutes_leg")) + "\n```"
    assert parse_label_response(text)[0]["label"] == "refutes_leg"


def test_parse_tolerates_surrounding_prose():
    text = "Here are the labels:\n" + _resp(("3", "y", "neither_or_inconclusive")) + "\nDone."
    assert parse_label_response(text) == [
        {"pmid": "3", "evidence": "y", "label": "neither_or_inconclusive"}]


def test_parse_drops_invalid_label_and_bad_pmid():
    text = ('{"labels": ['
            '{"pmid": "1", "evidence": "e", "label": "totally_made_up"}, '
            '{"pmid": 5, "evidence": "e", "label": "supports_leg"}, '
            '{"pmid": "ok", "evidence": "e", "label": "supports_leg"}]}')
    assert parse_label_response(text) == [{"pmid": "ok", "evidence": "e", "label": "supports_leg"}]


def test_parse_unparseable_returns_empty():
    assert parse_label_response("not json at all") == []
    assert parse_label_response("") == []
    assert parse_label_response('{"labels": "nope"}') == []


def test_tally_counts_buckets():
    labels = [
        {"pmid": "1", "label": "supports_leg"},
        {"pmid": "2", "label": "supports_leg"},
        {"pmid": "3", "label": "refutes_leg"},
        {"pmid": "4", "label": "neither_or_inconclusive"},
        {"pmid": "5", "label": "off_topic"},
        {"pmid": "6", "label": "off_topic"},
    ]
    assert tally_labels(labels) == {"support": 2, "refute": 1, "neither": 1, "off_topic": 2}


def test_tally_empty():
    assert tally_labels([]) == {"support": 0, "refute": 0, "neither": 0, "off_topic": 0}


def test_json_instruction_lists_labels():
    instr = json_output_instruction()
    assert '"labels"' in instr and "supports_leg" in instr and "off_topic" in instr
