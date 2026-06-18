"""U3 — per-leg labeling prompt + strict schema.

Run with: uv run python -m pytest tests/test_bridge_grounding_prompts.py -v
"""

from src.kestrel_backend.bridge_grounding.prompts import (
    LEG_LABELS,
    build_leg_prompt,
    clean_predicate,
    leg_label_schema,
    relation_phrase,
)

ABSTRACTS = [
    {"pmid": "111", "title": "HPV and cervix", "abstract": "HPV infection causes cervical cancer."},
    {"pmid": "222", "title": "Other", "abstract": "Unrelated review of cardiology."},
]


# --- schema ---------------------------------------------------------------------------

def test_schema_strict_and_required():
    s = leg_label_schema()
    assert s["additionalProperties"] is False
    item = s["properties"]["labels"]["items"]
    assert item["additionalProperties"] is False
    assert item["required"] == ["pmid", "evidence", "label"]
    assert item["properties"]["label"]["enum"] == list(LEG_LABELS)
    assert len(LEG_LABELS) == 4


def test_schema_property_order_evidence_before_label():
    # Forced CoT: the model must emit `evidence` before committing to `label`.
    keys = list(leg_label_schema()["properties"]["labels"]["items"]["properties"].keys())
    assert keys.index("evidence") < keys.index("label")


# --- clean_predicate / relation_phrase ------------------------------------------------

def test_clean_predicate():
    assert clean_predicate("biolink:gene_associated_with_condition") == "gene associated with condition"
    assert clean_predicate("biolink:affects") == "affects"
    assert clean_predicate(None) == ""
    assert clean_predicate("") == ""


def test_relation_phrase_forward():
    assert relation_phrase("HPV", "cervical cancer", "biolink:causes", True) == \
        '"HPV" causes "cervical cancer"'


def test_relation_phrase_reverse_swaps_subject_object():
    # Edge stored reversed: the asserted KG relation runs Y -> X.
    assert relation_phrase("INS", "type 2 diabetes", "biolink:gene_associated_with_condition", False) == \
        '"type 2 diabetes" gene associated with condition "INS"'


def test_relation_phrase_no_predicate_is_direction_agnostic():
    assert relation_phrase("A", "B", None, None) == 'an association between "A" and "B"'


# --- build_leg_prompt -----------------------------------------------------------------

def test_prompt_contains_names_relation_labels_and_pmids():
    p = build_leg_prompt("HPV", "cervical cancer", "biolink:causes", True, ABSTRACTS)
    assert "HPV" in p and "cervical cancer" in p
    assert '"HPV" causes "cervical cancer"' in p
    for label in LEG_LABELS:
        assert label in p
    assert "PMID:111" in p and "PMID:222" in p
    assert "Cite ONLY the PMIDs provided" in p


def test_prompt_includes_all_five_error_guards():
    p = build_leg_prompt("HPV", "cervical cancer", "biolink:causes", True, ABSTRACTS).lower()
    assert "co-mention" in p                 # topical co-mention guard
    assert "direction matters" in p          # directionality guard
    assert "belief" in p or "survey" in p    # opinion guard
    assert "epidemiology" in p               # association-only guard
    assert "negation" in p                   # negation/factuality guard


def test_prompt_evidence_before_label_instruction():
    p = build_leg_prompt("A", "B", "biolink:affects", True, ABSTRACTS)
    assert "evidence" in p.lower() and "before the `label`" in p.lower()


def test_prompt_directed_has_opposite_direction_rule():
    p = build_leg_prompt("A", "B", "biolink:causes", True, ABSTRACTS)
    assert "OPPOSITE direction" in p


def test_prompt_direction_agnostic_drops_opposite_rule():
    # When orientation is unknown (O3 fallback), no directional refute rule.
    p = build_leg_prompt("A", "B", None, None, ABSTRACTS, direction_known=False)
    assert "OPPOSITE direction" not in p
    assert 'an association between "A" and "B"' in p
