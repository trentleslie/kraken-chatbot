"""Unit 1: synthesis emits synthesis_context_stats (context-compression telemetry).

Plan: docs/plans/2026-06-22-004-feat-perf-report-context-insight-plan.md

The stats expose how the synthesis node compresses massive multi-analyte results into a
bounded LLM context: shown/total/elided per capped section + budget utilization. Counts that
depend on the min_members_for_recurrence filter come from a single shared helper
(count_recurrence_qualifying) that is locked to the aggregator's actual rendered output, so the
reported total cannot drift from the real cut.
"""

import pytest

from kestrel_backend.graph.nodes import synthesis
from kestrel_backend.graph.nodes.synthesis import (
    aggregate_shared_diseases,
    assemble_synthesis_context,
    count_recurrence_qualifying,
    _compute_context_stats,
)
from kestrel_backend.graph.pipeline_config import PipelineConfig, SynthesisConfig
from kestrel_backend.graph.state import (
    DiseaseAssociation,
    EntityResolution,
    Finding,
    Hypothesis,
    LiteratureSupport,
    NoveltyScore,
    PathwayMembership,
)


# --- fixture builders (mirror tests/test_synthesis_node.py) ---------------------------


def _finding(entity, tier=1, confidence="high", claim="claimX"):
    return Finding(
        entity=entity, claim=claim, tier=tier, predicate=None, source="direct_kg",
        pmids=[], confidence=confidence, logic_chain=None,
    )


def _entity(curie, name="GeneX", category="biolink:Gene"):
    return EntityResolution(
        raw_name=name, curie=curie, resolved_name=name, category=category,
        confidence=0.9, method="exact",
    )


def _novelty(curie, edge_count, classification="well_characterized"):
    return NoveltyScore(curie=curie, raw_name="x", edge_count=edge_count, classification=classification)


def _disease(entity, disease_curie, disease_name="DiseaseX", evidence_type="curated"):
    return DiseaseAssociation(
        entity_curie=entity, disease_curie=disease_curie, disease_name=disease_name,
        predicate="biolink:associated_with", source="test", pmids=[],
        evidence_type=evidence_type, discovery_preset="default",
    )


def _pathway(entity, pathway_curie, pathway_name="PathwayX"):
    return PathwayMembership(
        entity_curie=entity, pathway_curie=pathway_curie, pathway_name=pathway_name,
        predicate="biolink:participates_in", source="test", discovery_preset="default",
    )


def _hypothesis(title, grounded=False):
    lit = []
    if grounded:
        lit = [LiteratureSupport(
            paper_id="PMID:1", title="T", authors="A et al.", year=2020, relationship="supporting",
        )]
    return Hypothesis(
        title=title, tier=1, claim="c", supporting_entities=["NCBIGene:1"],
        structural_logic="logic", validation_steps=["step"], literature_support=lit,
    )


def _use_cfg(monkeypatch, **kw):
    cfg = PipelineConfig(synthesis=SynthesisConfig(**kw))
    monkeypatch.setattr(synthesis, "get_pipeline_config", lambda: cfg)
    return cfg


# --- count_recurrence_qualifying (the single-source filter) ---------------------------


def test_count_recurrence_qualifying_basic():
    # group A: 2 distinct members; group B: 1 member; group C: 3 members
    pairs = [("A", "x"), ("A", "y"), ("B", "x"), ("C", "x"), ("C", "y"), ("C", "z")]
    assert count_recurrence_qualifying(pairs, min_members=2) == 2   # A and C qualify, B does not


def test_count_recurrence_qualifying_dedupes_members():
    # duplicate (group, member) from parallel-branch reducers counts the member ONCE
    pairs = [("A", "x"), ("A", "x"), ("A", "y")]
    assert count_recurrence_qualifying(pairs, min_members=2) == 1
    assert count_recurrence_qualifying(pairs, min_members=3) == 0


def test_helper_matches_aggregator_rendered_rows():
    """R2: the helper count is locked to what the aggregator actually renders (no drift)."""
    diseases = []
    for i in range(5):  # 5 diseases each shared by 2 members → all qualify at min_members=2
        diseases += [_disease("NCBIGene:1", f"MONDO:{i}"), _disease("NCBIGene:2", f"MONDO:{i}")]
    diseases += [_disease("NCBIGene:3", "MONDO:99")]  # shared by 1 → does NOT qualify
    pairs = [(d.disease_curie, d.entity_curie) for d in diseases]
    total = count_recurrence_qualifying(pairs, min_members=2)
    assert total == 5
    out = aggregate_shared_diseases(diseases, min_members=2, max_items=3)
    rendered_rows = sum(1 for ln in out.splitlines() if ln.startswith("- **"))
    assert rendered_rows == min(total, 3) == 3   # cut matches min(total, max_items)


# --- _compute_context_stats ------------------------------------------------------------


def _module_state():
    """6 distinct entities → module mode; known findings/diseases/pathways/members."""
    entities = [_entity(f"NCBIGene:{i}") for i in range(1, 7)]
    novelty = [_novelty(f"NCBIGene:{i}", edge_count=100 - i) for i in range(1, 7)]
    # 4 diseases shared by ≥2 members (qualify), 1 shared by a single member (excluded)
    diseases = []
    for d in range(4):
        diseases += [_disease("NCBIGene:1", f"MONDO:{d}"), _disease("NCBIGene:2", f"MONDO:{d}")]
    diseases += [_disease("NCBIGene:3", "MONDO:solo")]
    # 3 pathways shared by ≥2
    pathways = []
    for p in range(3):
        pathways += [_pathway("NCBIGene:1", f"R-HSA-{p}"), _pathway("NCBIGene:2", f"R-HSA-{p}")]
    findings = [_finding(f"e{i}", tier=1) for i in range(10)] + [_finding(f"m{i}", tier=2) for i in range(5)]
    hyps = [_hypothesis("h1", grounded=True), _hypothesis("h2", grounded=True), _hypothesis("h3")]
    return {
        "resolved_entities": entities,
        "novelty_scores": novelty,
        "disease_associations": diseases,
        "pathway_memberships": pathways,
        "direct_findings": findings,
        "cold_start_findings": [],
        "hypotheses": hyps,
    }


def test_stats_module_mode_counts(monkeypatch):
    _use_cfg(monkeypatch, module_mode_min_entities=5, min_members_for_recurrence=2,
             max_findings_per_tier=8, max_aggregated_diseases=3, max_aggregated_pathways=2,
             max_member_table_rows=4)
    stats = _compute_context_stats(_module_state(), context="x" * 7000)

    assert stats["module_mode"] is True
    assert stats["distinct_entities"] == 6
    assert stats["module_mode_threshold"] == 5

    # findings: tier1 has 10 (cap 8 → 8 shown, 2 elided), tier2 has 5 (cap 8 → 5 shown, 0 elided)
    assert stats["sections"]["findings"] == {"shown": 13, "total": 15, "elided": 2}
    # diseases: 4 qualify (cap 3 → 3 shown, 1 elided)
    assert stats["sections"]["diseases"] == {"shown": 3, "total": 4, "elided": 1}
    # pathways: 3 qualify (cap 2 → 2 shown, 1 elided)
    assert stats["sections"]["pathways"] == {"shown": 2, "total": 3, "elided": 1}
    # member table: 6 members (cap 4 → 4 shown, 2 elided)
    assert stats["sections"]["member_table"] == {"shown": 4, "total": 6, "elided": 2}
    # literature: 2 of 3 hypotheses carry literature
    assert stats["literature"] == {"attached": 2, "total": 3}

    # budget readout
    assert stats["context_chars"] == 7000
    assert stats["context_est_tokens"] == round(7000 / 3.5)
    assert stats["max_context_chars"] == 350_000  # SynthesisConfig default (not overridden here)
    assert stats["window_tokens"] == 200_000
    assert 0 < stats["char_budget_pct"] <= 100
    assert stats["window_pct"] > 0


def test_stats_per_entity_mode_omits_capped_sections(monkeypatch):
    _use_cfg(monkeypatch, module_mode_min_entities=5, max_findings_per_tier=8)
    state = _module_state()
    state["resolved_entities"] = [_entity("NCBIGene:1"), _entity("NCBIGene:2")]  # 2 < threshold
    stats = _compute_context_stats(state, context="x" * 100)
    assert stats["module_mode"] is False
    assert "findings" in stats["sections"]              # findings always present
    assert "diseases" not in stats["sections"]          # uncapped per-entity → no compression row
    assert "pathways" not in stats["sections"]
    assert "member_table" not in stats["sections"]
    assert stats["literature"]["total"] == 3


def test_stats_findings_elision_aggregate(monkeypatch):
    _use_cfg(monkeypatch, module_mode_min_entities=5, max_findings_per_tier=150)
    state = {"resolved_entities": [_entity(f"NCBIGene:{i}") for i in range(6)],
             "novelty_scores": [], "disease_associations": [], "pathway_memberships": [],
             "direct_findings": [_finding(f"e{i}", tier=1) for i in range(200)],
             "cold_start_findings": [], "hypotheses": []}
    stats = _compute_context_stats(state, context="x")
    assert stats["sections"]["findings"] == {"shown": 150, "total": 200, "elided": 50}


def test_stats_empty_state_no_crash(monkeypatch):
    _use_cfg(monkeypatch, module_mode_min_entities=5)
    stats = _compute_context_stats(
        {"resolved_entities": [], "novelty_scores": [], "disease_associations": [],
         "pathway_memberships": [], "direct_findings": [], "cold_start_findings": [], "hypotheses": []},
        context="",
    )
    assert stats["module_mode"] is False
    assert stats["sections"]["findings"] == {"shown": 0, "total": 0, "elided": 0}
    assert stats["literature"] == {"attached": 0, "total": 0}


# --- assemble_synthesis_context out-param (backward-compatible) -----------------------


def test_assemble_out_param_does_not_change_context_string(monkeypatch):
    _use_cfg(monkeypatch, module_mode_min_entities=5)
    state = _module_state()
    plain = assemble_synthesis_context(state)
    stats = {}
    with_stats = assemble_synthesis_context(state, stats_out=stats)
    assert plain == with_stats          # context text byte-identical with/without the out-param
    assert stats["module_mode"] is True  # and the out-dict was populated


def test_assemble_default_returns_str(monkeypatch):
    _use_cfg(monkeypatch, module_mode_min_entities=5)
    assert isinstance(assemble_synthesis_context(_module_state()), str)


# --- run() emits the field (no-SDK fallback path, no live LLM call) -------------------


@pytest.mark.asyncio
async def test_run_emits_synthesis_context_stats(monkeypatch):
    _use_cfg(monkeypatch, module_mode_min_entities=5)
    monkeypatch.setattr(synthesis, "HAS_SDK", False)  # force deterministic fallback, no LLM
    result = await synthesis.run(_module_state())
    assert "synthesis_context_stats" in result
    assert result["synthesis_context_stats"]["module_mode"] is True


# --- R4: single-writer field is last-write-wins in the WS accumulator -----------------


def test_field_is_not_a_concat_field():
    from kestrel_backend.main import _get_concat_fields
    assert "synthesis_context_stats" not in _get_concat_fields()  # plain dict → last-write-wins
