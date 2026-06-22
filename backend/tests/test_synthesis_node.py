"""Module-aware synthesis context: cross-entity aggregation, per-member table, findings
cap, module-vs-per-entity assembly switch, and visible SDK-fallback degradation.

Plan: docs/plans/2026-06-22-001-feat-module-aware-synthesis-context-plan.md
"""

import logging

from kestrel_backend.graph.nodes import synthesis
from kestrel_backend.graph.nodes.synthesis import (
    aggregate_shared_diseases,
    aggregate_shared_pathways,
    format_findings_summary,
    format_member_table,
)
from kestrel_backend.graph.pipeline_config import PipelineConfig, SynthesisConfig
from kestrel_backend.graph.state import (
    DiseaseAssociation,
    EntityResolution,
    Finding,
    NoveltyScore,
    PathwayMembership,
)


def _finding(entity, tier=1, confidence="high", claim="claimX"):
    return Finding(
        entity=entity,
        claim=claim,
        tier=tier,
        predicate=None,
        source="direct_kg",
        pmids=[],
        confidence=confidence,
        logic_chain=None,
    )


def _use_cfg(monkeypatch, **synthesis_kwargs):
    cfg = PipelineConfig(synthesis=SynthesisConfig(**synthesis_kwargs))
    monkeypatch.setattr(synthesis, "get_pipeline_config", lambda: cfg)
    return cfg


def _entity(curie, name="GeneX", category="biolink:Gene"):
    return EntityResolution(
        raw_name=name,
        curie=curie,
        resolved_name=name,
        category=category,
        confidence=0.9,
        method="exact",
    )


def _novelty(curie, edge_count, classification="well_characterized"):
    return NoveltyScore(
        curie=curie,
        raw_name="x",
        edge_count=edge_count,
        classification=classification,
    )


def _disease(entity, disease_curie, disease_name="DiseaseX", evidence_type="curated"):
    return DiseaseAssociation(
        entity_curie=entity,
        disease_curie=disease_curie,
        disease_name=disease_name,
        predicate="biolink:associated_with",
        source="test",
        pmids=[],
        evidence_type=evidence_type,
        discovery_preset="default",
    )


def _pathway(entity, pathway_curie, pathway_name="PathwayX"):
    return PathwayMembership(
        entity_curie=entity,
        pathway_curie=pathway_curie,
        pathway_name=pathway_name,
        predicate="biolink:participates_in",
        source="test",
        discovery_preset="default",
    )


# --- Unit 2: aggregate_shared_diseases -------------------------------------------------


def test_shared_disease_lists_all_members():
    """A disease shared by 3 members is rendered with member count 3 and every member."""
    diseases = [
        _disease("NCBIGene:1", "MONDO:9", "Cancer"),
        _disease("NCBIGene:2", "MONDO:9", "Cancer"),
        _disease("NCBIGene:3", "MONDO:9", "Cancer"),
    ]
    out = aggregate_shared_diseases(diseases, min_members=2, max_items=30)
    assert "Cancer" in out
    assert "MONDO:9" in out
    assert "3" in out  # member count
    for curie in ("NCBIGene:1", "NCBIGene:2", "NCBIGene:3"):
        assert curie in out


def test_shared_disease_dedupes_duplicate_member():
    """An additive-reducer duplicate (same entity+disease twice) counts the member ONCE."""
    diseases = [
        _disease("NCBIGene:1", "MONDO:9", "Cancer"),
        _disease("NCBIGene:1", "MONDO:9", "Cancer"),  # duplicate from a parallel branch
        _disease("NCBIGene:2", "MONDO:9", "Cancer"),
    ]
    out = aggregate_shared_diseases(diseases, min_members=2, max_items=30)
    # 2 distinct members, not 3.
    assert "2" in out
    assert "3" not in out.split("Cancer")[1].split("\n")[0]


def test_shared_disease_dedupe_keeps_strongest_evidence():
    """Duplicate (entity, disease) differing in evidence_type keeps the STRONGER one."""
    diseases = [
        _disease("NCBIGene:1", "MONDO:9", "Cancer", evidence_type="predicted"),
        _disease("NCBIGene:1", "MONDO:9", "Cancer", evidence_type="gwas"),  # stronger
        _disease("NCBIGene:2", "MONDO:9", "Cancer", evidence_type="predicted"),
    ]
    out = aggregate_shared_diseases(diseases, min_members=2, max_items=30)
    assert "gwas" in out  # strongest evidence surfaced, not silently dropped
    # member counted once despite two rows for NCBIGene:1
    assert "2" in out


def test_disease_below_threshold_excluded():
    """A disease in only 1 member is excluded when min_members=2."""
    diseases = [
        _disease("NCBIGene:1", "MONDO:solo", "Lonely"),
        _disease("NCBIGene:1", "MONDO:shared", "Shared"),
        _disease("NCBIGene:2", "MONDO:shared", "Shared"),
    ]
    out = aggregate_shared_diseases(diseases, min_members=2, max_items=30)
    assert "Shared" in out
    assert "Lonely" not in out


def test_disease_cap_keeps_highest_member_count():
    """With max_items=N and N+ qualifying diseases, exactly N render, highest member-count kept."""
    diseases = []
    # disease A: 4 members; B: 3; C: 2; D: 2
    for e in ("g1", "g2", "g3", "g4"):
        diseases.append(_disease(e, "MONDO:A", "AAA"))
    for e in ("g1", "g2", "g3"):
        diseases.append(_disease(e, "MONDO:B", "BBB"))
    for e in ("g1", "g2"):
        diseases.append(_disease(e, "MONDO:C", "CCC"))
    out = aggregate_shared_diseases(diseases, min_members=2, max_items=2)
    assert "AAA" in out and "BBB" in out  # top 2 by member count
    assert "CCC" not in out


def test_disease_empty_and_single_entity_return_empty():
    """Empty input and single-entity input (nothing shared by >=2) both render nothing."""
    assert aggregate_shared_diseases([], min_members=2, max_items=30) == ""
    single = [
        _disease("NCBIGene:1", "MONDO:9", "Cancer"),
        _disease("NCBIGene:1", "MONDO:8", "Other"),
    ]
    assert aggregate_shared_diseases(single, min_members=2, max_items=30) == ""


# --- Unit 2: aggregate_shared_pathways -------------------------------------------------


def test_shared_pathway_distinct_member_count():
    """Pathway aggregation counts distinct members, dedupes, respects threshold."""
    pathways = [
        _pathway("NCBIGene:1", "GO:1", "Inflammation"),
        _pathway("NCBIGene:1", "GO:1", "Inflammation"),  # duplicate
        _pathway("NCBIGene:2", "GO:1", "Inflammation"),
        _pathway("NCBIGene:1", "GO:solo", "Solo"),  # 1 member only
    ]
    out = aggregate_shared_pathways(pathways, min_members=2, max_items=30)
    assert "Inflammation" in out
    assert "GO:1" in out
    assert "2" in out
    assert "Solo" not in out


# --- Unit 3: format_member_table -------------------------------------------------------


def test_member_table_rows_sorted_by_edge_count_desc():
    """One row per member, sorted by edge_count desc, showing bucket + edge_count."""
    entities = [_entity("g1", "Alpha"), _entity("g2", "Beta"), _entity("g3", "Gamma")]
    novelty = [
        _novelty("g1", 50, "moderate"),
        _novelty("g2", 900, "well_characterized"),
        _novelty("g3", 5, "sparse"),
    ]
    out = format_member_table(entities, novelty, [], max_rows=50)
    # rows present with bucket + edge_count
    assert "Beta" in out and "900" in out and "well_characterized" in out
    assert "Alpha" in out and "Gamma" in out
    # sorted desc: Beta (900) before Alpha (50) before Gamma (5)
    assert out.index("Beta") < out.index("Alpha") < out.index("Gamma")


def test_member_table_no_disease_shows_dash():
    """A member with no disease associations renders '—' for top disease."""
    entities = [_entity("g1", "Alpha")]
    novelty = [_novelty("g1", 50)]
    out = format_member_table(entities, novelty, [], max_rows=50)
    assert "Alpha" in out
    assert "—" in out


def test_member_table_top_disease_is_strongest_evidence():
    """'Top disease' is the entity's strongest-evidence association."""
    entities = [_entity("g1", "Alpha")]
    novelty = [_novelty("g1", 50)]
    diseases = [
        _disease("g1", "MONDO:weak", "WeakLink", evidence_type="predicted"),
        _disease("g1", "MONDO:strong", "StrongLink", evidence_type="gwas"),
    ]
    out = format_member_table(entities, novelty, diseases, max_rows=50)
    assert "StrongLink" in out
    assert "WeakLink" not in out


def test_member_table_graceful_join_both_directions():
    """A novelty curie with no resolved entity (and vice versa) renders without KeyError."""
    entities = [_entity("g1", "Alpha")]  # has no novelty
    novelty = [_novelty("g2", 100)]  # has no resolved entity
    out = format_member_table(entities, novelty, [], max_rows=50)
    assert "Alpha" in out  # resolved-only row
    assert "g2" in out  # novelty-only row


def test_member_table_empty_returns_empty():
    assert format_member_table([], [], [], max_rows=50) == ""


def test_member_table_caps_rows():
    """With max_rows=N and N+ members, exactly N rows (highest edge_count) + an elision line."""
    entities = [_entity(f"g{i}", f"E{i}") for i in range(7)]
    novelty = [_novelty(f"g{i}", i * 10) for i in range(7)]
    out = format_member_table(entities, novelty, [], max_rows=3)
    # top 3 by edge_count: g6(60), g5(50), g4(40)
    assert "E6" in out and "E5" in out and "E4" in out
    assert "E0" not in out and "E1" not in out
    assert "more members" in out


# --- Unit 4: findings cap + module-aware assembly + backstop ---------------------------


def test_findings_cap_per_tier_keeps_highest_confidence():
    """Cap per tier, ranked by confidence high->low; the rest are elided."""
    findings = [
        _finding("E_hi1", tier=1, confidence="high"),
        _finding("E_hi2", tier=1, confidence="high"),
        _finding("E_lo1", tier=1, confidence="low"),
        _finding("E_lo2", tier=1, confidence="low"),
        _finding("E_lo3", tier=1, confidence="low"),
    ]
    out = format_findings_summary(findings, [], max_per_tier=2)
    assert "E_hi1" in out and "E_hi2" in out
    assert "E_lo1" not in out
    assert "more (tier 1)" in out  # elision line names the remainder


def test_findings_unbounded_when_no_cap():
    """Backward-compatible: no cap renders every finding (existing callers unaffected)."""
    findings = [_finding(f"E{i}", tier=1, confidence="low") for i in range(40)]
    out = format_findings_summary(findings, [], max_per_tier=None)
    assert "E39" in out
    assert "more (tier 1)" not in out


def test_module_assembly_uses_aggregation_not_per_entity(monkeypatch):
    """At module scale, assembly emits aggregation + member table, NOT per-entity dumps."""
    _use_cfg(monkeypatch, module_mode_min_entities=3, min_members_for_recurrence=2)
    state = {
        "resolved_entities": [_entity("g1"), _entity("g2"), _entity("g3")],
        "disease_associations": [_disease("g1", "MONDO:9", "Cancer"), _disease("g2", "MONDO:9", "Cancer")],
        "pathway_memberships": [_pathway("g1", "GO:1"), _pathway("g2", "GO:1")],
        "novelty_scores": [_novelty("g1", 100), _novelty("g2", 200), _novelty("g3", 5)],
    }
    out = synthesis.assemble_synthesis_context(state)
    assert "Module-Level Disease Recurrence" in out
    assert "Member Prioritization Table" in out
    assert "## Disease Associations" not in out  # per-entity header omitted
    assert "## Pathway & Biological Process Memberships" not in out


def test_single_entity_keeps_per_entity_sections(monkeypatch):
    """A single-entity query keeps the per-entity report shape (R5)."""
    _use_cfg(monkeypatch, module_mode_min_entities=5)
    state = {
        "resolved_entities": [_entity("g1")],
        "disease_associations": [_disease("g1", "MONDO:9", "Cancer")],
        "pathway_memberships": [_pathway("g1", "GO:1")],
        "novelty_scores": [_novelty("g1", 100)],
    }
    out = synthesis.assemble_synthesis_context(state)
    assert "## Disease Associations" in out
    assert "Module-Level Disease Recurrence" not in out


def test_two_entity_boundary_respects_threshold(monkeypatch):
    """A 2-entity (pair) query keeps per-entity shape unless module_mode_min_entities<=2 (R5)."""
    state = {
        "resolved_entities": [_entity("g1"), _entity("g2")],
        "disease_associations": [_disease("g1", "MONDO:9", "Cancer"), _disease("g2", "MONDO:9", "Cancer")],
        "pathway_memberships": [],
        "novelty_scores": [_novelty("g1", 100), _novelty("g2", 200)],
    }
    _use_cfg(monkeypatch, module_mode_min_entities=5)
    assert "## Disease Associations" in synthesis.assemble_synthesis_context(state)
    _use_cfg(monkeypatch, module_mode_min_entities=2, min_members_for_recurrence=2)
    assert "Module-Level Disease Recurrence" in synthesis.assemble_synthesis_context(state)


def test_module_context_stays_under_token_budget(monkeypatch):
    """Core R1 guard: 50 well-char entities x inflated data stay under char + token budget."""
    cfg = _use_cfg(monkeypatch, module_mode_min_entities=5)
    entities = [_entity(f"g{i}", f"E{i}") for i in range(50)]
    novelty = [_novelty(f"g{i}", 500) for i in range(50)]
    diseases = [_disease(f"g{i}", f"MONDO:{j}", f"Disease{j}") for i in range(50) for j in range(40)]
    pathways = [_pathway(f"g{i}", f"GO:{j}", f"Pathway{j}") for i in range(50) for j in range(40)]
    findings = [_finding(f"E{i}-{k}", tier=((k % 3) + 1)) for i in range(50) for k in range(40)]
    state = {
        "resolved_entities": entities,
        "novelty_scores": novelty,
        "disease_associations": diseases,
        "pathway_memberships": pathways,
        "direct_findings": findings,
        "cold_start_findings": [],
    }
    out = synthesis.assemble_synthesis_context(state)
    assert len(out) < cfg.synthesis.max_context_chars
    assert len(out) / 3.5 < 100_000  # estimated tokens under budget


def test_context_backstop_logs_warning(monkeypatch, caplog):
    """If assembled context exceeds max_context_chars, a WARNING (with token estimate) is logged."""
    _use_cfg(monkeypatch, module_mode_min_entities=5, max_context_chars=50)
    state = {
        "resolved_entities": [_entity("g1")],
        "disease_associations": [_disease("g1", "MONDO:9", "Cancer")],
        "novelty_scores": [_novelty("g1", 100)],
    }
    with caplog.at_level(logging.WARNING):
        synthesis.assemble_synthesis_context(state)
    msgs = " ".join(r.getMessage().lower() for r in caplog.records)
    assert "token" in msgs or "max_context_chars" in msgs or "exceeds" in msgs


# NOTE: the run()-level visible-fallback degradation marker (formerly this file's "Unit 5") is
# now covered by tests/test_synthesis_fallback_marker.py, which shipped on dev with the
# performance-report feature (PR #84). That implementation also marks the empty-output case, so the
# duplicate tests here were dropped on merge rather than maintained in two places.
