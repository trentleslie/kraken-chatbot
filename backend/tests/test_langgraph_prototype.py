"""
Tests for the LangGraph workflow implementation.

Phase 1 tests: Intake, entity resolution, synthesis nodes
Phase 2 tests: Triage, conditional routing, parallel branch state merge
Phase 3 tests: Direct KG and Cold-Start real implementations
Phase 4a tests: Pathway enrichment node with shared neighbor analysis
Phase 4b tests: Integration (bridges + gaps) and temporal (conditional) nodes
Phase 5 tests: Synthesis hypothesis generation engine

Run with: uv run pytest tests/test_langgraph_prototype.py -v
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from src.kestrel_backend.graph.builder import (
    build_discovery_graph, route_after_triage, route_after_integration
)
from src.kestrel_backend.graph.state import (
    DiscoveryState, EntityResolution, NoveltyScore, Finding,
    DiseaseAssociation, PathwayMembership, InferredAssociation, AnalogueEntity,
    SharedNeighbor, BiologicalTheme, Bridge, GapEntity, TemporalClassification, Hypothesis
)
from src.kestrel_backend.graph.nodes import (
    intake, entity_resolution, triage, direct_kg, cold_start,
    pathway_enrichment, integration, temporal, synthesis
)


# =============================================================================
# Phase 1 Tests: Intake Node
# =============================================================================

class TestIntakeNode:
    """Tests for the intake node's parsing and detection logic."""

    @pytest.mark.asyncio
    async def test_discovery_mode_with_trigger_phrase(self):
        """Discovery triggers should be detected."""
        state: DiscoveryState = {
            "raw_query": "Analyze the relationship between glucose and insulin resistance"
        }
        result = await intake.run(state)
        assert result["query_type"] == "discovery"

    @pytest.mark.asyncio
    async def test_discovery_mode_with_multiple_entities(self):
        """Multiple entities should trigger discovery mode."""
        state: DiscoveryState = {
            "raw_query": "What do glucose, fructose, and mannose have in common?"
        }
        result = await intake.run(state)
        assert result["query_type"] == "discovery"
        assert len(result["raw_entities"]) >= 3

    @pytest.mark.asyncio
    async def test_retrieval_mode_simple_query(self):
        """Simple questions should be retrieval mode."""
        state: DiscoveryState = {
            "raw_query": "What is glucose?"
        }
        result = await intake.run(state)
        assert result["query_type"] == "retrieval"

    @pytest.mark.asyncio
    async def test_entity_extraction_comma_list(self):
        """Comma-separated entities should be extracted."""
        state: DiscoveryState = {
            "raw_query": "Analyze these metabolites: glucose, fructose, mannose"
        }
        result = await intake.run(state)
        entities_lower = [e.lower() for e in result["raw_entities"]]
        assert "glucose" in entities_lower or any("gluc" in e for e in entities_lower)

    @pytest.mark.asyncio
    async def test_longitudinal_detection(self):
        """Longitudinal study keywords should be detected."""
        state: DiscoveryState = {
            "raw_query": "Analyze the 5-year longitudinal OGTT study data for converters"
        }
        result = await intake.run(state)
        assert result["is_longitudinal"] is True
        assert result["duration_years"] == 5


# =============================================================================
# Phase 1 Tests: Entity Resolution Node
# =============================================================================

class TestEntityResolutionNode:
    """Tests for the entity resolution node with mocked SDK."""

    @pytest.mark.asyncio
    async def test_empty_entities(self):
        """Empty entity list should return empty results."""
        state: DiscoveryState = {
            "raw_entities": []
        }
        result = await entity_resolution.run(state)
        assert result["resolved_entities"] == []
        assert result["errors"] == []

    @pytest.mark.asyncio
    async def test_resolution_result_parsing(self):
        """Test JSON parsing from LLM response."""
        json_response = '{"curie": "CHEBI:17234", "name": "D-glucose", "category": "biolink:ChemicalEntity", "confidence": 0.95}'
        result = entity_resolution.parse_resolution_result("glucose", json_response)
        assert result.curie == "CHEBI:17234"
        assert result.resolved_name == "D-glucose"
        assert result.method == "exact"

    @pytest.mark.asyncio
    async def test_resolution_failure_handling(self):
        """Failed resolution should return method='failed'."""
        json_response = '{"curie": null, "name": null, "category": null, "confidence": 0.0}'
        result = entity_resolution.parse_resolution_result("unknownentity", json_response)
        assert result.method == "failed"
        assert result.curie is None

    @pytest.mark.asyncio
    async def test_batch_chunking(self):
        """Test that entities are chunked correctly."""
        items = list(range(15))
        chunks = entity_resolution.chunk(items, 6)
        assert len(chunks) == 3
        assert len(chunks[0]) == 6
        assert len(chunks[1]) == 6
        assert len(chunks[2]) == 3


# =============================================================================
# Phase 1 Tests: Synthesis Node
# =============================================================================

class TestSynthesisNode:
    """Tests for the synthesis node's report generation."""

    @pytest.mark.asyncio
    async def test_empty_entities_report(self):
        """Report should handle empty entity list."""
        with patch.object(synthesis, 'HAS_SDK', False):
            state: DiscoveryState = {
                "raw_query": "test query",
                "query_type": "discovery",
                "resolved_entities": [],
            }
            result = await synthesis.run(state)
            assert "No entities were provided" in result["synthesis_report"]

    @pytest.mark.asyncio
    async def test_successful_resolution_report(self):
        """Report should include resolved entity details."""
        with patch.object(synthesis, 'HAS_SDK', False):
            state: DiscoveryState = {
                "raw_query": "Analyze glucose",
                "query_type": "discovery",
                "resolved_entities": [
                    EntityResolution(
                        raw_name="glucose",
                        curie="CHEBI:17234",
                        resolved_name="D-glucose",
                        category="biolink:ChemicalEntity",
                        confidence=0.95,
                        method="exact",
                    )
                ],
            }
            result = await synthesis.run(state)
            report = result["synthesis_report"]
            assert "CHEBI:17234" in report
            assert "D-glucose" in report
            assert "1/1" in report  # success count

    @pytest.mark.asyncio
    async def test_report_includes_novelty_scores(self):
        """Report should include novelty classification."""
        with patch.object(synthesis, 'HAS_SDK', False):
            state: DiscoveryState = {
                "raw_query": "Analyze glucose",
                "query_type": "discovery",
                "resolved_entities": [
                    EntityResolution(
                        raw_name="glucose",
                        curie="CHEBI:17234",
                        resolved_name="D-glucose",
                        category="biolink:ChemicalEntity",
                        confidence=0.95,
                        method="exact",
                    )
                ],
                "novelty_scores": [
                    NoveltyScore(
                        curie="CHEBI:17234",
                        raw_name="glucose",
                        edge_count=500,
                        classification="well_characterized",
                    )
                ],
            }
            result = await synthesis.run(state)
            report = result["synthesis_report"]
            assert "Well-Characterized" in report
            assert "500 edges" in report

    @pytest.mark.asyncio
    async def test_report_includes_findings(self):
        """Report should include findings from analysis branches."""
        with patch.object(synthesis, 'HAS_SDK', False):
            state: DiscoveryState = {
                "raw_query": "Analyze glucose and novelty",
                "query_type": "discovery",
                "resolved_entities": [],
                "direct_findings": [
                    Finding(entity="CHEBI:17234", claim="Test finding from direct KG", tier=1, source="direct_kg")
                ],
                "cold_start_findings": [
                    Finding(entity="unknown", claim="Test finding from cold start", tier=3, source="cold_start")
                ],
            }
            result = await synthesis.run(state)
            report = result["synthesis_report"]
            assert "Tier 1" in report
class TestTriageNode:
    """Tests for the triage node's novelty scoring and classification."""

    @pytest.mark.asyncio
    async def test_classify_by_edge_count(self):
        """Test classification thresholds."""
        assert triage.classify_by_edge_count(500) == "well_characterized"
        assert triage.classify_by_edge_count(200) == "well_characterized"
        assert triage.classify_by_edge_count(100) == "moderate"
        assert triage.classify_by_edge_count(20) == "moderate"
        assert triage.classify_by_edge_count(10) == "sparse"
        assert triage.classify_by_edge_count(1) == "sparse"
        assert triage.classify_by_edge_count(0) == "cold_start"

    @pytest.mark.asyncio
    async def test_parse_edge_count_result(self):
        """Test JSON parsing from edge count response."""
        json_response = '{"curie": "CHEBI:17234", "edge_count": 250}'
        result = triage.parse_edge_count_result("CHEBI:17234", "glucose", json_response)
        assert result.curie == "CHEBI:17234"
        assert result.edge_count == 250
        assert result.classification == "well_characterized"

    @pytest.mark.asyncio
    async def test_empty_resolved_entities(self):
        """Empty resolved entities should return empty scores."""
        state: DiscoveryState = {"resolved_entities": []}
        result = await triage.run(state)
        assert result["novelty_scores"] == []
        assert result["well_characterized_curies"] == []

    @pytest.mark.asyncio
    async def test_failed_entities_go_to_cold_start(self):
        """Failed resolutions should be routed to cold_start."""
        state: DiscoveryState = {
            "resolved_entities": [
                EntityResolution(
                    raw_name="unknown_entity",
                    curie=None,
                    method="failed",
                    confidence=0.0,
                )
            ]
        }
        result = await triage.run(state)
        assert "unknown_entity" in result["cold_start_curies"]

    @pytest.mark.asyncio
    async def test_triage_with_mocked_sdk(self):
        """Test full triage with mocked edge counting."""
        mock_score = NoveltyScore(
            curie="CHEBI:17234",
            raw_name="glucose",
            edge_count=300,
            classification="well_characterized",
        )

        with patch.object(triage, 'count_edges_single', return_value=mock_score):
            state: DiscoveryState = {
                "resolved_entities": [
                    EntityResolution(
                        raw_name="glucose",
                        curie="CHEBI:17234",
                        resolved_name="D-glucose",
                        category="biolink:ChemicalEntity",
                        confidence=0.95,
                        method="exact",
                    )
                ]
            }
            result = await triage.run(state)
            assert len(result["novelty_scores"]) == 1
            assert "CHEBI:17234" in result["well_characterized_curies"]


# =============================================================================
# Phase 2 Tests: Routing Logic
# =============================================================================

class TestRoutingLogic:
    """Tests for the conditional routing function."""

    def test_route_both_branches(self):
        """Mixed entities should route to both branches."""
        state: DiscoveryState = {
            "well_characterized_curies": ["CHEBI:17234"],
            "moderate_curies": [],
            "sparse_curies": ["GENE:12345"],
            "cold_start_curies": [],
        }
        result = route_after_triage(state)
        assert isinstance(result, list)
        assert "direct_kg" in result
        assert "cold_start" in result

    def test_route_direct_only(self):
        """Only well-characterized should route to direct_kg."""
        state: DiscoveryState = {
            "well_characterized_curies": ["CHEBI:17234"],
            "moderate_curies": ["CHEBI:56789"],
            "sparse_curies": [],
            "cold_start_curies": [],
        }
        result = route_after_triage(state)
        assert result == "direct_kg"

    def test_route_cold_start_only(self):
        """Only sparse should route to cold_start."""
        state: DiscoveryState = {
            "well_characterized_curies": [],
            "moderate_curies": [],
            "sparse_curies": ["GENE:12345"],
            "cold_start_curies": ["unknown"],
        }
        result = route_after_triage(state)
        assert result == "cold_start"

    def test_route_to_pathway_enrichment_when_empty(self):
        """No entities should skip to pathway_enrichment (which handles empty gracefully)."""
        state: DiscoveryState = {
            "well_characterized_curies": [],
            "moderate_curies": [],
            "sparse_curies": [],
            "cold_start_curies": [],
        }
        result = route_after_triage(state)
        assert result == "pathway_enrichment"


# =============================================================================
# Phase 3 Tests: Direct KG Node
# =============================================================================

class TestDirectKGNode:
    """Tests for the direct_kg node's real implementation."""

    @pytest.mark.asyncio
    async def test_empty_entities(self):
        """Empty entity list should return empty results."""
        state: DiscoveryState = {
            "well_characterized_curies": [],
            "moderate_curies": [],
        }
        result = await direct_kg.run(state)
        assert result["direct_findings"] == []
        assert result["disease_associations"] == []
        assert result["pathway_memberships"] == []

    @pytest.mark.asyncio
    async def test_parse_direct_kg_result_with_diseases(self):
        """Test parsing disease associations from JSON response."""
        json_response = '''```json
{
  "diseases": [
    {"curie": "MONDO:0005148", "name": "Type 2 Diabetes", "predicate": "biolink:gene_associated_with_condition", "source": "GWAS Catalog", "pmids": ["PMID:12345"], "is_hub": false}
  ],
  "pathways": [],
  "interactions": [],
  "hub_flags": []
}
```'''
        diseases, pathways, findings, hub_flags = direct_kg.parse_direct_kg_result(
            "CHEBI:17234", "glucose", json_response
        )
        assert len(diseases) == 1
        assert diseases[0].disease_name == "Type 2 Diabetes"
        assert diseases[0].evidence_type == "gwas"
        assert len(findings) == 1
        assert findings[0].tier == 1

    @pytest.mark.asyncio
    async def test_parse_direct_kg_result_with_pathways(self):
        """Test parsing pathway memberships from JSON response."""
        json_response = '''{
  "diseases": [],
  "pathways": [
    {"curie": "GO:0006094", "name": "Gluconeogenesis", "predicate": "biolink:participates_in", "source": "Reactome"}
  ],
  "interactions": [],
  "hub_flags": []
}'''
        diseases, pathways, findings, hub_flags = direct_kg.parse_direct_kg_result(
            "CHEBI:17234", "glucose", json_response
        )
        assert len(pathways) == 1
        assert pathways[0].pathway_name == "Gluconeogenesis"
        assert len(findings) == 1
        assert "participates in" in findings[0].claim

    @pytest.mark.asyncio
    async def test_parse_direct_kg_result_with_hub_flags(self):
        """Test hub bias detection."""
        json_response = '''{
  "diseases": [
    {"curie": "MONDO:0005148", "name": "Diabetes", "predicate": "related_to", "source": "unknown", "is_hub": true}
  ],
  "pathways": [],
  "interactions": [],
  "hub_flags": ["MONDO:0005148", "GO:0008150"]
}'''
        diseases, pathways, findings, hub_flags = direct_kg.parse_direct_kg_result(
            "CHEBI:17234", "glucose", json_response
        )
        assert "MONDO:0005148" in hub_flags
        assert "GO:0008150" in hub_flags

    @pytest.mark.asyncio
    async def test_parse_direct_kg_result_fallback_on_invalid_json(self):
        """Test graceful fallback on invalid JSON."""
        invalid_response = "This is not JSON at all, just some text"
        diseases, pathways, findings, hub_flags = direct_kg.parse_direct_kg_result(
            "CHEBI:17234", "glucose", invalid_response
        )
        assert diseases == []
        assert pathways == []
        assert findings == []

    @pytest.mark.asyncio
    async def test_direct_kg_with_sdk_unavailable(self):
        """Test graceful handling when SDK is not available."""
        with patch.object(direct_kg, 'HAS_SDK', False):
            state: DiscoveryState = {
                "well_characterized_curies": ["CHEBI:17234"],
                "moderate_curies": [],
                "novelty_scores": [NoveltyScore(curie="CHEBI:17234", raw_name="glucose", edge_count=300, classification="well_characterized")],
            }
            result = await direct_kg.run(state)
            assert len(result["direct_findings"]) == 1
            assert "SDK unavailable" in result["direct_findings"][0].claim


# =============================================================================
# Phase 3 Tests: Cold-Start Node
# =============================================================================

class TestColdStartNode:
    """Tests for the cold_start node's real implementation."""

    @pytest.mark.asyncio
    async def test_empty_entities(self):
        """Empty entity list should return empty results."""
        state: DiscoveryState = {
            "sparse_curies": [],
            "cold_start_curies": [],
        }
        result = await cold_start.run(state)
        assert result["cold_start_findings"] == []
        assert result["inferred_associations"] == []
        assert result["analogues_found"] == []

    @pytest.mark.asyncio
    async def test_parse_cold_start_result_with_analogues(self):
        """Test parsing analogues from JSON response."""
        json_response = '''```json
{
  "analogues": [
    {"curie": "CHEBI:17234", "name": "D-glucose", "similarity": 0.85, "category": "biolink:ChemicalEntity"}
  ],
  "inferences": []
}
```'''
        analogues, inferences, findings = cold_start.parse_cold_start_result(
            "CHEBI:99999", "unknown_metabolite", 0, json_response
        )
        assert len(analogues) == 1
        assert analogues[0].name == "D-glucose"
        assert analogues[0].similarity == 0.85
        # Should generate a finding about the analogues
        assert len(findings) >= 1

    @pytest.mark.asyncio
    async def test_parse_cold_start_result_with_inferences(self):
        """Test parsing inferred associations from JSON response."""
        json_response = '''{
  "analogues": [
    {"curie": "CHEBI:17234", "name": "D-glucose", "similarity": 0.85}
  ],
  "inferences": [
    {
      "target_curie": "MONDO:0005148",
      "target_name": "Type 2 Diabetes",
      "predicate": "biolink:may_be_associated_with",
      "logic_chain": "X is similar to glucose (0.85). Glucose is associated with T2D. X may also be associated.",
      "supporting_analogues": 3,
      "confidence": "low",
      "validation_step": "Test X in diabetes cell model"
    }
  ]
}'''
        analogues, inferences, findings = cold_start.parse_cold_start_result(
            "CHEBI:99999", "unknown_metabolite", 0, json_response
        )
        assert len(inferences) == 1
        assert inferences[0].target_name == "Type 2 Diabetes"
        assert inferences[0].supporting_analogues == 3
        assert "glucose" in inferences[0].logic_chain.lower()
        assert len(findings) >= 1
        assert findings[0].tier == 3

    @pytest.mark.asyncio
    async def test_cold_start_findings_are_tier_3(self):
        """All cold-start findings must be Tier 3."""
        json_response = '''{
  "analogues": [],
  "inferences": [
    {
      "target_curie": "MONDO:0005148",
      "target_name": "Disease X",
      "predicate": "related_to",
      "logic_chain": "Inferred via similarity",
      "supporting_analogues": 1,
      "confidence": "low",
      "validation_step": "Check experimentally"
    }
  ]
}'''
        analogues, inferences, findings = cold_start.parse_cold_start_result(
            "CHEBI:99999", "unknown", 0, json_response
        )
        for f in findings:
            assert f.tier == 3, "Cold-start findings must be Tier 3"

    @pytest.mark.asyncio
    async def test_cold_start_includes_logic_chains(self):
        """Cold-start findings should include logic chains."""
        json_response = '''{
  "analogues": [],
  "inferences": [
    {
      "target_curie": "MONDO:0005148",
      "target_name": "Disease X",
      "predicate": "related_to",
      "logic_chain": "X similar to Y, Y connected to Z, therefore X may connect to Z",
      "supporting_analogues": 2,
      "confidence": "low",
      "validation_step": "Test experimentally"
    }
  ]
}'''
        analogues, inferences, findings = cold_start.parse_cold_start_result(
            "CHEBI:99999", "unknown", 0, json_response
        )
        assert len(findings) >= 1
        assert findings[0].logic_chain is not None
        assert "similar" in findings[0].logic_chain.lower()

    @pytest.mark.asyncio
    async def test_cold_start_with_sdk_unavailable(self):
        """Test graceful handling when SDK is not available."""
        with patch.object(cold_start, 'HAS_SDK', False):
            state: DiscoveryState = {
                "sparse_curies": ["GENE:99999"],
                "cold_start_curies": [],
                "novelty_scores": [NoveltyScore(curie="GENE:99999", raw_name="novel_gene", edge_count=5, classification="sparse")],
            }
            result = await cold_start.run(state)
            assert len(result["cold_start_findings"]) == 1
            assert "SDK unavailable" in result["cold_start_findings"][0].claim


# =============================================================================
# Phase 3 Tests: Synthesis with Structured Data
# =============================================================================

class TestSynthesisPhase3:
    """Tests for synthesis node with Phase 3 structured data."""

    @pytest.mark.asyncio
    async def test_report_includes_disease_associations(self):
        """Report should format disease associations with evidence types."""
        with patch.object(synthesis, 'HAS_SDK', False):
            state: DiscoveryState = {
                "raw_query": "Analyze glucose",
                "query_type": "discovery",
                "resolved_entities": [],
                "disease_associations": [
                    DiseaseAssociation(
                        entity_curie="CHEBI:17234",
                        disease_curie="MONDO:0005148",
                        disease_name="Type 2 Diabetes",
                        predicate="biolink:gene_associated_with_condition",
                        source="GWAS Catalog",
                        pmids=["PMID:12345"],
                        evidence_type="gwas",
                    )
                ],
            }
            result = await synthesis.run(state)
            report = result["synthesis_report"]
            assert "Disease Associations" in report
            assert "Type 2 Diabetes" in report
            assert "GWAS" in report

    @pytest.mark.asyncio
    async def test_report_includes_pathway_memberships(self):
        """Report should format pathway memberships."""
        with patch.object(synthesis, 'HAS_SDK', False):
            state: DiscoveryState = {
                "raw_query": "Analyze glucose",
                "query_type": "discovery",
                "resolved_entities": [],
                "pathway_memberships": [
                    PathwayMembership(
                        entity_curie="CHEBI:17234",
                        pathway_curie="GO:0006094",
                        pathway_name="Gluconeogenesis",
                        predicate="biolink:participates_in",
                        source="Reactome",
                    )
                ],
            }
            result = await synthesis.run(state)
            report = result["synthesis_report"]
            assert "Pathway" in report or "Biological Process" in report
            assert "Gluconeogenesis" in report

    @pytest.mark.asyncio
    async def test_report_includes_inferred_associations(self):
        """Report should format inferred associations with logic chains."""
        with patch.object(synthesis, 'HAS_SDK', False):
            state: DiscoveryState = {
                "raw_query": "Analyze unknown metabolite",
                "query_type": "discovery",
                "resolved_entities": [],
                "inferred_associations": [
                    InferredAssociation(
                        source_entity="CHEBI:99999",
                        target_curie="MONDO:0005148",
                        target_name="Type 2 Diabetes",
                        predicate="biolink:may_be_associated_with",
                        logic_chain="X similar to glucose, glucose associated with T2D",
                        supporting_analogues=3,
                        confidence="low",
                        validation_step="Test in cell model",
                    )
                ],
                "analogues_found": [
                    AnalogueEntity(curie="CHEBI:17234", name="D-glucose", similarity=0.85, category="biolink:ChemicalEntity")
                ],
            }
            result = await synthesis.run(state)
            report = result["synthesis_report"]
            assert "Inferred" in report
            assert "Tier 3" in report or "Speculative" in report
            assert "logic" in report.lower() or "Logic" in report

    @pytest.mark.asyncio
    async def test_report_includes_hub_warnings(self):
        """Report should include hub bias warnings."""
        with patch.object(synthesis, 'HAS_SDK', False):
            state: DiscoveryState = {
                "raw_query": "Analyze glucose",
                "query_type": "discovery",
                "resolved_entities": [],
                "hub_flags": ["GO:0008150", "MONDO:0000001"],
            }
            result = await synthesis.run(state)
            report = result["synthesis_report"]
            assert "Hub" in report or "hub" in report
            assert "GO:0008150" in report


class TestParallelBranches:
    """Tests for parallel branch state merging."""

    @pytest.mark.asyncio
    async def test_parallel_state_merge(self):
        """Critical: verify operator.add merges findings from parallel branches."""
        # Create mock resolutions and scores for mixed routing
        mock_resolution_well = EntityResolution(
            raw_name="glucose",
            curie="CHEBI:17234",
            resolved_name="D-glucose",
            category="biolink:ChemicalEntity",
            confidence=0.95,
            method="exact",
        )
        mock_resolution_sparse = EntityResolution(
            raw_name="novelty",
            curie="GENE:99999",
            resolved_name="Novel Gene",
            category="biolink:Gene",
            confidence=0.8,
            method="fuzzy",
        )

        # Mock edge counting to return well-characterized and sparse
        async def mock_count_edges(entity):
            if entity.curie == "CHEBI:17234":
                return NoveltyScore(
                    curie="CHEBI:17234",
                    raw_name="glucose",
                    edge_count=500,
                    classification="well_characterized",
                )
            else:
                return NoveltyScore(
                    curie="GENE:99999",
                    raw_name="novelty",
                    edge_count=5,
                    classification="sparse",
                )

        with patch.object(entity_resolution, 'resolve_single_entity') as mock_resolve:
            # Return different resolutions based on input
            async def resolve_side_effect(entity):
                if "glucose" in entity.lower():
                    return mock_resolution_well
                else:
                    return mock_resolution_sparse
            mock_resolve.side_effect = resolve_side_effect

            with patch.object(triage, 'count_edges_single', side_effect=mock_count_edges):
                # CRITICAL: Also mock direct_kg and cold_start HAS_SDK flags
                with patch.object(direct_kg, 'HAS_SDK', False):
                    with patch.object(cold_start, 'HAS_SDK', False):
                        graph = build_discovery_graph()
                        result = await graph.ainvoke({
                            "raw_query": "Analyze: glucose, novelty"
                        })

                        # Verify both branches produced findings
                        direct_findings = result.get("direct_findings", [])
                        cold_start_findings = result.get("cold_start_findings", [])

                        # Both branches should have findings
                        assert len(direct_findings) > 0, "direct_kg branch should produce findings"
                        assert len(cold_start_findings) > 0, "cold_start branch should produce findings"

                        # Synthesis should include all findings
                        report = result.get("synthesis_report", "")
                        assert "direct_kg" in report or "Tier 1" in report
                        assert "cold_start" in report or "Tier 3" in report


# =============================================================================
# End-to-End Tests
# =============================================================================

class TestEndToEnd:
    """End-to-end tests for the complete workflow."""

    @pytest.mark.asyncio
    async def test_graph_builds_successfully(self):
        """Graph should compile without errors."""
        graph = build_discovery_graph()
        assert graph is not None

    @pytest.mark.asyncio
    async def test_workflow_with_mocked_resolution(self):
        """Complete workflow with mocked entity resolution."""
        mock_resolution = EntityResolution(
            raw_name="glucose",
            curie="CHEBI:17234",
            resolved_name="D-glucose",
            category="biolink:ChemicalEntity",
            confidence=0.95,
            method="exact",
        )
        mock_score = NoveltyScore(
            curie="CHEBI:17234",
            raw_name="glucose",
            edge_count=300,
            classification="well_characterized",
        )

        with patch.object(entity_resolution, 'resolve_single_entity', return_value=mock_resolution):
            with patch.object(triage, 'count_edges_single', return_value=mock_score):
                with patch.object(direct_kg, 'HAS_SDK', False):
                    with patch.object(cold_start, 'HAS_SDK', False):
                        graph = build_discovery_graph()
                        result = await graph.ainvoke({
                            "raw_query": "Analyze these metabolites: glucose, fructose"
                        })

                        assert result["query_type"] == "discovery"
                        assert "synthesis_report" in result
                        assert len(result.get("resolved_entities", [])) > 0
                        assert len(result.get("novelty_scores", [])) > 0

    @pytest.mark.asyncio
    async def test_workflow_handles_sdk_unavailable(self):
        """Workflow should handle missing SDK gracefully."""
        with patch.object(entity_resolution, 'HAS_SDK', False):
            with patch.object(triage, 'HAS_SDK', False):
                with patch.object(direct_kg, 'HAS_SDK', False):
                    with patch.object(cold_start, 'HAS_SDK', False):
                        with patch.object(pathway_enrichment, 'HAS_SDK', False):
                            graph = build_discovery_graph()
                            result = await graph.ainvoke({
                                "raw_query": "What is glucose?"
                            })

                            # Should still complete without crashing
                            assert "synthesis_report" in result

    @pytest.mark.asyncio
    async def test_full_pipeline_with_triage(self):
        """intake -> entity_resolution -> triage -> [branches] -> synthesis"""
        mock_resolution = EntityResolution(
            raw_name="glucose",
            curie="CHEBI:17234",
            resolved_name="D-glucose",
            category="biolink:ChemicalEntity",
            confidence=0.95,
            method="exact",
        )
        mock_score = NoveltyScore(
            curie="CHEBI:17234",
            raw_name="glucose",
            edge_count=300,
            classification="well_characterized",
        )

        with patch.object(entity_resolution, 'resolve_single_entity', return_value=mock_resolution):
            with patch.object(triage, 'count_edges_single', return_value=mock_score):
                with patch.object(direct_kg, 'HAS_SDK', False):
                    with patch.object(cold_start, 'HAS_SDK', False):
                        graph = build_discovery_graph()
                        result = await graph.ainvoke({
                            "raw_query": "Analyze: glucose, KIF6, NLGN1"
                        })

                        # Verify triage populated
                        assert result.get("novelty_scores") is not None
                        assert result.get("synthesis_report") is not None

                        # Report should include classification
                        report = result["synthesis_report"]
                        assert "Well-Characterized" in report or "Tier" in report


# =============================================================================
# Phase 4a Tests: Pathway Enrichment Node
# =============================================================================

class TestPathwayEnrichmentNode:
    """Tests for the pathway enrichment node shared neighbor analysis."""

    @pytest.mark.asyncio
    async def test_empty_entities(self):
        """Empty entity list returns empty results."""
        state: DiscoveryState = {
            "resolved_entities": [],
        }
        result = await pathway_enrichment.run(state)
        assert result["shared_neighbors"] == []
        assert result["biological_themes"] == []
        assert "No valid entities" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_single_entity_returns_empty(self):
        """Single entity cannot have shared neighbors."""
        state: DiscoveryState = {
            "resolved_entities": [
                EntityResolution(
                    raw_name="glucose",
                    curie="CHEBI:17234",
                    resolved_name="D-glucose",
                    category="biolink:ChemicalEntity",
                    confidence=0.95,
                    method="exact",
                )
            ],
        }
        result = await pathway_enrichment.run(state)
        assert result["shared_neighbors"] == []
        assert result["biological_themes"] == []
        assert "Need at least 2 entities" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_sdk_unavailable_returns_placeholder(self):
        """SDK unavailable should return graceful error."""
        state: DiscoveryState = {
            "resolved_entities": [
                EntityResolution(
                    raw_name="glucose", curie="CHEBI:17234",
                    resolved_name="D-glucose", category="biolink:ChemicalEntity",
                    confidence=0.95, method="exact",
                ),
                EntityResolution(
                    raw_name="insulin", curie="HGNC:6081",
                    resolved_name="INS", category="biolink:Gene",
                    confidence=0.95, method="exact",
                ),
            ],
        }
        with patch.object(pathway_enrichment, "HAS_SDK", False):
            result = await pathway_enrichment.run(state)
            assert result["shared_neighbors"] == []
            assert "SDK not available" in result["errors"][0]

    def test_parse_enrichment_result_valid_json(self):
        """Parse valid JSON with shared neighbors and themes."""
        json_response = '''
        {
            "shared_neighbors": [
                {
                    "curie": "GO:0006915",
                    "name": "apoptotic process",
                    "category": "biolink:BiologicalProcess",
                    "degree": 245,
                    "is_hub": false,
                    "connected_inputs": ["HGNC:11998", "HGNC:7989"],
                    "predicates": ["biolink:participates_in"]
                }
            ],
            "themes": [
                {
                    "category": "biolink:BiologicalProcess",
                    "members": ["GO:0006915"],
                    "member_names": ["apoptotic process"],
                    "input_coverage": 2,
                    "top_non_hub": "GO:0006915"
                }
            ]
        }
        '''
        neighbors, themes, errors = pathway_enrichment.parse_enrichment_result(json_response)

        assert len(neighbors) == 1
        assert neighbors[0].curie == "GO:0006915"
        assert neighbors[0].name == "apoptotic process"
        assert neighbors[0].is_hub is False
        assert len(neighbors[0].connected_inputs) == 2

        assert len(themes) == 1
        assert themes[0].category == "biolink:BiologicalProcess"
        assert themes[0].input_coverage == 2
        assert len(errors) == 0

    def test_parse_enrichment_result_hub_detection(self):
        """Verify hub detection based on degree threshold."""
        json_response = '''
        {
            "shared_neighbors": [
                {
                    "curie": "GO:0005515",
                    "name": "protein binding",
                    "category": "biolink:MolecularActivity",
                    "degree": 5000,
                    "is_hub": false,
                    "connected_inputs": ["HGNC:1", "HGNC:2"],
                    "predicates": []
                }
            ],
            "themes": []
        }
        '''
        neighbors, themes, errors = pathway_enrichment.parse_enrichment_result(json_response)

        # Should be flagged as hub despite is_hub=false because degree > 1000
        assert len(neighbors) == 1
        assert neighbors[0].is_hub is True
        assert neighbors[0].degree == 5000

    def test_parse_enrichment_result_invalid_json(self):
        """Invalid JSON returns empty results with error."""
        json_response = "This is not valid JSON at all"
        neighbors, themes, errors = pathway_enrichment.parse_enrichment_result(json_response)
        assert len(neighbors) == 0
        assert len(themes) == 0
        assert len(errors) > 0


class TestPathwayEnrichmentIntegration:
    """Integration tests for pathway enrichment in full pipeline."""

    @pytest.mark.asyncio
    async def test_pipeline_includes_pathway_enrichment(self):
        """Verify pathway_enrichment node is in the pipeline."""
        graph = build_discovery_graph()
        node_names = list(graph.nodes.keys())
        assert "pathway_enrichment" in node_names

    @pytest.mark.asyncio
    async def test_synthesis_includes_enrichment_data(self):
        """Synthesis report should include pathway enrichment section."""
        mock_neighbor = SharedNeighbor(
            curie="GO:0006915",
            name="apoptotic process",
            category="biolink:BiologicalProcess",
            degree=245,
            is_hub=False,
            connected_inputs=["HGNC:11998", "HGNC:7989"],
            predicates=["biolink:participates_in"],
        )
        mock_theme = BiologicalTheme(
            category="biolink:BiologicalProcess",
            members=["GO:0006915"],
            member_names=["apoptotic process"],
            input_coverage=2,
            top_non_hub="GO:0006915",
        )

        state: DiscoveryState = {
            "raw_query": "Analyze: TP53, BRCA1",
            "query_type": "discovery",
            "resolved_entities": [
                EntityResolution(raw_name="TP53", curie="HGNC:11998",
                                resolved_name="TP53", category="biolink:Gene",
                                confidence=0.95, method="exact"),
                EntityResolution(raw_name="BRCA1", curie="HGNC:1100",
                                resolved_name="BRCA1", category="biolink:Gene",
                                confidence=0.95, method="exact"),
            ],
            "shared_neighbors": [mock_neighbor],
            "biological_themes": [mock_theme],
        }

        with patch.object(synthesis, 'HAS_SDK', False):
            result = await synthesis.run(state)
            report = result["synthesis_report"]

            assert "Pathway Enrichment" in report
            assert "Biological Themes" in report
            assert "apoptotic process" in report

# =============================================================================
# Phase 4b Tests: Integration Node (Bridges + Gap Analysis)
# =============================================================================

class TestIntegrationNode:
    """Tests for the integration node's bridge detection and gap analysis."""

    @pytest.mark.asyncio
    async def test_empty_findings_returns_empty(self):
        """No findings to integrate returns empty results."""
        state: DiscoveryState = {
            "disease_associations": [],
            "pathway_memberships": [],
            "inferred_associations": [],
            "biological_themes": [],
            "resolved_entities": [],
        }
        result = await integration.run(state)
        assert result["bridges"] == []
        assert result["gap_entities"] == []
        assert "No findings to integrate" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_sdk_unavailable_returns_error(self):
        """SDK unavailable should return graceful error."""
        state: DiscoveryState = {
            "resolved_entities": [
                EntityResolution(
                    raw_name="glucose", curie="CHEBI:17234",
                    resolved_name="D-glucose", category="biolink:ChemicalEntity",
                    confidence=0.95, method="exact",
                ),
            ],
            "disease_associations": [
                DiseaseAssociation(
                    entity_curie="CHEBI:17234",
                    disease_curie="MONDO:0005148",
                    disease_name="Type 2 Diabetes",
                    predicate="biolink:related_to",
                    source="test",
                )
            ],
        }
        with patch.object(integration, "HAS_SDK", False):
            result = await integration.run(state)
            assert result["bridges"] == []
            assert result["gap_entities"] == []
            assert "SDK not available" in result["errors"][0]

    def test_parse_integration_result_with_bridges(self):
        """Parse valid JSON with bridges."""
        json_response = '''
        {
            "bridges": [
                {
                    "path_description": "metabolite → gene → disease",
                    "entities": ["CHEBI:123", "HGNC:456", "MONDO:789"],
                    "entity_names": ["glucose", "SLC2A2", "diabetes"],
                    "predicates": ["affects_expression", "associated_with"],
                    "tier": 2,
                    "novelty": "known",
                    "significance": "Links metabolic state to disease risk"
                }
            ],
            "gaps": []
        }
        '''
        bridges, gaps, errors = integration.parse_integration_result(json_response)

        assert len(bridges) == 1
        assert bridges[0].path_description == "metabolite → gene → disease"
        assert len(bridges[0].entities) == 3
        assert bridges[0].tier == 2
        assert bridges[0].novelty == "known"
        assert len(errors) == 0

    def test_parse_integration_result_with_gaps(self):
        """Parse valid JSON with gap entities."""
        json_response = '''
        {
            "bridges": [],
            "gaps": [
                {
                    "name": "BCAAs",
                    "category": "biolink:ChemicalEntity",
                    "curie": null,
                    "expected_reason": "Canonical T2D conversion markers",
                    "absence_interpretation": "Not measured in cohort",
                    "is_informative": true
                }
            ]
        }
        '''
        bridges, gaps, errors = integration.parse_integration_result(json_response)

        assert len(gaps) == 1
        assert gaps[0].name == "BCAAs"
        assert gaps[0].is_informative is True
        assert "Open World" in gaps[0].absence_interpretation or "Not measured" in gaps[0].absence_interpretation
        assert len(errors) == 0

    def test_parse_integration_result_invalid_json(self):
        """Invalid JSON returns empty results with error."""
        json_response = "This is not valid JSON"
        bridges, gaps, errors = integration.parse_integration_result(json_response)
        assert len(bridges) == 0
        assert len(gaps) == 0
        assert len(errors) > 0

    def test_summarize_diseases(self):
        """Test disease summary generation for prompt."""
        diseases = [
            DiseaseAssociation(
                entity_curie="CHEBI:17234",
                disease_curie="MONDO:0005148",
                disease_name="Type 2 Diabetes",
                predicate="biolink:related_to",
                source="test",
            )
        ]
        summary = integration.summarize_diseases(diseases)
        assert "CHEBI:17234" in summary
        assert "Type 2 Diabetes" in summary

    def test_summarize_themes(self):
        """Test biological theme summary for prompt."""
        themes = [
            BiologicalTheme(
                category="biolink:BiologicalProcess",
                members=["GO:0006915"],
                member_names=["apoptosis"],
                input_coverage=3,
                top_non_hub="GO:0006915",
            )
        ]
        summary = integration.summarize_themes(themes)
        assert "BiologicalProcess" in summary
        assert "apoptosis" in summary


# =============================================================================
# Phase 4b Tests: Temporal Node (Longitudinal Studies)
# =============================================================================

class TestTemporalNode:
    """Tests for the temporal node's classification logic."""

    @pytest.mark.asyncio
    async def test_skips_non_longitudinal_studies(self):
        """Temporal node should skip when is_longitudinal is False."""
        state: DiscoveryState = {
            "is_longitudinal": False,
            "disease_associations": [
                DiseaseAssociation(
                    entity_curie="CHEBI:17234",
                    disease_curie="MONDO:0005148",
                    disease_name="Type 2 Diabetes",
                    predicate="biolink:related_to",
                    source="test",
                )
            ],
        }
        result = await temporal.run(state)
        assert result["temporal_classifications"] == []
        assert "not a longitudinal study" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_runs_for_longitudinal_studies(self):
        """Temporal node should run when is_longitudinal is True."""
        state: DiscoveryState = {
            "is_longitudinal": True,
            "duration_years": 5,
            "disease_associations": [
                DiseaseAssociation(
                    entity_curie="CHEBI:17234",
                    disease_curie="MONDO:0005148",
                    disease_name="Type 2 Diabetes",
                    predicate="biolink:related_to",
                    source="test",
                )
            ],
        }
        # With SDK unavailable, it should return an error but not skip
        with patch.object(temporal, "HAS_SDK", False):
            result = await temporal.run(state)
            # Should NOT contain "not a longitudinal study"
            errors = result.get("errors", [])
            assert not any("not a longitudinal study" in e for e in errors)
            # Should have SDK unavailable error instead
            assert any("SDK not available" in e for e in errors)

    @pytest.mark.asyncio
    async def test_empty_findings_returns_empty(self):
        """No findings to classify returns empty results."""
        state: DiscoveryState = {
            "is_longitudinal": True,
            "disease_associations": [],
            "inferred_associations": [],
            "bridges": [],
            "direct_findings": [],
        }
        result = await temporal.run(state)
        assert result["temporal_classifications"] == []
        assert "No findings available" in result["errors"][0]

    def test_parse_temporal_result_valid_json(self):
        """Parse valid temporal classification JSON."""
        json_response = '''
        {
            "classifications": [
                {
                    "entity": "CHEBI:17234",
                    "finding_claim": "Associated with insulin resistance",
                    "classification": "upstream_cause",
                    "reasoning": "Elevated glucose precedes T2D diagnosis",
                    "confidence": "high"
                },
                {
                    "entity": "AGE:12345",
                    "finding_claim": "Glycation product accumulation",
                    "classification": "downstream_consequence",
                    "reasoning": "AGEs form after chronic hyperglycemia",
                    "confidence": "moderate"
                }
            ]
        }
        '''
        classifications, errors = temporal.parse_temporal_result(json_response)

        assert len(classifications) == 2
        assert classifications[0].classification == "upstream_cause"
        assert classifications[0].confidence == "high"
        assert classifications[1].classification == "downstream_consequence"
        assert len(errors) == 0

    def test_parse_temporal_result_invalid_classification(self):
        """Invalid classification values should default to parallel_effect."""
        json_response = '''
        {
            "classifications": [
                {
                    "entity": "TEST:123",
                    "finding_claim": "Some finding",
                    "classification": "invalid_value",
                    "reasoning": "Test reasoning",
                    "confidence": "high"
                }
            ]
        }
        '''
        classifications, errors = temporal.parse_temporal_result(json_response)

        assert len(classifications) == 1
        assert classifications[0].classification == "parallel_effect"

    def test_parse_temporal_result_invalid_json(self):
        """Invalid JSON returns empty results with error."""
        json_response = "Not valid JSON"
        classifications, errors = temporal.parse_temporal_result(json_response)
        assert len(classifications) == 0
        assert len(errors) > 0

    def test_collect_findings_for_classification(self):
        """Test finding collection from state for temporal classification."""
        state: DiscoveryState = {
            "disease_associations": [
                DiseaseAssociation(
                    entity_curie="CHEBI:17234",
                    disease_curie="MONDO:0005148",
                    disease_name="Diabetes",
                    predicate="related_to",
                    source="test",
                )
            ],
            "inferred_associations": [],
            "bridges": [],
            "direct_findings": [],
        }
        findings = temporal.collect_findings_for_classification(state)
        assert len(findings) == 1
        assert findings[0]["entity"] == "CHEBI:17234"


# =============================================================================
# Phase 4b Tests: Routing After Integration
# =============================================================================

class TestIntegrationRouting:
    """Tests for conditional routing after integration node."""

    def test_route_to_temporal_for_longitudinal(self):
        """Longitudinal studies should route to temporal node."""
        state: DiscoveryState = {
            "is_longitudinal": True,
            "duration_years": 5,
        }
        result = route_after_integration(state)
        assert result == "temporal"

    def test_route_to_synthesis_for_non_longitudinal(self):
        """Non-longitudinal studies should skip temporal and route to synthesis."""
        state: DiscoveryState = {
            "is_longitudinal": False,
        }
        result = route_after_integration(state)
        assert result == "synthesis"

    def test_route_to_synthesis_when_flag_missing(self):
        """Missing is_longitudinal flag should default to synthesis."""
        state: DiscoveryState = {}
        result = route_after_integration(state)
        assert result == "synthesis"


# =============================================================================
# Phase 4b Tests: Synthesis with Bridges, Gaps, and Temporal
# =============================================================================

class TestSynthesisPhase4b:
    """Tests for synthesis node with Phase 4b data."""

    @pytest.mark.asyncio
    async def test_report_includes_bridges(self):
        """Report should format cross-type bridges."""
        with patch.object(synthesis, 'HAS_SDK', False):
            state: DiscoveryState = {
                "raw_query": "Analyze glucose",
                "query_type": "discovery",
                "resolved_entities": [],
                "bridges": [
                    Bridge(
                        path_description="metabolite -> gene -> disease",
                        entities=["CHEBI:17234", "HGNC:11100", "MONDO:0005148"],
                        entity_names=["glucose", "SLC2A2", "Type 2 Diabetes"],
                        predicates=["affects_expression", "associated_with"],
                        tier=2,
                        novelty="known",
                        significance="Links glucose metabolism to T2D risk",
                    )
                ],
            }
            result = await synthesis.run(state)
            report = result["synthesis_report"]
            assert "Cross-Type Bridges" in report
            assert "metabolite -> gene -> disease" in report
            assert "Links glucose metabolism" in report

    @pytest.mark.asyncio
    async def test_report_includes_gap_entities(self):
        """Report should format expected-but-absent entities."""
        with patch.object(synthesis, 'HAS_SDK', False):
            state: DiscoveryState = {
                "raw_query": "Analyze T2D conversion",
                "query_type": "discovery",
                "resolved_entities": [],
                "gap_entities": [
                    GapEntity(
                        name="BCAAs",
                        category="biolink:ChemicalEntity",
                        curie=None,
                        expected_reason="Canonical early T2D markers",
                        absence_interpretation="Not measured in this cohort",
                        is_informative=True,
                    )
                ],
            }
            result = await synthesis.run(state)
            report = result["synthesis_report"]
            assert "Gap Analysis" in report or "Expected-But-Absent" in report
            assert "BCAAs" in report
            assert "Informative" in report or "informative" in report

    @pytest.mark.asyncio
    async def test_report_includes_temporal_classifications(self):
        """Report should format temporal classifications for longitudinal studies."""
        with patch.object(synthesis, 'HAS_SDK', False):
            state: DiscoveryState = {
                "raw_query": "Analyze 5-year longitudinal T2D study",
                "query_type": "discovery",
                "is_longitudinal": True,
                "duration_years": 5,
                "resolved_entities": [],
                "temporal_classifications": [
                    TemporalClassification(
                        entity="CHEBI:17234",
                        finding_claim="Elevated glucose levels",
                        classification="upstream_cause",
                        reasoning="Glucose dysregulation precedes T2D diagnosis",
                        confidence="high",
                    ),
                    TemporalClassification(
                        entity="AGE:12345",
                        finding_claim="AGE accumulation",
                        classification="downstream_consequence",
                        reasoning="Glycation products form after chronic hyperglycemia",
                        confidence="moderate",
                    ),
                ],
            }
            result = await synthesis.run(state)
            report = result["synthesis_report"]
            assert "Temporal Analysis" in report
            assert "Upstream" in report
            assert "Downstream" in report
            assert "Longitudinal" in report

    @pytest.mark.asyncio
    async def test_report_shows_study_type_for_longitudinal(self):
        """Report header should indicate longitudinal study type."""
        with patch.object(synthesis, 'HAS_SDK', False):
            state: DiscoveryState = {
                "raw_query": "Analyze data",
                "query_type": "discovery",
                "is_longitudinal": True,
                "duration_years": 5,
                "resolved_entities": [],
            }
            result = await synthesis.run(state)
            report = result["synthesis_report"]
            assert "Longitudinal" in report
            assert "5 years" in report


class TestEndToEndPhase4b:
    """End-to-end tests for Phase 4b complete graph."""

    @pytest.mark.asyncio
    async def test_graph_has_9_nodes(self):
        """Graph should have 9 analysis nodes plus __start__."""
        graph = build_discovery_graph()
        node_names = list(graph.nodes.keys())

        expected_nodes = [
            "intake",
            "entity_resolution",
            "triage",
            "direct_kg",
            "cold_start",
            "pathway_enrichment",
            "integration",
            "temporal",
            "synthesis",
        ]

        for node in expected_nodes:
            assert node in node_names, f"Missing node: {node}"

        # Should have 9 analysis nodes + __start__
        assert len(node_names) == 10, f"Expected 10 nodes (9 + __start__), got {len(node_names)}: {node_names}"

    @pytest.mark.asyncio
    async def test_full_pipeline_non_longitudinal(self):
        """Full pipeline for non-longitudinal query should skip temporal node."""
        mock_resolution = EntityResolution(
            raw_name="glucose",
            curie="CHEBI:17234",
            resolved_name="D-glucose",
            category="biolink:ChemicalEntity",
            confidence=0.95,
            method="exact",
        )
        mock_score = NoveltyScore(
            curie="CHEBI:17234",
            raw_name="glucose",
            edge_count=300,
            classification="well_characterized",
        )

        with patch.object(entity_resolution, 'resolve_single_entity', return_value=mock_resolution):
            with patch.object(triage, 'count_edges_single', return_value=mock_score):
                with patch.object(direct_kg, 'HAS_SDK', False):
                    with patch.object(cold_start, 'HAS_SDK', False):
                        with patch.object(pathway_enrichment, 'HAS_SDK', False):
                            with patch.object(integration, 'HAS_SDK', False):
                                with patch.object(temporal, 'HAS_SDK', False):
                                    graph = build_discovery_graph()
                                    result = await graph.ainvoke({
                                        "raw_query": "Analyze glucose"
                                    })

                                    # Should complete without error
                                    assert "synthesis_report" in result

                                    # Temporal classifications should be empty
                                    assert result.get("temporal_classifications", []) == []

                                    # Report should NOT contain temporal section
                                    report = result["synthesis_report"]
                                    assert "Temporal Analysis" not in report

    @pytest.mark.asyncio
    async def test_full_pipeline_longitudinal(self):
        """Full pipeline for longitudinal query should execute temporal node."""
        mock_resolution = EntityResolution(
            raw_name="glucose",
            curie="CHEBI:17234",
            resolved_name="D-glucose",
            category="biolink:ChemicalEntity",
            confidence=0.95,
            method="exact",
        )
        mock_score = NoveltyScore(
            curie="CHEBI:17234",
            raw_name="glucose",
            edge_count=300,
            classification="well_characterized",
        )

        with patch.object(entity_resolution, 'resolve_single_entity', return_value=mock_resolution):
            with patch.object(triage, 'count_edges_single', return_value=mock_score):
                with patch.object(direct_kg, 'HAS_SDK', False):
                    with patch.object(cold_start, 'HAS_SDK', False):
                        with patch.object(pathway_enrichment, 'HAS_SDK', False):
                            with patch.object(integration, 'HAS_SDK', False):
                                with patch.object(temporal, 'HAS_SDK', False):
                                    graph = build_discovery_graph()
                                    result = await graph.ainvoke({
                                        "raw_query": "Analyze the 5-year longitudinal OGTT study data"
                                    })

                                    # Should complete without error
                                    assert "synthesis_report" in result

                                    # Should detect as longitudinal
                                    assert result.get("is_longitudinal") is True

                                    # Report should indicate longitudinal
                                    report = result["synthesis_report"]
                                    assert "Longitudinal" in report

    @pytest.mark.asyncio
    async def test_integration_and_temporal_in_pipeline(self):
        """Verify integration and temporal nodes are reachable."""
        graph = build_discovery_graph()
        node_names = list(graph.nodes.keys())

        assert "integration" in node_names
        assert "temporal" in node_names


# =============================================================================
# Integration Tests (require Kestrel API)
# =============================================================================

@pytest.mark.integration
class TestIntegration:
    """Integration tests requiring actual Kestrel API connectivity."""

    @pytest.mark.asyncio
    async def test_direct_kg_with_real_api(self):
        """Test direct_kg with actual Kestrel API calls."""
        # Skip if SDK not available
        if not direct_kg.HAS_SDK:
            pytest.skip("Claude Agent SDK not available")

        state: DiscoveryState = {
            "well_characterized_curies": ["CHEBI:17234"],  # glucose
            "moderate_curies": [],
            "novelty_scores": [NoveltyScore(curie="CHEBI:17234", raw_name="glucose", edge_count=300, classification="well_characterized")],
        }
        result = await direct_kg.run(state)

        # Should have real findings, not just "pending" stubs
        findings = result.get("direct_findings", [])
        assert len(findings) > 0
        # At least one finding should have real content (not "pending")
        assert any("pending" not in f.claim.lower() for f in findings)

    @pytest.mark.asyncio
    async def test_full_pipeline_with_real_analysis(self):
        """End-to-end with actual KG queries."""
        # Skip if SDK not available
        if not entity_resolution.HAS_SDK:
            pytest.skip("Claude Agent SDK not available")

        graph = build_discovery_graph()
        result = await graph.ainvoke({
            "raw_query": "Analyze: glucose"
        })

        findings = result.get("direct_findings", []) + result.get("cold_start_findings", [])
        # Should have at least one non-stub finding
        assert any("pending" not in f.claim.lower() and "unavailable" not in f.claim.lower() for f in findings)


# Run tests with: uv run pytest tests/test_langgraph_prototype.py -v


# =============================================================================
# Phase 5 Tests: Synthesis Hypothesis Generation Engine
# =============================================================================

class TestHypothesisModel:
    """Tests for the Hypothesis model in state.py."""

    def test_hypothesis_creation(self):
        """Hypothesis model should create with all required fields."""
        from src.kestrel_backend.graph.state import Hypothesis

        h = Hypothesis(
            title="Test hypothesis",
            tier=3,
            claim="Entity X may be associated with Y",
            supporting_entities=["CURIE:1", "CURIE:2"],
            contradicting_entities=[],
            structural_logic="X similar to Z, Z connected to Y",
            confidence="moderate",
            validation_steps=["Search literature", "Validate in cohort"],
            validation_gap_note="~18% of computational predictions reach clinical investigation",
        )

        assert h.title == "Test hypothesis"
        assert h.tier == 3
        assert len(h.supporting_entities) == 2
        assert len(h.validation_steps) == 2
        assert "18%" in h.validation_gap_note

    def test_hypothesis_immutable(self):
        """Hypothesis should be frozen (immutable)."""
        from src.kestrel_backend.graph.state import Hypothesis

        h = Hypothesis(
            title="Test",
            tier=2,
            claim="Claim",
            supporting_entities=["CURIE:1"],
            structural_logic="Logic",
            confidence="high",
            validation_steps=["Step 1"],
        )

        # Should raise on modification attempt
        with pytest.raises(Exception):  # ValidationError for frozen models
            h.title = "Modified"


class TestSynthesisPhase5:
    """Tests for Phase 5 synthesis node with hypothesis generation."""

    @pytest.mark.asyncio
    async def test_synthesis_returns_hypotheses(self):
        """Synthesis node should return hypotheses list."""
        state: DiscoveryState = {
            "raw_query": "Analyze glucose",
            "query_type": "discovery",
            "resolved_entities": [],
            "cold_start_findings": [],
            "bridges": [],
        }
        result = await synthesis.run(state)

        assert "hypotheses" in result
        assert isinstance(result["hypotheses"], list)

    @pytest.mark.asyncio
    async def test_hypotheses_from_cold_start_findings(self):
        """Hypotheses should be extracted from cold_start_findings."""
        from src.kestrel_backend.graph.state import Hypothesis

        state: DiscoveryState = {
            "raw_query": "Analyze unknown metabolite",
            "query_type": "discovery",
            "resolved_entities": [],
            "cold_start_findings": [
                Finding(
                    entity="UNKNOWN:123",
                    claim="Unknown metabolite may be associated with diabetes via glucose similarity",
                    tier=3,
                    source="cold_start",
                    confidence="low",
                    logic_chain="UNKNOWN:123 similar to glucose, glucose associated with diabetes",
                ),
            ],
            "bridges": [],
        }
        result = await synthesis.run(state)

        hypotheses = result["hypotheses"]
        assert len(hypotheses) >= 1

        # Find the hypothesis from the cold-start finding
        cs_hypothesis = next((h for h in hypotheses if "UNKNOWN:123" in h.supporting_entities), None)
        assert cs_hypothesis is not None
        assert cs_hypothesis.tier == 3
        assert "diabetes" in cs_hypothesis.claim or "glucose" in cs_hypothesis.claim

    @pytest.mark.asyncio
    async def test_hypotheses_from_bridges(self):
        """Hypotheses should be extracted from bridges."""
        state: DiscoveryState = {
            "raw_query": "Analyze bridge connections",
            "query_type": "discovery",
            "resolved_entities": [],
            "cold_start_findings": [],
            "bridges": [
                Bridge(
                    path_description="metabolite -> gene -> disease",
                    entities=["CHEBI:123", "HGNC:456", "MONDO:789"],
                    entity_names=["MetaboliteX", "GeneY", "DiseaseZ"],
                    predicates=["biolink:affects", "biolink:associated_with"],
                    tier=2,
                    novelty="inferred",
                    significance="MetaboliteX may influence DiseaseZ through GeneY regulation",
                ),
            ],
        }
        result = await synthesis.run(state)

        hypotheses = result["hypotheses"]
        assert len(hypotheses) >= 1

        bridge_hypothesis = next((h for h in hypotheses if "Bridge" in h.title), None)
        assert bridge_hypothesis is not None
        assert bridge_hypothesis.tier == 2
        assert "MetaboliteX" in bridge_hypothesis.claim or "DiseaseZ" in bridge_hypothesis.claim

    @pytest.mark.asyncio
    async def test_hypotheses_have_validation_steps(self):
        """All hypotheses should have non-empty validation_steps."""
        state: DiscoveryState = {
            "raw_query": "Test validation steps",
            "query_type": "discovery",
            "resolved_entities": [],
            "cold_start_findings": [
                Finding(
                    entity="TEST:1",
                    claim="Test finding with validation needs",
                    tier=3,
                    source="cold_start",
                    confidence="low",
                    logic_chain="Test logic",
                ),
            ],
            "bridges": [
                Bridge(
                    path_description="test path",
                    entities=["A", "B"],
                    entity_names=["EntityA", "EntityB"],
                    predicates=["relates_to"],
                    tier=3,
                    novelty="inferred",
                    significance="Test bridge significance",
                ),
            ],
        }
        result = await synthesis.run(state)

        hypotheses = result["hypotheses"]
        assert len(hypotheses) >= 2

        for h in hypotheses:
            assert len(h.validation_steps) > 0, f"Hypothesis '{h.title}' has no validation steps"
            assert all(step for step in h.validation_steps), f"Hypothesis '{h.title}' has empty validation step"

    @pytest.mark.asyncio
    async def test_hypotheses_include_validation_gap(self):
        """All hypotheses should include the ~18% validation gap note."""
        state: DiscoveryState = {
            "raw_query": "Test validation gap",
            "query_type": "discovery",
            "resolved_entities": [],
            "cold_start_findings": [
                Finding(
                    entity="GAP:1",
                    claim="Finding requiring calibration note",
                    tier=3,
                    source="cold_start",
                    confidence="low",
                    logic_chain="Inference logic",
                ),
            ],
            "bridges": [
                Bridge(
                    path_description="gap test path",
                    entities=["X", "Y"],
                    entity_names=["X", "Y"],
                    predicates=["related"],
                    tier=3,
                    novelty="inferred",
                    significance="Bridge with calibration",
                ),
            ],
        }
        result = await synthesis.run(state)

        hypotheses = result["hypotheses"]
        assert len(hypotheses) >= 2

        for h in hypotheses:
            assert "18%" in h.validation_gap_note, f"Hypothesis '{h.title}' missing ~18% validation gap"

    @pytest.mark.asyncio
    async def test_fallback_report_without_sdk(self):
        """When SDK unavailable, should use fallback_report."""
        # Temporarily disable SDK
        original_has_sdk = synthesis.HAS_SDK
        synthesis.HAS_SDK = False

        try:
            state: DiscoveryState = {
                "raw_query": "Test fallback",
                "query_type": "discovery",
                "resolved_entities": [
                    EntityResolution(
                        raw_name="glucose",
                        curie="CHEBI:17234",
                        resolved_name="D-glucose",
                        category="biolink:ChemicalEntity",
                        confidence=0.95,
                        method="exact",
                    )
                ],
            }
            result = await synthesis.run(state)

            report = result["synthesis_report"]
            assert "KRAKEN Analysis Report" in report
            assert "glucose" in report.lower()
        finally:
            synthesis.HAS_SDK = original_has_sdk

    @pytest.mark.asyncio
    async def test_fallback_report_structure(self):
        """Fallback report should have expected sections."""
        state: DiscoveryState = {
            "raw_query": "Test structure",
            "query_type": "discovery",
            "is_longitudinal": True,
            "duration_years": 5,
            "resolved_entities": [
                EntityResolution(
                    raw_name="glucose",
                    curie="CHEBI:17234",
                    resolved_name="D-glucose",
                    category="biolink:ChemicalEntity",
                    confidence=0.95,
                    method="exact",
                )
            ],
            "novelty_scores": [
                NoveltyScore(
                    curie="CHEBI:17234",
                    raw_name="glucose",
                    edge_count=300,
                    classification="well_characterized",
                )
            ],
            "bridges": [
                Bridge(
                    path_description="test bridge",
                    entities=["A", "B"],
                    entity_names=["A", "B"],
                    predicates=["relates"],
                    tier=2,
                    novelty="known",
                    significance="Test significance",
                ),
            ],
            "gap_entities": [
                GapEntity(
                    name="TestGap",
                    category="biolink:ChemicalEntity",
                    expected_reason="Expected in metabolic studies",
                    absence_interpretation="May not be measured in this platform",
                    is_informative=True,
                ),
            ],
        }

        # Use fallback directly
        report = synthesis.fallback_report(state)

        assert "KRAKEN Analysis Report" in report
        assert "Longitudinal" in report
        assert "Entity Resolution" in report
        assert "Entity Classification" in report or "Well-Characterized" in report
        assert "Cross-Type Bridges" in report
        assert "Gap Analysis" in report

    @pytest.mark.asyncio
    async def test_fdr_entities_in_context(self):
        """FDR entities should appear in synthesis context."""
        state: DiscoveryState = {
            "raw_query": "Test FDR",
            "query_type": "discovery",
            "resolved_entities": [],
            "fdr_entities": ["Glucose", "Insulin", "HOMA-IR"],
            "marginal_entities": ["Lactate"],
        }

        context = synthesis.assemble_synthesis_context(state)

        assert "FDR-Significant" in context
        assert "Glucose" in context
        assert "Marginal" in context
        assert "Lactate" in context

    @pytest.mark.asyncio
    async def test_hypotheses_from_inferred_associations(self):
        """Hypotheses should be extracted from inferred_associations."""
        state: DiscoveryState = {
            "raw_query": "Test inferred",
            "query_type": "discovery",
            "resolved_entities": [],
            "cold_start_findings": [],
            "bridges": [],
            "inferred_associations": [
                InferredAssociation(
                    source_entity="SPARSE:1",
                    target_curie="MONDO:123",
                    target_name="Type 2 Diabetes",
                    predicate="biolink:associated_with",
                    logic_chain="SPARSE:1 similar to glucose, glucose associated with T2D",
                    supporting_analogues=3,
                    confidence="low",
                    validation_step="Check GWAS Catalog for SPARSE:1",
                ),
            ],
        }
        result = await synthesis.run(state)

        hypotheses = result["hypotheses"]
        assert len(hypotheses) >= 1

        inf_hypothesis = next((h for h in hypotheses if "Inferred" in h.title), None)
        assert inf_hypothesis is not None
        assert inf_hypothesis.tier == 3
        assert "Type 2 Diabetes" in inf_hypothesis.claim
        assert "18%" in inf_hypothesis.validation_gap_note


class TestExtractHypotheses:
    """Unit tests for the extract_hypotheses function."""

    def test_skips_pending_findings(self):
        """Should skip findings with 'pending' in claim."""
        state: DiscoveryState = {
            "cold_start_findings": [
                Finding(
                    entity="TEST:1",
                    claim="Analysis pending due to SDK unavailability",
                    tier=3,
                    source="cold_start",
                    confidence="low",
                ),
            ],
            "bridges": [],
        }

        hypotheses = synthesis.extract_hypotheses(state)
        assert len(hypotheses) == 0

    def test_skips_bridges_without_significance(self):
        """Should skip bridges without significance."""
        state: DiscoveryState = {
            "cold_start_findings": [],
            "bridges": [
                Bridge(
                    path_description="empty bridge",
                    entities=["A", "B"],
                    tier=3,
                    novelty="inferred",
                    significance="",  # Empty significance
                ),
            ],
        }

        hypotheses = synthesis.extract_hypotheses(state)
        assert len(hypotheses) == 0

    def test_only_tier3_from_cold_start(self):
        """Should only extract Tier 3 hypotheses from cold_start_findings."""
        state: DiscoveryState = {
            "cold_start_findings": [
                Finding(
                    entity="TIER1:1",
                    claim="High confidence finding",
                    tier=1,
                    source="cold_start",
                    confidence="high",
                ),
                Finding(
                    entity="TIER3:1",
                    claim="Speculative finding",
                    tier=3,
                    source="cold_start",
                    confidence="low",
                    logic_chain="Test logic",
                ),
            ],
            "bridges": [],
        }

        hypotheses = synthesis.extract_hypotheses(state)
        # Should only have the Tier 3 finding
        assert len(hypotheses) == 1
        assert hypotheses[0].tier == 3


class TestEndToEndPhase5:
    """End-to-end tests for Phase 5 synthesis with hypotheses."""

    @pytest.mark.asyncio
    async def test_full_pipeline_produces_hypotheses(self):
        """Full pipeline should produce hypothesis objects."""
        mock_resolution = EntityResolution(
            raw_name="glucose",
            curie="CHEBI:17234",
            resolved_name="D-glucose",
            category="biolink:ChemicalEntity",
            confidence=0.95,
            method="exact",
        )
        mock_score = NoveltyScore(
            curie="CHEBI:17234",
            raw_name="glucose",
            edge_count=300,
            classification="well_characterized",
        )

        # Create a mock bridge that will be added during integration
        mock_bridge = Bridge(
            path_description="metabolite -> disease",
            entities=["CHEBI:17234", "MONDO:123"],
            entity_names=["D-glucose", "Diabetes"],
            predicates=["biolink:associated_with"],
            tier=2,
            novelty="known",
            significance="Glucose dysregulation is a hallmark of diabetes",
        )

        with patch.object(entity_resolution, 'resolve_single_entity', return_value=mock_resolution):
            with patch.object(triage, 'count_edges_single', return_value=mock_score):
                with patch.object(direct_kg, 'HAS_SDK', False):
                    with patch.object(cold_start, 'HAS_SDK', False):
                        with patch.object(pathway_enrichment, 'HAS_SDK', False):
                            # Mock integration to return our test bridge
                            async def mock_integration_run(state):
                                return {"bridges": [mock_bridge], "gap_entities": []}
                            with patch.object(integration, 'run', mock_integration_run):
                                with patch.object(temporal, 'HAS_SDK', False):
                                    with patch.object(synthesis, 'HAS_SDK', False):
                                        graph = build_discovery_graph()
                                        result = await graph.ainvoke({
                                            "raw_query": "Analyze glucose and its disease associations"
                                        })

                                        # Should have hypotheses
                                        assert "hypotheses" in result
                                        hypotheses = result["hypotheses"]
                                        assert len(hypotheses) >= 1

                                        # Check hypothesis structure
                                        for h in hypotheses:
                                            assert h.title
                                            assert h.claim
                                            assert len(h.validation_steps) > 0
                                            assert "18%" in h.validation_gap_note

    @pytest.mark.asyncio
    async def test_synthesis_report_exists(self):
        """Synthesis should always produce a report."""
        state: DiscoveryState = {
            "raw_query": "Simple test",
            "query_type": "retrieval",
            "resolved_entities": [],
        }
        result = await synthesis.run(state)

        assert "synthesis_report" in result
        assert len(result["synthesis_report"]) > 0
        assert "KRAKEN" in result["synthesis_report"] or "No entities" in result["synthesis_report"]

    @pytest.mark.asyncio
    async def test_hypothesis_reducer_merges_lists(self):
        """The operator.add reducer should merge hypothesis lists from parallel branches."""
        from src.kestrel_backend.graph.state import Hypothesis

        # Create two hypothesis objects
        h1 = Hypothesis(
            title="Hypothesis 1",
            tier=3,
            claim="Claim 1",
            supporting_entities=["A"],
            structural_logic="Logic 1",
            confidence="low",
            validation_steps=["Step 1"],
            validation_gap_note="~18% calibration",
        )
        h2 = Hypothesis(
            title="Hypothesis 2",
            tier=2,
            claim="Claim 2",
            supporting_entities=["B"],
            structural_logic="Logic 2",
            confidence="moderate",
            validation_steps=["Step 2"],
            validation_gap_note="~18% calibration",
        )

        # Test that lists can be merged (simulates parallel writes)
        list1 = [h1]
        list2 = [h2]
        import operator
        merged = operator.add(list1, list2)

        assert len(merged) == 2
        assert merged[0].title == "Hypothesis 1"
        assert merged[1].title == "Hypothesis 2"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
