"""
Tests for multi-hop query API integration across discovery pipeline nodes.

Run with: uv run pytest tests/test_multi_hop_integration.py -v

Tests cover:
1. kestrel_client.multi_hop_query wrapper function
2. integration.detect_bridges_via_api (cross-type bridge detection)
3. hypothesis_extraction.validate_bridge_hypotheses (bridge validation)
4. pathway_enrichment.find_two_hop_shared_neighbors (indirect connectivity)
"""

import pytest
import json
from unittest.mock import AsyncMock, patch

from src.kestrel_backend.kestrel_client import multi_hop_query
from src.kestrel_backend.graph.nodes.integration import detect_bridges_via_api, parse_multi_hop_result
from src.kestrel_backend.graph.nodes.hypothesis_extraction import validate_bridge_hypotheses
from src.kestrel_backend.graph.nodes.pathway_enrichment import find_two_hop_shared_neighbors
from src.kestrel_backend.graph.state import EntityResolution, Bridge


# =============================================================================
# Test multi_hop_query wrapper
# =============================================================================

class TestMultiHopQueryWrapper:
    """Tests for the kestrel_client.multi_hop_query function."""

    @pytest.mark.asyncio
    async def test_singly_pinned_query(self):
        """Test singly-pinned mode (start nodes only)."""
        with patch('src.kestrel_backend.kestrel_client.call_kestrel_tool', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {
                "content": [{"type": "text", "text": '{"results": []}'}],
                "isError": False,
            }

            result = await multi_hop_query(
                start_node_ids=["CHEBI:17234"],
                max_hops=2,
                limit=10,
            )

            # Verify the call was made correctly. The wrapper maps max_hops -> max_path_length.
            mock_call.assert_called_once()
            call_args = mock_call.call_args[0]
            assert call_args[0] == "multi_hop_query"
            assert call_args[1]["start_node_ids"] == ["CHEBI:17234"]
            assert call_args[1]["max_path_length"] == 2
            assert call_args[1]["limit"] == 10
            assert "end_node_ids" not in call_args[1]

            assert not result.get("isError")

    @pytest.mark.asyncio
    async def test_doubly_pinned_query(self):
        """Test doubly-pinned mode (start and end nodes)."""
        with patch('src.kestrel_backend.kestrel_client.call_kestrel_tool', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {
                "content": [{"type": "text", "text": '{"results": []}'}],
                "isError": False,
            }

            result = await multi_hop_query(
                start_node_ids=["CHEBI:17234"],
                end_node_ids=["MONDO:0005148"],
                max_hops=3,
            )

            call_args = mock_call.call_args[0]
            assert call_args[1]["start_node_ids"] == ["CHEBI:17234"]
            assert call_args[1]["end_node_ids"] == ["MONDO:0005148"]
            assert call_args[1]["max_path_length"] == 3

    @pytest.mark.asyncio
    async def test_validation_max_hops(self):
        """Test that max_hops is validated."""
        result = await multi_hop_query(
            start_node_ids=["CHEBI:17234"],
            max_hops=10,  # Invalid: > 5
        )

        assert result.get("isError") is True
        assert "max_hops must be between 1 and 5" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_validation_start_nodes_required(self):
        """Test that start_node_ids is required."""
        result = await multi_hop_query(
            start_node_ids=None,
            max_hops=2,
        )

        assert result.get("isError") is True
        assert "start_node_ids is required" in result["content"][0]["text"]


# =============================================================================
# Test integration.detect_bridges_via_api
# =============================================================================

class TestDetectBridgesViaAPI:
    """Tests for API-based bridge detection in integration node."""

    @pytest.mark.asyncio
    async def test_empty_entities(self):
        """Empty entity list should return empty bridges."""
        bridges, errors = await detect_bridges_via_api([])
        assert bridges == []
        assert errors == []

    @pytest.mark.asyncio
    async def test_single_category(self):
        """Single category cannot have cross-type bridges."""
        entities = [
            EntityResolution(
                raw_name="glucose",
                curie="CHEBI:17234",
                resolved_name="glucose",
                category="biolink:ChemicalEntity",
                confidence=1.0,
                method="exact",
            ),
            EntityResolution(
                raw_name="fructose",
                curie="CHEBI:28757",
                resolved_name="fructose",
                category="biolink:ChemicalEntity",
                confidence=1.0,
                method="exact",
            ),
        ]

        bridges, errors = await detect_bridges_via_api(entities)
        assert bridges == []

    @pytest.mark.asyncio
    async def test_cross_category_bridge_detection(self):
        """Test detecting bridges across different categories."""
        entities = [
            EntityResolution(
                raw_name="glucose",
                curie="CHEBI:17234",
                resolved_name="glucose",
                category="biolink:ChemicalEntity",
                confidence=1.0,
                method="exact",
            ),
            EntityResolution(
                raw_name="INS",
                curie="HGNC:6081",
                resolved_name="INS",
                category="biolink:Gene",
                confidence=1.0,
                method="exact",
            ),
        ]

        with patch('src.kestrel_backend.graph.nodes.integration.multi_hop_query', new_callable=AsyncMock) as mock_mhq:
            # Mock a successful path response
            # Real Kestrel shape: results -> per-result paths (CURIE-string lists) + end_node_id;
            # names from the top-level nodes dict (NOT a per-path {nodes,predicates} dict).
            mock_mhq.return_value = {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "results": [{"end_node_id": "HGNC:6081",
                                     "paths": [["CHEBI:17234", "HGNC:6081"]]}],
                        "nodes": {"CHEBI:17234": {"name": "glucose"},
                                  "HGNC:6081": {"name": "INS"}},
                    })
                }],
                "isError": False,
            }

            bridges, errors = await detect_bridges_via_api(entities, max_hops=2)

            assert len(bridges) == 1
            assert bridges[0].tier == 2  # 1-hop path (<=2 hops) -> tier 2
            assert "CHEBI:17234" in bridges[0].entities
            assert "HGNC:6081" in bridges[0].entities


class TestParseMultiHopResult:
    """Tests for parsing multi_hop_query results."""

    def test_parse_valid_path(self):
        """Test parsing a valid path result."""
        result = {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "results": [{
                        "end_node_id": "MONDO:0005148",
                        "paths": [["CHEBI:17234", "HGNC:6081", "MONDO:0005148"]],
                    }],
                    "nodes": {
                        "CHEBI:17234": {"name": "glucose"},
                        "HGNC:6081": {"name": "INS"},
                        "MONDO:0005148": {"name": "type 2 diabetes"},
                    },
                })
            }]
        }

        start_entities = [
            EntityResolution(
                raw_name="glucose",
                curie="CHEBI:17234",
                resolved_name="glucose",
                category="biolink:ChemicalEntity",
                confidence=1.0,
                method="exact",
            ),
        ]

        end_entities = [
            EntityResolution(
                raw_name="type 2 diabetes",
                curie="MONDO:0005148",
                resolved_name="type 2 diabetes",
                category="biolink:Disease",
                confidence=1.0,
                method="exact",
            ),
        ]

        bridges = parse_multi_hop_result(
            result,
            start_entities,
            end_entities,
            "biolink:ChemicalEntity",
            "biolink:Disease",
        )

        assert len(bridges) == 1
        bridge = bridges[0]
        assert len(bridge.entities) == 3
        assert bridge.entities[0] == "CHEBI:17234"
        assert bridge.entities[-1] == "MONDO:0005148"
        assert bridge.tier == 2  # 3 nodes = 2 hops = tier 2
        assert bridge.novelty == "known"
        # No edges in the response -> predicates/directions are all-None placeholders, hop-aligned.
        assert bridge.predicates == ["", ""]
        assert bridge.predicate_directions == [None, None]

    def test_parse_populates_hop_aligned_predicates_and_directions(self):
        """With edges/edge_schema, the builder fills Bridge.predicates + orientation per hop (U0)."""
        result = {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "results": [{
                        "end_node_id": "MONDO:0005148",
                        "paths": [["CHEBI:17234", "HGNC:6081", "MONDO:0005148"]],
                        "edge_ids": [1, 2],
                    }],
                    "nodes": {
                        "CHEBI:17234": {"name": "glucose"},
                        "HGNC:6081": {"name": "INS"},
                        "MONDO:0005148": {"name": "type 2 diabetes"},
                    },
                    "edge_schema": ["subject", "predicate", "object"],
                    "edges": {
                        # hop 1 forward: glucose -> INS
                        "1": ["CHEBI:17234", "biolink:affects", "HGNC:6081"],
                        # hop 2 stored REVERSED: disease -> gene (orientation must be recovered)
                        "2": ["MONDO:0005148", "biolink:gene_associated_with_condition", "HGNC:6081"],
                    },
                })
            }]
        }
        start = [EntityResolution(raw_name="glucose", curie="CHEBI:17234", resolved_name="glucose",
                                  category="biolink:ChemicalEntity", confidence=1.0, method="exact")]
        end = [EntityResolution(raw_name="type 2 diabetes", curie="MONDO:0005148",
                                resolved_name="type 2 diabetes", category="biolink:Disease",
                                confidence=1.0, method="exact")]
        bridges = parse_multi_hop_result(result, start, end,
                                         "biolink:ChemicalEntity", "biolink:Disease")
        bridge = bridges[0]
        assert bridge.predicates == ["biolink:affects", "biolink:gene_associated_with_condition"]
        # hop 1 runs with the path (forward); hop 2 edge is stored reversed -> forward=False.
        assert bridge.predicate_directions == [True, False]

    def test_parse_empty_result(self):
        """Test parsing empty result."""
        result = {"content": []}
        bridges = parse_multi_hop_result(result, [], [], "cat1", "cat2")
        assert bridges == []


# =============================================================================
# Test synthesis.validate_bridge_hypotheses
# =============================================================================

class TestValidateBridgeHypotheses:
    """Tests for bridge validation in synthesis node."""

    @pytest.mark.asyncio
    async def test_empty_bridges(self):
        """Empty bridge list should return empty."""
        validated = await validate_bridge_hypotheses([])
        assert validated == []

    @pytest.mark.asyncio
    async def test_tier2_bridges_unchanged(self):
        """Tier 2 bridges should not be validated (already high confidence)."""
        bridge = Bridge(
            path_description="test path",
            entities=["CHEBI:17234", "HGNC:6081"],
            entity_names=["glucose", "INS"],
            predicates=["biolink:affects"],
            tier=2,
            novelty="known",
            significance="Test bridge",
        )

        validated = await validate_bridge_hypotheses([bridge])
        assert len(validated) == 1
        assert validated[0].tier == 2

    @pytest.mark.asyncio
    async def test_tier3_upgrade_on_validation(self):
        """Tier 3 bridge should upgrade to Tier 2 if validated."""
        bridge = Bridge(
            path_description="test path",
            entities=["CHEBI:17234", "HGNC:6081"],
            entity_names=["glucose", "INS"],
            predicates=["biolink:affects"],
            tier=3,
            novelty="inferred",
            significance="Test bridge",
        )

        with patch('src.kestrel_backend.graph.nodes.hypothesis_extraction.multi_hop_query', new_callable=AsyncMock) as mock_mhq:
            # Mock successful validation — real shape with a reachable path.
            mock_mhq.return_value = {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "results": [{"end_node_id": "HGNC:6081",
                                     "paths": [["CHEBI:17234", "HGNC:6081"]]}],
                        "nodes": {"CHEBI:17234": {"name": "glucose"},
                                  "HGNC:6081": {"name": "INS"}},
                    })
                }],
                "isError": False,
            }

            validated = await validate_bridge_hypotheses([bridge])

            assert len(validated) == 1
            assert validated[0].tier == 2  # UPGRADED
            assert validated[0].novelty == "known"
            assert "[KG-validated]" in validated[0].significance

    @pytest.mark.asyncio
    async def test_tier3_remains_if_not_validated(self):
        """Tier 3 bridge should remain Tier 3 if validation fails."""
        bridge = Bridge(
            path_description="test path",
            entities=["CHEBI:17234", "HGNC:6081"],
            entity_names=["glucose", "INS"],
            predicates=["biolink:affects"],
            tier=3,
            novelty="inferred",
            significance="Test bridge",
        )

        with patch('src.kestrel_backend.graph.nodes.hypothesis_extraction.multi_hop_query', new_callable=AsyncMock) as mock_mhq:
            # Mock no paths found (real empty shape).
            mock_mhq.return_value = {
                "content": [{
                    "type": "text",
                    "text": json.dumps({"results": []})
                }],
                "isError": False,
            }

            validated = await validate_bridge_hypotheses([bridge])

            assert len(validated) == 1
            assert validated[0].tier == 3  # Remains Tier 3


# =============================================================================
# Test pathway_enrichment.find_two_hop_shared_neighbors
# =============================================================================

class TestFindTwoHopSharedNeighbors:
    """Tests for two-hop shared neighbor detection."""

    @pytest.mark.asyncio
    async def test_empty_entities(self):
        """Empty entity list should return empty."""
        neighbors, errors = await find_two_hop_shared_neighbors([])
        assert neighbors == {}
        assert errors == []

    @pytest.mark.asyncio
    async def test_single_entity(self):
        """Single entity cannot have shared neighbors."""
        neighbors, errors = await find_two_hop_shared_neighbors(["CHEBI:17234"])
        assert neighbors == {}

    @pytest.mark.asyncio
    async def test_shared_neighbor_detection(self):
        """Test detecting shared neighbors from two-hop paths."""
        with patch('src.kestrel_backend.graph.nodes.pathway_enrichment.multi_hop_query', new_callable=AsyncMock) as mock_mhq:
            # Mock responses for two entities sharing a neighbor
            def mock_response(start_node_ids, **kwargs):
                if start_node_ids == ["CHEBI:17234"]:
                    # Glucose reaches GO:0005737 and HGNC:6081 (real shape: CURIE-list path).
                    return {
                        "content": [{
                            "type": "text",
                            "text": json.dumps({
                                "results": [{"end_node_id": "HGNC:6081",
                                             "paths": [["CHEBI:17234", "GO:0005737", "HGNC:6081"]]}],
                                "nodes": {},
                            })
                        }],
                        "isError": False,
                    }
                elif start_node_ids == ["CHEBI:28757"]:
                    # Fructose also reaches GO:0005737 and HGNC:6081.
                    return {
                        "content": [{
                            "type": "text",
                            "text": json.dumps({
                                "results": [{"end_node_id": "HGNC:6081",
                                             "paths": [["CHEBI:28757", "GO:0005737", "HGNC:6081"]]}],
                                "nodes": {},
                            })
                        }],
                        "isError": False,
                    }
                return {"content": [], "isError": False}

            mock_mhq.side_effect = mock_response

            neighbors, errors = await find_two_hop_shared_neighbors(
                ["CHEBI:17234", "CHEBI:28757"],
                max_results_per_entity=50,
            )

            # Both entities reach GO:0005737 and HGNC:6081
            assert "GO:0005737" in neighbors
            assert "HGNC:6081" in neighbors
            assert neighbors["GO:0005737"] == 2
            assert neighbors["HGNC:6081"] == 2

    @pytest.mark.asyncio
    async def test_filters_single_connections(self):
        """Test that neighbors connected to only 1 input are filtered out."""
        with patch('src.kestrel_backend.graph.nodes.pathway_enrichment.multi_hop_query', new_callable=AsyncMock) as mock_mhq:
            def mock_response(start_node_ids, **kwargs):
                if start_node_ids == ["CHEBI:17234"]:
                    return {
                        "content": [{
                            "type": "text",
                            "text": json.dumps({
                                "results": [{"end_node_id": "UNIQUE:001",
                                             "paths": [["CHEBI:17234", "UNIQUE:001"]]}],
                                "nodes": {},
                            })
                        }],
                        "isError": False,
                    }
                elif start_node_ids == ["CHEBI:28757"]:
                    return {
                        "content": [{
                            "type": "text",
                            "text": json.dumps({
                                "results": [{"end_node_id": "UNIQUE:002",
                                             "paths": [["CHEBI:28757", "UNIQUE:002"]]}],
                                "nodes": {},
                            })
                        }],
                        "isError": False,
                    }
                return {"content": [], "isError": False}

            mock_mhq.side_effect = mock_response

            neighbors, errors = await find_two_hop_shared_neighbors(
                ["CHEBI:17234", "CHEBI:28757"],
            )

            # No shared neighbors (each unique neighbor connects to only 1 input)
            assert neighbors == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
