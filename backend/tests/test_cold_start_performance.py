"""
Performance tests for cold-start analysis node.

Tests the optimization changes for Issue #10:
- Top 5 sparse + top 3 cold-start entity selection
- Early termination for low-quality analogues
- Connection pooling for HTTP requests
- Reduced SDK inference timeout and increased parallelism
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from kestrel_backend.graph.nodes.cold_start import (
    run,
    score_entity_complexity,
    analyze_cold_start_entity,
)
from kestrel_backend.graph.state import (
    DiscoveryState,
    NoveltyScore,
    EntityResolution,
)


class TestEntityComplexityScoring:
    """Test entity complexity scoring for prioritization."""

    def test_cold_start_entities_highest_priority(self):
        """Cold-start entities (0 edges) should have highest priority."""
        assert score_entity_complexity(0) == 0.0

    def test_sparse_entities_sorted_by_edge_count(self):
        """Sparse entities should be sorted by edge count."""
        assert score_entity_complexity(1) < score_entity_complexity(5)
        assert score_entity_complexity(5) < score_entity_complexity(19)

    def test_linear_scoring(self):
        """Score should increase linearly with edge count."""
        assert score_entity_complexity(1) == 1.0
        assert score_entity_complexity(10) == 10.0
        assert score_entity_complexity(19) == 19.0


class TestEntityLimiting:
    """Test limiting to top 5 sparse + top 3 cold-start entities."""

    @pytest.mark.asyncio
    async def test_limits_sparse_entities(self):
        """Should limit to top 5 sparse entities by edge count."""
        # Create 10 sparse entities with varying edge counts
        novelty_scores = [
            NoveltyScore(curie=f"SPARSE:{i}", raw_name=f"Sparse {i}", edge_count=i, classification="sparse")
            for i in range(1, 11)
        ]

        state: DiscoveryState = {
            "sparse_curies": [f"SPARSE:{i}" for i in range(1, 11)],
            "cold_start_curies": [],
            "novelty_scores": novelty_scores,
            "resolved_entities": [],
        }

        # Mock analyze_cold_start_entity to avoid actual API calls
        with patch("kestrel_backend.graph.nodes.cold_start.analyze_cold_start_entity") as mock_analyze:
            mock_analyze.return_value = ([], [], [], [])

            result = await run(state)

            # Should have analyzed exactly 5 entities (top 5 by lowest edge count)
            assert mock_analyze.call_count == 5
            assert result["cold_start_skipped_count"] == 5

    @pytest.mark.asyncio
    async def test_limits_cold_start_entities(self):
        """Should limit to top 3 cold-start entities."""
        # Create 5 cold-start entities
        novelty_scores = [
            NoveltyScore(curie=f"COLD:{i}", raw_name=f"Cold {i}", edge_count=0, classification="cold_start")
            for i in range(5)
        ]

        state: DiscoveryState = {
            "sparse_curies": [],
            "cold_start_curies": [f"COLD:{i}" for i in range(5)],
            "novelty_scores": novelty_scores,
            "resolved_entities": [],
        }

        with patch("kestrel_backend.graph.nodes.cold_start.analyze_cold_start_entity") as mock_analyze:
            mock_analyze.return_value = ([], [], [], [])

            result = await run(state)

            # Should have analyzed exactly 3 entities
            assert mock_analyze.call_count == 3
            assert result["cold_start_skipped_count"] == 2

    @pytest.mark.asyncio
    async def test_prioritizes_by_edge_count(self):
        """Should prioritize entities with fewer edges."""
        # Create sparse entities with specific edge counts (not sorted)
        novelty_scores = [
            NoveltyScore(curie="SPARSE:HIGH", raw_name="High", edge_count=15, classification="sparse"),
            NoveltyScore(curie="SPARSE:LOW", raw_name="Low", edge_count=3, classification="sparse"),
            NoveltyScore(curie="SPARSE:MED", raw_name="Med", edge_count=8, classification="sparse"),
        ]

        state: DiscoveryState = {
            "sparse_curies": ["SPARSE:HIGH", "SPARSE:LOW", "SPARSE:MED"],
            "cold_start_curies": [],
            "novelty_scores": novelty_scores,
            "resolved_entities": [],
        }

        with patch("kestrel_backend.graph.nodes.cold_start.analyze_cold_start_entity") as mock_analyze:
            mock_analyze.return_value = ([], [], [], [])

            await run(state)

            # Should have called in order: LOW (3), MED (8), HIGH (15)
            calls = [call[0][0] for call in mock_analyze.call_args_list]
            assert calls == ["SPARSE:LOW", "SPARSE:MED", "SPARSE:HIGH"]


class TestEarlyTermination:
    """Test early termination for low-quality analogues."""

    @pytest.mark.asyncio
    async def test_skips_inference_for_low_quality_analogues(self):
        """Should skip SDK inference if all analogues have similarity < 0.7."""
        # Mock get_similar_entities to return low-quality analogues
        low_quality_analogues = [
            {"curie": "ANALOGUE:1", "name": "Poor Match 1", "similarity": 0.5, "category": "ChemicalEntity"},
            {"curie": "ANALOGUE:2", "name": "Poor Match 2", "similarity": 0.6, "category": "ChemicalEntity"},
        ]

        with patch("kestrel_backend.graph.nodes.cold_start.get_similar_entities") as mock_similar:
            mock_similar.return_value = low_quality_analogues

            # Mock query to ensure it's NOT called
            with patch("kestrel_backend.graph.nodes.cold_start.query") as mock_query:
                analogues, inferences, findings, errors = await analyze_cold_start_entity(
                    "TEST:001", "Test Entity", 5
                )

                # SDK inference should NOT have been called
                mock_query.assert_not_called()

                # Should have returned analogues but no inferences
                assert len(analogues) == 2
                assert len(inferences) == 0
                assert len(findings) == 1
                assert "max similarity < 0.7" in findings[0].claim
                assert "skipped inference" in findings[0].logic_chain

    @pytest.mark.asyncio
    async def test_runs_inference_for_quality_analogues(self):
        """Should run SDK inference if at least one analogue has similarity >= 0.7."""
        # Mock get_similar_entities to return quality analogues
        quality_analogues = [
            {"curie": "ANALOGUE:1", "name": "Good Match 1", "similarity": 0.85, "category": "ChemicalEntity"},
            {"curie": "ANALOGUE:2", "name": "Poor Match 2", "similarity": 0.5, "category": "ChemicalEntity"},
        ]

        with patch("kestrel_backend.graph.nodes.cold_start.get_similar_entities") as mock_similar:
            with patch("kestrel_backend.graph.nodes.cold_start.get_entity_connections") as mock_connections:
                with patch("kestrel_backend.graph.nodes.cold_start.query") as mock_query:
                    mock_similar.return_value = quality_analogues
                    mock_connections.return_value = {"edges": [], "summary": "No edges"}

                    # Mock SDK query to return immediately - use side_effect to create fresh generator per call
                    async def mock_query_gen(*args, **kwargs):
                        yield MagicMock(content=[MagicMock(text='{"inferences": []}')])

                    mock_query.side_effect = lambda *a, **kw: mock_query_gen()

                    analogues, inferences, findings, errors = await analyze_cold_start_entity(
                        "TEST:001", "Test Entity", 5
                    )

                    # SDK inference should have been called
                    mock_query.assert_called_once()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_performance_10_sparse_entities():
    """
    Integration test: 10 sparse entities should complete under 5 minutes.

    This test verifies the performance optimizations for Issue #10:
    - Limiting to top 5 entities
    - Batch processing with increased batch size
    - Reduced timeout and increased parallelism
    """
    # Create 10 sparse entities with varying edge counts
    novelty_scores = [
        NoveltyScore(curie=f"TEST:{i}", raw_name=f"Test {i}", edge_count=i, classification="sparse")
        for i in range(1, 11)
    ]

    state: DiscoveryState = {
        "sparse_curies": [f"TEST:{i}" for i in range(1, 11)],
        "cold_start_curies": [],
        "novelty_scores": novelty_scores,
        "resolved_entities": [],
    }

    # Mock external dependencies to avoid actual API calls
    # but still test internal logic and parallelism
    with patch("kestrel_backend.graph.nodes.cold_start.get_similar_entities") as mock_similar:
        with patch("kestrel_backend.graph.nodes.cold_start.get_entity_connections") as mock_connections:
            with patch("kestrel_backend.graph.nodes.cold_start.query") as mock_query:
                # Return quality analogues for testing
                mock_similar.return_value = [
                    {"curie": "ANALOGUE:1", "name": "Match 1", "similarity": 0.9, "category": "ChemicalEntity"}
                ]
                mock_connections.return_value = {"edges": [], "summary": "No edges"}

                # Mock SDK query to simulate processing time - use side_effect for fresh generator per call
                async def mock_query_gen(*args, **kwargs):
                    await asyncio.sleep(0.1)  # Simulate some processing
                    yield MagicMock(content=[MagicMock(text='{"inferences": []}')])

                mock_query.side_effect = lambda *a, **kw: mock_query_gen()

                start = time.time()
                result = await run(state)
                duration = time.time() - start

                # Should complete in reasonable time (much less than 5 minutes)
                # With mocks, should be very fast (< 10 seconds for batch processing)
                assert duration < 10.0, f"Processing took {duration:.1f}s, expected < 10s with mocks"

                # Should have limited to top 5 entities
                assert result["cold_start_skipped_count"] == 5

                # Should have some findings
                assert len(result["cold_start_findings"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
