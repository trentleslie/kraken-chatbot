"""Tests for assessment runner."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from kestrel_backend.assessment.cassette import CassetteRecorder
from kestrel_backend.assessment.runner import (
    run_assessment,
    run_single_assessment,
)


MOCK_STATE = {
    "raw_query": "NAD+ and aging",
    "resolved_entities": [],
    "direct_findings": [],
    "hypotheses": [],
    "errors": [],
}

SSE_RESPONSE = 'event: message\ndata: {"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"test"}]}}\n'


def _create_cassette(tmp_path: Path, query: str, query_hash: str) -> Path:
    """Create a minimal cassette file for replay testing."""
    recorder = CassetteRecorder()

    import httpx
    request = httpx.Request(
        "POST", "https://kestrel.nathanpricelab.com/mcp",
        json={"jsonrpc": "2.0", "method": "tools/call", "id": 1},
    )
    response = httpx.Response(status_code=200, text=SSE_RESPONSE)
    recorder.record_interaction(request, response)

    cassettes_dir = tmp_path / "cassettes"
    cassettes_dir.mkdir(parents=True, exist_ok=True)
    cassette_path = cassettes_dir / f"{query_hash}_run1.json"
    recorder.save(cassette_path, metadata={"query": query})

    return cassettes_dir


class TestRunSingleAssessment:
    """Test single query assessment runs."""

    async def test_live_mode_produces_result(self, tmp_path: Path):
        """Live mode should execute pipeline and return result dict."""
        with patch(
            "kestrel_backend.graph.runner.run_discovery",
            new_callable=AsyncMock,
            return_value=MOCK_STATE,
        ):
            result = await run_single_assessment(
                query="NAD+ and aging",
                mode="live",
            )

        assert result["query"] == "NAD+ and aging"
        assert result["mode"] == "live"
        assert result["error"] is None
        assert result["state"] is not None
        assert result["duration_seconds"] >= 0

    async def test_replay_mode_with_cassette(self, tmp_path: Path):
        """Replay mode should serve cached responses from cassette."""
        from kestrel_backend.assessment.capture import query_hash
        qhash = query_hash("NAD+ and aging")
        cassettes_dir = _create_cassette(tmp_path, "NAD+ and aging", qhash)

        with patch(
            "kestrel_backend.graph.runner.run_discovery",
            new_callable=AsyncMock,
            return_value=MOCK_STATE,
        ):
            result = await run_single_assessment(
                query="NAD+ and aging",
                mode="replay",
                cassettes_dir=cassettes_dir,
            )

        assert result["mode"] == "replay"
        assert result["error"] is None
        assert result["state"] is not None

    async def test_replay_mode_missing_cassette(self, tmp_path: Path):
        """Missing cassette should return error, not raise."""
        cassettes_dir = tmp_path / "cassettes"
        cassettes_dir.mkdir(parents=True, exist_ok=True)

        result = await run_single_assessment(
            query="unknown query",
            mode="replay",
            cassettes_dir=cassettes_dir,
        )

        assert result["error"] is not None
        assert "No cassette found" in result["error"]
        assert result["state"] is None

    async def test_replay_mode_requires_cassettes_dir(self):
        """Replay mode without cassettes_dir should raise ValueError."""
        with patch(
            "kestrel_backend.graph.runner.run_discovery",
            new_callable=AsyncMock,
        ):
            result = await run_single_assessment(
                query="test",
                mode="replay",
                cassettes_dir=None,
            )

        assert result["error"] is not None
        assert "cassettes_dir required" in result["error"]

    async def test_pipeline_failure_captured(self, tmp_path: Path):
        """Pipeline execution failure should be captured, not raised."""
        with patch(
            "kestrel_backend.graph.runner.run_discovery",
            new_callable=AsyncMock,
            side_effect=RuntimeError("SDK auth expired"),
        ):
            result = await run_single_assessment(
                query="failing query",
                mode="live",
            )

        assert result["error"] is not None
        assert "SDK auth expired" in result["error"]
        assert result["state"] is None

    async def test_metadata_passed_through(self, tmp_path: Path):
        """Query metadata should be included in result."""
        with patch(
            "kestrel_backend.graph.runner.run_discovery",
            new_callable=AsyncMock,
            return_value=MOCK_STATE,
        ):
            result = await run_single_assessment(
                query="test",
                mode="live",
                query_metadata={"path_type": "well-characterized"},
            )

        assert result["metadata"]["path_type"] == "well-characterized"


class TestRunAssessment:
    """Test full assessment runs across query dataset."""

    async def test_batch_assessment(self, tmp_path: Path):
        """Batch assessment should run all queries and produce summary."""
        queries = [
            {"query": "query A", "path_type": "well-characterized"},
            {"query": "query B", "path_type": "sparse"},
        ]
        queries_path = tmp_path / "queries.json"
        queries_path.write_text(json.dumps(queries))

        # Create cassettes for both queries
        from kestrel_backend.assessment.capture import query_hash
        for q in queries:
            _create_cassette(tmp_path, q["query"], query_hash(q["query"]))

        with patch(
            "kestrel_backend.graph.runner.run_discovery",
            new_callable=AsyncMock,
            return_value=MOCK_STATE,
        ):
            results = await run_assessment(
                queries_path=queries_path,
                mode="replay",
                cassettes_dir=tmp_path / "cassettes",
            )

        assert results["summary"]["total_queries"] == 2
        assert results["summary"]["successful"] == 2
        assert results["summary"]["failed"] == 0
        assert len(results["results"]) == 2

    async def test_batch_with_failure(self, tmp_path: Path):
        """Batch should continue after individual query failures."""
        queries = [
            {"query": "good query"},
            {"query": "bad query"},
        ]
        queries_path = tmp_path / "queries.json"
        queries_path.write_text(json.dumps(queries))

        call_count = 0

        async def mock_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("API error")
            return MOCK_STATE

        with patch(
            "kestrel_backend.graph.runner.run_discovery",
            new_callable=AsyncMock,
            side_effect=mock_run,
        ):
            results = await run_assessment(
                queries_path=queries_path,
                mode="live",
            )

        assert results["summary"]["successful"] == 1
        assert results["summary"]["failed"] == 1
        assert results["results"][0]["error"] is None
        assert results["results"][1]["error"] is not None
