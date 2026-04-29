"""Tests for pipeline capture infrastructure.

Tests the capture script's ability to record pipeline executions,
serialize state, and produce valid cassette + output files.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from kestrel_backend.assessment.capture import (
    PipelineCapture,
    _serialize_state,
    _serialize_value,
    query_hash,
)


class MockPydanticModel:
    """Minimal mock for frozen Pydantic BaseModel instances."""

    def __init__(self, data: dict):
        self._data = data

    def model_dump(self):
        return self._data


class TestSerializeState:
    """Test state serialization for JSON output."""

    def test_serialize_primitives(self):
        state = {"query": "NAD+", "count": 42, "score": 0.95, "active": True}
        result = _serialize_state(state)
        assert result == state

    def test_serialize_pydantic_model(self):
        model = MockPydanticModel({"name": "NAD+", "curie": "CHEBI:15422"})
        state = {"entity": model}
        result = _serialize_state(state)
        assert result == {"entity": {"name": "NAD+", "curie": "CHEBI:15422"}}

    def test_serialize_list_of_models(self):
        models = [
            MockPydanticModel({"name": "NAD+"}),
            MockPydanticModel({"name": "SIRT1"}),
        ]
        state = {"entities": models}
        result = _serialize_state(state)
        assert result == {"entities": [{"name": "NAD+"}, {"name": "SIRT1"}]}

    def test_serialize_nested_dict(self):
        state = {"analysis": {"phase": "intake", "results": [1, 2, 3]}}
        result = _serialize_state(state)
        assert result == {"analysis": {"phase": "intake", "results": [1, 2, 3]}}

    def test_serialize_none_values(self):
        state = {"data": None, "items": [None, "value"]}
        result = _serialize_state(state)
        assert result == {"data": None, "items": [None, "value"]}

    def test_serialize_empty_state(self):
        assert _serialize_state({}) == {}

    def test_serialize_tuple_to_list(self):
        state = {"pair": ("a", "b")}
        result = _serialize_state(state)
        assert result == {"pair": ["a", "b"]}


class TestQueryHash:
    """Test deterministic query hashing."""

    def test_same_query_same_hash(self):
        assert query_hash("NAD+ and aging") == query_hash("NAD+ and aging")

    def test_different_query_different_hash(self):
        assert query_hash("NAD+ and aging") != query_hash("SIRT1 and longevity")

    def test_hash_length(self):
        assert len(query_hash("test query")) == 12


class TestPipelineCapture:
    """Test the capture workflow."""

    async def test_capture_single_produces_files(self, tmp_path: Path):
        """Capture should produce cassette and output JSON files."""
        capture = PipelineCapture(tmp_path)

        # Mock run_discovery to return a simple state dict
        mock_state = {
            "raw_query": "NAD+ and aging",
            "resolved_entities": [
                MockPydanticModel({"raw_name": "NAD+", "curie": "CHEBI:15422"})
            ],
            "synthesis_report": "Test report",
            "hypotheses": [],
            "errors": [],
        }

        # Patch at the source module since capture_single imports it locally
        with patch(
            "kestrel_backend.graph.runner.run_discovery",
            new_callable=AsyncMock,
            return_value=mock_state,
        ):
            result = await capture.capture_single("NAD+ and aging", run_number=1)

        assert result["error"] is None
        assert result["run_number"] == 1
        assert result["query"] == "NAD+ and aging"

        # Check cassette file exists and is valid JSON
        cassette_path = Path(result["cassette_path"])
        assert cassette_path.exists()
        cassette_data = json.loads(cassette_path.read_text())
        assert "metadata" in cassette_data
        assert "interactions" in cassette_data
        assert cassette_data["metadata"]["query"] == "NAD+ and aging"

        # Check output file exists and is valid JSON
        output_path = Path(result["output_path"])
        assert output_path.exists()
        output_data = json.loads(output_path.read_text())
        assert output_data["raw_query"] == "NAD+ and aging"
        assert output_data["synthesis_report"] == "Test report"
        assert output_data["resolved_entities"] == [{"raw_name": "NAD+", "curie": "CHEBI:15422"}]

    async def test_capture_handles_pipeline_failure(self, tmp_path: Path):
        """Capture should handle pipeline failures gracefully."""
        capture = PipelineCapture(tmp_path)

        with patch(
            "kestrel_backend.graph.runner.run_discovery",
            new_callable=AsyncMock,
            side_effect=RuntimeError("API timeout"),
        ):
            result = await capture.capture_single("failing query", run_number=1)

        assert result["error"] is not None
        assert "API timeout" in result["error"]

        # Output file should contain the error
        output_path = Path(result["output_path"])
        assert output_path.exists()
        output_data = json.loads(output_path.read_text())
        assert "error" in output_data

    async def test_capture_empty_findings(self, tmp_path: Path):
        """Capture should produce valid JSON even with empty findings."""
        capture = PipelineCapture(tmp_path)

        mock_state = {
            "raw_query": "unknown entity xyz",
            "resolved_entities": [],
            "direct_findings": [],
            "hypotheses": [],
            "errors": [],
        }

        with patch(
            "kestrel_backend.graph.runner.run_discovery",
            new_callable=AsyncMock,
            return_value=mock_state,
        ):
            result = await capture.capture_single("unknown entity xyz", run_number=1)

        assert result["error"] is None
        output_data = json.loads(Path(result["output_path"]).read_text())
        assert output_data["resolved_entities"] == []
        assert output_data["hypotheses"] == []

    async def test_capture_batch_runs_all_queries(self, tmp_path: Path):
        """Batch capture should run each query the specified number of times."""
        capture = PipelineCapture(tmp_path)

        mock_state = {"raw_query": "test", "errors": []}
        call_count = 0

        async def mock_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_state

        queries = [
            {"query": "query A", "path_type": "well-characterized"},
            {"query": "query B", "path_type": "sparse"},
        ]

        with patch(
            "kestrel_backend.graph.runner.run_discovery",
            new_callable=AsyncMock,
            side_effect=mock_run,
        ):
            results = await capture.capture_batch(queries, runs_per_query=2)

        assert len(results) == 4  # 2 queries x 2 runs
        assert call_count == 4

        # Check summary file
        summary_path = tmp_path / "capture_summary.json"
        assert summary_path.exists()

    async def test_capture_directory_structure(self, tmp_path: Path):
        """Capture should create proper directory structure."""
        capture = PipelineCapture(tmp_path)

        assert (tmp_path / "cassettes").is_dir()
        assert (tmp_path / "outputs").is_dir()
