"""Baseline capture infrastructure for the discovery pipeline.

Records pipeline outputs and all external HTTP interactions (Kestrel KG,
OpenAlex, Semantic Scholar, Exa, PubMed) for later replay during assessment
runs. Uses the shared cassette module for transport-level HTTP interception.

Usage:
    python -m kestrel_backend.assessment.capture --query "NAD+ and aging" --output-dir assessment_data
    python -m kestrel_backend.assessment.capture --queries assessment_data/queries.json --runs 5
"""

import argparse
import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

from .cassette import CassetteRecorder

logger = logging.getLogger(__name__)


def _serialize_state(state: dict[str, Any]) -> dict[str, Any]:
    """Serialize a DiscoveryState dict to JSON-compatible format.

    Handles Pydantic models (frozen BaseModel instances) by calling model_dump().
    """
    serialized = {}
    for key, value in state.items():
        serialized[key] = _serialize_value(value)
    return serialized


def _serialize_value(value: Any) -> Any:
    """Recursively serialize a value to JSON-compatible format."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_serialize_value(item) for item in value]
    # Fallback: convert to string
    return str(value)


def query_hash(query: str) -> str:
    """Generate a short deterministic hash for a query string."""
    return hashlib.md5(query.encode()).hexdigest()[:12]


class PipelineCapture:
    """Captures pipeline execution results and HTTP interactions."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.cassettes_dir = output_dir / "cassettes"
        self.outputs_dir = output_dir / "outputs"
        self.cassettes_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    async def capture_single(
        self,
        query: str,
        run_number: int = 1,
        conversation_history: list[tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Run the pipeline for a single query, recording HTTP interactions.

        Returns a result dict with query, timing, output path, and cassette path.
        """
        qhash = query_hash(query)
        cassette_path = self.cassettes_dir / f"{qhash}_run{run_number}.json"
        output_path = self.outputs_dir / f"{qhash}_run{run_number}.json"

        recorder = CassetteRecorder()
        start_time = time.time()
        error = None
        state = None

        try:
            # Import here to avoid circular imports and heavy module loading at import time
            from ..graph.runner import run_discovery

            # We need to intercept httpx calls during pipeline execution.
            # respx works at the transport level, but for recording we need a
            # passthrough approach: make real requests and capture the responses.
            # We monkey-patch httpx.AsyncClient to wrap the send method.
            original_send = httpx.AsyncClient.send

            async def recording_send(self_client, request, *args, **kwargs):
                response = await original_send(self_client, request, *args, **kwargs)
                # Only record external API calls, not localhost
                if not str(request.url).startswith(("http://127.0.0.1", "http://localhost")):
                    recorder.record_interaction(request, response)
                return response

            httpx.AsyncClient.send = recording_send
            try:
                state = await run_discovery(query, conversation_history)
            finally:
                httpx.AsyncClient.send = original_send

        except Exception as e:
            error = str(e)
            logger.error("Pipeline execution failed for query '%s': %s", query[:50], e)

        duration = time.time() - start_time

        # Save cassette
        metadata = {
            "query": query,
            "query_hash": qhash,
            "run_number": run_number,
            "duration_seconds": round(duration, 2),
            "interaction_count": recorder.interaction_count,
            "error": error,
        }
        recorder.save(cassette_path, metadata=metadata)

        # Save pipeline output
        if state is not None:
            serialized = _serialize_state(state)
            output_path.write_text(json.dumps(serialized, indent=2, default=str))
            logger.info(
                "Captured query '%s' run %d: %d interactions, %.1fs",
                query[:50], run_number, recorder.interaction_count, duration,
            )
        else:
            output_path.write_text(json.dumps({"error": error}, indent=2))

        return {
            "query": query,
            "query_hash": qhash,
            "run_number": run_number,
            "duration_seconds": round(duration, 2),
            "interaction_count": recorder.interaction_count,
            "cassette_path": str(cassette_path),
            "output_path": str(output_path),
            "error": error,
        }

    async def capture_batch(
        self,
        queries: list[dict[str, Any]],
        runs_per_query: int = 5,
    ) -> list[dict[str, Any]]:
        """Run batch capture: all queries x N runs.

        Args:
            queries: List of query dicts with at least a "query" key
            runs_per_query: Number of runs per query for variance measurement

        Returns:
            List of result dicts, one per (query, run) pair
        """
        results = []
        total = len(queries) * runs_per_query
        completed = 0

        for q in queries:
            query_text = q["query"]
            for run_num in range(1, runs_per_query + 1):
                completed += 1
                logger.info(
                    "Capturing %d/%d: query='%s' run=%d",
                    completed, total, query_text[:50], run_num,
                )
                result = await self.capture_single(query_text, run_number=run_num)
                result["metadata"] = {k: v for k, v in q.items() if k != "query"}
                results.append(result)

        # Save batch summary
        summary_path = self.output_dir / "capture_summary.json"
        summary_path.write_text(json.dumps(results, indent=2, default=str))
        logger.info("Batch capture complete: %d total runs", len(results))

        return results


async def main():
    """CLI entry point for capture."""
    parser = argparse.ArgumentParser(description="Capture pipeline baselines")
    parser.add_argument("--query", type=str, help="Single query to capture")
    parser.add_argument("--queries", type=str, help="Path to queries.json for batch mode")
    parser.add_argument("--runs", type=int, default=5, help="Runs per query (default: 5)")
    parser.add_argument(
        "--output-dir", type=str, default="assessment_data",
        help="Output directory (default: assessment_data)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    capture = PipelineCapture(Path(args.output_dir))

    if args.query:
        result = await capture.capture_single(args.query)
        print(json.dumps(result, indent=2))
    elif args.queries:
        queries_path = Path(args.queries)
        queries = json.loads(queries_path.read_text())
        results = await capture.capture_batch(queries, runs_per_query=args.runs)
        print(f"Captured {len(results)} runs. Summary: {capture.output_dir / 'capture_summary.json'}")
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
