"""Assessment runner for the discovery pipeline.

Executes the pipeline against a query dataset in either live or replay mode,
collecting outputs for structural checks and quality scoring.

Live mode runs against real APIs. Replay mode uses the shared cassette module
to serve cached HTTP responses — LLM calls (Claude SDK query()) still execute
live, so each run requires SDK auth and incurs API costs.

Usage:
    python -m kestrel_backend.assessment.runner --mode replay --queries assessment_data/queries.json
    python -m kestrel_backend.assessment.runner --mode live --queries assessment_data/queries.json
"""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from .capture import PipelineCapture, _serialize_state, query_hash
from .cassette import setup_replay

logger = logging.getLogger(__name__)


async def run_single_assessment(
    query: str,
    mode: str = "replay",
    cassettes_dir: Path | None = None,
    query_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the pipeline for a single query and collect output for assessment.

    Args:
        query: The query text to run
        mode: "live" or "replay"
        cassettes_dir: Directory containing cassette files (required for replay mode)
        query_metadata: Optional metadata about the query (path_type, expected_entities, etc.)

    Returns:
        Result dict with query, state output, timing, mode, and any errors.
    """
    from ..graph.runner import run_discovery

    qhash = query_hash(query)
    start_time = time.time()
    state = None
    error = None

    try:
        if mode == "replay":
            if cassettes_dir is None:
                raise ValueError("cassettes_dir required for replay mode")

            # Find the cassette file for this query (use run 1 as default replay source)
            cassette_path = cassettes_dir / f"{qhash}_run1.json"
            if not cassette_path.exists():
                # Try to find any cassette for this query
                candidates = sorted(cassettes_dir.glob(f"{qhash}_run*.json"))
                if not candidates:
                    return {
                        "query": query,
                        "query_hash": qhash,
                        "mode": mode,
                        "duration_seconds": 0,
                        "state": None,
                        "error": f"No cassette found for query hash {qhash}",
                        "metadata": query_metadata,
                    }
                cassette_path = candidates[0]

            router, replayer = setup_replay(cassette_path)
            with router:
                state = await run_discovery(query)

        else:  # live mode
            state = await run_discovery(query)

    except Exception as e:
        error = str(e)
        logger.error("Assessment run failed for query '%s': %s", query[:50], e)

    duration = time.time() - start_time

    result = {
        "query": query,
        "query_hash": qhash,
        "mode": mode,
        "duration_seconds": round(duration, 2),
        "state": _serialize_state(state) if state is not None else None,
        "error": error,
        "metadata": query_metadata,
    }

    logger.info(
        "Assessment run complete: query='%s' mode=%s duration=%.1fs error=%s",
        query[:50], mode, duration, error,
    )

    return result


async def run_assessment(
    queries_path: str | Path,
    mode: str = "replay",
    cassettes_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run assessment across a full query dataset.

    Args:
        queries_path: Path to queries.json
        mode: "live" or "replay"
        cassettes_dir: Directory containing cassette files (defaults to assessment_data/cassettes)

    Returns:
        Assessment results dict with per-query results and aggregate summary.
    """
    queries_path = Path(queries_path)
    queries = json.loads(queries_path.read_text())

    if cassettes_dir is None:
        cassettes_dir = queries_path.parent / "cassettes"
    else:
        cassettes_dir = Path(cassettes_dir)

    results = []
    errors = 0
    total_duration = 0

    for i, q in enumerate(queries):
        query_text = q["query"]
        metadata = {k: v for k, v in q.items() if k != "query"}

        logger.info("Running assessment %d/%d: '%s'", i + 1, len(queries), query_text[:50])

        result = await run_single_assessment(
            query=query_text,
            mode=mode,
            cassettes_dir=cassettes_dir,
            query_metadata=metadata,
        )
        results.append(result)

        if result["error"]:
            errors += 1
        total_duration += result["duration_seconds"]

    summary = {
        "total_queries": len(queries),
        "successful": len(queries) - errors,
        "failed": errors,
        "total_duration_seconds": round(total_duration, 2),
        "mode": mode,
    }

    return {
        "summary": summary,
        "results": results,
    }


async def main():
    """CLI entry point for assessment runner."""
    parser = argparse.ArgumentParser(description="Run pipeline assessment")
    parser.add_argument("--queries", type=str, required=True, help="Path to queries.json")
    parser.add_argument("--mode", type=str, default="replay", choices=["live", "replay"],
                        help="Execution mode (default: replay)")
    parser.add_argument("--cassettes-dir", type=str, default=None,
                        help="Cassettes directory (default: assessment_data/cassettes)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for results JSON (default: stdout)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    results = await run_assessment(
        queries_path=args.queries,
        mode=args.mode,
        cassettes_dir=args.cassettes_dir,
    )

    output_json = json.dumps(results, indent=2, default=str)

    if args.output:
        Path(args.output).write_text(output_json)
        print(f"Results written to {args.output}")
    else:
        print(output_json)


if __name__ == "__main__":
    asyncio.run(main())
