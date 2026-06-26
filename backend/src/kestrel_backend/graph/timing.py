"""Per-node timing wrapper for the discovery graph.

Wraps each LangGraph node so its wall-clock duration is recorded into
``state['node_timings']`` (merged across nodes via the ``merge_node_timings``
reducer in :mod:`state`). Because the wrapper lives in the compiled graph, it
captures timing on every entry path (WebSocket, ``run_discovery``, LangGraph
Studio) and is concurrency-safe — each invocation writes into the per-run state,
not a shared handler instance.

Defensiveness (plan Unit 1, R4): the wrapper runs on every node, *outside* the
reporting node's try/except. A failure in the wrapper's own timing/merge logic
must never propagate into the pipeline — the wrapped node's result is returned
unchanged. The wrapped node call itself is *not* guarded: a node raising is the
node's own error and must surface as before.
"""

import functools
import logging
import time
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

NodeFn = Callable[[dict], Awaitable[Any]]


def _attach_timing(result: Any, name: str, duration: float) -> dict:
    """Return the node's result dict with ``node_timings`` for this node added.

    Shallow-copies a dict result so reducer-bound values (lists of frozen
    ``ModelUsageRecord`` / ``Finding``) pass through unchanged — never deep-copied
    or reconstructed. ``None`` / non-dict returns yield a dict carrying only the
    timing (LangGraph node returns are dicts or ``None``; a non-dict is already an
    invalid state update, so we surface timing rather than crash).
    """
    if isinstance(result, dict):
        merged = dict(result)
    else:
        if result is not None:
            # A non-dict return is an invalid state update — we cannot merge it, so
            # the node's output is dropped. Log it: silence would make a node that
            # accidentally returns a non-dict a zero-signal correctness landmine.
            logger.warning(
                "timed_node(%s): node returned non-dict %s; state update dropped",
                name,
                type(result).__name__,
            )
        merged = {}
    merged["node_timings"] = {name: duration}
    return merged


def timed_node(name: str, fn: NodeFn) -> NodeFn:
    """Wrap a graph node ``fn`` so it records its wall-clock duration.

    One duration per node execution (the node's ``await`` wall-clock). For the
    parallel ``direct_kg``/``cold_start`` superstep these overlap, so summed
    per-node durations can exceed total wall-clock — total wall-clock is exact,
    per-node is approximate under concurrency (surfaced as a caveat in the report).
    """

    @functools.wraps(fn)
    async def wrapped(state: dict) -> Any:
        start = time.time()
        result = await fn(state)  # node errors surface unchanged (not guarded)
        try:
            return _attach_timing(result, name, time.time() - start)
        except Exception:  # noqa: BLE001 - timing must never break the pipeline
            logger.warning(
                "timed_node(%s): timing/merge failed; passing node result through unchanged",
                name,
                exc_info=True,
            )
            return result

    # Keep functools.wraps' metadata but give the wrapper a distinct, traceable name.
    wrapped.__name__ = f"timed_{name}"
    return wrapped
