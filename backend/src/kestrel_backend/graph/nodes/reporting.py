"""Terminal reporting node: emit the per-node performance report to disk.

Wired as ``synthesis -> reporting -> END``, so it runs last on every entry path
(WebSocket, ``run_discovery``, LangGraph Studio) — a runner-level reporter would
miss Studio, which invokes the compiled graph directly.

Fail-safe (plan R4): the entire body is wrapped. On any failure it logs a
warning and returns ``{}`` — a reporting failure can never break or delay a
user-facing run. The reporter is read-only over state plus disk I/O.

The terminal ``reporting`` node is intentionally absent from
``NODE_STATUS_MESSAGES`` (so it stays invisible to the WS stream in v1) and is
excluded from its own report: the ``timed_node`` wrapper records
``node_timings["reporting"]`` only *after* this function returns, so the state
this function reads never contains it.
"""

import asyncio
import logging
from typing import Any

from ... import run_reports_io
from ..performance_report import build_report, render_markdown
from ..state import DiscoveryState

logger = logging.getLogger(__name__)


def _build_and_write(state: DiscoveryState, run_id: str) -> str:
    """Build the report and write it to disk. Pure-blocking; runs in a worker thread.

    Every blocking step lives here — ``get_git_sha`` (a subprocess with a 5s timeout,
    blocking on its first uncached call after each restart), ``json.dumps`` of the full
    state, and the file I/O + prune — so none of it stalls the asyncio event loop.
    """
    meta = {
        "run_id": run_id,
        "query": state.get("raw_query"),
        "mode": "pipeline",
        "biomapper_env": state.get("biomapper_env"),
        "timestamp": run_reports_io.utc_timestamp(),
        "git_sha": run_reports_io.get_git_sha(),
    }
    report = build_report(state, meta)
    markdown = render_markdown(report)
    result = run_reports_io.write_report(report, markdown, run_id=run_id)
    return str(result["json"])


async def run(state: DiscoveryState) -> dict[str, Any]:
    try:
        run_id = run_reports_io.new_run_id()
        # Offload ALL blocking work (git SHA lookup + report build + json.dumps + disk
        # write + prune) to a thread so it never stalls the event loop between
        # synthesis-complete and the final WebSocket message on the prod request path.
        report_path = await asyncio.to_thread(_build_and_write, state, run_id)
        logger.info("Performance report written: run_id=%s", run_id)
        return {"report_path": report_path}
    except Exception:  # noqa: BLE001 - reporting must never break a user-facing run (R4)
        logger.warning("Performance report failed; pipeline result unaffected", exc_info=True)
        return {}
