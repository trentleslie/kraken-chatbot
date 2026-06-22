"""Build the per-node pipeline performance report (JSON + Markdown).

Pure functions over the accumulated ``DiscoveryState`` plus run metadata — no I/O
(disk emission lives in :mod:`kestrel_backend.run_reports_io`). ``build_report``
returns a JSON-serializable dict; ``render_markdown`` renders it as
Outline-publishable GFM.

Schema (``report_version: 1``):
    run_id, query (full — JSON only), mode, biomapper_env, timestamp, git_sha,
    totals{wall_clock_s, input_tokens, output_tokens, cache_read_tokens,
           cache_creation_tokens, total_tokens, mcp_tool_calls, cost_usd},
    headline{top_bottleneck_node, top_cost_node},
    errors[],
    nodes[{node, status, duration_s, pct_of_total, input_tokens, output_tokens,
           cache_read_tokens, cache_creation_tokens, models[], mcp_tool_calls,
           cost_usd, findings}]

Security (plan Unit 4, R6): the full ``query`` is in the JSON only;
``render_markdown`` emits ``run_id`` and never the raw query, because the
Markdown is the Outline-bound artifact.
"""

from typing import Any

from .. import pricing
from .node_detail_extractors import extract_node_details

REPORT_VERSION = 1

# Canonical pipeline node order (matches builder.py). The terminal `reporting`
# node is intentionally excluded — it does not report on itself.
PIPELINE_NODES = [
    "intake",
    "entity_resolution",
    "triage",
    "direct_kg",
    "cold_start",
    "pathway_enrichment",
    "integration",
    "temporal",
    "hypothesis_extraction",
    "bridge_grounding",
    "literature_grounding",
    "synthesis",
]


def _field(obj: Any, name: str, default: Any = 0) -> Any:
    """Read ``name`` from a ModelUsageRecord (attr) or a plain dict."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _group_usages_by_node(model_usages: list) -> dict[str, list]:
    grouped: dict[str, list] = {}
    for rec in model_usages or []:
        node = _field(rec, "node_name", "") or "unknown"
        grouped.setdefault(node, []).append(rec)
    return grouped


def _node_token_totals(records: list) -> dict[str, int]:
    return {
        "input_tokens": sum(_field(r, "input_tokens", 0) for r in records),
        "output_tokens": sum(_field(r, "output_tokens", 0) for r in records),
        "cache_read_tokens": sum(_field(r, "cache_read_tokens", 0) for r in records),
        "cache_creation_tokens": sum(_field(r, "cache_creation_tokens", 0) for r in records),
        "mcp_tool_calls": sum(_field(r, "mcp_tool_calls", 0) for r in records),
    }


def _node_cost(records: list) -> float | None:
    """Per-node cost; None if every record's model is unknown (else sum of knowns)."""
    costs = [
        pricing.cost_of_record(
            model_name=_field(r, "model_name", "") or "",
            input_tokens=_field(r, "input_tokens", 0),
            output_tokens=_field(r, "output_tokens", 0),
            cache_read_tokens=_field(r, "cache_read_tokens", 0),
            cache_creation_tokens=_field(r, "cache_creation_tokens", 0),
        )
        for r in records
    ]
    known = [c for c in costs if c is not None]
    return sum(known) if known else None


def build_report(state: dict, meta: dict) -> dict:
    """Build the report dict from final state + run metadata.

    ``meta`` keys: ``run_id``, ``query``, ``mode``, ``biomapper_env``,
    ``timestamp``, ``git_sha``, and optionally ``wall_clock_s`` (the exact
    measured run duration; if absent, the sum of per-node durations is used as
    the percentage denominator).
    """
    node_timings: dict[str, float] = state.get("node_timings", {}) or {}
    model_usages = state.get("model_usages", []) or []
    errors = list(state.get("errors", []) or [])
    usages_by_node = _group_usages_by_node(model_usages)

    # Node set: canonical order, then any extras observed in timings/usages.
    seen = set(PIPELINE_NODES)
    extras = [n for n in (set(node_timings) | set(usages_by_node)) if n not in seen]
    ordered_nodes = PIPELINE_NODES + sorted(extras)

    # Denominator for pct_of_total: exact wall-clock if the caller measured it, else
    # the sum of per-node durations. The v1 reporting node has no true elapsed-time
    # source, so it omits wall_clock_s and this falls back to "summed_estimate" — which
    # overstates elapsed time when parallel branches overlap, and makes pct_of_total a
    # share-of-summed-work (always <=100%) rather than share-of-elapsed.
    summed = sum(node_timings.values())
    wall_clock = meta.get("wall_clock_s")
    if not wall_clock or wall_clock <= 0:
        wall_clock = summed
        wall_clock_source = "summed_estimate"
    else:
        wall_clock_source = "measured"
    denom = wall_clock if wall_clock and wall_clock > 0 else None

    nodes: list[dict] = []
    for name in ordered_nodes:
        ran = name in node_timings
        records = usages_by_node.get(name, [])
        if not ran and not records:
            # Node never executed on this run (conditional branch skipped).
            nodes.append({"node": name, "status": "skipped"})
            continue
        duration = node_timings.get(name)
        toks = _node_token_totals(records)
        models = sorted({_field(r, "model_name", "") for r in records if _field(r, "model_name", "")})
        summary, _details = extract_node_details(name, state) if ran else ("", {})
        nodes.append(
            {
                "node": name,
                "status": "ran",
                "duration_s": round(duration, 3) if duration is not None else None,
                "pct_of_total": (
                    round(100.0 * duration / denom, 1)
                    if duration is not None and denom
                    else None
                ),
                **toks,
                "models": models,
                "cost_usd": _node_cost(records),
                "findings": summary,
            }
        )

    tok = _node_token_totals(model_usages)  # same 5 fields, computed once
    totals = {
        "wall_clock_s": round(wall_clock, 3) if wall_clock else 0.0,
        "wall_clock_source": wall_clock_source,  # "measured" | "summed_estimate"
        **tok,
        "cost_usd": round(pricing.estimate_cost(model_usages), 6),
    }
    # total_tokens is billing tokens only; mcp_tool_calls is a call count, not tokens.
    totals["total_tokens"] = (
        tok["input_tokens"]
        + tok["output_tokens"]
        + tok["cache_read_tokens"]
        + tok["cache_creation_tokens"]
    )

    ran_nodes = [n for n in nodes if n["status"] == "ran"]
    top_bottleneck = max(
        (n for n in ran_nodes if n.get("duration_s") is not None),
        key=lambda n: n["duration_s"],
        default=None,
    )
    top_cost = max(
        (n for n in ran_nodes if n.get("cost_usd") is not None),
        key=lambda n: n["cost_usd"],
        default=None,
    )

    return {
        "report_version": REPORT_VERSION,
        "run_id": meta.get("run_id"),
        "query": meta.get("query"),  # full query — JSON only; markdown omits it
        "mode": meta.get("mode"),
        "biomapper_env": meta.get("biomapper_env"),
        "timestamp": meta.get("timestamp"),
        "git_sha": meta.get("git_sha"),
        "totals": totals,
        "headline": {
            "top_bottleneck_node": top_bottleneck["node"] if top_bottleneck else None,
            "top_cost_node": top_cost["node"] if top_cost else None,
        },
        "errors": errors,
        "nodes": nodes,
    }


def _fmt_cost(c: float | None) -> str:
    return f"${c:.4f}" if c is not None else "est. n/a"


def render_markdown(report: dict) -> str:
    """Render the report dict as Outline-publishable GFM. Never emits the raw query."""
    t = report.get("totals", {})
    h = report.get("headline", {})
    lines: list[str] = []
    lines.append("# Pipeline Performance Report")
    lines.append("")
    lines.append(f"- **Run:** `{report.get('run_id')}`")
    lines.append(f"- **Mode:** {report.get('mode')} · **biomapper_env:** {report.get('biomapper_env')}")
    lines.append(f"- **Timestamp:** {report.get('timestamp')} · **commit:** `{report.get('git_sha')}`")
    lines.append(
        f"- **Wall-clock:** {t.get('wall_clock_s')}s · "
        f"**tokens:** {t.get('input_tokens')} in / {t.get('output_tokens')} out "
        f"({t.get('cache_read_tokens')} cache-r, {t.get('cache_creation_tokens')} cache-w) · "
        f"**est. cost:** {_fmt_cost(t.get('cost_usd'))}"
    )
    lines.append(
        f"- **Top bottleneck:** {h.get('top_bottleneck_node') or 'n/a'} · "
        f"**Top cost:** {h.get('top_cost_node') or 'n/a'}"
    )
    lines.append("")
    if t.get("wall_clock_source") == "measured":
        lines.append(
            "> `%` is each node's share of measured wall-clock. Concurrent branches "
            "(`direct_kg` / `cold_start`) overlap, so per-node percentages can sum to >100%."
        )
    else:
        lines.append(
            "> Wall-clock is estimated as the **sum of per-node durations** (no measured "
            "elapsed time available); for concurrent branches this overstates true elapsed "
            "time. `%` is each node's share of that summed time."
        )
    lines.append("")
    lines.append("## Per-node")
    lines.append("")
    lines.append("| Node | Status | Duration (s) | % | In | Out | Cache r/w | Model | MCP | Est. $ | Findings |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")

    # Rows: ran nodes sorted by duration desc, then skipped nodes.
    ran = [n for n in report.get("nodes", []) if n.get("status") == "ran"]
    skipped = [n for n in report.get("nodes", []) if n.get("status") != "ran"]
    ran.sort(key=lambda n: (n.get("duration_s") or 0.0), reverse=True)
    for n in ran + skipped:
        if n.get("status") != "ran":
            lines.append(f"| {n['node']} | skipped | – | – | – | – | – | – | – | – | – |")
            continue
        models = ", ".join(n.get("models", [])) or "–"
        pct = f"{n['pct_of_total']}%" if n.get("pct_of_total") is not None else "–"
        dur = n["duration_s"] if n.get("duration_s") is not None else "–"
        cache = f"{n.get('cache_read_tokens', 0)}/{n.get('cache_creation_tokens', 0)}"
        findings = (n.get("findings") or "").replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {n['node']} | ran | {dur} | {pct} | {n.get('input_tokens', 0)} | "
            f"{n.get('output_tokens', 0)} | {cache} | {models} | "
            f"{n.get('mcp_tool_calls', 0)} | {_fmt_cost(n.get('cost_usd'))} | {findings} |"
        )

    lines.append("")
    lines.append("## Errors")
    lines.append("")
    errs = report.get("errors", [])
    if errs:
        for e in errs:
            lines.append(f"- {str(e).replace(chr(10), ' ')}")
    else:
        lines.append("None.")
    lines.append("")
    return "\n".join(lines)
