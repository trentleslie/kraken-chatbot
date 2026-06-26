"""Unit 4: report builder (state -> JSON dict + Markdown)."""

from kestrel_backend.graph import performance_report as pr
from kestrel_backend.graph.state import ModelUsageRecord


def _usage(node, model="claude-sonnet-4-6", **kw):
    return ModelUsageRecord(model_name=model, node_name=node, **kw)


def _meta(**kw):
    base = {
        "run_id": "abc123",
        "query": "SECRET PATIENT QUERY",
        "mode": "pipeline",
        "biomapper_env": "production",
        "timestamp": "20260622T120000Z",
        "git_sha": "deadbeef",
    }
    base.update(kw)
    return base


def test_happy_path_serial_nodes():
    state = {
        "node_timings": {"intake": 1.0, "synthesis": 3.0},
        "model_usages": [
            _usage("intake", input_tokens=100, output_tokens=10),
            _usage("synthesis", input_tokens=1000, output_tokens=500, mcp_tool_calls=2),
        ],
        "errors": [],
    }
    report = pr.build_report(state, _meta(wall_clock_s=4.0))

    by_node = {n["node"]: n for n in report["nodes"]}
    assert by_node["intake"]["status"] == "ran"
    assert by_node["intake"]["pct_of_total"] == 25.0
    assert by_node["synthesis"]["pct_of_total"] == 75.0
    assert by_node["synthesis"]["input_tokens"] == 1000
    assert by_node["synthesis"]["mcp_tool_calls"] == 2
    # Totals
    assert report["totals"]["input_tokens"] == 1100
    assert report["totals"]["output_tokens"] == 510
    assert report["totals"]["wall_clock_s"] == 4.0
    # Headline
    assert report["headline"]["top_bottleneck_node"] == "synthesis"
    assert report["headline"]["top_cost_node"] == "synthesis"


def test_conditional_node_skipped():
    state = {"node_timings": {"intake": 1.0}, "model_usages": [], "errors": []}
    report = pr.build_report(state, _meta())
    by_node = {n["node"]: n for n in report["nodes"]}
    # cold_start never ran on this path
    assert by_node["cold_start"]["status"] == "skipped"
    assert "duration_s" not in by_node["cold_start"]


def test_node_with_timing_but_no_usage():
    state = {"node_timings": {"triage": 0.5}, "model_usages": [], "errors": []}
    report = pr.build_report(state, _meta())
    triage = next(n for n in report["nodes"] if n["node"] == "triage")
    assert triage["status"] == "ran"
    assert triage["input_tokens"] == 0
    assert triage["cost_usd"] is None  # no usage records -> no cost


def test_unknown_model_cost_none_and_renders_na():
    state = {
        "node_timings": {"synthesis": 1.0},
        "model_usages": [_usage("synthesis", model="gpt-4o", input_tokens=1000)],
        "errors": [],
    }
    report = pr.build_report(state, _meta())
    synth = next(n for n in report["nodes"] if n["node"] == "synthesis")
    assert synth["cost_usd"] is None
    md = pr.render_markdown(report)
    assert "est. n/a" in md  # rendered, not crashed


def test_concurrency_pct_can_exceed_100_no_crash():
    # Two nodes overlapping the same 2s wall-clock window: durations sum to 4 but
    # wall-clock is 2 -> summed pct = 200%. Must not crash or clamp wrongly.
    state = {
        "node_timings": {"direct_kg": 2.0, "cold_start": 2.0},
        "model_usages": [],
        "errors": [],
    }
    report = pr.build_report(state, _meta(wall_clock_s=2.0))
    by_node = {n["node"]: n for n in report["nodes"]}
    assert by_node["direct_kg"]["pct_of_total"] == 100.0
    assert by_node["cold_start"]["pct_of_total"] == 100.0  # sums to 200%
    md = pr.render_markdown(report)
    assert "overlap" in md.lower()  # caveat present


def test_markdown_omits_raw_query_shows_run_id():
    state = {"node_timings": {"intake": 1.0}, "model_usages": [], "errors": []}
    report = pr.build_report(state, _meta(query="SECRET PATIENT QUERY", run_id="rid999"))
    # JSON keeps the full query for reproducibility...
    assert report["query"] == "SECRET PATIENT QUERY"
    # ...the Markdown (Outline-bound) must not.
    md = pr.render_markdown(report)
    assert "SECRET PATIENT QUERY" not in md
    assert "rid999" in md


def test_markdown_is_valid_gfm_table():
    state = {"node_timings": {"intake": 1.0}, "model_usages": [], "errors": []}
    md = pr.render_markdown(pr.build_report(state, _meta()))
    assert "| Node | Status |" in md
    assert "|---|---|---|---|---|---|---|---|---|---|---|" in md
    assert md.count("\n## ") >= 2  # Per-node + Errors sections


def test_errors_rendered_at_run_level():
    state = {
        "node_timings": {"synthesis": 1.0},
        "model_usages": [],
        "errors": ["synthesis: fell back to deterministic report"],
    }
    md = pr.render_markdown(pr.build_report(state, _meta()))
    assert "synthesis: fell back to deterministic report" in md


def test_no_errors_renders_none():
    state = {"node_timings": {"intake": 1.0}, "model_usages": [], "errors": []}
    md = pr.render_markdown(pr.build_report(state, _meta()))
    assert "## Errors\n\nNone." in md


def test_node_with_usage_but_no_timing():
    # Reverse of test_node_with_timing_but_no_usage: a node with model_usages but no
    # node_timings entry (the extras/usages_by_node path) is reported as ran.
    state = {
        "node_timings": {"intake": 1.0},
        "model_usages": [_usage("synthesis", input_tokens=500)],
        "errors": [],
    }
    report = pr.build_report(state, _meta())
    synth = next(n for n in report["nodes"] if n["node"] == "synthesis")
    assert synth["status"] == "ran"
    assert synth["duration_s"] is None
    assert synth["input_tokens"] == 500


def test_node_cost_mixed_known_and_unknown_models():
    # A node with one known + one unknown model -> cost is the sum of the known only.
    state = {
        "node_timings": {"synthesis": 1.0},
        "model_usages": [
            _usage("synthesis", model="claude-sonnet-4-6", input_tokens=1_000_000),  # $3
            _usage("synthesis", model="gpt-4o", input_tokens=1_000_000),  # unknown -> skipped
        ],
        "errors": [],
    }
    report = pr.build_report(state, _meta())
    synth = next(n for n in report["nodes"] if n["node"] == "synthesis")
    assert synth["cost_usd"] == 3.0  # known only, not None, not 0


def test_summed_estimate_when_wall_clock_absent():
    # The v1 reporting-node path: no wall_clock_s -> summed estimate, pct sums to 100%,
    # and the markdown caveat reflects the estimate.
    state = {
        "node_timings": {"intake": 1.0, "synthesis": 3.0},
        "model_usages": [],
        "errors": [],
    }
    report = pr.build_report(state, _meta())  # no wall_clock_s
    assert report["totals"]["wall_clock_source"] == "summed_estimate"
    assert report["totals"]["wall_clock_s"] == 4.0
    pcts = [n["pct_of_total"] for n in report["nodes"] if n.get("pct_of_total") is not None]
    assert sum(pcts) == 100.0  # share-of-summed, never exceeds 100
    md = pr.render_markdown(report)
    assert "sum of per-node durations" in md


def test_measured_caveat_in_register_voice():
    # Unit 3 / register: affirmative check that the rewrite happened (not merely that an
    # em-dash is absent, which the prior text already satisfied -> a permanently-green gate).
    state = {"node_timings": {"direct_kg": 2.0, "cold_start": 2.0}, "model_usages": [], "errors": []}
    md = pr.render_markdown(pr.build_report(state, _meta(wall_clock_s=2.0)))
    assert "reports each node's share of measured wall-clock" in md   # new register wording
    assert "`%` is each node's share" not in md                       # old wording is gone
    assert "—" not in md                                              # secondary guard: no em-dash


def test_summed_estimate_caveat_in_register_voice():
    state = {"node_timings": {"intake": 1.0, "synthesis": 3.0}, "model_usages": [], "errors": []}
    md = pr.render_markdown(pr.build_report(state, _meta()))  # no wall_clock_s -> summed estimate
    # Register rewrite is present and still communicates the summed-estimate meaning.
    assert "sum of per-node durations" in md
    assert "overstates true elapsed time" in md
    assert "reports each node's share of that summed total" in md
    assert "—" not in md


def _ctx_stats(module_mode=True):
    s = {
        "context_chars": 94225, "context_est_tokens": 26921,
        "max_context_chars": 350000, "char_budget_pct": 26.9,
        "window_tokens": 200000, "window_pct": 13.5,
        "module_mode": module_mode, "module_mode_threshold": 5, "distinct_entities": 24,
        "sections": {"findings": {"shown": 150, "total": 2433, "elided": 2283}},
        "literature": {"attached": 33, "total": 66},
    }
    if module_mode:
        s["sections"]["diseases"] = {"shown": 30, "total": 40, "elided": 10}
        s["sections"]["pathways"] = {"shown": 30, "total": 35, "elided": 5}
        s["sections"]["member_table"] = {"shown": 50, "total": 217, "elided": 167}
    return s


def test_context_management_module_mode():
    state = {"node_timings": {"synthesis": 1.0}, "model_usages": [], "errors": [],
             "synthesis_context_stats": _ctx_stats(module_mode=True)}
    report = pr.build_report(state, _meta())
    assert report["context_stats"]["sections"]["member_table"]["elided"] == 167  # JSON retains full structure
    md = pr.render_markdown(report)
    assert "## Context management" in md
    assert "| Findings | 150 | 2433 | 2283 |" in md
    assert "| Member table | 50 | 217 | 167 |" in md
    assert "| Diseases (recurrence) | 30 | 40 | 10 |" in md
    assert "33 of 66 hypotheses carry attached literature" in md
    assert "module-aware aggregation" in md
    assert "26.9% of the char budget" in md
    assert "13.5% of the 200K-token window" in md
    assert "—" not in md                                  # register: no em-dash in the section prose
    assert "## Context management" in md and "Context-management" not in md  # heading has no dash


def test_context_management_per_entity_mode():
    state = {"node_timings": {"synthesis": 1.0}, "model_usages": [], "errors": [],
             "synthesis_context_stats": _ctx_stats(module_mode=False)}
    md = pr.render_markdown(pr.build_report(state, _meta()))
    assert "## Context management" in md
    assert "| Findings | 150 | 2433 | 2283 |" in md
    assert "Diseases (recurrence)" not in md              # uncapped in per-entity mode → no row
    assert "Member table" not in md
    assert "per-entity" in md.lower()
    assert "uncapped" in md.lower()


def test_context_management_absent_degrades_gracefully():
    state = {"node_timings": {"synthesis": 1.0}, "model_usages": [], "errors": []}  # no stats
    report = pr.build_report(state, _meta())
    assert report["context_stats"] is None
    md = pr.render_markdown(report)                        # must not crash
    assert "## Context management" not in md
    assert "## Errors" in md                               # rest of the report intact


def test_report_is_json_serializable():
    import json

    state = {
        "node_timings": {"synthesis": 1.0},
        "model_usages": [_usage("synthesis", input_tokens=10)],
        "errors": ["x"],
    }
    report = pr.build_report(state, _meta())
    json.dumps(report)  # must not raise (no Pydantic objects leak into the dict)
