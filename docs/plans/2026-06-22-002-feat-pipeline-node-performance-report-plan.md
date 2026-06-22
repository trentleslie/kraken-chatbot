---
title: "feat: Per-node pipeline performance report (disk-only v1)"
type: feat
status: active
date: 2026-06-22
deepened: 2026-06-22
origin: docs/brainstorms/2026-06-22-pipeline-node-performance-report-requirements.md
---

# feat: Per-node pipeline performance report (disk-only v1)

## Overview

Auto-generate a per-node performance report at the end of **every** discovery pipeline run, written to disk as a JSON artifact plus a human-readable Markdown summary. The report attributes **time, tokens, estimated cost, output counts, and errors** to individual nodes so that bottlenecks (e.g. `pathway_enrichment` ~307s, the synthesis-overflow wall) and cost are visible per run without instrumenting each harness by hand.

The data is ~70% latent already: `main.py` accumulates `model_usages`/`errors`, `node_detail_extractors.py` extracts per-node counts, and `agent.py` already computes a token→$ estimate. The work is to lift that aggregation into a **shared, surface-agnostic reporter** that runs as a **graph-terminal node** (so it fires on the WebSocket path, `run_discovery`, and LangGraph Studio alike), add exact per-node timing, generalize cost to a per-model map, and emit durable artifacts fail-safely.

v1 is **disk-only**. The `performance_report` WebSocket event + frontend renderer are deferred to v2 (see origin: scope decision).

## Problem Frame

When a discovery run is slow, expensive, or degrades, there is no single durable artifact attributing time/tokens/cost/failures to individual nodes (see origin: Problem). The Brown eval harness hand-rolls a `node_timeline` for one harness only. Operators currently SSH for logs or read Langfuse call-by-call. The report's real job is **node-level bottleneck + cost + failure attribution**, emitted automatically.

This sits on the **prod request path** (Lightsail), so the reporter must be fail-safe (never break a user-facing run) and must not grow disk unbounded.

## Requirements Trace

- **R1.** Every discovery run writes a JSON + Markdown report pair to disk by default, no flag required (origin AC1; auto-save SOP).
- **R2.** Per-node exact duration is captured for every node that ran, on all entry paths (origin AC2, Open item #1, #3).
- **R3.** Per-node tokens (in/out/cache), model, MCP tool calls, and estimated cost appear and match `model_usages`; counts via `extract_node_details` (origin AC3).
- **R4.** Reporting failures never break or delay a run — proven by test (origin AC4; fail-safe).
- **R5.** A retention bound ships with the writer; `run_reports/` is mode 700, artifacts mode 600 (origin AC5, Delivery).
- **R6.** Query text handling: the disk **JSON** carries the full query (600 perms) for reproducibility; the **Markdown** (the Outline-bound artifact) never prints the raw query — it shows an opaque random `run_id` generated at run start and stored only in the disk JSON (not derivable, never sent to Langfuse/DB). Prod-run reports never auto-published. Full keyed-token/two-destination redaction remains v2 (origin Security decision; AC6).
- **R7.** Markdown renders cleanly as Outline-publishable GFM (origin AC7).
- **R8.** The Brown harness's bespoke `node_timeline` can cross-check against / be replaced by the shared reporter (origin AC8).
- **R9.** Report header pins `git_sha`, query, mode, `biomapper_env`, timestamp for reproducibility (origin AC9).

## Scope Boundaries

- Not replacing Langfuse tracing (LLM-call-level observability stays in Langfuse). The reporter derives cost from the **same** `model_usages` source so the two never diverge.
- Not a real-time dashboard; end-of-run artifact only.
- Not a billing system of record — `$` is an estimate.
- v1 reports errors at **run-level** + per-node ran/skipped status. Per-node error *rows* require node-tagged structured errors — not available from final state without touching nodes — and are deferred (see below).

### Deferred to Separate Tasks

- **WebSocket `performance_report` event + React renderer** (origin: v1 scope decision): v2. Includes loading/error/empty/no-event states, responsive layout, counts render contract, error-row treatment, and the cost-visibility (all-users vs operator-only) + `git_sha`-in-browser decisions.
- **Node-tagged structured per-node errors** (origin Open item #4): v2 — enables per-node error rows. v1 uses run-level errors.
- **Full Outline-safe redaction pipeline** (keyed-token + two-destination builder selection): v2, shipped with Outline publishing. v1 already keeps the raw query out of the markdown (opaque `run_id`, R6); v2 generalizes this (e.g. keyed HMAC, redacting other free-text fields if any appear). Note: v1's `run_id` is a *random opaque* id — sufficient because it is never stored anywhere correlatable (not in Langfuse/DB); a keyed HMAC is only needed if the token must be deterministically derivable.
- **Backfilling the unused `cost_usd` DB column** for pipeline turns (origin Open item #6): out of scope; cost is disk-only in v1.

## Context & Research

### Relevant Code and Patterns

- `backend/src/kestrel_backend/graph/builder.py` — `build_discovery_graph`; full tail chain is `…→hypothesis_extraction→bridge_grounding→literature_grounding→synthesis→END` (literature_grounding runs *before* synthesis). `synthesis` is the terminal functional node — `add_edge("synthesis", END)` at **line 196** is the exact edge to replace with `synthesis→reporting→END`. Conditional branches: triage→`[direct_kg|cold_start]`→pathway_enrichment (both branches run in the **same superstep**); integration→`[temporal|hypothesis_extraction]`. A terminal `reporting` node runs last on every path.
- `backend/src/kestrel_backend/graph/state.py` — `model_usages` (`Annotated[list, operator.add]`, ~451), `errors` (flat `list[str]`, ~455), `node_timings` (plain `dict[str,float]`, ~456, no reducer). `ModelUsageRecord` (~337) carries `node_name`, tokens, `mcp_tool_calls`; **frozen** (use `model_dump()`).
- `backend/src/kestrel_backend/graph/node_detail_extractors.py` — `extract_node_details(node_name, state) -> (summary, details)`; 12-node `_EXTRACTORS` dispatch. Reuse for counts; do not redesign.
- `backend/src/kestrel_backend/agent.py:340-348` — `TurnMetrics.cost_usd` hardcodes Sonnet 4.5 rates ($3/$15/$0.30/$3.75 per 1M). Generalize into a per-model map.
- `backend/src/kestrel_backend/main.py` — WS path: accumulates state (`CONCAT_LIST_FIELDS`), computes a **buggy** inline `node_timings` (605-610: first-seen start, overwrite every yield), passes it to Langfuse (686) + DB `add_turn` (714, where it is **not persisted**). Langfuse handler attached via `pipeline_config = {"callbacks": [handler]}` (575) — WS path only.
- `backend/src/kestrel_backend/graph/runner.py` — `run_discovery` (`ainvoke`, 46) and `stream_discovery` (`astream(stream_mode="updates")`, 83); both thread `config`; wrap yields as `{"type":"node_update","node":...,"node_output":...}` (85-89).
- `backend/langgraph.json` — Studio entry `kestrel_backend.graph.builder:build_discovery_graph` (bypasses runner; **terminal-node approach required for Studio coverage**).
- `backend/assessment_data/brown_c1_pilot_e2e.py` — artifact pattern: `Path(...)/f"...{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"`, `json.dumps(artifact, indent=2, default=str)`, `git_sha()` via `subprocess git rev-parse` (56-60), `_meta` block with reproduce-inputs. No retention.

### Institutional Learnings

- `docs/solutions/best-practices/reliable-long-running-llm-batch-runs-2026-06-07.md` — persist-by-default; **cost via per-unit deltas, not cumulative counters** (a 150× double-count bug). `model_usages` records are per-call deltas appended via `operator.add`, so summing them is correct — verify no cumulative counter is summed. Atomic write (temp + rename) so a mid-write crash leaves no corrupt JSON.
- `docs/solutions/best-practices/langfuse-sdk-v3-migration-2026-06-03.md` — exact **fail-safe-on-request-path** pattern: wrap instrumentation in try/except, degrade to no-op, never raise into the response. Langfuse already tracks per-node cost → derive from the same `model_usages` source to avoid divergence.
- `docs/solutions/best-practices/langgraph-pipeline-production-formalization.md` — template for lifting per-node logic into shared utilities + formalizing state contracts; test aggregation **structure/math**, not non-deterministic LLM values.
- `docs/solutions/best-practices/discovery-pipeline-one-graph-methods-within-nodes-2026-05-28.md` — report must distinguish "node did not run" (conditional branch skipped, e.g. cold_start/temporal) from "ran with zero".
- `docs/solutions/developer-experience/pytest-venv-path-spaces-module-invocation-2026-06-01.md` — run `cd backend && uv run python -m pytest tests/ -v -m "not integration"`; full suite has ~16-18 pre-existing failures → validate new test files individually.

### External References

- Per-model token pricing values come from the `claude-api` reference at implementation time (current Opus/Sonnet/Haiku rates), seeded from the existing `agent.py` Sonnet rates. No other external research needed — strongly patterned in-repo.

## Key Technical Decisions

- **Graph-terminal reporting node** (resolves Open item #1): place the reporter as a node after `synthesis` (`synthesis → reporting → END`). Only this covers all three entry paths (WS, `run_discovery`, Studio). A runner/`main.py`-level reporter would miss Studio (`langgraph.json` bypasses the runner).
- **`timed_node` wrapper + dict-merge reducer** (resolves Open items #2, #3): wrap each node at registration in `builder.py` with a timing wrapper that injects `{node_name: duration}` into the node's returned state; merge via a custom `node_timings` reducer. The reducer is **required, not just convenient**: `direct_kg` and `cold_start` write `node_timings` in the same superstep, and LangGraph raises `InvalidUpdateError` on concurrent writes to a non-reducer field. One duration per node execution (the wall-clock of that node's `await`); path-independent (baked into the compiled graph), concurrency-safe (per-run state, not a shared handler instance), and immune to the stream-delta multi-yield bug. Avoids editing all 12 node bodies. **Concurrency caveat (carried from origin):** for the parallel `direct_kg`/`cold_start` superstep, each node's `await` wall-clock overlaps the other, so summed per-node durations can exceed total wall-clock and `pct_of_total` can exceed 100%. Total wall-clock is exact; per-node durations are approximate under concurrency — surfaced as a caveat in the report. **Defensiveness (R4):** the wrapper and `merge_node_timings` run on *every* node, *outside* the reporting node's try/except — they must each be independently defensive (wrapper catches its own timing errors and still returns the node's result unchanged; reducer never raises on unexpected/None/non-dict values).
- **Reuse, don't rewrite** (resolves Open item #2): the reporter consumes the same `model_usages`/`errors`/`extract_node_details` the WS path already uses. **The disk artifact reads `node_timings` from the graph's merged final state inside the terminal node — it does not depend on `main.py`.** Unit 6 is still required because Unit 1's reducer change *regresses* `main.py`'s WS-path `node_timings`: `_get_concat_fields()` (main.py:84) only detects `operator.add`, and the accumulator's merge branch (main.py:599) is list-only, so a `dict` reducer field falls to last-write-wins → Langfuse would receive only the last node's timing. Unit 6 must add an **explicit dict-merge branch** for `node_timings` in `main.py`'s accumulator (do **not** rely on `CONCAT_LIST_FIELDS`), which also kills the pre-existing first-seen/overwrite bug. `add_turn` does **not** persist `node_timings` (no such column, database.py:156), so this benefits **Langfuse only**, not the DB.
- **Cost from `model_usages`, per-model map** (resolves Open item #6): generalize `agent.py` rates into `pricing.py` (model_name → rates, normalization, "last verified" date, unknown→`None` + warn). Same source as Langfuse → no divergence. Don't backfill the DB `cost_usd` column in v1.
- **Errors at run-level in v1** (resolves Open item #4): a terminal node sees only the flat `errors` list (no stream-delta origin). Report run-level errors + per-node ran/skipped status. Per-node error rows (node-tagged errors) → v2.
- **Retention = keep last N pairs, prune-on-write** (resolves Open item #5): default `N=200` (env-overridable), oldest pruned during the same fail-safe-wrapped write. No cron dependency. **Best-effort under concurrency**: `SDK_SEMAPHORE=4` allows concurrent runs, so two reporting nodes may prune simultaneously — prune must tolerate already-deleted files (swallow `FileNotFoundError`) and never count/delete an in-flight `*.tmp`; the cap is approximate, not exact, under concurrent writers. Keep prune cheap (stat+unlink only) since it runs on the request path.
- **Fail-safe + atomic write**: the entire reporting node body is wrapped try/except (logs a warning, never raises); disk write is temp-file + rename; dir 700 / files 600.
- **git_sha captured once** (resolves R9): memoized `get_git_sha()` (single `git rev-parse` at first call / startup), not a per-request subprocess; catch `CalledProcessError`/`FileNotFoundError` specifically and return `"unknown"` **without logging the exception detail** (avoid leaking the absolute repo path into logs). Correct on the deploy path (deploy does `git reset --hard` then `systemctl restart`, so the fresh process re-memoizes the new HEAD); the only stale case is manual prod branch-checkout *without* restart (documented testing flow) — acceptable.
- **Schema versioned** (resolves Open item #7): JSON carries `report_version: 1`; schema documented in the module docstring.
- **Serialization** (resolves Open item #8): `ModelUsageRecord.model_dump()`; `json.dumps(..., default=str)` backstop.

## Open Questions

### Resolved During Planning

- Entry-path coverage → graph-terminal node (Studio-safe).
- Per-node timing source → `timed_node` wrapper + merge reducer (exact, path-independent).
- Reuse vs rewrite → extract shared reporter; align `main.py` to authoritative timings.
- Error attribution → run-level + ran/skipped status v1; node-tagged errors v2.
- Retention → keep-last-N (200) prune-on-write.
- Cost vs Langfuse → derive from `model_usages`; no DB backfill v1.
- Schema versioning, frozen-model serialization → resolved above.

### Deferred to Implementation

- Verify the WS-accumulator dict-merge (Unit 6) against the installed `langgraph` version's `stream_mode="updates"` delta shape (the *design* is committed: an explicit dict branch; only the version-behavior confirmation is deferred).
- Final per-model rate values (pull current rates from `claude-api` reference; seed from `agent.py:340-348`).
- Exact module home for `pricing.py` / `run_reports_io.py` (`kestrel_backend/` vs `graph/`) — pick during implementation; keep surface-agnostic (no graph imports in pricing/io).

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation specification. The implementing agent should treat it as context, not code to reproduce.*

```mermaid
flowchart LR
    subgraph graph["compiled discovery graph (all entry paths)"]
      direction LR
      I[intake] --> ER[entity_resolution] --> T[triage]
      T --> S[...nodes...] --> SY[synthesis] --> R[reporting node NEW]
      R --> E((END))
    end
    W["timed_node wrapper (builder.py)"] -. "injects {node: duration}" .-> graph
    R --> RB["report builder (state -> report dict)"]
    RB --> PR["pricing.py (model_usages -> cost)"]
    RB --> ND["extract_node_details (counts)"]
    RB --> JM["JSON + Markdown render"]
    JM --> IO["run_reports_io: atomic write, 700/600, retention, git_sha"]
    IO --> D[("backend/run_reports/&lt;ts&gt;_&lt;run-id&gt;.{json,md}")]
```

Per-node timing is captured by the wrapper into reducer-merged `node_timings`; everything else the reporter needs (`model_usages`, `errors`, counts via `extract_node_details`) is already in the accumulated `DiscoveryState` when the terminal `reporting` node runs. The reporter is pure-from-state → trivially testable without running the LLM pipeline.

## Implementation Units

- [ ] **Unit 1: Per-node timing — `timed_node` wrapper + `node_timings` merge reducer**

**Goal:** Capture exact per-node duration on every entry path without editing node bodies.

**Requirements:** R2

**Dependencies:** None

**Files:**
- Modify: `backend/src/kestrel_backend/graph/state.py` (custom dict-merge reducer for `node_timings`)
- Create: `backend/src/kestrel_backend/graph/timing.py` (`timed_node(name, fn)` async wrapper)
- Modify: `backend/src/kestrel_backend/graph/builder.py` (wrap each `add_node` registration)
- Test: `backend/tests/test_timing_wrapper.py`

**Approach:**
- `timed_node` records `time.time()` before/after awaiting the wrapped node fn, shallow-copies the node's returned dict, and sets `node_timings={name: duration}`; returns the merged dict. One duration per node execution (the node's `await` wall-clock). Handles `None`/empty/non-dict returns. **Shallow-copy only** — pass reducer-bound values (lists of frozen `ModelUsageRecord`/`Finding`) through unchanged; never deep-copy or reconstruct frozen models.
- **Defensive (R4):** the wrapper runs on every node, outside the reporting node's guard — if timing/merge logic raises, it must catch internally and return the node's original result unchanged, so a timing bug can never break the pipeline.
- Replace `node_timings: dict[str, float]` with `Annotated[dict[str, float], merge_node_timings]`. The reducer is **required**: `direct_kg`+`cold_start` write `node_timings` in the same superstep, so LangGraph raises `InvalidUpdateError` without one. `merge_node_timings(a, b)` key-merges, treats `None`/missing as `{}`, and **never raises**.
- Wrap all node registrations in `builder.py` uniformly (a helper applied at each `add_node`).

**Patterns to follow:** existing `operator.add` reducer declarations in `state.py`; node registration block in `builder.py`.

**Test scenarios:**
- Happy path: wrapping a stub node that returns `{"errors": ["x"]}` yields a dict containing both `errors` and `node_timings={name: <float>}`.
- Edge case: wrapped node returns `None` / `{}` / a non-dict → result still carries `node_timings` for that node; non-dict is handled without raising.
- Edge case: `merge_node_timings` merges `{"a":1.0}` + `{"b":2.0}` → `{"a":1.0,"b":2.0}` (parallel branches); merging with `None`/`{}` does not raise.
- Edge case (defensiveness, R4): a wrapper whose timing logic is forced to raise still returns the node's original result unchanged (pipeline unaffected).
- Edge case: duration is non-negative and `>= sleep` for a stub that sleeps a known interval (not exact).
- Integration: a compiled two-branch graph (direct_kg + cold_start) runs without `InvalidUpdateError` and final `node_timings` contains both branch keys.

**Verification:** A full graph run leaves `state["node_timings"]` with one entry per executed node; skipped conditional nodes are absent (not zero); a forced timing error does not break the run.

- [ ] **Unit 2: Per-model pricing module**

**Goal:** Estimate $ cost from `model_usages`, generalized beyond hardcoded Sonnet rates.

**Requirements:** R3

**Dependencies:** None

**Files:**
- Create: `backend/src/kestrel_backend/pricing.py` (rate map + `estimate_cost(records)`)
- Test: `backend/tests/test_pricing.py`

**Approach:**
- Rate map keyed by normalized model id (handle free-form names like `anthropic/claude-sonnet-4-…`), fields: input/output/cache_read/cache_create per 1M; module-level `LAST_VERIFIED = "YYYY-MM-DD"`.
- Seed from `agent.py:340-348` Sonnet rates; add current Opus/Haiku rates from the `claude-api` reference.
- `estimate_cost` sums per-record cost (records are per-call deltas — safe to sum; do **not** sum a cumulative counter). Unknown model → cost `None` + `logger.warning`, never raise.
- Provide per-record and grouped-by-node helpers for the reporter.

**Patterns to follow:** `TurnMetrics.cost_usd` math in `agent.py`.

**Test scenarios:**
- Happy path: known model + known tokens → expected $ (assert against hand-computed value).
- Edge case: cache_read/cache_create tokens included in the total.
- Error path: unknown model id → returns `None`, logs warning, does not raise.
- Edge case: empty record list → `0.0` (not `None`).
- Edge case: free-form `anthropic/...` name normalizes to a known rate key.

**Verification:** `estimate_cost` reproduces the legacy Sonnet number for a Sonnet-only record set within rounding.

- [ ] **Unit 3: Artifact I/O — atomic write, perms, retention, git_sha**

**Goal:** Surface-agnostic durable writer that can never corrupt or unbounded-grow the prod disk.

**Requirements:** R1, R5, R6, R9

**Dependencies:** None

**Files:**
- Create: `backend/src/kestrel_backend/run_reports_io.py` (`write_report`, `prune`, `get_git_sha`, path helpers)
- Modify: `backend/.gitignore` (add `run_reports/` — **must ship in this unit** so the first prod run can't commit 600-perm PII artifacts)
- Test: `backend/tests/test_run_reports_io.py`

**Approach:**
- Output dir `backend/run_reports/` created without a world-readable window: `old=os.umask(0o077); os.makedirs(path, mode=0o700, exist_ok=True); os.umask(old)` (plus an explicit `os.chmod(0o700)`) — a bare `os.makedirs` leaves the dir at umask-default (e.g. 755) until a later chmod.
- Filename `<%Y%m%dT%H%M%SZ>_<run-id>.{json,md}` — run-id, **never** a query slug.
- Atomic write that is never world-readable even transiently: create the temp file with `os.open(tmp, O_WRONLY|O_CREAT|O_EXCL, 0o600)` (mode set at creation, not after), write, then `os.replace` to final. Do not `open()`-then-`chmod` (leaves a pre-chmod window on the PII-bearing file).
- `prune(max_pairs=env REPORT_RETENTION_MAX default 200)`: list completed report pairs (ignore `*.tmp`) by mtime, delete oldest beyond the cap; swallow `FileNotFoundError` (concurrent pruner already deleted it).
- `get_git_sha()`: memoized single `git rev-parse HEAD`; catch `CalledProcessError`/`FileNotFoundError` only, return `"unknown"` without logging the path.
- All functions are independently callable and raise normally here (the fail-safe wrapper lives in the node, Unit 5) — but `prune` failures must not block the write.

**Patterns to follow:** artifact/timestamp/`git_sha` conventions in `brown_c1_pilot_e2e.py`.

**Test scenarios:**
- Happy path: `write_report` produces both `.json` and `.md`; file mode is `600`, dir mode `700`.
- Edge case: the temp file is `600` at creation (assert mode before rename, not just after) — no world-readable window.
- Edge case: atomic write — no `.tmp` remains after success; a crash before rename leaves no partial final file.
- Edge case: `prune` with 205 existing pairs and cap 200 deletes the 5 oldest by mtime, keeps newest 200; ignores any `*.tmp`.
- Error/concurrency path: two `prune` calls targeting the same oldest file → the second's `FileNotFoundError` is swallowed; the just-written report persists.
- Happy path: `get_git_sha` returns a 40-char hex in a git repo; `"unknown"` when `git` unavailable (monkeypatch), with no path in logs.

**Verification:** Repeated writes keep the directory roughly bounded at the cap (approximate under concurrency); artifacts and dir are owner-only at every moment; `run_reports/` is git-ignored.

- [ ] **Unit 4: Report builder — state → report dict → JSON + Markdown**

**Goal:** Pure function turning final `DiscoveryState` + run metadata into the versioned report structure and an Outline-publishable Markdown rendering.

**Requirements:** R3, R6, R7, R9

**Dependencies:** Unit 2

**Files:**
- Create: `backend/src/kestrel_backend/graph/performance_report.py` (`build_report(state, meta) -> dict`, `render_markdown(report) -> str`)
- Test: `backend/tests/test_performance_report.py`

**Approach:**
- Run header carries `report_version=1`, `run_id` (opaque random id from meta), mode, `biomapper_env`, timestamp, `git_sha`, totals (wall-clock, tokens, est. cost), derived headline (top bottleneck by duration, top cost node). The **full `query` is in the JSON header only**; the **Markdown header shows `run_id`, never the raw query** (R6 — markdown is the Outline-bound artifact).
- `render_markdown(report)` MUST NOT emit the `query` field — assert this in tests. `build_report` returns one dict containing both `query` (full) and `run_id`; the JSON serializer keeps `query`, the markdown renderer reads only `run_id`.
- Per-node rows: status (`ran` if in `node_timings`, else `skipped`), `duration_s` + `pct_of_total`, tokens (in/out/cache) + model + `mcp_tool_calls` (group `model_usages` by `node_name`, `model_dump()` frozen records), cost (Unit 2), counts (`extract_node_details`). Run-level `errors` list in the header.
- **Concurrency caveat in output:** `pct_of_total` is `duration_s / total_wall_clock`; for the parallel `direct_kg`/`cold_start` superstep these overlap, so per-node sums can exceed 100%. Render a one-line note that concurrent branches overlap; total wall-clock is the exact denominator.
- Markdown: plain GFM tables, nodes ranked by duration then cost; no exotic embeds (R7).

**Patterns to follow:** brown harness `_meta`/`node_timeline` shape; `extract_node_details` return contract.

**Test scenarios:**
- Happy path: a synthetic serial-node state (2 sequential nodes + one skipped) → report dict with correct per-node tokens, summed totals, `pct_of_total` summing ~100% within tolerance, correct headline node.
- Edge case (concurrency): a state where two nodes' durations overlap the same wall-clock window → builder does **not** crash or clamp incorrectly; summed `pct_of_total` may exceed 100% and the overlap note is present.
- Edge case: a node in `node_timings` with no `model_usages` → tokens/cost `0`/`None`, status `ran`.
- Edge case: conditional node absent from `node_timings` → status `skipped`, excluded from the wall-clock denominator? (No — denominator is total wall-clock, not sum of durations; assert skipped nodes simply have no row contribution.)
- Edge case: unknown model → cost `None` rendered as `est. n/a`, never crashes the render.
- Security (R6): the rendered Markdown contains `run_id` and does **not** contain the raw query string (assert the query text is absent); the JSON does contain the full query.
- Happy path (R7): rendered Markdown is valid GFM (pipe tables, no raw HTML/embeds) — assert table header/separator rows present.
- Integration: feed a real recorded `DiscoveryState` fixture (or the brown harness's accumulated state) → totals reconcile with the harness's own `node_timeline` ordering (R8).

**Verification:** `build_report` is pure (no I/O); `render_markdown` output round-trips through a GFM parser without errors.

- [ ] **Unit 5: Reporting terminal node + graph wiring (fail-safe)**

**Goal:** Wire the reporter into the graph so it fires last on every path and can never break a run.

**Requirements:** R1, R4, R5

**Dependencies:** Unit 3, Unit 4

**Files:**
- Create: `backend/src/kestrel_backend/graph/nodes/reporting.py` (`run(state) -> dict`)
- Modify: `backend/src/kestrel_backend/graph/builder.py` (`add_node("reporting", ...)`; replace `add_edge("synthesis", END)` at line 196)
- Test: `backend/tests/test_reporting_node.py`

**Approach:**
- `reporting.run`: generate an opaque random `run_id` (e.g. `uuid4().hex` or `secrets.token_hex`), build report (Unit 4) passing `run_id` in meta, write JSON (full query) + MD (run_id only) via Unit 3 using `run_id` in the filename, prune. **Entire body wrapped in try/except** — log a warning and return `{}` on any failure; never raise (R4). Returns a minimal `{"report_path": ...}` for observability (optional). The `run_id` is stored only in the JSON artifact — never emitted to Langfuse/DB — so it cannot be correlated back to the query.
- Wiring: replace `add_edge("synthesis", END)` (builder.py:196) with `add_edge("synthesis", "reporting")` + `add_edge("reporting", END)`.
- **Do NOT add `reporting` to `NODE_STATUS_MESSAGES`** (protocol.py): the WS loop skips nodes not in that set (main.py:592), so `reporting` is intentionally invisible to the WS stream/accumulator in v1 (it writes its own disk artifact). Decide whether to exclude `reporting` from its own report (its duration is dominated by disk I/O) — recommended: exclude.

**Execution note:** Start with a failing test that mocks `write_report` to raise and asserts the run completes (fail-safe contract) before wiring real I/O.

**Patterns to follow:** existing node `run(state)` signatures in `graph/nodes/`; builder edge wiring.

**Test scenarios:**
- Happy path: invoking `reporting.run` on a synthetic terminal state writes a JSON+MD pair to a temp dir.
- Error path (R4): `write_report` raises → `reporting.run` logs a warning, returns without raising, run is unaffected.
- Error path (R4): `build_report` raises (mock a malformed state) → swallowed, no partial artifact, no raise.
- Integration: compiled graph reaches `reporting` after `synthesis` on both the direct_kg branch and the cold_start branch (assert node executed via a write side effect).

**Verification:** `END` is reached after `reporting` on every conditional path; a forced reporting failure leaves the pipeline result intact.

- [ ] **Unit 6: Prevent WS-path `node_timings` regression + WS integration test**

**Goal:** Unit 1's reducer change would regress `main.py`'s WS-path `node_timings` to last-write-wins; fix `main.py` to merge the dict, kill the pre-existing timing bug, and prove the report lands on the prod WebSocket path. (Required-for-v1: not an accuracy nice-to-have — without it, Langfuse receives only the last node's timing.)

**Requirements:** R2, R4, R8

**Dependencies:** Unit 1, Unit 5

**Files:**
- Modify: `backend/src/kestrel_backend/main.py` (explicit dict-merge branch for `node_timings` in the accumulator; source `node_timings` from accumulated state)
- Test: `backend/tests/test_pipeline_report_integration.py`

**Approach:**
- **Add an explicit dict-merge branch** for `node_timings` in the accumulation loop (main.py:595-602), *before* the list-concat check: `_get_concat_fields()` (main.py:84) only detects `operator.add`, and the concat branch (main.py:599) is list-only, so a `dict` reducer field otherwise falls to last-write-wins. Do **not** rely on `CONCAT_LIST_FIELDS`.
- Replace the first-seen/overwrite `node_start_times` logic (main.py:605-610) by reading the authoritative merged `node_timings` from accumulated state; pass that to Langfuse (main.py:686).
- **Scope note:** `add_turn` (database.py:154-156) has no `node_timings` column, so this benefits **Langfuse only** — node_timings is not persisted to the DB in v1 (consistent with Scope Boundaries). Do not claim DB persistence.
- Confirm no behavior change to the user-facing stream.

**Execution note:** Characterization-first — capture the current Langfuse metrics shape before changing the inline timing, so the refactor preserves the contract.

**Patterns to follow:** the accumulation block in `main.py` (add a dict branch alongside the existing list-concat branch).

**Test scenarios:**
- Integration: a mocked end-to-end discovery run over the WS handler writes exactly one report pair and streams its result normally.
- Happy path (regression guard): with the dict-merge branch, accumulated `node_timings` after a multi-node stream contains **all** nodes — assert it does NOT collapse to the last node only (the bug this unit prevents).
- Error path (R4): reporting node failure during a WS run → user still receives the full streamed response; warning logged.
- Happy path: `node_timings` passed to Langfuse equals the authoritative per-node durations (no monotonic overcount from the old first-seen/overwrite bug).
- Edge case (R8): a recorded multi-node run's report node ordering/coverage cross-checks against the brown harness `node_timeline` for the same inputs.

**Verification:** `main.py` accumulates `node_timings` for all nodes (not last-write-wins) and no longer uses the first-seen/overwrite logic; Langfuse receives accurate durations; a prod-shaped run produces a disk report.

- [ ] **Unit 7: Synthesis fallback marker (surface the silent overflow)**

**Goal:** Make the report able to flag the synthesis-overflow degradation that otherwise reads as `status=complete, errors=0`.

**Requirements:** R3 (failure attribution for the motivating case)

**Dependencies:** None (independent of Units 1-6; report picks it up via run-level `errors`)

**Files:**
- Modify: `backend/src/kestrel_backend/graph/nodes/synthesis.py` (emit a marker on the fallback paths)
- Test: `backend/tests/test_synthesis_fallback_marker.py`

**Approach:**
- At the fallback convergence points (synthesis.py:1009 empty-LLM-text, 1010-1012 SDK exception), append a distinct run-level marker to the node's returned `errors` (the `operator.add` reducer carries it), e.g. `"synthesis: fell back to deterministic report (LLM returned empty)"` and `"synthesis: fell back to deterministic report (SDK error: <type>)"`. Add `"errors"` to synthesis's return dict only on these degraded paths.
- The no-SDK path (1014-1015) is an environment condition, not a degradation — use a distinct, clearly-labeled marker (or omit) so it isn't confused with the overflow case.
- Keep the marker a short string (no raw report dump). The report's run-level errors list (Unit 4) surfaces it automatically; no reporter change needed.

**Execution note:** Test-first — assert the marker is emitted on each degraded path before touching the node.

**Patterns to follow:** existing `errors` returns in other nodes (e.g. `direct_kg.run` returning `{"errors": [...]}`); the SDK/fallback convergence comment block at synthesis.py:1021-1026.

**Test scenarios:**
- Happy path: LLM returns non-empty text → no fallback marker in `errors`.
- Degraded (the motivating case): LLM returns empty/whitespace text → `errors` contains the empty-fallback marker; `synthesis_report` is the deterministic fallback.
- Degraded: `query_with_usage` raises → `errors` contains the SDK-error marker; run still completes.
- Edge case: no-SDK environment → distinct marker (or none), not the overflow marker.

**Verification:** A synthesis run forced down each fallback path emits its marker; a normal run emits none; the marker appears in the report's run-level errors.

## System-Wide Impact

- **Interaction graph:** New terminal `reporting` node on the compiled graph affects all three entry paths (WS, `run_discovery`, Studio). The `timed_node` wrapper touches every node registration in `builder.py` (uniform, behavior-preserving).
- **Error propagation:** **Two new failure surfaces, not one.** (1) The reporting node is fully isolated by a try/except that never raises (R4). (2) The `timed_node` wrapper + `merge_node_timings` reducer run on *every* node, *outside* that guard — so they must be independently defensive (Unit 1): a timing/merge bug must never propagate into the pipeline. Pricing/IO functions raise normally but are only called inside the reporting guard.
- **State lifecycle risks:** `node_timings` becomes a reducer field (required — parallel branches write it). Confirmed readers are `main.py` only (Langfuse 686, the dropped `add_turn` pass 714); Unit 6 fixes the WS accumulator. Atomic temp+rename prevents partial-write corruption. Retention prune is best-effort under concurrency (tolerates already-deleted files), not exact.
- **API surface parity:** WS protocol unchanged in v1 (no `performance_report` event; `reporting` deliberately absent from `NODE_STATUS_MESSAGES`). Langfuse `node_timings` values get more accurate; `add_turn` payload shape unchanged (node_timings still not persisted).
- **Integration coverage:** Unit 5/6 integration tests prove the report lands on the real WS path and survives both conditional branches. **Studio path is not unit-testable** — verify manually (see Documentation/Operational Notes) since Studio adds a checkpointer that could in principle replay/interrupt the terminal node.
- **Failure-attribution in v1:** Unit 7 makes synthesis emit a run-level marker on its fallback paths, so the **synthesis-overflow silent fallback** (the motivating degradation, which otherwise reads as `status=complete, errors=0` — see memory `brown-c1-48-run-synthesis-overflow-wall`) is surfaced in the report's run-level errors. General per-node error *rows* (attributing arbitrary errors to their node) still need node-tagged errors (v2); v1 covers the one high-value case explicitly.
- **Unchanged invariants:** User-facing streaming output, classic mode, the WS protocol, and the synthesis report content are unchanged. The reporter is read-only over state plus disk I/O.

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Reporting failure breaks a prod run | Reporting node try/except never raises; **and** `timed_node`/reducer are independently defensive (run outside that guard); proven by Unit 1 + Unit 5/6 error-path tests (R4). |
| Unbounded disk growth on single Lightsail box | Retention cap (keep last N=200) prunes on every write (R5); best-effort under concurrent runs; no prior art, so it ships *with* the writer — never the writer first. |
| Corrupt JSON if process dies mid-write | Atomic temp-file (`O_EXCL`, mode `600` at creation) + `os.replace`. |
| User query (PII/PHI) leaks via filenames / disk / git | Run-id (not query slug) in filenames; 600 files / 700 dir created without a world-readable window; `run_reports/` git-ignored in the same unit (Unit 3). Outline-redaction variant deferred to v2 (built with publishing). |
| Cost number diverges from Langfuse | Both derive from the same `model_usages`; pricing map carries a "last verified" date; unknown model → `None` not a guess. |
| `node_timings` reducer change regresses Langfuse timings | `_get_concat_fields()` only detects `operator.add`; Unit 6 adds an explicit dict-merge branch (does not rely on `CONCAT_LIST_FIELDS`); regression-guard test asserts all nodes present. Confirmed only `main.py` reads the field. |
| Touching every node registration introduces regressions | Wrapper is behavior-preserving (only adds `node_timings`); covered by Unit 1 + the existing pipeline test suite (run new test files individually given ~16-18 pre-existing suite failures). |

## Documentation / Operational Notes

- Document the JSON schema (`report_version: 1`) in the `performance_report.py` module docstring.
- Note `REPORT_RETENTION_MAX` env var in `backend/.env.example` and deploy docs; default 200.
- `run_reports/` git-ignore entry ships in Unit 3 (not as a later cleanup) so the first prod run can't commit PII artifacts.
- **Manual Studio verification** (not unit-testable): run one query in LangGraph Studio (`uv run langgraph dev`) and confirm exactly one report pair is written and `reporting` runs to `END` once (no checkpointer replay/duplicate). Record the result when implementing Unit 5.
- Testing: `cd backend && uv run python -m pytest tests/test_*report* tests/test_pricing.py tests/test_timing_wrapper.py tests/test_reporting_node.py tests/test_run_reports_io.py tests/test_synthesis_fallback_marker.py -v` (module invocation; validate these files individually given ~16-18 pre-existing full-suite failures).

## Sources & References

- **Origin document:** [docs/brainstorms/2026-06-22-pipeline-node-performance-report-requirements.md](docs/brainstorms/2026-06-22-pipeline-node-performance-report-requirements.md)
- Related code: `graph/builder.py`, `graph/state.py`, `graph/node_detail_extractors.py`, `agent.py`, `main.py`, `assessment_data/brown_c1_pilot_e2e.py`
- Learnings: `docs/solutions/best-practices/reliable-long-running-llm-batch-runs-2026-06-07.md`, `docs/solutions/best-practices/langfuse-sdk-v3-migration-2026-06-03.md`, `docs/solutions/best-practices/langgraph-pipeline-production-formalization.md`, `docs/solutions/developer-experience/pytest-venv-path-spaces-module-invocation-2026-06-01.md`
