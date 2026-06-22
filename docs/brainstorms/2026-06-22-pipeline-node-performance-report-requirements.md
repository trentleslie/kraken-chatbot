# Per-Node Pipeline Performance Report — Requirements

_Brainstorm date: 2026-06-22 · Status: requirements — reviewed (7-persona document-review applied) + scope decided (disk-only v1, queries sensitive) · ready for ce:plan · Branch context: `perf/bridge-grounding-parallel-cache`_

## Problem

The KRAKEN discovery pipeline runs 12 nodes per query, each making paid LLM calls and MCP graph queries. When a run is slow, expensive, or degrades (e.g. the synthesis-overflow wall, `pathway_enrichment` at ~307s), there is **no single durable artifact that attributes time, tokens, cost, and failures to individual nodes**. Today the data is ~70% latent — and much of the aggregation logic already exists, scattered and not durably emitted:

- Every node already times itself with `time.time()` and logs duration. The `DiscoveryState.node_timings` *TypedDict field* (`graph/state.py:456`) is never written, **but `main.py` already computes a local `node_timings` dict** (main.py:523, 605-610) inside the WebSocket handler and persists it to the Langfuse trace (main.py:686) and the DB turn metrics (main.py:714). So timing capture is not greenfield — it exists, only inside the WS path, and (see Key architectural constraint) it has a known overwrite bug.
- `main.py` also already maintains an **accumulator** (`accumulated_state` merged via `CONCAT_LIST_FIELDS`, main.py:522, 595-602) that drains every streamed node delta — exactly the "accumulate as it streams" mechanism this report needs.
- `model_usages` (`graph/state.py:451`, frozen `ModelUsageRecord` at `:337`) already captures per-node `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_creation_tokens`, `mcp_tool_calls`, and `available_tools`, each tagged with `node_name`.
- Per-node **counts already have extractors**: `graph/node_detail_extractors.py` implements `extract_node_details(node_name, state)` with a 12-node `_EXTRACTORS` dispatch (entities resolved, edges, bridges, hypotheses, etc.), already called at main.py:631.
- `errors` (`graph/state.py:455`) is a flat `list[str]` accumulated via `operator.add` — **it carries no node field** (see Per-node rows for the attribution consequence).
- The Brown eval harness (`backend/assessment_data/brown_c1_pilot_e2e.py`) hand-rolls a `node_timeline` + coverage counts — reinventing a slice of this report for one harness only.

The missing piece is therefore **not a new accumulator but one shared, surface-agnostic reporter** that lifts the existing per-node aggregation (timing + tokens + cost + counts + errors) out of the WS handler, fixes the timing bug, adds cost + serialization + durable emission, and runs for *every* entry path — not just the WebSocket one.

## Goal

Auto-generate a per-node performance report at the end of **every** discovery pipeline run (prod WebSocket, LangGraph Studio, and eval harnesses alike). The report's job is **node-level bottleneck + cost + failure attribution** — not a raw timing dump.

### Non-goals

- Not replacing Langfuse tracing (LLM-call-level observability stays in Langfuse).
- Not a real-time dashboard. This is an end-of-run artifact.
- Not a billing system of record — the `$` figure is an estimate (see Cost below).

## Scope decision (load-bearing)

**Every pipeline run.** The aggregator lives in the streaming path (`stream_discovery()` in `graph/runner.py`) and fires once per run. Two consequences follow directly:

1. **Fail-safe is mandatory — at two distinct points.** This sits on the prod request path, and aggregation now runs *during* the stream, not only at the end. So two guards are required: (a) **per-yield accumulation** must be individually wrapped — an exception draining a single delta (e.g. a non-serializable `node_output`) logs a warning and skips that delta but **never aborts the stream**; (b) **end-of-run emission** (serialization + disk write) must be wrapped so any failure (disk full, serialization error, missing field) logs a warning and is swallowed. Together these guarantee a reporting failure can **never** break or delay a user-facing run.
2. **Auto-save by default.** Per the artifact-hygiene SOP, and because every run makes paid LLM calls (expensive to reproduce), the artifact is written by default. An `--out`/path override is allowed, but there is no flag that gates *whether* it saves.

### v1 scope — disk-only (decided 2026-06-22)

**v1 ships the disk artifact (JSON + Markdown) only.** The `performance_report` WebSocket event + frontend renderer are **deferred to v2**. Rationale: the disk artifact fully serves the stated dev/research audience, and the WS path carries a large UI-spec surface (loading/error/empty/no-event states, responsive layout for a wide node table, counts render contract, error-row treatment) that is not worth gating v1 on. Consequences:

- The reporter is **surface-agnostic** and owns disk emission only. No `protocol.py` event type and no `client/src/` work in v1.
- The browser-payload concerns (cost visibility to users, `git_sha` exposure) are **moot for v1** — nothing report-related is sent to the browser. They return as open decisions when v2 (the WS event) is scoped.
- v2, when scoped, must specify the deferred UI states above and the cost-visibility decision (all authenticated users vs operator-only).

## Key architectural constraint

`stream_discovery()` uses `stream_mode="updates"`, so the runner does **not** retain full accumulated state — the underlying graph yields one node's delta per step, and the caller accumulates. Two implementation facts must be pinned (both verified against source):

- **Yield shape.** `stream_discovery()` does not yield raw `{node_name: node_output}` — it wraps each as `{"type": "node_update", "node": node_name, "node_output": node_output}` (runner.py:85-89). A reporter placed *inside* the runner consumes the raw `event.items()` loop; one placed at the caller consumes the wrapped form. `main.py` also filters out `__start__` and non-status nodes (main.py:592) — the reporter must replicate that filtering or it will time/cost phantom nodes.
- **Reuse the existing accumulator.** The reporter must **accumulate as it streams** by draining each node's delta into running `model_usages` / `errors` lists — which is what main.py already does. Prefer extracting that logic over rewriting it (see Open items: reuse-vs-rewrite decision).

**Per-node timing — known bug + multi-yield rule.** The existing main.py timing (605-610) records `node_start_times[node]` only on first sighting but overwrites `node_timings[node]` on *every* sighting. Nodes with `operator.add` reducers (e.g. `resolved_entities`, `direct_findings`, `model_usages`) yield **multiple deltas** per run from parallel `asyncio.gather` batches, so for those nodes `duration = now − first_seen` grows monotonically and double-counts unrelated intervening wall-clock — not merely "approximate concurrent work." The plan MUST define an explicit accumulation rule for multi-yield nodes (e.g. sum per-delta intervals, or measure first-start→last-yield) and pick the per-node timing *source* (see Open items). Caveat to retain: with truly parallel branches, any inter-yield attribution is approximate; **total wall-clock is exact**.

## Report contents

### Run-level header (reproducibility — SOP)

- `query` (and `mode`: classic vs discovery)
- `timestamp` (run start, ISO 8601)
- `git_sha` (current commit; pins code version). Capture as a **build/deploy-time constant** (e.g. baked into an env var at deploy), not a per-request `git rev-parse` subprocess — the every-run scope means this runs on the prod request path. Keep `git_sha` and `biomapper_env` in the **disk artifact only**; omit them from the browser-facing WS payload (no need to fingerprint the deployed commit to end users).
- `biomapper_env` (production/dev toggle — affects entity resolution behavior)
- Totals: wall-clock seconds, total tokens (in/out/cache), total estimated `$`
- Derived headline: top bottleneck node (by duration) and top cost node (by `$`)

### Per-node rows

| Field | Source |
|---|---|
| `node` | stream key |
| `status` (ok / error / skipped) | derived from `errors` + whether node ran |
| `duration_s` + `pct_of_total` | populated `node_timings` |
| `input_tokens` / `output_tokens` / `cache_read_tokens` / `cache_creation_tokens` | `model_usages` |
| `model(s)` | `model_usages.model_name` |
| `cost_usd` (estimate) | tokens × per-model pricing map |
| `mcp_tool_calls` | `model_usages.mcp_tool_calls` |
| `counts` (node-specific findings: entities resolved, edges, bridges, hypotheses, etc.) | **reuse `node_detail_extractors.extract_node_details`** (already exists, 12-node dispatch) — do not redesign |
| `errors` | `errors` is a flat `list[str]` with **no node field** — attribute by the stream delta the error arrived in (see below) |

> **Error attribution caveat.** Because `errors` carries no node tag, per-node attribution is only possible via *which streamed delta the error appeared in*. This works in the streaming path but is approximate under parallel branches and is **lost entirely in the non-streaming `run_discovery()` path** (no per-delta visibility). The plan must either accept delta-origin attribution (and document its limits) or change nodes to emit structured, node-tagged errors. Acceptance criteria are worded accordingly.

## Output formats — JSON + Markdown

Both emitted per run.

- **JSON** — machine-readable; canonical artifact for diffing across runs and feeding eval tooling. Stable schema (versioned with a `report_version` field).
- **Markdown** — human-readable; ranks nodes by time and cost so a bottleneck is visible at a glance. **Ultimate destination: the BioMapper collection in the Phenome Health wiki (Outline).** The markdown should therefore be Outline-publishable as-is (publishable via the `publish-wiki` skill — plain GFM tables, no exotic embeds). **Publication is a manual, curated step — never per-run auto-publish.** See Security for query-text handling in Outline-bound reports.

## Delivery — disk only (v1)

- **Disk:** timestamped artifact written by default. Proposed location: a dedicated run-reports directory (e.g. `backend/run_reports/<timestamp>_<run-id>.{json,md}`) rather than `assessment_data/` (which is eval-specific). Use a run-id/hash in the filename, **not** a query slug — a query slug leaks user query text into directory listings.
  - **Retention is a v1 requirement, not an open item.** Because the writer is default-on for every prod run on a single shared Lightsail box, an unbounded writer can fill prod disk and take down the user-facing service the fail-safe was meant to protect. The writer and a retention bound MUST ship together — **do not merge the writer ahead of the pruner.** (Specific policy — e.g. keep last N pairs vs total-size cap — is a decision below.)
  - **File permissions:** create `run_reports/` mode `700` and write each artifact mode `600` (owner `ubuntu` only). Artifacts contain the raw query and cost data; default umask would leave them group/world-readable.
- **WebSocket: deferred to v2** (see *v1 scope* above). No `protocol.py` event type or `client/src/` renderer in v1.

## Security — query-text handling (decided 2026-06-22)

User queries are treated as **potentially sensitive** (may contain PII/PHI/proprietary hypotheses). Therefore:

- **Disk artifact** stores the **full raw query** (needed for reproducibility), protected by `600` perms / `700` directory and the retention bound.
- **Outline-bound reports** (the curated Markdown that may be published to the wiki) must **not** carry the raw prod query — replace it with the run-id / a hash. Prod-run reports are **never auto-published**; publishing is a deliberate, per-report human action.
- **Eval-harness reports** (synthetic queries, e.g. Brown module) carry no sensitive data and may be published freely with the raw query intact.
- Because the WS event is deferred to v2, no cost/token/query data crosses the browser trust boundary in v1; the cost-visibility (all users vs operator-only) and `git_sha`-exposure decisions return when v2 is scoped.

## Cost estimation — tokens + `$` estimate

Derive `cost_usd` from a per-model pricing map (input/output/cache rates per model id).

- **Drift risk (must document):** the pricing table is maintained by hand and can diverge from actual billing. Mitigations: keep the map in one well-commented constant with a "last verified" date; label the figure "estimated"; include raw token counts so the reader can recompute. Unknown model id → cost `null` + a logged warning, never a crash.

## Acceptance criteria

1. Running any discovery query writes a JSON + Markdown report pair to disk by default, with no flag required.
2. `node_timings` is populated for every node that ran, using the chosen timing source and multi-yield accumulation rule. Define a **concrete tolerance**: the sum of per-node durations is within a stated bound of total wall-clock (note that under multi-yield re-yields a naive sum can *exceed* wall-clock — the chosen rule must prevent that).
3. Per-node tokens, MCP tool calls, model, and estimated cost all appear and match `model_usages`. Counts appear via `extract_node_details`. Errors that **can be attributed** (by stream-delta origin) appear on their node; the flat-list limitation is documented.
4. Reporting failures never break a run — **proven by test**, not assumed: (a) a mid-stream accumulation error (mock a non-serializable delta) is logged and skipped without aborting the stream; (b) an end-of-run serialization/disk error is logged and swallowed while the run completes and streams its result normally.
5. A retention bound is enforced (ships with the writer): old report pairs are pruned per the chosen policy; `run_reports/` is mode `700` and artifacts mode `600`.
6. The disk artifact stores the full query; any Outline-bound report substitutes a run-id/hash for the raw prod query; prod-run reports are not auto-published.
7. The Markdown renders cleanly when published to Outline (no broken tables/embeds).
8. The Brown harness's bespoke `node_timeline` can be replaced by (or cross-checks against) the shared reporter.
9. Report header pins `git_sha`, query, mode, `biomapper_env`, and timestamp for reproducibility (disk artifact).
10. _(v2)_ When the WS `performance_report` event is added: it is emitted and rendered with loading/error/empty/no-event states; `git_sha`/`biomapper_env` excluded from the browser payload; cost-visibility decision (all users vs operator-only) made.

## Open items / decisions for the plan

Genuine forks (no single correct answer — to be resolved before/at planning):

1. **Reporter placement / entry-path coverage.** Placing the reporter only in `stream_discovery()` does **not** cover LangGraph Studio (which invokes `build_discovery_graph` directly) or the non-streaming `run_discovery()` (`ainvoke`). The Goal claims "prod WebSocket, Studio, and eval harnesses alike." Decide: lift aggregation into a **graph-level mechanism** (e.g. a final reporting node / callback) that fires for all three paths, or **narrow the Goal** to drop Studio/`run_discovery`.
2. **Reuse vs rewrite.** Extract main.py's existing accumulator + timing + `extract_node_details` usage into the shared reporter and have main.py *call* it (single source of truth), or build the reporter alongside (two accumulators on one stream — divergence risk). Recommended: extract.
3. **Per-node timing source.** Use each node's **exact in-node `time.time()` self-timing** (already measured/logged per node) as the `node_timings` source — exact, immune to the parallel-branch/multi-yield problem — and use inter-yield timestamps only for total wall-clock; vs. fix the stream-delta inter-yield approach. Recommended: in-node self-timing.
4. **Error attribution.** Accept stream-delta-origin attribution (approximate; lost in `run_discovery()`), or change nodes to emit structured node-tagged errors.
5. **Retention policy specifics.** A bound is required (see Delivery); choose the mechanism — keep last N pairs, total-size cap, or time-based TTL — and who/what prunes.
6. **Cost vs Langfuse + pricing-map.** Confirm Langfuse does not already attribute cost per pipeline node before maintaining a parallel hand-kept pricing map; pin pricing-map ownership, "last verified" date, and the model ids actually in use (note `ModelUsageRecord.model_name` is free-form, e.g. `anthropic/claude-sonnet-4-…`). Also decide whether to backfill the unused `cost_usd` DB column or keep cost disk-only.
7. **JSON schema versioning** + where the schema is documented (`report_version` field).
8. `ModelUsageRecord` is a **frozen** Pydantic model — the reporter must `model_dump()` it when serializing (a likely cause of the AC-4 serialization failure if missed).
