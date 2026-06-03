---
title: "feat: Langfuse pipeline observability for live demo"
type: feat
status: active
date: 2026-06-03
origin: docs/brainstorms/langfuse-pipeline-observability-demo-requirements.md
---

# feat: Langfuse pipeline observability for live demo

## Overview

Upgrade the KRAKEN discovery pipeline's Langfuse tracing so a benchmark prompt submitted
at `kraken.expertintheloop.io` (pipeline mode) streams a clean, graph-accurate trace into
Langfuse Cloud live — per-node spans with **nested LLM generations carrying token + cost**.
This is the observability half of the demo; LangGraph Studio (run locally) is the
architecture-exhibit half and needs no code change beyond a topology sanity check.

## Problem Frame

The goal is a live demo: type a prompt in the production chat, watch the discovery pipeline
unfold in Langfuse (see origin: `docs/brainstorms/langfuse-pipeline-observability-demo-requirements.md`).

**Correction to the origin doc's premise (discovered during planning research):** the origin
doc states the pipeline is "not instrumented — Langfuse sees nothing from it." That was based
only on `runner.py` (which has no callbacks). In fact `main.py:handle_pipeline_mode`
(lines ~478–693) **already** wraps pipeline runs in *manual* Langfuse spans: it creates a
`discovery_pipeline` trace, opens a `node_{name}` child span per streamed node event, sets
`session_id`, sends `trace.id` to the client, and `/api/feedback` scores against that id.

So the real work is **upgrading** existing tracing, not adding it from scratch. Two gaps:
1. The manual node spans are **wall-clock-only** (reconstructed from streamed events in
   `main.py`) — no LLM-level token/cost data, and not graph-accurate.
2. LLM calls inside nodes (Claude Agent SDK via `query_with_usage()`) emit **no Langfuse
   generations** at all.

**Chosen approach (user-confirmed):** replace the manual node spans with the real Langfuse
LangGraph `CallbackHandler` attached at the `astream` call, keep an enclosing current-span in
`main.py` for trace attributes / `trace.id` / feedback, and add LLM generations inside the
shared `query_with_usage()` chokepoint so they auto-nest under each node span via v3's OTel
active-context propagation.

## Requirements Trace

- **R1.** A pipeline-mode query (prod + dev) produces a single coherent Langfuse trace with a
  span per executed node, visible within seconds. *(origin: Success Criteria #1)*
- **R2.** Each LLM-bearing node shows a nested generation with model, latency, and
  input/output/cache token counts (and cost). *(origin: In Scope #1, trace-depth decision)*
- **R3.** At least one other engineer can access the Langfuse project and see the traces.
  *(origin: In Scope #3 / Success Criteria #2)*
- **R4.** LangGraph Studio renders the discovery topology (10 nodes — the 9 core nodes plus
  `literature_grounding`) locally from `backend/langgraph.json`. *(origin: Success Criteria #3)*
- **R5.** Instrumentation does not couple pipeline latency to Langfuse availability and does
  not truncate streamed traces. *(origin: Open Questions — sync/async)*
- **R6.** Dev and prod traces are distinguishable. *(origin: Open Questions — dev/prod)*
- **R7.** Full prompt/output content is captured now; a governance follow-up is recorded
  before long-term prod reliance. *(origin: Decisions Resolved)*

## Scope Boundaries

- Do **not** change classic-mode tracing (`agent.py`) beyond what is strictly required; it
  works today and is the reference pattern, not a target.
- Do **not** add content masking/scrubbing now (full-capture decision stands); see governance
  follow-up under Documentation / Operational Notes.
- Do **not** alter the `ModelUsageRecord` schema or the AstaBench `model_usages` aggregation.

### Deferred to Separate Tasks

- **Formal data-classification / DPA call** on real-query sensitivity, and revisiting Langfuse
  masking if proprietary/regulated content is in scope: separate governance task before
  long-term prod reliance (origin governance follow-up).
- **Langfuse project member-role hardening + secret rotation runbook**: separate ops task.
- **Running Studio on the server / LangGraph Platform re-architecture / astabench demo**: out
  of scope per origin non-goals.

## Context & Research

### Relevant Code and Patterns

- `backend/src/kestrel_backend/agent.py` (lines ~14, 36–51, 358–589) — **reference v3 pattern**:
  `from langfuse import get_client`; lazy `_get_langfuse()` returns `None` when disabled/keys
  missing; `start_span` / `trace.update_trace(session_id=...)` / child spans / `trace.update`
  / explicit `trace.end()` / surfaces `trace.id` to client. Mirror the guard + lifecycle.
- `backend/src/kestrel_backend/main.py` `handle_pipeline_mode` (lines ~478–693) +
  `_get_pipeline_langfuse()` (lines ~36–44) — existing manual pipeline trace + per-node spans
  to **reconcile (replace)**; `trace.id` → `PipelineCompleteMessage`; `/api/feedback` →
  `langfuse.score(trace_id=...)` (lines ~378–391) must keep working.
- `backend/src/kestrel_backend/graph/runner.py` — `run_discovery` (`ainvoke`, ~line 39) and
  `stream_discovery` (`astream`, ~line 71); neither passes `config`/`callbacks` today.
- `backend/src/kestrel_backend/graph/sdk_utils.py` `query_with_usage()` (lines ~176–292) —
  single chokepoint for all node LLM calls; already accumulates `node_name`,
  `input/output/cache` tokens; `DEFAULT_MODEL_NAME = "anthropic/claude-sonnet-4-20250514"`.
- `backend/src/kestrel_backend/config.py` (lines ~34–38, 92–95) — `langfuse_enabled`,
  `langfuse_public_key`, `langfuse_secret_key`, `langfuse_base_url`.
- `backend/src/kestrel_backend/graph/state.py` — `ModelUsageRecord` (~287–305), `model_usages`
  reducer (~389).

### Institutional Learnings

- `docs/solutions/best-practices/sdk-stdio-mcp-unavailable-migrate-to-http-data-in-prompt-2026-06-02.md`
  — `query_with_usage()` already runs `classify_mcp_degradation` (counts `mcp__*` tool blocks).
  Attach that signal as generation metadata so silently-degraded nodes are visible in traces.
- `docs/solutions/best-practices/langgraph-json-src-layout-import-2026-05-06.md` — Studio
  topology needs the dotted-import form `kestrel_backend.graph.builder:build_discovery_graph`
  with `"dependencies": ["."]`. First thing to check if the DAG doesn't render (R4).
- `docs/solutions/developer-experience/pytest-venv-path-spaces-module-invocation-2026-06-01.md`
  — run tests with `cd backend && uv run python -m pytest ...` (module form), never bare
  `uv run pytest`. If stubbing `langfuse`/`langchain` in `sys.modules`, give package stubs a
  real `__path__` or unrelated test collection breaks.
- `docs/solutions/best-practices/langgraph-pipeline-production-formalization.md` — assessment
  runs make **real** Claude SDK calls and will emit Langfuse traces; guard assessment/test
  runs with `langfuse_enabled=False` to avoid polluting the project.
- `docs/plans/2026-05-07-001-feat-model-usages-cost-tracking-plan.md` — origin of
  `query_with_usage()`; SDK field map `cache_creation_input_tokens → cache_creation_tokens`,
  `cache_read_input_tokens → cache_read_tokens`; model name is not SDK-exposed (constant).

### External References

- Langfuse Python **v3** (installed 3.14.1) is OTel-based. v3↔LangGraph specifics:
  - Import: `from langfuse.langchain import CallbackHandler` (v2 `langfuse.callback` is gone);
    requires the **`langchain`** package installed, not just `langchain-core`.
  - Attach via `astream(state, config={"callbacks": [handler]})`; one trace per invocation,
    span per node. Attach at **one** point only (invoke-time) to avoid duplicate traces.
  - **Nesting:** manual generations created *inside a node coroutine* auto-nest under that
    node's span via OTel active context — **no `trace_id`/parent threading**. Caveat:
    breaks if the LLM call is offloaded to a different thread/detached task.
  - **Trace attributes:** set via an enclosing `start_as_current_observation(as_type="span")`
    + `propagate_attributes(trace_name=, session_id=, user_id=, tags=, metadata=)` around the
    `astream` call (the v3 CallbackHandler constructor takes no trace attributes).
  - **Flush:** background/non-blocking export thread; do **not** flush per-message; call
    `langfuse.flush()` on FastAPI shutdown so a deploy/restart doesn't drop in-flight spans.
  - **Environment:** `LANGFUSE_TRACING_ENVIRONMENT=production|development` distinguishes the
    two deployments within one project (regex `^(?!langfuse)[a-z0-9-_]+$`, ≤40 chars).
  - **Cost:** auto-computed from a recognized `model` + `usage_details`; pass `cost_details`
    only to override.
  - Docs: langfuse.com/integrations/frameworks/langchain ;
    langfuse.com/docs/observability/sdk/upgrade-path/python-v2-to-v3 ;
    langfuse.com/docs/observability/features/environments ;
    langfuse.com/docs/observability/features/queuing-batching

## Key Technical Decisions

- **Real CallbackHandler over enriched manual spans** (user-confirmed): graph-accurate node
  spans + clean OTel nesting for generations. Requires replacing `main.py`'s manual
  `node_{name}` spans to avoid duplicate/overlapping spans.
- **Generations at the shared chokepoint** (`query_with_usage()`), not per-node — one change
  covers all 6 LLM-bearing nodes; `node_name` + token counts are already there.
- **Trace attributes via enclosing current-span + `propagate_attributes`** in `main.py`, which
  also yields the `trace.id` for the client + feedback round-trip (preserves existing wiring).
- **Background export + shutdown flush only** — satisfies R5 (no latency coupling, no
  truncation) without per-request flush.
- **One Langfuse project, environment-tagged** (`LANGFUSE_TRACING_ENVIRONMENT`) — satisfies R6.
  Access-isolation tradeoff (engineers added for the demo can see prod traces) is **accepted
  for now** under the full-capture decision; the governance follow-up covers revisiting it.
- **Let Langfuse compute cost from model + `usage_details`**; fall back to explicit
  `cost_details` (mirroring `agent.py`'s `TurnMetrics` pricing) only if the model string isn't
  recognized.

## Open Questions

### Resolved During Planning

- *Instrumentation approach?* → Real CallbackHandler + enclosing span + generations (above).
- *Sync vs async / flush?* → Background export; no per-message flush; flush on shutdown.
- *Dev vs prod separation?* → One project + `LANGFUSE_TRACING_ENVIRONMENT` tagging.
- *Trace name/session/tags in v3?* → Enclosing span + `propagate_attributes`.

### Deferred to Implementation

- Exact `usage_details` key names for cache tokens against Langfuse's cost schema, and whether
  `anthropic/claude-sonnet-4-20250514` is a Langfuse-recognized model for auto-cost (else pass
  `cost_details`). Verify when wiring the generation.
- ~~Orphaned-generation check~~ — **promoted to a blocking gate in Unit 2** (verify nesting on a
  gather-heavy LLM node: `cold_start` or `pathway_enrichment` — not `direct_kg`, which emits no
  generation). Fallback (explicit parent capture) is pre-scoped into Unit 2.
- **Demo prerequisite to verify (blocking for the live demo, not for the code):** confirm the
  production frontend can actually trigger **pipeline mode** (`agent_mode="pipeline"` over the
  WebSocket). If the chat UI only sends classic mode, the live pipeline demo can't run from the
  website. Check `client/src/` before demo day.
- `LANGFUSE_BASE_URL` (config) vs `LANGFUSE_HOST` (read by `get_client()`) — confirm the
  US-region host is actually set on both servers; reconcile in config so it isn't dead config.

## Implementation Units

- [ ] **Unit 1: Dependencies + Langfuse host/env config**

**Goal:** Make the v3 CallbackHandler importable and the client correctly targeted, and make
required env vars discoverable.

**Requirements:** R1, R6 (enabling)

**Dependencies:** None

**Files:**
- Modify: `backend/pyproject.toml` (add `langchain`; bump `langfuse>=3.0.0`)
- Modify: `backend/src/kestrel_backend/config.py` (reconcile host; read `LANGFUSE_TRACING_ENVIRONMENT`)
- Modify: `backend/.env.example` (add commented `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`,
  `LANGFUSE_HOST`, `LANGFUSE_TRACING_ENVIRONMENT`)
- Modify: `backend/.../assessment/runner.py` (force `langfuse_enabled=False` for assessment
  `run_discovery` calls so live assessments with prod keys present don't pollute the demo project)
- Test: `backend/tests/test_config.py` (or existing config test module)

**Approach:**
- Add `langchain` (not just `langchain-core`) so `from langfuse.langchain import CallbackHandler`
  resolves; bump the stale `langfuse>=2.0.0` pin to `>=3.0.0`. `uv sync` locally + both servers.
  **Pin caution:** the installed `langchain-core` is already 1.x and `langgraph>=0.2.0` is present;
  an unpinned `langchain` resolves to 1.x. Pin `langchain` to a range known-compatible with
  `langfuse 3.14.1`'s handler and the installed `langchain-core`/`langgraph`, and resolve any
  resolver conflict here (deferred-resolver item) — this dependency gates the entire approach.
- Ensure the Langfuse client targets the US host: export `langfuse_base_url` to `LANGFUSE_HOST`
  before `get_client()` is first used, or pass `host=` explicitly — so `langfuse_base_url` stops
  being dead config. Keep `https://us.cloud.langfuse.com` default.
- Surface `LANGFUSE_TRACING_ENVIRONMENT` (read by the SDK from env) and document it in `.env.example`.

**Patterns to follow:** `config.py` existing `get_settings()` env reads; `agent.py:_get_langfuse()`.

**Test scenarios:**
- Happy path: with all `LANGFUSE_*` env vars set, settings expose the expected host/environment.
- Edge case: `LANGFUSE_ENABLED=true` but keys unset → settings still load; no crash (silent-disable
  path intact).
- Edge case: `LANGFUSE_HOST` unset but `LANGFUSE_BASE_URL` set → resolved host is the base_url
  (no dead config).

**Verification (smoke-test gate before Unit 4):** `from langfuse.langchain import CallbackHandler`
imports in the backend venv AND a trivial `astream(config={"callbacks":[handler]})` against a
toy graph produces a trace in Langfuse — confirm both before building on the handler.
`uv run python -m pytest tests/ -m "not integration"` passes.

- [ ] **Unit 2: LLM generations in `query_with_usage()`**

**Goal:** Emit a Langfuse generation per node LLM call with model, input, output, token usage,
and cost — auto-nesting under the active node span.

**Requirements:** R2, R5, R7

**Dependencies:** Unit 1

**Files:**
- Modify: `backend/src/kestrel_backend/graph/sdk_utils.py`
- Test: `backend/tests/graph/test_sdk_utils_langfuse.py` (new)

**Approach:**
- Wrap the SDK call in `langfuse.start_as_current_generation(name=f"llm:{node_name}",
  model=model_name, input=prompt)`; on completion `update(output=text,
  usage_details={input/output/cache token mapping}, metadata={mcp_tool_calls, degradation verdict})`.
- Guard on `_get_langfuse()` returning non-`None` (reuse the `agent.py` guard); when disabled,
  run the SDK call with no generation (zero behavior change, R7 capture only when enabled).
- Map SDK token fields per the model_usages plan (`cache_creation_input_tokens →
  cache_creation_tokens`, etc.). Let Langfuse compute cost from model+usage; note the explicit
  `cost_details` fallback as a deferred item.
- Context-manager must exit (record output/usage) before returning so the generation finalizes.
- **Scope:** the 6 LLM-bearing nodes that call `query_with_usage` are `entity_resolution`,
  `cold_start`, `pathway_enrichment`, `integration`, `temporal`, `synthesis`. (`direct_kg` keeps a
  `model_usages` list but does **not** call `query_with_usage`, so it emits no generation;
  `intake`/`triage`/`literature_grounding` make no LLM call.)
- **BLOCKING nesting gate (promoted from a deferred item — this is load-bearing for R2):** several
  LLM-bearing nodes fan out via `asyncio.gather` (`entity_resolution`, `cold_start`,
  `pathway_enrichment`). v3 nesting relies on OTel active context propagating into those child
  tasks. asyncio copies the `contextvars` context at task creation, so it *likely* holds — but it
  MUST be verified on a gather-heavy node (`cold_start` or `pathway_enrichment`), not a
  single-`await` node, before relying on it. **Pre-scope the fallback in this unit:** if
  generations float to trace root, capture the node's active span/trace context before `gather`
  and attach it explicitly to each generation.

**Patterns to follow:** `agent.py` v3 span lifecycle; `sdk_utils.py` existing usage accumulation;
degradation signal from the stdio-MCP learning.

**Test scenarios:**
- Happy path: Langfuse enabled (mock client) → one generation created with `model=DEFAULT_MODEL_NAME`,
  `usage_details` reflecting input/output/cache tokens from a stubbed `ResultMessage.usage`.
- Nil/disabled path: `_get_langfuse()` returns `None` → no generation created; `(text, record)`
  return value unchanged.
- Error path: SDK `query()` raises → generation context still exits (no leaked/open generation),
  error recorded; exception propagates as before.
- Edge case: usage absent but `mcp_tool_calls > 0` → generation still records metadata; token
  fields zero.
- Integration: token-field mapping matches `ModelUsageRecord` values for the same call (no drift
  between what AstaBench records and what Langfuse shows).

**Verification:** A mocked node call produces a generation with correct token mapping; disabled
path is a no-op.

- [ ] **Unit 3: Thread callbacks config through `runner.py`**

**Goal:** Let callers attach the CallbackHandler to the graph execution.

**Requirements:** R1

**Dependencies:** Unit 1

**Files:**
- Modify: `backend/src/kestrel_backend/graph/runner.py`
- Test: `backend/tests/graph/test_runner_config.py` (new)

**Approach:**
- Add an optional `config: dict | None = None` param to `stream_discovery` and `run_discovery`;
  pass it straight to `astream(initial_state, stream_mode="updates", config=config)` /
  `ainvoke(initial_state, config=config)`. Default `None` preserves current behavior.
- Do not construct the handler here (keep `runner.py` Langfuse-agnostic); the caller owns it.

**Patterns to follow:** existing `runner.py` signatures; LangGraph `config={"callbacks": [...]}`.

**Test scenarios:**
- Happy path: `stream_discovery(query, config={"callbacks":[sentinel]})` → graph `astream` called
  with that config (assert via mock graph).
- Backward-compat: no `config` arg → `astream`/`ainvoke` called without callbacks (unchanged).

**Verification:** Mocked graph receives the config; existing pipeline tests still pass.

- [ ] **Unit 4: Rewire `main.py:handle_pipeline_mode` to the CallbackHandler**

**Goal:** Add the real handler under an enclosing trace span and — once node/generation nesting
is verified — replace the manual per-node spans, preserving `trace.id` → client and feedback.

**Requirements:** R1, R2, R6, R7

**Dependencies:** Unit 1, Unit 2, Unit 3

**Files:**
- Modify: `backend/src/kestrel_backend/main.py`
- Test: `backend/tests/test_pipeline_tracing.py` (new or extend existing pipeline test)

**Approach — sequenced so working tracing is never deleted before the new path is proven:**

*Phase A — add handler + enclosing span alongside the existing manual spans (no deletion yet):*
- When `_get_pipeline_langfuse()` is non-`None`, wrap the **entire** `async for event in
  stream_discovery(...)` loop in `with langfuse.start_as_current_observation(as_type="span",
  name="discovery_pipeline") as span:` so the OTel active context stays live across every
  iteration that drives `astream`. **Wrapping only the call (not the loop) detaches node spans** —
  `stream_discovery` is an async generator; `astream` runs lazily during iteration.
- Set trace attributes via the **top-level** `propagate_attributes` context manager
  (`from langfuse import propagate_attributes` — it is NOT a client method): `trace_name`,
  `session_id=str(conversation_id)`, `user_id`, `tags=["pipeline"]`,
  `metadata={"mode":"pipeline","version":"2.0"}` (metadata values must be **strings** — typed
  `Dict[str,str]`). Construct `CallbackHandler()` and pass `config={"callbacks":[handler]}` into
  `stream_discovery`.
- Verify on dev (gather-heavy node, per Unit 2 gate): handler node spans appear, generations nest
  correctly, and the enclosing span's trace id equals the trace the handler writes into.

*Phase B — gated on Phase A verification passing: remove the manual spans:*
- Remove the manual `trace.start_span(name=f"node_{node_name}", ...)` open/update/end blocks (the
  handler now provides node spans). Keep streaming the same `node_update` events to the client.
- **Rollback:** if nesting/identity verification fails, do NOT remove the manual spans — keep them
  and treat the handler swap as a follow-up; the demo can run on the (working) manual spans +
  generations.

*Both phases:*
- Source `trace.id` from the enclosing span (inside the `with` block) for `PipelineCompleteMessage`.
- **Fix the broken feedback path (verified):** Langfuse 3.14.1 has no `.score()` — `main.py`'s
  `/api/feedback` call (`langfuse.score(trace_id=...)`, ~line 383) currently throws `AttributeError`
  swallowed as a warning, so scoring is silently dead today. Change it to the v3
  `langfuse.create_score(...)` (confirm the `trace_id`/scope kwarg shape against 3.14.1) and add a
  test that asserts the score call is actually made (the current mock-based `test_feedback.py`
  masks the missing method).
- Drop the per-request `langfuse.flush()` (moved to shutdown, Unit 5). Keep error-path span closing
  via the context manager.
- When Langfuse is disabled: run `stream_discovery` with no config/handler (full degrade path).

**Execution note:** Start with a test pinning the Phase A contract (handler attached + trace.id
sent, manual spans still present) before touching the existing block; gate Phase B on the live
nesting/identity check.

**Patterns to follow:** `agent.py` enclosing-span + `update_trace(session_id=...)`; existing
`handle_pipeline_mode` client-message flow; the existing `/api/feedback` handler (fixing
`score`→`create_score`).

**Test scenarios:**
- Happy path (enabled): handler constructed and passed into `stream_discovery`; enclosing span
  created with `session_id=conversation_id`; `PipelineCompleteMessage` carries a non-null `trace_id`.
- Regression (Phase B only): no `node_{name}` manual spans are created (assert the old span calls
  are gone).
- Disabled path: `langfuse_enabled=False`/keys missing → no handler, no enclosing span, pipeline
  still streams node updates and completes (no crash).
- Error path: a node raises mid-stream → enclosing span closes, error surfaced to client, no leaked
  spans.
- Feedback (real, not mock-masked): `/api/feedback` invokes `langfuse.create_score(...)` (assert the
  method is actually called with the trace_id + value) — guards against the silent `score()` regression.

**Verification:** A pipeline run (enabled) yields one trace, handler-driven node spans, nested
generations from Unit 2, and a working `trace_id` round-trip; disabled run is unaffected.

- [ ] **Unit 5: Flush on FastAPI shutdown**

**Goal:** Prevent trace loss on deploy/restart without per-request flushing.

**Requirements:** R5

**Dependencies:** Unit 1

**Files:**
- Modify: `backend/src/kestrel_backend/main.py` (lifespan/shutdown)
- Test: `backend/tests/test_shutdown_flush.py` (new) or assert in existing app-lifecycle test

**Approach:**
- In the FastAPI `lifespan` shutdown phase (or `shutdown` event), if Langfuse is enabled call
  `langfuse.flush()` (optionally `shutdown()`), guarded so it no-ops when disabled.

**Patterns to follow:** existing app lifespan/startup in `main.py`; `agent.py` flush usage.

**Test scenarios:**
- Happy path: shutdown with Langfuse enabled → `flush()` invoked (mock client).
- Disabled path: shutdown with Langfuse disabled → no flush call, no error.

**Verification:** App shutdown triggers a single guarded flush.

- [ ] **Unit 6: Server env + demo verification (gating prerequisite)**

**Goal:** Make traces actually appear in Langfuse Cloud for both deployments and confirm the
demo surfaces work end to end.

**Requirements:** R1, R3, R4, R6

**Dependencies:** Unit 4, Unit 5 (deployed to dev first per workflow)

**Files:**
- Modify (server-side, not committed): `backend/.env` on prod (:8000) and dev (:8006) —
  `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST=https://us.cloud.langfuse.com`,
  `LANGFUSE_TRACING_ENVIRONMENT=production|development`
- Verify: `backend/langgraph.json` (dotted-import topology for Studio)

**Approach:**
- This unit is verification/ops, not feature code. Treat key presence + a confirmed test trace as
  a **blocking gate** (recall the silent-failure mode: enabled-but-keyless ⇒ empty Langfuse, no error).
- Deploy via the normal `dev` branch flow first (dev-kraken, :8006), verify, then prod.
- Add the demo's chosen benchmark prompt(s); confirm they route through pipeline mode.
- Add the other engineer(s) to the Langfuse project (R3).

**Test scenarios:** `Test expectation: none — operational verification unit (no behavioral code change).`

**Verification:**
- A pipeline query at `dev-kraken.expertintheloop.io` then `kraken.expertintheloop.io` produces a
  trace within seconds, with node spans + nested generations showing token/cost, tagged with the
  correct environment.
- **Live feedback-identity check:** submit a feedback score for the returned `trace_id` on a dev
  trace and confirm it lands on the **same** trace that shows the node spans/generations (proves the
  enclosing-span id == handler-emitted trace id — a unit test with a mocked client cannot prove this).
- A second engineer can open the project and see the same traces.
- `cd backend && uv run langgraph dev` renders the 10-node DAG locally (per the dotted-import learning).

## System-Wide Impact

- **Interaction graph:** WebSocket `handle_pipeline_mode` → `stream_discovery` (astream) → graph
  nodes → `query_with_usage`. The CallbackHandler hooks graph node events; generations hook the
  shared SDK chokepoint. `/api/feedback` depends on the `trace_id` round-trip — preserved by
  sourcing `trace.id` from the enclosing span.
- **Error propagation:** Langfuse failures must never break the pipeline. The guard
  (`_get_langfuse()` → `None`) plus background export keep tracing entirely off the critical path;
  context managers ensure spans/generations close on node errors.
- **State lifecycle risks:** generations/spans must finalize before stream completion (context
  exit); on deploy/restart, the shutdown flush prevents dropping queued spans.
- **API surface parity:** classic mode (`agent.py`) tracing is unchanged; `query_with_usage` is
  graph-only, so the generation change does not touch classic mode.
- **Unchanged invariants:** `ModelUsageRecord` schema and `model_usages` aggregation (AstaBench
  cost tracking); `/api/feedback` scoring contract; classic-mode `agent.py` trace shape; the
  client-facing `PipelineCompleteMessage.trace_id` field.

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Removing manual node spans breaks the `trace.id` → client / feedback wiring | Source `trace.id` from the enclosing current-span; add a test pinning the round-trip (Unit 4). |
| Duplicate/overlapping node spans | Attach the handler at **one** point only (invoke-time config in `runner.py`); remove all manual `node_{name}` spans. |
| Generations float to trace root for `asyncio.gather` fan-out nodes | Verify nesting on a gather-heavy node (`direct_kg`/`cold_start`); fall back to explicit parent capture if orphaned (deferred item). |
| Silent failure: enabled-but-keyless or wrong host ⇒ empty Langfuse, no error | Unit 6 gating verification (keys + `LANGFUSE_HOST` + live test trace) before the demo. |
| Assessment/test runs emit real traces, polluting the project (note: `langfuse_enabled` defaults to **True** and `assessment/runner.py` calls `run_discovery` without disabling it — the guard is not currently in force) | **Actually wire it** (not aspirational): force `langfuse_enabled=False` for `assessment/runner.py`'s `run_discovery` calls (env or explicit override). Add as a concrete task in Unit 1's config work. |
| `langfuse_base_url` is dead config (host actually from `LANGFUSE_HOST`) | Unit 1 reconciles host resolution so US region is guaranteed. |
| Frontend may not trigger `agent_mode="pipeline"` | Verify `client/src/` before demo day (deferred prerequisite); the live demo depends on it. |

## Documentation / Operational Notes

- **Branch & PR workflow (PR-only, even for dev — per `CLAUDE.md`):**
  1. Create a feature branch off `dev` (e.g. `feat/langfuse-pipeline-observability`).
  2. Open a PR **feature → `dev`**; **Greptile** auto-reviews; address feedback; merge to `dev`.
  3. Merging to `dev` auto-deploys to `dev-kraken.expertintheloop.io` (:8006, `kraken_dev`) via
     `.github/workflows/deploy-dev.yml` — run the Unit 6 **dev** verification there (incl. the live
     nesting + feedback-identity checks; this also gates Unit 4 Phase B).
  4. After dev verification, open a **separate PR `dev` → `main`** for production (auto-deploys to
     `kraken.expertintheloop.io`). Never deploy directly; never push straight to `main`.
- **Server-side prerequisite (Unit 6):** Langfuse keys + `LANGFUSE_HOST` + `LANGFUSE_TRACING_ENVIRONMENT`
  must be set in `backend/.env` on **both** services (these are not in git); confirm a live test trace
  before treating the demo as ready.
- **Governance follow-up (R7):** before relying on prod traces long-term, make the formal
  data-classification / DPA call on real-query content and revisit masking. Full capture is
  intentional for now; the demo itself uses curated benchmark prompts.
- **Access (R3):** decide member roles (least-privilege) when adding engineers; secret
  storage/rotation runbook is a separate ops task (Deferred to Separate Tasks).
- **Testing:** run `cd backend && uv run python -m pytest tests/ -v -m "not integration"`; if
  stubbing `langfuse`/`langchain` in `sys.modules`, give package stubs a real `__path__`.

## Sources & References

- **Origin document:** [docs/brainstorms/langfuse-pipeline-observability-demo-requirements.md](docs/brainstorms/langfuse-pipeline-observability-demo-requirements.md)
- Reference pattern: `backend/src/kestrel_backend/agent.py` (v3 span lifecycle)
- Chokepoint: `backend/src/kestrel_backend/graph/sdk_utils.py:query_with_usage`
- Existing pipeline trace to reconcile: `backend/src/kestrel_backend/main.py:handle_pipeline_mode`
- Learnings: `docs/solutions/best-practices/langgraph-json-src-layout-import-2026-05-06.md`,
  `docs/solutions/developer-experience/pytest-venv-path-spaces-module-invocation-2026-06-01.md`,
  `docs/solutions/best-practices/sdk-stdio-mcp-unavailable-migrate-to-http-data-in-prompt-2026-06-02.md`
- External: Langfuse v3 LangChain/LangGraph integration + v2→v3 upgrade + environments + queuing/batching docs (langfuse.com)
