---
title: "Langfuse v2 → v3 SDK migration gotchas (OpenTelemetry, LangGraph tracing)"
date: 2026-06-03
category: docs/solutions/best-practices
module: kestrel_backend (observability / Langfuse)
problem_type: best_practice
component: tooling
severity: high
applies_when:
  - Integrating or upgrading the Langfuse Python SDK at v3 (3.x, OpenTelemetry-based)
  - Adding a LangGraph/LangChain CallbackHandler or per-node LLM generations
  - Debugging "Langfuse is enabled but no traces appear" or silently-dead feedback scoring
tags: [langfuse, observability, opentelemetry, langgraph, sdk-migration, tracing, feedback, async]
---

# Langfuse v2 → v3 SDK migration gotchas (OpenTelemetry, LangGraph tracing)

## Context

The KRAKEN backend ran `langfuse 3.14.1` (a ground-up OpenTelemetry rewrite) while the code and
config were still written against the v2 API. Most v2 calls **fail silently** under v3 rather than
erroring loudly, so the integration looked healthy while doing nothing. Wiring real pipeline
observability (a LangGraph `CallbackHandler` + per-node LLM generations) surfaced a cluster of
non-obvious v3 gotchas, each verified against the live SDK/KG. This doc is the quick-reference +
prevention checklist; the working code reference is the implementation plan
`docs/plans/2026-06-03-001-feat-langfuse-pipeline-observability-plan.md` (PR #66).

## Guidance

For each item: **v2 behavior → v3 fix → how to verify.** Several are silent failures.

### 1. `langfuse.score()` was removed → use `create_score()` (silently-dead feedback)
v3 has no `.score()`. The old call raised `AttributeError` that a broad `except` swallowed as a
warning, so `/api/feedback` scoring had been **silently dead**. Mock-based tests masked it
(`test_feedback.py` never exercised the score branch). (session history)

```python
# v2 (silently dead in v3):
langfuse.score(trace_id=tid, name="user_feedback", value=v, comment=c)
# v3:
langfuse.create_score(trace_id=tid, name="user_feedback", value=v, comment=c)
```
Verify: a test that asserts `create_score` is **actually called** (not a MagicMock that auto-creates
any method) — e.g. `assert lf.create_score.called and not lf.score.called`.

### 2. `langfuse.langchain.CallbackHandler` needs the full `langchain` package
`from langfuse.langchain import CallbackHandler` raises
`ModuleNotFoundError: Please install langchain` when only `langchain-core` is present (it is a
transitive dep via langgraph; the full package is **not**). Add `langchain` explicitly. (A reviewer
suggested `langchain-core` suffices — empirically false; the import check requires `langchain`.)
```toml
# backend/pyproject.toml
"langfuse>=3.0.0",   # was the stale ">=2.0.0" while 3.14.1 was installed
"langchain>=0.3.0",  # required by langfuse.langchain.CallbackHandler
```
Verify: `from langfuse.langchain import CallbackHandler; CallbackHandler()` imports + constructs in
the project venv.

### 3. `LANGFUSE_HOST`, not `LANGFUSE_BASE_URL` (dead config → wrong region)
`get_client()` reads **`LANGFUSE_HOST`** from the environment. A v2-era `LANGFUSE_BASE_URL` setting
is silently ignored, so the client falls back to the default (EU) host even though config "set" the
US region. Reconcile: export/derive `LANGFUSE_HOST` (e.g. from the old `LANGFUSE_BASE_URL` if unset)
before the client is first used.
Verify: `get_client().auth_check() is True` and `os.environ["LANGFUSE_HOST"]` is the intended region.

### 4. Enabled-but-keyless = silent empty project
`LANGFUSE_ENABLED` defaulting `true` with **no keys** produces a client that records nothing — no
error, just an empty Langfuse. Treat "keys present **and** a test trace arrives" as a gating check,
and document the keys in `.env.example`. Guard client creation on `enabled and public_key and
secret_key` (mirror `agent.py:_get_langfuse()`).

### 5. Generations nest via OTel active context — but only when created in-context
v3 is OpenTelemetry-based. A manual generation created **inside a LangGraph node** auto-nests under
that node's CallbackHandler span via the OTel active context — **no parent/trace-id threading**.
This propagation **does** survive `asyncio.create_task`/`gather` (LangGraph's executor copies the
contextvars context at task creation) — verified live on a gather-heavy node. Two real caveats:
- A generation from an **errored** run may not export (the run aborts before the span flushes).
- For trace-level attributes, use the **top-level** `propagate_attributes(...)` (a context manager,
  not a client method) inside an enclosing `start_as_current_observation(as_type="span", ...)`, and
  ensure that `with` wraps the **entire** `astream` loop (it is an async generator; wrapping only
  the call detaches the node spans).

### 6. `trace.trace_id` ≠ `trace.id`
On a v3 span object, `.id` is the **span/observation** id and `.trace_id` is the **trace** id. Feedback
scoring (`create_score(trace_id=...)`) and any client round-trip must send **`trace.trace_id`**.
(session history — verified by runtime introspection.)

### 7. Flush is background; flush on shutdown, not per-request
Export is non-blocking (OTel batch processor on a background thread), so do **not** `flush()` per
request (it adds latency and couples the request to Langfuse availability). Call `langfuse.flush()`
once on FastAPI shutdown/lifespan so a deploy/restart doesn't drop queued spans.

### 8. `start_as_current_generation` is deprecated
Prefer `start_as_current_observation(as_type="generation", ...)`; the old typed helper still works
but warns.

### 9. Distinguish environments with `LANGFUSE_TRACING_ENVIRONMENT`
One project, `LANGFUSE_TRACING_ENVIRONMENT=development|production` (regex `^(?!langfuse)[a-z0-9-_]+$`).
The SDK reads it from the env directly; it tags every trace and filters in the UI.

## Why This Matters

The defining trait of this migration is **silent failure**: removed methods are swallowed, dead env
vars are ignored, keyless clients no-op, and `MagicMock`-based tests pass against an API that no
longer exists. An integration can look green (health check "healthy", no exceptions) while recording
nothing or scoring nothing. The only reliable signals are (a) `auth_check()`, (b) a real test trace
viewed in the Langfuse UI, and (c) tests that exercise the **real** v3 client object, not mocks.

## When to Apply

- Bumping `langfuse` across the 2→3 boundary, or finding 3.x already installed under v2-era code.
- Adding LangGraph/LangChain tracing, per-node generations, or feedback scoring.
- Diagnosing missing traces, missing token/cost, or feedback that "submits" but never appears.

## Examples

**Reliability: never let tracing setup break the request.** Wrap CallbackHandler + enclosing-span
setup in `try/except` (a reviewer P1) — on failure, release partially-entered context managers and
degrade to no tracing so the client still gets a response:
```python
if langfuse:
    try:
        handler = CallbackHandler()
        trace = span_stack.enter_context(langfuse.start_as_current_observation(as_type="span", ...))
        span_stack.enter_context(propagate_attributes(trace_name=..., session_id=..., tags=[...]))
    except Exception as e:
        logger.warning("Langfuse trace setup failed; tracing disabled this request: %s", e)
        span_stack.close(); handler = None; trace = None
```

**Testing pitfalls that hid v3 bugs (fold these into your tests):**
- A `MagicMock` trace auto-creates any method, so `trace.start_span()` "passes" even if it didn't
  exist — drive at least one test through the **real keyless v3 client** so v2/v3 API mismatches fail
  loudly. (session history)
- Module-identity: patching `kestrel_backend.graph.runner` while importing `src.kestrel_backend.main`
  targets a *different* module object — the mock never applies. Import via the same package path.
  (session history)
- `@lru_cache`'d `get_settings()` means a `LANGFUSE_ENABLED=false` env override has no effect once
  cached — call `get_settings.cache_clear()` after setting it (e.g. in assessment runners that must
  not pollute the project). (session history)
- Validate new Langfuse tests by running the file **in isolation** — the backend full suite has a
  pre-existing monkeypatch isolation bug (`'kestrel_backend.graph' has no attribute 'graph'`) that
  fails 16–18 tests nondeterministically; new tests look like regressions but pass alone.
  (auto memory [claude])

## Related

- `docs/plans/2026-06-03-001-feat-langfuse-pipeline-observability-plan.md` — working implementation
  reference for every gotcha above (PR #66).
- `docs/brainstorms/langfuse-pipeline-observability-demo-requirements.md` — origin requirements;
  flagged the silent-failure GATING risk and the `langchain` import issue.
- `docs/solutions/best-practices/sdk-stdio-mcp-unavailable-migrate-to-http-data-in-prompt-2026-06-02.md`
  — the shared-layer SDK instrumentation (`classify_mcp_degradation`) now surfaces as generation metadata.
- `docs/solutions/best-practices/langgraph-pipeline-production-formalization.md` — why assessment runs
  must disable Langfuse to avoid polluting the project.
- GitHub: issue #19 (Langfuse session tracking + feedback — exposed the `.score()` regression), PR #66.
