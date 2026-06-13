# Langfuse Pipeline Observability + Live Demo — Requirements

**Status:** Ready for planning
**Date:** 2026-06-03
**Scope:** Standard

## Goal

Enable a live demo where a benchmark prompt submitted at the production site
(`kraken.expertintheloop.io`) streams its **discovery-pipeline** run into Langfuse in
real time, with LangGraph Studio used **locally** to present the 9-node architecture.

## Background

Two clarifications shaped this scope:

- **LangGraph Studio (`langgraph dev`) is a local interactive debugger.** It runs its
  own in-memory copy of the graph on `:2024` and can only show runs invoked *through
  Studio itself*. Queries submitted via the website run **in-process inside FastAPI**
  (`backend/src/kestrel_backend/graph/runner.py`), so they never reach Studio's server
  and **cannot be mirrored in Studio**.
- **Langfuse Cloud is the shared observability layer** (`https://us.cloud.langfuse.com`,
  configured in `backend/src/kestrel_backend/config.py`). It is the correct tool for
  watching live dev/prod runs and supports multi-user team access.

**Critical current-state finding:** the discovery pipeline is **not instrumented for
Langfuse**. `runner.py` invokes the graph with no observability hook:

> **Correction (added during planning, 2026-06-03):** This is incomplete. `runner.py` has
> no callbacks, but `main.py:handle_pipeline_mode` (lines ~478–693) **already** wraps
> pipeline runs in *manual* Langfuse spans (a `discovery_pipeline` trace + per-node
> `node_{name}` spans + `trace.id` to the client). So pipeline mode already produces a
> live trace — it just lacks LLM-level token/cost and graph-accurate spans. The work is an
> **upgrade**, not net-new. See `docs/plans/2026-06-03-001-feat-langfuse-pipeline-observability-plan.md`.


- `graph.ainvoke(initial_state)` (line ~39) — no `config`, no callbacks
- `graph.astream(initial_state, stream_mode="updates")` (line ~71) — same

The 9 nodes do not route through the traced helpers in `agent.py`. So today Langfuse
sees the **classic agent** but effectively **nothing from the discovery pipeline**.

## In Scope

1. **Instrument the discovery pipeline for Langfuse.** Attach a Langfuse LangChain/
   LangGraph `CallbackHandler` via `config={"callbacks": [...]}` on the `ainvoke`/
   `astream` calls in `runner.py`, so a run traces as a nested tree: one trace per query,
   a span per node (Intake → Entity Resolution → … → Synthesis).
   **Important (verified against code):** the CallbackHandler yields **node-level chain
   spans only**. The nodes call the Claude Agent SDK's `query()` directly (via
   `graph/sdk_utils.py:query_with_usage()`), **not** a LangChain ChatModel — so the
   handler has no hook into those calls. **LLM-level generations, latencies, and token
   counts will NOT auto-nest.** **Decision (2026-06-03): in scope.** Add explicit Langfuse
   generations inside `query_with_usage()` (which already accumulates `node_name` and
   input/output/cache tokens), following the manual-span pattern already used in
   `agent.py` for classic mode, to capture per-node LLM prompts/outputs, latency, and
   token/cost counts.
   The instrumentation must also guard on `settings.langfuse_enabled` + key presence,
   mirroring `agent.py:_get_langfuse()` (returns `None` when disabled), and pass
   `callbacks` only when a handler exists, so local/dev runs without keys don't break.
   The live-chat demo path is `stream_discovery` (`astream`), so that call must be
   instrumented, not just `ainvoke`.
2. **Verify Langfuse is operating in dev and prod.** Confirm `LANGFUSE_PUBLIC_KEY` /
   `LANGFUSE_SECRET_KEY` are set in server `.env` for both `kraken-backend` (prod, :8000)
   and `kraken-backend-dev` (dev, :8006), and that traces actually arrive.
3. **Add other engineers to the Langfuse project** (team/member access).
4. **Run Studio locally** (`cd backend && uv run langgraph dev`) to present the 9-node
   DAG topology as the architecture exhibit.

## Out of Scope (Non-goals)

- Running `langgraph dev` / Studio **on the Lightsail server**.
- Re-architecting onto LangGraph Platform/Server (Docker + Postgres + Redis + licensing)
  so Studio could mirror website traffic.
- Making Studio observe live website queries (not possible without the re-architecture
  above).
- astabench demo — separate track, demoed locally, its own thing.

## Target Demo Flow

1. Open Studio locally → show the 9-node DAG (Intake → … → Synthesis) as the architecture
   blueprint.
2. Go to `kraken.expertintheloop.io` → submit a benchmarking prompt in the live chat.
3. Watch that **production run** stream into Langfuse live — node spans, latencies, token
   counts, LLM calls.
4. *(Optional)* Step through one query **inside Studio** to show node-level state
   inspection — explicitly framed as a separate Studio-native run, not the website query.

## Success Criteria

- A query at `kraken.expertintheloop.io` (prod) and `dev-kraken.expertintheloop.io` (dev)
  each produce a single coherent Langfuse trace covering all executed pipeline nodes,
  visible within seconds of submission.
- At least one other engineer can log into the Langfuse project and see the same traces.
- Studio renders the discovery graph topology locally from `backend/langgraph.json`.

## Key Risks / Unknowns (for planning to resolve)

- **Langfuse keys on the server — unverified + silent-failure mode (GATING).**
  `backend/.env.example` has no Langfuse keys; confirm they're set in the live `.env` for
  both prod and dev **before** the demo. Note the failure mode: `langfuse_enabled`
  defaults to `true`, but if `langfuse_public_key`/`langfuse_secret_key` are unset,
  `get_client()` produces no working client and **traces silently never appear — no
  error, just an empty Langfuse**. Treat "keys present in both server `.env` files AND a
  test trace arrives" as a blocking prerequisite, not a post-hoc check. Also add commented
  `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` entries to
  `backend/.env.example` so future deploys include them.
- **Langfuse SDK version / handler API (RESOLVED — verified in venv).** Installed
  `langfuse` is **3.14.1** (v3); `pyproject.toml` pins the stale `langfuse>=2.0.0` —
  bump to `>=3.0.0`. The v2 path `langfuse.callback` does **not** exist. The v3 path is
  `from langfuse.langchain import CallbackHandler`, but it raises `ModuleNotFoundError`
  today because only `langchain-core` (1.2.14) is installed, **not** `langchain`.
  **Action:** add `langchain` to `backend/pyproject.toml`, `uv sync` locally and on prod
  (:8000) + dev (:8006), then import from `langfuse.langchain`.
- **`langgraph.json` currency.** Verify `backend/langgraph.json` declares all 9 nodes/edges
  of the current discovery pipeline and its entry point references the live builder
  function — Studio's topology view (Success Criteria #3) depends on it.
- **Dev vs prod separation.** Decide whether dev and prod share one Langfuse project
  (tagged by environment) or use two projects, so demo traces are unambiguous. See the
  security trade-off under Open Questions (shared project = added engineers can read prod
  traces).

## Decisions Resolved (2026-06-03)

- **Trace depth:** Both node-level spans AND LLM-level generations with token/cost counts
  (see In Scope #1). Wrap `query_with_usage()` following `agent.py`'s classic-mode pattern.
- **Content-capture posture:** Capture **full content** (prompts + LLM outputs) now to
  move fast for the demo. **Follow-up (before long-term prod reliance):** make the formal
  data-classification / DPA call with the team on real-query sensitivity, and revisit
  Langfuse masking if proprietary/regulated content is in scope. Demo itself uses curated
  benchmark prompts, so it's safe regardless.

## Open Questions

- One Langfuse project for dev+prod (tag by environment) or two separate projects?
  (Security note: a shared project lets added engineers read prod traces — two projects
  mirror the existing prod/dev isolation.)
- Which benchmark prompt(s) to use for the demo? Confirm the chosen prompt routes through
  the **discovery pipeline** (`stream_discovery`), not classic agent mode.
- Sync vs async CallbackHandler / flush timing — avoid coupling prod `astream` latency to
  Langfuse availability and avoid truncating streamed traces.
- Langfuse project member roles (least-privilege) + offboarding; secret storage/rotation
  standard for `.env`.
