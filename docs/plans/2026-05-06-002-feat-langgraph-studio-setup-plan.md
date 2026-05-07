---
title: "feat: Set up LangGraph Studio for local pipeline development"
type: feat
status: active
date: 2026-05-06
---

# feat: Set up LangGraph Studio for local pipeline development

## Overview

Add LangGraph Studio configuration so developers can run `langgraph dev` locally and connect to the browser-based Studio UI for visualizing and debugging the KRAKEN discovery pipeline. This is dev tooling only — no production impact.

## Problem Frame

The 10-node discovery pipeline (Intake → Entity Resolution → Triage → [Direct KG | Cold-Start] → Pathway Enrichment → Integration → [Temporal] → Synthesis → Literature Grounding → END) has no visual debugging tool. LangGraph Studio provides real-time graph topology visualization and node execution streaming via a browser UI at `smith.langchain.com/studio/`, connecting to a local server on port 2024.

## Requirements Trace

- R1. `langgraph-cli[inmem]` installed as a dev dependency via `uv`
- R2. `backend/langgraph.json` correctly references `build_discovery_graph` entry point
- R3. `.env` is gitignored and has guidance for required keys
- R4. Graph imports cleanly and `langgraph dev` starts without crashing
- R5. CLAUDE.md documents Studio setup for other developers
- R6. All changes committed to `dev` branch only

## Scope Boundaries

- Do NOT modify graph code (`builder.py`, nodes, state) — config only
- Do NOT install Docker, Redis, or Postgres — `[inmem]` mode avoids all of that
- Do NOT commit `.env` files
- Do NOT modify FastAPI server (`main.py`) or production deployment configs
- Do NOT SSH into remote servers
- Graph invocations will fail without a Claude OAuth session. Topology visualization (the success criterion) only requires that `langgraph dev` starts and Studio loads the graph structure — no OAuth needed for that

## Context & Research

### Relevant Code and Patterns

- `backend/pyproject.toml` — uses `hatchling` build system, `uv` for dependency management, dev deps in `[dependency-groups] dev`
- `backend/src/kestrel_backend/graph/builder.py` — `build_discovery_graph() -> StateGraph` returns `workflow.compile()` (a `CompiledStateGraph`)
- `backend/.env.example` — exists with `KESTREL_API_KEY`, `HOST`, `PORT`, etc.
- Graph nodes use `claude_agent_sdk` (Claude Code SDK with OAuth) — NOT LangChain's `ChatAnthropic`. No `ANTHROPIC_API_KEY` is needed.
- `.gitignore` — already ignores `backend/.env` (line 10)
- Package uses `src/` layout — Python import path is `kestrel_backend.graph.builder`, file path is `./src/kestrel_backend/graph/builder.py`

### Institutional Learnings

- Project follows config-flagged patterns with documented rationale per config field (from `docs/solutions/best-practices/langgraph-pipeline-production-formalization.md`)
- `uv sync` / `uv run` is the standard for dependency management

## Key Technical Decisions

- **`[inmem]` install variant**: Avoids Docker/Redis/Postgres infrastructure entirely. The in-memory mode is sufficient for local visualization and debugging.
- **`langgraph.json` graph reference format**: Uses dotted-import format `kestrel_backend.graph.builder:build_discovery_graph` (not file path — file-path format breaks relative imports with `src/` layout). The function already returns a compiled graph, which is what Studio expects.
- **Dev dependency only**: `langgraph-cli[inmem]` goes in `[dependency-groups] dev`, not main dependencies — it's a dev tool.
- **CLAUDE.md placement**: New "LangGraph Studio" section fits after "Commands" section, as it's a dev workflow tool.
- **No `ANTHROPIC_API_KEY` needed**: The project uses Claude Code SDK (OAuth-based auth), not a direct API key. `ANTHROPIC_API_KEY` should NOT be added to `.env.example` — it would be misleading.
- **LangSmith account required for Studio UI**: A free LangSmith account at `smith.langchain.com` is needed to access the browser-based Studio UI. The `langgraph dev` server itself starts without any LangSmith credentials.
- **LangSmith tracing not needed**: The project already has full observability via Langfuse (`backend/src/kestrel_backend/agent.py`, `main.py`), which traces Claude SDK calls including tool-level spans. `LANGSMITH_API_KEY` / `LANGCHAIN_TRACING_V2` are unnecessary — Studio's value here is topology visualization and interactive debugging, not tracing.

## Open Questions

### Resolved During Planning

- **Does `.env` need updating?** No — `.env.example` exists for reference but `.env` is user-managed and gitignored. The task should verify keys are documented but not overwrite existing files.
- **Is `build_discovery_graph` the right entry point?** Yes — it returns `workflow.compile()` which is a compiled `CompiledStateGraph`, exactly what Studio expects.
- **Does the project need `ANTHROPIC_API_KEY`?** No — the graph uses Claude Code SDK (OAuth). Do not add `ANTHROPIC_API_KEY` to `.env.example`.
- **Is a LangSmith account required?** Yes — a free account at `smith.langchain.com` is needed to access the Studio UI in the browser. The local server starts without it.
- **Is LangSmith tracing needed?** No — the project already has full observability via Langfuse, which traces Claude SDK calls with tool-level spans in both classic mode (`agent.py`) and pipeline mode (`main.py`). Studio's value is visualization and interactive debugging, not tracing.

### Deferred to Implementation

- **Will all node imports resolve cleanly?** Some nodes may have runtime import dependencies that fail at import time. The implementer should run the import test and fix any issues encountered.
- **Exact Studio URL format**: The URL printed by `langgraph dev` may vary by version. Implementer follows whatever the CLI outputs.

## Implementation Units

- [x] **Unit 1: Add langgraph-cli dev dependency**

  **Goal:** Install `langgraph-cli[inmem]` as a dev dependency

  **Requirements:** R1

  **Dependencies:** None

  **Files:**
  - Modify: `backend/pyproject.toml`

  **Approach:**
  - Run `cd backend && uv add --dev "langgraph-cli[inmem]"`
  - Verify with `uv run langgraph --version`
  - This will update both `pyproject.toml` and `uv.lock`

  **Patterns to follow:**
  - Existing dev dependency group in `backend/pyproject.toml` under `[dependency-groups] dev`

  **Test scenarios:**
  - Happy path: `uv run langgraph --version` prints a version number without error

  **Verification:**
  - `langgraph-cli` appears in dev dependencies in `pyproject.toml`
  - `uv run langgraph --version` succeeds

- [x] **Unit 2: Create langgraph.json and verify graph import**

  **Goal:** Create Studio configuration and confirm the graph builds cleanly

  **Requirements:** R2, R3, R4

  **Dependencies:** Unit 1

  **Files:**
  - Create: `backend/langgraph.json`

  **Approach:**
  - Create `backend/langgraph.json` with `"graphs": {"kraken_discovery": "./src/kestrel_backend/graph/builder.py:build_discovery_graph"}`, `"dependencies": ["."]`, `"env": ".env"`
  - **Graph reference fallback:** If the file-path format fails, try the dotted-import format: `"kestrel_backend.graph.builder:build_discovery_graph"`. The `src/` layout with hatchling can resolve differently depending on `langgraph-cli` version.
  - **Missing `.env` handling:** If `backend/.env` does not exist (fresh clone), test whether `langgraph dev` starts anyway. Preferred remediation if it fails: keep `"env": ".env"` in `langgraph.json` and document `cp .env.example .env` as a prerequisite in CLAUDE.md, noting that topology visualization works without real keys filled in.
  - Test graph import: `uv run python -c "from kestrel_backend.graph.builder import build_discovery_graph; g = build_discovery_graph(); print(f'Graph built: {len(g.nodes)} nodes')"`
  - If import fails, diagnose and fix import-time issues (not graph logic)
  - Verify `.env` is already gitignored (it is — `.gitignore` line 10)

  **Patterns to follow:**
  - LangGraph Studio config format (JSON with `graphs`, `dependencies`, `env` keys)

  **Test scenarios:**
  - Happy path: Graph import prints "Graph built: 10 nodes" (or current node count)
  - Error path: If import fails, error message identifies the missing module or circular import

  **Verification:**
  - `backend/langgraph.json` exists with correct graph reference
  - Graph imports and builds without error
  - `.env` remains gitignored

- [x] **Unit 3: Start langgraph dev and verify Studio connects**

  **Goal:** Confirm `langgraph dev` starts and Studio UI renders the graph topology

  **Requirements:** R4

  **Dependencies:** Unit 2

  **Files:**
  - No file changes — runtime verification only

  **Approach:**
  - Run `cd backend && uv run langgraph dev` from the backend directory
  - Confirm server starts on port 2024
  - Open the printed Studio URL in browser
  - Verify graph topology is visible (10 nodes with connections)
  - Success = topology renders. Invocations will fail without Claude OAuth session — that's expected.
  - Note: Accessing the Studio URL at `smith.langchain.com/studio/` requires a free LangSmith account.

  **Test scenarios:**
  - Happy path: Server starts, prints Studio URL, graph topology visible in browser
  - Error path: If Docker-related errors appear, confirm `[inmem]` variant was installed correctly

  **Verification:**
  - `langgraph dev` starts without crashing
  - Studio URL loads and shows graph topology

- [x] **Unit 4: Update CLAUDE.md and commit to dev**

  **Goal:** Document Studio setup and commit all changes to `dev` branch

  **Requirements:** R5, R6

  **Dependencies:** Unit 3

  **Files:**
  - Modify: `CLAUDE.md` (project root)

  **Approach:**
  - Add "LangGraph Studio (Local Development)" section to CLAUDE.md after the "Commands" section
  - Include: setup commands, configuration file references, LangSmith account prerequisite, `cp .env.example .env` prerequisite for fresh clones, note that LangSmith tracing is unnecessary (Langfuse already covers observability), and notes about dev-only scope
  - Stage `backend/langgraph.json`, `backend/pyproject.toml`, `backend/uv.lock`, and `CLAUDE.md`
  - Do NOT stage `.env` — verify it's excluded
  - Commit to `dev` branch with descriptive message
  - Delete `AGENT-TASK-langgraph-studio.md` (task file cleanup)

  **Patterns to follow:**
  - Existing CLAUDE.md section structure (heading + subheadings + code blocks)

  **Test scenarios:**

  Test expectation: none -- documentation and commit, no behavioral change

  **Verification:**
  - CLAUDE.md has Studio section with setup instructions
  - Commit is on `dev` branch
  - `.env` is not staged
  - `AGENT-TASK-langgraph-studio.md` is deleted

## System-Wide Impact

- **Interaction graph:** None — Studio runs as a separate process from FastAPI, no code coupling
- **Error propagation:** Import-time failures in graph nodes will surface when starting `langgraph dev` — these need diagnosing but are pre-existing issues
- **State lifecycle risks:** None — in-memory mode, no persistent state
- **API surface parity:** Not applicable — dev tooling only
- **Unchanged invariants:** FastAPI server (`main.py`), production deployment, graph logic, all node implementations remain untouched

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Graph nodes have import-time failures that block `langgraph dev` | Run import test in Unit 2; fix import issues only, not graph logic |
| `langgraph-cli` version incompatibility with existing `langgraph>=0.2.0` | `uv` will resolve version constraints; if conflict, report to user |
| Invocations fail without Claude OAuth session | Expected — topology visualization is the success criterion, not invocations |
| Developer confusion about tracing setup | CLAUDE.md documents that Langfuse handles observability; LangSmith tracing env vars are not needed |

## Sources & References

- Task file: `AGENT-TASK-langgraph-studio.md` (to be deleted after completion)
- Related code: `backend/src/kestrel_backend/graph/builder.py` — `build_discovery_graph()`
- Related code: `backend/pyproject.toml` — dependency management
- LangGraph Studio docs: LangChain documentation for `langgraph dev` and `langgraph.json` configuration
