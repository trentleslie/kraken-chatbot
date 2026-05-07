---
title: "Triage Node NameError: undefined variable in Tier 2 fallback path"
date: 2026-05-06
category: runtime-errors
module: discovery-pipeline
problem_type: runtime_error
component: tooling
symptoms:
  - "NameError: name 'resolved_entities' is not defined"
  - "Triage node crashes when Tier 1 API edge counting fails and Tier 2 LLM fallback triggers"
  - "Pipeline halts at triage node — downstream nodes (direct_kg, cold_start, etc.) never execute"
root_cause: logic_error
resolution_type: code_fix
severity: high
tags:
  - triage
  - tier2-fallback
  - edge-counting
  - nameerror
  - langgraph-studio
---

# Triage Node NameError: undefined variable in Tier 2 fallback path

## Problem

The triage node's Tier 2 LLM fallback path referenced an undefined variable `resolved_entities` on line 314 of `triage.py`. This caused a `NameError` whenever Tier 1 API edge counting failed for any entity, halting the entire discovery pipeline at the triage stage.

## Symptoms

- `NameError("name 'resolved_entities' is not defined")` at `backend/src/kestrel_backend/graph/nodes/triage.py:314`
- Only triggers when Tier 1 (Kestrel API `one_hop_query`) fails and falls back to Tier 2 (Claude Agent SDK)
- Pipeline stops at triage — no entities are classified, so downstream nodes never execute

## What Didn't Work

N/A — straightforward undefined variable bug. Discovered via LangGraph Studio's node execution visualization, which showed the pipeline progressing through `__start__ → intake → entity_resolution → triage → NameError`. (session history)

This was the first end-to-end execution of the Tier 2 fallback path. Prior test runs and production usage always had Kestrel MCP reachable, so Tier 1 always succeeded. The bug was introduced during pipeline refactoring (SDK boilerplate cleanup) but was never caught because the fallback path had no test coverage. (session history)

## Solution

```python
# Before (line 314)
entity = resolved_entities[idx]

# After (line 314)
entity = valid_entities[idx]
```

## Why This Works

The variable `resolved_entities` was never defined anywhere in `triage.py`. The correct variable is `valid_entities`, defined on line 259:

```python
valid_entities = [e for e in resolved if e.curie and e.method != "failed"]
```

The bug was wrong on two levels:
1. `resolved_entities` is undefined — `resolved` is the raw entity list, `valid_entities` is the filtered subset
2. Even if `resolved` had been used, the indexing would be semantically wrong — `tier1_failed_indices` contains indices into `valid_entities`, not into the full `resolved` list

## Prevention

- **Lint for undefined variables**: Enable pyflakes or pylint rules that catch undefined name references. `ruff check --select F821` specifically catches this class of bug.
- **Test the Tier 2 fallback path**: Add a test that mocks Tier 1 API calls to fail, forcing the Tier 2 LLM path to execute. The current test suite only exercised Tier 1 success paths.
- **Use LangGraph Studio for end-to-end debugging**: This bug was discovered via Studio's node execution visualization. Running the pipeline locally (where Kestrel MCP may not be reachable) naturally exercises fallback paths that production rarely hits.

## Related Issues

- `docs/solutions/best-practices/langgraph-pipeline-production-formalization.md` — documents triage semaphore configuration and pipeline architecture
- `docs/plans/2026-05-06-002-feat-langgraph-studio-setup-plan.md` — the Studio setup that led to discovering this bug
