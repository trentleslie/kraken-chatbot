---
title: "Stale `direction` arg makes MCP one_hop_query error → triage buckets every entity as cold_start"
date: 2026-06-11
category: logic-errors
module: discovery-pipeline
problem_type: logic_error
component: triage
severity: critical
applies_when:
  - "All/most entities classified as cold_start regardless of how well they resolved"
  - "Triage edge counts are 0 / Tier 1 'API isError' in logs"
  - "Calling a Kestrel MCP tool with an argument the REST endpoint tolerates"
tags:
  - kestrel
  - mcp
  - triage
  - cold-start
  - direction-param
  - request-validation
  - mcp-vs-rest
related:
  - .claude/skills/kestrel-api/SKILL.md
  - docs/solutions/logic-errors/kestrel-multi-hop-response-shape-paths-key-2026-06-11.md
---

# Stale `direction` arg → MCP error → triage cold_starts everything

## Symptom

Every (or nearly every) resolved entity is classified `cold_start`, regardless of resolution quality —
e.g. a 25/25 cold-start incident. Logs show Tier-1 triage "API isError".

## Root cause

`triage.py:76` (`count_edges_via_api`) calls the Kestrel **MCP** `one_hop_query` with a stale
`"direction": "both"` argument:

```python
result = await call_kestrel_tool("one_hop_query", {
    "start_node_ids": curie, "mode": "preview", "direction": "both", "limit": 10000,
})
```

The current Kestrel **MCP** tool layer **strictly rejects unknown keyword arguments**. `direction` is
not a real parameter, so the call returns `isError=True` with `"Unexpected keyword argument"`.
`count_edges_via_api` treats `isError` as a *transient* failure, retries once with identical args (same
deterministic error), exhausts attempts, and `return None`. The caller defaults a `None` score to
`cold_start`. Result: **100% of entities → cold_start**, deterministically.

Verified live 2026-06-11 (`NCBIGene:3949`, via `call_kestrel_tool` / MCP):

| Call | Result |
|------|--------|
| `one_hop_query(..., direction="both")` | `isError=True` — `"Unexpected keyword argument"` |
| `one_hop_query(...)` (no `direction`) | `isError=False`, `results_count=5305` |

## Why it was mis-diagnosed as harmless

The Kestrel **REST** endpoint `/api/one-hop` silently ignores unknown fields (Pydantic ignore-extra) and
returns 200. An investigation that probed **REST** concluded `direction` was "harmless / does not narrow
results." But the pipeline calls **MCP** (`kestrel_client.py` → `KESTREL_MCP_URL`, `tools/call`), which
validates strictly. **The two interfaces validate request args differently — always verify against the
interface the code actually uses (MCP).**

## Fix

Drop `direction` from the `one_hop_query` call in `triage.py:76` (one line). Done on `origin/amy`
(PR #70) — this is a **functional fix for the cold-start incident**, not cleanup. Until it lands, triage
buckets everything as cold_start, so no downstream change (e.g. Biomapper entity resolution) can be
measured — triage erases the signal first.

## Prevention

- Use `.claude/skills/kestrel-api/SKILL.md`; never send `direction` to Kestrel.
- Verify Kestrel request-arg changes against **MCP** (`call_kestrel_tool`), not REST.
- Consider making `count_edges_via_api` distinguish a **deterministic** `isError` (e.g. validation /
  "Unexpected keyword argument") from a transient one — retrying identical args against a deterministic
  rejection only delays the cold_start default and hides the real cause. A validation `isError` should
  fail loudly, not silently degrade to cold_start.
