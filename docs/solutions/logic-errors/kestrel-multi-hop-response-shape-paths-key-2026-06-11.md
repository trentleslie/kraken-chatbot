---
title: "Kestrel multi_hop_query parsed with a non-existent top-level \"paths\" key (silent-fallback bug)"
date: 2026-06-11
last_updated: 2026-06-16
category: logic-errors
module: discovery-pipeline
problem_type: logic_error
component: kestrel-client
severity: high
root_cause: wrong_api
resolution_type: code_fix
applies_when:
  - "Parsing a Kestrel multi_hop_query / subgraph_query response"
  - "Building bridges or shared-neighbor counts from KG paths"
  - "Reviewing code that uses .get(key, data) silent fallbacks"
  - "A node silently produces zero bridges, or over-validates every bridge to Tier 2"
tags:
  - kestrel
  - knowledge-graph
  - response-parsing
  - silent-fallback
  - multi-hop-query
  - bridges
  - test-mocks-wrong-shape
related:
  - .claude/skills/kestrel-api/SKILL.md
  - docs/kestrel-api-reference.md
  - backend/tests/test_kestrel_parse.py
  - docs/solutions/logic-errors/kestrel-direction-param-triage-cold-start-2026-06-11.md
---

# Kestrel `multi_hop_query` parsed with a non-existent top-level `"paths"` key

> **Status (updated 2026-06-16): FIXED in PR #72** (merged to `dev`, merge commit `3d54f68`;
> fix commits `5fe0d37` + `4525242`). Resolved as **Unit 2A** of the ground-before-synthesis plan.
> Added one tested helper `kestrel_client.parse_kestrel_response()` (fails loudly to empty, never the
> raw dict) and routed all three nodes through it. Found via Spike 0, which needed real bridge
> hypotheses to A/B against and instead got `bridges=0, hypotheses=0`. Verified: a well-characterized
> panel (urolithin A, spermidine, SIRT1, PINK1, NFE2L2) went **0 → 20 bridges → 102 hypotheses**;
> 38 targeted tests pass; zero new failures vs the pre-existing broken suite (git stash/compare).
> Sections below preserve the original 2026-06-11 diagnosis with statuses flipped to FIXED.

## Summary

Three discovery-pipeline nodes parsed the Kestrel `multi_hop_query` response as
`paths = data.get("paths", data)`. The real response has **no** top-level `"paths"` key — it is
`{"results": [...], "nodes": {...}, "edges": {...}, "edge_schema": [...]}`, and each `result` carries its
own `paths` (a list of **CURIE-string lists**) plus `end_node_id`. Because `"paths"` is absent,
`.get("paths", data)` **silently returns the whole `data` dict** instead of failing — producing garbage
downstream rather than a `KeyError`.

The bug is wrong on **two axes**: (1) top-level key (`paths` vs `results`), and (2) per-path element model
(a path is a list of CURIE strings, not a dict with `nodes`/`predicates`/`node_names`).

## Why it is dangerous (per call site)

| Guard after the parse | Effect of the whole-dict fallback |
|---|---|
| `if paths and len(paths) > 0:` (synthesis) | A non-empty dict is truthy with `len > 0` → **every bridge is "validated" and over-upgraded Tier 3 → Tier 2** on garbage. False confidence in the report. Highest severity. |
| `if not isinstance(paths, list): return` (integration) | The dict isn't a list → **silently returns zero bridges** from every `multi_hop_query`. Feature is dead and looks like "no paths found". |

## Provenance

- **Introduced:** commit `bf07a72` (trentleslie, 2026-02-22, "feat: integrate Kestrel multi_hop_query API
  into discovery pipeline"). The identical line was **copy-pasted into all three node files in one
  commit**. The introduced comment captures the wrong assumption verbatim:
  `# Expected format: {"paths": [...]} or direct list of paths`.
- **Why the guess was plausible:** a `paths` key *does* exist — nested **inside each `result`**
  (`result["paths"]`), not at the top level. The author confused the nested per-result key for a
  top-level one.
- **Why it survived ~3.5 months:** two independent silencers, both still present —
  (a) `backend/tests/test_multi_hop_integration.py` mocked the **wrong** shape `{"paths": [{...}]}`
  (~10 occurrences), so tests exercised and certified the buggy branch; there was no integration test
  against a real Kestrel response. (b) The `isinstance(...)/len(...)` guards turned the shape mismatch
  into a silent "no data" (or, in synthesis, a false "validated") instead of an error.
- **Partial fix:** commit `5d901de` (amykglen, 2026-06-09, PR #70) fixed **`pathway_enrichment.py` only**,
  and (correctly) removed a phantom `direction: "both"` arg from `triage.py`. It did **not** fix
  `integration.py` / `synthesis.py` and did **not** correct the test mocks.

## All instances (exhaustive sweep, 2026-06-11) — now resolved (PR #72)

### Call sites — all FIXED (was buggy)
All three now route through `kestrel_client.parse_kestrel_response()` (no remaining
`data.get("paths", data)` in the codebase — grep-clean):
| File | Function | Resolution |
|---|---|---|
| `backend/src/kestrel_backend/graph/nodes/synthesis.py` | `validate_bridge_hypotheses` | gate is now `if parsed["n_paths"] > 0:` (real path count, not `len(dict)`) → upgrades only genuinely path-verified bridges |
| `backend/src/kestrel_backend/graph/nodes/integration.py` | `parse_multi_hop_result` | rebuilds `Bridge`s from `parsed["paths"]` (CURIE-string `curies` + `names` from the top-level `nodes` dict) — the deeper Bridge-build rewrite this doc flagged is done |
| `backend/src/kestrel_backend/graph/nodes/pathway_enrichment.py` | `find_two_hop_shared_neighbors` | already correct post-#70; now deduped through the helper via `parsed["end_node_ids"]`, preserving its exact reachable-set semantics |

### Test mocks asserting the wrong shape — REWRITTEN
`backend/tests/test_multi_hop_integration.py`'s ~6 wrong-shape mocks (`{"paths": [{"nodes": [...],
"predicates": [...], "node_names": [...]}]}`) were rewritten to the real
`{"results": [{"end_node_id": ..., "paths": [["A","B"]]}], "nodes": {...}}` shape (two pre-existing
wrapper assertions, `max_hops` → `max_path_length`, were also fixed). New
`backend/tests/test_kestrel_parse.py` covers the helper directly: real / empty / malformed /
old-bug-shape / `end_node_ids` / short-path / missing-name cases.

### Correct references (use as the parsing pattern)
- `backend/src/kestrel_backend/graph/nodes/direct_kg.py:549-557` — `body.get("results", [])` → per-result
  `result.get("paths", [])` as CURIE-string lists.
- `backend/tests/code_on_graph_spike/kestrel_rest.py:128-138` — `parse_paths()`; correct-shape spike tests.

### Suspect but safe (chained `.get` that terminates in `[]`, not the container)
- `backend/src/kestrel_backend/graph/nodes/cold_start.py:127, 188` — defensible (degrade to `[]`), low risk.

## Fix pattern (implemented — `kestrel_client.parse_kestrel_response`, PR #72)

The shipped helper reads `data["results"]`, treats each `result["paths"]` as CURIE-string lists, maps
names via `data["nodes"]`, and **falls back to empty + a log, never the raw dict**. It returns
`{"paths": [...], "nodes": {...}, "end_node_ids": [...], "n_paths": N}` so each caller takes what it
needs (`paths` for bridges, `n_paths` for the synthesis validate gate, `end_node_ids` for
pathway_enrichment's shared-neighbor counting). The key guard:

```python
results = data.get("results", [])          # the REAL key — never "paths"
if not isinstance(results, list):
    logger.warning("parse_kestrel_response: 'results' not a list (keys=%s)", list(data)[:5])
    return empty                            # EMPTY ([]/{}), never the raw dict
```

The default in `.get(key, default)` is meant to supply a value of the *same type* for an optionally
absent key; using `data` (the container) conflated "key absent" with "different shape" — two failures
that need opposite handling. Defaulting to typed-empty + a `logger.warning` makes a shape mismatch
visible instead of letting it impersonate a successful-but-empty parse.

**Behavior change is real and per-caller** (confirmed, not asserted neutral): synthesis yields fewer
Tier-2 upgrades; integration **newly emits** bridges it previously dropped (0 → 20 on the verification
panel). This was **Unit 2A** of `docs/plans/2026-06-11-002-feat-ground-before-synthesis-plan.md`,
landed first so the reorder (PR 1) moves an already-correct function.

## Prevention

- Use `.claude/skills/kestrel-api/SKILL.md` when parsing any Kestrel response.
- Never `.get(key, data)` / `.get(key, body)` — fall back to `[]`/`{}` and log on shape mismatch.
- Mock the **real** shape (`{"results": [{"paths": [["A","B"]]}], "nodes": {...}}`) in tests; a wrong-shape
  mock certifies the bug.
- Prefer at least one test/fixture captured from a real Kestrel response for response-shape parsers.

## The `direction` parameter — NOT a red herring (separate, dominant bug)

`one_hop_query` / `multi_hop_query` have **no** `direction` parameter, but the consequence depends on
the interface — and the pipeline uses **MCP**:

- **MCP** (`call_kestrel_tool`, what the pipeline calls): **rejects** an unknown `direction` arg with
  `isError=True`, `"Unexpected keyword argument"`. Verified live 2026-06-11 on `NCBIGene:3949`:
  with `direction:"both"` → `isError`; without → `results_count=5305`.
- **REST** (`/api/one-hop`): silently ignores extras (Pydantic ignore-extra) and returns 200.

An earlier investigation tested **REST** and wrongly concluded "harmless / does not narrow results." On
the **MCP** path that production uses, `triage.py:76` sends `direction:"both"` on every entity →
`isError` → `count_edges_via_api`'s retry (identical args, identical error) → exhausted → `return None`
→ **caller defaults the entity to `cold_start`**. This deterministically buckets **every** resolved
entity as cold_start (the 25/25 cold-start incident) — a **dominant, separate bug** from the
response-parsing bug above. Removing `direction` (PR #70, `triage.py`) **fixes** it; it is not mere
cleanup. **Lesson:** verify Kestrel *request-arg* behavior against MCP, not REST — they validate
differently. See `docs/solutions/logic-errors/kestrel-direction-param-triage-cold-start-2026-06-11.md`.
