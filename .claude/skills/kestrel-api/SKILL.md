---
name: kestrel-api
description: Use when calling the Kestrel biomedical knowledge graph (multi_hop_query, one_hop_query, subgraph_query, get_edges via call_kestrel_tool / the MCP server), parsing its JSON responses, or debugging missing/duplicate/over-validated bridges, empty shared-neighbor counts, or wrong path results.
---

# Kestrel API

## Overview

Kestrel is the biomedical knowledge-graph backing the discovery pipeline, reached two ways (same schemas):
- **MCP** `https://kestrel.nathanpricelab.com/mcp` — what `kestrel_client.py` / the pipeline call.
- **REST** `https://kestrel.nathanpricelab.com/api` — OpenAPI at `/api/openapi.json`, Swagger at `/api/docs`.

Auth: `X-API-Key: $KESTREL_API_KEY`. Full endpoint catalog: **`docs/kestrel-api-reference.md`** (re-pull `/api/openapi.json` to refresh).

**⚠️ Two access styles in this repo validate args differently — use the production one to verify.**
- **Production discovery nodes** call `call_kestrel_tool(...)` / `multi_hop_query(...)` from
  `kestrel_client.py` — **MCP-over-HTTP** (HTTP POST of JSON-RPC `tools/call` to `KESTREL_MCP_URL` /`/mcp`).
  This goes through the MCP tool layer, which **strictly rejects unknown keyword arguments**
  (`isError: "Unexpected keyword argument"`). All six SDK nodes were migrated onto this path (off stdio
  MCP) in PR #60 — "HTTP, not stdio-MCP", but still the MCP endpoint with strict validation.
- **The code-on-graph spike** (`backend/tests/code_on_graph_spike/kestrel_rest.py`) calls the **pure REST**
  `/api` endpoints, which **silently ignore** unknown args (Pydantic ignore-extra).

So **verify request-arg behavior with `call_kestrel_tool` (MCP-over-HTTP), not the REST client** — a REST
probe passes an arg the production MCP path rejects, hiding a real failure (this is exactly how the
`direction` cold-start bug was first mis-diagnosed as harmless).

## The #1 gotcha: the response has NO top-level `"paths"` key

A `multi_hop_query` / `subgraph_query` response is:

```json
{ "results": [ { "end_node_id": "MONDO:0005148",
                 "paths": [ ["CHEBI:4167", "MONDO:0005148"] ],
                 "score": 0.81, "degree": 11051, "edge_ids": [130916, ...] } ],
  "nodes": { "CHEBI:4167": {"name": "...", "categories": [...]}, ... },
  "edges": { ... }, "edge_schema": [ ... ] }
```

Two things people get wrong (this bug shipped to prod for 3.5 months across 3 nodes):
1. **`"paths"` is per-result, not top-level.** Read `data["results"]`, then each `result["paths"]`.
2. **Each path is a list of CURIE strings**, e.g. `["CHEBI:4167","MONDO:0005148"]` — **not** a dict with `nodes`/`predicates`. Map CURIEs → names via the top-level `data["nodes"]` dict; get predicates from `edges`/`edge_schema`.

### Never write the silent-fallback parse

```python
# ❌ NEVER — "paths" is absent, so this returns the ENTIRE dict, which is truthy.
#    synthesis: len(dict)>0 → over-validates every bridge to Tier 2 on garbage.
#    integration: dict isn't a list → silently returns ZERO bridges.
paths = data.get("paths", data)

# ✅ Read the real key; fall back to EMPTY, never to the container.
results = data.get("results", [])
if not isinstance(results, list):
    logger.warning("kestrel: unexpected multi_hop shape: %r", list(data)[:5]); results = []
for res in results:
    for path in res.get("paths", []):        # path is a list of CURIE strings
        ...
node_names = data.get("nodes", {})           # CURIE -> {name, categories, ...}
```

**Rule: parse Kestrel responses by the real key, and fall back to an empty value (`[]`/`{}`) that fails loudly via a log — never `.get(key, data)` / `.get(key, body)`, which masks a shape mismatch as "no data".** Correct references already in the repo: `direct_kg.py:549-557` (`body.get("results", [])` → per-result `result.get("paths", [])`) and `backend/tests/code_on_graph_spike/kestrel_rest.py:128` (`parse_paths`).

## Quick reference

| Tool | Response top-level keys | Notes |
|------|------------------------|-------|
| `multi_hop_query` | `results, nodes, edges, edge_schema` | each result: `paths` (CURIE-string lists), `end_node_id`, `score`, `degree`, `edge_ids`. `max_hops`→`max_path_length`, `predicate_filter`→`predicate` in the client wrapper |
| `one_hop_query` (mode `slim`/`full`) | `results, nodes, edges, edge_schema` | each result: `end_node_id`, `edge_ids`, `edge_count`, `degree`, `score` |
| `one_hop_query` (mode `preview`) | `results_count, nodes_count, summary` | counts only, **no** result list — used by triage for edge counts (`results_count`) |
| `subgraph_query` | `nodes, edges` (+results) | connecting subgraph; parse `nodes`/`edges` dicts |
| `get_edges` | edge objects | `slim` (bool) controls attribute depth |

`mode` default is `slim`. `edge_schema` is the column order for the compact `edges` tuples: `[subject, predicate, object, qualifiers, primary_knowledge_source, supporting_sources, aggregator_knowledge_source, knowledge_level, agent_type, id]`.

## Common mistakes

| Mistake | Reality |
|---------|---------|
| `data.get("paths", data)` | No top-level `paths`; this returns the whole dict → over-validates or silently drops. Use `data["results"]`. |
| Treating a path as `{"nodes": [...]}` | A path is a **list of CURIE strings**; names come from `data["nodes"]`. |
| Passing `direction: "forward"/"both"` | **No `direction` param exists**, and the **MCP** tool **rejects** it: `isError=True`, `"Unexpected keyword argument"` (verified live 2026-06-11 on `NCBIGene:3949`). One-hop is already bidirectional. ⚠️ `triage.py` sends `direction:"both"`; its `isError → retry → None → cold_start` fallback then buckets **every** entity as cold_start (the 25/25 cold-start incident). The `direction` field in the unused `kestrel_tools.py` wrapper is a phantom. |
| Mocking `{"paths": [...]}` in a test | That shape Kestrel never returns; it certifies the bug as "passing". Mock the real `{"results": [{"paths": [["A","B"]]}], "nodes": {...}}`. |
| Trusting a `mode:slim` parse for triage counts | `preview` mode returns `results_count`, not a result list. |
| Date/time filters on edges | The KG has **no date field**; publications cover ~18% of edges. Temporal queries are not supported (see reference doc). |

## Verifying a parser

Before trusting a Kestrel parser, assert it against the **real** shape and the empty case:
- real: `{"results": [{"paths": [["A","B"]], "end_node_id": "B"}], "nodes": {"A": {...}, "B": {...}}}` → produces a path/bridge.
- empty: `{"results": [], "nodes": {}, "edges": {}}` → produces nothing, **no** fallback to the dict, no false "path exists".
- malformed: missing `results`, non-list `results`, non-dict body → empty + log, never a raw-dict return or `KeyError`.
