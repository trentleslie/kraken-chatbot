---
title: "Kestrel API Reference (KESTREL API v0.1.0)"
type: reference
date: 2026-05-29
source: live OpenAPI spec — https://kestrel.nathanpricelab.com/api/openapi.json
base_url: https://kestrel.nathanpricelab.com/api
mcp_url: https://kestrel.nathanpricelab.com/mcp
auth: "X-API-Key header (KESTREL_API_KEY)"
note: "Captured from the live spec + live probing on 2026-05-29. Re-pull /api/openapi.json to refresh."
---

# Kestrel API Reference

Two interfaces to the same KG:
- **REST** — `https://kestrel.nathanpricelab.com/api` (this doc; OpenAPI at `/api/openapi.json`, Swagger at `/api/docs`)
- **MCP** — `https://kestrel.nathanpricelab.com/mcp` (JSON-RPC; what `kestrel_client.py` / the discovery pipeline call). Tool set mirrors the REST query endpoints (`one_hop_query`, `multi_hop_query`, `get_edges`, …).

**Auth:** `X-API-Key: $KESTREL_API_KEY` (in `backend/.env`).

## Endpoint catalog (22)

### Graph query (POST)
| Endpoint | Body schema | Purpose |
|---|---|---|
| `/one-hop` | OneHopRequest | 1-hop neighbors from start node(s); optional end-node/predicate/category filters |
| `/multi-hop` | MultiHopRequest | Server-side multi-hop paths (singly- or doubly-pinned); `max_path_length`, `beam_width≤20000` |
| `/subgraph` | SubgraphRequest | Connecting subgraph among `node_ids` (paths, not just shared neighbors) |
| `/get-nodes` | GetNodesRequest | Fetch nodes by CURIE; `canonicalize`, `slim`, `truncate_long_fields` |
| `/get-edges` | GetEdgesRequest | Fetch edges by integer `edge_ids`; **`slim` (bool)** controls attribute depth |
| `/canonicalize` | CanonicalizeRequest | Map CURIEs → canonical IDs; `by_input` |

### Search (POST)
| Endpoint | Body | Purpose |
|---|---|---|
| `/text-search` | TextSearchRequest | Lexical node search (`search_text`, `category`, `prefix`, `limit`) |
| `/vector-search` | VectorSearchRequest | Embedding search |
| `/hybrid-search` | HybridSearchRequest | Lexical+vector; `method` selects strategy |
| `/similar-nodes` | SimilarNodesRequest | Nearest nodes to a `node_id` |

### Vocabulary / metadata (GET)
`/traversal-options` (constraint fields + operators) · `/categories` · `/predicates` · `/prefixes` · `/primary-knowledge-sources` · `/aggregator-knowledge-sources` · `/knowledge-levels` · `/agent-types` · `/provided-by` · `/qualifiers` · `/metagraph` · `/health`

## Ranking presets (the sparsity ladder)
`RankingParams.preset` ∈ **`established` · `hidden_gems` · `frontier` · `deep_dive` · `speculative` · `long_shot`** (6). Plus tunable weights: `confidence_weight`, `degree_weight`, `evidence_weight`, `path_length_decay` (and `*_resolved` variants).

## Response depth: `mode`
`one-hop` / `multi-hop` / `subgraph` take **`mode: slim | full | preview`** (default `slim`). `get-edges` / `get-nodes` use a **`slim` boolean**. *Earlier confusion note: it is `mode:"full"` (or `slim:false` on get-edges), NOT `"slim_false"`.*

## Constraints (filtering mechanism)
`Constraint{ field, operator, value }`; operators `eq, ne, gt, lt, gte, lte` (scalar) / `in, not_in` (list). **17 constrainable fields:**
`predicate, knowledge_level, agent_type, primary_knowledge_source, aggregator_knowledge_source, provided_by, qualifiers, confidence, degree, degree_percentile, edge_count, chemical_formula, exact_mass, prefix, end_node_category, intermediate_node_category, upstream_kg`.

## Edge schema (what an edge carries)
Slim edge tuple:
`[subject, predicate, object, qualifiers, primary_knowledge_source, supporting_sources, aggregator_knowledge_source, knowledge_level, agent_type, id]`
`slim:false` adds nested per-source `attributes` and, **on some edges only**, a `publications` list of PMIDs.

- `knowledge_level` ∈ `knowledge_assertion | prediction | logical_entailment | statistical_association | not_provided`
- `agent_type` ∈ `manual_agent | automated_agent | computational_model | text_mining_agent | not_provided`

## ⚠️ Temporal & provenance limitations (probed live 2026-05-29)
**Critical for any time-sliced / publication-date evaluation:**
- **No date field anywhere.** A 120-edge sample around `MONDO:0005148` had **0/120** edges with any date attribute. The KG is a current snapshot with no time dimension.
- **No date constraint.** None of the 17 constrainable fields is temporal — you cannot query "edges known before T."
- **Publications are sparse (~18%).** Only **22/120** edges carried a `publications` list, and those are bare **PMIDs with no date** (would need PubMed eutils PMID→date resolution; earliest-PMID is only a proxy for "first asserted").
- **The undatable ~82%** are predictions / computed / aggregator-only edges (DrugApprovals, HPO entailments, hetionet, disgenet) — exactly the edge types a discovery pipeline competes with.
- **Implication:** a faithful literature-date holdout on this KG is **not directly feasible**; see `docs/solutions/best-practices/verify-temporal-provenance-before-kg-holdout-eval-2026-05-29.md` (the generalizable lesson) and `AGENT-TASK-temporal-eval-port.md` (the workstream).
