# LangGraph Discovery Pipeline: Node → Tool → How/Why

**Date:** 2026-06-03
**Source:** Generated from a 10-agent read-only mapping of `backend/src/kestrel_backend/graph/nodes/*.py` (one agent per node).
**Purpose:** Reference map of which Kestrel KG primitives and LLM/SDK calls each discovery node makes, and how/why — used to scope where a code-on-graph iterative query loop fits (see `docs/code-on-graph-feasibility.md` and `docs/brainstorms/code-on-graph-spike-requirements.md`).

## Pipeline order

```
intake → entity_resolution → triage → [direct_kg | cold_start] → pathway_enrichment
                                                                        ↓
                                                                  ► integration ◄   ← code-on-graph first target
                                                                        ↓
                                                              [temporal] → synthesis → literature_grounding → END
```

## Node → Tool → How/Why

| Node | Tool called | How & why in the workflow |
|---|---|---|
| **intake** | *none (heuristic)* | Pure regex parsing of the raw query — extracts entities, aliases, type hints, study context. No KG or LLM calls; just produces structured fields for downstream nodes. |
| **entity_resolution** | `hybrid_search` | **Tier 1** primary resolution: `limit=1` top hit, score→confidence mapping (>1.5→0.95, >0.6→0.70), parallelized across entities. Also drives Tier 1.5 alias retries and Tier 2 prefetch over spelling variants. |
| | `text_search` | **Tier 2** complement to `hybrid_search` in prefetch — different lexical/semantic matching, merged + deduped by CURIE to broaden recall for chemical-name variants/abbreviations. |
| | Claude SDK *(query_with_usage)* | **Tier 2** LLM **selection** (`allowed_tools=[]`, `max_turns=1`): picks the best CURIE *from the prefetched candidate set* — never searches the KG itself. R1a contract rejects any CURIE not in the candidate set. |
| **triage** | `one_hop_query` | One call/entity in `mode='preview'`, `limit=10000` — returns only the **edge count** (~100ms, no payload). Classifies into well-characterized (≥200) / moderate (20–199) / sparse (1–19) / cold-start (0) to route the next branch. |
| **direct_kg** | `one_hop_query` | **6 parallel calls** (3 categories × 2 presets) — filtered by `biolink:Disease` / `biolink:BiologicalProcess` / `biolink:Gene`, presets `established` + `hidden_gems`, `limit=25`. Extracts disease/pathway associations + PMIDs. *(The fixed fan-out discussed for code-on-graph.)* |
| | `multi_hop_query` | Singly-pinned, `max_hops=2`, `limit=10`, in-query hub guard (`degree < 5000`). Builds mechanistic-chain findings (score ≥ 0.7). Semaphore-gated (`MULTI_HOP_SEMAPHORE=6`). |
| | *none (heuristic)* | Post-query hub guard (drops degree ≥ 5000 end-nodes) + flags high-degree entities to warn downstream of spurious-association risk. |
| **cold_start** | `similar_nodes` | Finds well-characterized **analogues** for a sparse entity (`limit`≈5–10) — bootstraps inference by borrowing connected neighbors from similar entities. |
| | `one_hop_query` | Pulls each analogue's real neighbors (≤50 edges, grouped by predicate) — these become concrete evidence embedded in the inference prompt. |
| | Claude SDK *(query_with_usage)* | Single-turn (`allowed_tools=[]`) reasoning over the analogue evidence to infer associations for the sparse entity. Graceful fallback to Tier-3 speculative findings on low similarity / timeout / no-SDK. |
| **pathway_enrichment** | `multi_hop_query` | **Phase A**: singly-pinned `max_hops=2` from each entity (filtered to edge_count ≥ 20); nodes appearing in ≥2 entity neighborhoods become **shared neighbors**. |
| | `get_entity_connections` *(→ one_hop)* | HTTP prefetch of one-hop neighbors (parallel) embedded in-prompt; gates Phase B if <2 entities have real data (anti-hallucination, issue #44). |
| | Claude SDK *(query_with_usage)* | **Phase B**: reasons over prefetched neighbors (`allowed_tools=[]`) to produce shared neighbors + biological themes. |
| **integration** ⭐ | `multi_hop_query` | **Doubly-pinned bridge detection** (`detect_bridges_via_api`): groups entities by category, picks ≤2 reps/category, `max_hops=3`, **max 3 category pairs**. Produces `Bridge` objects (Tier 2: 1–2 hops, Tier 3: longer). *(The spike's first target.)* |
| | `subgraph_query` | Flag-gated demo feature: hub-filtered connecting-structure query (`max_path_length=2`, `degree < threshold`, `mode='slim'`) → one summary `Bridge` over the top entities. |
| | Claude SDK *(query_with_usage)* | **Gap analysis only** (not bridges): reasons over accumulated findings (`max_turns=5`) for expected-but-absent entities under Open World Assumption. |
| **temporal** | Claude SDK *(query_with_usage)* | Pure LLM classification (`allowed_tools=[]`, `max_turns=3`) of findings as UPSTREAM_CAUSE / DOWNSTREAM_CONSEQUENCE / PARALLEL_EFFECT. No KG calls (longitudinal studies only). |
| **synthesis** | `multi_hop_query` | **Bridge validation** (`validate_bridge_hypotheses`): doubly-pinned `limit=1` existence check per Tier-3 bridge; verified paths upgrade Tier 3→2. *(The validation site the feasibility doc said NOT to prototype at.)* |
| | Claude SDK *(query_with_usage)* | Final report synthesis (`allowed_tools=[]`, `max_turns=1`) over all assembled state; `fallback_report()` if SDK unavailable. |
| **literature_grounding** | *none (KG)* — reuses state PMIDs | Harvests PMIDs already collected by direct_kg/cold_start as free `source='kg'` evidence. |
| | external: OpenAlex / Exa / PubMed / Semantic Scholar | **4 parallel literature searches** per hypothesis; S2 optionally uses an LLM classifier (supporting/contradicting/tangential). Merged + deduped by priority (KG > PubMed > OpenAlex > Exa > S2). |

## What this reveals about code-on-graph scope

- **Every KG-querying node uses the *fetch-then-reason* pattern** — issue a fixed query (or fixed fan-out), then reason with `allowed_tools=[]`. There is **no node where the LLM steers querying based on results**. That's the universal gap code-on-graph fills.
- **`integration` is the cleanest prototype target** ⭐ — its `multi_hop_query` bridge detection is *doubly-pinned* (start + end), so an endpoint-to-endpoint spike maps to it almost directly, and its fan-out is the most arbitrary (`max 3 pairs`, `≤2 reps/category`).
- **`direct_kg` is the highest-volume target** — `one_hop` ×6 **per entity** + `multi_hop` — the biggest aggregate payoff, but a fuzzier success metric and higher blast radius, so it is the **phase-2 port**, not the spike.
- **`synthesis`'s `multi_hop_query` is the trap** — it is a *validation* call (`limit=1`, "does this path exist?"), exactly where an iterate-until-it-returns loop would be confirmation-seeking. Fenced off deliberately.

## Kestrel primitive usage matrix

| Primitive | Nodes that call it |
|---|---|
| `one_hop_query` | triage, direct_kg, pathway_enrichment, cold_start, integration |
| `multi_hop_query` | pathway_enrichment, integration, direct_kg, synthesis |
| `subgraph_query` | integration, direct_kg |
| `similar_nodes` | cold_start |
| `hybrid_search` | entity_resolution, integration |
| `text_search` | entity_resolution |
