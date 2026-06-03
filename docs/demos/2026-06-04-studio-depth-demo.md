# LangGraph Studio Demo — Reasoning Depth (one-hop / multi-hop / subgraph)

**Audience:** 6/4 data-science meeting · **Plan:** `docs/plans/2026-05-30-001-feat-discovery-depth-demo-slice-plan.md`

**What it shows:** a single Studio run of the discovery pipeline in which the `direct_kg` and
`integration` nodes visibly exercise three reasoning modes over the Kestrel knowledge graph:
one-hop associations, **multi-hop mechanistic chains**, and a **connecting subgraph** between entities.

> **Framing (say this on the slide and out loud):** this demonstrates the pipeline's *reasoning
> capability* — that it can traverse the graph at depth and surface mechanistic chains and connecting
> structure. It is **not** a claim that the deeper output is *measurably better* discovery; that
> requires the evaluation work tracked separately (parent plan finding #1). If asked "is this better?",
> answer: *"This shows the capability is wired and produces sensible, hub-filtered structure; whether it
> improves discovery quality is a measurement question we're setting up, not one this demo answers."*

---

## The curated query

**Entity:** type 2 diabetes — `MONDO:0005148` (a well-characterized entity; verified 2026-06-01 to
produce legible output in all three modes against the live graph).

> *Type 2 diabetes mechanisms and connected drug targets*

This single query triggers: one-hop disease/pathway associations in `direct_kg`; multi-hop chains for
the well-characterized entity (e.g. `MONDO:0005148 → CHEBI:28077 → CHEBI:5384`); and, in `integration`,
a connecting subgraph among the resolved entities (≈25 intermediate nodes, hundreds of edges, with
diabetes drugs like troglitazone and targets like PPARG).

## Setup (before the meeting)

1. **Credentials in `backend/.env`:** a **valid `KESTREL_API_KEY`** (data queries 403 without it),
   `ANTHROPIC_API_KEY`, and Claude SDK OAuth fresh (`claude login` if expired — prod/dev share
   `~/.claude`).
2. **Flip the demo flags** (default off so the pipeline is normally inert). For the demo, set the two
   flags to `True` in `backend/src/kestrel_backend/graph/pipeline_config.py`:
   - `DirectKGConfig.multi_hop_enabled = True`
   - `IntegrationConfig.subgraph_enabled = True`
   **Revert both to `False` after the demo** (they gate unmeasured behavior).
3. Launch Studio: `cd backend && uv run langgraph dev` → open the printed Studio URL.

## Dry-run checklist (run once the day before; must pass with margin)

- [ ] `KESTREL_API_KEY` valid — a live `multi_hop_query` returns results, not 403.
- [ ] Claude SDK OAuth fresh (the LLM nodes don't return `AUTH_ERROR`).
- [ ] Both flags `True`; pipeline imported without error (`uv run python -m pytest tests/test_depth_demo_slice.py`).
- [ ] The query produces, in one run: ≥1 one-hop finding, ≥1 `direct_kg_multi_hop` finding with a
      non-empty `logic_chain`, and ≥1 subgraph Bridge in `integration`.
- [ ] **Margin:** all three modes visible **and** end-to-end run completes in a comfortable wall-clock
      window for a live audience (no node hanging on a slow KG call).
- [ ] **Recorded fallback captured** — screen-record a known-good run; this recording is the *primary*
      artifact shown on stage. Go live only if the dry run passed with margin.

## Narration script (per on-screen node)

| Node | What the audience sees | What to say |
|------|------------------------|-------------|
| `pipeline_init` / `intake` / `entity_resolution` | the query resolves to `MONDO:0005148` | "We resolve the question to a knowledge-graph entity." |
| `triage` | the entity classified **well-characterized** | "Triage sees this is a dense, well-studied entity — so it's eligible for deeper traversal." |
| `direct_kg` — one-hop | disease/pathway associations | "First, direct one-hop associations — what the graph directly links to diabetes." |
| `direct_kg` — **multi-hop** | findings with `source=direct_kg_multi_hop` and a `logic_chain` like `MONDO:0005148 → CHEBI:… → CHEBI:…` | "Now the depth: multi-hop **mechanistic chains** — diabetes → intermediate → target — hub nodes filtered out **in the query itself** so we don't resurface generic hubs." |
| `integration` — **subgraph** | a Bridge: *"Connecting subgraph among …"* with intermediate-node and edge counts | "And the connecting **subgraph** — the structure linking the entities, e.g. diabetes drugs and their targets." |
| `synthesis` | the report incorporating the above | "All of this feeds the synthesis report." |

## If the core build / flags aren't ready (fallback)

If multi-hop or subgraph can't be shown (flag/key/Studio issue): demo the shipped **one-hop** pipeline
live, show this doc's recorded run for multi-hop + subgraph, and present the depth modes as "wired and
verified against the live graph on 2026-06-01" with the plan as evidence.
