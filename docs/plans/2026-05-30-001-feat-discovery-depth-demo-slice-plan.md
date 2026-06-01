---
title: "feat: Reasoning-depth demo slice (multi-hop + subgraph) for the 6/4 Studio demo"
type: feat
status: active
date: 2026-05-30
deepened: 2026-05-30
origin: docs/brainstorms/kestrel-deck-deliverables-requirements.md
parent_plan: docs/plans/2026-05-25-001-feat-kestrel-api-depth-plan.md
---

# feat: Reasoning-depth demo slice (multi-hop + subgraph) for the 6/4 Studio demo

## Overview

A **thin, demo-first carve-out** of the kestrel-api-depth plan, created to unblock the 6/4 Studio
reasoning-depth demo (Deliverable 2 of `docs/brainstorms/kestrel-deck-deliverables-requirements.md`)
**independent of the full per-entity ToolStrategy engine**. It adds the two visible reasoning modes the
demo needs — **multi-hop** in `direct_kg` and **`/subgraph`** in `integration` — driven by the entity
classification Triage **already produces** (`NoveltyScore.classification`), not by the new
`tool_strategies` dict.

This deliberately defers, to the parent plan, everything the demo does not need: the `ToolStrategy`
type and per-entity routing (parent Units 1, 4), the `pipeline_init` node + `/traversal-options`/
`/metagraph` (parent Unit 2), `/canonicalize` (parent Unit 3), Cold-Start/Pathway adaptation (parent
Unit 6), and `validation_tier`/attrition (parent Unit 8). It carries forward the parent review's
non-negotiable correctness fixes (hub-awareness, the MCP-tool-name spike, the concurrency cap).

## Problem Frame

The 6/4 deck's depth story (Studio demo) needs one-hop / multi-hop / subgraph reasoning visible on
screen. In the parent plan, the units that deliver this (Unit 5 multi-hop, Unit 7 subgraph) are gated
on the full ToolStrategy engine (Unit 4 → Unit 1/3), so the demo can't ship until ~two-thirds of an
8-unit plan lands — and it inherits the strategy-routing bet whose *quality* the team can't yet measure
(parent D1; review finding #1). The additive multi-hop/subgraph **calls** need none of that: they only
need to know which entities are well-characterized, which Triage already classifies. (See origin:
`docs/brainstorms/kestrel-deck-deliverables-requirements.md`; parent:
`docs/plans/2026-05-25-001-feat-kestrel-api-depth-plan.md`.)

## Requirements Trace

- **D1.** `direct_kg` issues a `multi_hop_query` for entities Triage classified `well_characterized`,
  producing mechanistic-chain `Finding`s (advances parent R5, sans ToolStrategy).
- **D2.** `integration` issues a `/subgraph` call for selected entity pairs, producing connecting-path
  `Bridge`s (advances parent R6, sans ToolStrategy).
- **D3.** Both structural streams are **hub-aware** — paths through high-degree nodes are suppressed or
  down-ranked before becoming `Finding`/`Bridge` (parent D6 / review C2). Non-negotiable. **Mechanism
  (confirmed 2026-06-01):** add a `degree` / `degree_percentile` constraint to the `multi_hop_query` /
  `subgraph_query` `constraints` list — the live API accepts it and filters in-query (no `mode:"full"`
  parse or `get_nodes` lookup needed). See Spike Results (2026-06-01) / RC1.
- **D4.** A curated Studio demo query + narration script + dry-run checklist + recorded fallback that
  visibly exercises one-hop, multi-hop, and subgraph (Deliverable 2; brainstorm S1–S4).

## Scope Boundaries

- **In scope:** Additive multi-hop call in `direct_kg`; additive `/subgraph` call in `integration`;
  hub-aware filtering of both; classification-gating via existing `NoveltyScore`; a concurrency cap on
  the new calls; the Studio demo artifact.
- **Out of scope (this slice):** `ToolStrategy` type + per-entity routing; preset expansion beyond the
  current two; `pipeline_init`/`/traversal-options`/`/metagraph`; `/canonicalize`; Cold-Start/Pathway
  changes; `validation_tier`/attrition; any *quality* claim about depth (this slice is demonstration,
  not measured improvement — see Open Questions).
- **Deferred to the parent plan:** all of the above, plus signal_type (parent D1) and the novelty_signal
  question (parent D3).

## Context & Research

### Relevant Code and Patterns

- **`multi_hop_query()`** — `backend/src/kestrel_backend/kestrel_client.py` (~lines 340–403):
  `start_node_ids`, optional `end_node_ids`, `max_hops`, `predicate_filter`, `limit`. Verified to exist;
  currently used **only** in `integration.py` (doubly-pinned). This slice uses it singly-pinned in
  `direct_kg`.
- **Existing classification** — triage writes the **state key** `well_characterized_curies`
  (`state.py`), already consumed by `direct_kg.py` via `state.get("well_characterized_curies", [])`.
  Read that key; no `tool_strategies` needed, and do **not** re-derive it from `novelty_scores`.
- **One-hop / Tier-1 call pattern + dedup** — `direct_kg.py` (`analyze_via_api`, `_merge_into_deduped`);
  run multi-hop in parallel with existing preset calls.
- **Subgraph parse target** — `integration.py` `detect_bridges_via_api` (~line 527) + `parse_multi_hop_result`
  (~lines 272–360); build `Bridge`s from subgraph paths the same way.
- **Existing hub guard** — `edge_count > hub_threshold` (1000/5000), currently applied to one-hop only.
- **Config pattern** — per-node Pydantic configs in `pipeline_config.py` with `Field(description=...)`.

### Institutional Learnings

- Standalone/new structural results without degree-correction "resurface hubs as findings" (parent D2/D6;
  review C2). Apply the existing hub threshold to multi-hop/subgraph path nodes.
- Tier-1 calls use a bare `asyncio.gather` with **no** semaphore (`SDK_SEMAPHORE` guards only Tier-2);
  the new fan-out must wire its own cap (review C6).
- Tier-2 SDK fallback paths are historically under-tested
  (`docs/solutions/runtime-errors/triage-tier2-undefined-variable-2026-05-06.md`).

## Key Technical Decisions

- **Gate on existing classification, not `tool_strategies`.** The demo needs "well-characterized →
  multi-hop"; `NoveltyScore.classification` already provides it. This removes the dependency on parent
  Units 1/3/4 entirely and avoids the unmeasurable per-entity-quality bet for the deck.
- **Behind config flags, default off → flip for the demo.** `multi_hop_enabled` / `subgraph_enabled`
  default `False`; the demo run flips them. Keeps the change inert for everyone else until measured.
- **Hub-awareness is a first-class acceptance criterion, not a fold-in note.** Every multi-hop/subgraph
  path passes the hub-threshold filter (or degree down-weighting) before emission, with a test.
- **MCP-tool-name spike gates the subgraph unit.** `multi_hop_query` is confirmed in code; `/subgraph`
  is **not** verified as an MCP tool name — run `tools/list` first (review C4). If absent, add a direct
  REST path or cut D2 from the slice and demo one-hop + multi-hop only.
- **`Finding`/`Bridge` unchanged.** No `signal_type` (parent D1, deferred); reuse existing models with
  `source="direct_kg_multi_hop"` and existing `Bridge` fields.

## Open Questions

### Resolved During Planning

- **Can multi-hop/subgraph ship without the ToolStrategy engine?** Yes — gate on existing
  `NoveltyScore.classification`. This is the whole point of the slice.
- **Does `multi_hop_query` exist / is it usable singly-pinned?** Yes (verified, `kestrel_client.py`).

### Deferred to Implementation

- **Is `/subgraph` a callable MCP tool?** Resolve via the `tools/list` spike before D2 (review C4).
- **`/subgraph` response shape** — connecting paths vs shared neighbors; parse accordingly after probing.
- **Multi-hop latency per entity** — cap via `multi_hop_limit` + the new semaphore; profile on the demo query.

### Explicitly NOT resolved here (belongs to the parent plan / finding #1)

- **Whether depth improves discovery *quality*.** This slice demonstrates reasoning depth qualitatively
  (Studio); it makes **no** measured-improvement claim. The outcome metric / kill criteria for the
  per-entity strategy engine stay with the parent plan.

## Implementation Units

- [ ] **Unit 1: Multi-hop in `direct_kg` (classification-gated, hub-aware)**

**Goal:** For `well_characterized` entities, issue a singly-pinned `multi_hop_query` and emit
hub-filtered mechanistic-chain `Finding`s, in parallel with existing one-hop.

**Requirements:** D1, D3

**Dependencies:** None (uses existing classification + existing `multi_hop_query`)

**Files:**
- Modify: `backend/src/kestrel_backend/graph/nodes/direct_kg.py`
- Modify: `backend/src/kestrel_backend/graph/pipeline_config.py`
- Test: `backend/tests/test_depth_demo_slice.py`

**Approach:**
- Read the well-characterized CURIEs from the **existing state key** `state.get("well_characterized_curies",
  [])` (written by triage, already consumed by `direct_kg.py`) — not `tool_strategies`, and not by
  re-filtering `novelty_scores`.
- For each, `multi_hop_query(start_node_ids=[curie], max_hops=2, limit=10)`; run alongside the existing
  one-hop preset calls.
- Parse into `Finding(source="direct_kg_multi_hop", logic_chain=<Drug→Gene→Pathway→Disease>)`. Confirm the
  response (in the chosen `mode`) actually carries `node_names` — `parse_multi_hop_result` returns `[]`
  silently on a wrong shape, so a slim response would yield zero findings (RC2).
- **Hub-aware (D3) — see RC1:** the existing one-hop guard cannot see path interiors. Source
  intermediate-node degree explicitly — preferred: push a `degree`/`degree_percentile` **constraint into
  the `multi_hop_query` call** (constrainable per the API reference); alternative: `mode:"full"` + parse
  per-node degree, or a bounded `get_nodes` degree-lookup (the `pathway_enrichment` pattern). Drop/down-rank
  paths through `> hub_threshold` intermediates before emitting. If degree is unavailable, disable multi-hop
  for the demo (D3 invariant).
- **Concurrency (C6):** wrap the new calls in a per-node `asyncio.Semaphore` (reuse `batch_size`).
- Config: `DirectKGConfig.multi_hop_enabled: bool = False`, `multi_hop_max_hops: int = 2`,
  `multi_hop_limit: int = 10`.

**Execution note:** Add a test for the hub-suppression path first — it is the load-bearing guard.

**Patterns to follow:** `analyze_via_api` task creation; `multi_hop_query` usage in `integration.py`
(adapt doubly→singly pinned); existing one-hop hub-threshold check.

**Test scenarios:**
- Happy path: a `well_characterized` entity → multi-hop call made → `Finding`s with non-empty
  `logic_chain` and `source="direct_kg_multi_hop"`.
- Edge case (D3): a path through a `>hub_threshold` intermediate node → suppressed or down-ranked
  (explicit assertion).
- Edge case: entity not `well_characterized` → no multi-hop call.
- Edge case: multi-hop returns no paths → no findings, no error.
- Error path: multi-hop times out / errors → one-hop results still returned.
- Config: `multi_hop_enabled=False` → no multi-hop calls (default inert).

**Verification:** With the flag on, a well-characterized entity yields hub-filtered multi-hop findings
visible in the node output; one-hop behavior unchanged; no concurrency blow-up on a multi-entity query.

---

- [ ] **Unit 2: `/subgraph` in `integration` (hub-aware) — gated on the MCP-tool spike**

**Goal:** For selected entity pairs, issue `/subgraph` and emit hub-filtered connecting-path `Bridge`s,
augmenting existing bridge detection.

**Requirements:** D2, D3

**Dependencies:** The MCP-tool-name spike (below) must pass; otherwise demo one-hop + multi-hop only.

**Files:**
- Modify: `backend/src/kestrel_backend/graph/nodes/integration.py`
- Modify: `backend/src/kestrel_backend/graph/pipeline_config.py` (**create** `IntegrationConfig` — it
  does not exist yet; register on `PipelineConfig`, add the `get_pipeline_config().integration` load in
  `integration.py` mirroring `direct_kg.py`)
- Test: `backend/tests/test_depth_demo_slice.py`

**Approach:**
- **Spike DONE (2026-05-31, see Spike Results):** the tool is **`subgraph_query`** (not `subgraph`),
  and it exists. It takes `node_ids` (a set), `constraints`, `mode`, `ranking`, `max_path_length`, and
  **returns nodes + edges — NOT enumerated paths**, so `parse_multi_hop_result` does **not** apply.
- For top entity pairs (existing pair-selection logic), `call_kestrel_tool("subgraph_query",
  {"node_ids": [...], ...})`; write a **new parser** for the nodes+edges response → `Bridge`s; run in
  parallel with existing `detect_bridges_via_api`.
- **Hub-aware (D3) — see RC1:** source subgraph intermediate-node degree the same way as Unit 1 (degree
  constraint in the `/subgraph` call, or `mode:"full"`, or `get_nodes` lookup); drop/down-rank hub paths
  before emitting; disable subgraph if degree is unavailable.
- Map subgraph paths to frozen `Bridge` fields: `tier = 2 if hops<=2 else 3`, `novelty="known"`, and
  generate `path_description`/`significance` (required fields) — else Pydantic raises (RC3).
- **No cross-stream dedup in this slice** (multi-hop vs subgraph same-path dedup is parent Unit 7 scope;
  duplicate paths on screen are harmless for a demo).
- **Concurrency (C6):** semaphore (a hardcoded constant for integration — `IntegrationConfig` has no
  `batch_size`) around the new calls. Config: `subgraph_enabled: bool = False`, `max_subgraph_pairs: int = 3`.

**Execution note:** Probe the `/subgraph` response shape (connecting paths vs shared neighbors) before
writing the parser.

**Patterns to follow:** `detect_bridges_via_api`; `Bridge` construction from `parse_multi_hop_result`;
per-node config load in `direct_kg.py`.

**Test scenarios:**
- Happy path: entity pair with a connecting subgraph → `Bridge`s with subgraph-derived paths.
- Edge case (D3): subgraph path through a `>hub_threshold` node → suppressed/down-ranked.
- Edge case: subgraph returns shared neighbors but no connecting path → no bridges added.
- Edge case: `subgraph_enabled=False` → no call (default inert).
- Error path: `/subgraph` errors/times out → existing bridge detection still runs.
- Integration: subgraph bridges appear alongside multi-hop bridges in synthesis input.

**Verification:** With the flag on (and the spike passed), subgraph connections appear in bridge
analysis; existing multi-hop bridge detection unchanged; integration profiled with the new calls.

---

- [ ] **Unit 3: Studio demo query + narration + fallback (Deliverable 2)**

**Goal:** A curated Studio run that visibly exercises one-hop, multi-hop, and subgraph, with a script a
non-author can present and a recorded fallback.

**Requirements:** D4 (brainstorm S1–S4)

**Dependencies:** Units 1 & 2 (for multi-hop + subgraph to be visible). If Unit 2's spike fails, demo
one-hop + multi-hop and present subgraph as "next."

**Files:**
- Create: `docs/demos/2026-06-04-studio-depth-demo.md` (query, narration, dry-run checklist)
- (No production code; Studio is already wired via `backend/langgraph.json`.)

**Approach:**
- Pick a curated query with a well-characterized entity (→ one-hop + multi-hop) and ≥1 entity pair with
  a known connecting path (→ subgraph). Reuse entities validated during Units 1/2.
- Write a narration script mapping each on-screen node to what the audience sees (endpoint, path type).
- Dry-run checklist: env keys present, Kestrel reachable, flags on, query produces multi-hop + subgraph
  findings. Record a screen capture as the live-demo fallback.

**Test expectation:** none — documentation/demo artifact. Verification is the dry run.

**Verification:** A single Studio run (recorded as primary; live only if the dry run passes with margin)
shows one-hop, multi-hop, and subgraph findings in the node outputs; the script lets a non-author present.

## System-Wide Impact

- **Interaction graph:** Additive calls inside `direct_kg` and `integration`; no new nodes, no topology
  change, no entry-point change (unlike parent Unit 2). Synthesis consumes the extra findings/bridges.
- **Error propagation:** New calls follow the existing per-call error pattern; individual failures
  gracefully degrade (the slice falls back to one-hop / existing bridge detection).
- **State lifecycle risks:** None new — reuses existing reducer fields for findings/bridges; no new
  read-before-fork dependencies (the classification is already produced upstream of `direct_kg`).
- **API surface parity:** WebSocket/response shape unchanged; the extra findings are additive.
- **Integration coverage:** Verify multi-hop + subgraph results coexist with one-hop and dedup correctly;
  verify the hub filter actually suppresses a hub path end-to-end (not just in a unit mock).
- **Unchanged invariants:** `Finding`/`Bridge` models, existing one-hop behavior, preset behavior, and
  the full per-entity strategy engine (untouched — deferred to the parent plan).

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| `/subgraph` has no MCP tool name (review C4) | `tools/list` spike gates Unit 2; fall back to one-hop + multi-hop demo, present subgraph as "next" |
| Multi-hop/subgraph resurface hubs as "findings" | Hub-aware filter is a first-class acceptance criterion (D3) with a test, not a fold-in note |
| New fan-out saturates the shared Kestrel server | Per-node semaphore (review C6) + `multi_hop_limit`/`max_subgraph_pairs`; profile on the demo query |
| Demo shows "depth" but it's noise, not insight | Slice claims qualitative demonstration only; no measured-quality claim (finding #1 stays with parent) |
| Divergence from the parent plan | This slice's multi-hop/subgraph logic should be written so the parent's Unit 5/7 can later wrap it with ToolStrategy gating rather than replace it |

## Documentation / Operational Notes

- When this lands, the parent plan's Units 5 & 7 reduce to "wrap the shipped multi-hop/subgraph calls
  with ToolStrategy gating + preset expansion" — note that in the parent plan.
- After the demo, `ce:compound` candidates: the standalone classification-gated structural-call pattern
  and the `/subgraph` MCP invocation recipe.

## Sources & References

- **Origin (Deliverable 2):** [docs/brainstorms/kestrel-deck-deliverables-requirements.md](docs/brainstorms/kestrel-deck-deliverables-requirements.md)
- **Parent plan (full build):** [docs/plans/2026-05-25-001-feat-kestrel-api-depth-plan.md](docs/plans/2026-05-25-001-feat-kestrel-api-depth-plan.md) — see its `## Review Corrections (2026-05-30)` (C2 hub-awareness, C4 MCP spike, C6 semaphore carried into this slice)
- Related code: `backend/src/kestrel_backend/kestrel_client.py` (`multi_hop_query`),
  `backend/src/kestrel_backend/graph/nodes/direct_kg.py`, `.../integration.py`, `.../triage.py`
- Institutional learnings: `docs/solutions/runtime-errors/triage-tier2-undefined-variable-2026-05-06.md`

---

## Review Corrections (2026-05-30)

Code-verified fixes from a multi-persona review. The review confirmed the slice's **core premise is
sound** (`well_characterized_curies` is readable in `direct_kg` with no ToolStrategy; `multi_hop_query`
supports singly-pinned; `IntegrationConfig` genuinely doesn't exist; the spike gate is correct).

### RC1. D3 hub guard is unimplementable as "reuse the one-hop guard" — **P0, must fix before Unit 1**
The one-hop guard iterates `novelty_scores` (input entities only); `parse_multi_hop_result` extracts only
`nodes`/`predicates`/`node_names` — **no per-node degree** for intermediate path nodes. So "reuse the
one-hop hub guard" silently does nothing on path interiors and the demo can resurface hubs as findings —
exactly what D3 prevents. **Fix (folded into Units 1 & 2):** explicitly source intermediate-node degree —
preferred is a `degree`/`degree_percentile` **constraint pushed into the `multi_hop_query`/`/subgraph`
call** (constrainable per `docs/kestrel-api-reference.md`); alternatives are `mode:"full"` + parse, or a
bounded `get_nodes` lookup. **Invariant + test:** if path-node degree is unavailable, **disable** the
structural stream for the demo (don't emit unfiltered); test that a fixture path through a `>hub_threshold`
intermediate is suppressed. *(Note: this also tightens parent-plan correction C2, which said "reuse the
existing hub guard" — same gap; reflect there.)*

### RC2. Multi-hop response shape must be probed in the same spike
`parse_multi_hop_result` returns `[]` silently on an unexpected shape, and `slim` mode may omit
`node_names`/degree that the `logic_chain` rendering and the hub filter both need. **Fix:** the pre-Unit-2
`tools/list` spike must ALSO confirm the singly-pinned `multi_hop_query` response (in the chosen mode)
carries `node_names` + degree; add a test asserting a **non-empty** `logic_chain`, not just that a `Finding`
is emitted.

### RC3. Subgraph → frozen `Bridge` field mapping
`Bridge` is frozen with `tier: Literal[2,3]`, `novelty: Literal["known","inferred"]`, required
`path_description`/`significance`. Map subgraph paths to valid values (`tier=2 if hops<=2 else 3`,
`novelty="known"`) or Pydantic raises.

### RC4. Front-load a candidate-query spike (Unit 0) — demo-existence risk
The demo needs ONE query that simultaneously yields a `well_characterized` entity (→ multi-hop) AND a
connected pair (→ subgraph), legibly. `well_characterized` is ≥200 edges while `hub_threshold` is 5000, so
such entities have dense 2-hop neighborhoods that are *more* likely to traverse hubs (couples with RC1).
**Fix:** add a **Unit 0** — name 2–3 concrete candidate CURIEs and run them through the *current* pipeline
**today** to confirm legible multi-hop + subgraph output exists; pre-commit a fallback query. Run the cheap
`tools/list` `/subgraph` spike on **day 1**.

### RC5. The demo must not assert quality the slice disclaims (D4 acceptance criterion)
A Studio screen showing `Drug→Gene→Pathway→Disease` chains implicitly claims they're meaningful. **Fix:**
make the no-quality-claim framing a **required, testable element of the Unit 3 narration script** — an
explicit on-script line framing depth as "reasoning capability demonstrated, not validated improvement,"
plus a prepared answer to "is this actually better?". Also independently verify one-hop vs multi-hop are
**visually distinct** in Studio before relying on the subgraph-failed fallback (both render as `Finding`s;
the only difference is a longer `logic_chain`).

### RC6. Scope trims (keep the slice thin)
- **`IntegrationConfig`:** for the demo, a minimal module-level config/flag suffices; a full registered
  `PipelineConfig` entry is parent Unit 7's job (parent C7). If created here, mark parent Unit 7 "Modify."
- **Tests:** trim to smoke-level for a flags-default-off demo — happy path, default-off-inert, error-degrade,
  and the RC1 hub-suppression fixture. Full integration coverage (multi-hop+subgraph+one-hop coexistence,
  end-to-end dedup) belongs in the parent's `test_kestrel_api_depth.py`, not here.
- **Hub test framing:** assert "the filter runs and suppresses a known fixture hub path," not a calibrated
  quality threshold (the slice makes no measured-quality claim).

### RC7. Dry-run checklist additions
Add **OAuth token freshness** (prod/dev share `~/.claude`; if expired, Studio loses Claude access) and a
quantified "passes with margin" bar (all three modes visible + acceptable per-node latency) to Unit 3's
dry-run checklist; keep the recorded capture as the definitive fallback.

---

## Spike Results (2026-05-31) — `tmp/spike_kestrel_tools.py`, `tmp/spike_kestrel_schemas.py`

Ran the RC4/RC1/RC2 spikes against the live Kestrel MCP server (`tools/list` succeeded). Findings:

- **`subgraph_query` exists** (21 tools total). The slice's `subgraph` name was wrong → use
  `subgraph_query`. It **returns nodes + edges, not paths** → needs its own parser (Unit 2 updated).
- **RC1 RESOLVED — degree IS sourceable.** Both `multi_hop_query` and `subgraph_query` expose a generic
  **`constraints`** param (objects) **and** **`mode`** (default `"slim"`; `"full"` available); `get_nodes`
  fetches node data by CURIE. So push a degree/`degree_percentile` constraint into the call (preferred),
  or use `mode:"full"`, or a `get_nodes` lookup — all three confirmed present.
- **`multi_hop_query` MCP params** are `max_path_length`/`min_path_length` (default 3/1) + `beam_width`,
  `predicate`, `end_node_category`, `constraints`, `mode`, `ranking` — the `multi_hop_query()` wrapper
  translates `max_hops`; verify the translation when wiring Unit 1.
- **RC8 — BLOCKER: `KESTREL_API_KEY` is INVALID/expired.** `tools/list` works, but a real
  `multi_hop_query` returns embedded error *"Invalid API key. Provide valid key in an X-API-Key header."*
  The header is correct (`X-API-Key`) and a 43-char key is set, so the **key itself is stale**. **This
  blocks RC2** (couldn't observe whether `slim` carries `node_names` / `full` carries degree) **and the
  demo itself** — no live KG query runs until the key is refreshed. *Action: refresh `KESTREL_API_KEY` in
  `backend/.env`, then re-run `tmp/spike_kestrel_schemas.py` with a real query to settle RC2.*

> **RC1/RC2 status:** the *mechanism* is resolved (constraints/mode/get_nodes), but the *response-shape*
> confirmation (RC2) is pending a valid API key. Refresh the key first.

**Key-status probe (2026-05-31, `tmp/spike_kestrel_probe.py`):** the current `KESTREL_API_KEY` authenticates
**nothing that touches graph data** — every data query returns HTTP 403 "Invalid API key." Only
unauthenticated metadata/vocabulary tools respond:
- **Work without the key (6+):** `health_check`, `get_metagraph` (16.3M nodes / 112.5M edges),
  `get_traversal_options`, `get_valid_predicates` / `get_valid_categories` / `get_valid_prefixes`.
- **403 — need a valid key (all data access):** `get_nodes`, `get_edges`, `one_hop_query`,
  `multi_hop_query`, `subgraph_query`, `similar_nodes`, `text_search`, `hybrid_search`, `vector_search`.
(Also: corrected arg names — `one_hop_query`→`start_node_ids`, search
tools→`search_text`, `similar_nodes`→`node_id` singular, `get_edges`→`edge_ids`.)

> **CORRECTION (2026-06-01):** the "key invalid / 403" finding above was a **test-harness artifact** —
> a bare `load_dotenv()` resolved from the wrong directory and sent *no* key (which the server reports
> as "Invalid API key"). With `backend/.env` explicitly loaded, **the key works reliably** (verified:
> 11/11 data calls OK, HTTP 200, no rate-limit headers, identical under burst vs spaced). Data access is
> **not** blocked. The key fingerprint is `len=43`.

## Spike Results (2026-06-01) — all demo-slice unknowns RESOLVED (`tmp/spike_depth_unblock.py`)

- **RC1 (hub guard) — SOLVED, cleanest mechanism.** `degree` and `degree_percentile` are **constrainable
  node fields** (`get_traversal_options` lists them, operators `gt/lt/gte/lte`, scope `node`; plus
  `edge_count` scope `result`, and `intermediate_node_category`). Pushing
  `{"field":"degree","operator":"lt","value":5000}` (and `degree_percentile lte 0.99`) **into the
  `multi_hop_query`/`subgraph_query` `constraints` list was accepted and returned results.** So D3's
  hub filter is an **in-query degree constraint** — no `mode:"full"` parse and no `get_nodes` lookup
  needed. (Multi-hop results also carry a `degree` field for belt-and-suspenders post-filtering.)
- **RC2 (response shapes) — RESOLVED.** `multi_hop_query` result keys: `start_node_ids`, `end_node_id`,
  `paths`, `edge_ids`, `path_count`, `degree`, `score`, `score_components` (same keys in `slim`/`full`;
  `paths` + `edge_ids` give the chain, resolve names via `get_nodes`/`get_edges`). **`subgraph_query`
  returns `{nodes, edges, edge_schema, summary}`** — `nodes`/`edges` are **dicts** (not a `paths` list),
  so Unit 2 needs its **own parser** (edges are arrays: `[subj, predicate, obj, …, sources, level, agent]`).
- **RC4 (demo query) — CONFIRMED viable.** `MONDO:0005148` (type 2 diabetes) yields: multi-hop ranked
  paths (e.g. → Muraglitazar, a PPAR agonist), and a `subgraph_query([MONDO:0005148, NCBIGene:5468])`
  with 5 nodes / 109 edges of real connecting structure (e.g. `CHEBI:8228 —in_clinical_trials_for→
  MONDO:0005148`, troglitazone). One curated diabetes query exercises one-hop + multi-hop + subgraph.
- **Net:** every load-bearing unknown for Units 1–3 is now resolved against the live API. The slice is
  implementation-ready; the hub-guard mechanism (the prior P0) is the simplest of the RC1 options.
