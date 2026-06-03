---
title: "feat: Code-on-Graph go/no-go spike harness"
type: feat
status: active
date: 2026-06-03
origin: docs/brainstorms/code-on-graph-spike-requirements.md
deepened: 2026-06-03
---

# feat: Code-on-Graph Go/No-Go Spike Harness

## Overview

Decide whether an LLM-driven **iterative** Kestrel query-refinement loop ("code-on-graph") beats kraken's **static** query plan at endpoint-to-endpoint bridge discovery — **staged** so the idea can be killed cheaply:

- **Phase 0 — Kill-test (N=50 stratified gold set):** **10 type-2-diabetes + 10 Alzheimer's** hand-curated anchors (direct CURIEs, difficulty-labeled — **already measured & committed**, `backend/tests/fixtures/code_on_graph_spike/gold_set_anchors.json`) **+ 30 randomly-sampled DrugMechDB indications** (crosswalked to Kestrel CURIEs, reachability-filtered). The random 30 carry the McNemar inference; the 20 anchors set R0 + per-stratum relevance. Both arms + a recall gate (paired McNemar + recall-lift + hallucinated-CURIE=0). A recall NO-GO or any hallucinated CURIE **kills the idea here.**
- **Phase 1 — Defensibility layer (runs only on a passing Phase-0 recall signal, toward PROCEED):** the EITL expert precision arm, the treatment-fairness guard, and the reproducibility manifest. (The gold set is already built in Phase 0; Phase 1 only scales it if the powered N exceeds 50.)

Nothing in the production pipeline changes — `integration.py` is touched only *after* a PROCEED verdict.

## Problem Frame

Kraken's discovery nodes all "fetch-then-reason": fixed query (or fixed fan-out), then reason with `allowed_tools=[]`. No node lets the LLM steer querying based on what it finds; code-on-graph is that missing loop. It's a speculative bet — CoG's published wins come from keeping the KG out of the prompt, which kraken already does, so an iterate loop *adds* turns/calls against an already-lean baseline (origin: `docs/brainstorms/code-on-graph-spike-requirements.md`). Staging exists because a fully defensible gate re-implements the L+ build's riskiest component as throwaway and needs a live human EITL campaign — disproportionate to a kill-test. Serves Lance's discovery themes #2 (mechanism of action) and #4 (drug repurposing) — `docs/discovery-pipeline-requirements.md`.

## Requirements Trace

- R1. Standalone throwaway harness; no production node change until PROCEED (origin R1).
- R2. Compare static baseline vs minimal LLM iterate-loop on identical inputs (origin R2).
- R3. Endpoint-to-endpoint: start CURIE + gold-target CURIE → bridge node(s) (origin R3).
- R4. Iterate-loop emits a typed JSON query spec (structured-spec executor); self-corrects to a hard turn cap (origin R4).
- R5–R7. Recall reference = labeled gold bridges. **Phase 0 gold set = 50 stratified**: 10 T2D + 10 Alzheimer's hand-curated anchors (direct CURIEs) + 30 random DrugMechDB indications (crosswalked, Kestrel-reachable hop-agnostically). N target 50, floor = powered-N (origin R5–R7).
- R8. Precision reference = EITL expert votes on off-gold bridges, blinded, ≥2 reviewers/pair + κ — **Phase 1 only** (origin R8).
- R9. Grounding contract: every emitted CURIE must reference a node returned by an executed Kestrel call; hallucinated CURIE = hard fail (origin R9).
- R10. Report recall, false-confirmation (Phase 1), hallucinated-CURIE rate, cost (LLM + Kestrel calls separately) (origin R10).
- P1–P5. McNemar significance, R0 anchor, cost rule, treatment-fairness guard (Phase 1), frozen definitions (origin Pre-Registration).

## Scope Boundaries

- **No production node changes** (`integration.py:detect_bridges_via_api`, `direct_kg.py`, `pathway_enrichment.py`, a future `graph_reasoning` node).
- **No arbitrary-Python sandbox** — structured-spec executor only.
- **No local subgraph materialization / relation-ranking** — Kestrel stays remote.
- **No category-pair framing in the gate** — endpoint-to-endpoint only.
- **Not run at `synthesis.py:validate_bridge_hypotheses`** (confirmation-seeking site).

### Deferred to Separate Tasks
- Category-pair confirmation arm + L+ `graph_reasoning` build: separate plan, only on PROCEED. **Kill-propagation rule:** a Phase-0 recall NO-GO also kills the category-pair arm.
- Repurposing prerequisites (analyte-anchored entry; bridges terminating on drug/intervention nodes): separate discovery-pipeline work.

## Live Probe Findings (2026-06-03) — verified against live Kestrel

These three findings (from building the anchor probe) materially shape the plan:

1. **Use Kestrel REST `/api`, not `/mcp`.** Kestrel's MCP server (`mcp_server.py`) forwards each tool call to its own REST API but **does not reliably pass `X-API-Key`** during streamable-http tool calls → intermittent "Invalid API key" (it is *not* rate-limiting; auth is a plain membership check with no lockout). The REST `/api` (`https://kestrel.nathanpricelab.com/api`, what biomapper2 uses) is rock-solid (8/8). **The harness hits REST `/api` directly** (`/hybrid-search`, `/multi-hop`, `/one-hop`, `/get-nodes`), reusing kraken's *parsing/grounding* helpers but not its MCP client. (Worth reporting the `/mcp` bug upstream — kraken's whole pipeline rides the flaky path.)
2. **The static baseline genuinely misses canonical bridges.** Even with correct resolution, `multi-hop` (limit=100) between drug and disease did **not** recover the gold bridge for metformin→AMPK, memantine→NMDA(GRIN2B), bromocriptine→DRD2, pramlintide→CALCR — the specific gold path is buried below the top-100 ranked paths for hub diseases (T2D deg 11051, AD deg 7771). **This is a strong pro-spike signal** (where iteration could win) **and** a measurement caveat: the baseline's recall ceiling is partly a `limit`/ranking artifact, so the "identical evidence budget" decision must pin the `limit` for both arms and treat it as a frozen knob.
3. **Entity resolution is fragile.** Naive `hybrid-search` top-hit mis-resolved ~5 gene symbols (PPARG, SLC5A2, GRIN1, PSEN1, APP) to **non-human orthologs** (degree 11–15 vs thousands). The committed anchors use **direct canonical human `NCBIGene` CURIEs**; `anchors.py`/`gold_set.py` resolution must be taxon-robust (prefer human / highest-degree / category-filtered), validating the spike's whole emphasis on grounding.

## Context & Research

### Relevant Code and Patterns
- `backend/tests/recall_gate.py` — **the structural template**: standalone `__main__`, `load_dotenv()`, committed JSON fixture + `_threshold`, per-item table, exit 0/1. `_same_entity`/`_ambiguous` via `equivalent_ids` — precedent for "did we recover the gold bridge?" and the 1→many identity problem.
- `backend/src/kestrel_backend/assessment/{runner,cassette,scorer,capture}.py` — argparse runner that continues past failures, `LANGFUSE_ENABLED=false`; respx record/replay keyed on `method:url:md5(body)` (**raises on unmatched body** — governs the cassette decision); scipy stats; Pydantic→JSON.
- **Kestrel REST `/api`** (the access path — finding #1): `POST /hybrid-search {search_text, limit, category}` → `{name:[{id,score,neighbors_count}]}`; `POST /multi-hop {start_node_ids,end_node_ids,max_path_length(2–5),min_path_length,limit,mode}` → `{"results":[{...,"paths":[[curie,…],…]}]}`; `POST /one-hop {…,mode:"preview"}` → degree; `POST /get-nodes` → `equivalent_ids`. Auth: `X-API-Key` header. A thin httpx client replaces the kraken MCP client.
- `backend/src/kestrel_backend/graph/nodes/integration.py` — `parse_multi_hop_result` **hardcodes `paths[:10]`** → the spike needs its own **cap-free shared parser** over `results[].paths[]`, used identically by both arms.
- `backend/src/kestrel_backend/graph/nodes/entity_resolution.py` — `_canonical_curie` + the R1a membership gate: the grounding template (transport-agnostic, reusable over REST).
- `backend/src/kestrel_backend/graph/sdk_utils.py` — `query_with_usage` → `(text, ModelUsageRecord)`; `ClaudeAgentOptions(allowed_tools=[], …)` — **no `temperature` field exists**; tracks tokens, not Kestrel-call/turn counts (harness adds its own).
- `backend/src/kestrel_backend/kestrel_tools.py` — dormant; its **12 tool names + schemas are the enumerated verb whitelist** for the structured-spec executor (the spec maps to REST endpoints).
- `expert-in-the-loop` `shared/schema.ts` (`pairs` 7 required cols; `votes` supersession → κ uses **active votes only**); `ColumnMapper.tsx` auto-maps `llm_*`/`resolution_layer`; `review.tsx` **renders** `llm*` + metadata when `showMetadataPanel` on.

### Institutional Learnings
- `docs/solutions/best-practices/sdk-stdio-mcp-unavailable-migrate-to-http-data-in-prompt-2026-06-02.md` — fetch KG facts over HTTP, reason with `allowed_tools=[]`; never count a fabricated bridge as a recall hit.
- `docs/solutions/developer-experience/pytest-venv-path-spaces-module-invocation-2026-06-01.md` — `cd backend && uv run python -m pytest`.
- `docs/solutions/best-practices/langgraph-pipeline-production-formalization.md` — pre-commit thresholds; cassette replay does NOT remove LLM nondeterminism → variance bands.
- `docs/solutions/best-practices/verify-temporal-provenance-before-kg-holdout-eval-2026-05-29.md` — score recall under OWA.
- `docs/solutions/runtime-errors/triage-tier2-undefined-variable-2026-05-06.md` — force-execute the loop in tests; `ruff check --select F821`.

### External References
- **DrugMechDB** (`github.com/SuLab/DrugMechDB`, **CC0-1.0**): single `indication_paths.yaml`; drug=`links[0].source`, disease=`links[-1].target`, ordered bridges = interior nodes. **Used in Phase 0** for the random-30 slice.
- **Crosswalks (Phase 0, random slice):** DrugBank→CHEBI via UniChem `wholeSourceMapping` (src 2→7); MeSH→MONDO via MONDO `mondo_exactmatch_mesh.sssom.tsv`; UniProt→NCBIGene via UniProt `idmapping.dat.gz`. The 20 anchors use direct CURIEs (no crosswalk).
- **Stats** (`statsmodels`): `contingency_tables.mcnemar(table, exact=True)` primary, `exact=False, correction=False` sensitivity; power depends on discordance π_D not N (Monte-Carlo the exact test); `inter_rater.cohens_kappa`/`fleiss_kappa`; TREC pooling → "pooled recall".

## Key Technical Decisions

- **Kestrel access = REST `/api` via a thin httpx client** (finding #1), not the kraken MCP client/`/mcp`. Reuse kraken's `_canonical_curie`/`_same_entity` parsing+grounding helpers (transport-agnostic) over REST. Cassette wraps the REST httpx layer.
- **Staged build.** Phase 0 (stratified 50-item gold set, both arms, recall gate) can issue a NO-GO without the EITL human campaign, fairness guard, or hashed manifest. (DrugMechDB crosswalk ETL lives in Phase 0 by deliberate choice, to make the random-30 inferential slice genuinely random.)
- **Stratified gold set (N=50), random tranche carries the inference.** 10 T2D + 10 Alzheimer's hand-curated anchors (direct CURIEs, `difficulty∈{easy,hard}`, **measured & committed** — 14 easy / 6 hard) anchor R0 + per-stratum relevance; **30 random DrugMechDB indications** supply most McNemar discordance and avoid hand-pick selection bias. Recall lift reported **per stratum** (T2D / Alzheimer's / random). `hub_degree` captured on hub disease endpoints.
- **No determinism claim — the gate is a variance-band measurement.** The Agent SDK exposes **no temperature control**; run each item **K≥3 times**, per-item hit = majority of K, report the variance band. INCONCLUSIVE if McNemar flips across the band.
- **Cassette policy.** Static baseline (one fixed query/trial) replays from a cassette; the **iterate arm runs live** (record-as-you-go); a cassette miss mid-loop **fails the trial loudly** (`terminal_state=cassette-miss`). McNemar pairing is per-item.
- **Cap-free shared parser, total-evidence budget held equal.** Both arms use one harness parser over `results[].paths[]` (no `paths[:10]` cap). Hold the **aggregate distinct-path budget** equal (not per-call) — *and pin the `multi-hop` `limit` as a frozen knob* (finding #2: with hub diseases, `limit` determines whether the gold path surfaces, for both arms).
- **Recall and precision are disjoint, and recall is the kill axis.** Recall = gold-path hits (OWA); EITL-validated off-gold bridges do **not** feed back into recall. Precision/false-confirmation (Phase 1) is a PROCEED-confirmation criterion, never a kill criterion. On single-path gold the exploratory iterate arm may be penalized on recall and rewarded on precision — they can pull opposite ways, which is why recall+McNemar gates the kill.
- **Grounding** reuses `_canonical_curie` + `equivalent_ids` (port `_ambiguous`) over REST `/get-nodes`; per-trial violation rolls up to a harness-wide count; any > 0 → hard NO-GO.
- **Pilot is baseline-only.** R0 (per stratum) from a baseline-only run on ~15 items + conservative π_D prior (0.25) sets powered-N via Monte-Carlo exact-McNemar. Iterate arm built only after N confirmed.
- **Frozen config, plain committed file (Phase 0); hashed manifest (Phase 1 only).** Per-turn Kestrel-call cap frozen from principle, not pilot.

## Open Questions

### Resolved During Planning
- Kestrel access? → REST `/api` (finding #1); `/mcp` is flaky.
- multi-hop response shape? → `{"results":[{…,"paths":[[curie,…]]}]}` (verified).
- Determinism? → none; variance-band, K≥3, majority hit.
- Pilot circularity? → baseline-only R0 + conservative π_D prior.
- DrugMechDB license / extraction? → CC0; walk `links`.
- McNemar variant? → exact primary, asymptotic uncorrected sensitivity.
- The 20 anchors + difficulty? → **done & committed** (`gold_set_anchors.json`).

### Deferred to Implementation
- DrugMechDB random-sample **seed** must be frozen in config (P5) for reproducible random 30.
- UniChem `src_id`s (DrugBank=2, ChEBI=7) — verify before hardcoding.
- Do ≥30 random DrugMechDB indications survive crosswalk + reachability at the powered N? (anchors are crosswalk-independent.)
- The frozen `multi-hop` `limit` value (finding #2) — pilot-informed but frozen before the gate.

## Output Structure

    backend/tests/code_on_graph_spike/
      __init__.py
      config.py                 # frozen Pydantic pre-registration + DrugMechDB sample seed
      kestrel_rest.py           # thin httpx REST client (/api) + cap-free path parser + grounding
      anchors.py                # load/validate the 20 committed anchors (direct CURIEs)
      drugmechdb.py             # parse indication_paths.yaml + gold-path extraction
      crosswalk.py              # DrugBank→CHEBI, MeSH→MONDO, UniProt→NCBIGene
      gold_set.py               # random-30 + reachability filter + merge with anchors → unified 50
      baseline.py               # static multi-hop arm (+ cassette replay)
      pilot.py                  # baseline-only R0 (per stratum) + conservative π_D → powered-N
      iterate_loop.py           # structured-spec executor → REST, live record-as-you-go, K reruns
      recall_scorer.py          # majority-of-K recall vs gold (OWA), per-stratum, McNemar table
      gate_recall.py            # Phase-0 recall gate → NO-GO or proceed-to-Phase-1
      # ---- Phase 1 (only on a passing recall signal) ----
      eitl_export.py / fairness_guard.py / gate_full.py / manifest.py / run_spike.py
      test_*.py                 # companion mocked-Kestrel unit tests
    backend/tests/fixtures/code_on_graph_spike/
      gold_set_anchors.json     # [DONE] 20 measured T2D/AD anchors, direct CURIEs, difficulty-tagged
      gold_set.json             # unified 50 (anchors + crosswalked random 30), trial_id keyed
      config_frozen.json

## High-Level Technical Design

> *Directional guidance for review, not implementation specification.*

```
PHASE 0 (kill-test, N=50 stratified, Kestrel REST /api)      PHASE 1 (only if recall passes)
 gold set = 20 anchors [DONE] + 30 random DrugMechDB           EITL blinded precision arm → κ, false-confirm
   └ baseline-only pilot → R0 (per stratum), π_D=0.25 → N      fairness guard (under-spec seed + neg control)
   └ baseline arm (REST multi-hop, cassette replay) ─┐         full gate + verdict lattice + hashed manifest
   └ iterate arm (REST, live, K≥3 reruns)           ─┴→ majority hit/miss
   └ recall gate: McNemar(exact) + lift (per stratum) + hallucinated=0
        ├ recall NO-GO or hallucinated>0 → KILL (stop)
        └ recall passes → PHASE 1
```

Iterate-loop control (per item, K reruns; no temperature control → variance band):
```
LLM emits JSON spec {verb∈whitelist, start, end, max_path_length≤5, predicate, degree_constraint}
  → validate vs the 12-verb whitelist  → dispatch to REST /api (live, record-as-you-go)
  → transport error: retry transparently, max 2, OUTSIDE turn budget; then terminal_state=transport-failed
  → valid-empty / malformed-spec: feed back, consumes 1 turn (cap 5)
  → grounding: emitted CURIE ∈ canonical(returned ∪ equivalent_ids), else grounding-violation (hard-fail++)
  → stop: bridge found | turn cap | empty-exhausted | aggregate-path-budget hit
```

## Implementation Units

### Phase 0 — Kill-test (N=50 stratified; issues a cheap NO-GO; gates Phase 1)

- [x] **Unit 0.1a: Disease anchors (20) — curated & measured** *(done this session)*
  `gold_set_anchors.json` committed: 10 T2D + 10 AD, direct human CURIEs, difficulty measured live (14 easy / 6 hard). Remaining 0.1 work is `config.py` + `anchors.py` loader/validator below.

- [ ] **Unit 0.1: REST client + frozen config + anchor loader**

**Goal:** The thin REST client, the frozen pre-registration config, and a validating loader for the committed anchors.

**Requirements:** R3, R5, P1, P3, P5

**Dependencies:** None (anchors fixture already exists)

**Files:**
- Create: `backend/tests/code_on_graph_spike/kestrel_rest.py`, `config.py`, `anchors.py`
- Test: `test_config.py`, `test_anchors.py`

**Approach:**
- `kestrel_rest.py`: httpx client to `https://kestrel.nathanpricelab.com/api` with `X-API-Key`; helpers `hybrid_search`, `multi_hop`, `one_hop`, `get_nodes`; **cap-free** parser over `results[].paths[]`; grounding via `_canonical_curie`+`equivalent_ids`. (Finding #1 — no MCP.)
- `config.py`: frozen Pydantic — recall-lift 15% (+relative if R0>85%), α=0.05, McNemar exact primary, turn cap 5, per-turn Kestrel-call cap, **frozen `multi-hop` limit** (finding #2), aggregate-path budget, K≥3, majority-hit rule, N floor, **DrugMechDB sample seed**.
- `anchors.py`: load `gold_set_anchors.json`, re-validate each CURIE resolves to the *human* node (taxon-robust — finding #3), capture `hub_degree`. ≥2 known-easy items serve as the loop smoke test.
- **Bridge unit (frozen):** hit = a returned path containing **all** gold interior node(s) (`_canonical_curie`+`equivalent_ids`); "any-one-interior" reported for sensitivity only.

**Test scenarios:**
- Happy: config loads, thresholds + seed typed, immutable; anchors load, all 20 CURIEs resolve to human nodes.
- Edge: an anchor resolves to a low-degree ortholog → flagged (finding #3 guard).
- Edge: bad-prefix CURIE → validation error, not silent skip.
- Test: `kestrel_rest` parser extracts `results[].paths[]` correctly (mocked).

- [ ] **Unit 0.2: DrugMechDB random-30 + crosswalk + unified gold set**

**Goal:** Randomly sample 30 DrugMechDB indications, crosswalk to Kestrel CURIEs, reachability-filter, merge with anchors → unified 50.

**Requirements:** R5, R6, R7, P5

**Dependencies:** Unit 0.1

**Files:** Create `drugmechdb.py`, `crosswalk.py`, `gold_set.py`, fixture `gold_set.json`; Test `test_drugmechdb.py`, `test_crosswalk.py`, `test_gold_set.py`

**Approach:**
- `drugmechdb.py`: parse `indication_paths.yaml` (CC0, pinned SHA); drug=`links[0].source`, disease=`links[-1].target`, ordered interior bridges by walking links (DFS fallback on branched); normalize prefixes.
- `crosswalk.py`: DrugBank→CHEBI (UniChem), MeSH→MONDO (MONDO SSSOM), GO/UniProt passthrough; **frozen 1→many tie-break** (one Kestrel-reachable CHEBI; reuse `_same_entity`); **taxon-robust gene resolution** (finding #3); coverage report.
- `gold_set.py`: **random-sample with frozen seed** → **hop-agnostic reachability** filter (≤ frozen-`limit`/`max_path_length`, NOT the 2-hop baseline); exclude >5-hop gold + node-types absent from Kestrel; over-sample raw until 30 survive. Merge with anchors → `gold_set.json` (unified 50, `trial_id` keyed, `stratum∈{t2d,alzheimers,random}`).

**Execution note:** Run the crosswalk + reachability filter early; if <30 random survive, surface before building arms.

**Test scenarios:** Happy: 2-link record → correct extraction. Edge: multi-intermediate → ordered list; branched → DFS/flagged. Edge: 1→2 CHEBI → excluded. Edge: >5-hop or non-Kestrel node-type → excluded stratum. Edge: same seed → identical 30 (reproducibility). Integration: unified set = 50 (or flags shortfall), strata labelled.

- [ ] **Unit 0.3: Static baseline arm + baseline-only pilot**

**Goal:** The static arm (REST multi-hop) and R0 (per stratum) / π_D / powered-N.

**Requirements:** R2, R3, R5, R7, P1, P2, R10

**Dependencies:** Unit 0.1, Unit 0.2

**Files:** Create `baseline.py`, `pilot.py`; Test `test_baseline.py`, `test_pilot.py`

**Approach:**
- `baseline.py`: `kestrel_rest.multi_hop(start, end, max_path_length=2, limit=<frozen>)`; parse to the frozen bridge unit (cap-free); cassette-replay (one fixed body/trial). Record per-trial `{trial_id, method, bridges, hop_counts, kestrel_calls, terminal_state}`.
- `pilot.py`: baseline-only on a ~15-item subset spanning all three strata → **R0 per stratum** (anchors ≈ high R0; random tranche lower); conservative π_D=0.25; Monte-Carlo exact-McNemar → powered N (target 50, floor = powered-N). If the surviving set can't clear it → INCONCLUSIVE before building the iterate arm.

**Test scenarios:** Happy: known-easy anchor → baseline returns gold bridge → hit. Edge: empty → miss `terminal_state=empty`. Error: transport error → retried (max 2) → `transport-failed`, not a miss. Happy: pilot computes R0 + powered-N; Monte-Carlo matches closed-form. Edge: R0=86%→relative form, 84%→absolute (frozen at pilot). Edge: N<floor → INCONCLUSIVE.

- [ ] **Unit 0.4: Iterate-loop arm (structured-spec executor, K reruns, live REST)**

**Goal:** The treatment — JSON-spec query loop over REST with grounding, bounded cost, variance band.

**Requirements:** R2, R4, R9, R10, P3

**Dependencies:** Unit 0.1, Unit 0.3

**Files:** Create `iterate_loop.py`; Test `test_iterate_loop.py`

**Approach:**
- LLM (`query_with_usage`, `allowed_tools=[]`) emits a typed JSON spec validated against the **12-verb whitelist**; harness dispatches to **REST `/api` live with record-as-you-go**.
- **No temperature control** → K≥3 reruns; per-item hit = majority; record variance band. Cassette miss → `terminal_state=cassette-miss`.
- **Transport vs empty:** transport errors retry (max 2) *outside* the turn budget; only valid-empty/malformed-spec consumes a turn (cap 5). **Grounding:** emitted CURIE ∉ canonical(returned ∪ equivalent_ids) → grounding-violation (hard-fail++). **Aggregate-path budget** equal to baseline. Persist transcript + raw responses; LLM-call & Kestrel-call counters separate (incl. `get-nodes` grounding lookups).

**Execution note:** Test-first against failure paths; `ruff check --select F821`.

**Test scenarios:** Happy: loop finds gold bridge → hit, majority-of-K stable. Edge: turn cap → `turn-cap-hit`, miss. Error: malformed/invalid-verb → consumes a turn, re-prompted. Error (R9): unreturned CURIE → grounding-violation (assert `chebi:6801` vs `CHEBI:6801` is grounded). Error: transport error mid-loop → retried outside turn budget; cassette miss → fail-loud. Integration: LLM vs Kestrel counters recorded separately.

- [ ] **Unit 0.5: Recall gate (Phase-0 kill decision)**

**Goal:** Apply the recall-axis gate and decide KILL vs proceed-to-Phase-1.

**Requirements:** R5, R9, R10, P1, P2, P3

**Dependencies:** Unit 0.3, Unit 0.4

**Files:** Create `recall_scorer.py`, `gate_recall.py`; Test `test_recall_scorer.py`, `test_gate_recall.py`

**Approach:**
- `recall_scorer.py`: majority-of-K per-item hit/miss for both arms vs gold bridge (OWA); paired `[[a,b],[c,d]]` table **overall and per stratum**; pooled recall + concordant-miss count.
- `gate_recall.py`: recall lift (abs/relative per frozen P1) **and** McNemar (`mcnemar(table, exact=True)`, α=0.05; asymptotic sensitivity); hallucinated-CURIE roll-up; cost worst-case ≤3× baseline. **Phase-0 lattice:** `hallucinated>0`→NO-GO → lift absent or McNemar n.s.→NO-GO → N<powered or variance-band flip→INCONCLUSIVE → else **proceed to Phase 1**.

**Test scenarios:** Happy: lift + McNemar sig + hallucinated=0 → proceed. Edge: lift ≥15% but p>0.05 → NO-GO. Edge: hallucinated=1 → NO-GO override. Edge: McNemar flips across band → INCONCLUSIVE. Integration: scorer table feeds gate; verdict reproduced on the committed fixture.

### Phase 1 — Defensibility layer (built only on a passing Phase-0 recall signal)

> Gold set and both arms already exist. Phase 1 adds expert precision, the fairness guard, the manifest, and — only if powered N > 50 — a larger DrugMechDB sample (reusing Unit 0.2).

- [ ] **Unit 1.1: EITL blinded precision arm**

**Goal:** Expert false-confirmation judgment on off-gold bridges (PROCEED-confirmation).

**Requirements:** R8, R10, P5

**Dependencies:** Phase-0 pool; **Precondition (do first):** read `client/src/pages/review.tsx`, confirm `display.showMetadataPanel=false`, choose CSV headers that do **not** match the `ColumnMapper` auto-aliases (`llm_*`, `resolution_layer`); `eitl_export.py` asserts no method-correlated field maps to a rendered column.

**Files:** Create `eitl_export.py`; Test `test_eitl_export.py`

**Approach:** Pool both arms' off-gold bridges, dedup by unique bridge (`method` set-valued in metadata), shuffle. Emit 7 required EITL columns; **all method-correlated signal (`llm_*`, `method`, `on_drugmechdb_path`, `trial_id`) in metadata under non-auto-mapped headers, never rendered**. Ingest **active votes only**; `no_match`=spurious, `unsure` excluded; pair-level aggregation frozen (≥2 reviewers; 1-1→3rd adjudicator). κ on raw labels + report raw agreement & prevalence; κ<0.6 → precision INCONCLUSIVE.

**Test scenarios:** Happy: pool → 7-col CSV ingests. Edge (blinding): no rendered column carries `llm_*`/method; export fails if it would. Edge: same bridge both arms → one pair, method={static,iterate}. Edge: superseded votes excluded. Edge: 1-1 → adjudication. Edge: κ<floor → INCONCLUSIVE.

- [ ] **Unit 1.2: Treatment-fairness guard (discriminating)**

**Goal:** Separate "iterate genuinely loses" from "weak impl" without being theatre.

**Requirements:** P4

**Dependencies:** Unit 0.4

**Files:** Create `fairness_guard.py`; Test `test_fairness_guard.py`

**Approach:** Disjoint held-out known-reachable set. Seed spec **under-specified by ≥1 degree the loop must recover** (correct endpoints, wrong/absent predicate or max_path_length). **Negative control:** a steering-disabled loop must FAIL the guard. Below pass-fraction → kill reported INCONCLUSIVE. Runs before the full gate; one-way info barrier.

**Test scenarios:** Happy: real loop recovers ≥ pass-fraction → passes. Edge: steering-disabled control fails. Edge: held-out overlaps gate set → disjointness assertion. Edge: N-budget too small → guard skipped with logged caveat.

- [ ] **Unit 1.3: Full gate + manifest + orchestrator**

**Goal:** The defensible PROCEED/NO-GO/INCONCLUSIVE verdict with reproducibility.

**Requirements:** Gate criteria, P1–P4, R10

**Dependencies:** Units 1.1–1.2, Unit 0.5

**Files:** Create `manifest.py`, `gate_full.py`, `run_spike.py`; Test `test_gate_full.py`

**Approach:** Hashed artifact manifest (DrugMechDB SHA, crosswalk versions, **Kestrel API version**, model id, loop seed spec, thresholds, **EITL UI commit SHA + blinding-verified flag**); gate refuses on hash mismatch. **Verdict lattice:** `hallucinated>0`→NO-GO → any fail→NO-GO → any inconclusive (N<powered / κ<floor / fairness-guard fail / McNemar n.s. / variance-band flip)→INCONCLUSIVE → else PROCEED. `run_spike.py` chains Phase 0→1, per-criterion + per-stratum table, exit-code by verdict.

**Test scenarios:** Happy: all pass + McNemar sig → PROCEED. Edge: κ<floor → INCONCLUSIVE. Edge: manifest hash mismatch → refuses. Edge: fairness-guard fail → INCONCLUSIVE. Integration: end-to-end tiny fixture → reproducible verdict.

## System-Wide Impact

- **Interaction graph:** Standalone — no LangGraph node/contract modified. Reads live Kestrel REST `/api`; Phase 1 creates an EITL campaign.
- **Error propagation:** transport errors retry (max 2) then `transport-failed`, never a miss or hallucination; per-trial failures captured; cassette miss fails loud.
- **State lifecycle:** `trial_id` minted in the gold set, threaded through blinded EITL votes back to gold; raw responses + transcripts persisted.
- **API surface parity:** none.
- **Integration coverage:** DrugMechDB→Kestrel mapping, blinded-export→vote-rejoin, live iterate vs cassette-baseline pairing — exercise on a small live slice.
- **Unchanged invariants:** `integration.py`, `synthesis.py`, the DAG, all state contracts.

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Kestrel `/mcp` intermittently drops `X-API-Key` (finding #1) | Harness uses REST `/api` directly (8/8 reliable); report `/mcp` bug upstream |
| Static baseline + `limit` + hub disease buries the gold path (finding #2) | Pin `multi-hop` `limit` as a frozen knob applied identically to both arms; treat baseline recall ceiling as measured, not assumed |
| Entity resolution mis-resolves to non-human orthologs (finding #3) | Anchors use direct human CURIEs (committed); `anchors.py`/`gold_set.py` taxon-robust resolution + degree sanity check |
| No temperature control → loop nondeterministic | K≥3 reruns, majority hit, variance band; INCONCLUSIVE if McNemar flips |
| Cassette can't replay loop's novel queries | Loop runs live/record-as-you-go; baseline replays; miss = fail-loud |
| EITL UI renders `llm_*`/metadata → un-blinds | Phase-1 precondition reads `review.tsx`; non-auto-mapped headers; `showMetadataPanel=false`; export assertion |
| Aggregate-evidence asymmetry (multi-call loop > single-call baseline) | Hold the aggregate distinct-path budget equal, not per-call |
| Crosswalk attrition drops the random 30 below N | Anchors are crosswalk-independent (a floor); over-sample raw DrugMechDB until 30 survive; surface shortfall before building arms |
| Well-characterized anchors → low discordance | The 30 random items carry the McNemar discordance; per-stratum lift reporting makes the easy-vs-hard split visible |

## Documentation / Operational Notes
- Throwaway, not in CI; run `cd backend && uv run python tests/code_on_graph_spike/run_spike.py`; unit tests `cd backend && uv run python -m pytest tests/code_on_graph_spike/`.
- Phase 1 creates one EITL `custom` campaign — coordinate around its production launch.
- After verdict, archive `backend/tests/code_on_graph_spike/` to `docs/experiments/` (NO-GO/INCONCLUSIVE) or promote to `assessment/` (PROCEED).

## Sources & References
- **Origin:** [docs/brainstorms/code-on-graph-spike-requirements.md](docs/brainstorms/code-on-graph-spike-requirements.md)
- Feasibility: `docs/code-on-graph-feasibility.md`; node map: `docs/pipeline-node-tool-map.md`; pipeline wants: `docs/discovery-pipeline-requirements.md`
- Measured anchors: `backend/tests/fixtures/code_on_graph_spike/gold_set_anchors.json`
- Code: `backend/tests/recall_gate.py`, `backend/src/kestrel_backend/assessment/{runner,cassette,scorer,capture}.py`, `backend/src/kestrel_backend/graph/nodes/{integration,entity_resolution}.py`, `backend/src/kestrel_backend/graph/sdk_utils.py`, `backend/src/kestrel_backend/kestrel_tools.py`, `expert-in-the-loop/{shared/schema.ts,client/src/components/ColumnMapper.tsx,client/src/pages/review.tsx}`
- Kestrel REST: `https://kestrel.nathanpricelab.com/api` (`/hybrid-search`, `/multi-hop`, `/one-hop`, `/get-nodes`); source `Phenome-Health/kestrel` (`kestrel/api.py`, `kestrel/mcp_server.py`)
- External: DrugMechDB (CC0); UniChem; MONDO SSSOM; statsmodels `contingency_tables.mcnemar`, `inter_rater`; TREC pooling (Buckley & Voorhees, Zobel)
