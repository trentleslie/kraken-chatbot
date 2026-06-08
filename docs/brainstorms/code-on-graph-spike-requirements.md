---
date: 2026-06-03
topic: code-on-graph-spike
---

# Code-on-Graph Go/No-Go Spike

## Problem Frame

The feasibility assessment (`docs/code-on-graph-feasibility.md`) lands on a measurement-gated "recommend-prototype" for adapting the Code-on-Graph (CoG) iterative query-refinement loop into kraken's discovery pipeline. But the verdict is honest that this is a **speculative bet**: CoG's published wins come mostly from keeping the KG out of the prompt, which kraken already does (`allowed_tools=[]`, fetch-then-reason). Against an already-lean baseline, an iterate-and-refine loop *adds* turns/calls, so CoG's numbers give ≈zero predictive evidence for kraken's payoff.

Committing to the full build (new `graph_reasoning` node + structured-spec executor + state contracts — rated L/L+ in the feasibility doc) before knowing whether iterate-beats-static is true would risk significant effort on an unvalidated hypothesis.

This spike is the **cheapest experiment that could kill the idea**: a standalone, throwaway harness that measures whether an iterative query-refinement loop beats kraken's static query plan for endpoint-to-endpoint bridge discovery, judged against pre-committed thresholds fixed *before* results are seen.

## Requirements

**Experiment Design**
- R1. The spike is a **standalone throwaway harness**, not a modification of the production `integration.py:detect_bridges_via_api` node. The production node is only touched *after* the gate passes.
- R2. The harness compares two conditions on the same inputs: (a) **static baseline** — a fixed 2-hop `multi_hop_query` between the two endpoints; (b) **iterate-loop treatment** — a minimal LLM-driven query-refinement loop.
- R3. Task framing is **endpoint-to-endpoint**: each trial supplies a start CURIE and a gold-target CURIE; the task is to find the bridge node(s) connecting them. (This deliberately differs from the production node's category-pair fan-out — it isolates "find the connector" cleanly.)
- R4. The iterate-loop emits a **typed JSON query spec** (structured-spec executor per the feasibility doc), not arbitrary Python — preserving kraken's no-arbitrary-execution safety posture. It self-corrects on empty/error results up to the hard turn cap.

**Recall Reference — DrugMechDB labeled gold bridges** *(supersedes the earlier BioHopR choice — see Key Decisions)*
- R5. Recall reference = **DrugMechDB** (SuLab, *Scientific Data* 2023; `github.com/SuLab/DrugMechDB`): 4,583 manually curated **drug → [intermediate path] → disease** mechanism-of-action paths. The curated **intermediate nodes are the gold bridges**, so recall is well-defined as "did the method recover the curated mechanism's bridge node(s)?". This fixes the BioHopR problem (its gold was the *target* node, not a labeled *bridge*).
- R6. Build the combined query+gold set by mapping DrugMechDB identifiers (DrugBank/MeSH/UniProt/GO) → Kestrel CURIEs (CHEBI/MONDO/NCBIGene-HGNC/GO; GO maps directly, **DrugBank↔CHEBI** and **MeSH↔MONDO** need standard crosswalks), then keeping items whose drug↔disease pair is **reachable in Kestrel verified hop-agnostically** (any path ≤ N hops, or both endpoints individually present) — **NOT** via the 2-hop baseline query. Verifying reachability with the baseline query would keep only items the static baseline already solves and silently delete iteration's headroom (review finding #1).
- R7. The filtered set must contain **≥ 30 Kestrel-reachable items** (else **inconclusive**, not pass/fail). Run this filter as **step-0, before building the harness**; report the surviving count plus its hop-length and drug/disease-category composition so a non-representative or baseline-favorable slice is visible.

**Precision Reference — EITL expert judgment on off-gold bridges**
- R8. DrugMechDB catalogs **one** mechanism per indication, so a bridge a method reports that is *not* on the curated path is not automatically wrong — it may be a legitimately valid, uncatalogued mechanism. **expert-in-the-loop** (`../expert-in-the-loop`, the pairwise-voting platform: Campaigns → Pairs → Votes) supplies the precision/false-confirmation judgment on these off-gold bridges: experts vote real-but-uncatalogued vs spurious, **≥ 2 reviewers per pair, blinded to which method produced the bridge, with an inter-rater κ check**. This is where expert labels are irreplaceable; recall stays automated against DrugMechDB.

**Grounding Contract**
- R9. **Grounding contract** (the runtime-membership variant; do not conflate with the prior plan's static-prefetch R1a): any CURIE the iterate-loop emits must reference a node actually returned by an executed Kestrel call. Hallucinated CURIEs are a hard failure (see Success Criteria).

**Measurement**
- R10. Report, for both conditions: **recall** (vs DrugMechDB gold bridges), **false-confirmation rate** (EITL-judged spurious off-gold bridges ÷ all bridges reported), **hallucinated-CURIE rate**, and per-trial **cost** as a distribution (p50/p95) — never mean alone. Record LLM-call count and Kestrel-call count separately. Zero-denominator convention: a method reporting no off-gold bridges has false-confirmation rate 0 (passing).

## Success Criteria

The spike **PROCEEDs to the L+ build** only if **ALL** of the following hold (Balanced threshold package, pre-committed):

- Recall lift ≥ **15% absolute** OR the iterate-loop recovers ≥ **50% of the gold bridges the static baseline missed** — **and** the lift is **significant by a paired McNemar test (α = 0.05)** on per-item hit/miss (see Pre-Registration). A point estimate alone does not pass.
- False-confirmation rate ≤ **10%** (EITL-judged spurious off-gold bridges; zero-denominator → 0, passing)
- Hallucinated-CURIE rate = **0** (hard fail — non-negotiable, reuses the R9 grounding contract)
- Cost: hard turn cap = **5**; **worst-case cost (turn-cap × per-turn) ≤ 3× the static baseline** — gated on the controllable worst case, not an unstable empirical p95 at small N (p50/p95 reported descriptively only)
- N ≥ the **power-calc'd N** from the step-0 pilot (floor **30**; if the pilot shows 30 is underpowered for a 15-pt McNemar effect, the floor rises — below the powered N the result is **inconclusive**, never a kill)

**Transfer interpretation (endpoint-to-endpoint → category-pair):** the spike tests endpoint-to-endpoint, but the production target `detect_bridges_via_api` is category-pair. So a **no-go is a strong kill** (fails the easier task), while a **go licenses a category-pair confirmation arm, not the full L+ build directly** (review finding #6).

Any single criterion missed = **no-go** (or inconclusive if the N floor is not met). The thresholds are fixed before results are observed; results are not re-litigated against adjusted bars.

## Pre-Registration (frozen before any data is seen)

These resolve the review's validity findings. All are fixed *before* the harness runs; none may be re-chosen after seeing results.

**P1 — Baseline anchor + attainability (resolves the #baseline-anchor finding).** Step-0 measures static-baseline recall **R0** on the pilot set. The 15%-absolute criterion is only valid if headroom allows it: if **R0 > 85%**, the absolute criterion is unattainable and the gate uses the relative form ("recover ≥ 50% of static's misses"). R0 is reported with the result.

**P2 — Significance test (resolves the #power finding).** Lift is tested with a **paired McNemar test** on per-item static-vs-iterate hit/miss (paired design shrinks variance vs two independent proportions), **α = 0.05**, two-sided. The step-0 pilot computes the **N required for 80% power** to detect a 15-pt effect given the observed discordance rate; the run uses `max(30, powered-N)`. A point estimate that clears 15% but fails McNemar does **not** pass.

**P3 — Cost rule (resolves the #p95 finding).** The cost gate is the **controllable worst case** (turn-cap × per-turn cost) ≤ 3× baseline, *not* an empirical p95 (unstable as the ~28th order statistic at N≈30). p50/p95 are reported descriptively. LLM-call and Kestrel-call counts recorded separately.

**P4 — Treatment-fairness guard (resolves the weak-impl-vs-genuine-loss confound).** On a held-out subset of items with a *known-reachable* gold bridge, confirm the iterate-loop **can** recover it when handed a near-oracle plan. If it cannot, a kill is attributed to a weak implementation → **inconclusive, not no-go**. The minimal-but-fair loop spec (JSON query-spec schema, self-correction prompt, executor surface) is frozen here, before the gate.

**P5 — Frozen definitions (resolves the forking-paths finding).** Fixed before data: the **hop-agnostic reachability rule** (R6), the **bridge unit** (which intermediate node(s) on a returned path are scored), the **EITL vote → false-confirmation mapping** (`no_match` = spurious; `unsure` excluded from the denominator), and **method-provenance blinding** in EITL. No definition is tuned after preliminary numbers are visible.

## Scope Boundaries

- **No production node changes.** `detect_bridges_via_api`, `direct_kg.py`, `pathway_enrichment.py`, and a new `graph_reasoning` node are explicitly out of scope for the spike — they are the *reward* if the gate passes, not part of the experiment.
- **No arbitrary-Python sandbox.** Structured-spec executor only (R4). Full LLM-generated-Python fidelity is out of scope.
- **No local subgraph materialization / local relation-ranking.** Kestrel is remote-only; replicating CoG's in-memory-subgraph + DistilBERT ranking is an unscoped sub-project, excluded here.
- **No category-pair framing in the gate itself.** The endpoint-to-endpoint gate (R3) excludes the noisier category-pair fit; category-pair returns only as a **confirmation arm on a go** (see Transfer interpretation), not as part of the kill/keep gate.
- **Not a validation-site prototype.** Per the feasibility doc, the spike must not run at `synthesis.py:validate_bridge_hypotheses` (confirmation-seeking risk).

## Key Decisions

- **Cheapest go/no-go over full requirements**: Validate the speculative hypothesis before committing L+ effort against an already-lean baseline.
- **Single comparison over staged headroom-probe**: User chose to build a minimal iterate-loop and compare directly, accepting the validity risk (weak-impl vs iterate-genuinely-losing), mitigated by the "minimal but fair" structured-spec design (R4) and pre-committed thresholds.
- **DrugMechDB (recall) + EITL (precision), superseding BioHopR**: BioHopR's gold was the *target* node, not a labeled *bridge*, and building it was a crosswalk sub-project that could still yield N<30. DrugMechDB gives **labeled gold bridges** (the curated mechanism path) *and* the query endpoints in one artifact; EITL covers precision on off-gold bridges where DrugMechDB's single-path gold is incomplete. This is the "hand-curated" path upgraded to a published, defensible methodology, and it removes review findings #1/#4/#5 rather than patching them. (BioHopR was the prior choice; it is retained only as an optional external cross-check if the spike proceeds.)
- **Endpoint-to-endpoint over category-pair (for the gate)**: Cleaner recall metric and isolates the "find the connector" capability; category-pair transfer is handled explicitly as a post-go confirmation arm rather than ignored.
- **Balanced threshold package**: Lift must clear noise on a usable N; false-confirm capped at the level the feasibility doc worries about; chosen over Strict (risks false-kill of a modest-but-real win) and Exploratory (risks greenlighting a marginal idea).

## Alignment with Discovery Pipeline Requirements

Cross-checked against `docs/discovery-pipeline-requirements.md` (Lance's wants, 2026-06-02). The spike is **strategically central, not a side-quest** — it de-risks the mechanistic-path-finding engine behind two of the four guiding themes:

| Lance's requirement | How this spike serves it |
|---|---|
| **Mechanism of action** (theme #2) | DrugMechDB *is* an MoA-path database; benchmarking bridge discovery against it directly tests MoA-quality reasoning. |
| **Drug repurposing** (theme #4) | Defined as "multi-hop from implicated biology to drugs/interventions" — i.e., doubly-pinned bridge discovery, the exact `integration` function the spike targets. |
| **Multi-hop to drug/intervention** (core must-have, §2) | The `integration` node's `multi_hop_query` bridge detection — improving it improves a stated must-have. |
| **Subgraph mechanistic extraction** (core must-have, §2) | Bridges *are* the mechanistic-chain extraction. |

The spike is **neutral** on the other requirement clusters (module/ranking inputs, ORA/GSEA multiomic enrichment, network centrality, drug safety) — different machinery.

**Two repurposing prerequisites the spike's "go" does NOT by itself satisfy** (already flagged as gaps in the requirements doc and carried forward here):
1. **Analyte-anchored entry.** DrugMechDB paths run drug→bridge→disease; Lance's repurposing runs *biology(analyte/module)→drug*. The spike validates path-finding fidelity (which transfers), but production repurposing queries are analyte-anchored, not drug-anchored — needs separate confirmation.
2. **Bridges must terminate on drug/intervention nodes.** `integration` today finds cross-type *biology* bridges; serving repurposing requires bridges that reach drug/intervention nodes (the requirements-doc gap row "Confirm it terminates in drug/intervention nodes").

## Dependencies / Assumptions

- **Kestrel ↔ DrugMechDB namespace overlap is sufficient.** Verified in-codebase: Kestrel uses biolink-style CURIEs (MONDO disease, CHEBI compound, GO, NCBIGene/HGNC gene). DrugMechDB uses DrugBank (drug), MeSH (disease), UniProt (protein), GO. **GO maps directly; DrugBank↔CHEBI and MeSH↔MONDO need standard crosswalks** — assumption that enough items survive mapping + hop-agnostic reachability to clear N≥30.
- **DrugMechDB is downloadable and license-compatible** (`github.com/SuLab/DrugMechDB`, *Scientific Data* 2023) — confirm license at planning.
- **EITL is usable for a labeling campaign** — can host a `custom` campaign, blind method provenance on Pairs, capture ≥2 votes/pair with export, without disrupting its imminent production launch.
- **No checkpointer** in `build_discovery_graph` (plain `compile()`, verified at `builder.py:180/201`) — irrelevant to the standalone harness, but relevant to the eventual node build.
- **Kestrel `multi_hop_query` exists** and serves as the static baseline primitive (referenced in `kestrel_client.py`).

## Outstanding Questions

### Resolve Before Planning
*(none — all product/scope decisions are resolved)*

*Statistical power (R0 anchor, McNemar, cost rule), the treatment-fairness guard, and the frozen definitions are now resolved in the **Pre-Registration** section above — no longer open.*

### Deferred to Planning
- [Affects R6][Needs research] Build the **DrugBank↔CHEBI** and **MeSH↔MONDO** crosswalks — how complete must they be, and do enough items survive the step-0 hop-agnostic reachability filter to clear the powered N? (Empirical; run step-0 first.)
- [Affects R8][Technical] EITL campaign mechanics: confirm the review UI does **not** render `resolution_layer`/provenance during voting (else provenance moves to `*_metadata`), and pin the κ acceptance threshold + label-export → metric pipeline.
- [Affects repurposing prerequisites][Technical] Confirm `integration` bridges can terminate on **drug/intervention** nodes, and scope an **analyte-anchored** entry variant — both required for the drug-repurposing theme, neither tested by the endpoint-to-endpoint gate.
- [Affects R5][Needs research] Does any author reference implementation for CoG exist yet? (Feasibility doc open question #4.)

## Appendix: EITL Ingestion Format

Verified against `../expert-in-the-loop` `shared/schema.ts` (`pairs` table) and the smart-mapping aliases in `client/src/components/ColumnMapper.tsx`. One **CSV row = one candidate bridge** (the unit experts vote on). Smart mapping auto-detects columns whose header matches any listed alias, so a CSV using these names ingests with zero manual mapping.

**7 required columns**

| EITL field | Smart-map aliases (auto-detected) | Spike value |
|---|---|---|
| `source_text` | `source_text`, `sourceText`, `source_name`, `question_text`, `source` | The endpoint-pair query, e.g. `"Metformin → Type 2 Diabetes (proposed mechanism)"` |
| `source_id` | `source_id`, `sourceId`, `source_code`, `questionnaire_item_id` | Composite endpoint id, e.g. `CHEBI:6801|MONDO:0005148` |
| `source_dataset` | `source_dataset`, `sourceDataset`, `source_system`, `dataset` | `kraken-spike-query` (or `DrugMechDB`) |
| `target_text` | `target_text`, `targetText`, `target_name`, `target` | Human-readable proposed bridge path, e.g. `"AMPK (NCBIGene:5562) —activates→ …"` |
| `target_id` | `target_id`, `targetId`, `target_code` | Bridge node CURIE(s) / path id |
| `target_dataset` | `target_dataset`, `targetDataset`, `target_system` | `kraken-integration-bridge` |
| `pair_type` | `pair_type`, `pairType`, `type`, `mapping_type` | `bridge_validity` (free-text since migration-001) |

Vote semantics: binary `match` = valid mechanistic bridge (real) · `no_match` = spurious · `unsure` (excluded from the false-confirmation denominator per P5).

**Optional LLM columns** (purpose-built for the iterate-loop's LLM-generated bridges)

| EITL field | Smart-map aliases | Spike value |
|---|---|---|
| `llm_confidence` | `llm_confidence`, `confidence`, `score` | iterate-loop's confidence in the bridge (0–1) |
| `llm_model` | `llm_model`, `model` | model id, e.g. `claude-opus-4-8` |
| `llm_reasoning` | `llm_reasoning`, `reasoning`, `explanation` | the loop's rationale for proposing the bridge |

**Optional metadata** (any fields we want; mapped into `source_metadata` / `target_metadata` jsonb)
- `method` — `static` vs `iterate` ⚠️ **blinding-critical**: keep in metadata, confirm it is not rendered during voting (P5). Do **not** put it in `resolution_layer` if that field is displayed.
- `hop_count`, `path_length`, `predicates`, `hub_degree` (max intermediate-node degree), `on_drugmechdb_path` (bool — is this the gold bridge?), `trial_id`, `kestrel_call_count`.

> `evidence_status` defaults to `unreviewed`; `resolution_layer` defaults to `unspecified`. Neither is required on import.

## Next Steps
-> `/ce:plan` for structured implementation planning of the spike harness.
