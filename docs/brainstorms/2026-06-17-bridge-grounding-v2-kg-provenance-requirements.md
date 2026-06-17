# Requirements: Bridge-Grounding v2 — KG-Provenance Gate + LLM Confirmation

**Date:** 2026-06-17
**Status:** requirements (brainstorm output) — feeds `ce:plan`
**Supersedes approach in:** `docs/plans/2026-06-08-002-feat-bridge-grounding-scorer-plan.md` (v1, co-occurrence)
**Origin of pivot:** `backend/assessment_data/BRIDGE_GROUNDING_TIER_A_FINDINGS.md` (v1 Tier A FAILED, margin −0.333)

## Problem Frame

v1 scored each 3-node `Bridge` by retrieving PubMed **co-occurrence** abstracts per leg and having an
LLM label support/refute. Its Tier A calibration **failed** (margin −0.333): the hard negative
coffee→caffeine→pancreatic cancer scored highest of the panel, above real mechanisms. Root cause —
**co-occurrence ≠ mechanism**: the pool/labeling reward "studied association" (epidemiology, reviews)
indistinguishably from mechanism. skimgpt avoids this with a *learned relevance classifier*
("porpoise" via Triton) that we deliberately did not port; the lightweight `off_topic` label was an
insufficient substitute.

**Key reframe (from live Kestrel probe, 2026-06-17):** v1 was reconstructing from raw text what the
**bridge's own leg edges already encode in the KG.** Each leg of a bridge IS a Kestrel edge with a
typed predicate, a knowledge source, a knowledge level, and supporting PMIDs. The mechanism signal we
need is largely already there.

## Verified Feasibility Findings (live Kestrel, `one_hop_query` mode=full)

- **Predicate vocabulary separates mechanism from association.** Causal: `biolink:causes`,
  `contributes_to`, `affects`, `regulates`, `directly_physically_interacts_with`, `catalyzes`,
  `binds`, `treats`. Associative: `biolink:associated_with`, `gene_associated_with_condition`,
  `genetically_associated_with`, `correlated_with`. (Vague middle: `biolink:related_to`.)
- **Edges carry rich provenance:** `primary_knowledge_source` (curated DBs — CTD, SIGNOR, IntAct,
  DrugCentral, hetionet — vs `infores:text-mining-provider-targeted`), `knowledge_level`
  (`knowledge_assertion` vs prediction), `publications` (PMIDs — e.g. 2581/3663 TP53 edges have
  them), and directional `qualifiers`.
- **Kestrel is a Translator-style aggregate** (SPOKE + ROBOKOP + ~30 sources). It subsumes the
  mechanism-typed-relation role SemMedDB would have played → **no SemMedDB dependency needed.**
- The `multi_hop` edge tuples (which U0 already parses) carry `predicate`, `primary_knowledge_source`,
  `knowledge_level`; per-edge `publications` come via `get_edges`/edge_ids (planning detail).

## Chosen Approach — KG-Provenance Gate + LLM Confirmation (hybrid)

**Primary signal = per-leg KG edge provenance.** For each leg edge, combine:
- **Predicate-type weight** — causal predicates score high, associative predicates are gated down,
  `related_to`/vague are neutral/low.
- **Source quality** — curated mechanism sources (CTD, SIGNOR, …) > text-mining > prediction-only.
- **Knowledge level** — `knowledge_assertion` > lower levels.
- **Publication support** — count/presence of edge PMIDs (honest "thin evidence" signal).

**LLM/literature is now a CONFIRMATION layer, not the primary signal.** It reads the *edge's own
publications* (not raw co-occurrence pools) to (a) confirm the asserted-relation strength and
(b) flag **contestation** (a causal edge whose literature disputes it). It no longer has to
distinguish mechanism from association from scratch — the KG already typed that.

This drops the substrate that failed (raw co-occurrence pools), leverages U0, and makes the gate
essentially free (metadata already in the multi_hop response).

## Requirements

- **R1.** Compute a per-leg grounding signal from the leg edge's `predicate`, `primary_knowledge_source`,
  `knowledge_level`, and `publications` — sourced from the KG, not co-occurrence retrieval.
- **R2.** Gate mechanism vs association via the Biolink predicate type (causal high, associative low).
  Spurious-but-studied chains (associative edges) must score low.
- **R3.** Compose the two legs into a chain verdict (retain weaker-leg gating from v1/U4, but over the
  new per-leg signal); keep the stronger leg as a secondary ranking key (reuse U1's model shape).
- **R4.** LLM confirmation runs over the **edge's own publications**, scoped to confirm strength /
  detect contestation — not to re-derive the relation.
- **R5.** Reuse the U8 Tier A harness + panel and the pre-registered thresholds; v2 must **pass**
  where v1 failed — coffee→caffeine→pancreatic must score LOW, positives must clear the floor, margin
  ≥ +0.30. (Same gate, new scorer.)
- **R6.** Keep v1's plumbing that is substrate-independent: U0 (predicate/orientation derivation),
  U1 (`BridgeGrounding`/`grounded_bridges` model + state), the qualitative-band rendering, default-off
  enablement, before-synthesis topology. Retire/park U2 co-occurrence retrieval + U3 from-scratch
  labeling as the primary path.

## Success Criteria

- v2 **passes Tier A** on the existing panel (evaluable + margin ≥ +0.30; hard negative below the
  negative ceiling). This is the same falsification gate v1 failed.
- The signal is computed primarily from KG metadata (cheap, deterministic gate) with bounded LLM use.
- coffee→caffeine→pancreatic (and beta-carotene→oxidative→lung) score as negatives; HPV and H.pylori
  score as positives.

## Scope Boundaries / Non-Goals

- Single-bridge grounding only — **no pairwise competing-bridge ranking** (skimgpt's comparator path
  remains a separate future workstream).
- Beta-Binomial CI / resampling — still deferred (the v1 U9 seam carries over).
- Tier B time-sliced calibration — still deferred behind the temporal-eval port.
- **Keep the v1 co-occurrence scorer code (eval-only) as a baseline/comparator** — do not delete; it
  is the negative-control the v2 is measured against.

## Open Questions for Planning

- **Per-leg edge provenance retrieval:** use the `multi_hop` compact edges (predicate + source +
  knowledge_level, already parsed by U0) and fetch `publications` via `get_edges` by edge_id? Confirm
  the bridge construction preserves per-leg edge_ids.
- **Predicate → weight mapping:** the exact causal/associative/neutral partition of the Biolink
  predicate set, and how to treat the vague `related_to`.
- **Source-quality tiering:** which `infores:*` sources are curated-mechanism vs text-mining vs
  prediction; is `knowledge_level` a cleaner signal than source identity?
- **Contestation detection:** how the LLM confirmation flags a refuted causal edge (and whether that
  down-weights or just annotates).
- **Multi-source legs:** a leg may have multiple edges of different predicate types between the same
  pair — how to aggregate (best predicate? curated-source-preferred? union of publications?).
- **Calibration reuse:** swap the scorer behind the U8 harness; do the synthetic panel bridges need
  real KG edges (vs the synthetic `predicate_directions` fixtures)? Likely yes — v2 needs real leg
  edge provenance, so the panel chains must resolve to real Kestrel edges (ties to O2/entity names).

## Sources & References

- v1 failure analysis: `backend/assessment_data/BRIDGE_GROUNDING_TIER_A_FINDINGS.md`
- v1 plan (substrate-independent units to reuse): `docs/plans/2026-06-08-002-feat-bridge-grounding-scorer-plan.md`
- skimgpt mechanism mining (porpoise gate = learned Triton model): `docs/references/km-gpt-dch-competing-hypotheses.md`
- Kestrel API (edge schema, sources, modes): `.claude/skills/kestrel-api`
- Live Kestrel probe (2026-06-17): typed predicates + provenance confirmed on CHEBI:27732, MONDO:0005192, NCBIGene:7157.
