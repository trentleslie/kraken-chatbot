# Bridge-Grounding v2 (KG-Provenance) — Investigation & Reframe

**Outcome: v2's KG-provenance gate is NOT a reliable per-bridge mechanism-confidence scorer.**
Three near-zero-cost Kestrel probes converged: the curated-causal signal is real but **weak and sparse**.
Decision (2026-06-17): **reframe from a mechanism-confidence SCORE to a deterministic evidence-provenance
LABEL** (honest transparency, no LLM, no calibration gate). This supersedes the scorer framing in
`docs/plans/2026-06-17-001-feat-bridge-grounding-v2-kg-provenance-plan.md`.

## The probes (all Kestrel-only; no LLM/PubMed; artifacts persisted)

| Probe | What it measured | Result |
|-------|------------------|--------|
| `kg_provenance_probe.py` | curated-AND-causal on DIRECT A→C drug→disease edges | 6/6 pos, 1/5 neg — **misleadingly optimistic** (tested direct edges, not bridge legs; positives are FDA drugs with guaranteed curated `treats`; `treats` counted as causal) |
| `kg_bridge_leg_probe.py` | curated-causal coverage on REAL 2-hop bridge LEGS (A→B→C, discovered B) | **23%** of legs curated-causal; **only ~5% of bridges have BOTH legs curated-causal** → a strict gate marks ~95% of real bridges `insufficient` |
| `kg_bridge_graded_probe.py` | does a TIER-WEIGHTED graded chain score separate mech vs spurious endpoints? | **AUC 0.61** per-bridge (0.70 per-endpoint-best) — weak; 3/5 real mechanisms score identically to spurious (curated-`related_to` ceiling at 0.5) |
| (combine) KG provenance + edge `publications` | does literature backing sharpen it? | **AUC 0.610 → 0.588** — no improvement (publications cover 64% of legs in BOTH classes → popularity trap, anti-discriminative) |

## Why the scorer fails (consistent across probes)

1. **Curated-causal coverage is sparse on real bridges** (23% of legs; ~5% both-legs). The pipeline
   surfaces novel/sparse connections through discovered middle nodes — exactly the bridges whose legs
   are NOT in curated mechanism DBs. The score would measure *curated-DB coverage*, not bridge validity.
2. **The curated-`related_to` (neutral) tier dominates** both real and spurious bridges → no discrimination.
3. **Literature backing is anti-discriminative** (spurious chains are well-studied too) — the v1 failure
   mode in a new form.
4. The initial 6/6 "validation" over-claimed: it tested direct curated drug→disease edges (selection
   bias: positives = FDA drugs, negatives = debunked), not the legs of discovered bridges.

## The reframe — evidence provenance LABEL (not a score)

Surface what the KG honestly provides, deterministically, per bridge:
- **Per leg:** best evidence tier ∈ {`curated-causal`, `curated-associative`, `curated-neutral`,
  `text-mined`, `none`} (from the leg edges' `knowledge_level`/`agent_type`/predicate-class).
- **Per bridge:** a chain summary, e.g. "both legs curated-causal" (strongest) → "text-mined only" →
  "no KG edge".

Honest, deterministic, cheap (no LLM), shippable, low-risk. **No Tier A calibration gate** is needed —
there is no confidence claim to falsify; it reports what evidence exists, not how true the bridge is.
Reuses U0/U1 (predicate + provenance extraction, the model/state), drops the scorer + LLM layers.

## Reusable assets

- `backend/assessment_data/kg_provenance_probe.py`, `kg_bridge_leg_probe.py`,
  `kg_bridge_graded_probe.py` (artifacts under `kg_bridge_leg_runs/`, gitignored).
- v1 (co-occurrence) and v2 (KG-provenance) both characterized: per-bridge mechanism grounding from
  available signals is not tractable; evidence-transparency labeling is.
- Binding upstream dependency throughout: entity resolution returns wrong-namespace CURIEs
  (`[[biomapper2-wraps-kestrel-hgnc-marker]]`) — the highest-leverage fix for the whole pipeline.
