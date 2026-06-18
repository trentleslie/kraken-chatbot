# Bridge-Grounding Tier A Calibration — Findings

**Verdict: FAILED (the riskiest assumption is NOT supported).** Per the plan's gate, do NOT proceed
to ranking-trust / U5–U7 without a methodology rework. Run artifacts (gitignored, recomputable from
frozen label snapshots) under `assessment_data/bridge_grounding_tier_a_runs/`.

## Runs

| Run | Panel | Verdict | Why |
|-----|-------|---------|-----|
| `20260616T014211Z` | v1 middle nodes (E7 oncoprotein, estrogen receptor signaling, …) | `not_evaluable` | 3/5 chains had a binding leg < 5 labeled abstracts — sparse middle-node co-occurrence (O2) |
| `20260616T022439Z` | revised middle nodes (probe-verified ≥15 PMIDs/leg) | **`passed=False`, margin=−0.333** | Fully evaluable, but the scorer does not separate real mechanisms from spurious chains |

## The result (run 2, headline support_fraction = weaker leg)

| Chain | Polarity | Headline | Note |
|-------|----------|----------|------|
| coffee → caffeine → pancreatic cancer | **hard negative** | **0.62** | **highest of all** — caffeine↔pancreatic association studies labeled "support" |
| HPV → cervical dysplasia → cervical cancer | positive | 0.54 | dysplasia→cancer leg 23/45 off-topic |
| beta-carotene → oxidative stress → lung cancer | negative | 0.54 | both legs are real *associations* → scores like a positive |
| HRT → estrogen → coronary heart disease | negative | 0.49 | HRT→estrogen leg 86% off-topic; estrogen→CHD 16 "support" |
| H. pylori → chronic gastritis → peptic ulcer | positive | 0.29 | gastritis→ulcer leg 15 "neither" dragged a TRUE positive down |
| MMR → bowel disease → autism | negative | 0.11 | correctly refuted (4 refutes on the MMR→bowel leg) |

margin = min(positives 0.29) − max(negatives 0.62) = **−0.333** (pre-registered floor: ≥ +0.30).

## Root cause — co-occurrence ≠ mechanism (confirmed)

The score tracks "are these two terms co-studied with positive-sounding language", NOT "is this a
mechanism". Four reinforcing failure modes, all visible in the per-leg counts:

1. **Association ≠ mechanism leaks through labeling.** caffeine→pancreatic cancer got 13 "support"
   (epidemiology/association abstracts) despite being a known null chain. The U3 prompt's
   epidemiology-association guard exists but is not applied strictly enough by the labeler.
2. **Real positives are fragile under min-leg gating.** H. pylori→ulcer is real, but the
   gastritis→ulcer leg returned 15 "neither" (0.29) and the min dragged the whole chain down.
3. **Off-topic pool pollution** on broad middle nodes (estrogen→CHD leg 86% off-topic; HPV
   dysplasia→cancer 23/45 off-topic). The `off_topic` label catches some but the pools are noisy.
4. **Spurious-but-studied chains score like mechanisms** (beta-carotene→oxidative→lung 0.54): both
   legs are real associations, so the chain reads as supported even though it is a refuted/non-mechanism.

This is precisely the plan's stated riskiest assumption: skimgpt gated its pools with a *learned
relevance classifier* ("porpoise") that we deliberately did NOT port; the lightweight `off_topic`
LLM label was the stand-in, and it is **not sufficient** to recover mechanism from co-occurrence.

## Implications

- **Methodology-level failure, not tuning.** Better middle nodes fixed *evaluability* but not
  separation. The hard negative scoring highest is a structural property of co-occurrence + abstract
  labeling, not a threshold artifact.
- **Candidate reworks (a real decision, not a tweak):** (a) port/replace the learned relevance
  pre-filter; (b) move from co-occurrence pools to relation-extraction (SemRep/SemMedDB predications)
  so the pool is already mechanism-typed; (c) make labeling far stricter on association-vs-mechanism
  (and test the guard against this panel); (d) reconsider whether abstract-only + LLM can ever
  separate, i.e. whether the v1 framing is viable at all.
- **The gate worked exactly as intended:** ~2 paid runs definitively answered the load-bearing
  question with NO *before* building U5–U7 or surfacing a misleading score to researchers.

## SDK reproducibility caveats (verified empirically)

The Claude Agent SDK exposes no `temperature`, AND setting `ClaudeAgentOptions(model=...)` (prefixed
or bare) breaks the bundled CLI subprocess (exit 1) — only the no-model path works. So neither temp
nor model is pinnable via the SDK; reproducibility rests on the frozen per-abstract label snapshots.
