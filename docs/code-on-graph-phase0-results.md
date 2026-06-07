# Code-on-Graph — Phase-0 Results (Recall Kill-Test)

**Date:** 2026-06-07
**Spike:** `backend/tests/code_on_graph_spike/` (branch `feat/code-on-graph-spike`)
**Pre-registration:** `backend/tests/code_on_graph_spike/config.py` (frozen 2026-06-03; git timestamp is the freeze proof)
**Plan:** `docs/plans/2026-06-03-002-feat-code-on-graph-spike-plan.md` · **Feasibility:** `docs/code-on-graph-feasibility.md`
**Verdict:** ⚠️ **Not a clean NO-GO and not a clean PROCEED** — significant-but-modest, grounded iteration advantage; the pre-registered *primary* metric was the wrong yardstick for multi-bridge gold.

---

## TL;DR

- **Question:** does an LLM-driven *iterative* Kestrel query-refinement loop recover mechanistic bridge nodes that Kraken's *static one-shot* query plan misses — enough to justify building it?
- **Primary metric (strict: recover *all* gold interior nodes in one path):** baseline 0.21 → iterate 0.26, McNemar exact b=0/c=4, **p=0.125 (n.s.)** → NO-GO.
- **Secondary metric (any-one: recover *any* gold interior node — pre-registered):** baseline 0.33 → iterate **0.48**, b=0/c=13, **p=0.00024 (highly significant)**. Iterate **never loses a single item**; biggest gain is on the *hardest* stratum (random 13→23).
- **The strict NO-GO is largely a measurement artifact.** The "recover all 3 interior nodes in one returned path" bar is structurally near-impossible for the multi-bridge DrugMechDB `random` items — both arms fail ~all 70 (2/70 vs 3/70). That concordant-miss wash diluted the lift and starved McNemar of discordant pairs.
- **Grounding is clean.** No iterate win depended on an ungrounded query. Finding-level hallucination is ~0 by construction (a "hit" only counts paths returned by executed Kestrel calls). The grounding violations that exist are incidental query-argument leakage on *non-winning* items.
- **Honest read:** iteration genuinely surfaces buried bridge nodes, significantly and grounded — but the *magnitude* (+14.4% absolute) lands just under the pre-registered PROCEED thresholds. This is a judgment call on whether moderate, real lift justifies the L+ build, not an automatic kill.

---

## 1. Background & hypothesis

Every Kraken discovery node today "fetches then reasons": it runs one fixed query (or a fixed fan-out), then the LLM summarizes the result with `allowed_tools=[]`, `max_turns=1`. **No node lets the LLM steer querying based on what it finds.** The "Code-on-Graph" paper (arXiv:2606.03705) is that missing loop.

The feasibility review (`docs/code-on-graph-feasibility.md`) established that CoG's published wins come mostly from *keeping the KG out of the prompt* — which Kraken already does — so its numbers give ~zero predictive evidence for Kraken's payoff. The single genuinely testable claim:

> **H1:** iterative query-refinement recovers bridge nodes that a static, equal-depth one-shot query misses.

Phase 0 is a deliberately cheap **kill-test** of H1 on a recall metric, before any production code (`integration.py`) is touched or any expert-precision (EITL) campaign is built.

## 2. Method

### Two arms (paired, per gold item)

- **`baseline` (control)** — one `multi_hop(start, end, max_path_length=5, limit=100)`. This is Kraken's current behavior at full reach. (`baseline.py`)
- **`iterate` (treatment)** — **Turn 0 is the baseline query, for free**, so iterate ⊇ baseline by construction and any delta is the *marginal* value of iterating, not extra search depth. The LLM then emits one typed JSON query spec per turn (`multi_hop` / `one_hop` / `hybrid_search`), the harness validates it against a verb whitelist, dispatches to Kestrel REST, accumulates distinct paths, and feeds results back — up to a hard **5-turn cap**. No temperature control exists, so each item is run **K=3 times, scored by majority**. (`iterate_loop.py`)

The LLM never sees the gold bridge; the harness scores recovery identically for both arms.

### Gold set (N=90, stratified)

DrugMechDB drug→mechanism→disease chains, START (drug) and END (disease) given, **interior bridge node(s) hidden** as the gold answer:

| Stratum | N | Character |
|---|---|---|
| `alzheimers` | 10 | hand-picked anchors, ~1-bridge |
| `t2d` | 10 | hand-picked anchors, ~1-bridge |
| `random` | 70 | randomly sampled (seed `20260603`), mostly **3-bridge / 4-hop** — hard |

N=90 = the pre-registered **powered floor** (80% power at a 15pp effect, discordance prior π_d=0.25).

### Pre-registered gate (`config.py`, frozen before any results)

- **Recall lift (P1):** absolute lift ≥ **0.15**, *or* (if baseline R0 > 0.85, unattainable) recover ≥ **50%** of static's misses.
- **Significance (P2):** McNemar **exact** binomial, two-sided α=0.05, primary.
- **Cost (P3):** worst-case loop ≤ 3× baseline; hard 5-turn cap; ≤8 Kestrel calls/turn.
- **Grounding (R9):** every CURIE the loop emits must be a node returned by a prior executed call (or an `equivalent_id` of one). Pre-registered as: any hallucinated CURIE > 0 → **NO-GO override**.
- **Bridge unit:** primary = **strict** (returned path contains *all* gold interior nodes); pre-registered secondary = **any-one** (path contains *any* gold interior node).

## 3. Worked example — `metformin → type-2 diabetes` (gold bridge = AMPK)

| Field | Value |
|---|---|
| START (given) | `CHEBI:6801` (metformin) |
| END (given) | `MONDO:0005148` (type-2 diabetes mellitus) |
| Gold bridge (hidden) | `NCBIGene:5562` (PRKAA1 / AMPK) |

**Baseline:** `multi_hop(CHEBI:6801, MONDO:0005148, depth=5, limit=100)` → 100 paths, **AMPK absent** → miss. The bridge was *buried* in the top-100, not absent.

**Iterate** (real transcript; `gv=0`, fully grounded):
```
turn 0  seed: multi_hop(CHEBI:6801, MONDO:0005148, depth=5)              → 100 paths (== baseline, AMPK absent)
turn 1  one_hop(CHEBI:6801)
turn 2  multi_hop(CHEBI:6801, MONDO:0005148, max_path_length=4,
                  predicate="biolink:affects")                          → predicate-filtered re-query
turn 3-5 one_hop(CHEBI:6801) ×3                                         → terminal: turn-cap-hit
ITERATE hit: True | turns: 5 | gv: 0 | n_paths: 199 | AMPK recovered: True
```

The turn-2 **`biolink:affects`-predicate-filtered re-query** surfaced a slice of the neighborhood the single broad query buried; cumulative paths grew 100→199 and AMPK appeared. Discordant iterate-only win, grounded.

**Contrast cases:** `empagliflozin → T2D` (`CHEBI:82720`) — iterate won via the **seed alone** (`gv=0`); when the broad query already finds the bridge, iterate simply inherits it. And the **pre-fix `iterate 0/16`** false-kill — the LLM wasn't handed START/END and emitted `hybrid_search("Please provide the two nodes…")`, recovering nothing; fixed in commit `302ca62` (START/END hard-coded into the system prompt). All winning transcripts are post-fix.

## 4. Results

| Metric | baseline | iterate | abs lift | McNemar (exact) | reads as |
|---|---|---|---|---|---|
| **Strict** (all interior nodes, one path) — *primary* | 0.21 (19/90) | 0.26 (23/90) | +0.04 | b=0, c=4, **p=0.125** | NO-GO |
| **Any-one** (any interior node) — *secondary* | 0.33 (30/90) | **0.48 (43/90)** | **+0.144** | b=0, c=13, **p=0.00024** | significant win |

**Per-stratum (any-one):** `t2d` 7→10 · `alzheimers` 10→10 (already maxed) · **`random` 13→23 (+10 on the hardest tranche)**.

**Per-stratum (strict):** `t2d` 7→10 · `alzheimers` 10→10 · **`random` 2/70 → 3/70** ← the dead stratum.

Iterate **never loses** under either metric (b=0 throughout): the Turn-0 seed guarantees iterate ⊇ baseline, so every discordant pair is an iterate-only win.

**Against the magnitude thresholds (any-one):** abs lift +0.144 lands *just under* the 0.15 bar; iterate recovers 13/60 = 21.7% of baseline's misses, under the 50% bar. So even on the favorable metric, the **magnitude** is sub-threshold while the **significance** is decisive.

## 5. Grounding-violation analysis

The pre-registered "any hallucinated CURIE → NO-GO" rule nearly killed the spike on the strict run (18 violations across 9 items). Investigation (re-running flagged items instrumented) resolved it:

1. **Findings are never fabricated.** A "hit" only counts paths *returned by an executed Kestrel call*. A hallucinated identifier can at most produce a query that returns nothing or returns real data — it can never *become* a fake bridge. Finding-level hallucination is structurally ~0.
2. **Violations are real but intermittent** — pure non-determinism (no temperature control). The same item (`empagliflozin t2d-04`) scored `gv=2` on the live run and `gv=0` on re-run. So it is not a tracking false-positive; the LLM *sometimes* drops a memorized CURIE in as a **query argument**.
3. **No win was leakage-driven.** All discordant iterate wins had `gv=0`. The 18 violations sit entirely on *non-winning* items. Leakage is incidental, never load-bearing.

**Implication for the gate:** the pre-registered override conflates two things. R9's intent is to prevent *fabricated bridges in findings* (here ~0 by construction). The 18 counts are *query-steering leakage*, a strictly weaker concern. The principled refinement (a pre-verdict clarification, not a post-hoc tune): **hard-fail only if a hit itself depended on an ungrounded query** (checkable, and here: zero); **report `gv` as a caveat** otherwise.

## 6. Diagnosis — why the primary metric mis-fires

The strict bridge-unit demands that a *single returned path* contain *all* interior nodes of a curated 3-node DrugMechDB mechanism. Kestrel rarely holds the exact curated 3-node path, so on the 70 `random` items **both arms fail together** (2/70 vs 3/70 strict). That concordant-miss wash does two things:

1. **Dilutes the lift** — the +0.95-ish recall on anchors is swamped by ~67/70 double-misses, pulling overall strict lift to +0.04.
2. **Starves McNemar** — only discordant pairs carry signal; with ~all-random concordant, just 4 discordant pairs remain → p=0.125 regardless of how real the anchor effect is.

Under the appropriate question for a *discovery* task — *did iteration recover more of the gold mechanism's bridge nodes?* — iteration wins significantly and most on the hard cases. The strict NO-GO is an artifact of the success bar, not evidence that iteration fails.

## 7. Verdict & recommendation

**Significant-but-modest, grounded advantage.**

- ✅ Highly significant by the pre-registered secondary (p=0.0002), never loses, biggest gain on hard items, no leakage-driven wins.
- ⚠️ Magnitude is borderline: +14.4% absolute (just under the 15% line), recovers ~22% of misses (under the 50% line).
- ⚠️ The strict primary returned NO-GO, but for a **measurement reason** (wrong bridge-unit for multi-bridge gold), not a substantive one.

**Recommendation:** treat this as a **defensible lean-PROCEED to a scoped Phase 1**, conditional on two corrections being pre-registered *before* re-running, not after:

1. **Fix the bridge-unit** for multi-bridge mechanisms — "fraction of gold interior nodes recovered (any path)" as primary, with strict retained as a sensitivity report. This is the single biggest lever on the verdict and the current choice is defensibly wrong for DrugMechDB.
2. **Split the grounding gate** — hard-fail on finding-level hallucination (a hit depending on an ungrounded query); report query-argument leakage as a caveat.

If, after those corrections, the magnitude still sits below the pre-registered lift bar, that is itself the answer: iteration helps *moderately* and the call becomes a cost/benefit judgment on the L+ build (a dedicated `graph_reasoning` node) rather than a recall question. Phase 1 should also add the **treatment-fairness guard** (under-spec seed + negative control) — the pre-fix `0/16` false-kill is concrete proof that arm-implementation quality, not iteration itself, can dominate the result.

## 8. Threats to validity / limitations

- **Reimplementation from paper** — CoG hyperparameters (depth, top-K, retry N, schema→class mapper) are unrecoverable; Freebase-QA gains do not transfer to open-ended discovery.
- **No temperature control** — K=3 majority dampens but does not remove run-to-run flapping; the grounding-violation counts are themselves non-deterministic.
- **`hybrid_search` degeneracy** — a small number of iterate runs wasted turns on malformed/clarifying `hybrid_search` specs (see the pre-fix transcript); the post-fix prompt mitigates but does not fully prevent this. Worth a spec-validation tightening in Phase 1.
- **Kestrel top-100 truncation** — `parse_multi_hop_result` truncates to the top paths; the entire effect under test is "iteration surfaces what the single top-100 buried," so this metric is sensitive to the `limit` and ranking, applied identically to both arms.
- **Reproducibility gap (important):** the original full N=90 run was **not persisted to JSON** (`--out` was not passed, and the runner did not yet auto-save). The numbers in this report are transcribed from the live run's progress output and the instrumented re-runs captured in the working session, not from a saved artifact. The runner now **persists every run by default** to `runs/phase0_n<N>_<timestamp>.json` (`run_phase0.py`), so this cannot recur. **Before acting on this verdict, re-run to produce a durable result file** — and pin `drugmechdb_commit_sha` (currently `UNPINNED` in `config.py`).

## 9. Reproduce

```bash
cd backend
# smoke (1 item, K=1):
uv run python -m tests.code_on_graph_spike.run_phase0 --limit 1 --k 1
# full powered run (N=90 × K=3; ~2h of live Kestrel + SDK):
uv run python -m tests.code_on_graph_spike.run_phase0
# every run auto-saves to tests/code_on_graph_spike/runs/phase0_n<N>_<timestamp>.json
# (override with --out PATH). exit code: 0 = PROCEED, 1 = NO-GO, 2 = INCONCLUSIVE
```

---

*Phase-0 kill-test per the staged plan; production pipeline unchanged. Phase 1 (DrugMechDB-scale gold set, EITL precision arm, fairness guard, reproducibility manifest) runs only on a passing recall signal.*
