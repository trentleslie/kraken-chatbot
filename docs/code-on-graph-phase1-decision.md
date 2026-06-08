# Code-on-Graph — Phase-1 Decision Memo

**Date:** 2026-06-07
**Inputs:** N=100 corrected-gate result (`docs/code-on-graph-phase0-results.md`) · feasibility (`docs/code-on-graph-feasibility.md`) · gate-corrections plan (`docs/plans/2026-06-07-001-fix-cog-spike-gate-corrections-plan.md`)
**Purpose:** Frame the build/don't-build decision the spike teed up, and specify what a Phase-1 run must add to make it cleanly answerable.

---

## Where the spike left it

Phase-0 (N=100, corrected gate) returned **INCONCLUSIVE**, but it settled most of the uncertainty:

- **Settled — iteration helps, significantly and grounded.** Any-one recall **0.37 → 0.50 (+0.13)**, McNemar **p=0.0002**, iterate never loses (b=0, c=13); biggest gains on **t2d (+0.30)** and the hard **random** tranche (+0.125). **Finding-level hallucinations = 0** — no win depended on an ungrounded query.
- **Settled — the metric and the grounding rule.** The any-one primary (vs the strict measurement artifact) and the finding-level grounding override are now correct and validated; the 48 query-arg-leakage events are caveat-only.
- **Unsettled — three things**, which are the entire Phase-1 agenda below: **(1) magnitude vs the bar, (2) reproducibility under non-determinism, (3) precision and cost — both unmeasured.**

## The crux: does ~+0.13 grounded recall justify the L+ build?

The L+ build is a dedicated `graph_reasoning` node in `builder.py:build_discovery_graph` (new `@validate_state` contracts, `operator.add`-reduced fields, error propagation) — the largest, permanent commitment on the integration ladder, vs cheaper rungs (modify `integration.py:detect_bridges_via_api` at "L"; `direct_kg`/`pathway_enrichment` at "M").

**The benefit** is real but **below the pre-registered bar**: any-one lift **+0.13 < 0.15**. By the letter of the pre-registration that is a NO-GO; the question is whether a *significant, grounded* +0.13 on the hardest mechanistic queries is worth building despite falling just short of a line we set before seeing the data.

**The cost** (corrected — the N=100 `cost_advisory` over-reported Kestrel calls ~150× because `rest.kestrel_calls` is a shared cumulative counter that was summed across loops; now fixed to a per-loop delta):
- True iterate-arm cost ≈ **1,919 Kestrel + 1,500 LLM calls** for 100 items × K=3 → ~**6–7 Kestrel + 5 LLM calls per discovery at K=1**, vs baseline's **1 Kestrel, 0 LLM**.
- **Every N=100 loop ran to the 5-turn cap** (all 300 terminal states `turn-cap-hit`) — the LLM never voluntarily stopped, so every query paid full cost even when the bridge was found on turn 1. A **stagnation early-exit** (stop after `stagnation_patience` consecutive 0-new-path turns; `config.stagnation_patience`) is now in the arm to cut that waste. This makes the cost denominator partly *reducible*.

So the trade is: **+13% bridge recall on hard queries** for a **~5–7× Kestrel multiple + a new per-query LLM cost the lean static plan doesn't incur + variable latency + a permanent new node to maintain.** This is the "efficiency can invert" risk from the feasibility doc, confirmed empirically.

## Unsettled 1 — stabilize the arm (the temp-0 vs larger-K fork)

INCONCLUSIVE was emitted because **one discordant pair flapped** across the K=3 reruns (no temperature control → same item, different LLM query specs → sometimes finds the bridge). Discordant cells carry all the McNemar signal, so one sample-dependent cell makes the p-value untrustworthy → the gate refuses a verdict.

Two stabilization paths, with a real tension:
- **Temperature 0** → near-deterministic specs → identical reruns → no flap → trustworthy McNemar. *But* the K-rerun **variance band becomes theatre** (three identical runs measure nothing), and — **open question** — it is unverified whether `ClaudeAgentOptions` (the Agent SDK the iterate arm uses) **exposes temperature at all**; it may require switching the arm's LLM call to the raw Anthropic Messages API.
- **Larger K + flapping tolerance** → keeps non-determinism but dampens it via majority and an explicit tolerance instead of zero-tolerance. More expensive, less clean, but keeps a meaningful variance measure.

**Recommendation:** verify Agent-SDK temperature support first. If available, run temp-0 as the primary (clean point-verdict) with a small temp>0 K-band reported as a sensitivity. If not, go larger-K with a defined flapping tolerance. Either way, **a stabilized run converts INCONCLUSIVE into a real verdict** — most likely a clean **NO-GO-on-magnitude** if the lift holds at ~+0.13, though temp-0 may shift the lift either way (a less exploratory LLM could find fewer bridges, or be more consistent on the ones it does).

## Unsettled 2 — precision / EITL (and why NOT "only the 13")

Recall asks *"did we find the known answer"*; **precision asks *"of what we'd actually report, how much is right."*** These are different axes, and the recall gate never measured the second.

**Do not grade only the 13 CoG-won items** — that is the wrong sample and biases the result:
- The 13 are **correct by construction** (they matched curated gold); grading them is circular → trivially ~100%, learns nothing.
- The real risk of an iterate-until-found loop is **false confirmation**: surfacing spurious-but-real hub paths as findings. That risk lives in the **off-gold** output (`intermediates`), which recall ignored.

**Correct EITL design** (TREC-style pooling — Buckley & Voorhees, Zobel):
- Grade the **candidate bridges each arm actually surfaces as findings** (off-gold intermediates + claimed mechanism), not just the gold node.
- **Both arms**, **blinded** to arm and to gold.
- Focus effort on **discordant items** (where arms differ — concordant items carry no comparative signal), but on each grade *both arms'* surfaced findings.
- Output: **precision per arm + false-confirmation rate**. The decisive question: *does iteration's +13 recall come with proportionally more junk findings?* If so, the precision tax can erase the recall win. κ floor (`config.kappa_floor`) gates inter-rater reliability.

## Recommended Phase-1 shape

1. **Re-pre-register** a fresh frozen config (the cost-counter fix is neutral, but the **stagnation early-exit changes the arm**, so the N=100 result and any new run are not directly comparable). Keep the kill thresholds; add the stabilization choice and `stagnation_patience` to the freeze.
2. **Stabilized recall re-run** (temp-0 or larger-K) → a clean PROCEED / NO-GO / still-INCONCLUSIVE on magnitude.
3. **EITL precision arm** as specified above → precision + false-confirmation rate.
4. **Decision gate:** PROCEED to the L+ build only if recall lift clears the bar *and* iteration's precision/false-confirmation is no worse than baseline. A significant-but-sub-threshold recall lift with a clean precision profile is a judgment call (honor the bar, or accept the bar was slightly high); a sub-threshold lift *with* a precision tax is a clean NO-GO.

## Already done (this session)

- ✅ Cost-counter overcount fixed (per-loop Kestrel delta) — `iterate_loop.py`.
- ✅ Stagnation early-exit added (`config.stagnation_patience`, default 2) — cuts the always-run-to-cap waste; flagged as an arm change requiring re-pre-registration.
- ✅ Both covered by tests (`test_iterate_loop.py`): stagnation stop, and per-loop-delta cost.

## Open questions for Phase 1

- Does `ClaudeAgentOptions` expose temperature, or must the iterate arm move to the raw Messages API for determinism?
- Is the EITL expert campaign (live human grading) in scope/budget, or is a lighter automated-judge precision proxy acceptable as a first pass?
- What absolute recall lift, at what precision parity, actually justifies the L+ build's permanent cost? (The pre-registered 0.15 was a guess; Phase-1 should set it deliberately against the now-measured cost.)
