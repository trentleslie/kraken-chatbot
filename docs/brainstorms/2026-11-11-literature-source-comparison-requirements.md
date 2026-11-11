---
date: 2026-11-11
topic: literature-source-comparison
---

# Literature-Source Comparison for KRAKEN Grounding

## Problem Frame

The discovery pipeline's `literature_grounding` node attaches papers to each KG-derived hypothesis. It fans out to five sources (KG edge publications, PubMed, OpenAlex, Exa, Semantic Scholar) and merges the results. On dev run `cdf9fa68` (modules test panel, T2D, 2026-11-09), every one of the 30 kept papers came from OpenAlex (16 of 48 raw) and Exa (14 of 56 raw). KG returned 0, PubMed 0 raw, and S2 was 403. Nobody has measured whether the papers that do come back are evidence for the hypotheses, or which sources are worth keeping. Elicit and Consensus are available but not wired in.

PubMed, OpenAlex and Exa currently receive one natural-language sentence (`build_search_query`, called without the entity-name map; OpenAlex truncates it to 200 chars, Exa to 500). S2 runs three strategies (that sentence, an entity-name query, claim keywords) and merges them. A single sentence likely suits the semantic engines (Exa, OpenAlex) and disadvantages Boolean/MeSH engines (PubMed). A comparison that only uses the deployed query would partly measure query fit, not source quality.

**Outcome wanted:** a one-time, evidence-backed **source-mix decision**: which sources KRAKEN keeps, adds, fixes the query for, reweights, or drops, and whether a certified link-evidence filter should sit in front of what users see. This is not a reusable benchmark.

Three of the five sources returned nothing on that run because of plumbing, not quality. So the work starts with a cheap gate (G0 below), and the full study runs only if the gate shows the mix is still wrong.

## Requirements

**Gate (runs first)**
- G0. **Lifted 2026-11-11 (Trent):** the spike proceeds now, without waiting for the S2 key. The harness uses its own verified PubMed client (R7a), so #126 cannot contaminate it; S2 is reported unavailable until the new key lands. The original gate, kept for the record: fix or explain the three zero-yield sources: PubMed `raw=0` (#126, separate PR), install the new S2 key, and find out why KG edge publications return 0. Then rerun about 3 normal pipeline runs on different panels and record per-source yield and failure mode. The full study (R1-R16) proceeds only if, after the fixes, the kept papers still come mostly from one or two sources or still look off-link on a quick read. Otherwise the study stops here, and the outcome is recorded as "fixed plumbing; mix decision not needed".


**Hit definition and judging**
- R1. A returned paper is a **hit** for a hypothesis only if it reports evidence on the specific relationship the hypothesis claims, whether that evidence supports or refutes it. Discussing both entities without addressing the link is not a hit.
- R2. Each paper is judged from its title and abstract by an LLM judge with a fixed prompt, **blind to which source returned it**. The judge records hit/not-hit, direction (supports / refutes / n.a.) and a one-line reason.
- R2a. **Abstract normalization:** every returned paper is resolved to a canonical ID (PMID and/or DOI, through the NCBI ID converter or OpenAlex) and its abstract is fetched from one resolver regardless of source (PMID → PubMed EFetch, DOI → OpenAlex `abstract_inverted_index`). Source-written summaries (Elicit, Consensus, Exa highlights) are not judged. The per-source share of papers with no retrievable abstract is reported; those papers are excluded from precision and counted separately.
- R3. The judge is certified against a blinded human audit. Trent labels a stratified sample of about 50 judgments without seeing the judge's verdict. The sample is stratified by source and agreement is reported per source as well as pooled. If agreement on hit/not-hit is below the threshold set in planning, judge-scored numbers are not reported as findings.

**Hypothesis set** ("hypotheses" below means the full evaluated set: real hypotheses plus controls)
- R4. **Real hypotheses:** about 10 hypotheses taken from actual KRAKEN pipeline outputs, spanning several panels and hypothesis types (bridges, inferred roles), with their source run recorded. They are drawn from the pre-grounding stage, or planning confirms that no pipeline stage filters or ranks hypotheses using grounding output (survivorship would favor the incumbent sources).
- R5. **Known-true controls:** about 6 links with well-established literature, each with at least one pre-registered landmark paper (PMID) chosen before any source is queried. Example: BCAA → insulin resistance.
- R6. **Fabricated-false controls:** about 4 plausible-sounding links with no supporting literature, each verified before the run by an explicit search across all sources. Any paper the judge marks "supports" on these is a false-support event. Every false-support event is reviewed by Trent (not sampled), and results are reported as raw counts, not rates.
- R6a. Every hypothesis, controls included, is pinned as a serialized `Hypothesis` (title and claim written in the same template as pipeline bridges) plus `disease_context` plus the entity-name map, so Arm 1 queries are reproducible.

**Sources and query arms**
- R7a. **Error-transparent calls:** the harness classifies each call as ok / empty / error / unavailable from the raw HTTP response, not from the result count. The existing clients turn errors into `[]` (OpenAlex on any HTTP error, S2 on everything but 429, Exa without a key, PubMed on generic exceptions), so the harness either calls each endpoint directly or wraps the clients so status codes reach it. PubMed is queried through a direct E-utilities call checked against a known query with a known PMID count, independent of the #126 code path.
- R7. Sources in scope, in tiers:
  - **Tier 1 (decides keep/fix/drop):** PubMed, OpenAlex, Exa, Semantic Scholar.
  - **KG edge publications:** measured, but **kept on product grounds** (KRAKEN's grounding comes from its own KG). Excluded from drop/reweight verdicts; its numbers inform why it yields what it does.
  - **Consensus is promoted into the spike (Trent, 2026-11-11).** Access is the claude.ai Consensus connector on the free plan, driven by in-session tool calls rather than the script (no REST key; the API is approval-only). Its raw results are written to the run directory like every other source. The free plan returns no DOI or PMID, so its papers are resolved to canonical IDs by title through OpenAlex, and the unresolved share is reported (R11). Consensus ranks the S2/OpenAlex/PubMed corpus with its own engine, so it is not a substitute for S2.
  - **Tier 2 (add candidates, only if budget remains):** Elicit. Each must first pass the R13 gate (headless API, automated-use terms, per-query cost at public volume). Consensus is dropped, not worked around, if it cannot run from a script with little effort.
 A source that cannot run is recorded as **unavailable with a reason**, never scored as 0 hits. This applies to S2 without a new key, and to Consensus if it cannot be scripted headlessly.
- R8. **Arm 1 (as-deployed):** each source receives exactly the call `parallel_search_hypothesis` makes today, with the same arguments. For S2 that is the deployed three-strategy merge, cut to the top 5. KG edge publications have no query string; they take part in Arm 1 only (hypothesis entities → edges → PMIDs) and skip Arm 2.
- R9. **Arm 2 (native):** each source whose natural query form differs also receives a query in that form, built by a fixed, pre-registered rule per source. The rules are frozen and hashed before the control links and landmarks are chosen, and are never tuned against the scored controls. PubMed: Boolean of entity names (+ synonyms) AND the disease. Elicit and Consensus: a question. Sources whose native form is the sentence skip Arm 2.
- R10. Each source returns its top 5 per query. Arms and sources are compared at equal k.

**Headline: what users see**
- R10a. **End-to-end arm:** run the production merge-and-filter (`merge_literature` and the kept-paper cutoff) on the same hypotheses for the current source mix and 1-2 candidate mixes chosen from the per-source diagnostics, then judge the kept set. Kept-set evidence-precision and false-support are the **headline numbers**; per-source numbers (R11) are diagnostics that explain them. This is also what grounds any "reweight" verdict.
- R10b. **Judge-as-filter option:** for each mix in R10a, also report kept-set precision and false-support after dropping papers the certified judge marks not-hit. That way the recommendation can be "add a link-evidence filter" instead of, or as well as, a source change.

**Diagnostics (per source)**
- R11. Per source × arm:
  - evidence-precision@5, computed as hits ÷ papers returned (up to 5, excluding papers with no abstract), with the returned count reported alongside;
  - unique hits (hits no other source found), with papers matched across sources by canonical DOI/PMID (normalized-title fallback), and the per-source share that could not be resolved reported;
  - landmark recall on the known-true controls;
  - false-support rate on the fabricated-false controls;
  - median latency;
  - failure mode per call (ok / empty / error / unavailable);
  - cost per query.
- R13. Licensing/ToS for automated use in a public-facing tool is recorded per source as a yes/no/unknown adoption gate, separate from the quality numbers. It is checked **before** a Tier 2 source is scored, and an "add" verdict requires a "yes".

**Decision rule (pre-registered)**
- R16. Before the run, a written rule maps the numbers to verdicts: thresholds for keep / add / fix query / reweight / drop and for adding the judge filter. Confidence intervals are bootstrapped over hypotheses, not papers. A difference inside the noise band means **keep as-is**, not a ranking. Hypotheses where no source found any hit are reported separately as false-support/noise probes, not precision probes.

**Budget**
- R17. Hard cap for the full study: **$25** of LLM judging plus Elicit/Consensus credits, and **about 1 hour** of Trent's audit time (about 50 judgments, R3, plus every false-support event, R6). If the timing run projects over the cap, these pre-decided cuts apply in order: drop Tier 2 sources; drop Arm 2 for sources whose native form is the sentence; trim real hypotheses from 10 to 6.

**Reproducibility**
- R14. Every raw response, judgment and timing is persisted by default to a timestamped run directory, never behind a flag. Inputs are pinned: hypothesis set, query rules, judge prompt and model, repo SHA, KG build fingerprint, and date per source call.
- R15. The run is resumable. A 5xx or timeout is retried and never cached as an empty result.

## Success Criteria
- G0 is recorded first. If the gate stops the study, the success criterion is the recorded gate outcome.
- A recommendation table: for each source, **keep / add / fix query / reweight / drop**, plus a yes/no on the judge filter. Each verdict is backed by the headline numbers (R10a/R10b) and diagnostics (R11) under the pre-registered rule (R16), with judge certification (R3) passed.
- The PubMed question is answered with one of three outcomes: **client failure** (the verified direct call returns results where the pipeline client returns 0, so the cause is #126); **query failure** (Arm 1 low, Arm 2 fine → fix the query); or **corpus/ranking failure** (both low on the verified client). "Low" and "fine" use the pre-registered decision rule.
- Both planted controls behave as expected (see Validation controls) before any number is reported.
- Every reported number traces to a file in the run directory.

## Validation controls (BootLoops)
- **Planted truth:** the known-true links (R5) must show landmark recall above zero in at least one source. If no source finds any landmark paper, the harness or judge is broken and the run halts. Any available source with zero landmark recall across all known-true controls is investigated before its numbers are used. Landmark recall is a sanity check, not a decision input on its own.
- **Planted corruption:** one known-true link is re-run with its entities swapped for unrelated ones. The judge must not mark the original landmark paper as a hit for the corrupted link. If it does, the run halts, the same as a planted-truth failure.
- **Independence:** the LLM judge produces the hit labels; Trent's blinded audit certifies them (R3). The audit sample is drawn by rule before he sees any verdict, so the two cannot fail the same way. Landmark recall is certified by canonical-ID match against the pre-registered list (each landmark registered as a PMID plus DOI pair), not by the judge.
- **Timing:** a single-hypothesis timing run across all sources comes before the full run. A multi-hour full-run projection means cutting k or hypotheses, not scaling up.

## Scope Boundaries
- No change to the production `literature_grounding` node in this work. Any source-mix, query or filter change it recommends is a separate PR.
- Not a reusable benchmark, Inspect solver, or AstaBench scoring. The 2026-05-29 AstaBench plan stays deferred.
- No full-text reading. Judgments use title and abstract only, which is a stated limitation.
- No relevance scoring beyond hit/not-hit plus direction. No graded scale.
- The #126 fix and the S2 key install are prerequisites (G0) handled in their own PRs/ops steps, not inside the study harness.
- No ongoing per-source health monitoring in this work. Production logging of per-source failure modes is a recommended follow-up (the silent S2/PubMed failures went unnoticed).

## Key Decisions
- **Source-mix decision over reusable benchmark:** Trent wants the call made, not a harness to maintain.
- **Hit = evidence on the link (supports or refutes):** that is what grounding exists for. Topical papers that name both entities are what make a false hypothesis look supported.
- **Two query arms:** without them, PubMed and question-style tools are judged on our query, which would bias the decision toward the semantic engines we already use.
- **Fix-first gate:** the cited symptoms are plumbing failures; the cheap fix may make the study unnecessary (Trent, 2026-11-11).
- **Headline = what users see:** sources are scored on the merged, kept set, with a judge-as-filter option, because a source's isolated top 5 is not what reaches the report.
- **KG kept on identity:** KG-native grounding is KRAKEN's positioning; dropping it is not a decision this study makes.
- **True/false controls double as planted truth:** they cost little extra and give real recall and false-support numbers.

## Dependencies / Assumptions
- A new S2 key arrives before the full run; otherwise S2 is reported unavailable (R7).
- The Elicit REST key at `~/.config/elicit/token` has enough credits for about 20 hypotheses × 2 arms × top 5. (Unverified: check credits in planning.)
- Consensus has a headless/API path usable from a script. (Unverified: the available connector is an interactive MCP.)
- KG edge publications depend on the KG build at run time; record its fingerprint.
- Elicit is reachable only through the personal helper (`~/.local/bin/elicit.py`: forces IPv4, browser User-Agent), so Tier 2 runs on pop-os. Its corpus and mode settings must be fixed per arm.

## Outstanding Questions

### Resolve Before Planning
(none)

### Deferred to Planning
- [Affects R3][Technical] Judge model choice (preferably a different family from the hypothesis generator) and the agreement threshold (e.g., ≥85% on hit/not-hit, or Cohen's κ ≥ 0.7).
- [Affects R16][Technical] The specific verdict thresholds and noise band.
- [Affects G0][Needs research] Why KG edge publications yield 0 (collection path in state vs. coverage).
- [Affects R4][Technical] Which past runs supply the 10 real hypotheses, and the selection rule (pre-registered, not cherry-picked).
- [Affects R5, R6][Needs research] The specific control links and their landmark PMIDs, verified against PubMed before the run.
- [Affects R9][Technical] The exact native-query rule per source, including PubMed synonym expansion (MeSH or KG synonyms).
- [Affects R7][Needs research] Consensus API availability and terms; Elicit API rate limits and cost per call.
- [Affects R11][Technical] How "KG edge publications" is queried standalone (hypothesis entities → edges → PMIDs) outside a pipeline run.

## Next Steps
→ `/ce:plan` for structured planning.
