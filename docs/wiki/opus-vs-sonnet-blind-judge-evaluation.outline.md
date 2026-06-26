# Opus 4.8 vs Sonnet 4: Blind LLM-as-Judge Evaluation of Discovery Reports

This document reports a blind, pairwise large-language-model-as-judge evaluation comparing the discovery reports produced by two models, Claude Opus 4.8 and Claude Sonnet 4, across all eighteen Frailty WGCNA modules run through the Kraken discovery pipeline. Each module was analysed twice, once per model, holding every other pipeline component constant; the two reports for a module were then scored head-to-head by an impartial judge that did not know which model produced which report. We graded on ten parameters drawn from the current literature on hypothesis-generation and biomedical-report evaluation. The result is a like-for-like quality comparison that counting hypotheses cannot provide. The per-module discovery outputs are indexed in the [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm) and synthesized in the [Cross-Module Discovery Synthesis](https://phwiki.phenoma.ai/doc/frailty-multi-omics-wgcna-modules-cross-module-discovery-synthesis-dGektQw73Q).

## Overview

Opus 4.8 was judged the better report in **11 of 18 modules** (Sonnet 7, no ties); excluding the two modules whose Opus run was independently degraded by a known pipeline bug or run variance, Opus led **10 of 16**. Opus scored greater than or equal to Sonnet on **every one** of the ten parameters, with the largest advantages in testability, novelty, groundedness, and mechanistic specificity. The margins are small in absolute terms: both models produced high-quality reports (grand means 4.54 for Opus and 4.41 for Sonnet on a 1 to 5 scale), so the correct reading is that Opus is **modestly but consistently** stronger, not categorically superior. Two facts make the Opus edge credible rather than noise. First, the judge was Sonnet, so any tendency toward stylistic self-preference would favour the Sonnet report; the Opus advantage survives that conservative bias. Second, the comparison inverts the naive hypothesis-count story: Opus generated nearly twice as many hypotheses as Sonnet for the Brown module, yet the blind judge preferred Sonnet's Brown report on quality, confirming that hypothesis volume is not a proxy for report quality.

## Method

The evaluation followed the blind-pairwise protocol recommended for model comparison. For each of the eighteen modules, the Sonnet and Opus synthesis reports were written to neutral files labelled only "A" and "B", with the A/B assignment randomized per module and the model identity recorded in a separate manifest withheld from the judge. An independent judge agent read both reports in full and scored each on the ten-parameter rubric (integers 1 to 5), declared a per-parameter winner and an overall preference with a confidence rating, and returned a structured result. Scores were then de-blinded against the manifest and aggregated. The judge model was Claude Sonnet 4; using the weaker contestant as judge makes the Opus result conservative, because a judge is more likely to reward output resembling its own style. Pipeline components other than the model were held fixed (same modules, same knowledge graph, same `direct_kg` concurrency bound, same `pathway_enrichment` timeout headroom for Opus).

## The ten-parameter rubric

The rubric synthesizes three converging literatures: scientific hypothesis-generation evaluation (Scientific Rigor, Novelty, Testability, Feasibility, Significance, Clarity), biomedical report and summarization faithfulness (groundedness, hallucination, factuality), and retrieval-augmented-generation groundedness (faithfulness, coverage). Eight core parameters are supplemented by two add-ons (coverage and epistemic honesty) tailored to these knowledge-graph reports, which carry explicit evidence tiers.

| Parameter | What it measures |
|---|---|
| Groundedness | Load-bearing claims supported by cited KG evidence; no fabrication |
| Evidence-tier calibration | The `[KG Evidence]` / `[Inferred]` / `[Model Knowledge]` tags match the actual support; no over-claiming |
| Plausibility | Mechanistically sound, consistent with established biology |
| Novelty | Non-obvious links beyond textbook restatement |
| Mechanistic specificity | Names entities, directions, and pathways rather than vague association |
| Testability | Proposes falsifiable predictions with plausible experiments |
| Significance | Relevance and impact to the disease or biology in scope |
| Clarity | Structure, readability, absence of repetition |
| Coverage (add-on) | Uses the breadth of the module's evidence rather than cherry-picking |
| Epistemic honesty (add-on) | Flags caveats and limitations (resolution failures, hub bias, degraded nodes) honestly |

## Results

### Overall outcome

Opus 4.8 won 11 modules, Sonnet 4 won 7, with no ties. Excluding the two modules whose Opus run was degraded (Tan, where the unfixed `degree=None` pathway-enrichment bug stripped the shared-neighbour themes; and Magenta, where direct-KG yield was low from run variance), Opus led 10 of 16.

### Per-parameter means (all 18 modules, 1 to 5)

| Parameter | Sonnet | Opus 4.8 | Delta (Opus - Sonnet) |
|---|---|---|---|
| Groundedness | 4.39 | 4.61 | +0.22 |
| Evidence-tier calibration | 4.56 | 4.61 | +0.06 |
| Plausibility | 4.61 | 4.67 | +0.06 |
| Novelty | 3.89 | 4.17 | +0.28 |
| Mechanistic specificity | 4.50 | 4.72 | +0.22 |
| Testability | 4.33 | 4.67 | +0.33 |
| Significance | 4.17 | 4.17 | 0.00 |
| Clarity | 4.56 | 4.56 | 0.00 |
| Coverage | 4.56 | 4.67 | +0.11 |
| Epistemic honesty | 4.56 | 4.61 | +0.06 |
| **Grand mean** | **4.41** | **4.54** | **+0.13** |

Opus never scored below Sonnet on any parameter mean. The two parameters where the models tied (significance, clarity) are the ones least sensitive to model capability: significance is largely fixed by which entities the module contains, and both models write cleanly in the imposed register. The parameters where Opus separated most (testability, novelty, mechanistic specificity, groundedness) are exactly the dimensions the hypothesis-generation literature identifies as the hardest to do well, which is consistent with a more capable model showing its advantage where the task is most demanding. Novelty is the lowest-scoring parameter for both models (3.89 and 4.17), indicating that genuine non-obviousness remains the common weak point regardless of model.

### Per-module winners

| Module | Winner | Confidence | | Module | Winner | Confidence |
|---|---|---|---|---|---|---|
| Brown | Sonnet | med | | Tan* | Sonnet | high |
| Blue | Opus | high | | Lightcyan | Opus | high |
| Turquoise | Opus | med | | Grey60 | Sonnet | med |
| Black | Opus | high | | Lightgreen | Opus | med |
| Green | Sonnet | med | | Lightyellow | Opus | low |
| Midnightblue | Opus | med | | Royalblue | Opus | med |
| Pink | Sonnet | high | | Darkred | Sonnet | med |
| Magenta* | Opus | med | | Darkgreen | Opus | low |
| Purple | Sonnet | low | | Grey | Opus | high |

Asterisk marks a module whose Opus run was independently degraded; Tan's degradation is the likely reason Sonnet won it with high confidence. Where the judge expressed high confidence it favoured Opus four times (Blue, Black, Lightcyan, Grey) versus twice for Sonnet (Pink, Tan), and the high-confidence Sonnet win on Tan is attributable to the degraded Opus run rather than to model quality.

## Interpretation

The judges' written rationales are consistent across modules and explain the Opus edge concretely: Opus reports more often applied evidence-tier tags inline and precisely, named specific genetic instruments and assays for validation (for example cis-mQTL and Mendelian-randomization designs), distinguished hub-flagged entities from module-specific biology, and built longer mechanistic chains. Sonnet's wins (Brown, Pink, Green, Purple, Grey60, Darkred) were typically credited to tighter narrative cohesion, a structured member-prioritization table, or a sharper single sub-network, and in two cases to a substantive entity-resolution correction that Sonnet got right. The overall picture is therefore not that one model is right and the other wrong; both are competent, and the margin is the kind of incremental quality difference that compounds across a large analysis rather than a step change on any single report.

## Limitations

Several limitations bound this evaluation. The judge is a single LLM rather than a panel or a domain expert, and although blind randomized scoring mitigates position and identity bias, residual self-preference (the Sonnet judge favouring Sonnet-style prose) cannot be fully excluded; it would bias against the observed Opus result, so the true Opus margin may be slightly larger, not smaller. The 1 to 5 scale compresses at the top (most scores fall in the 4 to 5 band), which understates differences; a finer scale or forced-ranking would separate the models more. Novelty was judged from each report's own attached literature rather than a fresh retrieval pass, so novelty scores are less rigorous than a fully retrieval-augmented check would be. Two Opus runs were degraded by factors unrelated to the model (the `degree=None` pathway-enrichment bug and run-to-run knowledge-graph variance), which penalizes Opus; the n=16 figures exclude them. Finally, the model-independent pipeline nodes (triage, entity resolution, direct-KG counts) are identical by construction, so this evaluation measures only the report-writing and hypothesis-synthesis stages, which is precisely the part the model controls.

## Conclusion

Under a conservative blind protocol, Opus 4.8 produces consistently higher-quality discovery reports than Sonnet 4, winning the majority of modules and leading on every rubric parameter, with the clearest gains in testability, novelty, mechanistic specificity, and groundedness. The absolute margin is modest because both models already perform well. The practical recommendation follows the use case: Opus 4.8 is the better default when report quality and hypothesis depth are the priority and the roughly 1.2 to 1.5 times cost is acceptable, while Sonnet 4 remains a strong, cheaper option whose output the judge still preferred on a meaningful minority of modules. A stronger future evaluation would use a multi-model judge panel, a cross-family judge to remove self-preference, a finer or rank-based scale, and a fully retrieval-augmented novelty check.

## Related

- [Cross-Module Discovery Synthesis](https://phwiki.phenoma.ai/doc/frailty-multi-omics-wgcna-modules-cross-module-discovery-synthesis-dGektQw73Q)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

## Sources (rubric provenance)

- The Evolving Role of LLMs in Scientific Innovation: Evaluator, Collaborator, Scientist (arXiv 2507.11810)
- A Survey on Hypothesis Generation for Scientific Discovery in the Era of LLMs (arXiv 2504.05496)
- YESciEval: Robust LLM-as-a-Judge for Scientific Question Answering (arXiv 2505.14279)
- Development, validation, and usage of metrics to evaluate the quality of clinical research hypotheses (BMC Medical Research Methodology, 2025)
- AI Idea Bench 2025 / LiveIdeaBench (arXiv 2504.14191)
- Current and future state of evaluation of LLMs for medical summarization tasks (npj Health Systems, 2024)
