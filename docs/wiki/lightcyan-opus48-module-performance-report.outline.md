# Lightcyan Module Run on Opus 4.8: Pipeline Performance Report (27-analyte, dev, 2026-06-24)

This is the per-node performance report emitted automatically at the end of the full-module **Lightcyan** discovery run on dev (commit `a7d7fd6`). All SDK-backed nodes ran on the Opus 4.8 model; this report is published for direct model comparison against the Sonnet baseline. The Kraken performance-reporter (a terminal graph node) attributes wall-clock, tokens, estimated cost, output counts, and errors across the twelve pipeline nodes, and the Context management section reports how synthesis compresses the accumulated evidence into a bounded context.

## Related

- Discovery analysis from this run: [Lightcyan Module Run on Opus 4.8: Discovery Output](https://phwiki.phenoma.ai/doc/lightcyan-module-run-on-opus-48-discovery-output-27-analyte-dev-2026-06-24-iPUy9JfzZv)
- Model comparison baseline (Sonnet): [Lightcyan Module Run: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/lightcyan-module-run-pipeline-performance-report-27-analyte-dev-2026-06-23-nakWIM0Nkc)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---

## Pipeline Performance Report

- **Run:** `8f7810dd51694eeb80ae8e544ef75473`
- **Mode:** pipeline · **biomapper_env:** dev
- **Timestamp:** 20260624T174159Z · **commit:** `a7d7fd63deffdbbb25f8b30e8dee40cefdf6c3cd`
- **Wall-clock:** 771.328s · **tokens:** 23 in / 27458 out (156048 cache-r, 161753 cache-w) · **est. cost:** $1.7755
- **Top bottleneck:** integration · **Top cost:** cold_start

> Wall-clock is estimated as the sum of per-node durations, because no measured elapsed time was available; for concurrent branches this sum overstates true elapsed time. The percentage column reports each node's share of that summed total.

### Per-node

| Node | Status | Duration (s) | % | In | Out | Cache r/w | Model | MCP | Est. $ | Findings |
|---|---|---|---|---|---|---|---|---|---|---|
| integration | ran | 180.969 | 23.5% | 3 | 3962 | 27252/32362 | claude-opus-4-8 | 0 | $0.3150 | 10 bridges, 15 gap entities |
| synthesis | ran | 152.5 | 19.8% | 3 | 6612 | 21466/39605 | claude-opus-4-8 | 0 | $0.4236 | 59 hypotheses generated, 26272 char report |
| pathway_enrichment | ran | 127.893 | 16.6% | 3 | 9735 | 0/35937 | claude-opus-4-8 | 0 | $0.4680 | 28 shared neighbors (24 non-hub), 6 biological themes |
| cold_start | ran | 121.055 | 15.7% | 14 | 7149 | 107330/53849 | claude-opus-4-8 | 0 | $0.5690 | 21 analogues, 23 inferred associations, 26 findings |
| direct_kg | ran | 61.441 | 8.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 182 disease associations, 159 pathways, 889 findings |
| literature_grounding | ran | 60.658 | 7.9% | 0 | 0 | 0/0 | – | 0 | est. n/a | 32 papers, 31 from Exa across 14/59 hypotheses |
| entity_resolution | ran | 23.561 | 3.1% | 0 | 0 | 0/0 | – | 0 | est. n/a | 27/27 entities resolved (11 biomapper, 16 fuzzy) |
| triage | ran | 23.255 | 3.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 5 well-characterized, 12 moderate, 8 sparse, 2 cold-start |
| bridge_grounding | ran | 13.38 | 1.7% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| hypothesis_extraction | ran | 6.614 | 0.9% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| intake | ran | 0.002 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 27 entities extracted, query_type=discovery |
| temporal | skipped | – | – | – | – | – | – | – | – | – |

### Context management

> Synthesis compresses the accumulated evidence into a bounded context; elision at module scale is expected, and it is the compression that keeps the assembled context within the model's token window.

- **Context:** 116423 chars (~33264 est. tokens) · 33.3% of the char budget (350000) · 16.6% of the 200K-token window
- **Mode:** module-aware aggregation (24 distinct entities; threshold 5)
- **Literature grounding:** 14 of 59 hypotheses carry attached literature

| Section | Shown | Total | Elided |
|---|---|---|---|
| Findings | 143 | 915 | 772 |
| Diseases (recurrence) | 29 | 29 | 0 |
| Pathways (recurrence) | 10 | 10 | 0 |
| Member table | 24 | 24 | 0 |

### Errors

None.