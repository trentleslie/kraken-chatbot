# Grey Module Run on Opus 4.8: Pipeline Performance Report (46-analyte, dev, 2026-06-24)

This is the per-node performance report emitted automatically at the end of the full-module **Grey** discovery run on dev (commit `a7d7fd6`). All SDK-backed nodes ran on the Opus 4.8 model; this report is published for direct model comparison against the Sonnet baseline. The Kraken performance-reporter (a terminal graph node) attributes wall-clock, tokens, estimated cost, output counts, and errors across the twelve pipeline nodes, and the Context management section reports how synthesis compresses the accumulated evidence into a bounded context.

## Related

- Discovery analysis from this run: [Grey Module Run on Opus 4.8: Discovery Output](https://phwiki.phenoma.ai/doc/grey-module-run-on-opus-48-discovery-output-46-analyte-dev-2026-06-24-RiItPWqChE)
- Model comparison baseline (Sonnet): [Grey Module Run: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/grey-module-run-pipeline-performance-report-46-analyte-dev-2026-06-23-WhfflHp2gV)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---

## Pipeline Performance Report

- **Run:** `5492901f1de049b5b704efb9b2868524`
- **Mode:** pipeline · **biomapper_env:** dev
- **Timestamp:** 20260624T171940Z · **commit:** `a7d7fd63deffdbbb25f8b30e8dee40cefdf6c3cd`
- **Wall-clock:** 1161.98s · **tokens:** 19 in / 31584 out (48011 cache-r, 249896 cache-w) · **est. cost:** $2.3756
- **Top bottleneck:** integration · **Top cost:** cold_start

> Wall-clock is estimated as the sum of per-node durations, because no measured elapsed time was available; for concurrent branches this sum overstates true elapsed time. The percentage column reports each node's share of that summed total.

### Per-node

| Node | Status | Duration (s) | % | In | Out | Cache r/w | Model | MCP | Est. $ | Findings |
|---|---|---|---|---|---|---|---|---|---|---|
| integration | ran | 252.465 | 21.7% | 3 | 4738 | 26545/33714 | claude-opus-4-8 | 0 | $0.3424 | 20 bridges, 16 gap entities |
| pathway_enrichment | ran | 215.66 | 18.6% | 3 | 8208 | 0/54255 | claude-opus-4-8 | 0 | $0.5443 | 53 shared neighbors (44 non-hub), 8 biological themes |
| cold_start | ran | 171.997 | 14.8% | 10 | 11260 | 0/115147 | claude-opus-4-8 | 0 | $1.0012 | 15 analogues, 39 inferred associations, 40 findings |
| synthesis | ran | 171.738 | 14.8% | 3 | 7378 | 21466/46780 | claude-opus-4-8 | 0 | $0.4876 | 99 hypotheses generated, 28912 char report |
| direct_kg | ran | 147.15 | 12.7% | 0 | 0 | 0/0 | – | 0 | est. n/a | 1118 disease associations, 1060 pathways, 3421 findings, 7 hub flags |
| literature_grounding | ran | 67.148 | 5.8% | 0 | 0 | 0/0 | – | 0 | est. n/a | 23 papers, 23 from Exa across 15/99 hypotheses |
| bridge_grounding | ran | 57.663 | 5.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| entity_resolution | ran | 39.802 | 3.4% | 0 | 0 | 0/0 | – | 0 | est. n/a | 46/46 entities resolved (41 biomapper, 5 fuzzy) |
| triage | ran | 38.357 | 3.3% | 0 | 0 | 0/0 | – | 0 | est. n/a | 30 well-characterized, 6 moderate, 9 sparse, 1 cold-start |
| intake | ran | 0.001 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 46 entities extracted, query_type=discovery |
| hypothesis_extraction | ran | 0.0 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| temporal | skipped | – | – | – | – | – | – | – | – | – |

### Context management

> Synthesis compresses the accumulated evidence into a bounded context; elision at module scale is expected, and it is the compression that keeps the assembled context within the model's token window.

- **Context:** 125143 chars (~35755 est. tokens) · 35.8% of the char budget (350000) · 17.9% of the 200K-token window
- **Mode:** module-aware aggregation (46 distinct entities; threshold 5)
- **Literature grounding:** 15 of 99 hypotheses carry attached literature

| Section | Shown | Total | Elided |
|---|---|---|---|
| Findings | 150 | 3461 | 3311 |
| Diseases (recurrence) | 30 | 157 | 127 |
| Pathways (recurrence) | 30 | 154 | 124 |
| Member table | 46 | 46 | 0 |

### Errors

- Exception analyzing NCBIGene:614927: 1 validation error for AnalogueEntity similarity   Input should be less than or equal to 1 [type=less_than_equal, input_value=1.000130295753479, input_type=float]     For further information visit https://errors.pydantic.dev/2.12/v/less_than_equal