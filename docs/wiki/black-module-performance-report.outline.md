# Black Module Run: Pipeline Performance Report (109-analyte, dev, 2026-06-23)

This is the per-node performance report emitted automatically at the end of the full-module **Black** discovery run on dev (commit `813fc3f`). The Kraken performance-reporter (a terminal graph node) attributes wall-clock, tokens, estimated cost, output counts, and errors across the twelve pipeline nodes, and the Context management section reports how synthesis compresses the accumulated evidence into a bounded context.

## Related

- Discovery analysis from this run: [Black Module Run: Discovery Output](https://phwiki.phenoma.ai/doc/black-module-run-discovery-output-109-analyte-dev-2026-06-23-UTui957Bee)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---

## Pipeline Performance Report

- **Run:** `9d28ebe4768645a19d287c56e554586b`
- **Mode:** pipeline · **biomapper_env:** dev
- **Timestamp:** 20260623T163002Z · **commit:** `813fc3fb835aed414fc314f0942524a48ca6fbb9`
- **Wall-clock:** 998.076s · **tokens:** 22 in / 32693 out (21466 cache-r, 286411 cache-w) · **est. cost:** $1.5709
- **Top bottleneck:** pathway_enrichment · **Top cost:** cold_start

> Wall-clock is estimated as the sum of per-node durations, because no measured elapsed time was available; for concurrent branches this sum overstates true elapsed time. The percentage column reports each node's share of that summed total.

### Per-node

| Node | Status | Duration (s) | % | In | Out | Cache r/w | Model | MCP | Est. $ | Findings |
|---|---|---|---|---|---|---|---|---|---|---|
| pathway_enrichment | ran | 306.834 | 30.7% | 3 | 17866 | 0/58599 | anthropic/claude-sonnet-4-20250514 | 0 | $0.4877 | 133 shared neighbors (114 non-hub), 8 biological themes |
| cold_start | ran | 153.582 | 15.4% | 14 | 5916 | 21466/139408 | anthropic/claude-sonnet-4-20250514 | 0 | $0.6180 | 24 analogues, 17 inferred associations, 22 findings |
| synthesis | ran | 141.321 | 14.2% | 3 | 6147 | 0/60830 | anthropic/claude-sonnet-4-20250514 | 0 | $0.3203 | 40 hypotheses generated, 27008 char report |
| entity_resolution | ran | 105.236 | 10.5% | 0 | 0 | 0/0 | – | 0 | est. n/a | 108/108 entities resolved (35 fuzzy, 70 biomapper, 3 exact) |
| literature_grounding | ran | 79.49 | 8.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 33 papers, 29 from Exa across 15/40 hypotheses |
| triage | ran | 77.562 | 7.8% | 0 | 0 | 0/0 | – | 0 | est. n/a | 19 well-characterized, 27 moderate, 46 sparse, 16 cold-start |
| direct_kg | ran | 73.283 | 7.3% | 0 | 0 | 0/0 | – | 0 | est. n/a | 387 disease associations, 198 pathways, 1042 findings, 1 hub flags |
| integration | ran | 60.276 | 6.0% | 2 | 2764 | 0/27574 | anthropic/claude-sonnet-4-20250514 | 0 | $0.1449 | 1 bridges, 13 gap entities |
| hypothesis_extraction | ran | 0.49 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| intake | ran | 0.003 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 108 entities extracted, query_type=discovery |
| bridge_grounding | ran | 0.0 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| temporal | skipped | – | – | – | – | – | – | – | – | – |

### Context management

> Synthesis compresses the accumulated evidence into a bounded context; elision at module scale is expected, and it is the compression that keeps the assembled context within the model's token window.

- **Context:** 105998 chars (~30285 est. tokens) · 30.3% of the char budget (350000) · 15.1% of the 200K-token window
- **Mode:** module-aware aggregation (104 distinct entities; threshold 5)
- **Literature grounding:** 15 of 40 hypotheses carry attached literature

| Section | Shown | Total | Elided |
|---|---|---|---|
| Findings | 131 | 1064 | 933 |
| Diseases (recurrence) | 30 | 65 | 35 |
| Pathways (recurrence) | 13 | 13 | 0 |
| Member table | 50 | 104 | 54 |

### Errors

None.