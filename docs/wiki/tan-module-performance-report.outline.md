# Tan Module Run: Pipeline Performance Report (43-analyte, dev, 2026-06-23)

This is the per-node performance report emitted automatically at the end of the full-module **Tan** discovery run on dev (commit `813fc3f`). The Kraken performance-reporter (a terminal graph node) attributes wall-clock, tokens, estimated cost, output counts, and errors across the twelve pipeline nodes, and the Context management section reports how synthesis compresses the accumulated evidence into a bounded context.

## Related

- Discovery analysis from this run: [Tan Module Run: Discovery Output](https://phwiki.phenoma.ai/doc/tan-module-run-discovery-output-43-analyte-dev-2026-06-23-xSd82cW2It)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---

## Pipeline Performance Report

- **Run:** `1805ba6ebcde4456bb98f0dad16540b4`
- **Mode:** pipeline · **biomapper_env:** dev
- **Timestamp:** 20260623T182910Z · **commit:** `813fc3fb835aed414fc314f0942524a48ca6fbb9`
- **Wall-clock:** 792.568s · **tokens:** 19 in / 24498 out (134361 cache-r, 132988 cache-w) · **est. cost:** $0.9065
- **Top bottleneck:** integration · **Top cost:** cold_start

> Wall-clock is estimated as the sum of per-node durations, because no measured elapsed time was available; for concurrent branches this sum overstates true elapsed time. The percentage column reports each node's share of that summed total.

### Per-node

| Node | Status | Duration (s) | % | In | Out | Cache r/w | Model | MCP | Est. $ | Findings |
|---|---|---|---|---|---|---|---|---|---|---|
| integration | ran | 157.232 | 19.8% | 3 | 4313 | 48497/6976 | anthropic/claude-sonnet-4-20250514 | 0 | $0.1054 | 10 bridges, 15 gap entities |
| synthesis | ran | 154.232 | 19.5% | 3 | 6699 | 21466/36972 | anthropic/claude-sonnet-4-20250514 | 0 | $0.2456 | 49 hypotheses generated, 25901 char report |
| pathway_enrichment | ran | 152.444 | 19.2% | 3 | 7038 | 21466/16731 | anthropic/claude-sonnet-4-20250514 | 0 | $0.1748 | 33 shared neighbors (28 non-hub), 6 biological themes |
| cold_start | ran | 94.834 | 12.0% | 10 | 6448 | 42932/72309 | anthropic/claude-sonnet-4-20250514 | 0 | $0.3808 | 15 analogues, 19 inferred associations, 20 findings |
| literature_grounding | ran | 72.021 | 9.1% | 0 | 0 | 0/0 | – | 0 | est. n/a | 29 papers, 29 from Exa across 15/49 hypotheses |
| direct_kg | ran | 62.71 | 7.9% | 0 | 0 | 0/0 | – | 0 | est. n/a | 422 disease associations, 315 pathways, 1307 findings |
| entity_resolution | ran | 35.876 | 4.5% | 0 | 0 | 0/0 | – | 0 | est. n/a | 39/39 entities resolved (38 biomapper, 1 fuzzy) |
| triage | ran | 34.678 | 4.4% | 0 | 0 | 0/0 | – | 0 | est. n/a | 13 well-characterized, 5 moderate, 21 sparse, 0 cold-start |
| bridge_grounding | ran | 28.539 | 3.6% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| intake | ran | 0.002 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 39 entities extracted, query_type=discovery |
| hypothesis_extraction | ran | 0.0 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| temporal | skipped | – | – | – | – | – | – | – | – | – |

### Context management

> Synthesis compresses the accumulated evidence into a bounded context; elision at module scale is expected, and it is the compression that keeps the assembled context within the model's token window.

- **Context:** 101091 chars (~28883 est. tokens) · 28.9% of the char budget (350000) · 14.4% of the 200K-token window
- **Mode:** module-aware aggregation (37 distinct entities; threshold 5)
- **Literature grounding:** 15 of 49 hypotheses carry attached literature

| Section | Shown | Total | Elided |
|---|---|---|---|
| Findings | 129 | 1327 | 1198 |
| Diseases (recurrence) | 30 | 52 | 22 |
| Pathways (recurrence) | 30 | 32 | 2 |
| Member table | 37 | 37 | 0 |

### Errors

None.