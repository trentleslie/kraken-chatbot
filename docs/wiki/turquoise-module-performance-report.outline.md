# Turquoise Module Run: Pipeline Performance Report (107-analyte, dev, 2026-06-23)

This is the per-node performance report emitted automatically at the end of the full-module **Turquoise** discovery run on dev (commit `813fc3f`). The Kraken performance-reporter (a terminal graph node) attributes wall-clock, tokens, estimated cost, output counts, and errors across the twelve pipeline nodes, and the Context management section reports how synthesis compresses the accumulated evidence into a bounded context.

## Related

- Discovery analysis from this run: [Turquoise Module Run: Discovery Output](https://phwiki.phenoma.ai/doc/turquoise-module-run-discovery-output-107-analyte-dev-2026-06-23-tuVcWPUCy6)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---

## Pipeline Performance Report

- **Run:** `b40c2a77df984a5eb43b1b4d9535ff37`
- **Mode:** pipeline · **biomapper_env:** dev
- **Timestamp:** 20260623T165532Z · **commit:** `813fc3fb835aed414fc314f0942524a48ca6fbb9`
- **Wall-clock:** 1555.356s · **tokens:** 17 in / 11683 out (83598 cache-r, 127478 cache-w) · **est. cost:** $0.6784
- **Top bottleneck:** pathway_enrichment · **Top cost:** cold_start

> Wall-clock is estimated as the sum of per-node durations, because no measured elapsed time was available; for concurrent branches this sum overstates true elapsed time. The percentage column reports each node's share of that summed total.

### Per-node

| Node | Status | Duration (s) | % | In | Out | Cache r/w | Model | MCP | Est. $ | Findings |
|---|---|---|---|---|---|---|---|---|---|---|
| pathway_enrichment | ran | 751.692 | 48.3% | 0 | 0 | 0/0 | – | 0 | est. n/a | 0 shared neighbors (0 non-hub), 0 biological themes |
| cold_start | ran | 167.445 | 10.8% | 12 | 2047 | 64398/71841 | anthropic/claude-sonnet-4-20250514 | 0 | $0.3195 | 18 analogues, 0 inferred associations, 6 findings |
| synthesis | ran | 150.767 | 9.7% | 3 | 7252 | 19200/33483 | anthropic/claude-sonnet-4-20250514 | 0 | $0.2401 | 28 hypotheses generated, 25648 char report |
| direct_kg | ran | 115.749 | 7.4% | 0 | 0 | 0/0 | – | 0 | est. n/a | 1138 disease associations, 1022 pathways, 3415 findings, 10 hub flags |
| entity_resolution | ran | 97.383 | 6.3% | 0 | 0 | 0/0 | – | 0 | est. n/a | 106/106 entities resolved (99 biomapper, 7 fuzzy) |
| triage | ran | 95.868 | 6.2% | 0 | 0 | 0/0 | – | 0 | est. n/a | 77 well-characterized, 14 moderate, 14 sparse, 1 cold-start |
| literature_grounding | ran | 60.014 | 3.9% | 0 | 0 | 0/0 | – | 0 | est. n/a | 17 papers, 17 from Exa across 15/28 hypotheses |
| integration | ran | 58.435 | 3.8% | 2 | 2384 | 0/22154 | anthropic/claude-sonnet-4-20250514 | 0 | $0.1188 | 22 bridges, 11 gap entities |
| bridge_grounding | ran | 58.001 | 3.7% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| intake | ran | 0.002 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 106 entities extracted, query_type=discovery |
| hypothesis_extraction | ran | 0.0 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| temporal | skipped | – | – | – | – | – | – | – | – | – |

### Context management

> Synthesis compresses the accumulated evidence into a bounded context; elision at module scale is expected, and it is the compression that keeps the assembled context within the model's token window.

- **Context:** 77386 chars (~22110 est. tokens) · 22.1% of the char budget (350000) · 11.1% of the 200K-token window
- **Mode:** module-aware aggregation (106 distinct entities; threshold 5)
- **Literature grounding:** 15 of 28 hypotheses carry attached literature

| Section | Shown | Total | Elided |
|---|---|---|---|
| Findings | 115 | 3421 | 3306 |
| Diseases (recurrence) | 30 | 156 | 126 |
| Pathways (recurrence) | 30 | 148 | 118 |
| Member table | 50 | 106 | 56 |

### Errors

- SDK query timed out after 480s