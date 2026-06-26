# Grey Module Run: Pipeline Performance Report (46-analyte, dev, 2026-06-23)

This is the per-node performance report emitted automatically at the end of the full-module **Grey** discovery run on dev (commit `813fc3f`). The Kraken performance-reporter (a terminal graph node) attributes wall-clock, tokens, estimated cost, output counts, and errors across the twelve pipeline nodes, and the Context management section reports how synthesis compresses the accumulated evidence into a bounded context.

## Related

- Discovery analysis from this run: [Grey Module Run: Discovery Output](https://phwiki.phenoma.ai/doc/grey-module-run-discovery-output-46-analyte-dev-2026-06-23-96ZUVsxoW5)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---

## Pipeline Performance Report

- **Run:** `93f075216d9f43519aa5ed006d19590a`
- **Mode:** pipeline · **biomapper_env:** dev
- **Timestamp:** 20260623T181650Z · **commit:** `813fc3fb835aed414fc314f0942524a48ca6fbb9`
- **Wall-clock:** 964.783s · **tokens:** 19 in / 24595 out (134024 cache-r, 149816 cache-w) · **est. cost:** $0.9710
- **Top bottleneck:** pathway_enrichment · **Top cost:** pathway_enrichment

> Wall-clock is estimated as the sum of per-node durations, because no measured elapsed time was available; for concurrent branches this sum overstates true elapsed time. The percentage column reports each node's share of that summed total.

### Per-node

| Node | Status | Duration (s) | % | In | Out | Cache r/w | Model | MCP | Est. $ | Findings |
|---|---|---|---|---|---|---|---|---|---|---|
| pathway_enrichment | ran | 224.278 | 23.2% | 3 | 7409 | 0/54255 | anthropic/claude-sonnet-4-20250514 | 0 | $0.3146 | 50 shared neighbors (39 non-hub), 8 biological themes |
| cold_start | ran | 169.629 | 17.6% | 10 | 7662 | 85864/29282 | anthropic/claude-sonnet-4-20250514 | 0 | $0.2505 | 15 analogues, 25 inferred associations, 28 findings |
| synthesis | ran | 144.301 | 15.0% | 3 | 6148 | 21466/38864 | anthropic/claude-sonnet-4-20250514 | 0 | $0.2444 | 73 hypotheses generated, 23518 char report |
| direct_kg | ran | 112.519 | 11.7% | 0 | 0 | 0/0 | – | 0 | est. n/a | 859 disease associations, 851 pathways, 2674 findings, 7 hub flags |
| integration | ran | 98.654 | 10.2% | 3 | 3376 | 26694/27415 | anthropic/claude-sonnet-4-20250514 | 0 | $0.1615 | 20 bridges, 14 gap entities |
| literature_grounding | ran | 63.788 | 6.6% | 0 | 0 | 0/0 | – | 0 | est. n/a | 23 papers, 23 from Exa across 15/73 hypotheses |
| bridge_grounding | ran | 61.655 | 6.4% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| triage | ran | 46.464 | 4.8% | 0 | 0 | 0/0 | – | 0 | est. n/a | 30 well-characterized, 6 moderate, 9 sparse, 1 cold-start |
| entity_resolution | ran | 43.492 | 4.5% | 0 | 0 | 0/0 | – | 0 | est. n/a | 46/46 entities resolved (41 biomapper, 5 fuzzy) |
| intake | ran | 0.002 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 46 entities extracted, query_type=discovery |
| hypothesis_extraction | ran | 0.0 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| temporal | skipped | – | – | – | – | – | – | – | – | – |

### Context management

> Synthesis compresses the accumulated evidence into a bounded context; elision at module scale is expected, and it is the compression that keeps the assembled context within the model's token window.

- **Context:** 102940 chars (~29411 est. tokens) · 29.4% of the char budget (350000) · 14.7% of the 200K-token window
- **Mode:** module-aware aggregation (46 distinct entities; threshold 5)
- **Literature grounding:** 15 of 73 hypotheses carry attached literature

| Section | Shown | Total | Elided |
|---|---|---|---|
| Findings | 137 | 2702 | 2565 |
| Diseases (recurrence) | 30 | 130 | 100 |
| Pathways (recurrence) | 30 | 108 | 78 |
| Member table | 46 | 46 | 0 |

### Errors

- Exception analyzing NCBIGene:614927: 1 validation error for AnalogueEntity similarity   Input should be less than or equal to 1 [type=less_than_equal, input_value=1.000130295753479, input_type=float]     For further information visit https://errors.pydantic.dev/2.12/v/less_than_equal