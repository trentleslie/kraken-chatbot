# Blue Module Run: Pipeline Performance Report (149-analyte, dev, 2026-06-23)

This is the per-node performance report emitted automatically at the end of the full-module **Blue** discovery run on dev (commit `813fc3f`). The Kraken performance-reporter (a terminal graph node) attributes wall-clock, tokens, estimated cost, output counts, and errors across the twelve pipeline nodes, and the Context management section reports how synthesis compresses the accumulated evidence into a bounded context.

## Related

- Discovery analysis from this run: [Blue Module Run: Discovery Output](https://phwiki.phenoma.ai/doc/blue-module-run-discovery-output-149-analyte-dev-2026-06-23-9dAc5UFomv)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---

## Pipeline Performance Report

- **Run:** `fa16ada695124f7e852bc0f755330ce8`
- **Mode:** pipeline · **biomapper_env:** dev
- **Timestamp:** 20260623T161231Z · **commit:** `813fc3fb835aed414fc314f0942524a48ca6fbb9`
- **Wall-clock:** 1446.328s · **tokens:** 22 in / 34681 out (0 cache-r, 345974 cache-w) · **est. cost:** $1.8177
- **Top bottleneck:** pathway_enrichment · **Top cost:** cold_start

> Wall-clock is estimated as the sum of per-node durations, because no measured elapsed time was available; for concurrent branches this sum overstates true elapsed time. The percentage column reports each node's share of that summed total.

### Per-node

| Node | Status | Duration (s) | % | In | Out | Cache r/w | Model | MCP | Est. $ | Findings |
|---|---|---|---|---|---|---|---|---|---|---|
| pathway_enrichment | ran | 570.208 | 39.4% | 3 | 19727 | 0/86626 | anthropic/claude-sonnet-4-20250514 | 0 | $0.6208 | 139 shared neighbors (125 non-hub), 13 biological themes |
| synthesis | ran | 182.078 | 12.6% | 3 | 8191 | 0/74096 | anthropic/claude-sonnet-4-20250514 | 0 | $0.4007 | 44 hypotheses generated, 28835 char report |
| cold_start | ran | 147.494 | 10.2% | 14 | 4574 | 0/159206 | anthropic/claude-sonnet-4-20250514 | 0 | $0.6657 | 21 analogues, 10 inferred associations, 14 findings |
| entity_resolution | ran | 132.748 | 9.2% | 0 | 0 | 0/0 | – | 0 | est. n/a | 146/146 entities resolved (135 biomapper, 2 exact, 9 fuzzy) |
| triage | ran | 124.938 | 8.6% | 0 | 0 | 0/0 | – | 0 | est. n/a | 95 well-characterized, 24 moderate, 24 sparse, 3 cold-start |
| direct_kg | ran | 104.517 | 7.2% | 0 | 0 | 0/0 | – | 0 | est. n/a | 1266 disease associations, 1167 pathways, 3806 findings, 13 hub flags |
| integration | ran | 67.722 | 4.7% | 2 | 2189 | 0/26046 | anthropic/claude-sonnet-4-20250514 | 0 | $0.1305 | 20 bridges, 13 gap entities |
| literature_grounding | ran | 63.644 | 4.4% | 0 | 0 | 0/0 | – | 0 | est. n/a | 38 papers, 26 from OpenAlex, 12 from Exa across 15/44 hypotheses |
| bridge_grounding | ran | 47.33 | 3.3% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| hypothesis_extraction | ran | 5.646 | 0.4% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| intake | ran | 0.003 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 146 entities extracted, query_type=discovery |
| temporal | skipped | – | – | – | – | – | – | – | – | – |

### Context management

> Synthesis compresses the accumulated evidence into a bounded context; elision at module scale is expected, and it is the compression that keeps the assembled context within the model's token window.

- **Context:** 143053 chars (~40872 est. tokens) · 40.9% of the char budget (350000) · 20.4% of the 200K-token window
- **Mode:** module-aware aggregation (146 distinct entities; threshold 5)
- **Literature grounding:** 15 of 44 hypotheses carry attached literature

| Section | Shown | Total | Elided |
|---|---|---|---|
| Findings | 126 | 3820 | 3694 |
| Diseases (recurrence) | 30 | 136 | 106 |
| Pathways (recurrence) | 30 | 168 | 138 |
| Member table | 50 | 146 | 96 |

### Errors

None.