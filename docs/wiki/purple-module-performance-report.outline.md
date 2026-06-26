# Purple Module Run: Pipeline Performance Report (51-analyte, dev, 2026-06-23)

This is the per-node performance report emitted automatically at the end of the full-module **Purple** discovery run on dev (commit `813fc3f`). The Kraken performance-reporter (a terminal graph node) attributes wall-clock, tokens, estimated cost, output counts, and errors across the twelve pipeline nodes, and the Context management section reports how synthesis compresses the accumulated evidence into a bounded context.

## Related

- Discovery analysis from this run: [Purple Module Run: Discovery Output](https://phwiki.phenoma.ai/doc/purple-module-run-discovery-output-51-analyte-dev-2026-06-23-Sahgv1nyKF)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---

## Pipeline Performance Report

- **Run:** `51130f9e093a410fb49a846eaa418ae4`
- **Mode:** pipeline · **biomapper_env:** dev
- **Timestamp:** 20260623T180226Z · **commit:** `813fc3fb835aed414fc314f0942524a48ca6fbb9`
- **Wall-clock:** 783.07s · **tokens:** 23 in / 22521 out (150262 cache-r, 135084 cache-w) · **est. cost:** $0.8895
- **Top bottleneck:** pathway_enrichment · **Top cost:** cold_start

> Wall-clock is estimated as the sum of per-node durations, because no measured elapsed time was available; for concurrent branches this sum overstates true elapsed time. The percentage column reports each node's share of that summed total.

### Per-node

| Node | Status | Duration (s) | % | In | Out | Cache r/w | Model | MCP | Est. $ | Findings |
|---|---|---|---|---|---|---|---|---|---|---|
| pathway_enrichment | ran | 179.528 | 22.9% | 3 | 9908 | 21466/26955 | anthropic/claude-sonnet-4-20250514 | 0 | $0.2562 | 0 shared neighbors (0 non-hub), 3 biological themes |
| cold_start | ran | 162.716 | 20.8% | 12 | 4833 | 85864/51486 | anthropic/claude-sonnet-4-20250514 | 0 | $0.2914 | 18 analogues, 12 inferred associations, 18 findings |
| synthesis | ran | 126.308 | 16.1% | 3 | 5704 | 0/49908 | anthropic/claude-sonnet-4-20250514 | 0 | $0.2727 | 40 hypotheses generated, 21653 char report |
| direct_kg | ran | 83.046 | 10.6% | 0 | 0 | 0/0 | – | 0 | est. n/a | 148 disease associations, 106 pathways, 1097 findings, 2 hub flags |
| literature_grounding | ran | 64.179 | 8.2% | 0 | 0 | 0/0 | – | 0 | est. n/a | 20 papers, 20 from Exa across 13/40 hypotheses |
| integration | ran | 47.11 | 6.0% | 2 | 2052 | 21466/4917 | anthropic/claude-sonnet-4-20250514 | 0 | $0.0557 | 10 bridges, 12 gap entities |
| entity_resolution | ran | 43.085 | 5.5% | 3 | 24 | 21466/1818 | anthropic/claude-sonnet-4-20250514 | 0 | $0.0136 | 47/47 entities resolved (4 biomapper, 42 fuzzy, 1 exact) |
| bridge_grounding | ran | 42.941 | 5.5% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| triage | ran | 34.155 | 4.4% | 0 | 0 | 0/0 | – | 0 | est. n/a | 4 well-characterized, 30 moderate, 11 sparse, 2 cold-start |
| intake | ran | 0.002 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 47 entities extracted, query_type=discovery |
| hypothesis_extraction | ran | 0.0 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| temporal | skipped | – | – | – | – | – | – | – | – | – |

### Context management

> Synthesis compresses the accumulated evidence into a bounded context; elision at module scale is expected, and it is the compression that keeps the assembled context within the model's token window.

- **Context:** 71461 chars (~20417 est. tokens) · 20.4% of the char budget (350000) · 10.2% of the 200K-token window
- **Mode:** module-aware aggregation (35 distinct entities; threshold 5)
- **Literature grounding:** 13 of 40 hypotheses carry attached literature

| Section | Shown | Total | Elided |
|---|---|---|---|
| Findings | 125 | 1115 | 990 |
| Diseases (recurrence) | 27 | 27 | 0 |
| Pathways (recurrence) | 4 | 4 | 0 |
| Member table | 35 | 35 | 0 |

### Errors

- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'