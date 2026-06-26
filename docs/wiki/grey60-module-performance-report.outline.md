# Grey60 Module Run: Pipeline Performance Report (32-analyte, dev, 2026-06-23)

This is the per-node performance report emitted automatically at the end of the full-module **Grey60** discovery run on dev (commit `813fc3f`). The Kraken performance-reporter (a terminal graph node) attributes wall-clock, tokens, estimated cost, output counts, and errors across the twelve pipeline nodes, and the Context management section reports how synthesis compresses the accumulated evidence into a bounded context.

## Related

- Discovery analysis from this run: [Grey60 Module Run: Discovery Output](https://phwiki.phenoma.ai/doc/grey60-module-run-discovery-output-32-analyte-dev-2026-06-23-GPq51xU6sG)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---

## Pipeline Performance Report

- **Run:** `d25edac120ff4adaa36705308cb582d6`
- **Mode:** pipeline · **biomapper_env:** dev
- **Timestamp:** 20260623T184848Z · **commit:** `813fc3fb835aed414fc314f0942524a48ca6fbb9`
- **Wall-clock:** 538.046s · **tokens:** 24 in / 19381 out (128796 cache-r, 157144 cache-w) · **est. cost:** $0.9187
- **Top bottleneck:** synthesis · **Top cost:** cold_start

> Wall-clock is estimated as the sum of per-node durations, because no measured elapsed time was available; for concurrent branches this sum overstates true elapsed time. The percentage column reports each node's share of that summed total.

### Per-node

| Node | Status | Duration (s) | % | In | Out | Cache r/w | Model | MCP | Est. $ | Findings |
|---|---|---|---|---|---|---|---|---|---|---|
| synthesis | ran | 156.825 | 29.1% | 3 | 6621 | 0/49224 | anthropic/claude-sonnet-4-20250514 | 0 | $0.2839 | 51 hypotheses generated, 25842 char report |
| cold_start | ran | 111.587 | 20.7% | 16 | 7897 | 85864/97243 | anthropic/claude-sonnet-4-20250514 | 0 | $0.5089 | 24 analogues, 24 inferred associations, 27 findings |
| literature_grounding | ran | 69.041 | 12.8% | 0 | 0 | 0/0 | – | 0 | est. n/a | 19 papers, 19 from Exa across 14/51 hypotheses |
| integration | ran | 59.255 | 11.0% | 2 | 2691 | 21466/4825 | anthropic/claude-sonnet-4-20250514 | 0 | $0.0649 | 0 bridges, 10 gap entities |
| pathway_enrichment | ran | 57.917 | 10.8% | 3 | 2172 | 21466/5852 | anthropic/claude-sonnet-4-20250514 | 0 | $0.0610 | 0 shared neighbors (0 non-hub), 3 biological themes |
| entity_resolution | ran | 33.026 | 6.1% | 0 | 0 | 0/0 | – | 0 | est. n/a | 33/33 entities resolved (7 fuzzy, 25 biomapper, 1 exact) |
| triage | ran | 26.037 | 4.8% | 0 | 0 | 0/0 | – | 0 | est. n/a | 3 well-characterized, 3 moderate, 21 sparse, 6 cold-start |
| direct_kg | ran | 24.355 | 4.5% | 0 | 0 | 0/0 | – | 0 | est. n/a | 91 disease associations, 10 pathways, 265 findings |
| intake | ran | 0.002 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 33 entities extracted, query_type=discovery |
| hypothesis_extraction | ran | 0.0 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| bridge_grounding | ran | 0.0 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| temporal | skipped | – | – | – | – | – | – | – | – | – |

### Context management

> Synthesis compresses the accumulated evidence into a bounded context; elision at module scale is expected, and it is the compression that keeps the assembled context within the model's token window.

- **Context:** 77163 chars (~22047 est. tokens) · 22.0% of the char budget (350000) · 11.0% of the 200K-token window
- **Mode:** module-aware aggregation (33 distinct entities; threshold 5)
- **Literature grounding:** 14 of 51 hypotheses carry attached literature

| Section | Shown | Total | Elided |
|---|---|---|---|
| Findings | 134 | 292 | 158 |
| Diseases (recurrence) | 5 | 5 | 0 |
| Pathways (recurrence) | 0 | 0 | 0 |
| Member table | 33 | 33 | 0 |

### Errors

- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be a valid integer [type=int_type, input_value=None, input_type=NoneType]     For further information visit https://errors.pydantic.dev/2.12/v/int_type
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'