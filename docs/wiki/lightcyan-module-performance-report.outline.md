# Lightcyan Module Run: Pipeline Performance Report (27-analyte, dev, 2026-06-23)

This is the per-node performance report emitted automatically at the end of the full-module **Lightcyan** discovery run on dev (commit `813fc3f`). The Kraken performance-reporter (a terminal graph node) attributes wall-clock, tokens, estimated cost, output counts, and errors across the twelve pipeline nodes, and the Context management section reports how synthesis compresses the accumulated evidence into a bounded context.

## Related

- Discovery analysis from this run: [Lightcyan Module Run: Discovery Output](https://phwiki.phenoma.ai/doc/lightcyan-module-run-discovery-output-27-analyte-dev-2026-06-23-Uwabc4tJ77)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---

## Pipeline Performance Report

- **Run:** `f9d2b2cb65c64b23a4a4c19e2d8313bb`
- **Mode:** pipeline · **biomapper_env:** dev
- **Timestamp:** 20260623T184001Z · **commit:** `813fc3fb835aed414fc314f0942524a48ca6fbb9`
- **Wall-clock:** 687.262s · **tokens:** 23 in / 22261 out (112967 cache-r, 203527 cache-w) · **est. cost:** $1.1311
- **Top bottleneck:** integration · **Top cost:** cold_start

> Wall-clock is estimated as the sum of per-node durations, because no measured elapsed time was available; for concurrent branches this sum overstates true elapsed time. The percentage column reports each node's share of that summed total.

### Per-node

| Node | Status | Duration (s) | % | In | Out | Cache r/w | Model | MCP | Est. $ | Findings |
|---|---|---|---|---|---|---|---|---|---|---|
| integration | ran | 169.956 | 24.7% | 3 | 3683 | 27103/32739 | anthropic/claude-sonnet-4-20250514 | 0 | $0.1862 | 10 bridges, 12 gap entities |
| synthesis | ran | 133.844 | 19.5% | 3 | 6117 | 0/60253 | anthropic/claude-sonnet-4-20250514 | 0 | $0.3177 | 61 hypotheses generated, 24344 char report |
| cold_start | ran | 100.629 | 14.6% | 14 | 7409 | 64398/96064 | anthropic/claude-sonnet-4-20250514 | 0 | $0.4907 | 21 analogues, 24 inferred associations, 27 findings |
| pathway_enrichment | ran | 98.77 | 14.4% | 3 | 5052 | 21466/14471 | anthropic/claude-sonnet-4-20250514 | 0 | $0.1365 | 0 shared neighbors (0 non-hub), 6 biological themes |
| literature_grounding | ran | 66.131 | 9.6% | 0 | 0 | 0/0 | – | 0 | est. n/a | 35 papers, 31 from Exa across 14/61 hypotheses |
| direct_kg | ran | 48.67 | 7.1% | 0 | 0 | 0/0 | – | 0 | est. n/a | 182 disease associations, 155 pathways, 756 findings |
| entity_resolution | ran | 24.954 | 3.6% | 0 | 0 | 0/0 | – | 0 | est. n/a | 27/27 entities resolved (11 biomapper, 16 fuzzy) |
| triage | ran | 22.552 | 3.3% | 0 | 0 | 0/0 | – | 0 | est. n/a | 5 well-characterized, 12 moderate, 8 sparse, 2 cold-start |
| bridge_grounding | ran | 14.907 | 2.2% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| hypothesis_extraction | ran | 6.849 | 1.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| intake | ran | 0.002 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 27 entities extracted, query_type=discovery |
| temporal | skipped | – | – | – | – | – | – | – | – | – |

### Context management

> Synthesis compresses the accumulated evidence into a bounded context; elision at module scale is expected, and it is the compression that keeps the assembled context within the model's token window.

- **Context:** 115240 chars (~32926 est. tokens) · 32.9% of the char budget (350000) · 16.5% of the 200K-token window
- **Mode:** module-aware aggregation (24 distinct entities; threshold 5)
- **Literature grounding:** 14 of 61 hypotheses carry attached literature

| Section | Shown | Total | Elided |
|---|---|---|---|
| Findings | 143 | 783 | 640 |
| Diseases (recurrence) | 29 | 29 | 0 |
| Pathways (recurrence) | 10 | 10 | 0 |
| Member table | 24 | 24 | 0 |

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
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be a valid integer [type=int_type, input_value=None, input_type=NoneType]     For further information visit https://errors.pydantic.dev/2.12/v/int_type
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be a valid integer [type=int_type, input_value=None, input_type=NoneType]     For further information visit https://errors.pydantic.dev/2.12/v/int_type
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'