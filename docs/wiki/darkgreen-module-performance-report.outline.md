# Darkgreen Module Run: Pipeline Performance Report (20-analyte, dev, 2026-06-23)

This is the per-node performance report emitted automatically at the end of the full-module **Darkgreen** discovery run on dev (commit `813fc3f`). The Kraken performance-reporter (a terminal graph node) attributes wall-clock, tokens, estimated cost, output counts, and errors across the twelve pipeline nodes, and the Context management section reports how synthesis compresses the accumulated evidence into a bounded context.

## Related

- Discovery analysis from this run: [Darkgreen Module Run: Discovery Output](https://phwiki.phenoma.ai/doc/darkgreen-module-run-discovery-output-20-analyte-dev-2026-06-23-zILJChS0TE)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---

## Pipeline Performance Report

- **Run:** `da22b171b0d54093ab5f8f73e3f15005`
- **Mode:** pipeline · **biomapper_env:** dev
- **Timestamp:** 20260623T194744Z · **commit:** `813fc3fb835aed414fc314f0942524a48ca6fbb9`
- **Wall-clock:** 721.778s · **tokens:** 21 in / 24733 out (91415 cache-r, 193458 cache-w) · **est. cost:** $1.1240
- **Top bottleneck:** integration · **Top cost:** cold_start

> Wall-clock is estimated as the sum of per-node durations, because no measured elapsed time was available; for concurrent branches this sum overstates true elapsed time. The percentage column reports each node's share of that summed total.

### Per-node

| Node | Status | Duration (s) | % | In | Out | Cache r/w | Model | MCP | Est. $ | Findings |
|---|---|---|---|---|---|---|---|---|---|---|
| integration | ran | 197.571 | 27.4% | 3 | 3405 | 27017/34022 | anthropic/claude-sonnet-4-20250514 | 0 | $0.1868 | 0 bridges, 13 gap entities |
| synthesis | ran | 143.813 | 19.9% | 3 | 5958 | 0/54770 | anthropic/claude-sonnet-4-20250514 | 0 | $0.2948 | 71 hypotheses generated, 23720 char report |
| cold_start | ran | 123.738 | 17.1% | 12 | 9663 | 42932/95573 | anthropic/claude-sonnet-4-20250514 | 0 | $0.5163 | 18 analogues, 35 inferred associations, 36 findings |
| pathway_enrichment | ran | 104.161 | 14.4% | 3 | 5707 | 21466/9093 | anthropic/claude-sonnet-4-20250514 | 0 | $0.1262 | 0 shared neighbors (0 non-hub), 7 biological themes |
| literature_grounding | ran | 64.699 | 9.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 29 papers, 24 from Exa across 13/71 hypotheses |
| direct_kg | ran | 45.863 | 6.4% | 0 | 0 | 0/0 | – | 0 | est. n/a | 161 disease associations, 38 pathways, 454 findings |
| triage | ran | 22.323 | 3.1% | 0 | 0 | 0/0 | – | 0 | est. n/a | 3 well-characterized, 8 moderate, 9 sparse, 1 cold-start |
| entity_resolution | ran | 19.607 | 2.7% | 0 | 0 | 0/0 | – | 0 | est. n/a | 21/21 entities resolved (3 fuzzy, 17 biomapper, 1 exact) |
| intake | ran | 0.001 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 21 entities extracted, query_type=discovery |
| hypothesis_extraction | ran | 0.0 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| bridge_grounding | ran | 0.0 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| temporal | skipped | – | – | – | – | – | – | – | – | – |

### Context management

> Synthesis compresses the accumulated evidence into a bounded context; elision at module scale is expected, and it is the compression that keeps the assembled context within the model's token window.

- **Context:** 95004 chars (~27144 est. tokens) · 27.1% of the char budget (350000) · 13.6% of the 200K-token window
- **Mode:** module-aware aggregation (21 distinct entities; threshold 5)
- **Literature grounding:** 13 of 71 hypotheses carry attached literature

| Section | Shown | Total | Elided |
|---|---|---|---|
| Findings | 142 | 490 | 348 |
| Diseases (recurrence) | 26 | 26 | 0 |
| Pathways (recurrence) | 8 | 8 | 0 |
| Member table | 21 | 21 | 0 |

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
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be a valid integer [type=int_type, input_value=None, input_type=NoneType]     For further information visit https://errors.pydantic.dev/2.12/v/int_type
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
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be a valid integer [type=int_type, input_value=None, input_type=NoneType]     For further information visit https://errors.pydantic.dev/2.12/v/int_type
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be a valid integer [type=int_type, input_value=None, input_type=NoneType]     For further information visit https://errors.pydantic.dev/2.12/v/int_type
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be a valid integer [type=int_type, input_value=None, input_type=NoneType]     For further information visit https://errors.pydantic.dev/2.12/v/int_type
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be a valid integer [type=int_type, input_value=None, input_type=NoneType]     For further information visit https://errors.pydantic.dev/2.12/v/int_type
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be a valid integer [type=int_type, input_value=None, input_type=NoneType]     For further information visit https://errors.pydantic.dev/2.12/v/int_type
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be a valid integer [type=int_type, input_value=None, input_type=NoneType]     For further information visit https://errors.pydantic.dev/2.12/v/int_type
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'