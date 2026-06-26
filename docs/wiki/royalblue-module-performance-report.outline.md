# Royalblue Module Run: Pipeline Performance Report (27-analyte, dev, 2026-06-23)

This is the per-node performance report emitted automatically at the end of the full-module **Royalblue** discovery run on dev (commit `813fc3f`). The Kraken performance-reporter (a terminal graph node) attributes wall-clock, tokens, estimated cost, output counts, and errors across the twelve pipeline nodes, and the Context management section reports how synthesis compresses the accumulated evidence into a bounded context.

## Related

- Discovery analysis from this run: [Royalblue Module Run: Discovery Output](https://phwiki.phenoma.ai/doc/royalblue-module-run-discovery-output-27-analyte-dev-2026-06-23-9lq8RXNhPN)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---

## Pipeline Performance Report

- **Run:** `66a9e63362fb4042bbaac9c1d5d96b8d`
- **Mode:** pipeline · **biomapper_env:** dev
- **Timestamp:** 20260623T192154Z · **commit:** `813fc3fb835aed414fc314f0942524a48ca6fbb9`
- **Wall-clock:** 621.529s · **tokens:** 24 in / 30593 out (128796 cache-r, 169938 cache-w) · **est. cost:** $1.1349
- **Top bottleneck:** synthesis · **Top cost:** cold_start

> Wall-clock is estimated as the sum of per-node durations, because no measured elapsed time was available; for concurrent branches this sum overstates true elapsed time. The percentage column reports each node's share of that summed total.

### Per-node

| Node | Status | Duration (s) | % | In | Out | Cache r/w | Model | MCP | Est. $ | Findings |
|---|---|---|---|---|---|---|---|---|---|---|
| synthesis | ran | 153.977 | 24.8% | 3 | 7143 | 0/60052 | anthropic/claude-sonnet-4-20250514 | 0 | $0.3323 | 123 hypotheses generated, 25595 char report |
| cold_start | ran | 115.805 | 18.6% | 16 | 13882 | 85864/97746 | anthropic/claude-sonnet-4-20250514 | 0 | $0.6006 | 24 analogues, 49 inferred associations, 49 findings |
| pathway_enrichment | ran | 114.284 | 18.4% | 3 | 7159 | 21466/6961 | anthropic/claude-sonnet-4-20250514 | 0 | $0.1399 | 0 shared neighbors (0 non-hub), 8 biological themes |
| literature_grounding | ran | 63.699 | 10.2% | 0 | 0 | 0/0 | – | 0 | est. n/a | 10 papers, 10 from Exa across 10/123 hypotheses |
| integration | ran | 60.753 | 9.8% | 2 | 2409 | 21466/5179 | anthropic/claude-sonnet-4-20250514 | 0 | $0.0620 | 25 bridges, 12 gap entities |
| bridge_grounding | ran | 43.31 | 7.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| direct_kg | ran | 26.747 | 4.3% | 0 | 0 | 0/0 | – | 0 | est. n/a | 137 disease associations, 95 pathways, 473 findings |
| entity_resolution | ran | 24.539 | 3.9% | 0 | 0 | 0/0 | – | 0 | est. n/a | 25/25 entities resolved (10 biomapper, 14 fuzzy, 1 exact) |
| triage | ran | 18.413 | 3.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 3 well-characterized, 7 moderate, 9 sparse, 6 cold-start |
| intake | ran | 0.002 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 25 entities extracted, query_type=discovery |
| hypothesis_extraction | ran | 0.0 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| temporal | skipped | – | – | – | – | – | – | – | – | – |

### Context management

> Synthesis compresses the accumulated evidence into a bounded context; elision at module scale is expected, and it is the compression that keeps the assembled context within the model's token window.

- **Context:** 105481 chars (~30137 est. tokens) · 30.1% of the char budget (350000) · 15.1% of the 200K-token window
- **Mode:** module-aware aggregation (24 distinct entities; threshold 5)
- **Literature grounding:** 10 of 123 hypotheses carry attached literature

| Section | Shown | Total | Elided |
|---|---|---|---|
| Findings | 150 | 522 | 372 |
| Diseases (recurrence) | 12 | 12 | 0 |
| Pathways (recurrence) | 18 | 18 | 0 |
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
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be a valid integer [type=int_type, input_value=None, input_type=NoneType]     For further information visit https://errors.pydantic.dev/2.12/v/int_type
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be a valid integer [type=int_type, input_value=None, input_type=NoneType]     For further information visit https://errors.pydantic.dev/2.12/v/int_type
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be a valid integer [type=int_type, input_value=None, input_type=NoneType]     For further information visit https://errors.pydantic.dev/2.12/v/int_type
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be a valid integer [type=int_type, input_value=None, input_type=NoneType]     For further information visit https://errors.pydantic.dev/2.12/v/int_type
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be a valid integer [type=int_type, input_value=None, input_type=NoneType]     For further information visit https://errors.pydantic.dev/2.12/v/int_type
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