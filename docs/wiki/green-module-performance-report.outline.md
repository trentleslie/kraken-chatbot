# Green Module Run: Pipeline Performance Report (83-analyte, dev, 2026-06-23)

This is the per-node performance report emitted automatically at the end of the full-module **Green** discovery run on dev (commit `813fc3f`). The Kraken performance-reporter (a terminal graph node) attributes wall-clock, tokens, estimated cost, output counts, and errors across the twelve pipeline nodes, and the Context management section reports how synthesis compresses the accumulated evidence into a bounded context.

## Related

- Discovery analysis from this run: [Green Module Run: Discovery Output](https://phwiki.phenoma.ai/doc/green-module-run-discovery-output-83-analyte-dev-2026-06-23-tAMwedxCJR)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---

## Pipeline Performance Report

- **Run:** `8e957fd71812445e93708195cfc965f0`
- **Mode:** pipeline · **biomapper_env:** dev
- **Timestamp:** 20260623T171010Z · **commit:** `813fc3fb835aed414fc314f0942524a48ca6fbb9`
- **Wall-clock:** 798.009s · **tokens:** 27 in / 23288 out (128796 cache-r, 196322 cache-w) · **est. cost:** $1.1242
- **Top bottleneck:** cold_start · **Top cost:** cold_start

> Wall-clock is estimated as the sum of per-node durations, because no measured elapsed time was available; for concurrent branches this sum overstates true elapsed time. The percentage column reports each node's share of that summed total.

### Per-node

| Node | Status | Duration (s) | % | In | Out | Cache r/w | Model | MCP | Est. $ | Findings |
|---|---|---|---|---|---|---|---|---|---|---|
| cold_start | ran | 157.141 | 19.7% | 16 | 6589 | 85864/97820 | anthropic/claude-sonnet-4-20250514 | 0 | $0.4915 | 24 analogues, 16 inferred associations, 19 findings |
| synthesis | ran | 149.807 | 18.8% | 3 | 7001 | 0/55508 | anthropic/claude-sonnet-4-20250514 | 0 | $0.3132 | 45 hypotheses generated, 27099 char report |
| pathway_enrichment | ran | 133.632 | 16.7% | 3 | 6703 | 21466/13694 | anthropic/claude-sonnet-4-20250514 | 0 | $0.1583 | 1 shared neighbors (0 non-hub), 3 biological themes |
| direct_kg | ran | 78.536 | 9.8% | 0 | 0 | 0/0 | – | 0 | est. n/a | 141 disease associations, 212 pathways, 1337 findings |
| entity_resolution | ran | 70.602 | 8.8% | 3 | 424 | 21466/1799 | anthropic/claude-sonnet-4-20250514 | 0 | $0.0196 | 56/56 entities resolved (13 biomapper, 42 fuzzy, 1 semantic) |
| literature_grounding | ran | 58.955 | 7.4% | 0 | 0 | 0/0 | – | 0 | est. n/a | 20 papers, 20 from Exa across 11/45 hypotheses |
| integration | ran | 58.798 | 7.4% | 2 | 2571 | 0/27501 | anthropic/claude-sonnet-4-20250514 | 0 | $0.1417 | 10 bridges, 11 gap entities |
| bridge_grounding | ran | 48.183 | 6.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| triage | ran | 42.354 | 5.3% | 0 | 0 | 0/0 | – | 0 | est. n/a | 6 well-characterized, 24 moderate, 19 sparse, 7 cold-start |
| intake | ran | 0.003 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 56 entities extracted, query_type=discovery |
| hypothesis_extraction | ran | 0.0 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| temporal | skipped | – | – | – | – | – | – | – | – | – |

### Context management

> Synthesis compresses the accumulated evidence into a bounded context; elision at module scale is expected, and it is the compression that keeps the assembled context within the model's token window.

- **Context:** 85108 chars (~24317 est. tokens) · 24.3% of the char budget (350000) · 12.2% of the 200K-token window
- **Mode:** module-aware aggregation (45 distinct entities; threshold 5)
- **Literature grounding:** 11 of 45 hypotheses carry attached literature

| Section | Shown | Total | Elided |
|---|---|---|---|
| Findings | 126 | 1356 | 1230 |
| Diseases (recurrence) | 12 | 12 | 0 |
| Pathways (recurrence) | 2 | 2 | 0 |
| Member table | 45 | 45 | 0 |

### Errors

- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be a valid integer [type=int_type, input_value=None, input_type=NoneType]     For further information visit https://errors.pydantic.dev/2.12/v/int_type
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
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be a valid integer [type=int_type, input_value=None, input_type=NoneType]     For further information visit https://errors.pydantic.dev/2.12/v/int_type
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'