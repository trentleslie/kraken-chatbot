# Magenta Module Run: Pipeline Performance Report (47-analyte, dev, 2026-06-23)

This is the per-node performance report emitted automatically at the end of the full-module **Magenta** discovery run on dev (commit `813fc3f`). The Kraken performance-reporter (a terminal graph node) attributes wall-clock, tokens, estimated cost, output counts, and errors across the twelve pipeline nodes, and the Context management section reports how synthesis compresses the accumulated evidence into a bounded context.

## Related

- Discovery analysis from this run: [Magenta Module Run: Discovery Output](https://phwiki.phenoma.ai/doc/magenta-module-run-discovery-output-47-analyte-dev-2026-06-23-NALz0ymcml)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---

## Pipeline Performance Report

- **Run:** `10eb169d8c0d407ea9e0868a7eef9e04`
- **Mode:** pipeline · **biomapper_env:** dev
- **Timestamp:** 20260623T175037Z · **commit:** `813fc3fb835aed414fc314f0942524a48ca6fbb9`
- **Wall-clock:** 896.798s · **tokens:** 25 in / 28479 out (155830 cache-r, 186997 cache-w) · **est. cost:** $1.1752
- **Top bottleneck:** pathway_enrichment · **Top cost:** cold_start

> Wall-clock is estimated as the sum of per-node durations, because no measured elapsed time was available; for concurrent branches this sum overstates true elapsed time. The percentage column reports each node's share of that summed total.

### Per-node

| Node | Status | Duration (s) | % | In | Out | Cache r/w | Model | MCP | Est. $ | Findings |
|---|---|---|---|---|---|---|---|---|---|---|
| pathway_enrichment | ran | 217.127 | 24.2% | 3 | 8611 | 21466/23069 | anthropic/claude-sonnet-4-20250514 | 0 | $0.2221 | 0 shared neighbors (0 non-hub), 6 biological themes |
| synthesis | ran | 176.045 | 19.6% | 3 | 8185 | 21466/35489 | anthropic/claude-sonnet-4-20250514 | 0 | $0.2623 | 48 hypotheses generated, 31163 char report |
| cold_start | ran | 145.421 | 16.2% | 16 | 8576 | 85864/97162 | anthropic/claude-sonnet-4-20250514 | 0 | $0.5188 | 24 analogues, 22 inferred associations, 26 findings |
| integration | ran | 128.174 | 14.3% | 3 | 3107 | 27034/31277 | anthropic/claude-sonnet-4-20250514 | 0 | $0.1720 | 0 bridges, 10 gap entities |
| direct_kg | ran | 85.776 | 9.6% | 0 | 0 | 0/0 | – | 0 | est. n/a | 808 disease associations, 610 pathways, 2241 findings, 2 hub flags |
| literature_grounding | ran | 61.102 | 6.8% | 0 | 0 | 0/0 | – | 0 | est. n/a | 26 papers, 21 from Exa across 12/48 hypotheses |
| entity_resolution | ran | 44.897 | 5.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 47/47 entities resolved (35 biomapper, 3 exact, 9 fuzzy) |
| triage | ran | 38.254 | 4.3% | 0 | 0 | 0/0 | – | 0 | est. n/a | 19 well-characterized, 7 moderate, 15 sparse, 6 cold-start |
| intake | ran | 0.002 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 47 entities extracted, query_type=discovery |
| hypothesis_extraction | ran | 0.0 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| bridge_grounding | ran | 0.0 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| temporal | skipped | – | – | – | – | – | – | – | – | – |

### Context management

> Synthesis compresses the accumulated evidence into a bounded context; elision at module scale is expected, and it is the compression that keeps the assembled context within the model's token window.

- **Context:** 93747 chars (~26785 est. tokens) · 26.8% of the char budget (350000) · 13.4% of the 200K-token window
- **Mode:** module-aware aggregation (47 distinct entities; threshold 5)
- **Literature grounding:** 12 of 48 hypotheses carry attached literature

| Section | Shown | Total | Elided |
|---|---|---|---|
| Findings | 133 | 2267 | 2134 |
| Diseases (recurrence) | 30 | 139 | 109 |
| Pathways (recurrence) | 30 | 95 | 65 |
| Member table | 47 | 47 | 0 |

### Errors

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
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'