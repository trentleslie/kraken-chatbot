# Lightgreen Module Run: Pipeline Performance Report (29-analyte, dev, 2026-06-23)

This is the per-node performance report emitted automatically at the end of the full-module **Lightgreen** discovery run on dev (commit `813fc3f`). The Kraken performance-reporter (a terminal graph node) attributes wall-clock, tokens, estimated cost, output counts, and errors across the twelve pipeline nodes, and the Context management section reports how synthesis compresses the accumulated evidence into a bounded context.

## Related

- Discovery analysis from this run: [Lightgreen Module Run: Discovery Output](https://phwiki.phenoma.ai/doc/lightgreen-module-run-discovery-output-29-analyte-dev-2026-06-23-AbI5HbhH8b)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---

## Pipeline Performance Report

- **Run:** `115c098282bc46b9b7e8b84e821fa993`
- **Mode:** pipeline · **biomapper_env:** dev
- **Timestamp:** 20260623T190138Z · **commit:** `813fc3fb835aed414fc314f0942524a48ca6fbb9`
- **Wall-clock:** 812.212s · **tokens:** 23 in / 27360 out (198839 cache-r, 107250 cache-w) · **est. cost:** $0.8723
- **Top bottleneck:** cold_start · **Top cost:** cold_start

> Wall-clock is estimated as the sum of per-node durations, because no measured elapsed time was available; for concurrent branches this sum overstates true elapsed time. The percentage column reports each node's share of that summed total.

### Per-node

| Node | Status | Duration (s) | % | In | Out | Cache r/w | Model | MCP | Est. $ | Findings |
|---|---|---|---|---|---|---|---|---|---|---|
| cold_start | ran | 186.114 | 22.9% | 14 | 9842 | 107330/53456 | anthropic/claude-sonnet-4-20250514 | 0 | $0.3803 | 24 analogues, 34 inferred associations, 36 findings |
| integration | ran | 170.106 | 20.9% | 3 | 3540 | 48577/10745 | anthropic/claude-sonnet-4-20250514 | 0 | $0.1080 | 0 bridges, 12 gap entities |
| synthesis | ran | 159.898 | 19.7% | 3 | 6819 | 21466/30556 | anthropic/claude-sonnet-4-20250514 | 0 | $0.2233 | 70 hypotheses generated, 25649 char report |
| pathway_enrichment | ran | 118.945 | 14.6% | 3 | 7159 | 21466/12493 | anthropic/claude-sonnet-4-20250514 | 0 | $0.1607 | 0 shared neighbors (0 non-hub), 4 biological themes |
| literature_grounding | ran | 65.243 | 8.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 19 papers, 18 from Exa across 12/70 hypotheses |
| direct_kg | ran | 59.495 | 7.3% | 0 | 0 | 0/0 | – | 0 | est. n/a | 108 disease associations, 101 pathways, 654 findings |
| entity_resolution | ran | 28.678 | 3.5% | 0 | 0 | 0/0 | – | 0 | est. n/a | 30/30 entities resolved (14 fuzzy, 15 biomapper, 1 exact) |
| triage | ran | 23.73 | 2.9% | 0 | 0 | 0/0 | – | 0 | est. n/a | 5 well-characterized, 10 moderate, 11 sparse, 4 cold-start |
| intake | ran | 0.002 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 30 entities extracted, query_type=discovery |
| hypothesis_extraction | ran | 0.0 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| bridge_grounding | ran | 0.0 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| temporal | skipped | – | – | – | – | – | – | – | – | – |

### Context management

> Synthesis compresses the accumulated evidence into a bounded context; elision at module scale is expected, and it is the compression that keeps the assembled context within the model's token window.

- **Context:** 80141 chars (~22897 est. tokens) · 22.9% of the char budget (350000) · 11.4% of the 200K-token window
- **Mode:** module-aware aggregation (30 distinct entities; threshold 5)
- **Literature grounding:** 12 of 70 hypotheses carry attached literature

| Section | Shown | Total | Elided |
|---|---|---|---|
| Findings | 144 | 690 | 546 |
| Diseases (recurrence) | 10 | 10 | 0 |
| Pathways (recurrence) | 2 | 2 | 0 |
| Member table | 30 | 30 | 0 |

### Errors

- SDK inference timed out for PUBCHEM.COMPOUND:11728391 after 60s
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
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be a valid integer [type=int_type, input_value=None, input_type=NoneType]     For further information visit https://errors.pydantic.dev/2.12/v/int_type