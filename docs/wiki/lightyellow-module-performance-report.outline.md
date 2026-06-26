# Lightyellow Module Run: Pipeline Performance Report (19-analyte, dev, 2026-06-23)

This is the per-node performance report emitted automatically at the end of the full-module **Lightyellow** discovery run on dev (commit `813fc3f`). The Kraken performance-reporter (a terminal graph node) attributes wall-clock, tokens, estimated cost, output counts, and errors across the twelve pipeline nodes, and the Context management section reports how synthesis compresses the accumulated evidence into a bounded context.

## Related

- Discovery analysis from this run: [Lightyellow Module Run: Discovery Output](https://phwiki.phenoma.ai/doc/lightyellow-module-run-discovery-output-19-analyte-dev-2026-06-23-iDbMdsgIiU)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---

## Pipeline Performance Report

- **Run:** `565fbf4355674cf38127db60443ccb21`
- **Mode:** pipeline · **biomapper_env:** dev
- **Timestamp:** 20260623T191140Z · **commit:** `813fc3fb835aed414fc314f0942524a48ca6fbb9`
- **Wall-clock:** 618.03s · **tokens:** 18 in / 22991 out (42932 cache-r, 192884 cache-w) · **est. cost:** $1.0811
- **Top bottleneck:** synthesis · **Top cost:** cold_start

> Wall-clock is estimated as the sum of per-node durations, because no measured elapsed time was available; for concurrent branches this sum overstates true elapsed time. The percentage column reports each node's share of that summed total.

### Per-node

| Node | Status | Duration (s) | % | In | Out | Cache r/w | Model | MCP | Est. $ | Findings |
|---|---|---|---|---|---|---|---|---|---|---|
| synthesis | ran | 163.564 | 26.5% | 3 | 7448 | 0/59823 | anthropic/claude-sonnet-4-20250514 | 0 | $0.3361 | 96 hypotheses generated, 28694 char report |
| cold_start | ran | 120.646 | 19.5% | 10 | 10147 | 42932/73317 | anthropic/claude-sonnet-4-20250514 | 0 | $0.4401 | 18 analogues, 37 inferred associations, 39 findings |
| pathway_enrichment | ran | 97.879 | 15.8% | 3 | 2943 | 0/32816 | anthropic/claude-sonnet-4-20250514 | 0 | $0.1672 | 0 shared neighbors (0 non-hub), 4 biological themes |
| literature_grounding | ran | 62.755 | 10.2% | 0 | 0 | 0/0 | – | 0 | est. n/a | 37 papers, 37 from Exa across 15/96 hypotheses |
| integration | ran | 55.915 | 9.0% | 2 | 2453 | 0/26928 | anthropic/claude-sonnet-4-20250514 | 0 | $0.1378 | 20 bridges, 10 gap entities |
| bridge_grounding | ran | 51.195 | 8.3% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| direct_kg | ran | 29.782 | 4.8% | 0 | 0 | 0/0 | – | 0 | est. n/a | 333 disease associations, 276 pathways, 1011 findings, 1 hub flags |
| entity_resolution | ran | 21.936 | 3.5% | 0 | 0 | 0/0 | – | 0 | est. n/a | 19/19 entities resolved (17 biomapper, 2 fuzzy) |
| triage | ran | 14.357 | 2.3% | 0 | 0 | 0/0 | – | 0 | est. n/a | 9 well-characterized, 3 moderate, 6 sparse, 1 cold-start |
| intake | ran | 0.001 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 19 entities extracted, query_type=discovery |
| hypothesis_extraction | ran | 0.0 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| temporal | skipped | – | – | – | – | – | – | – | – | – |

### Context management

> Synthesis compresses the accumulated evidence into a bounded context; elision at module scale is expected, and it is the compression that keeps the assembled context within the model's token window.

- **Context:** 107135 chars (~30610 est. tokens) · 30.6% of the char budget (350000) · 15.3% of the 200K-token window
- **Mode:** module-aware aggregation (19 distinct entities; threshold 5)
- **Literature grounding:** 15 of 96 hypotheses carry attached literature

| Section | Shown | Total | Elided |
|---|---|---|---|
| Findings | 145 | 1050 | 905 |
| Diseases (recurrence) | 30 | 58 | 28 |
| Pathways (recurrence) | 23 | 23 | 0 |
| Member table | 19 | 19 | 0 |

### Errors

- SDK inference timed out for CHEBI:88956 after 60s
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be a valid integer [type=int_type, input_value=None, input_type=NoneType]     For further information visit https://errors.pydantic.dev/2.12/v/int_type
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be a valid integer [type=int_type, input_value=None, input_type=NoneType]     For further information visit https://errors.pydantic.dev/2.12/v/int_type
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be a valid integer [type=int_type, input_value=None, input_type=NoneType]     For further information visit https://errors.pydantic.dev/2.12/v/int_type
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be a valid integer [type=int_type, input_value=None, input_type=NoneType]     For further information visit https://errors.pydantic.dev/2.12/v/int_type
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be a valid integer [type=int_type, input_value=None, input_type=NoneType]     For further information visit https://errors.pydantic.dev/2.12/v/int_type
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'