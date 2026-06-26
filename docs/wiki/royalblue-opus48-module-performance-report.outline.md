# Royalblue Module Run on Opus 4.8: Pipeline Performance Report (27-analyte, dev, 2026-06-24)

This is the per-node performance report emitted automatically at the end of the full-module **Royalblue** discovery run on dev (commit `a7d7fd6`). All SDK-backed nodes ran on the Opus 4.8 model; this report is published for direct model comparison against the Sonnet baseline. The Kraken performance-reporter (a terminal graph node) attributes wall-clock, tokens, estimated cost, output counts, and errors across the twelve pipeline nodes, and the Context management section reports how synthesis compresses the accumulated evidence into a bounded context.

## Related

- Discovery analysis from this run: [Royalblue Module Run on Opus 4.8: Discovery Output](https://phwiki.phenoma.ai/doc/royalblue-module-run-on-opus-48-discovery-output-27-analyte-dev-2026-06-24-bOraLLXTDT)
- Model comparison baseline (Sonnet): [Royalblue Module Run: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/royalblue-module-run-pipeline-performance-report-27-analyte-dev-2026-06-23-5VhvILmveG)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---

## Pipeline Performance Report

- **Run:** `d1c010ce22444736bb7edd1967d709c4`
- **Mode:** pipeline · **biomapper_env:** dev
- **Timestamp:** 20260624T182232Z · **commit:** `a7d7fd63deffdbbb25f8b30e8dee40cefdf6c3cd`
- **Wall-clock:** 694.483s · **tokens:** 24 in / 38388 out (150262 cache-r, 150221 cache-w) · **est. cost:** $1.9738
- **Top bottleneck:** synthesis · **Top cost:** cold_start

> Wall-clock is estimated as the sum of per-node durations, because no measured elapsed time was available; for concurrent branches this sum overstates true elapsed time. The percentage column reports each node's share of that summed total.

### Per-node

| Node | Status | Duration (s) | % | In | Out | Cache r/w | Model | MCP | Est. $ | Findings |
|---|---|---|---|---|---|---|---|---|---|---|
| synthesis | ran | 163.168 | 23.5% | 3 | 7110 | 21466/38644 | claude-opus-4-8 | 0 | $0.4300 | 123 hypotheses generated, 25290 char report |
| pathway_enrichment | ran | 161.077 | 23.2% | 3 | 14696 | 21466/8707 | claude-opus-4-8 | 0 | $0.4326 | 0 shared neighbors (0 non-hub), 8 biological themes |
| cold_start | ran | 110.358 | 15.9% | 16 | 13651 | 85864/97746 | claude-opus-4-8 | 0 | $0.9952 | 24 analogues, 49 inferred associations, 49 findings |
| integration | ran | 66.02 | 9.5% | 2 | 2931 | 21466/5124 | claude-opus-4-8 | 0 | $0.1160 | 25 bridges, 12 gap entities |
| literature_grounding | ran | 60.883 | 8.8% | 0 | 0 | 0/0 | – | 0 | est. n/a | 10 papers, 10 from Exa across 10/123 hypotheses |
| bridge_grounding | ran | 53.145 | 7.7% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| direct_kg | ran | 37.403 | 5.4% | 0 | 0 | 0/0 | – | 0 | est. n/a | 137 disease associations, 95 pathways, 476 findings |
| entity_resolution | ran | 22.172 | 3.2% | 0 | 0 | 0/0 | – | 0 | est. n/a | 25/25 entities resolved (10 biomapper, 14 fuzzy, 1 exact) |
| triage | ran | 20.256 | 2.9% | 0 | 0 | 0/0 | – | 0 | est. n/a | 3 well-characterized, 7 moderate, 9 sparse, 6 cold-start |
| intake | ran | 0.002 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 25 entities extracted, query_type=discovery |
| hypothesis_extraction | ran | 0.0 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| temporal | skipped | – | – | – | – | – | – | – | – | – |

### Context management

> Synthesis compresses the accumulated evidence into a bounded context; elision at module scale is expected, and it is the compression that keeps the assembled context within the model's token window.

- **Context:** 106095 chars (~30313 est. tokens) · 30.3% of the char budget (350000) · 15.2% of the 200K-token window
- **Mode:** module-aware aggregation (24 distinct entities; threshold 5)
- **Literature grounding:** 10 of 123 hypotheses carry attached literature

| Section | Shown | Total | Elided |
|---|---|---|---|
| Findings | 150 | 525 | 375 |
| Diseases (recurrence) | 12 | 12 | 0 |
| Pathways (recurrence) | 18 | 18 | 0 |
| Member table | 24 | 24 | 0 |

### Errors

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
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
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
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'