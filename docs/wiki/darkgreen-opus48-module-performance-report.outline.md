# Darkgreen Module Run on Opus 4.8: Pipeline Performance Report (20-analyte, dev, 2026-06-24)

This is the per-node performance report emitted automatically at the end of the full-module **Darkgreen** discovery run on dev (commit `a7d7fd6`). All SDK-backed nodes ran on the Opus 4.8 model; this report is published for direct model comparison against the Sonnet baseline. The Kraken performance-reporter (a terminal graph node) attributes wall-clock, tokens, estimated cost, output counts, and errors across the twelve pipeline nodes, and the Context management section reports how synthesis compresses the accumulated evidence into a bounded context.

## Related

- Discovery analysis from this run: [Darkgreen Module Run on Opus 4.8: Discovery Output](https://phwiki.phenoma.ai/doc/darkgreen-module-run-on-opus-48-discovery-output-20-analyte-dev-2026-06-24-CUJzRhbjhg)
- Model comparison baseline (Sonnet): [Darkgreen Module Run: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/darkgreen-module-run-pipeline-performance-report-20-analyte-dev-2026-06-23-3rOl51fryq)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---

## Pipeline Performance Report

- **Run:** `d837f729eb8b415e933d4a5f24373ff8`
- **Mode:** pipeline · **biomapper_env:** dev
- **Timestamp:** 20260624T184937Z · **commit:** `a7d7fd63deffdbbb25f8b30e8dee40cefdf6c3cd`
- **Wall-clock:** 632.065s · **tokens:** 20 in / 29793 out (85864 cache-r, 166738 cache-w) · **est. cost:** $1.8300
- **Top bottleneck:** synthesis · **Top cost:** cold_start

> Wall-clock is estimated as the sum of per-node durations, because no measured elapsed time was available; for concurrent branches this sum overstates true elapsed time. The percentage column reports each node's share of that summed total.

### Per-node

| Node | Status | Duration (s) | % | In | Out | Cache r/w | Model | MCP | Est. $ | Findings |
|---|---|---|---|---|---|---|---|---|---|---|
| synthesis | ran | 163.546 | 25.9% | 3 | 6784 | 0/56506 | claude-opus-4-8 | 0 | $0.5228 | 69 hypotheses generated, 26408 char report |
| pathway_enrichment | ran | 144.893 | 22.9% | 3 | 10753 | 21466/9093 | claude-opus-4-8 | 0 | $0.3364 | 0 shared neighbors (0 non-hub), 6 biological themes |
| cold_start | ran | 108.326 | 17.1% | 12 | 9487 | 42932/95573 | claude-opus-4-8 | 0 | $0.8560 | 18 analogues, 34 inferred associations, 35 findings |
| integration | ran | 62.536 | 9.9% | 2 | 2769 | 21466/5566 | claude-opus-4-8 | 0 | $0.1148 | 0 bridges, 13 gap entities |
| literature_grounding | ran | 62.482 | 9.9% | 0 | 0 | 0/0 | – | 0 | est. n/a | 32 papers, 26 from Exa across 13/69 hypotheses |
| direct_kg | ran | 51.583 | 8.2% | 0 | 0 | 0/0 | – | 0 | est. n/a | 161 disease associations, 38 pathways, 456 findings |
| triage | ran | 20.975 | 3.3% | 0 | 0 | 0/0 | – | 0 | est. n/a | 3 well-characterized, 8 moderate, 9 sparse, 1 cold-start |
| entity_resolution | ran | 17.721 | 2.8% | 0 | 0 | 0/0 | – | 0 | est. n/a | 21/21 entities resolved (3 fuzzy, 17 biomapper, 1 exact) |
| intake | ran | 0.001 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 21 entities extracted, query_type=discovery |
| hypothesis_extraction | ran | 0.0 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| bridge_grounding | ran | 0.0 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| temporal | skipped | – | – | – | – | – | – | – | – | – |

### Context management

> Synthesis compresses the accumulated evidence into a bounded context; elision at module scale is expected, and it is the compression that keeps the assembled context within the model's token window.

- **Context:** 99800 chars (~28514 est. tokens) · 28.5% of the char budget (350000) · 14.3% of the 200K-token window
- **Mode:** module-aware aggregation (21 distinct entities; threshold 5)
- **Literature grounding:** 13 of 69 hypotheses carry attached literature

| Section | Shown | Total | Elided |
|---|---|---|---|
| Findings | 143 | 491 | 348 |
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
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be a valid integer [type=int_type, input_value=None, input_type=NoneType]     For further information visit https://errors.pydantic.dev/2.12/v/int_type
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be a valid integer [type=int_type, input_value=None, input_type=NoneType]     For further information visit https://errors.pydantic.dev/2.12/v/int_type
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be a valid integer [type=int_type, input_value=None, input_type=NoneType]     For further information visit https://errors.pydantic.dev/2.12/v/int_type
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be a valid integer [type=int_type, input_value=None, input_type=NoneType]     For further information visit https://errors.pydantic.dev/2.12/v/int_type
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be a valid integer [type=int_type, input_value=None, input_type=NoneType]     For further information visit https://errors.pydantic.dev/2.12/v/int_type
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be a valid integer [type=int_type, input_value=None, input_type=NoneType]     For further information visit https://errors.pydantic.dev/2.12/v/int_type
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be a valid integer [type=int_type, input_value=None, input_type=NoneType]     For further information visit https://errors.pydantic.dev/2.12/v/int_type
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'