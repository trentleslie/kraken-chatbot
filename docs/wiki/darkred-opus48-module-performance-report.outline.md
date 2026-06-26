# Darkred Module Run on Opus 4.8: Pipeline Performance Report (26-analyte, dev, 2026-06-24)

This is the per-node performance report emitted automatically at the end of the full-module **Darkred** discovery run on dev (commit `a7d7fd6`). All SDK-backed nodes ran on the Opus 4.8 model; this report is published for direct model comparison against the Sonnet baseline. The Kraken performance-reporter (a terminal graph node) attributes wall-clock, tokens, estimated cost, output counts, and errors across the twelve pipeline nodes, and the Context management section reports how synthesis compresses the accumulated evidence into a bounded context.

## Related

- Discovery analysis from this run: [Darkred Module Run on Opus 4.8: Discovery Output](https://phwiki.phenoma.ai/doc/darkred-module-run-on-opus-48-discovery-output-26-analyte-dev-2026-06-24-P7JQj2iKkp)
- Model comparison baseline (Sonnet): [Darkred Module Run: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/darkred-module-run-pipeline-performance-report-26-analyte-dev-2026-06-23-rTsggJIalU)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---

## Pipeline Performance Report

- **Run:** `f0289f18672b4c4fb9a8970c2f289b56`
- **Mode:** pipeline · **biomapper_env:** dev
- **Timestamp:** 20260624T183945Z · **commit:** `a7d7fd63deffdbbb25f8b30e8dee40cefdf6c3cd`
- **Wall-clock:** 1092.356s · **tokens:** 26 in / 57497 out (168652 cache-r, 212641 cache-w) · **est. cost:** $2.8509
- **Top bottleneck:** pathway_enrichment · **Top cost:** pathway_enrichment

> Wall-clock is estimated as the sum of per-node durations, because no measured elapsed time was available; for concurrent branches this sum overstates true elapsed time. The percentage column reports each node's share of that summed total.

### Per-node

| Node | Status | Duration (s) | % | In | Out | Cache r/w | Model | MCP | Est. $ | Findings |
|---|---|---|---|---|---|---|---|---|---|---|
| pathway_enrichment | ran | 377.715 | 34.6% | 6 | 36760 | 34336/66371 | claude-opus-4-8 | 0 | $1.3510 | 0 shared neighbors (0 non-hub), 0 biological themes |
| integration | ran | 235.325 | 21.5% | 3 | 4750 | 48452/13115 | claude-opus-4-8 | 0 | $0.2250 | 16 bridges, 19 gap entities |
| synthesis | ran | 134.105 | 12.3% | 3 | 5650 | 0/57630 | claude-opus-4-8 | 0 | $0.5015 | 86 hypotheses generated, 22201 char report |
| cold_start | ran | 125.766 | 11.5% | 14 | 10337 | 85864/75525 | claude-opus-4-8 | 0 | $0.7735 | 21 analogues, 34 inferred associations, 36 findings |
| direct_kg | ran | 71.198 | 6.5% | 0 | 0 | 0/0 | – | 0 | est. n/a | 178 disease associations, 107 pathways, 652 findings |
| literature_grounding | ran | 62.232 | 5.7% | 0 | 0 | 0/0 | – | 0 | est. n/a | 17 papers, 17 from Exa across 15/86 hypotheses |
| bridge_grounding | ran | 34.079 | 3.1% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| entity_resolution | ran | 28.048 | 2.6% | 0 | 0 | 0/0 | – | 0 | est. n/a | 26/26 entities resolved (22 biomapper, 1 exact, 3 fuzzy) |
| triage | ran | 19.98 | 1.8% | 0 | 0 | 0/0 | – | 0 | est. n/a | 5 well-characterized, 11 moderate, 8 sparse, 2 cold-start |
| hypothesis_extraction | ran | 3.906 | 0.4% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| intake | ran | 0.001 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 26 entities extracted, query_type=discovery |
| temporal | skipped | – | – | – | – | – | – | – | – | – |

### Context management

> Synthesis compresses the accumulated evidence into a bounded context; elision at module scale is expected, and it is the compression that keeps the assembled context within the model's token window.

- **Context:** 102298 chars (~29228 est. tokens) · 29.2% of the char budget (350000) · 14.6% of the 200K-token window
- **Mode:** module-aware aggregation (25 distinct entities; threshold 5)
- **Literature grounding:** 15 of 86 hypotheses carry attached literature

| Section | Shown | Total | Elided |
|---|---|---|---|
| Findings | 150 | 688 | 538 |
| Diseases (recurrence) | 21 | 21 | 0 |
| Pathways (recurrence) | 11 | 11 | 0 |
| Member table | 25 | 25 | 0 |

### Errors

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