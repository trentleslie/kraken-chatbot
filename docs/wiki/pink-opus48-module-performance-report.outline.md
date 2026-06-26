# Pink Module Run on Opus 4.8: Pipeline Performance Report (54-analyte, dev, 2026-06-24)

This is the per-node performance report emitted automatically at the end of the full-module **Pink** discovery run on dev (commit `098093e`). All SDK-backed nodes ran on the Opus 4.8 model; this report is published for direct model comparison against the Sonnet baseline. The Kraken performance-reporter (a terminal graph node) attributes wall-clock, tokens, estimated cost, output counts, and errors across the twelve pipeline nodes, and the Context management section reports how synthesis compresses the accumulated evidence into a bounded context.

## Related

- Discovery analysis from this run: [Pink Module Run on Opus 4.8: Discovery Output](https://phwiki.phenoma.ai/doc/pink-module-run-on-opus-48-discovery-output-54-analyte-dev-2026-06-24-QdbiKKbhTA)
- Model comparison baseline (Sonnet): [Pink Module Run: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/pink-module-run-pipeline-performance-report-54-analyte-dev-2026-06-23-sBZjXUAb57)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---

## Pipeline Performance Report

- **Run:** `1d7ad997710049dbb1a6761b0de1c426`
- **Mode:** pipeline · **biomapper_env:** dev
- **Timestamp:** 20260624T022026Z · **commit:** `098093ea146164cb1ed09db4b8dcc4e8fb1c39a9`
- **Wall-clock:** 621.845s · **tokens:** 34 in / 23351 out (218972 cache-r, 171703 cache-w) · **est. cost:** $1.7666
- **Top bottleneck:** synthesis · **Top cost:** cold_start

> Wall-clock is estimated as the sum of per-node durations, because no measured elapsed time was available; for concurrent branches this sum overstates true elapsed time. The percentage column reports each node's share of that summed total.

### Per-node

| Node | Status | Duration (s) | % | In | Out | Cache r/w | Model | MCP | Est. $ | Findings |
|---|---|---|---|---|---|---|---|---|---|---|
| synthesis | ran | 137.68 | 22.1% | 3 | 6033 | 0/54679 | claude-opus-4-8 | 0 | $0.4926 | 44 hypotheses generated, 23088 char report |
| cold_start | ran | 108.75 | 17.5% | 16 | 9341 | 108606/76053 | claude-opus-4-8 | 0 | $0.7632 | 24 analogues, 20 inferred associations, 24 findings |
| pathway_enrichment | ran | 88.777 | 14.3% | 3 | 5220 | 21466/8165 | claude-opus-4-8 | 0 | $0.1923 | 0 shared neighbors (0 non-hub), 3 biological themes |
| integration | ran | 86.274 | 13.9% | 3 | 2685 | 48234/6024 | claude-opus-4-8 | 0 | $0.1289 | 0 bridges, 10 gap entities |
| entity_resolution | ran | 60.872 | 9.8% | 9 | 72 | 40666/26782 | claude-opus-4-8 | 0 | $0.1896 | 53/53 entities resolved (14 biomapper, 35 fuzzy, 4 exact) |
| literature_grounding | ran | 59.49 | 9.6% | 0 | 0 | 0/0 | – | 0 | est. n/a | 36 papers, 7 from OpenAlex, 10 from Exa across 14/44 hypotheses |
| direct_kg | ran | 41.035 | 6.6% | 0 | 0 | 0/0 | – | 0 | est. n/a | 113 disease associations, 88 pathways, 466 findings |
| triage | ran | 38.965 | 6.3% | 0 | 0 | 0/0 | – | 0 | est. n/a | 3 well-characterized, 6 moderate, 31 sparse, 13 cold-start |
| intake | ran | 0.002 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 53 entities extracted, query_type=discovery |
| hypothesis_extraction | ran | 0.0 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| bridge_grounding | ran | 0.0 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| temporal | skipped | – | – | – | – | – | – | – | – | – |

### Context management

> Synthesis compresses the accumulated evidence into a bounded context; elision at module scale is expected, and it is the compression that keeps the assembled context within the model's token window.

- **Context:** 91841 chars (~26240 est. tokens) · 26.2% of the char budget (350000) · 13.1% of the 200K-token window
- **Mode:** module-aware aggregation (41 distinct entities; threshold 5)
- **Literature grounding:** 14 of 44 hypotheses carry attached literature

| Section | Shown | Total | Elided |
|---|---|---|---|
| Findings | 131 | 490 | 359 |
| Diseases (recurrence) | 7 | 7 | 0 |
| Pathways (recurrence) | 3 | 3 | 0 |
| Member table | 41 | 41 | 0 |

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