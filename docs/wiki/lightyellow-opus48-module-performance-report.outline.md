# Lightyellow Module Run on Opus 4.8: Pipeline Performance Report (19-analyte, dev, 2026-06-24)

This is the per-node performance report emitted automatically at the end of the full-module **Lightyellow** discovery run on dev (commit `a7d7fd6`). All SDK-backed nodes ran on the Opus 4.8 model; this report is published for direct model comparison against the Sonnet baseline. The Kraken performance-reporter (a terminal graph node) attributes wall-clock, tokens, estimated cost, output counts, and errors across the twelve pipeline nodes, and the Context management section reports how synthesis compresses the accumulated evidence into a bounded context.

## Related

- Discovery analysis from this run: [Lightyellow Module Run on Opus 4.8: Discovery Output](https://phwiki.phenoma.ai/doc/lightyellow-module-run-on-opus-48-discovery-output-19-analyte-dev-2026-06-24-1nxVv5Nmha)
- Model comparison baseline (Sonnet): [Lightyellow Module Run: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/lightyellow-module-run-pipeline-performance-report-19-analyte-dev-2026-06-23-WFgy5WFfxw)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---

## Pipeline Performance Report

- **Run:** `7177ec76bb374a64b08e284dd3f21e1e`
- **Mode:** pipeline · **biomapper_env:** dev
- **Timestamp:** 20260624T181122Z · **commit:** `a7d7fd63deffdbbb25f8b30e8dee40cefdf6c3cd`
- **Wall-clock:** 725.641s · **tokens:** 19 in / 23201 out (134265 cache-r, 135504 cache-w) · **est. cost:** $1.4942
- **Top bottleneck:** integration · **Top cost:** cold_start

> Wall-clock is estimated as the sum of per-node durations, because no measured elapsed time was available; for concurrent branches this sum overstates true elapsed time. The percentage column reports each node's share of that summed total.

### Per-node

| Node | Status | Duration (s) | % | In | Out | Cache r/w | Model | MCP | Est. $ | Findings |
|---|---|---|---|---|---|---|---|---|---|---|
| integration | ran | 191.944 | 26.5% | 3 | 4030 | 26935/33960 | claude-opus-4-8 | 0 | $0.3265 | 20 bridges, 16 gap entities |
| synthesis | ran | 143.677 | 19.8% | 3 | 6639 | 21466/38343 | claude-opus-4-8 | 0 | $0.4164 | 92 hypotheses generated, 26454 char report |
| cold_start | ran | 112.242 | 15.5% | 10 | 9831 | 85864/30385 | claude-opus-4-8 | 0 | $0.4787 | 18 analogues, 35 inferred associations, 37 findings |
| pathway_enrichment | ran | 84.679 | 11.7% | 3 | 2701 | 0/32816 | claude-opus-4-8 | 0 | $0.2726 | 0 shared neighbors (0 non-hub), 4 biological themes |
| literature_grounding | ran | 60.95 | 8.4% | 0 | 0 | 0/0 | – | 0 | est. n/a | 37 papers, 37 from Exa across 15/92 hypotheses |
| bridge_grounding | ran | 53.109 | 7.3% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| direct_kg | ran | 45.609 | 6.3% | 0 | 0 | 0/0 | – | 0 | est. n/a | 333 disease associations, 276 pathways, 1015 findings, 1 hub flags |
| entity_resolution | ran | 17.128 | 2.4% | 0 | 0 | 0/0 | – | 0 | est. n/a | 19/19 entities resolved (17 biomapper, 2 fuzzy) |
| triage | ran | 16.301 | 2.2% | 0 | 0 | 0/0 | – | 0 | est. n/a | 9 well-characterized, 3 moderate, 6 sparse, 1 cold-start |
| intake | ran | 0.001 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 19 entities extracted, query_type=discovery |
| hypothesis_extraction | ran | 0.0 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| temporal | skipped | – | – | – | – | – | – | – | – | – |

### Context management

> Synthesis compresses the accumulated evidence into a bounded context; elision at module scale is expected, and it is the compression that keeps the assembled context within the model's token window.

- **Context:** 107023 chars (~30578 est. tokens) · 30.6% of the char budget (350000) · 15.3% of the 200K-token window
- **Mode:** module-aware aggregation (19 distinct entities; threshold 5)
- **Literature grounding:** 15 of 92 hypotheses carry attached literature

| Section | Shown | Total | Elided |
|---|---|---|---|
| Findings | 147 | 1052 | 905 |
| Diseases (recurrence) | 30 | 58 | 28 |
| Pathways (recurrence) | 23 | 23 | 0 |
| Member table | 19 | 19 | 0 |

### Errors

- SDK inference timed out for CHEBI:88956 after 60s
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal