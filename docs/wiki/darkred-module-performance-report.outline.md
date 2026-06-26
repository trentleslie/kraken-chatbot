# Darkred Module Run: Pipeline Performance Report (26-analyte, dev, 2026-06-23)

This is the per-node performance report emitted automatically at the end of the full-module **Darkred** discovery run on dev (commit `813fc3f`). The Kraken performance-reporter (a terminal graph node) attributes wall-clock, tokens, estimated cost, output counts, and errors across the twelve pipeline nodes, and the Context management section reports how synthesis compresses the accumulated evidence into a bounded context.

## Related

- Discovery analysis from this run: [Darkred Module Run: Discovery Output](https://phwiki.phenoma.ai/doc/darkred-module-run-discovery-output-26-analyte-dev-2026-06-23-z1Qvn5R3I1)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---

## Pipeline Performance Report

- **Run:** `8895633dcb544f74b1997b8b16e51789`
- **Mode:** pipeline · **biomapper_env:** dev
- **Timestamp:** 20260623T193616Z · **commit:** `813fc3fb835aed414fc314f0942524a48ca6fbb9`
- **Wall-clock:** 909.645s · **tokens:** 23 in / 38152 out (177168 cache-r, 132652 cache-w) · **est. cost:** $1.1229
- **Top bottleneck:** pathway_enrichment · **Top cost:** pathway_enrichment

> Wall-clock is estimated as the sum of per-node durations, because no measured elapsed time was available; for concurrent branches this sum overstates true elapsed time. The percentage column reports each node's share of that summed total.

### Per-node

| Node | Status | Duration (s) | % | In | Out | Cache r/w | Model | MCP | Est. $ | Findings |
|---|---|---|---|---|---|---|---|---|---|---|
| pathway_enrichment | ran | 202.74 | 22.3% | 3 | 17492 | 21466/12870 | anthropic/claude-sonnet-4-20250514 | 0 | $0.3171 | 0 shared neighbors (0 non-hub), 0 biological themes |
| integration | ran | 197.064 | 21.7% | 3 | 3677 | 26906/33540 | anthropic/claude-sonnet-4-20250514 | 0 | $0.1890 | 16 bridges, 12 gap entities |
| synthesis | ran | 160.997 | 17.7% | 3 | 7219 | 0/54306 | anthropic/claude-sonnet-4-20250514 | 0 | $0.3119 | 76 hypotheses generated, 27023 char report |
| cold_start | ran | 139.648 | 15.4% | 14 | 9764 | 128796/31936 | anthropic/claude-sonnet-4-20250514 | 0 | $0.3049 | 21 analogues, 29 inferred associations, 31 findings |
| direct_kg | ran | 66.044 | 7.3% | 0 | 0 | 0/0 | – | 0 | est. n/a | 178 disease associations, 107 pathways, 644 findings |
| literature_grounding | ran | 63.781 | 7.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 17 papers, 17 from Exa across 15/76 hypotheses |
| bridge_grounding | ran | 29.852 | 3.3% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| entity_resolution | ran | 25.196 | 2.8% | 0 | 0 | 0/0 | – | 0 | est. n/a | 26/26 entities resolved (22 biomapper, 1 exact, 3 fuzzy) |
| triage | ran | 20.906 | 2.3% | 0 | 0 | 0/0 | – | 0 | est. n/a | 5 well-characterized, 11 moderate, 8 sparse, 2 cold-start |
| hypothesis_extraction | ran | 3.416 | 0.4% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| intake | ran | 0.001 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 26 entities extracted, query_type=discovery |
| temporal | skipped | – | – | – | – | – | – | – | – | – |

### Context management

> Synthesis compresses the accumulated evidence into a bounded context; elision at module scale is expected, and it is the compression that keeps the assembled context within the model's token window.

- **Context:** 91799 chars (~26228 est. tokens) · 26.2% of the char budget (350000) · 13.1% of the 200K-token window
- **Mode:** module-aware aggregation (25 distinct entities; threshold 5)
- **Literature grounding:** 15 of 76 hypotheses carry attached literature

| Section | Shown | Total | Elided |
|---|---|---|---|
| Findings | 141 | 675 | 534 |
| Diseases (recurrence) | 21 | 21 | 0 |
| Pathways (recurrence) | 11 | 11 | 0 |
| Member table | 25 | 25 | 0 |

### Errors

- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-1, input_type=int]     For further information visit https://errors.pydantic.dev/2.12/v/greater_than_equal