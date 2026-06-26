# Pink Module Run: Pipeline Performance Report (54-analyte, dev, 2026-06-23)

This is the per-node performance report emitted automatically at the end of the full-module **Pink** discovery run on dev (commit `813fc3f`). The Kraken performance-reporter (a terminal graph node) attributes wall-clock, tokens, estimated cost, output counts, and errors across the twelve pipeline nodes, and the Context management section reports how synthesis compresses the accumulated evidence into a bounded context.

## Related

- Discovery analysis from this run: [Pink Module Run: Discovery Output](https://phwiki.phenoma.ai/doc/pink-module-run-discovery-output-54-analyte-dev-2026-06-23-2wJf2HBLjK)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---

## Pipeline Performance Report

- **Run:** `2c4ac10bf2c04abbb617e8b45c68a528`
- **Mode:** pipeline · **biomapper_env:** dev
- **Timestamp:** 20260623T173655Z · **commit:** `813fc3fb835aed414fc314f0942524a48ca6fbb9`
- **Wall-clock:** 721.486s · **tokens:** 34 in / 25549 out (155746 cache-r, 243482 cache-w) · **est. cost:** $1.3431
- **Top bottleneck:** integration · **Top cost:** cold_start

> Wall-clock is estimated as the sum of per-node durations, because no measured elapsed time was available; for concurrent branches this sum overstates true elapsed time. The percentage column reports each node's share of that summed total.

### Per-node

| Node | Status | Duration (s) | % | In | Out | Cache r/w | Model | MCP | Est. $ | Findings |
|---|---|---|---|---|---|---|---|---|---|---|
| integration | ran | 176.248 | 24.4% | 3 | 3904 | 48416/11558 | anthropic/claude-sonnet-4-20250514 | 0 | $0.1164 | 0 bridges, 12 gap entities |
| synthesis | ran | 150.883 | 20.9% | 3 | 6543 | 0/54829 | anthropic/claude-sonnet-4-20250514 | 0 | $0.3038 | 51 hypotheses generated, 24173 char report |
| cold_start | ran | 109.297 | 15.1% | 16 | 10161 | 64398/120571 | anthropic/claude-sonnet-4-20250514 | 0 | $0.6239 | 24 analogues, 24 inferred associations, 27 findings |
| pathway_enrichment | ran | 85.338 | 11.8% | 3 | 4868 | 0/29631 | anthropic/claude-sonnet-4-20250514 | 0 | $0.1841 | 0 shared neighbors (0 non-hub), 3 biological themes |
| entity_resolution | ran | 71.308 | 9.9% | 9 | 73 | 42932/26893 | anthropic/claude-sonnet-4-20250514 | 0 | $0.1149 | 53/53 entities resolved (14 biomapper, 35 fuzzy, 4 exact) |
| literature_grounding | ran | 60.796 | 8.4% | 0 | 0 | 0/0 | – | 0 | est. n/a | 20 papers, 14 from Exa across 11/51 hypotheses |
| triage | ran | 38.993 | 5.4% | 0 | 0 | 0/0 | – | 0 | est. n/a | 3 well-characterized, 6 moderate, 30 sparse, 14 cold-start |
| direct_kg | ran | 28.621 | 4.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 113 disease associations, 88 pathways, 469 findings |
| intake | ran | 0.002 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 53 entities extracted, query_type=discovery |
| hypothesis_extraction | ran | 0.0 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| bridge_grounding | ran | 0.0 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| temporal | skipped | – | – | – | – | – | – | – | – | – |

### Context management

> Synthesis compresses the accumulated evidence into a bounded context; elision at module scale is expected, and it is the compression that keeps the assembled context within the model's token window.

- **Context:** 91769 chars (~26220 est. tokens) · 26.2% of the char budget (350000) · 13.1% of the 200K-token window
- **Mode:** module-aware aggregation (41 distinct entities; threshold 5)
- **Literature grounding:** 11 of 51 hypotheses carry attached literature

| Section | Shown | Total | Elided |
|---|---|---|---|
| Findings | 137 | 496 | 359 |
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