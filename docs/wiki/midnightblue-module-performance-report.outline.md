# Midnightblue Module Run: Pipeline Performance Report (76-analyte, dev, 2026-06-23)

This is the per-node performance report emitted automatically at the end of the full-module **Midnightblue** discovery run on dev (commit `813fc3f`). The Kraken performance-reporter (a terminal graph node) attributes wall-clock, tokens, estimated cost, output counts, and errors across the twelve pipeline nodes, and the Context management section reports how synthesis compresses the accumulated evidence into a bounded context.

## Related

- Discovery analysis from this run: [Midnightblue Module Run: Discovery Output](https://phwiki.phenoma.ai/doc/midnightblue-module-run-discovery-output-76-analyte-dev-2026-06-23-sVfv9adUa0)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---

## Pipeline Performance Report

- **Run:** `0af38ea593e54c2e8094b7a768586c9d`
- **Mode:** pipeline · **biomapper_env:** dev
- **Timestamp:** 20260623T172511Z · **commit:** `813fc3fb835aed414fc314f0942524a48ca6fbb9`
- **Wall-clock:** 931.15s · **tokens:** 23 in / 27314 out (107330 cache-r, 186624 cache-w) · **est. cost:** $1.1418
- **Top bottleneck:** pathway_enrichment · **Top cost:** cold_start

> Wall-clock is estimated as the sum of per-node durations, because no measured elapsed time was available; for concurrent branches this sum overstates true elapsed time. The percentage column reports each node's share of that summed total.

### Per-node

| Node | Status | Duration (s) | % | In | Out | Cache r/w | Model | MCP | Est. $ | Findings |
|---|---|---|---|---|---|---|---|---|---|---|
| pathway_enrichment | ran | 189.853 | 20.4% | 3 | 10468 | 0/48561 | anthropic/claude-sonnet-4-20250514 | 0 | $0.3391 | 76 shared neighbors (62 non-hub), 9 biological themes |
| synthesis | ran | 166.322 | 17.9% | 3 | 7539 | 21466/37000 | anthropic/claude-sonnet-4-20250514 | 0 | $0.2583 | 49 hypotheses generated, 29864 char report |
| cold_start | ran | 159.277 | 17.1% | 12 | 5651 | 64398/73104 | anthropic/claude-sonnet-4-20250514 | 0 | $0.3783 | 18 analogues, 17 inferred associations, 22 findings |
| direct_kg | ran | 100.196 | 10.8% | 0 | 0 | 0/0 | – | 0 | est. n/a | 465 disease associations, 386 pathways, 1593 findings |
| entity_resolution | ran | 83.582 | 9.0% | 3 | 25 | 0/22636 | anthropic/claude-sonnet-4-20250514 | 0 | $0.0853 | 76/76 entities resolved (51 biomapper, 25 fuzzy) |
| integration | ran | 80.622 | 8.7% | 2 | 3631 | 21466/5323 | anthropic/claude-sonnet-4-20250514 | 0 | $0.0809 | 10 bridges, 15 gap entities |
| literature_grounding | ran | 58.729 | 6.3% | 0 | 0 | 0/0 | – | 0 | est. n/a | 23 papers, 23 from Exa across 15/49 hypotheses |
| triage | ran | 57.199 | 6.1% | 0 | 0 | 0/0 | – | 0 | est. n/a | 15 well-characterized, 22 moderate, 29 sparse, 10 cold-start |
| bridge_grounding | ran | 35.369 | 3.8% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| intake | ran | 0.002 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 76 entities extracted, query_type=discovery |
| hypothesis_extraction | ran | 0.0 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| temporal | skipped | – | – | – | – | – | – | – | – | – |

### Context management

> Synthesis compresses the accumulated evidence into a bounded context; elision at module scale is expected, and it is the compression that keeps the assembled context within the model's token window.

- **Context:** 97277 chars (~27793 est. tokens) · 27.8% of the char budget (350000) · 13.9% of the 200K-token window
- **Mode:** module-aware aggregation (71 distinct entities; threshold 5)
- **Literature grounding:** 15 of 49 hypotheses carry attached literature

| Section | Shown | Total | Elided |
|---|---|---|---|
| Findings | 130 | 1615 | 1485 |
| Diseases (recurrence) | 30 | 71 | 41 |
| Pathways (recurrence) | 30 | 39 | 9 |
| Member table | 50 | 71 | 21 |

### Errors

- Exception analyzing CHEBI:25982: 1 validation error for AnalogueEntity similarity   Input should be less than or equal to 1 [type=less_than_equal, input_value=1.0002281665802002, input_type=float]     For further information visit https://errors.pydantic.dev/2.12/v/less_than_equal
- Exception analyzing CHEBI:132918: 1 validation error for AnalogueEntity similarity   Input should be less than or equal to 1 [type=less_than_equal, input_value=1.0002069473266602, input_type=float]     For further information visit https://errors.pydantic.dev/2.12/v/less_than_equal