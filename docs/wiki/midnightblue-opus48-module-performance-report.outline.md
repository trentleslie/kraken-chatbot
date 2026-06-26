# Midnightblue Module Run on Opus 4.8: Pipeline Performance Report (76-analyte, dev, 2026-06-24)

This is the per-node performance report emitted automatically at the end of the full-module **Midnightblue** discovery run on dev (commit `098093e`). All SDK-backed nodes ran on the Opus 4.8 model; this report is published for direct model comparison against the Sonnet baseline. The Kraken performance-reporter (a terminal graph node) attributes wall-clock, tokens, estimated cost, output counts, and errors across the twelve pipeline nodes, and the Context management section reports how synthesis compresses the accumulated evidence into a bounded context.

## Related

- Discovery analysis from this run: [Midnightblue Module Run on Opus 4.8: Discovery Output](https://phwiki.phenoma.ai/doc/midnightblue-module-run-on-opus-48-discovery-output-76-analyte-dev-2026-06-24-teHlsdAuZb)
- Model comparison baseline (Sonnet): [Midnightblue Module Run: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/midnightblue-module-run-pipeline-performance-report-76-analyte-dev-2026-06-23-r0R7cTBAoA)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---

## Pipeline Performance Report

- **Run:** `eb7b849cc9874350bfdab7f5e17425a8`
- **Mode:** pipeline · **biomapper_env:** dev
- **Timestamp:** 20260624T021035Z · **commit:** `098093ea146164cb1ed09db4b8dcc4e8fb1c39a9`
- **Wall-clock:** 1119.414s · **tokens:** 24 in / 24369 out (91027 cache-r, 232704 cache-w) · **est. cost:** $2.1093
- **Top bottleneck:** integration · **Top cost:** cold_start

> Wall-clock is estimated as the sum of per-node durations, because no measured elapsed time was available; for concurrent branches this sum overstates true elapsed time. The percentage column reports each node's share of that summed total.

### Per-node

| Node | Status | Duration (s) | % | In | Out | Cache r/w | Model | MCP | Est. $ | Findings |
|---|---|---|---|---|---|---|---|---|---|---|
| integration | ran | 286.293 | 25.6% | 3 | 4872 | 48095/15556 | claude-opus-4-8 | 0 | $0.2431 | 10 bridges, 16 gap entities |
| synthesis | ran | 176.282 | 15.7% | 3 | 8039 | 0/59431 | claude-opus-4-8 | 0 | $0.5724 | 56 hypotheses generated, 30334 char report |
| cold_start | ran | 165.755 | 14.8% | 12 | 5683 | 21466/116424 | claude-opus-4-8 | 0 | $0.8805 | 18 analogues, 21 inferred associations, 25 findings |
| pathway_enrichment | ran | 155.729 | 13.9% | 3 | 5751 | 0/40123 | claude-opus-4-8 | 0 | $0.3946 | 41 shared neighbors (34 non-hub), 9 biological themes |
| direct_kg | ran | 96.543 | 8.6% | 0 | 0 | 0/0 | – | 0 | est. n/a | 436 disease associations, 340 pathways, 1592 findings |
| entity_resolution | ran | 78.392 | 7.0% | 3 | 24 | 21466/1170 | claude-opus-4-8 | 0 | $0.0187 | 76/76 entities resolved (51 biomapper, 24 fuzzy, 1 exact) |
| triage | ran | 60.153 | 5.4% | 0 | 0 | 0/0 | – | 0 | est. n/a | 15 well-characterized, 22 moderate, 29 sparse, 10 cold-start |
| literature_grounding | ran | 58.792 | 5.3% | 0 | 0 | 0/0 | – | 0 | est. n/a | 31 papers, 12 from OpenAlex, 19 from Exa across 15/56 hypotheses |
| bridge_grounding | ran | 41.472 | 3.7% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| intake | ran | 0.002 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 76 entities extracted, query_type=discovery |
| hypothesis_extraction | ran | 0.0 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| temporal | skipped | – | – | – | – | – | – | – | – | – |

### Context management

> Synthesis compresses the accumulated evidence into a bounded context; elision at module scale is expected, and it is the compression that keeps the assembled context within the model's token window.

- **Context:** 107408 chars (~30688 est. tokens) · 30.7% of the char budget (350000) · 15.3% of the 200K-token window
- **Mode:** module-aware aggregation (71 distinct entities; threshold 5)
- **Literature grounding:** 15 of 56 hypotheses carry attached literature

| Section | Shown | Total | Elided |
|---|---|---|---|
| Findings | 140 | 1617 | 1477 |
| Diseases (recurrence) | 30 | 69 | 39 |
| Pathways (recurrence) | 30 | 37 | 7 |
| Member table | 50 | 71 | 21 |

### Errors

- Exception analyzing CHEBI:25982: 1 validation error for AnalogueEntity similarity   Input should be less than or equal to 1 [type=less_than_equal, input_value=1.0002281665802002, input_type=float]     For further information visit https://errors.pydantic.dev/2.12/v/less_than_equal
- Exception analyzing CHEBI:132918: 1 validation error for AnalogueEntity similarity   Input should be less than or equal to 1 [type=less_than_equal, input_value=1.0002069473266602, input_type=float]     For further information visit https://errors.pydantic.dev/2.12/v/less_than_equal