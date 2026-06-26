# Blue Module Run on Opus 4.8: Pipeline Performance Report (149-analyte, dev, 2026-06-24)

This is the per-node performance report emitted automatically at the end of the full-module **Blue** discovery run on dev (commit `098093e`). All SDK-backed nodes ran on the Opus 4.8 model; this report is published for direct model comparison against the Sonnet baseline. The Kraken performance-reporter (a terminal graph node) attributes wall-clock, tokens, estimated cost, output counts, and errors across the twelve pipeline nodes, and the Context management section reports how synthesis compresses the accumulated evidence into a bounded context.

## Related

- Discovery analysis from this run: [Blue Module Run on Opus 4.8: Discovery Output](https://phwiki.phenoma.ai/doc/blue-module-run-on-opus-48-discovery-output-149-analyte-dev-2026-06-24-w2zZRNlC3t)
- Model comparison baseline (Sonnet): [Blue Module Run: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/blue-module-run-pipeline-performance-report-149-analyte-dev-2026-06-23-Yh2cTvs6GD)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---

## Pipeline Performance Report

- **Run:** `24b83ce0b7c9411ca7969fc73628fb87`
- **Mode:** pipeline · **biomapper_env:** dev
- **Timestamp:** 20260624T010826Z · **commit:** `098093ea146164cb1ed09db4b8dcc4e8fb1c39a9`
- **Wall-clock:** 1427.45s · **tokens:** 23 in / 24431 out (110773 cache-r, 241908 cache-w) · **est. cost:** $2.1782
- **Top bottleneck:** pathway_enrichment · **Top cost:** cold_start

> Wall-clock is estimated as the sum of per-node durations, because no measured elapsed time was available; for concurrent branches this sum overstates true elapsed time. The percentage column reports each node's share of that summed total.

### Per-node

| Node | Status | Duration (s) | % | In | Out | Cache r/w | Model | MCP | Est. $ | Findings |
|---|---|---|---|---|---|---|---|---|---|---|
| pathway_enrichment | ran | 457.841 | 32.1% | 3 | 11030 | 21466/48193 | claude-opus-4-8 | 0 | $0.5877 | 78 shared neighbors (61 non-hub), 10 biological themes |
| cold_start | ran | 173.375 | 12.1% | 14 | 2263 | 64398/94547 | claude-opus-4-8 | 0 | $0.6798 | 21 analogues, 1 inferred associations, 8 findings |
| synthesis | ran | 164.975 | 11.6% | 3 | 7379 | 0/70589 | claude-opus-4-8 | 0 | $0.6257 | 29 hypotheses generated, 26892 char report |
| integration | ran | 144.281 | 10.1% | 3 | 3759 | 24909/28579 | claude-opus-4-8 | 0 | $0.2851 | 20 bridges, 17 gap entities |
| direct_kg | ran | 131.385 | 9.2% | 0 | 0 | 0/0 | – | 0 | est. n/a | 1210 disease associations, 1071 pathways, 3696 findings, 13 hub flags |
| triage | ran | 129.507 | 9.1% | 0 | 0 | 0/0 | – | 0 | est. n/a | 95 well-characterized, 24 moderate, 24 sparse, 3 cold-start |
| entity_resolution | ran | 125.195 | 8.8% | 0 | 0 | 0/0 | – | 0 | est. n/a | 146/146 entities resolved (135 biomapper, 2 exact, 9 fuzzy) |
| literature_grounding | ran | 59.942 | 4.2% | 0 | 0 | 0/0 | – | 0 | est. n/a | 38 papers, 26 from OpenAlex, 12 from Exa across 15/29 hypotheses |
| bridge_grounding | ran | 35.541 | 2.5% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| hypothesis_extraction | ran | 5.404 | 0.4% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| intake | ran | 0.003 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 146 entities extracted, query_type=discovery |
| temporal | skipped | – | – | – | – | – | – | – | – | – |

### Context management

> Synthesis compresses the accumulated evidence into a bounded context; elision at module scale is expected, and it is the compression that keeps the assembled context within the model's token window.

- **Context:** 133384 chars (~38110 est. tokens) · 38.1% of the char budget (350000) · 19.1% of the 200K-token window
- **Mode:** module-aware aggregation (146 distinct entities; threshold 5)
- **Literature grounding:** 15 of 29 hypotheses carry attached literature

| Section | Shown | Total | Elided |
|---|---|---|---|
| Findings | 125 | 3704 | 3579 |
| Diseases (recurrence) | 30 | 157 | 127 |
| Pathways (recurrence) | 30 | 178 | 148 |
| Member table | 50 | 146 | 96 |

### Errors

- Exception analyzing CHEBI:132918: 1 validation error for AnalogueEntity similarity   Input should be less than or equal to 1 [type=less_than_equal, input_value=1.0002069473266602, input_type=float]     For further information visit https://errors.pydantic.dev/2.12/v/less_than_equal