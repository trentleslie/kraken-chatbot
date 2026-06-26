# Grey60 Module Run on Opus 4.8: Pipeline Performance Report (32-analyte, dev, 2026-06-24)

This is the per-node performance report emitted automatically at the end of the full-module **Grey60** discovery run on dev (commit `a7d7fd6`). All SDK-backed nodes ran on the Opus 4.8 model; this report is published for direct model comparison against the Sonnet baseline. The Kraken performance-reporter (a terminal graph node) attributes wall-clock, tokens, estimated cost, output counts, and errors across the twelve pipeline nodes, and the Context management section reports how synthesis compresses the accumulated evidence into a bounded context.

## Related

- Discovery analysis from this run: [Grey60 Module Run on Opus 4.8: Discovery Output](https://phwiki.phenoma.ai/doc/grey60-module-run-on-opus-48-discovery-output-32-analyte-dev-2026-06-24-BjOIDVTfbs)
- Model comparison baseline (Sonnet): [Grey60 Module Run: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/grey60-module-run-pipeline-performance-report-32-analyte-dev-2026-06-23-xpx5udQQOy)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---

## Pipeline Performance Report

- **Run:** `694406121c3046f9807bbd12ba878ed3`
- **Mode:** pipeline · **biomapper_env:** dev
- **Timestamp:** 20260624T175004Z · **commit:** `a7d7fd63deffdbbb25f8b30e8dee40cefdf6c3cd`
- **Wall-clock:** 496.584s · **tokens:** 24 in / 17370 out (64398 cache-r, 217547 cache-w) · **est. cost:** $1.8262
- **Top bottleneck:** synthesis · **Top cost:** cold_start

> Wall-clock is estimated as the sum of per-node durations, because no measured elapsed time was available; for concurrent branches this sum overstates true elapsed time. The percentage column reports each node's share of that summed total.

### Per-node

| Node | Status | Duration (s) | % | In | Out | Cache r/w | Model | MCP | Est. $ | Findings |
|---|---|---|---|---|---|---|---|---|---|---|
| synthesis | ran | 135.017 | 27.2% | 3 | 6152 | 0/45316 | claude-opus-4-8 | 0 | $0.4370 | 44 hypotheses generated, 24340 char report |
| cold_start | ran | 109.384 | 22.0% | 16 | 6780 | 64398/118709 | claude-opus-4-8 | 0 | $0.9437 | 24 analogues, 20 inferred associations, 24 findings |
| literature_grounding | ran | 65.787 | 13.2% | 0 | 0 | 0/0 | – | 0 | est. n/a | 13 papers, 13 from Exa across 10/44 hypotheses |
| pathway_enrichment | ran | 54.616 | 11.0% | 3 | 2163 | 0/27318 | claude-opus-4-8 | 0 | $0.2248 | 0 shared neighbors (0 non-hub), 3 biological themes |
| integration | ran | 48.172 | 9.7% | 2 | 2275 | 0/26204 | claude-opus-4-8 | 0 | $0.2207 | 0 bridges, 11 gap entities |
| entity_resolution | ran | 36.056 | 7.3% | 0 | 0 | 0/0 | – | 0 | est. n/a | 33/33 entities resolved (7 fuzzy, 25 biomapper, 1 exact) |
| triage | ran | 25.521 | 5.1% | 0 | 0 | 0/0 | – | 0 | est. n/a | 3 well-characterized, 3 moderate, 21 sparse, 6 cold-start |
| direct_kg | ran | 22.03 | 4.4% | 0 | 0 | 0/0 | – | 0 | est. n/a | 91 disease associations, 10 pathways, 263 findings |
| intake | ran | 0.002 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 33 entities extracted, query_type=discovery |
| hypothesis_extraction | ran | 0.0 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| bridge_grounding | ran | 0.0 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| temporal | skipped | – | – | – | – | – | – | – | – | – |

### Context management

> Synthesis compresses the accumulated evidence into a bounded context; elision at module scale is expected, and it is the compression that keeps the assembled context within the model's token window.

- **Context:** 63096 chars (~18027 est. tokens) · 18.0% of the char budget (350000) · 9.0% of the 200K-token window
- **Mode:** module-aware aggregation (33 distinct entities; threshold 5)
- **Literature grounding:** 10 of 44 hypotheses carry attached literature

| Section | Shown | Total | Elided |
|---|---|---|---|
| Findings | 129 | 287 | 158 |
| Diseases (recurrence) | 5 | 5 | 0 |
| Pathways (recurrence) | 0 | 0 | 0 |
| Member table | 33 | 33 | 0 |

### Errors

- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: 1 validation error for SharedNeighbor degree   Input should be a valid integer [type=int_type, input_value=None, input_type=NoneType]     For further information visit https://errors.pydantic.dev/2.12/v/int_type
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'