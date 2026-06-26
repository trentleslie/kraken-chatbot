# Green Module Run on Opus 4.8: Pipeline Performance Report (83-analyte, dev, 2026-06-24)

This is the per-node performance report emitted automatically at the end of the full-module **Green** discovery run on dev (commit `098093e`). All SDK-backed nodes ran on the Opus 4.8 model; this report is published for direct model comparison against the Sonnet baseline. The Kraken performance-reporter (a terminal graph node) attributes wall-clock, tokens, estimated cost, output counts, and errors across the twelve pipeline nodes, and the Context management section reports how synthesis compresses the accumulated evidence into a bounded context.

## Related

- Discovery analysis from this run: [Green Module Run on Opus 4.8: Discovery Output](https://phwiki.phenoma.ai/doc/green-module-run-on-opus-48-discovery-output-83-analyte-dev-2026-06-24-jHR3fmEpRH)
- Model comparison baseline (Sonnet): [Green Module Run: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/green-module-run-pipeline-performance-report-83-analyte-dev-2026-06-23-6H1guU0r1I)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---

## Pipeline Performance Report

- **Run:** `ba46e947a1404e09b7ad1c3d040244b1`
- **Mode:** pipeline · **biomapper_env:** dev
- **Timestamp:** 20260624T015320Z · **commit:** `098093ea146164cb1ed09db4b8dcc4e8fb1c39a9`
- **Wall-clock:** 718.387s · **tokens:** 27 in / 23721 out (171728 cache-r, 159006 cache-w) · **est. cost:** $1.6728
- **Top bottleneck:** pathway_enrichment · **Top cost:** cold_start

> Wall-clock is estimated as the sum of per-node durations, because no measured elapsed time was available; for concurrent branches this sum overstates true elapsed time. The percentage column reports each node's share of that summed total.

### Per-node

| Node | Status | Duration (s) | % | In | Out | Cache r/w | Model | MCP | Est. $ | Findings |
|---|---|---|---|---|---|---|---|---|---|---|
| pathway_enrichment | ran | 143.399 | 20.0% | 3 | 7651 | 21466/16182 | claude-opus-4-8 | 0 | $0.3032 | 0 shared neighbors (0 non-hub), 3 biological themes |
| synthesis | ran | 136.032 | 18.9% | 3 | 5852 | 21466/37155 | claude-opus-4-8 | 0 | $0.3893 | 49 hypotheses generated, 24889 char report |
| cold_start | ran | 122.87 | 17.1% | 16 | 7818 | 107330/76354 | claude-opus-4-8 | 0 | $0.7264 | 24 analogues, 18 inferred associations, 21 findings |
| direct_kg | ran | 76.357 | 10.6% | 0 | 0 | 0/0 | – | 0 | est. n/a | 127 disease associations, 180 pathways, 1169 findings |
| entity_resolution | ran | 62.968 | 8.8% | 3 | 468 | 0/23265 | claude-opus-4-8 | 0 | $0.1571 | 56/56 entities resolved (13 biomapper, 42 fuzzy, 1 semantic) |
| literature_grounding | ran | 60.147 | 8.4% | 0 | 0 | 0/0 | – | 0 | est. n/a | 30 papers, 16 from OpenAlex, 12 from Exa across 11/49 hypotheses |
| integration | ran | 48.231 | 6.7% | 2 | 1932 | 21466/6050 | claude-opus-4-8 | 0 | $0.0969 | 10 bridges, 10 gap entities |
| triage | ran | 40.245 | 5.6% | 0 | 0 | 0/0 | – | 0 | est. n/a | 6 well-characterized, 24 moderate, 19 sparse, 7 cold-start |
| bridge_grounding | ran | 28.135 | 3.9% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| intake | ran | 0.003 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 56 entities extracted, query_type=discovery |
| hypothesis_extraction | ran | 0.0 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| temporal | skipped | – | – | – | – | – | – | – | – | – |

### Context management

> Synthesis compresses the accumulated evidence into a bounded context; elision at module scale is expected, and it is the compression that keeps the assembled context within the model's token window.

- **Context:** 97695 chars (~27913 est. tokens) · 27.9% of the char budget (350000) · 14.0% of the 200K-token window
- **Mode:** module-aware aggregation (45 distinct entities; threshold 5)
- **Literature grounding:** 11 of 49 hypotheses carry attached literature

| Section | Shown | Total | Elided |
|---|---|---|---|
| Findings | 129 | 1190 | 1061 |
| Diseases (recurrence) | 9 | 9 | 0 |
| Pathways (recurrence) | 0 | 0 | 0 |
| Member table | 45 | 45 | 0 |

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
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'
- Error parsing shared neighbor: '>' not supported between instances of 'NoneType' and 'int'