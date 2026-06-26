# Purple Module Run on Opus 4.8: Pipeline Performance Report (51-analyte, dev, 2026-06-24)

This is the per-node performance report emitted automatically at the end of the full-module **Purple** discovery run on dev (commit `098093e`). All SDK-backed nodes ran on the Opus 4.8 model; this report is published for direct model comparison against the Sonnet baseline. The Kraken performance-reporter (a terminal graph node) attributes wall-clock, tokens, estimated cost, output counts, and errors across the twelve pipeline nodes, and the Context management section reports how synthesis compresses the accumulated evidence into a bounded context.

## Related

- Discovery analysis from this run: [Purple Module Run on Opus 4.8: Discovery Output](https://phwiki.phenoma.ai/doc/purple-module-run-on-opus-48-discovery-output-51-analyte-dev-2026-06-24-yj8TkaGYGM)
- Model comparison baseline (Sonnet): [Purple Module Run: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/purple-module-run-pipeline-performance-report-51-analyte-dev-2026-06-23-CBbswHQIxY)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---

## Pipeline Performance Report

- **Run:** `ed391692d4c14bd89060f7bfff603a41`
- **Mode:** pipeline · **biomapper_env:** dev
- **Timestamp:** 20260624T024340Z · **commit:** `098093ea146164cb1ed09db4b8dcc4e8fb1c39a9`
- **Wall-clock:** 803.67s · **tokens:** 18 in / 20521 out (69322 cache-r, 171531 cache-w) · **est. cost:** $1.6198
- **Top bottleneck:** integration · **Top cost:** synthesis

> Wall-clock is estimated as the sum of per-node durations, because no measured elapsed time was available; for concurrent branches this sum overstates true elapsed time. The percentage column reports each node's share of that summed total.

### Per-node

| Node | Status | Duration (s) | % | In | Out | Cache r/w | Model | MCP | Est. $ | Findings |
|---|---|---|---|---|---|---|---|---|---|---|
| integration | ran | 190.954 | 23.8% | 3 | 3758 | 47856/10657 | claude-opus-4-8 | 0 | $0.1845 | 10 bridges, 15 gap entities |
| synthesis | ran | 139.327 | 17.3% | 3 | 6352 | 0/50405 | claude-opus-4-8 | 0 | $0.4738 | 40 hypotheses generated, 23201 char report |
| pathway_enrichment | ran | 136.918 | 17.0% | 3 | 6174 | 0/39350 | claude-opus-4-8 | 0 | $0.4003 | 25 shared neighbors (14 non-hub), 3 biological themes |
| cold_start | ran | 116.914 | 14.5% | 6 | 4213 | 21466/47835 | claude-opus-4-8 | 0 | $0.4151 | 9 analogues, 12 inferred associations, 18 findings |
| literature_grounding | ran | 63.213 | 7.9% | 0 | 0 | 0/0 | – | 0 | est. n/a | 17 papers, 17 from Exa across 11/40 hypotheses |
| direct_kg | ran | 42.282 | 5.3% | 0 | 0 | 0/0 | – | 0 | est. n/a | 96 disease associations, 70 pathways, 659 findings, 2 hub flags |
| entity_resolution | ran | 41.269 | 5.1% | 3 | 24 | 0/23284 | claude-opus-4-8 | 0 | $0.1461 | 47/47 entities resolved (4 biomapper, 42 fuzzy, 1 exact) |
| bridge_grounding | ran | 37.391 | 4.7% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| triage | ran | 35.399 | 4.4% | 0 | 0 | 0/0 | – | 0 | est. n/a | 4 well-characterized, 30 moderate, 11 sparse, 2 cold-start |
| intake | ran | 0.002 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 47 entities extracted, query_type=discovery |
| hypothesis_extraction | ran | 0.0 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| temporal | skipped | – | – | – | – | – | – | – | – | – |

### Context management

> Synthesis compresses the accumulated evidence into a bounded context; elision at module scale is expected, and it is the compression that keeps the assembled context within the model's token window.

- **Context:** 73142 chars (~20898 est. tokens) · 20.9% of the char budget (350000) · 10.4% of the 200K-token window
- **Mode:** module-aware aggregation (35 distinct entities; threshold 5)
- **Literature grounding:** 11 of 40 hypotheses carry attached literature

| Section | Shown | Total | Elided |
|---|---|---|---|
| Findings | 128 | 677 | 549 |
| Diseases (recurrence) | 14 | 14 | 0 |
| Pathways (recurrence) | 4 | 4 | 0 |
| Member table | 35 | 35 | 0 |

### Errors

None.