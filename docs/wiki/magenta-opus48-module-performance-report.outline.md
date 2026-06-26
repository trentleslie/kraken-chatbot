# Magenta Module Run on Opus 4.8: Pipeline Performance Report (47-analyte, dev, 2026-06-24)

This is the per-node performance report emitted automatically at the end of the full-module **Magenta** discovery run on dev (commit `098093e`). All SDK-backed nodes ran on the Opus 4.8 model; this report is published for direct model comparison against the Sonnet baseline. The Kraken performance-reporter (a terminal graph node) attributes wall-clock, tokens, estimated cost, output counts, and errors across the twelve pipeline nodes, and the Context management section reports how synthesis compresses the accumulated evidence into a bounded context.

## Related

- Discovery analysis from this run: [Magenta Module Run on Opus 4.8: Discovery Output](https://phwiki.phenoma.ai/doc/magenta-module-run-on-opus-48-discovery-output-47-analyte-dev-2026-06-24-dubaOsQuxw)
- Model comparison baseline (Sonnet): [Magenta Module Run: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/magenta-module-run-pipeline-performance-report-47-analyte-dev-2026-06-23-fqDYpvYB74)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---

## Pipeline Performance Report

- **Run:** `a1f40b08cc6e4110b4213725f666a984`
- **Mode:** pipeline · **biomapper_env:** dev
- **Timestamp:** 20260624T023046Z · **commit:** `098093ea146164cb1ed09db4b8dcc4e8fb1c39a9`
- **Wall-clock:** 638.068s · **tokens:** 22 in / 21766 out (171728 cache-r, 99958 cache-w) · **est. cost:** $1.2549
- **Top bottleneck:** synthesis · **Top cost:** pathway_enrichment

> Wall-clock is estimated as the sum of per-node durations, because no measured elapsed time was available; for concurrent branches this sum overstates true elapsed time. The percentage column reports each node's share of that summed total.

### Per-node

| Node | Status | Duration (s) | % | In | Out | Cache r/w | Model | MCP | Est. $ | Findings |
|---|---|---|---|---|---|---|---|---|---|---|
| synthesis | ran | 160.746 | 25.2% | 3 | 7393 | 21466/25837 | claude-opus-4-8 | 0 | $0.3571 | 13 hypotheses generated, 30776 char report |
| pathway_enrichment | ran | 160.543 | 25.2% | 3 | 7892 | 0/40013 | claude-opus-4-8 | 0 | $0.4474 | 26 shared neighbors (20 non-hub), 7 biological themes |
| cold_start | ran | 78.607 | 12.3% | 14 | 3434 | 128796/30482 | claude-opus-4-8 | 0 | $0.3408 | 21 analogues, 3 inferred associations, 10 findings |
| integration | ran | 67.024 | 10.5% | 2 | 3047 | 21466/3626 | claude-opus-4-8 | 0 | $0.1096 | 0 bridges, 12 gap entities |
| literature_grounding | ran | 52.367 | 8.2% | 0 | 0 | 0/0 | – | 0 | est. n/a | 18 papers, 3 from OpenAlex, 15 from Exa across 8/13 hypotheses |
| entity_resolution | ran | 42.222 | 6.6% | 0 | 0 | 0/0 | – | 0 | est. n/a | 47/47 entities resolved (35 biomapper, 3 exact, 9 fuzzy) |
| triage | ran | 40.138 | 6.3% | 0 | 0 | 0/0 | – | 0 | est. n/a | 19 well-characterized, 7 moderate, 15 sparse, 6 cold-start |
| direct_kg | ran | 36.419 | 5.7% | 0 | 0 | 0/0 | – | 0 | est. n/a | 243 disease associations, 247 pathways, 797 findings, 2 hub flags |
| intake | ran | 0.002 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 47 entities extracted, query_type=discovery |
| hypothesis_extraction | ran | 0.0 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| bridge_grounding | ran | 0.0 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| temporal | skipped | – | – | – | – | – | – | – | – | – |

### Context management

> Synthesis compresses the accumulated evidence into a bounded context; elision at module scale is expected, and it is the compression that keeps the assembled context within the model's token window.

- **Context:** 67321 chars (~19235 est. tokens) · 19.2% of the char budget (350000) · 9.6% of the 200K-token window
- **Mode:** module-aware aggregation (47 distinct entities; threshold 5)
- **Literature grounding:** 8 of 13 hypotheses carry attached literature

| Section | Shown | Total | Elided |
|---|---|---|---|
| Findings | 115 | 807 | 692 |
| Diseases (recurrence) | 30 | 46 | 16 |
| Pathways (recurrence) | 30 | 32 | 2 |
| Member table | 47 | 47 | 0 |

### Errors

None.