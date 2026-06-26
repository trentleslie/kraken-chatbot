# Turquoise Module Run on Opus 4.8: Pipeline Performance Report (107-analyte, dev, 2026-06-24)

This is the per-node performance report emitted automatically at the end of the full-module **Turquoise** discovery run on dev (commit `098093e`). All SDK-backed nodes ran on the Opus 4.8 model; this report is published for direct model comparison against the Sonnet baseline. The Kraken performance-reporter (a terminal graph node) attributes wall-clock, tokens, estimated cost, output counts, and errors across the twelve pipeline nodes, and the Context management section reports how synthesis compresses the accumulated evidence into a bounded context.

## Related

- Discovery analysis from this run: [Turquoise Module Run on Opus 4.8: Discovery Output](https://phwiki.phenoma.ai/doc/turquoise-module-run-on-opus-48-discovery-output-107-analyte-dev-2026-06-24-dOkMVtry8A)
- Model comparison baseline (Sonnet): [Turquoise Module Run: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/turquoise-module-run-pipeline-performance-report-107-analyte-dev-2026-06-23-HE1isJ8kuw)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---

## Pipeline Performance Report

- **Run:** `8e69a871ae994d31933c953a1a5a2745`
- **Mode:** pipeline · **biomapper_env:** dev
- **Timestamp:** 20260624T012759Z · **commit:** `098093ea146164cb1ed09db4b8dcc4e8fb1c39a9`
- **Wall-clock:** 1284.09s · **tokens:** 21 in / 23745 out (133229 cache-r, 172464 cache-w) · **est. cost:** $1.7382
- **Top bottleneck:** pathway_enrichment · **Top cost:** synthesis

> Wall-clock is estimated as the sum of per-node durations, because no measured elapsed time was available; for concurrent branches this sum overstates true elapsed time. The percentage column reports each node's share of that summed total.

### Per-node

| Node | Status | Duration (s) | % | In | Out | Cache r/w | Model | MCP | Est. $ | Findings |
|---|---|---|---|---|---|---|---|---|---|---|
| pathway_enrichment | ran | 301.075 | 23.4% | 3 | 5841 | 0/52979 | claude-opus-4-8 | 0 | $0.4772 | 40 shared neighbors (21 non-hub), 8 biological themes |
| cold_start | ran | 196.572 | 15.3% | 12 | 4253 | 85864/50970 | claude-opus-4-8 | 0 | $0.4679 | 18 analogues, 6 inferred associations, 10 findings |
| synthesis | ran | 190.02 | 14.8% | 3 | 8582 | 0/62955 | claude-opus-4-8 | 0 | $0.6080 | 38 hypotheses generated, 29828 char report |
| integration | ran | 147.807 | 11.5% | 3 | 5069 | 47365/5560 | claude-opus-4-8 | 0 | $0.1852 | 22 bridges, 17 gap entities |
| direct_kg | ran | 122.677 | 9.6% | 0 | 0 | 0/0 | – | 0 | est. n/a | 1166 disease associations, 1082 pathways, 3441 findings, 10 hub flags |
| triage | ran | 94.895 | 7.4% | 0 | 0 | 0/0 | – | 0 | est. n/a | 77 well-characterized, 14 moderate, 14 sparse, 1 cold-start |
| entity_resolution | ran | 92.35 | 7.2% | 0 | 0 | 0/0 | – | 0 | est. n/a | 106/106 entities resolved (99 biomapper, 7 fuzzy) |
| bridge_grounding | ran | 77.158 | 6.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| literature_grounding | ran | 61.535 | 4.8% | 0 | 0 | 0/0 | – | 0 | est. n/a | 23 papers, 6 from OpenAlex, 17 from Exa across 15/38 hypotheses |
| intake | ran | 0.002 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 106 entities extracted, query_type=discovery |
| hypothesis_extraction | ran | 0.0 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| temporal | skipped | – | – | – | – | – | – | – | – | – |

### Context management

> Synthesis compresses the accumulated evidence into a bounded context; elision at module scale is expected, and it is the compression that keeps the assembled context within the model's token window.

- **Context:** 105460 chars (~30131 est. tokens) · 30.1% of the char budget (350000) · 15.1% of the 200K-token window
- **Mode:** module-aware aggregation (106 distinct entities; threshold 5)
- **Literature grounding:** 15 of 38 hypotheses carry attached literature

| Section | Shown | Total | Elided |
|---|---|---|---|
| Findings | 120 | 3451 | 3331 |
| Diseases (recurrence) | 30 | 169 | 139 |
| Pathways (recurrence) | 30 | 168 | 138 |
| Member table | 50 | 106 | 56 |

### Errors

None.