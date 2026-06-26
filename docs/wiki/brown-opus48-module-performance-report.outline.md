# Brown Module Run on Opus 4.8: Pipeline Performance Report (203-analyte, dev, 2026-06-23)

This is the per-node performance report emitted automatically at the end of the full-module **Brown** discovery run on dev (commit `f8ea4c5`). All SDK-backed nodes ran on the Opus 4.8 model; this report is published for direct model comparison against the Sonnet baseline. The Kraken performance-reporter (a terminal graph node) attributes wall-clock, tokens, estimated cost, output counts, and errors across the twelve pipeline nodes, and the Context management section reports how synthesis compresses the accumulated evidence into a bounded context.

## Related

- Discovery analysis from this run: [Brown Module Run on Opus 4.8: Discovery Output](https://phwiki.phenoma.ai/doc/brown-module-run-on-opus-48-discovery-output-203-analyte-dev-2026-06-23-qWTUmlQDLh)
- Model comparison baseline (Sonnet): [Brown Module Run: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/brown-module-full-run-pipeline-performance-report-203-analyte-dev-2026-06-22-IosiN9wigV)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---

## Pipeline Performance Report

- **Run:** `81891ccb63454ab1805e98deb72dc93b`
- **Mode:** pipeline · **biomapper_env:** dev
- **Timestamp:** 20260623T215726Z · **commit:** `f8ea4c5f3fd7d20228e0c9411ec8d3b037bf0951`
- **Wall-clock:** 1370.413s · **tokens:** 24 in / 37442 out (85864 cache-r, 293135 cache-w) · **est. cost:** $2.8112
- **Top bottleneck:** pathway_enrichment · **Top cost:** cold_start

> Wall-clock is estimated as the sum of per-node durations, because no measured elapsed time was available; for concurrent branches this sum overstates true elapsed time. The percentage column reports each node's share of that summed total.

### Per-node

| Node | Status | Duration (s) | % | In | Out | Cache r/w | Model | MCP | Est. $ | Findings |
|---|---|---|---|---|---|---|---|---|---|---|
| pathway_enrichment | ran | 597.812 | 43.6% | 3 | 16467 | 21466/62753 | claude-opus-4-8 | 0 | $0.8146 | 117 shared neighbors (97 non-hub), 10 biological themes |
| cold_start | ran | 175.851 | 12.8% | 16 | 10958 | 42932/141014 | claude-opus-4-8 | 0 | $1.1768 | 24 analogues, 33 inferred associations, 35 findings |
| synthesis | ran | 167.511 | 12.2% | 3 | 7434 | 0/83537 | claude-opus-4-8 | 0 | $0.7080 | 98 hypotheses generated, 27951 char report |
| triage | ran | 150.726 | 11.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 81 well-characterized, 41 moderate, 61 sparse, 11 cold-start |
| direct_kg | ran | 80.696 | 5.9% | 0 | 0 | 0/0 | – | 0 | est. n/a | 1112 disease associations, 953 pathways, 3166 findings, 11 hub flags |
| literature_grounding | ran | 63.991 | 4.7% | 0 | 0 | 0/0 | – | 0 | est. n/a | 32 papers, 32 from Exa across 15/98 hypotheses |
| integration | ran | 60.309 | 4.4% | 2 | 2583 | 21466/5831 | claude-opus-4-8 | 0 | $0.1118 | 30 bridges, 12 gap entities |
| bridge_grounding | ran | 52.677 | 3.8% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| entity_resolution | ran | 19.181 | 1.4% | 0 | 0 | 0/0 | – | 0 | est. n/a | 194/194 entities resolved (135 biomapper, 49 fuzzy, 10 exact) |
| hypothesis_extraction | ran | 1.653 | 0.1% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| intake | ran | 0.004 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 194 entities extracted, query_type=discovery |
| temporal | skipped | – | – | – | – | – | – | – | – | – |

### Context management

> Synthesis compresses the accumulated evidence into a bounded context; elision at module scale is expected, and it is the compression that keeps the assembled context within the model's token window.

- **Context:** 156573 chars (~44735 est. tokens) · 44.7% of the char budget (350000) · 22.4% of the 200K-token window
- **Mode:** module-aware aggregation (181 distinct entities; threshold 5)
- **Literature grounding:** 15 of 98 hypotheses carry attached literature

| Section | Shown | Total | Elided |
|---|---|---|---|
| Findings | 143 | 3201 | 3058 |
| Diseases (recurrence) | 30 | 178 | 148 |
| Pathways (recurrence) | 30 | 141 | 111 |
| Member table | 50 | 181 | 131 |

### Errors

None.