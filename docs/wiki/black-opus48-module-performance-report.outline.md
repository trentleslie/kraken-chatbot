# Black Module Run on Opus 4.8: Pipeline Performance Report (109-analyte, dev, 2026-06-24)

This is the per-node performance report emitted automatically at the end of the full-module **Black** discovery run on dev (commit `098093e`). All SDK-backed nodes ran on the Opus 4.8 model; this report is published for direct model comparison against the Sonnet baseline. The Kraken performance-reporter (a terminal graph node) attributes wall-clock, tokens, estimated cost, output counts, and errors across the twelve pipeline nodes, and the Context management section reports how synthesis compresses the accumulated evidence into a bounded context.

## Related

- Discovery analysis from this run: [Black Module Run on Opus 4.8: Discovery Output](https://phwiki.phenoma.ai/doc/black-module-run-on-opus-48-discovery-output-109-analyte-dev-2026-06-24-KqKTyfz1AZ)
- Model comparison baseline (Sonnet): [Black Module Run: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/black-module-run-pipeline-performance-report-109-analyte-dev-2026-06-23-EOscfYQbEB)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---

## Pipeline Performance Report

- **Run:** `5ccc3cf448d142c0972d565809f8eb45`
- **Mode:** pipeline · **biomapper_env:** dev
- **Timestamp:** 20260624T014218Z · **commit:** `098093ea146164cb1ed09db4b8dcc4e8fb1c39a9`
- **Wall-clock:** 933.595s · **tokens:** 22 in / 24821 out (107330 cache-r, 194306 cache-w) · **est. cost:** $1.8887
- **Top bottleneck:** pathway_enrichment · **Top cost:** synthesis

> Wall-clock is estimated as the sum of per-node durations, because no measured elapsed time was available; for concurrent branches this sum overstates true elapsed time. The percentage column reports each node's share of that summed total.

### Per-node

| Node | Status | Duration (s) | % | In | Out | Cache r/w | Model | MCP | Est. $ | Findings |
|---|---|---|---|---|---|---|---|---|---|---|
| pathway_enrichment | ran | 229.823 | 24.6% | 3 | 10754 | 0/47115 | claude-opus-4-8 | 0 | $0.5633 | 79 shared neighbors (67 non-hub), 8 biological themes |
| cold_start | ran | 172.437 | 18.5% | 14 | 5397 | 107330/53260 | claude-opus-4-8 | 0 | $0.5215 | 24 analogues, 17 inferred associations, 22 findings |
| synthesis | ran | 145.388 | 15.6% | 3 | 6439 | 0/66459 | claude-opus-4-8 | 0 | $0.5764 | 40 hypotheses generated, 27040 char report |
| entity_resolution | ran | 99.49 | 10.7% | 0 | 0 | 0/0 | – | 0 | est. n/a | 108/108 entities resolved (35 fuzzy, 70 biomapper, 3 exact) |
| triage | ran | 86.935 | 9.3% | 0 | 0 | 0/0 | – | 0 | est. n/a | 19 well-characterized, 27 moderate, 46 sparse, 16 cold-start |
| direct_kg | ran | 85.175 | 9.1% | 0 | 0 | 0/0 | – | 0 | est. n/a | 537 disease associations, 319 pathways, 1733 findings, 1 hub flags |
| literature_grounding | ran | 61.212 | 6.6% | 0 | 0 | 0/0 | – | 0 | est. n/a | 39 papers, 22 from OpenAlex, 15 from Exa across 14/40 hypotheses |
| integration | ran | 52.242 | 5.6% | 2 | 2231 | 0/27472 | claude-opus-4-8 | 0 | $0.2275 | 1 bridges, 11 gap entities |
| hypothesis_extraction | ran | 0.891 | 0.1% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| intake | ran | 0.003 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 108 entities extracted, query_type=discovery |
| bridge_grounding | ran | 0.0 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| temporal | skipped | – | – | – | – | – | – | – | – | – |

### Context management

> Synthesis compresses the accumulated evidence into a bounded context; elision at module scale is expected, and it is the compression that keeps the assembled context within the model's token window.

- **Context:** 127041 chars (~36297 est. tokens) · 36.3% of the char budget (350000) · 18.1% of the 200K-token window
- **Mode:** module-aware aggregation (104 distinct entities; threshold 5)
- **Literature grounding:** 14 of 40 hypotheses carry attached literature

| Section | Shown | Total | Elided |
|---|---|---|---|
| Findings | 130 | 1755 | 1625 |
| Diseases (recurrence) | 30 | 92 | 62 |
| Pathways (recurrence) | 30 | 31 | 1 |
| Member table | 50 | 104 | 54 |

### Errors

None.