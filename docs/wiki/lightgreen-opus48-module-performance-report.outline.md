# Lightgreen Module Run on Opus 4.8: Pipeline Performance Report (29-analyte, dev, 2026-06-24)

This is the per-node performance report emitted automatically at the end of the full-module **Lightgreen** discovery run on dev (commit `a7d7fd6`). All SDK-backed nodes ran on the Opus 4.8 model; this report is published for direct model comparison against the Sonnet baseline. The Kraken performance-reporter (a terminal graph node) attributes wall-clock, tokens, estimated cost, output counts, and errors across the twelve pipeline nodes, and the Context management section reports how synthesis compresses the accumulated evidence into a bounded context.

## Related

- Discovery analysis from this run: [Lightgreen Module Run on Opus 4.8: Discovery Output](https://phwiki.phenoma.ai/doc/lightgreen-module-run-on-opus-48-discovery-output-29-analyte-dev-2026-06-24-PFkGc57Uxz)
- Model comparison baseline (Sonnet): [Lightgreen Module Run: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/lightgreen-module-run-pipeline-performance-report-29-analyte-dev-2026-06-23-dYmjwNiqBg)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---

## Pipeline Performance Report

- **Run:** `2f5015f529d54a869dda8325412f8293`
- **Mode:** pipeline · **biomapper_env:** dev
- **Timestamp:** 20260624T175951Z · **commit:** `a7d7fd63deffdbbb25f8b30e8dee40cefdf6c3cd`
- **Wall-clock:** 635.225s · **tokens:** 22 in / 23327 out (64398 cache-r, 209872 cache-w) · **est. cost:** $1.9272
- **Top bottleneck:** cold_start · **Top cost:** cold_start

> Wall-clock is estimated as the sum of per-node durations, because no measured elapsed time was available; for concurrent branches this sum overstates true elapsed time. The percentage column reports each node's share of that summed total.

### Per-node

| Node | Status | Duration (s) | % | In | Out | Cache r/w | Model | MCP | Est. $ | Findings |
|---|---|---|---|---|---|---|---|---|---|---|
| cold_start | ran | 174.117 | 27.4% | 14 | 9332 | 42932/117854 | claude-opus-4-8 | 0 | $0.9914 | 24 analogues, 33 inferred associations, 35 findings |
| synthesis | ran | 145.304 | 22.9% | 3 | 6322 | 21466/31054 | claude-opus-4-8 | 0 | $0.3629 | 68 hypotheses generated, 24238 char report |
| pathway_enrichment | ran | 84.735 | 13.3% | 3 | 5320 | 0/33959 | claude-opus-4-8 | 0 | $0.3453 | 29 shared neighbors (28 non-hub), 4 biological themes |
| direct_kg | ran | 64.438 | 10.1% | 0 | 0 | 0/0 | – | 0 | est. n/a | 108 disease associations, 101 pathways, 654 findings |
| literature_grounding | ran | 59.191 | 9.3% | 0 | 0 | 0/0 | – | 0 | est. n/a | 15 papers, 15 from Exa across 10/68 hypotheses |
| integration | ran | 55.129 | 8.7% | 2 | 2353 | 0/27005 | claude-opus-4-8 | 0 | $0.2276 | 0 bridges, 11 gap entities |
| entity_resolution | ran | 26.795 | 4.2% | 0 | 0 | 0/0 | – | 0 | est. n/a | 30/30 entities resolved (14 fuzzy, 15 biomapper, 1 exact) |
| triage | ran | 25.514 | 4.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 5 well-characterized, 10 moderate, 11 sparse, 4 cold-start |
| intake | ran | 0.002 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 30 entities extracted, query_type=discovery |
| hypothesis_extraction | ran | 0.0 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| bridge_grounding | ran | 0.0 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| temporal | skipped | – | – | – | – | – | – | – | – | – |

### Context management

> Synthesis compresses the accumulated evidence into a bounded context; elision at module scale is expected, and it is the compression that keeps the assembled context within the model's token window.

- **Context:** 83362 chars (~23818 est. tokens) · 23.8% of the char budget (350000) · 11.9% of the 200K-token window
- **Mode:** module-aware aggregation (30 distinct entities; threshold 5)
- **Literature grounding:** 10 of 68 hypotheses carry attached literature

| Section | Shown | Total | Elided |
|---|---|---|---|
| Findings | 143 | 689 | 546 |
| Diseases (recurrence) | 10 | 10 | 0 |
| Pathways (recurrence) | 2 | 2 | 0 |
| Member table | 30 | 30 | 0 |

### Errors

- SDK inference timed out for PUBCHEM.COMPOUND:11728391 after 60s