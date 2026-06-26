# Brown Module Full Run: Pipeline Performance Report (203-analyte, dev, 2026-06-22)

> **Updated 2026-06-23 (corrected re-run, commit `edf351f`).** The first full-module run was degraded by a triage-concurrency bug (unbounded edge-count fan-out → knowledge-graph overload → hub genes silently cold-started). With that fixed, this report's profile shifts: triage now records zero measurement failures and routes the recovered hub genes into the direct-KG path, which is why `pathway_enrichment` is now the clear dominant node.

This document is the per-node performance report emitted automatically at the end of the corrected full-module Brown WGCNA discovery run described in the companion [Discovery Output](https://phwiki.phenoma.ai/doc/brown-module-full-run-discovery-output-203-analyte-dev-2026-06-22-giJ0OvxUln). The report is produced by the Kraken pipeline's performance-reporter instrumentation, a terminal graph node that runs on every discovery execution and attributes wall-clock time, token consumption, estimated cost, output counts, and errors to each of the twelve pipeline nodes. This instance is the full module at dev commit `edf351f`, which integrates the synthesis-context, intake-robustness, and triage-reliability fixes.

We highlight three results. First, `pathway_enrichment` is now both the dominant latency and the dominant cost contributor (823 s, 49% of summed node time; $1.08 of a $2.27 estimated run). This is a direct consequence of fixing triage: correcting the false cold-starts roughly doubled the entities routed to direct-KG analysis (122 well-characterized plus moderate, versus 66 in the buggy run), so pathway enrichment genuinely computes shared neighbors for the recovered hub genes rather than skipping them. It is the clear next optimization target for module-scale latency. Second, triage recorded zero measurement failures: bounding its edge-count concurrency eliminated the thundering-herd timeouts that had silently downgraded well-characterized entities. Third, the Context management section shows the compression caps engaging at scale without breaching the budget: the member prioritization table shows 50 of 181 entities (131 elided) and the assembled context occupies 34% of its character budget. The wall-clock figure is a summed-duration estimate; for the concurrently executed branches this overstates true elapsed time (the run completed in approximately 1,602 s of real time), and per-node percentages represent each node's share of summed work.

## Related

- Discovery analysis from this run: [Brown Module Full Run: Discovery Output](https://phwiki.phenoma.ai/doc/brown-module-full-run-discovery-output-203-analyte-dev-2026-06-22-giJ0OvxUln)
- Pilot-scale precursor: [Brown Module C1 Pilot: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/brown-module-c1-pilot-pipeline-performance-report-24-analyte-dev-2026-06-22-vk616thIl8)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---

## Pipeline Performance Report

- **Run:** `e257479d9cb5473bb8c6300954df1b97`
- **Mode:** pipeline · **biomapper_env:** dev
- **Timestamp:** 20260623T094328Z · **commit:** `edf351f1e6199a46fefe99f204897fc05c7c4737`
- **Wall-clock:** 1685.208s · **tokens:** 23 in / 52700 out (114914 cache-r, 384886 cache-w) · **est. cost:** $2.2684
- **Top bottleneck:** pathway_enrichment · **Top cost:** pathway_enrichment

> Wall-clock is estimated as the sum of per-node durations, because no measured elapsed time was available; for concurrent branches this sum overstates true elapsed time. The percentage column reports each node's share of that summed total.

### Per-node

| Node | Status | Duration (s) | % | In | Out | Cache r/w | Model | MCP | Est. $ | Findings |
|---|---|---|---|---|---|---|---|---|---|---|
| pathway_enrichment | ran | 822.982 | 48.8% | 4 | 38031 | 114914/127723 | anthropic/claude-sonnet-4-20250514 | 0 | $1.0839 | 175 shared neighbors (154 non-hub), 12 biological themes |
| entity_resolution | ran | 169.325 | 10.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 194/194 entities resolved (135 biomapper, 49 fuzzy, 10 exact) |
| synthesis | ran | 167.037 | 9.9% | 3 | 7139 | 0/71611 | anthropic/claude-sonnet-4-20250514 | 0 | $0.3756 | 54 hypotheses generated, 26432 char report |
| triage | ran | 150.886 | 9.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 81 well-characterized, 41 moderate, 61 sparse, 11 cold-start |
| cold_start | ran | 128.585 | 7.6% | 14 | 5490 | 0/159110 | anthropic/claude-sonnet-4-20250514 | 0 | $0.6791 | 21 analogues, 9 inferred associations, 15 findings |
| direct_kg | ran | 83.47 | 5.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 965 disease associations, 782 pathways, 2729 findings, 11 hub flags |
| literature_grounding | ran | 60.541 | 3.6% | 0 | 0 | 0/0 | – | 0 | est. n/a | 32 papers, 32 from Exa across 15/54 hypotheses |
| bridge_grounding | ran | 51.63 | 3.1% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| integration | ran | 49.096 | 2.9% | 2 | 2040 | 0/26442 | anthropic/claude-sonnet-4-20250514 | 0 | $0.1298 | 30 bridges, 11 gap entities |
| hypothesis_extraction | ran | 1.651 | 0.1% | 0 | 0 | 0/0 | – | 0 | est. n/a | Node completed |
| intake | ran | 0.006 | 0.0% | 0 | 0 | 0/0 | – | 0 | est. n/a | 194 entities extracted, query_type=discovery |
| temporal | skipped | – | – | – | – | – | – | – | – | – |

### Context management

> Synthesis compresses the accumulated evidence into a bounded context; elision at module scale is expected, and it is the compression that keeps the assembled context within the model's token window.

- **Context:** 120168 chars (~34334 est. tokens) · 34.3% of the char budget (350000) · 17.2% of the 200K-token window
- **Mode:** module-aware aggregation (181 distinct entities; threshold 5)
- **Literature grounding:** 15 of 54 hypotheses carry attached literature

| Section | Shown | Total | Elided |
|---|---|---|---|
| Findings | 121 | 2744 | 2623 |
| Diseases (recurrence) | 30 | 161 | 131 |
| Pathways (recurrence) | 30 | 95 | 65 |
| Member table | 50 | 181 | 131 |

### Errors

None.
