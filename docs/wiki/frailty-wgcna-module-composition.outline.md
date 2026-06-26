# Frailty Multi-Omics WGCNA Modules: Dataset Composition

This document describes the composition of the eighteen weighted gene co-expression network analysis (WGCNA) modules that are the input to the Kraken discovery-pipeline analyses collected here. It is the reference for what each module contains before any pipeline processing, and it defines the analyte counts used throughout the per-module and cross-module reports. The discovery analyses themselves are indexed in the master index for this collection and synthesized in the cross-module syntheses.

## Overview

The dataset partitions 1,197 frailty multi-omics analytes into eighteen co-expression modules by WGCNA. Each analyte is one of three measurement types: a protein (identified by gene symbol), a metabolite (identified by chemical name), or a chemistry species. Of the 1,197 analytes, 1,148 carry a usable name and are therefore submittable to the discovery pipeline; the 49 unnameable analytes are chemistry rows with an empty chemical name and are inherently un-submittable. The modules range in size from Brown (217 analytes, 203 named) down to Darkgreen (20 analytes), and they vary widely in protein-to-metabolite balance, which is the single most important determinant of how each module behaves in the pipeline. Protein-heavy modules (Blue, Turquoise, Grey) resolve to well-characterized knowledge-graph hubs and exercise the direct-knowledge-graph and pathway-enrichment paths; all-metabolite modules (Black, Green, Purple, Grey60, Lightgreen, Darkgreen, Royalblue) are sparser in the graph and exercise the cold-start and sparse-entity paths.

## Per-module composition

The "named" column is the count of analytes with a usable identifier; it is the number submitted to the discovery pipeline. The difference between total and named is chemistry rows lacking a chemical name.

| Module | Total | Proteins | Metabolites | Chemistry | Named (submitted) |
|---|---|---|---|---|---|
| Brown | 217 | 50 | 153 | 14 | 203 |
| Blue | 150 | 93 | 56 | 1 | 149 |
| Turquoise | 114 | 61 | 46 | 7 | 107 |
| Black | 109 | 0 | 109 | 0 | 109 |
| Green | 84 | 1 | 82 | 1 | 83 |
| Midnightblue | 78 | 6 | 70 | 2 | 76 |
| Pink | 54 | 2 | 52 | 0 | 54 |
| Magenta | 51 | 9 | 38 | 4 | 47 |
| Purple | 51 | 2 | 49 | 0 | 51 |
| Grey | 51 | 28 | 18 | 5 | 46 |
| Tan | 43 | 12 | 31 | 0 | 43 |
| Lightcyan | 33 | 4 | 23 | 6 | 27 |
| Grey60 | 32 | 0 | 32 | 0 | 32 |
| Lightgreen | 29 | 0 | 29 | 0 | 29 |
| Lightyellow | 28 | 5 | 14 | 9 | 19 |
| Royalblue | 27 | 1 | 26 | 0 | 27 |
| Darkred | 26 | 2 | 24 | 0 | 26 |
| Darkgreen | 20 | 0 | 20 | 0 | 20 |
| **Total** | **1,197** | **283** | **891** | **49 (named) + 49 (unnamed)** | **1,148** |

## Notes on composition

Three points govern how this composition maps onto the discovery analyses. First, the named count is the reachable maximum per module; Brown, for example, submits 203 of its 217 rows because 14 chemistry entries have no chemical name. Reaching the full named count required an intake-parsing fix that preserves internal commas in chemical names and admits parenthetical synonyms; before that fix the harness submitted far fewer analytes. Second, the protein-to-metabolite balance predicts the triage outcome: protein-heavy modules land mostly in the well-characterized and moderate buckets, while all-metabolite modules shift toward sparse and cold-start, which is biology (metabolites are genuinely sparser in the knowledge graph) rather than a pipeline failure. Third, the Grey module is, by WGCNA convention, the unassigned or leftover bucket holding analytes that did not cluster into a coherent co-expression module; its downstream analysis is therefore interpreted with more caution than the other seventeen, which are genuine co-expression modules. The Grey60 module, despite the similar name, is a distinct and coherent module.
