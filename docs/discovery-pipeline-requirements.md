# Kraken Discovery Pipeline — Requirements Overview

**Date:** 2026-06-03
**Source:** Lance's list of wants for the Kraken chatbot discovery pipeline (meeting 2026-06-02). Captured live in daily note `Active 🎯/Planning/Daily Notes/2026-06-02.md`.
**Context:** Lance is tinkering with two external tools — **[whyspr.bio](https://whyspr.bio)** and **biomni** — as idea sources. They are *not* the target system; the goal is to **fold the useful whyspr-style capabilities into our own discovery-pipeline requirements.**
**Related:** `discovery-pipeline.md` (current LangGraph architecture), `pipeline-node-tool-map.md` (node→tool map), `code-on-graph-feasibility.md` (iterative query-loop feasibility), `brainstorms/code-on-graph-spike-requirements.md` (the go/no-go spike).

---

## TL;DR

- The discovery pipeline should take a **ranked, module-organized analyte list** as input and produce **mechanistically-grounded, multiomic-enriched interpretation** that terminates in **actionable drug/intervention hypotheses**.
- Four interpretation themes Lance is anchored on: **drug safety, mechanism of action, network centrality, drug repurposing.**
- Core interpretation = KG-grounded LLM reasoning + subgraph mechanistic extraction + multiomic enrichment (ORA/GSEA) + multi-hop to drug/intervention.
- A second tier of **"nice to have"** agents (single-omic enrichments, covariate-aware regression, WGCNA) would deepen the analysis but aren't blocking.
- Several of these map cleanly onto the existing pipeline nodes; this doc flags where the gaps are vs. what already exists.

---

## Guiding Themes

These are the lenses Lance wants the pipeline's output to serve:

| Theme | What it means for output |
|---|---|
| **Drug safety** | Surface safety-relevant signals (off-target, toxicity-adjacent pathways) when proposing interventions |
| **Mechanism of action** | Explain *why* an analyte/module matters mechanistically, not just that it's correlated |
| **Network centrality** | Rank/weight analytes by their structural importance in the KG / co-expression network |
| **Drug repurposing** | Multi-hop from implicated biology to existing drugs/interventions as repurposing candidates |

---

## 1. Inputs to the agentic system

The pipeline should accept, per study:

- **List of all analytes by module and connectivity** — analytes grouped into modules, annotated with their network connectivity.
- **Rankings by regression coefficients** — analytes ordered by effect size from the upstream statistical model.

> *Implication:* the pipeline consumes a **pre-computed module/ranking structure**, not raw measurements. Module detection + regression happen upstream (or via the "nice to have" agents below).

---

## 2. Core interpretation requirements (must-have)

Interpretation should be produced from a combination of:

- **KG-grounded LLM interpretation** — LLM reasoning explicitly grounded in Kestrel KG evidence (fetch-then-reason, KG kept out of the prompt).
- **LLM interpretation** — complementary ungrounded/general LLM reasoning layer.
- **Subgraph mechanistic extraction** — pull the relevant mechanistic subgraph for implicated analytes/modules to support MoA explanations.
- **Multiomic enrichment:**
  - **ORA** (over-representation analysis) against **WikiPathways, KEGG, Reactome**.
  - **GSEA** ranked by **connectivity** and **limma** statistics.
- **Multi-hop to drug/intervention** — traverse from implicated biology to drugs/interventions (serves repurposing + safety themes).

---

## 3. Nice-to-have requirements (second tier)

Valuable enhancements, not blocking for a first cut:

- **Single-omic enrichments:**
  - Metabolite pathway enrichment
  - Lipid class enrichment
  - GO enrichment
- **Network centrality** — first-class centrality scoring of analytes (also one of the four guiding themes; promote if it proves load-bearing).
- **Regression agent with covariates (limma)** — an agent that runs covariate-adjusted regression to produce/refresh the ranking input.
- **WGCNA agent** — an agent that performs weighted gene co-expression network analysis to produce the module/connectivity input.

> *Note:* the limma and WGCNA agents would let the pipeline **generate its own inputs** (modules + rankings) rather than requiring them pre-computed — closing the loop on Section 1.

---

## 4. Mapping to the current pipeline (gap analysis)

Cross-referenced against `discovery-pipeline.md` (10-node LangGraph) and `pipeline-node-tool-map.md`:

| Requirement | Current pipeline coverage | Gap / action |
|---|---|---|
| Module + connectivity input | Entity resolution handles CURIEs, not modules | **Gap** — needs module/ranking ingestion (or WGCNA agent) |
| Regression-coefficient ranking | Not present | **Gap** — needs limma regression agent or upstream import |
| KG-grounded LLM interpretation | ✅ Core of direct-KG / integration nodes | Covered |
| Subgraph mechanistic extraction | Partially (integration / bridges) | Confirm depth vs. MoA needs |
| Multiomic enrichment (ORA/GSEA) | Pathway enrichment node exists | **Partial** — confirm ORA source coverage (Wiki/KEGG/Reactome) + GSEA by limma/connectivity |
| Multi-hop to drug/intervention | Integration/multi-hop primitives exist | Confirm it terminates in drug/intervention nodes |
| MoA / bridge-discovery *quality* (themes #2, #4) | `integration` uses a fixed `max_pairs=3` fan-out; no iterative query refinement | **Spike in progress** — `brainstorms/code-on-graph-spike-requirements.md` tests whether an iterative query loop beats the static plan, benchmarked on DrugMechDB MoA paths + EITL. A "go" needs two repurposing follow-ups: analyte-anchored entry + bridges terminating on drug/intervention nodes. |
| Single-omic enrichments | Not present | Nice-to-have |
| Network centrality scoring | Not first-class | Nice-to-have → possibly promote |
| WGCNA / limma agents | Not present | Nice-to-have (input generators) |

*(Coverage column is provisional — verify against the live node implementations before committing to the requirement set.)*

---

## 5. whyspr.bio features to evaluate for folding in

Action item from Lance: review **whyspr.bio** specifically for the above capabilities and decide which to adopt. Candidate areas where whyspr's approach may be worth borrowing:

- Multiomic enrichment UX/stack (ORA + GSEA across multiple pathway DBs)
- Module → mechanism → drug narrative structure
- Centrality-based ranking presentation

*(biomni is the second tinkering tool — note any complementary capabilities as they surface.)*

---

## Open questions

1. Are modules + regression rankings **provided** to the pipeline, or should it **compute** them (WGCNA + limma agents)?
2. Which enrichment libraries/services back ORA and GSEA — reuse existing or new dependency?
3. Does "drug safety" need a dedicated safety-evidence source, or is it derived from multi-hop KG context?
4. Priority order for the nice-to-haves — which single, if any, gets promoted to must-have for the next milestone?

---

*Draft requirements overview — to be refined with Lance. Source notes preserved in the 2026-06-02 daily note.*
