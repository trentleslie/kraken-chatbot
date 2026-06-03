---
date: 2026-05-29
topic: kestrel-deck-deliverables
relates_to:
  - docs/brainstorms/kestrel-api-depth-requirements.md
  - docs/plans/2026-05-25-001-feat-kestrel-api-depth-plan.md
  - docs/plans/2026-05-07-001-feat-model-usages-cost-tracking-plan.md
---

# 6/4 Deck Deliverables: AstaBench Lit-Grounding Solver + Studio Reasoning-Depth Demo

## Problem Frame

The 6/4 data-science meeting deck needs to *show* the discovery pipeline's value, on top
of the already-planned core reasoning-depth build (R1–R10 in
`docs/brainstorms/kestrel-api-depth-requirements.md`, planned in
`docs/plans/2026-05-25-001-feat-kestrel-api-depth-plan.md`). This document covers the two
**presentation deliverables** that sit on top of that build:

1. **AstaBench integration** — runs KRAKEN's literature grounding inside the field-standard
   eval harness (feasibility) + a qualitative comparison. A benchmark *number* is a stretch,
   not a committed output (see Deliverable 1 scope + the SQA-scoring caveats).
2. **LangGraph Studio demo** — qualitative, visible reasoning depth.

The core R1–R10 build is **not** re-opened here; it is brainstormed and planned already.

**Who is affected:** The data-science meeting audience, who need an intuitive picture of
reasoning depth (Studio) plus evidence that the grounding is sound (AstaBench).

### Key finding from research: no shipped AstaBench task measures KRAKEN's actual job

AstaBench ships ~11 tasks in 4 categories (lit / code / data / discovery). None is a
biomedical KG-traversal hypothesis-discovery task. `DiscoveryBench` is tabular-data
discovery (load a CSV); `E2E-Bench` is autonomous experimentation (expects code +
artifacts). Wrapping the *whole* pipeline against either produces an off-task, caveat-heavy
number.

**The literature-grounding seam resolves this.** KRAKEN's `literature_grounding` node
(`backend/src/kestrel_backend/graph/nodes/literature_grounding.py`) does exactly what the
AstaBench **Literature** category scores: take claims, search PubMed / OpenAlex / Exa /
Semantic Scholar, classify each as supporting/contradicting/neutral, attach citations with
key passages. That maps directly onto **ScholarQABench2 (`sqa`)** metrics —
`citation_precision`, `citation_recall`, `answer_precision`, `ingredient_recall`. Scoping
the eval to this component gives an *on-task* measurement of **citation grounding** —
`citation_precision` / `citation_recall`.

> **Caveat (review 2026-05-29):** SQA's aggregate weights all 4 metrics equally
> (0.25 each), and **2 of the 4 — `ingredient_recall` and `answer_precision` — grade the
> ANSWER TEXT content** via a rubric LLM judge, not the citations. A *thin synthesis*
> answer carries little ingredient content, so those two metrics score near-zero regardless
> of citation quality. The "on-task, independent of domain" claim holds **only for the two
> citation metrics**; the aggregate SQA score is not a clean signal for this reduced solver.

---

## Deliverable 1 — AstaBench Lit-Grounding-Only Solver

**Shape:** A reduced Inspect AI solver entry point that takes an SQA research question →
runs a *thin answer-synthesis* step (question → a few claims forming a sectioned answer) →
runs `literature_grounding` to cite each claim → formats output as SQA's
`{sections: [{title, text, citations}]}` JSON. The KG path (intake, entity_resolution,
triage, direct_kg, cold_start, pathway_enrichment, integration, full synthesis) is
**bypassed**.

**Deck claim:** "KRAKEN's literature grounding **runs as a real AstaBench solver** (feasibility
proven), and here's **qualitatively** what it surfaces vs. AstaBench's own tools for the same
question." A ScholarQA citation precision/recall *number* is nice-to-have, not committed.

**6/4 scope (FINAL, decided 2026-05-29 post-review):** **Descoped to the spike + L6.** The
committed 6/4 work is: (1) the de-risking spike (see Next Steps) and (2) **L6 — the
qualitative side-by-side**, built as a thin direct-call script (`thin-synthesis →
literature_grounding → comparison table`), **not** a full Inspect solver. **L1 (Inspect
solver entry), L3 (SQA output adapter), L4 (cost wiring), and L5 (scored run) are DEFERRED
to post-6/4** — they serve only the stretch scored number, whose signal is compromised
(thin-answer metrics + 2025-05 corpus cutoff + empty snippets per the review). **L2** is
still needed for L6 but only in its minimal direct-call form (no Inspect harness).

**Comparison framing (decided 2026-05-29):** The comparison is **qualitative, not a scored
head-to-head** — deliberately, to avoid the agent-vs-tools/corpus confound. For one (or a
few) curated query, run AstaBench's standardized literature search tools AND KRAKEN's
`literature_grounding`, and show **which papers/citations each surfaces** side by side. No
fairness claim, no score comparison — a narrative slide. Both arms **run beforehand**,
presented statically (no live run on stage).

### Requirements

- **L1.** *(DEFERRED post-6/4)* Add a reduced solver entry path (in the `kraken-chatbot-solver`
  repo at `../kraken-chatbot-solver`) distinct from the existing full-pipeline
  `kraken_discovery_solver` — e.g. `kraken_lit_grounding_solver`, runnable via
  `inspect eval --solver ... astabench/sqa_dev`. Not needed for L6.
- **L2.** *(COMMITTED 6/4, minimal form — a direct function call, no Inspect harness.)*
  Input bridge: convert an SQA question into the `list[Hypothesis]` (claim text)
  that `literature_grounding` consumes. A **thin synthesis** step generates a minimal
  sectioned answer whose claims become the hypotheses to ground. **Not purely "thin"
  (review):** `Hypothesis` is a frozen model requiring `title`, `tier`, `claim`,
  `supporting_entities`, `structural_logic`, and `validation_steps` — the bridge must
  synthesize placeholder values for the KG-derived fields (`supporting_entities`,
  `structural_logic`, `validation_steps`) the bypassed path normally fills. Empty
  `supporting_entities` also disables the node's entity-focused search strategy, so
  retrieval differs from the full pipeline. The thin synthesis is the only net-new
  *reasoning* component, but the bridge itself is non-trivial.
- **L3.** *(DEFERRED post-6/4 — only needed for the scored run.)* Output adapter: convert
  grounded hypotheses + their `LiteratureSupport`
  (title, authors, year, doi, key_passage, relationship, source) into SQA's required
  `{sections: [{title, text, citations: [{id, snippets, title, metadata}]}]}` JSON. **Adapter
  constraints (review, code-verified):** (a) write the JSON to `state.output.completion` —
  the SQA scorers read `state.output.completion`, **not** `metadata` (the existing
  full-pipeline solver writes structured state to metadata — do not copy that habit);
  (b) each citation `id` must appear **verbatim** in the section `text` (scorer matches
  `id in sentence`); (c) `snippets` must be genuine paper excerpts **distinct** from the
  section text and the title — the scorer's `filter_citation` drops snippets equal to the
  title or contained in the text; (d) **`key_passage` is empty (`""`) for KG-, PubMed-, and
  OpenAlex-sourced citations** — only Exa (highlights) and S2 (`extract_key_passage`)
  populate it, so a majority of citations would be filtered out, sinking `citation_recall`.
  The `key_passage → snippets` mapping is **not** a mechanical rename.
- **L4.** *(DEFERRED post-6/4 — only needed for the scored run.)* Cost tracking: wire
  `record_model_usage_with_inspect` for the **thin-synthesis** LLM call (the only LLM call
  the reduced solver itself owns and can wrap). **HTTP-only is
  conditional (review, code-verified):** `literature_grounding` is HTTP-only **only when
  `use_llm_classifier=False`, which is the default**. The `classify_relationship_llm` path
  is a raw Claude Agent SDK `query()` call that (a) only runs when the flag is flipped and
  (b) returns no usage object, so `record_model_usage_with_inspect` cannot observe it
  without changing node internals (out of scope). **Decision: pin `use_llm_classifier=False`
  for the deck run** (relationships default to `supporting`), keeping L4 to one tracked call.
  (The per-node `model_usages` machinery exists for the 8 instrumented LLM nodes; this node
  was never instrumented because the plan classified it HTTP-only.)
- **L5.** *(DEFERRED post-6/4)* Run on the SQA **validation** split and produce a results
  bundle (score + per-metric breakdown + cost). Depends on L1/L3/L4 and is signal-compromised
  per the review — revisit only after the deferred solver work and search date-bounding.
- **L6.** *(COMMITTED 6/4 — the primary artifact.)* Qualitative comparison: for a curated
  query set, run AstaBench's standardized literature search tools AND KRAKEN's
  `literature_grounding`, and produce a **side-by-side of the papers/citations each surfaces**
  (title, source, key passage, relationship where available). No scoring, no fairness claim.
  Pre-computed; static deck slide. Built as a thin direct-call script, not an Inspect solver.
  See strategic decision #2 (anti-anecdote query protocol) before finalizing the query set.

### Success Criteria

**Required (6/4):**
- `literature_grounding` runs on ≥1 curated query (via the bridge) without erroring,
  producing citations.
- A qualitative side-by-side slide (L6): papers/citations KRAKEN surfaces vs. AstaBench's
  tools for the curated query set, pre-computed.

**Nice-to-have (stretch):**
- The reduced solver runs end-to-end against `astabench/sqa_dev` and its output validates
  against the SQA scorer's expected JSON (no "Invalid output format" zero-scores). *Note:
  "validates/parses" ≠ "scores well" — see the SQA-scoring caveats; expect low aggregate
  scores from thin answers + the 2025-05 corpus cutoff.*
- The thin-synthesis LLM call cost-tracked; a `citation_precision` / `citation_recall`
  number + cost on the SQA validation split, with caveats stated.

### Scope Boundaries

- **In scope:** Reduced solver entry path; question→claims thin synthesis; SQA output
  adapter; cost wiring for the reduced path; the qualitative side-by-side (L6, primary deck
  artifact). A scored validation-split run is nice-to-have (L5).
- **Out of scope:** The full-pipeline solver against SQA; DiscoveryBench / E2E-Bench;
  building a custom biomedical task; **replacing** `literature_grounding`'s 4 search APIs
  with AstaBench's standardized tools *inside the node* (the deferred "fair variant" — note
  this is distinct from L6, which *invokes* AstaBench's tools standalone, alongside KRAKEN,
  purely to tabulate the comparison); changes to the core R1–R10 build; modifying
  `literature_grounding`'s search/scoring logic.
- **Not changing:** `literature_grounding` node internals; the existing
  `kraken_discovery_solver`; the completed `model_usages` instrumentation.

### Key Decisions

- **Lit-grounding-only, not full pipeline:** Isolates the component whose job actually
  matches a lit task, producing an on-task number instead of a caveated off-task one.
- **SQA as the target task:** Its citation precision/recall metrics are exactly what
  literature grounding governs; it needs no Docker sandbox and no provided dataset (unlike
  DiscoveryBench / E2E-Bench).
- **Thin synthesis is required, not optional:** SQA gives a *question*, but
  `literature_grounding` consumes *claims/hypotheses*. A minimal answer generator bridges
  the two. The deck claim is therefore about *grounding discipline on top of a minimal
  answer*, not the full discovery value — state this honestly.
- **Custom-toolset + corpus-date caveat (deck honesty, code-verified):** KRAKEN uses its
  own 4 search APIs, not AstaBench's standardized search tools. Beyond the `--toolset custom`
  leaderboard-comparability point, there is a deeper **validity** issue: SQA freezes its
  search corpus at `inserted_before=2025-05` so the provided tools match the corpus the gold
  citations were computed against. KRAKEN's APIs apply **no date filter**, so it can cite
  papers published after the cutoff that the gold set never contained — structurally
  depressing `citation_recall`/`citation_precision`. This is not just "may score lower"; the
  scored comparison is partly invalid unless KRAKEN's search is date-bounded. State it.
- **Domain caveat:** SQA questions are broad scientific QA; `literature_grounding`'s query
  builder is biomedical-tuned. Citation precision/recall is more grounding-discipline than
  domain, but non-biomedical questions may underperform — note it.

### Open Questions (deferred to planning)

- **[L6] How to invoke AstaBench's search tools standalone?** The qualitative arm needs
  AstaBench's standardized lit search tools run on a query *outside* a full solver/scorer
  loop (e.g. call the tool factory `make_native_search_tools` / `make_asta_mcp_tools` from
  `astabench/evals/sqa/task.py` directly, or a 1-sample `inspect eval` whose trace exposes
  the retrieved papers). Spike which is simplest.
- **[L6] Query selection:** pick 1–3 biomedical questions where the side-by-side is
  legible (KRAKEN surfaces mechanistic/KG-adjacent papers; AstaBench surfaces
  search-relevant papers). Curate, don't cherry-pick dishonestly — note any selection.
- **[L2] How thin is the thin synthesis?** One-claim-per-question vs. a few-claim sectioned
  answer. Spike: does grounding the literal question vs. grounding generated sub-claims
  score better on SQA? Decide empirically on the validation split.
- **[L2/L3] Mini-graph vs. direct call:** Build a 2-node mini-graph
  (thin_synthesis → literature_grounding) reusing `build_discovery_graph` plumbing, or call
  the node functions directly from the solver? Planning decision.
- **[L1] Where does the reduced entry live** — a new `build_*` in `kestrel_backend.graph`
  (importable by the solver) vs. assembled solver-side? Prefer the seam that keeps
  `kraken-chatbot` unmodified if feasible.
- **[Deferred capability] AstaBench standardized search tools:** A later "fair" variant
  that swaps `literature_grounding`'s 4 APIs for AstaBench's `state.tools`, enabling a
  clean standardized-vs-custom ablation. Out of scope for 6/4; capture as the obvious
  follow-on.

---

## Deliverable 2 — LangGraph Studio Reasoning-Depth Demo

**Shape:** A live Studio run of a single curated query that visibly exercises **one-hop,
multi-hop, and subgraph** reasoning so node-level depth is on screen.
Studio is already wired (`backend/langgraph.json` registers `kraken_discovery`); launch via
`cd backend && langgraph dev`.

**Deck claim:** "Here is the pipeline reasoning at depth — watch one-hop, multi-hop, and
subgraph paths light up per node."

### Requirements

- **S1.** A curated demo query that provably triggers all three reasoning modes: a
  well-characterized entity (→ one-hop + multi-hop in `direct_kg`) and at least one entity
  pair with a connecting path (→ `/subgraph` in integration).
- **S2.** A short narration script mapping each on-screen node to what the audience is
  seeing (which endpoint, which preset, which path type).
- **S3.** A dry-run checklist confirming Studio renders the depth live (env keys present,
  Kestrel reachable, query produces multi-hop + subgraph findings).

### Dependency / Gating (critical)

- **S1–S3 depend on the core R1–R10 build**, specifically **Unit 5** (multi-hop in
  `direct_kg`, R5) and **Unit 7** (`/subgraph` in integration, R6) of
  `docs/plans/2026-05-25-001-feat-kestrel-api-depth-plan.md`. Both are **unchecked** as of
  2026-05-29.
- **If Units 5 & 7 are not merged before 6/4**, the Studio demo can only show the current
  one-hop pipeline. **Fallback (S4):** demo the shipped one-hop pipeline + the AstaBench
  lit-grounding result, and present multi-hop/subgraph as "next" with the plan as evidence.

### Success Criteria

- **If Units 5 & 7 land:** a Studio run (recorded as primary, live only if the S3 dry-run
  passes with margin) in which one-hop, multi-hop, and subgraph findings are each visible in
  the node outputs.
- **If they slip (S4 fallback):** the success criterion **degrades** to — the shipped
  one-hop pipeline is shown in Studio + the AstaBench qualitative comparison, with
  multi-hop/subgraph presented as "next" via the plan. (Note: this is a roadmap, not a
  delivered depth demo — see the strategic risk in Dependencies & Sequencing.)
- The narration script lets a non-author present it.
- A recorded fallback (screen capture) exists in case of live-demo failure.

### Open Questions (deferred to planning)

- **[S1] Which entities?** Needs a known well-characterized entity + a known connected pair
  in the Kestrel KG. Spike during/after core-build implementation (the plan already calls
  for validating multi-hop/subgraph on representative entities — reuse those).
- **[S4] Live vs. recorded:** Decide whether 6/4 shows a live run or a safer recording.

---

## Dependencies & Sequencing (6/4 timeline)

| Deliverable | Depends on | Can start now? |
|---|---|---|
| AstaBench lit-grounding solver | Shipped `literature_grounding`; existing solver scaffold; completed `model_usages` | **Yes — parallel, no core-build dependency** |
| Studio reasoning-depth demo | Core build Units 5 (R5 multi-hop) & 7 (R6 subgraph) | **No — gated on core build** |

**Implication:** Prioritize the core build (Units 5 & 7 at minimum) for the Studio demo to
land, while the AstaBench lit-grounding solver proceeds independently. If the core build
slips, the AstaBench deliverable + the Studio fallback (S4) still give the deck a
substantive story.

## Dependencies / Assumptions

- **Spike VERIFIED (2026-05-29, `tmp/spike_litground.py`):** importing
  `kestrel_backend.graph.nodes.literature_grounding` and calling `run()` directly on a
  hand-crafted `Hypothesis` works **standalone** — no DB, no Kestrel, no full-graph init
  (~0.6s import, 21s run, 0 errors). Returned 3 real, on-topic TREM2/AD citations
  (OpenAlex ×2, PubMed ×1) with valid DOIs. **Confirmed findings:** S2 rate-limited (429,
  contributed 0) under naive parallel calls — pace/curate for L6; and **all snippets were
  empty** (OpenAlex/PubMed `key_passage=""`), empirically validating the L3/L5 deferral.
  Calling the node function directly (not via `build_discovery_graph()`) avoids the DB deps.
- AstaBench at `~/projects/asta-bench/` runs locally in Docker; SQA is a lit task and does
  **not** require the Docker python-session sandbox (unlike DiscoveryBench/E2E). **But the
  stretch scored run (L5) additionally needs:** an LLM grader credential (SQA scorer defaults
  to `google/gemini-2.5-flash`) **and** `HF_TOKEN` for the dataset download — neither is just
  "literature APIs + one LLM call."
- The thin-synthesis + (default-off) classifier both use the **Claude Agent SDK**, which on
  the shared server uses one `~/.claude/` OAuth token across prod and dev (per `CLAUDE.md`);
  if it expires, all SDK calls fail. Run locally / verify token before the deck.
- ScholarQABench2 input/scoring contract per `astabench/evals/sqa/task.py` remains stable.

## Next Steps

→ **De-risking spike first (cheapest, highest-value):** run a single `sqa_dev` sample
  through a direct `thin-synthesis → literature_grounding` call to (a) confirm the reduced
  path runs without import-time DB/Kestrel deps and (b) measure whether it scores
  non-trivially — *before* committing the deck narrative or building the full solver.
→ Set a **go/no-go date** for the Studio depth demo gated on core-build Units 5 & 7.
→ `/ce:plan` for the AstaBench lit-grounding work (scope depends on the open strategic
  decisions below).
→ Core build (Units 5 & 7) gates the Studio demo — sequence accordingly.

## Open Strategic Decisions (from 2026-05-29 document review)

These three need a human call before planning; they are not auto-resolvable:

1. **Descope L1–L4 for 6/4?** The committed primary artifact (L6 qualitative side-by-side)
   needs only a thin direct-call script + a comparison table — *not* a runnable Inspect
   solver (L1), the SQA JSON output adapter (L3), or cost wiring (L4), which serve only the
   stretch scored run (L5). Given L5's score is compromised (thin-answer metrics + corpus
   cutoff + empty snippets), is the solver+adapter worth building for 6/4, or deferred?
2. **L6 anti-anecdote protocol.** A single author-curated biomedical query + KRAKEN's
   biomedical-tuned tools is structurally favorable to KRAKEN; a data-science audience may
   read it as rigged. Pre-register a small fixed query set (chosen by a stated rule, incl.
   ≥1 query where KRAKEN isn't expected to win), or keep it illustrative?
3. **Protect the depth thesis.** The deck's headline (reasoning depth) rests entirely on
   gated, unbuilt Units 5 & 7 with only a roadmap fallback, while the AstaBench arm
   benchmarks a *bypassed* node. Carve a minimal multi-hop+subgraph slice as a hard gate, or
   build a non-gated pre-recorded depth trace, so the headline survives a build slip?
