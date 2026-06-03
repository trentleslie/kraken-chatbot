---
title: "feat: AstaBench × KRAKEN literature-grounding qualitative comparison (L6)"
type: feat
status: active
date: 2026-05-29
origin: docs/brainstorms/kestrel-deck-deliverables-requirements.md
---

# feat: AstaBench × KRAKEN literature-grounding qualitative comparison (L6)

## Overview

Build a standalone script that produces the **primary deck artifact (L6)**: a
qualitative side-by-side showing, for a pre-registered set of 3–5 biomedical questions,
which papers/citations KRAKEN's `literature_grounding` surfaces vs. a **comparison-reference search**
on the same synthesized claims. Per Decision A (hedge, below), the committed reference is a
**non-gated public-search baseline** (labeled "general search"); the **AstaBench MCP search arm** is an
optional swap-in if `ASTA_TOOL_KEY` is provisioned. The output is a static, pre-computed deck
slide (markdown table + JSON snapshot) — no scoring, no fairness claim.

The de-risking spike (2026-05-29, `tmp/spike_litground.py`) already proved the load-bearing
assumption: `literature_grounding.run()` executes standalone (no DB/Kestrel) and returns
real, on-topic citations. This plan turns that spike into a re-renderable, multi-query
comparison artifact.

## Problem Frame

The 6/4 data-science deck needs evidence the discovery pipeline's literature grounding is
sound. Wrapping the whole pipeline against a public benchmark scores off-task; scoping to the
`literature_grounding` node maps onto AstaBench's Literature category. Per the descope
decision (see origin), the committed 6/4 work is **not** a scored Inspect solver but this
**qualitative comparison** — the lowest-risk artifact that still uses AstaBench's tooling as
the comparison reference. (See origin: `docs/brainstorms/kestrel-deck-deliverables-requirements.md`)

## Requirements Trace

- **L2** (minimal form). Thin-synthesis bridge: biomedical question → a few claims as frozen
  `Hypothesis` objects, via one Claude Agent SDK call. Direct function call, no Inspect harness.
- **L6** (primary). For a curated query set, run KRAKEN's `literature_grounding` AND a
  comparison-reference search (committed: non-gated public baseline; optional: AstaBench MCP arm)
  **on the same synthesized claims**, and produce a side-by-side of the papers each surfaces
  (title, source, year, DOI). Pre-computed static slide.
- **Query protocol** (decision #2, resolved). The query set is **pre-registered**: 3–5
  biomedical questions chosen by a stated rule *before* seeing results, including ≥1 query
  where KRAKEN is **not** expected to win, to defuse the cherry-pick read.

## Scope Boundaries

- **In scope:** A standalone comparison script (thin synthesis → `literature_grounding` →
  AstaBench-tools arm → tabulation); a pre-registered query manifest; a reproducible
  markdown + JSON deck artifact.
- **Out of scope (true non-goals):** Scoring against the SQA rubric; any fairness/causal
  claim; modifying `literature_grounding` internals; the full-pipeline solver.

### Deferred to Separate Tasks

- **Inspect solver L1/L3/L4 + scored run L5** — deferred post-6/4 (see origin); their signal
  is compromised (thin-answer metrics + 2025-05 corpus cutoff + empty snippets).
- **Studio reasoning-depth demo (Deliverable 2)** — gated on core-build Units 5 & 7; separate
  plan (`docs/plans/2026-05-25-001-feat-kestrel-api-depth-plan.md`).
- **Strategic decision #3 (protect the depth thesis)** — belongs to the Studio/core-build
  track, not this L6 plan.

## Context & Research

### Relevant Code and Patterns

- **Thin-synthesis SDK call:** `query_with_usage(prompt, options, node_name) -> (text,
  ModelUsageRecord|None)` in `backend/src/kestrel_backend/graph/sdk_utils.py`. Gate on
  `HAS_SDK` (raises `RuntimeError` when false). Mirror the direct-construction pattern in
  `backend/src/kestrel_backend/graph/nodes/synthesis.py` (SDK call site): `ClaudeAgentOptions(
  system_prompt=..., allowed_tools=[], max_turns=1, permission_mode="bypassPermissions")`.
  `query_with_usage` joins text blocks with `""`. Wrap in try/except and fall back on SDK error.
- **Hypothesis construction:** mirror `extract_hypotheses()` in
  `backend/src/kestrel_backend/graph/nodes/synthesis.py`. `Hypothesis` (in
  `backend/src/kestrel_backend/graph/state.py`, `frozen=True`) requires `title`, `tier`,
  `claim`, `supporting_entities`, `structural_logic`, `validation_steps`; optional
  `contradicting_entities=[]`, `confidence="moderate"`, `validation_gap_note=""`,
  `literature_support=[]`.
- **Node invocation:** `literature_grounding.run(state)` in
  `backend/src/kestrel_backend/graph/nodes/literature_grounding.py`. Input contract
  (`backend/src/kestrel_backend/graph/state_contracts.py` `LiteratureGroundingInput`) requires
  only `hypotheses: list[Any]`; `_ContractBase` ignores extras, so `{"hypotheses": [...]}`
  validates. Returns `{"hypotheses": [...grounded with .literature_support], "literature_errors":
  [...], "synthesis_report": str}`.
- **Citation fields for the table:** `LiteratureSupport` (in `state.py`): `title`, `authors`,
  `year`, `doi`, `url`, `relevance_score`, `relationship`, `key_passage`, `citation_count`,
  `source` (`kg|openalex|s2|exa|pubmed`). `build_references_table(hypotheses)` in
  `literature_grounding.py` is a ready-made markdown formatter to reuse or mirror.
- **Config:** instantiate/observe `LiteratureGroundingConfig` explicitly; `use_llm_classifier`
  defaults `False` (keep it — relationships default to `supporting`, and the LLM path is an
  untracked SDK call).
- **dotenv precedent:** `backend/src/kestrel_backend/config.py` calls `load_dotenv()`.
- **AstaBench tool factories** (repo `~/projects/asta-bench`, not this repo):
  `astabench/tools/native_provider_tools.py` `make_native_search_tools(inserted_before)` and
  `astabench/tools/asta_tools.py` `make_asta_mcp_tools(insertion_date)` — both return Inspect
  `Tool` objects (web_search-backed / MCP-server-backed). SQA uses `inserted_before="2025-05"`.

### Institutional Learnings

- `docs/solutions/runtime-errors/triage-tier2-undefined-variable-2026-05-06.md` — standalone
  node invocation (no Kestrel) exercises the **under-tested Tier-2 SDK fallback** path. Run
  `ruff check --select F821` over `literature_grounding.py` and the new glue before trusting output.
- `docs/solutions/best-practices/langgraph-pipeline-production-formalization.md` — Claude SDK
  `query()` is **non-deterministic even when external HTTP is mocked**; snapshot the synthesized
  claims yourself so the deck artifact is reproducible. `LiteratureGroundingConfig` is the
  canonical config-flag example.
- `docs/solutions/best-practices/discovery-pipeline-one-graph-methods-within-nodes-2026-05-28.md`
  — a state field written once and read-only is a plain dict (no reducer); the minimal input
  dict should not rely on reducer accumulation. Tagged `astabench`.
- No prior art exists for **S2 429 rate-limit handling** (the spike hit 429), `model_usages`
  mechanics, or AstaBench/Inspect integration — these are `ce:compound` candidates afterward.

### External References

- AstaBench README (`~/projects/asta-bench/README.md`) — lit tasks provide standardized search
  tools with date/corpus restrictions; alternative tools "may score lower." For L6 (qualitative,
  unscored) this is a labeling caveat, not a scoring concern.

## Key Technical Decisions

- **Script home: `backend/scripts/` (new dir), run via `uv run` from `backend/`.** It needs the
  `kestrel_backend` package + deps + `backend/.env`. `tmp/` lacks the env; the solver repo wraps
  the whole graph (wrong layer); `backend/tests/` is for pytest, not a live-API script.
- **`load_dotenv()` before importing `kestrel_backend.graph`.** `EXA_API_KEY` and `S2_API_KEY`
  are captured as module-level constants at import time (`exa_client.py`, `semantic_scholar.py`);
  importing the node before loading `.env` silently disables those sources. This ordering is a
  hard constraint.
- **Snapshot everything to JSON (audit trail).** Because SDK synthesis + live literature APIs are
  non-deterministic, persist the synthesized claims and both arms' raw results to a JSON file with a
  capture timestamp. This makes the slide **deterministically re-renderable from the frozen capture**
  — **not** re-runnable to the same result (a data-science audience must not read "reproducible" as
  "regenerable"). The slide carries a one-line provenance note: point-in-time snapshot, captured
  <date>.
- **AstaBench arm: the two factories are NOT interchangeable (review, code-verified).** Split them:
  - `make_asta_mcp_tools(insertion_date="2025-05")` returns **directly-awaitable MCP `Tool`s** (it
    opens an MCP streamable-HTTP connection; no Inspect `TaskState`/sandbox needed). This is the
    viable standalone path — **but requires `ASTA_TOOL_KEY`**, and raises `ValueError` (not a falsy
    return) when the key is unset. Call the factory **once** and reuse the tool list (it opens a
    connection per call via an internal threaded `asyncio.run`).
  - `make_native_search_tools(inserted_before="2025-05")` returns provider-side `web_search` `Tool`s
    that are **resolved inside model `generate()`** and are **NOT standalone-awaitable** — the
    Anthropic/OpenAI/Gemini providers raise `ValueError("No valid provider found")` outside a solver.
    The native arm therefore requires the **`inspect eval` + configured model** path, not direct await,
    and needs a provider model key + (for `sqa_dev`) `HF_TOKEN` for the gated dataset.
  - **Credentials are the gating risk:** `ASTA_TOOL_KEY` and `HF_TOKEN` are **absent from
    `backend/.env`** (verified). Both arm mechanisms are blocked until they are provisioned — see
    Open Strategic Decisions.
  - Wrap factory calls in `try/except (ValueError, RuntimeError)` and record an explicit
    "no AstaBench results (reason)" cell on failure — do **not** rely on a falsy return.
- **Keep `use_llm_classifier=False`.** Avoids untracked SDK calls inside the node and keeps the
  KRAKEN arm purely HTTP; relationships default to `supporting`.
- **Pace S2.** The spike hit 429 under naive parallel calls. Run queries sequentially with a
  delay (the node's own `S2_DELAY` is 1.1s with key / 10s without); accept that S2 may contribute
  little and that OpenAlex/PubMed/Exa carry the KRAKEN arm.

## Open Questions

### Resolved During Planning

- **Does the reduced path run standalone?** Yes — spike-verified (import 0.6s, run 21s, 0 errors,
  real citations).
- **Query protocol?** Pre-registered 3–5 set with ≥1 non-favorable query (decision #2).
- **Where does the script live?** `backend/scripts/` (research-confirmed convention fit).
- **`use_llm_classifier`?** `False` (default) for the deck run.

### Deferred to Implementation

- **Exact thin-synthesis granularity** — 1 claim vs. a few per question. Start with 1–3 claims;
  tune by eyeballing whether grounding produces a legible spread of papers.
- **S2 contribution under pacing** — may still 429; acceptable, documented as a result note.
- **MCP-tool result schema** — confirm what `make_asta_mcp_tools` results actually contain
  (title/doi/year may be sparse) by spiking one question before wiring the normalizer.

## Strategic Decisions (from 2026-05-30 plan review — RESOLVED)

- **Decision A — comparison arm (RESOLVED: hedge).** Credential spike (2026-05-30) confirmed
  `ASTA_TOOL_KEY` and `HF_TOKEN` are **not provisioned anywhere on the system** and require out-of-band
  action (AI2 key request + gated HF license), and `make_native_search_tools` isn't standalone-awaitable.
  **Committed path:** a **non-gated public-search baseline** (KRAKEN's own S2/OpenAlex clients used
  plainly) as the comparison reference, labeled "general search" — *not* "AstaBench." **Optional swap-in:**
  the AstaBench MCP arm (`make_asta_mcp_tools`, falls back to baseline on missing key) if `ASTA_TOOL_KEY`
  is provisioned before the deck.
- **Decision B — input asymmetry (RESOLVED: like-with-like).** Both arms consume the **same
  synthesized claim strings** (not raw question vs. claims), so the side-by-side compares retrieval
  behavior on identical inputs. The remaining difference is the search/grounding logic, which is what
  the slide is about. Note the thin-synthesis hop as a stated limitation on the slide.

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not
> implementation specification. The implementing agent should treat it as context, not code to
> reproduce.*

```
load_dotenv()                      # BEFORE importing kestrel_backend.graph
  ↓
query manifest (3–5 pre-registered Qs, ≥1 non-favorable, with selection rule)
  ↓  for each question:
  ├── thin_synthesis(Q) ── Claude SDK (query_with_usage, allowed_tools=[]) ──► [Hypothesis(claim),...]
  │                                                    │  (SAME claim strings feed both arms — Decision B)
  ├── KRAKEN arm:    literature_grounding.run({"hypotheses":[...]}) ──► LiteratureSupport[]  (source-tagged)
  │
  └── Comparison arm (same claims):
        committed: baseline = plain S2/OpenAlex search ──► papers[]   (no creds; label "general search")
        optional : make_asta_mcp_tools(insertion_date="2025-05") direct-await ──► papers[]  (iff ASTA_TOOL_KEY)
  ↓
snapshot {question, claims, kraken_results, astabench_results} → JSON
  ↓
render side-by-side markdown table (per query: what each surfaced, overlap) → deck slide
```

## Implementation Units

- [ ] **Unit 1: Script scaffold + env loading + pre-registered query manifest**

**Goal:** Create `backend/scripts/` with the comparison-script entry point, correct
`.env`-before-import ordering, and a pre-registered query manifest.

**Requirements:** L6 (query protocol)

**Dependencies:** None

**Files:**
- Create: `backend/scripts/litground_comparison.py` (entry: `main()`, `asyncio.run`)
- Create: `backend/scripts/comparison_queries.py` (or a `.json`) — the query manifest
- Test: `backend/tests/test_litground_comparison.py`

**Approach:**
- Call `load_dotenv()` at the very top, **before** any `from kestrel_backend.graph...` import
  (enforce by import order / a small loader module imported first).
- Manifest: 3–5 biomedical questions, each with `{id, question, rationale, expected_kraken_edge:
  bool}`. Encode the selection rule in a module docstring/comment: questions chosen before
  running; ≥1 with `expected_kraken_edge=False` (a broad/methodological question where general
  search should do at least as well).
- CLI: optional `--query-id` to run one, default runs all; `--out` path for the JSON snapshot.

**Patterns to follow:**
- `load_dotenv()` usage in `backend/src/kestrel_backend/config.py`.
- `uv run` invocation convention from `backend/CLAUDE.md`.

**Test scenarios:**
- Happy path: manifest loads and exposes 3–5 entries; each has the required keys.
- Edge case: at least one entry has `expected_kraken_edge=False` (asserts the anti-anecdote rule).
- Edge case: `--query-id` selects exactly one question; unknown id errors clearly.

**Verification:**
- `cd backend && uv run python scripts/litground_comparison.py --help` runs; manifest validates;
  no `kestrel_backend` import occurs before `load_dotenv()` (grep/import-order check).

---

- [ ] **Unit 2: Thin-synthesis bridge (question → Hypothesis claims)**

**Goal:** One Claude Agent SDK call turns a question into 1–3 claims, built as frozen
`Hypothesis` objects and snapshotted.

**Requirements:** L2 (minimal form)

**Dependencies:** Unit 1

**Files:**
- Modify: `backend/scripts/litground_comparison.py` (add `thin_synthesis(question) -> list[Hypothesis]`)
- Test: `backend/tests/test_litground_comparison.py`

**Approach:**
- Build `ClaudeAgentOptions(system_prompt=..., allowed_tools=[], max_turns=1,
  permission_mode="bypassPermissions")`; call `query_with_usage(prompt, options,
  node_name="thin_synthesis")`; gate on `HAS_SDK`; try/except with a clear error on SDK failure.
- Parse the response into 1–3 claim strings; construct `Hypothesis` per claim mirroring
  `extract_hypotheses()` — **`title`** = first ~10 words of the claim, truncated (mirrors how
  `extract_hypotheses()` derives short titles; `title` is a required field), `claim` = full claim
  string, `supporting_entities=[]`, `structural_logic="Synthesized from the question (thin synthesis
  bridge)."`, `validation_steps=["Literature corroboration via multi-source search."]`, `tier=1`,
  `confidence="moderate"`, `literature_support=[]` (the node fills it).
- Snapshot the question + synthesized claims into the run record.

**Execution note:** Snapshot the synthesized claims to JSON — SDK output is non-deterministic, and
the deck artifact must be reproducible from the snapshot.

**Patterns to follow:**
- SDK call site + `ClaudeAgentOptions` construction in `synthesis.py`.
- `extract_hypotheses()` Hypothesis construction in `synthesis.py`.

**Test scenarios:**
- Happy path (SDK mocked): a stubbed response string → ≥1 valid frozen `Hypothesis` with all
  required fields populated; `claim` non-empty.
- Edge case: response yielding zero parseable claims → raises/handled with a clear message, not a
  silent empty list.
- Edge case (`HAS_SDK=False`): surfaces a clear "SDK unavailable" error rather than an opaque crash.

**Verification:**
- With SDK available, `thin_synthesis("...TREM2...")` returns frozen Hypotheses; claims captured
  in the snapshot.

---

- [ ] **Unit 3: KRAKEN arm — invoke `literature_grounding` standalone**

**Goal:** Run `literature_grounding.run()` on the synthesized claims and normalize the returned
`LiteratureSupport` into comparison rows.

**Requirements:** L6 (KRAKEN arm)

**Dependencies:** Unit 2

**Files:**
- Modify: `backend/scripts/litground_comparison.py` (add `run_kraken_arm(hypotheses) -> list[dict]`)
- Test: `backend/tests/test_litground_comparison.py`

**Approach:**
- Build minimal state `{"hypotheses": [...]}`; `await literature_grounding.run(state)`.
- Extract `result["hypotheses"][*].literature_support`; normalize each `LiteratureSupport` to a
  row `{title, source, year, doi, url, relationship, key_passage, relevance_score}`.
- Pace S2: run queries sequentially; tolerate 429 (record `literature_errors`); do not fail the run.
- Keep `use_llm_classifier=False`.

**Execution note:** Standalone invocation exercises the under-tested Tier-2 SDK fallback path —
run `ruff check --select F821` over `literature_grounding.py` and this glue before trusting output
(per the triage-NameError learning).

**Patterns to follow:**
- `tmp/spike_litground.py` (the verified working invocation) as the reference shape.
- `build_references_table()` for table/normalization cues.

**Test scenarios:**
- Happy path (pure): given a `Hypothesis` carrying known `literature_support`, the normalizer
  emits one correctly-typed row per citation with the right `source` tag.
- Edge case: empty `key_passage` (KG/PubMed/OpenAlex sources) → row still emitted, `key_passage=""`
  handled (not dropped) for the qualitative table.
- Error path: `literature_errors` populated (e.g., S2 429) → recorded in the snapshot, run continues.
- Integration (live, manual): one real question yields ≥1 real citation with a valid DOI (as the
  spike showed).

**Verification:**
- For the manifest queries, KRAKEN rows are produced and captured in the snapshot; 429s are noted,
  not fatal.

---

- [ ] **Unit 4: AstaBench arm — standardized search tools (spike-first)**

**Goal:** Obtain, for each question, the papers a **comparison-reference search** surfaces,
normalized into the shared comparison-row schema. Per Decision A (hedge), the committed reference is
a **non-gated public-search baseline**; the AstaBench MCP arm is an **optional swap-in** if
`ASTA_TOOL_KEY` is provisioned in time.

**Requirements:** L6 (comparison arm)

**Dependencies:** Unit 2 (consumes the same synthesized claims — see Decision B)

**Files:**
- Modify: `backend/scripts/litground_comparison.py` (add `run_baseline_arm(claims) -> list[dict]`
  and an optional `run_astabench_arm(claims)`)
- Test: `backend/tests/test_litground_comparison.py`

**Approach — committed: non-gated public-search baseline:**
- Reuse KRAKEN's own non-gated search clients as a *plain* baseline (no grounding logic): e.g.
  `backend/src/kestrel_backend/semantic_scholar.py` `search_papers` and/or `openalex.py`
  `search_works`, queried directly with the **same input the KRAKEN arm grounds** (Decision B:
  like-with-like — feed the baseline the synthesized claim strings, not the raw question).
- Date-bound the baseline to a 2025-05 cutoff where the client supports it, for rough parity.
- Normalize to `{title, source: "baseline", year, doi/url, snippet}` (shared schema).
- Label the column honestly: "general public search," not "AstaBench."

**Approach — optional AstaBench MCP swap-in (only if `ASTA_TOOL_KEY` arrives):**
- Call `make_asta_mcp_tools(insertion_date="2025-05")` **once**, reuse the returned MCP `Tool`s, and
  `await` them per claim. These are genuinely standalone-awaitable (MCP connection, no Inspect solver
  context). (`make_native_search_tools` is provider-side `web_search`, **not** standalone-awaitable —
  do not use it for this script.)
- Wrap factory construction + calls in `try/except (ValueError, RuntimeError)` — the factory **raises**
  (missing `ASTA_TOOL_KEY` → `ValueError`), it does not return falsy. On failure, **fall back to the
  baseline arm**, never crash. Record which arm produced the results in the snapshot metadata.

**Execution note:** Build the baseline arm first (no new creds, low risk). If `ASTA_TOOL_KEY` is
provisioned, **spike the MCP direct-await path against ONE claim** (it was not covered by
`tmp/spike_litground.py` — least-verified integration) before wiring all queries; time-box it.

**Patterns to follow:**
- `~/projects/asta-bench/astabench/evals/sqa/task.py` for how the tools are wired with
  `inserted_before` and `use_tools`.

**Test scenarios:**
- Happy path (pure): given a fixture baseline-search payload, the normalizer emits correctly-typed
  `baseline` rows sharing the comparison schema.
- Edge case (optional AstaBench path): `make_asta_mcp_tools` **raises** `ValueError` (missing
  `ASTA_TOOL_KEY`) → caught, **falls back to the baseline arm**, no crash. (Factories raise, they do
  not return falsy.)
- Edge case: search returns zero papers → empty-with-reason cell, run continues.
- Integration (live, manual): one claim returns ≥1 baseline paper; the baseline gets the **same
  claim input** as the KRAKEN arm (Decision B).

**Verification:**
- For the manifest queries, AstaBench rows are produced (or an explicit empty-with-reason is
  recorded); the working mechanism is noted in the snapshot.

---

- [ ] **Unit 5: Comparison tabulation + deck artifact**

**Goal:** Merge both arms into a per-query side-by-side, compute simple overlap, and emit the
reproducible deck artifact (markdown + JSON snapshot).

**Requirements:** L6 (primary)

**Dependencies:** Units 2, 3, 4

**Files:**
- Modify: `backend/scripts/litground_comparison.py` (add `build_comparison(snapshot) -> str` +
  snapshot writer)
- Test: `backend/tests/test_litground_comparison.py`

**Approach:**
- For each question, render two columns (KRAKEN vs AstaBench) listing surfaced papers
  (**title, source, year, DOI**); compute overlap by normalized DOI then title+year. `key_passage`
  is captured in the JSON snapshot but kept out of the slide table for legibility; `relationship` is
  omitted from the table because with `use_llm_classifier=False` it is constant (`supporting`) for
  the KRAKEN arm and carries no signal — the slide must not imply relationship classification was performed.
- Emit: (1) a markdown table per question (the slide content) + a short header stating the
  pre-registration rule, the input-asymmetry limitation (see Decision B), the custom-toolset/date
  caveats, and the snapshot provenance note; (2) the full JSON snapshot (questions, claims, both
  arms' raw rows incl. `key_passage`, overlap, which comparison arm ran (baseline vs AstaBench MCP),
  errors).
- Frame as "qualitative difference in retrieval behavior," not superiority; surface the
  `expected_kraken_edge=False` query result honestly even if KRAKEN does worse.

**Patterns to follow:**
- `build_references_table()` markdown-table construction in `literature_grounding.py`.

**Test scenarios:**
- Happy path: two row-lists → a markdown table with both columns; counts correct.
- Edge case: overlap detection — same paper via different sources (DOI match; missing-DOI
  title+year match) is counted once as overlap.
- Edge case: one arm empty → table renders the populated side + an explicit empty-with-reason cell.
- Happy path: JSON snapshot round-trips (write then re-read reproduces the same table).

**Verification:**
- Running the full script over the manifest writes a JSON snapshot and prints/saves the markdown
  slide. A render-from-snapshot path (no live calls) is a low-cost nice-to-have for re-rendering the
  frozen capture — not a hard requirement; do not build a second divergent code path for it.

## System-Wide Impact

- **Interaction graph:** Standalone script under `backend/scripts/`; imports `literature_grounding`
  + `sdk_utils` + `state` but does **not** run the graph, FastAPI, DB, or Kestrel. No production
  code paths change.
- **Error propagation:** Live-API failures (S2 429, missing Exa/Asta creds) are captured into the
  snapshot and rendered as explicit "empty-with-reason" cells; the script never hard-fails on a
  single source.
- **State lifecycle risks:** None persistent — the only writes are the JSON snapshot + markdown
  artifact. `Hypothesis`/`LiteratureSupport` are frozen; use `model_copy` (the node already does).
- **API surface parity:** None — no shared API changes. The script is additive and isolated.
- **Integration coverage:** The live arms (Units 3, 4) are integration/manual; the testable value
  is the pure normalization + tabulation logic (Units 3–5 helpers) tested against fixtures.
- **Testing scope (right-sized for a deck script):** Keep automated tests to the two pure
  normalizers — Unit 3 row-shape and Unit 5 merge/overlap-dedup — plus the Unit 2 happy-path with a
  mocked SDK. Skip CLI-arg, `HAS_SDK=False`, and JSON round-trip tests; they are not load-bearing for
  a one-run artifact. **A green test run does NOT mean the artifact is achievable** — that depends on
  the manual Unit 4 MCP spike + live-API yield, which tests cannot cover.
- **Unchanged invariants:** `literature_grounding` internals, `LiteratureGroundingConfig` defaults
  (`use_llm_classifier=False`), the existing solver, and the `model_usages` instrumentation are all
  untouched.

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| AstaBench creds (`ASTA_TOOL_KEY`/`HF_TOKEN`) unprovisioned (verified) | **Resolved (Decision A): hedge** — committed non-gated baseline; AstaBench MCP arm is an optional swap-in that falls back to baseline if the key is missing |
| `make_native_search_tools` is NOT standalone-awaitable (provider-side `web_search`) | Do not use it; the optional AstaBench arm uses the **MCP** factory only |
| AstaBench MCP factory **raises** (`ValueError`) on missing key, not falsy | `try/except` → fall back to the baseline arm; never crash |
| Asta MCP / S2 own rate limits (429/504) | Internal retries + pacing; tolerate residual as empty-with-reason |
| Input asymmetry between arms | **Resolved (Decision B):** both arms consume the same synthesized claim strings (like-with-like) |
| S2 429 rate-limiting yields a thin KRAKEN arm | Sequential pacing + `S2_DELAY`; OpenAlex/PubMed/Exa carry the arm; record 429s as result notes, not failures |
| `.env` loaded after importing the node → EXA/S2 silently disabled | Hard ordering rule: `load_dotenv()` before any `kestrel_backend.graph` import; Unit 1 import-order check |
| Standalone run hits an untested Tier-2 SDK fallback bug | `ruff check --select F821` over the node + glue (per triage-NameError learning) before relying on output |
| SDK + live APIs are non-deterministic → slide not reproducible | Snapshot claims + raw results to JSON; render the slide from the snapshot |
| Missing AstaBench creds (`ASTA_TOOL_KEY`/provider keys) | Detect falsy tool factories; record explicit "no AstaBench results (missing creds)"; degrade, don't crash |
| Comparison reads as cherry-picked | Pre-registered 3–5 query set with ≥1 `expected_kraken_edge=False`; print the selection rule + non-favorable result on the slide |

## Documentation / Operational Notes

- After the deck, consider `ce:compound` for two current knowledge gaps: the standalone-node
  invocation recipe and S2 429 handling (both flagged as missing prior art).
- The JSON snapshot is the durable artifact; keep it alongside the slide so the comparison is
  auditable.

## Sources & References

- **Origin document:** [docs/brainstorms/kestrel-deck-deliverables-requirements.md](docs/brainstorms/kestrel-deck-deliverables-requirements.md)
- Verified spike: `tmp/spike_litground.py`
- Related code: `backend/src/kestrel_backend/graph/nodes/literature_grounding.py`,
  `backend/src/kestrel_backend/graph/nodes/synthesis.py`,
  `backend/src/kestrel_backend/graph/sdk_utils.py`,
  `backend/src/kestrel_backend/graph/state.py`
- AstaBench tools: `~/projects/asta-bench/astabench/tools/native_provider_tools.py`,
  `~/projects/asta-bench/astabench/tools/asta_tools.py`,
  `~/projects/asta-bench/astabench/evals/sqa/task.py`
- Institutional learnings: `docs/solutions/runtime-errors/triage-tier2-undefined-variable-2026-05-06.md`,
  `docs/solutions/best-practices/langgraph-pipeline-production-formalization.md`,
  `docs/solutions/best-practices/discovery-pipeline-one-graph-methods-within-nodes-2026-05-28.md`
