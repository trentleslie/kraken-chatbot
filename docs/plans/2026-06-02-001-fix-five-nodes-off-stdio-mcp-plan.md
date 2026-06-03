---
title: "fix: Migrate the five remaining SDK nodes off the broken stdio MCP (#61)"
type: fix
status: active
date: 2026-06-02
deepened: 2026-06-02
---

# fix: Migrate the five remaining SDK nodes off the broken stdio MCP (#61)

## Overview

PR #60 fixed `pathway_enrichment` — one of six discovery-pipeline nodes that configure the Kestrel **stdio MCP** server via `uvx mcp-client-kestrel`, a package that does not exist on PyPI (`graph/sdk_utils.py:35-36`; root-cause record in `graph/nodes/cold_start.py:1-18`; note `cold_start` is *not* one of the six — it was migrated earlier and already runs HTTP-only with `allowed_tools=[]`). The stdio subprocess can never launch, so on every SDK call these nodes pay a doomed-spawn cost, and when their SDK tier fires it produces output not grounded in the knowledge graph. (The *degree* of fabrication varies — e.g. triage's prompt funnels a failed query to `edge_count: 0`, so it degrades toward a default rather than a confident wrong number — but in no case is the SDK tier doing real KG work.)

This migrates the **five remaining** nodes — `entity_resolution`, `direct_kg`, `triage`, `integration`, `temporal` — off the broken path, **right-sized per node** (not a blanket copy of the pathway_enrichment template), and retires the dead `get_kestrel_mcp_config` stdio path. Verified (issue #61): all five are low-medium severity — their authoritative outputs come from HTTP Tier 1; the broken SDK degrades only fallback/supplementary tiers. The goal is a correctness guarantee: **no SDK tier can silently surface a fabricated CURIE, association, or edge-count as a KG fact.** (Temporal *direction* and integration *gap* judgments are intentionally retained model reasoning over already-grounded findings — they annotate/relabel real CURIE-bearing findings, and gap CURIEs are **nulled/membership-checked at the data boundary (R1b)** so a gap can never surface a model-authored CURIE. The guarantee is about fabricated KG *facts*, not about eliminating all model inference.)

## Problem Frame

Each node requests `mcp__kestrel__*` tools in `allowed_tools` and passes `mcp_servers={"kestrel": get_kestrel_mcp_config()}`. The tools never register. Two distinct shapes exist:

- **Tier-2 fallbacks** (`entity_resolution`, `direct_kg`, `triage`): the SDK tier fires *only when HTTP Tier 1 already failed* for an entity. It uses MCP tools to re-query the KG and reason. A naive "HTTP prefetch" must therefore use a *different/broader* query than the one that just failed, or it adds nothing.
- **Reason-over-context** (`temporal`, `integration` gap analysis): the SDK already receives the data it needs in the prompt (findings summaries / study context) and uses MCP tools only to *validate/expand*. These don't need a prefetch — they need the tools removed.

`integration` is the subtlest: its **bridges already come from HTTP** (`api_bridges` = `detect_bridges_via_api` (`integration.py:617`) + `detect_subgraphs_via_api` (`integration.py:624`), concatenated at `:628`; both call `call_kestrel_tool`); only the speculative **gap analysis** (tier-3 "expected but absent") uses the broken SDK (`integration.py:680-699`, which parses *gaps only* and discards the LLM bridges with `_`).

## Requirements Trace

- R1. No SDK tier may present hallucinated training-data as KG facts — each either uses real HTTP data, reasons only over already-provided context, or honestly fails/degrades.
- R1a. **(closes R1 for the non-empty `entity_resolution` case)** Any CURIE returned by the select-from-candidates SDK call MUST be validated as a member of the prefetched candidate set; a CURIE the model emits that was not in the candidates is treated as `method="failed"`, never surfaced as a resolved fact. (The empty-candidate guard alone does not close this — see Key Decisions.)
  - **Membership semantics (required, or R1a doesn't actually close the path):** compare on **exact CURIE strings after a single canonical normalization applied identically to both sides** (e.g. uppercase the prefix, trim) — **fuzzy/equivalent-namespace matching is prohibited**, since a lenient match could admit a *different* node that merely normalizes alike. On a match, **surface the matched candidate's own `curie`/`name`/`category`, never the model's emitted strings** (the model selects; it does not author the fact). Key the candidate set by the canonical form so lookup is O(1) and the surfaced object is always the prefetched one.
- R1b. **(closes R1 for the `integration` gap path)** Integration gap entities MUST NOT surface a model-authored CURIE as a KG fact. **Construct each `GapEntity` with `curie` nulled** (or kept only if it's a member of the already-grounded in-prompt finding CURIEs — same discipline as R1a). Since gaps are by definition "expected but *absent* from these findings," unconditional nulling is acceptable and simplest. **`GapEntity` is `frozen=True` (`state.py:183`), so apply this at construction inside `parse_integration_result` (pass `curie=None`/the gated value into the constructor) — NOT by post-parse mutation like `gap.curie = None`, which raises `ValidationError`** (alternatively rebuild via `model_copy(update={"curie": None})`). A model can otherwise emit a plausible training-data CURIE that `parse_integration_result` preserves (`integration.py:518`, `curie=g.get("curie")`) into the unvalidated `GapEntity.curie` (`state.py:187`), which `synthesis.py:496` renders as a backtick CURIE indistinguishable from a real one — the same fabrication class as R1a. (Consumer audit confirmed: `synthesis.py:496/506` is the only renderer of a gap CURIE; `literature_grounding` reads `resolution.curie`, not gap CURIEs; `node_detail_extractors` builds gap items without curie — so nulling at construction covers every consumer.)
- R2. Stop configuring the nonexistent stdio MCP server on every SDK call (remove `mcp_servers` + the `mcp__*` allowlist), eliminating the failed-spawn overhead.
- R3. Reuse PR #60's shared pieces — `query_with_usage` diagnostics, `classify_mcp_degradation`, and the `cold_start`/`pathway_enrichment` HTTP-prefetch + `allowed_tools=[]` data-in-prompt pattern. No new abstractions.
- R4. Preserve each node's authoritative HTTP Tier-1 output and its findings `source` tags unchanged.
- R5. Retire `get_kestrel_mcp_config` + `KESTREL_COMMAND`/`KESTREL_ARGS` from `graph/sdk_utils.py` once no graph node references them.

## Scope Boundaries

- Not changing any node's **Tier-1 HTTP** logic, output schema, or finding `source` strings.
- Not changing `pathway_enrichment` (done in #60) or `cold_start` (already HTTP, `allowed_tools=[]`).
- Not touching classic-mode `agent.py` — it has its *own* `_get_kestrel_mcp_config` (`agent.py:86`) and stdio config; separate surface.
- Not changing the LangGraph topology, state schema, or reducers.
- **Single PR** (user decision) covering all five nodes + the `sdk_utils` cleanup.

## Context & Research

### Relevant Code and Patterns

- **Template (prefetch + data-in-prompt):** `graph/nodes/pathway_enrichment.py` (`prefetch_one_hop_neighbors`, `_build_inference_user_prompt`, the emptiness guard, `allowed_tools=[]`, `SDK_SEMAPHORE`) and `graph/nodes/cold_start.py:148-225` (`get_entity_connections`).
- **Shared guard/diagnostics:** `graph/sdk_utils.py` — `classify_mcp_degradation`, the `query_with_usage` `mcp_tool_calls`/`available_tools` instrumentation (from #60).
- **Per-node SDK tiers to migrate:**
  - `entity_resolution.py:259-320` (`resolve_single_entity`, prompt `f"Resolve: {entity}"`, tools hybrid_search/text_search/get_nodes/get_node_info/get_neighbors); Tier-1 HTTP is `resolve_via_api` (`:96`, `call_kestrel_tool("hybrid_search", ...)`).
  - `direct_kg.py:536-600` (`analyze_single_entity`, Tier-2 `llm_fallback`); Tier-1 HTTP is `analyze_via_api` (`:292`, 6 `call_kestrel_tool("one_hop_query")` calls).
  - `triage.py:167-235` (`count_edges_single`); Tier-1 HTTP is `count_edges_via_api` (`:64`, `call_kestrel_tool("one_hop_query", mode="preview")`).
  - `integration.py:680-699` (gap-analysis SDK call); bridges via `detect_subgraphs_via_api` (`:414`).
  - `temporal.py:290-310` (SDK classify, `allowed_tools=["mcp__kestrel__one_hop_query"]  # Minimal - just for validation`), reasons over `build_study_context_for_temporal` (`:138`).

### Institutional Learnings

- `docs/solutions/best-practices/sdk-stdio-mcp-unavailable-migrate-to-http-data-in-prompt-2026-06-02.md` — the migration + degradation-guard pattern from #60.
- `docs/solutions/best-practices/discovery-pipeline-one-graph-methods-within-nodes-2026-05-28.md` — two-tier (HTTP + SDK) node design context.
- `docs/plans/2026-06-01-001-fix-pathway-enrichment-mcp-degradation-plan.md` — the #60 plan this follows.

## Key Technical Decisions

- **Three outcomes, assigned by what the SDK tier actually needs** (not one template):
  1. **Drop the tools** where the data is already in-context — `temporal` (validation-only tool) and `integration` gap analysis (reasons over the provided findings summaries). *The load-bearing safety argument is that the stdio MCP was **always broken**, so these tools never registered — gap analysis has therefore only ever reasoned over the in-prompt context, and dropping the (already-dead) tool config is behaviorally inert. We do not rely on the weaker, runtime-unverifiable claim "the model chose not to call the tools."*
  2. **Drop the fallback to the honest existing default** where a prefetch would re-fail — `direct_kg` and `triage`. **The "same endpoint, same args → re-fails" premise is asymmetric across these two nodes and must not be bundled:** `analyze_via_api` (direct_kg) returns `None` *only* from its top-level `except` — a total failure, so a fresh prefetch genuinely re-fails. `count_edges_via_api` (triage), by contrast, returns `None` on per-call `isError`/empty/parse-fail for a *single* CURIE against an otherwise-healthy endpoint — a transient condition a retry could clear. So: direct_kg → remove the SDK tier, yield no findings (honest). triage → remove the SDK tier, **and add a single in-place retry of `count_edges_via_api` on per-call `None` before falling to the default** (cheap, directly addresses the flaky mode). A transient-failure *prefetch* is **out of scope** for this PR — a post-deploy follow-on if logs later justify it (see Open Questions). This PR ships drop (direct_kg) / drop-plus-retry (triage).
  3. **Prefetch + select** where the fallback genuinely adds reach — `entity_resolution` only.
- **`entity_resolution` is the one true prefetch — and it must preserve Tier-2's variant-search behavior.** Tier-2 fires when Tier-1 `hybrid_search` scored the *raw name* too low; its value is iterative reformulation (synonyms, de-hyphenation, prefix/gene-symbol heuristics across `max_turns=5`). A single raw-name prefetch would drop exactly those variant-spelled entities to cold_start — a recall regression, not just an "honest fail." So the prefetch must issue **multiple HTTP queries** (raw + de-hyphenated + prefix-stripped + gene-symbol variants, via `text_search` *and* `hybrid_search`, `limit>1`) to build a candidate set comparable to the live loop; the SDK then selects from candidates with `allowed_tools=[]`. This **requires rewriting `RESOLUTION_PROMPT`/`RETRY_PROMPT`** from "search the KG" to "select the best CURIE from these candidates." Empty candidate set → `method="failed"` (no fabrication).
- **The static variant list is an *approximation* of the live loop, not an equivalent — so gate it against ground truth.** The live Tier-2 is open-ended: across `max_turns=5` the LLM generates *new* reformulations conditioned on prior results, and it could (in principle) traverse via `get_nodes`/`get_node_info`/`get_neighbors` — none of which a fixed 4-variant search prefetch reproduces. The fixed list is therefore a strict subset and *can* lose recall. **The gate must use an absolute, hand-labeled target — not "% of the pre-change baseline":** by this plan's own premise the broken Tier-2 SDK loop produces ungrounded output, so a relative gate compares against a non-functional or HTTP-only baseline and passes tautologically. Instead: a **repo-committed fixture of ≥20 hard-variant entities, each with a hand-labeled expected CURIE** (mined from #60 `FALLBACK_EVENT` entities that reached Tier-2), and the prefetch+select path must resolve **≥95% of those to the correct labeled CURIE**. On a miss, **reclassify `entity_resolution` to honest-fail** (drop Tier-2 → cold_start; the variant-string SDK loop is a deferred follow-on, not built here — user decision) — do not silently ship a regression.
- **The empty-candidate guard does NOT close the hallucination path on its own (R1a).** With a non-empty candidate set, `parse_resolution_result` returns whatever CURIE the model emits — including a plausible training-data CURIE never in the candidates. The load-bearing protection is the **membership check (R1a)**: reject any returned CURIE not present in the prefetched candidate set → `method="failed"`. The prompt rewrite alone is *not* sufficient.
- **The correctness guarantee is "no real data → honest failure," not the MCP classifier.** With `allowed_tools=[]` the `classify_mcp_degradation` guard is inert by design; the active protection is per-node honest-default behavior on empty/failed data **plus the data-boundary CURIE validation: entity_resolution membership check (R1a) and integration gap-CURIE nulling (R1b)**.
- **`integration` gap analysis drops tools AND nulls model-authored gap CURIEs (R1b).** Two distinct changes: (1) drop the broken tool config — behaviorally inert, since the stdio MCP never functioned and gap analysis already reasons only over the in-prompt study context (gaps are tier-3 speculative, "expected but absent from these findings"); (2) **the tool drop alone does NOT close the fabrication path** — the model can still emit a CURIE in its gap output that flows to synthesis, so R1b nulls/membership-checks `GapEntity.curie` at construction. Bridges (HTTP) are untouched.
- **`temporal` simply drops its validation-only tool.** Classification already reasons over `build_study_context_for_temporal`; `allowed_tools=[]` removes the doomed spawn with no behavioral loss.
- **Reuse, don't abstract.** Each node's prefetch is small and uses node-specific tools; mirror `cold_start.get_entity_connections` per node rather than forcing a shared helper.

## Open Questions

### Resolved During Planning

- Severity / blocking? Verified low-medium, fallback-only, non-blocking (issue #61).
- One PR vs many? Single PR (user decision).
- Per-node approach? Three outcomes (review-refined): **drop-tools** (temporal, integration-gap), **drop-fallback to honest default** (direct_kg = drop only; triage = drop **+ single in-place retry** for its transient per-call `None` mode), **prefetch+select** (entity_resolution only — the one with genuine variant-search value-add).
- `entity_resolution` prompt? **Rewrite** `RESOLUTION_PROMPT`/`RETRY_PROMPT` from "search" to "select from candidates" (required, not optional) — with multi-variant prefetch to avoid a recall regression.
- `entity_resolution` hallucination closure? **Membership check (R1a)** is mandatory: a returned CURIE not in the prefetched candidate set → `method="failed"`. The empty-candidate guard alone is insufficient.
- `entity_resolution` recall safety? **Pre-committed go/no-go gate against a repo-committed, hand-labeled fixture** (≥20 hard-variant entities, ≥95% resolved to the correct labeled CURIE — absolute, *not* relative to the non-functional Tier-2 baseline). Build/freeze the fixture before coding.
- **Gate-miss behavior? (user decision)** **Reclassify `entity_resolution` to honest-fail** — drop Tier-2 entirely (consistent with `direct_kg`); gate-failing variants route to cold_start. Honest failure is always in scope and satisfies the correctness goal; a recall hit is accepted rather than introducing a third resolution strategy in this fix-PR. The variant-string SDK loop is **not** built here — if production later shows the recall hit matters, it is a separate follow-on issue.

### Deferred to Implementation

- **`text_search`'s response envelope (request param already resolved).** The request param is `search_text` (confirmed in `kestrel_tools.py`, matching `hybrid_search`) — reuse it. **Still verify the response envelope before reusing `resolve_via_api`'s parse path:** if `text_search` returns a different shape than the `{search_text: [results]}` envelope `resolve_via_api` parses (`entity_resolution.py:143-150`), the parse silently yields zero candidates for *all* `text_search` results — partially defeating the variant search with no error signal. Treat as a **Unit 3 precondition**, not an optional deferral.
- **Optional `direct_kg`/`triage` prefetch — out of scope for this PR.** This PR commits to **drop** (`direct_kg`) and **drop + single retry** (`triage`); both fully satisfy R1–R5 without a prefetch. A transient-failure prefetch is a **post-deploy follow-on**, not an in-PR log-gated decision: after merge, check the #60 `FALLBACK_EVENT`/`mcp_tool_calls=0` lines (`sudo journalctl -u kraken-backend-dev | grep FALLBACK_EVENT`); only if they show a meaningful per-call/transient rate that the triage retry doesn't already clear, open a separate issue to add a thin `one_hop_query` prefetch. Do not build it speculatively here.
- **`triage_failed` marker (optional).** Whether to add a distinct marker so Synthesis can flag reduced confidence for entities rerouted to cold-start by a failed count, vs. leaving the pre-existing `cold_start` default (Unit 5).

## High-Level Technical Design

> *This illustrates the intended per-node approach and is directional guidance for review, not implementation specification.*

| Node | Broken SDK tier does | Migration pattern | Empty/failure behavior |
|---|---|---|---|
| `temporal` | classify longitudinal; tool "just for validation" | **drop tools** (`allowed_tools=[]`, no `mcp_servers`); reason over study context | unchanged (classification still runs) |
| `integration` (gap) | speculative gap analysis | **drop tools** + **null model-authored gap CURIEs at construction (R1b)**; reason over provided findings context | gap CURIEs nulled (R1b); empty gaps possible; bridges unaffected, already HTTP |
| `entity_resolution` | Tier-2 name→CURIE *variant search* | **prefetch + select**: multi-variant HTTP `text_search`+`hybrid_search` (`limit>1`) → SDK selects CURIE (`allowed_tools=[]`); **rewrite prompt** to "select from candidates"; **validate returned CURIE ∈ candidates (R1a)** | no candidates → `method="failed"`; transport failure → `method="failed"`; CURIE not in candidates → `method="failed"` |
| `direct_kg` | Tier-2 association lookup (fires on *total* Tier-1 failure) | **drop fallback** → honest no-findings (prefetch only if logs show per-call failures) | no findings |
| `triage` | Tier-2 edge counting (fires on *per-call/transient* Tier-1 failure) | **drop fallback + single in-place retry** on per-call `None` → existing default (prefetch only if logs justify) | default classification = `cold_start` (reroutes entity — see Unit 5) |
| `sdk_utils` | — | **retire** `get_kestrel_mcp_config` + `KESTREL_*` | n/a |

## Implementation Units

- [ ] **Unit 1: `temporal` — drop the validation-only MCP tool**

**Goal:** Stop configuring the broken stdio server in temporal; reason over the already-provided study context.

**Requirements:** R1, R2, R4

**Dependencies:** None

**Files:**
- Modify: `backend/src/kestrel_backend/graph/nodes/temporal.py` (SDK options; **edit the `from ..sdk_utils import ...` line to remove only the dead symbols** — `get_kestrel_mcp_config`, `McpStdioServerConfig`, `KESTREL_COMMAND`, `KESTREL_ARGS` — while **keeping** the still-used `HAS_SDK`, `query_with_usage`, `ClaudeAgentOptions` (note: `chunk` is **not** imported/used in this file); Unit 6 deletes the symbols from `sdk_utils`)
- Test: `backend/tests/test_langgraph_prototype.py` (`TestTemporalNode`)

**Approach:**
- Set `allowed_tools=[]`, remove `mcp_servers` and the `get_kestrel_mcp_config()` call. The classification prompt already includes `build_study_context_for_temporal` output — no prefetch needed. With `allowed_tools=[]` the MCP classifier is inert (no false degradation).
- **Prune the now-dead `..sdk_utils` import names** (see Files) so Unit 6's symbol deletion can't leave a dangling import that breaks graph build.
- If `TEMPORAL_PROMPT` still instructs tool use (e.g. "use one_hop_query…"), prune that reference so the model doesn't waste a turn or emit "tools not available" phrasing under `allowed_tools=[]`.
- **`system_prompt` structure:** the current temporal call embeds the full prompt in the user turn and passes no `system_prompt` (unlike the `cold_start` pattern, which splits a `system_prompt`). Keeping the existing user-turn structure is fine for parity — but make it a *deliberate* choice: do not silently diverge from the cited `cold_start` pattern. If aligning with `cold_start`, split `TEMPORAL_PROMPT` into `system_prompt` + user turn explicitly; otherwise note that the user-turn structure is retained intentionally.

**Execution note:** Characterization-first — capture current temporal classification output on a longitudinal fixture before the change to assert parity.

**Patterns to follow:** `cold_start` SDK call (`allowed_tools=[]`, reason over provided data).

**Test scenarios:**
- Happy path: longitudinal study fixture → temporal classifications produced (parity with pre-change shape).
- Edge case: non-longitudinal → still skips (routing unchanged).
- Error path: `HAS_SDK is False` → existing graceful return unchanged.

**Verification:** temporal produces classifications with `allowed_tools=[]` and no `mcp_servers`; no reference to `get_kestrel_mcp_config` remains in the file.

- [ ] **Unit 2: `integration` gap analysis — drop tools, reason over context**

**Goal:** Migrate the gap-analysis SDK call off MCP, **and close the gap-CURIE fabrication path (R1b)**; leave the HTTP bridges untouched.

**Requirements:** R1, R1b, R2, R4

**Dependencies:** None

**Files:**
- Modify: `backend/src/kestrel_backend/graph/nodes/integration.py` (gap-analysis SDK options; **null/membership-check `GapEntity.curie` after parse (R1b)**; **edit the `..sdk_utils` import line to remove only the dead symbols**, keeping the still-used `HAS_SDK`/`query_with_usage`/`ClaudeAgentOptions` (note: `chunk` is **not** imported/used in this file))
- Test: `backend/tests/test_langgraph_prototype.py` (`TestIntegrationNode`)

**Approach:**
- Set `allowed_tools=[]`, remove `mcp_servers`/`get_kestrel_mcp_config`. **The tool-drop safety argument is structural, not behavioral:** the stdio MCP was always broken, so the gap-analysis tools never registered — gap analysis has only ever reasoned over the in-prompt study context (findings summaries). Dropping the (already-dead) tool config is therefore **behaviorally inert**. (We do not rely on the runtime-unverifiable claim "the model chose not to call the tools.") `bridges = api_bridges` (HTTP) path stays exactly as-is.
- **Close the gap-CURIE hole (R1b — the tool drop alone does NOT close it):** `parse_integration_result` currently keeps `curie=g.get("curie")` (`integration.py:518`) → unvalidated `GapEntity.curie` (`state.py:187`) → rendered by `synthesis.py:496`. **`GapEntity` is `frozen=True` (`state.py:183`)**, so do this **at construction** — pass `curie=None` (or the membership-gated value) into the `GapEntity(...)` call at `integration.py:515-525`; do **not** write `gap.curie = None` post-parse (raises `ValidationError`). Gaps are "expected but *absent*," so the CURIE is not needed to convey the gap. Consumer audit (already done): `synthesis.py:496/506` is the sole renderer; `literature_grounding` reads `resolution.curie` not gap CURIEs; `node_detail_extractors` omits the gap curie — nulling at construction covers all. (Membership variant: source the allowlist from the in-prompt finding CURIEs actually placed in the prompt, not raw model output.)
- Prune the dead `..sdk_utils` import names (see Files) and any tool-usage instruction left in the gap prompt / `INTEGRATION_PROMPT`.

**Execution note:** Characterization-first — assert bridges (HTTP) are unchanged and gaps still parse; capture a model-emitted non-null gap CURIE pre-change and assert it is nulled post-change.

**Patterns to follow:** the existing `bridges = api_bridges` / `parse_integration_result` flow (`integration.py:698-702`); R1a's membership discipline for the optional keep-if-grounded variant.

**Test scenarios:**
- Happy path: study context with an obvious absent marker → a gap is produced; bridges from `api_bridges` unchanged.
- Edge case: no informative gaps → `gaps == []`; bridges still emitted.
- **R1b path: model emits a non-null gap `curie` → it is nulled (or rejected) and does NOT appear in the synthesis report.**
- Integration: output dict still carries `bridges`, `gap_entities`, `direct_findings` with the same `source` tags.

**Verification:** gaps are produced with `allowed_tools=[]`; bridges and their source/HTTP path are byte-unchanged; **a model-emitted gap CURIE never reaches the synthesis report (R1b)**; no `get_kestrel_mcp_config` reference remains.

- [ ] **Unit 3: `entity_resolution` Tier-2 — HTTP search-candidate prefetch + SDK select**

**Goal:** Replace the MCP-tool search fallback with an HTTP `text_search`+`hybrid_search` prefetch; the SDK selects the best CURIE; no candidates → honest `failed`.

**Requirements:** R1, R1a, R2, R3, R4

**Dependencies:** None (uses existing `call_kestrel_tool`)

**Files:**
- Modify: `backend/src/kestrel_backend/graph/nodes/entity_resolution.py` (`resolve_single_entity`; add a multi-variant prefetch helper; **rewrite `RESOLUTION_PROMPT`/`RETRY_PROMPT`** to "select from candidates"; **add post-parse candidate-membership validation (R1a)**; remove only the dead `..sdk_utils` import names)
- Test: `backend/tests/test_langgraph_prototype.py` (`TestEntityResolution*`) or a new `backend/tests/test_entity_resolution.py`

**Approach:**
- **Scope (mirrors Units 4/5):** this PR builds a **fixed-variant HTTP prefetch + membership-gated selection**. The open-ended `max_turns=5` reformulation loop is **not** reproduced or re-implemented here; if the recall gate is missed, **reclassify to honest-fail (drop Tier-2 → cold_start)** — the variant-string SDK loop is a deferred follow-on. Do not attempt the iterative loop as an in-PR secondary fallback.
- **Preserve Tier-2's variant-search behavior, server-side.** Tier-2 fires when Tier-1 `hybrid_search` scored the *raw name* too low — **but note `resolve_via_api` returns `None` on `is_error`/empty/parse-fail too, not only on score `< 0.6`.** So a low score (prefetch helps) and a transport/endpoint failure (prefetch re-fails — same trap as direct_kg/triage) both route here. The prefetch is the right fix for the *score* case; on a *transport* failure it must fail honestly (`method="failed"`), not imply recovery. Issue **multiple prefetch queries** (raw + de-hyphenated + prefix-stripped + gene-symbol forms) across `text_search` *and* `hybrid_search`, with `limit > 1` (e.g. 5-10) and iterate the full results list — do **not** copy `resolve_via_api`'s `limit:1`/`results[0]`. Dedup into a candidate set of `{curie, name, category, score}`.
- **Rewrite the SDK call to select, not search:** new prompt = "select the single best CURIE for `<entity>` from these candidates, or null if none fit"; `allowed_tools=[]`; parse via `parse_resolution_result`. **No candidates → `method="failed"`** without invoking the SDK.
- **Validate the selection (R1a — this, not the prompt, closes the hallucination path):** after parsing, **check the returned CURIE is a member of the prefetched candidate set** via exact match on a single canonical normalization applied to both sides (uppercase prefix + trim); **no fuzzy matching**. On a hit, surface the **candidate's** `curie`/`name`/`category`, not the model's emitted fields. A model can emit a plausible training-data CURIE that was never in the candidates; reject it → `method="failed"`. The prompt rewrite reduces but does not eliminate this. (Beware: `resolve_via_api` reads `top.get("id") or top.get("curie")` while `parse_resolution_result` reads `data.get("curie")` — two independently-formatted strings, which is exactly why the normalization must be applied to both.)
- **The static variant list is an approximation, not an equivalent of `max_turns=5`.** It cannot reproduce the live loop's turn-by-turn reformulation or its `get_nodes`/`get_neighbors` traversal, so recall *can* drop. **Gate on the pre-committed ground-truth fixture (see Execution note); on a miss, reclassify `entity_resolution` to honest-fail (drop Tier-2, route to cold_start) — the variant-string SDK loop is a follow-on, not built here** (user decision). Do not silently ship the static-list prefetch if it underperforms.
- `text_search`'s request param is `search_text` (confirmed, matches `hybrid_search`); **verify its response envelope** matches the `{search_text: [results]}` shape `resolve_via_api` parses before reusing that parse path, and add a branch if it differs — see Deferred.
- **Parse robustness (harden proactively — do NOT defer to the gate):** `parse_resolution_result`'s `\{[^{}]+\}` regex cannot match *nested* JSON. Keep the new "select from candidates" prompt's output a **flat** object (`{"curie": ..., "confidence": ...}`) **and** ensure the parser either handles any plausible nested shape or rejects it → `method="failed"` (honest). The recall gate validates *recall on the fixture entities*, not *parser correctness on corner cases*, so a silent parse failure on an out-of-fixture shape would slip through — fix it at the parser, not by hoping the fixture exercises it.

**Execution note:** Characterization-first against a **pre-committed, repo-committed fixture** of ≥20 hard-variant entities (hyphenation/gene-symbol/prefix cases that motivated `max_turns=5`, mined from #60 `FALLBACK_EVENT` entities), **each carrying a hand-labeled expected CURIE**. **Go/no-go gate: the prefetch+select path must resolve ≥95% of the fixture to its labeled CURIE** (absolute target — not relative to the non-functional Tier-2 baseline). Build and freeze the fixture *before* implementing so the gate is reviewable and not self-graded. On a miss, **reclassify to honest-fail (drop Tier-2 → cold_start)**, not a shipped regression; do not assume recall can only improve.

**Patterns to follow:** `resolve_via_api` (`:96-200`) for the HTTP call/parse; `pathway_enrichment.prefetch_one_hop_neighbors` for the gather/errored handling.

**Test scenarios:**
- Happy path: mocked multi-query prefetch returns candidates → SDK selects the best → resolved CURIE.
- Edge case (the point of the node): a variant form (de-hyphenated / gene-symbol) surfaces a candidate the raw name missed → resolves; assert recall ≥ a curated set of hard entities.
- Error path: no candidate across all variants → `method="failed"`, SDK not called (no fabricated CURIE).
- Error path: **SDK returns a CURIE not in the candidate set → `method="failed"` (R1a membership check), no fabricated CURIE surfaced.**
- Error path: prefetch hits a transport failure (all variant queries error) → `method="failed"`, not a false resolution.
- Error path: `HAS_SDK is False` → existing failed-resolution shape unchanged.

**Verification:** Tier-2 resolves from HTTP candidates with `allowed_tools=[]`; the committed hard-variant fixture resolves at **≥95% to its hand-labeled CURIEs** (else reclassify to honest-fail: drop Tier-2 → cold_start); empty candidates **and out-of-set selections** yield `failed`, never a hallucinated CURIE.

- [ ] **Unit 4: `direct_kg` Tier-2 — drop the broken SDK fallback (default) → honest no-findings**

**Goal:** Remove the `llm_fallback` SDK tier that fires only on total Tier-1 failure and can only fabricate; entities whose Tier-1 failed contribute no `direct_kg` findings (honest), not hallucinated associations.

**Requirements:** R1, R2, R4

**Dependencies:** None

**Files:**
- Modify: `backend/src/kestrel_backend/graph/nodes/direct_kg.py` (`analyze_single_entity` + its call site; remove **only** the dead `..sdk_utils` import names — and if dropping the Tier-2 batching loop removes the file's last `chunk(...)` call, drop `chunk` too; otherwise keep it)
- Test: `backend/tests/test_langgraph_prototype.py` (`TestDirectKG*`) or new `backend/tests/test_direct_kg.py`

**Approach:**
- **Default: drop the SDK Tier-2.** `analyze_via_api` returns `None` only from its top-level `except` (a total failure; per-call errors are swallowed with `continue`), so a fresh `one_hop_query` prefetch would call the *same* failing endpoint with the *same* CURIE and re-fail. Remove the `mcp_servers`/`allowed_tools` SDK call; an entity whose Tier-1 failed simply yields no `direct_kg` findings.
- **Prefetch is out of scope (post-deploy follow-on).** `direct_kg`'s `None` is total-failure-only, so a prefetch would re-fail the same endpoint; drop is the complete fix. If post-deploy logs ever show per-call/transient failures, a thin HTTP `one_hop_query` prefetch + `allowed_tools=[]` extraction (parse via `parse_direct_kg_result`, `source="direct_kg:llm_fallback"`, empty → no findings) is a *separate* issue — not built here (see Open Questions).

**Execution note:** Characterization-first — capture current fallback output, then assert the dropped path yields no fabricated associations.

**Patterns to follow:** the existing Tier-1 `analyze_via_api` result handling (Tier-2 reached only when it returns `None`).

**Test scenarios:**
- Happy path: Tier-1 succeeds → Tier-2 never invoked (unchanged); associations come from HTTP.
- Error path: Tier-1 returns `None` for an entity → no `direct_kg` findings for it (no fabricated associations), pipeline continues.
- Error path: `HAS_SDK is False` → unchanged.

**Verification:** no `mcp_servers`/`mcp__*` config remains in `direct_kg.py`; a failed-Tier-1 entity yields zero fabricated associations.

- [ ] **Unit 5: `triage` Tier-2 — drop the broken SDK fallback (default); name the reroute honestly**

**Goal:** Remove the edge-counting SDK fallback that fires only on total Tier-1 failure; a failed count falls to triage's existing default classification — and the plan states the routing consequence plainly.

**Requirements:** R1, R2, R4

**Dependencies:** None

**Files:**
- Modify: `backend/src/kestrel_backend/graph/nodes/triage.py` (`count_edges_single` + call site; remove **only** the dead `..sdk_utils` import names — and if dropping the Tier-2 batching loop removes the file's last `chunk(...)` call, drop `chunk` too; otherwise keep it)
- Test: `backend/tests/test_langgraph_prototype.py` (`TestTriageNode`) or new `backend/tests/test_triage.py`

**Approach:**
- **Default: drop the SDK Tier-2.** `count_edges_via_api` already funnels failures toward `edge_count=0`, and the SDK fallback (with broken tools) can't do better. Remove the SDK call.
- **Add a single in-place retry of `count_edges_via_api` on per-call `None` before defaulting** (this node, unlike `direct_kg`). `count_edges_via_api` returns `None` on per-call `isError`/empty/parse-fail for a *single* CURIE against an otherwise-healthy endpoint. **Scope the retry to the genuinely time-varying causes (exception/timeout/`isError`)** — those a retry can clear. A *stable* empty/malformed envelope for a specific CURIE is **deterministic**: an identical-args retry re-fails, so it correctly falls to the documented `cold_start` reroute (that is expected, not a failure of the retry). A small backoff/jitter is optional; do not claim the retry "handles transient failures" for the deterministic class. A genuinely well-characterized entity should not be downgraded because one preview query *flaked* — that is the case this targets. (Do **not** add this to `direct_kg`, whose `None` is total-failure-only.)
- **Name the consequence honestly (this is NOT a neutral "unknown"):** `classify_by_edge_count` has no `unknown` bucket — a count that fails *even after the retry* → `edge_count=0` → `cold_start`, which `route_after_triage` sends to the cold-start analogue branch instead of `direct_kg`. So a *genuinely well-characterized* entity whose triage still failed is downgraded to cold-start analysis. This is **pre-existing** behavior (the current default is already `cold_start`), so the migration doesn't worsen it — but decide whether to leave it or add a distinct `triage_failed` marker so Synthesis can flag reduced confidence. Record the decision.
- **Prefetch is out of scope (post-deploy follow-on).** Beyond the retry, a thin `one_hop_query` prefetch could occasionally succeed, but it is **not built here** — a separate issue only if post-deploy logs show transient failures the single retry didn't clear (see Open Questions).

**Execution note:** Characterization-first on the classification output + routing.

**Patterns to follow:** `count_edges_via_api` (`:64`) + `classify_by_edge_count`.

**Test scenarios:**
- Happy path: Tier-1 counts an edge → correct `classify_by_edge_count` bucket (unchanged).
- Error path: Tier-1 returns `None` once then succeeds on retry → correct bucket (assert the retry recovers the transient case, no reroute).
- Error path: Tier-1 returns `None` on both attempts → entity defaults to `cold_start` (assert the reroute, no SDK-guessed count).
- Error path: `HAS_SDK is False` → unchanged.

**Verification:** no `mcp_servers`/`mcp__*` config remains in `triage.py`; a transient per-call `None` is recovered by one retry; counts failing after retry default to `cold_start` (documented reroute), never an SDK-guessed number.

- [ ] **Unit 6: Retire the dead stdio config in `sdk_utils.py`**

**Goal:** Remove `get_kestrel_mcp_config` + `KESTREL_COMMAND`/`KESTREL_ARGS` once no graph node references them.

**Requirements:** R5

**Dependencies:** Units 1-5 (all graph callers removed)

**Files:**
- Modify: `backend/src/kestrel_backend/graph/sdk_utils.py` (remove `get_kestrel_mcp_config`, `KESTREL_COMMAND`, `KESTREL_ARGS`, and the `McpStdioServerConfig` re-export — review confirmed `agent.py` imports it directly from `claude_agent_sdk.types`, not from `sdk_utils`, so nothing imports it from here once the five nodes are pruned)
- Test: `backend/tests/test_sdk_utils.py` (delete `TestKestrelConfig` **and** remove `KESTREL_COMMAND`/`KESTREL_ARGS`/`get_kestrel_mcp_config` from the module-level import — otherwise the test module fails at collection)

**Approach:**
- After Units 1-5 prune their import lines, remove the symbols. Classic `agent.py` keeps its own `_get_kestrel_mcp_config` — leave it.

**Test expectation:** none beyond removing `TestKestrelConfig` + its imports; import-cleanliness covered by the suite.

**Verification:** `grep -rn "get_kestrel_mcp_config\|KESTREL_COMMAND\|KESTREL_ARGS\|McpStdioServerConfig" backend/src/kestrel_backend/graph/` returns nothing; **also confirm nothing outside `graph/` imports the removed re-export** with `grep -rn "from .*sdk_utils import.*McpStdioServerConfig" backend/src/` (expected: none — `agent.py` imports `McpStdioServerConfig` directly from `claude_agent_sdk.types`, not from `sdk_utils`); **`uv run python -c "import kestrel_backend.graph.builder"` succeeds** (proves no node left a dangling import — `builder.py` eagerly imports all node modules); suite collects and imports clean.

## System-Wide Impact

- **Interaction graph:** all five nodes feed Integration/Synthesis; outputs (CURIEs, associations, edge-count classifications, gaps, temporal classifications) keep their schemas and `source` tags, so downstream consumers are unaffected.
- **Error propagation:** the change *strengthens* honesty — failure modes that previously hallucinated now return `failed`/`unknown`/empty, via two mechanisms: **tool-drop** (temporal, integration bridges) and **data-boundary CURIE validation** (entity_resolution membership check R1a; integration gap-CURIE nulling R1b). No new exceptions introduced.
- **State lifecycle:** no state-schema or reducer changes; no new fields.
- **API surface parity:** this completes the parity started by #60 — after it, all six SDK nodes are off the broken stdio path and `cold_start`-style.
- **Unchanged invariants:** every node's HTTP Tier-1 logic, output schema, and finding `source` strings; classic-mode `agent.py`; LangGraph topology.

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| **Stale `..sdk_utils` import after Unit 6 deletes symbols → ImportError breaks the whole graph** (`builder.py` eagerly imports all node modules) | Each of Units 1-5 edits its `..sdk_utils` import to remove **only** the dead symbols (keeping each file's still-used subset of `HAS_SDK`/`query_with_usage`/`ClaudeAgentOptions`/`chunk` — `chunk` only in the batching nodes, not temporal/integration); Unit 6 greps **all** dead symbols (not just `get_kestrel_mcp_config`) and runs `import kestrel_backend.graph.builder` as a smoke gate. |
| **`entity_resolution` recall regression**: a fixed-variant prefetch is a strict *subset* of the live `max_turns=5` loop (no turn-by-turn reformulation, no `get_neighbors` traversal) and can drop variant-spelled entities to cold_start | Multi-variant prefetch (de-hyphenated / prefix / gene-symbol forms) to rebuild a comparable candidate set; **pre-committed go/no-go gate against a repo-committed, hand-labeled fixture (≥95% to correct CURIE — absolute, not relative to the non-functional Tier-2 baseline)**; on a miss, **reclassify to honest-fail (drop Tier-2 → cold_start)** — do not assume "can only improve." |
| **`entity_resolution` hallucinated CURIE survives the empty-candidate guard**: with non-empty candidates the model can emit a training-data CURIE never in the set | **R1a membership check**: reject any returned CURIE not present in the prefetched candidate set → `method="failed"`. The prompt rewrite alone does not close this. |
| `direct_kg`/`triage` prefetch would re-fail (same endpoint Tier-1 just failed on) | `direct_kg`: `None` is total-failure-only → **drop**, no retry. `triage`: `None` is transient per-call → **drop + single in-place retry** before defaulting. Build a *prefetch* only if logs show transient failures the retry didn't clear (committed default if logs unavailable: no prefetch). |
| `triage` failed count silently reroutes a well-characterized entity to cold-start | Single retry recovers the transient case; residual reroute is pre-existing behavior (current default is already `cold_start`); plan names it explicitly and offers an optional `triage_failed` marker decision (Unit 5). |
| **`entity_resolution` Tier-1 `None` conflates score-too-low with transport failure** — prefetch helps the former, re-fails the latter | Prefetch is correct for the score case; on transport failure (all variant queries error) it returns `method="failed"`, never a false resolution. |
| Larger single-PR diff across five node files | Units 1+2 (drop-tools) are independently safe; the `direct_kg`/`triage` simplification (drop, not prefetch) shrinks the contested surface; characterization-first per node; the builder import smoke test gates the whole PR. |

## Documentation / Operational Notes

- Run tests with `cd backend && uv run python -m pytest tests/ -v -m "not integration"` (venv-path-spaces learning).
- After merge, close #61 and note in the #60 compound doc that all six SDK nodes are migrated.
- Production logs: the #60 `mcp_tool_calls=0` diagnostics for these nodes should disappear once migrated — a quick post-deploy confirmation.

## Sources & References

- Issue: #61 (this work); PR #60 (pathway_enrichment precedent), PR #24 (cold_start precedent).
- Plan: `docs/plans/2026-06-01-001-fix-pathway-enrichment-mcp-degradation-plan.md`.
- Learning: `docs/solutions/best-practices/sdk-stdio-mcp-unavailable-migrate-to-http-data-in-prompt-2026-06-02.md`.
- Code: `graph/nodes/{entity_resolution,direct_kg,triage,integration,temporal}.py`, `graph/sdk_utils.py`, `graph/nodes/cold_start.py:148-225`.
