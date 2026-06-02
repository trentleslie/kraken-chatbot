---
title: "fix: Guard + migrate pathway_enrichment Phase B off broken stdio MCP (#44)"
type: fix
status: active
date: 2026-06-01
deepened: 2026-06-01
---

# fix: Guard + migrate pathway_enrichment Phase B off broken stdio MCP (#44)

## Overview

Issue #44 reports that the `pathway_enrichment` Phase B SDK agent intermittently announces *"The Kestrel MCP tools … are not available in my current tool set"* and then hallucinates one-hop analysis from training-data knowledge. A multi-agent investigation (see Sources) found the cause is **not** a race: the SDK stdio MCP server is configured to launch `uvx mcp-client-kestrel` (`graph/sdk_utils.py:35-36`), but **that package does not exist on PyPI** — already discovered and documented in `graph/nodes/cold_start.py:13-18`, where PR #24 migrated cold_start to direct HTTP for exactly this reason. The tools never register; the agent's report is literally true; the "intermittency" is just `uvx` cold-resolution failing different ways per spawn.

This plan is organized as two work stages, **delivered together in a single PR** (decision below). Landing the guard and the migration at once means real HTTP data flows from the moment it merges, so there is no interim window where Phase B renders an empty enrichment section:
1. **Guard + instrumentation** (low-risk, additive): detect the degraded condition structurally, drop the hallucinated findings, flag the run as degraded, and add diagnostics at the shared SDK layer so the same latent failure becomes observable across all SDK nodes.
2. **HTTP migration** (durable fix): convert Phase B to fetch one-hop data via the HTTP Kestrel client and reason over it in-prompt, mirroring the proven `cold_start` / PR #24 pattern — eliminating the stdio MCP dependency entirely.

## Problem Frame

`pathway_enrichment` runs two sub-analyses: **Phase A** two-hop shared-neighbor analysis over the **HTTP** client (`find_two_hop_shared_neighbors` → `multi_hop_query`, `pathway_enrichment.py:170-256,342-348`) which always works, and **Phase B** one-hop analysis via the **SDK + stdio MCP** (`pathway_enrichment.py:370-440`) which is the only stdio-MCP call and is broken. "Kestrel is operational" in the issue refers to the healthy HTTP endpoint and says nothing about the stdio subprocess. Under `permission_mode="bypassPermissions"` and `max_turns=25`, the tool-less agent fills the gap by fabricating shared neighbors, which then flow into Integration and Synthesis as if KG-grounded — a correctness/credibility risk in a research-grade pipeline.

## Requirements Trace

- R1. Hallucinated `pathway_enrichment` (Phase B) output must not propagate to Integration/Synthesis. (#44 core risk)
- R1b. The same broken stdio path exists in five other SDK nodes; that pipeline-wide hallucination exposure is explicitly out of scope here, acknowledged, and tracked under a separate issue. Stage 1 instrumentation quantifies it. (R1 is scoped to one node — it is not a pipeline-wide guarantee.)
- R2. When Phase B is degraded, the run must be flagged so downstream nodes and the UI can disclose the gap.
- R3. The degraded condition must be detectable from a structural signal, not a brittle string match alone.
- R4. Phase B must obtain real one-hop KG data instead of relying on the nonexistent stdio MCP server (durable fix).
- R5. Phase A two-hop findings (real, HTTP-derived) must be preserved even when Phase B is dropped.
- R6. No behavioral change to the other five SDK nodes in this work; surface their shared latent exposure for separate follow-up.

## Scope Boundaries

- Not changing Phase A (`find_two_hop_shared_neighbors`, HTTP) — it works.
- Not changing the HTTP `kestrel_client`, `graph/builder.py` topology, or working nodes' semaphores.
- Not touching classic-mode `agent.py:86` stdio config (separate surface, classic mode).
- Not adding a feature to detect *every* form of SDK misbehavior — scope is MCP-tool-unavailability degradation.

### Deferred to Separate Tasks

- **Migrate the other five SDK nodes off the broken stdio MCP** (`entity_resolution`, `direct_kg`, `integration`, `temporal`, `triage` — all call `get_kestrel_mcp_config()` + `mcp_servers={"kestrel": …}`): file a new issue. They have HTTP primary tiers that mask the SDK-tier degradation to varying degrees; each needs an individual audit. Stage 1 instrumentation here will quantify their exposure first. **Verify the "HTTP primary tier" claim per node before relying on it** — two cases are higher-severity than Phase B: (a) if `entity_resolution`'s SDK tier has no HTTP fallback, degraded CURIE resolution would poison *every* downstream node (including the fixed `pathway_enrichment`); (b) `integration` is both a *consumer* of pathway_enrichment output and itself a broken stdio node, so even after this plan its own SDK call can still hallucinate into Synthesis.
- **Add `pathway_enrichment` to the `get_semaphore` map** (`pipeline_config.py:207-215` omits it): becomes moot once Stage 2 acquires `SDK_SEMAPHORE` directly (mirroring cold_start), so handled inside Unit 5 rather than as standalone work.

## Context & Research

### Relevant Code and Patterns

- **Canonical HTTP precedent:** `graph/nodes/cold_start.py:1-18,460,490` — `allowed_tools=[]`, prefetch KG data via HTTP `call_kestrel_tool(...)`, `max_turns=1`, `async with SDK_SEMAPHORE`, reason over real data in-prompt. Stage 2 mirrors this exactly. Do not invent a new approach.
- **Broken config:** `graph/sdk_utils.py:35-50` (`KESTREL_COMMAND="uvx"`, `KESTREL_ARGS=["mcp-client-kestrel"]`, `get_kestrel_mcp_config()` builds `McpStdioServerConfig` with no `env`).
- **SDK stream loop:** `graph/sdk_utils.py:120-202` `query_with_usage` — iterates `async for event in query(...)`, collects text + usage; the natural place to also count `mcp__*` tool-use blocks and capture the init tool list.
- **Phase B target:** `graph/nodes/pathway_enrichment.py:370-440` (options, `wait_for` timeout, parse, findings build); output dict keys `shared_neighbors` / `biological_themes` / `errors`.
- **Usage record:** `graph/state.py:287` `ModelUsageRecord` — extend with optional diagnostics fields (additive, non-breaking).
- **Two-tier patterns** in `direct_kg.py` show how nodes blend HTTP Tier-1 with SDK Tier-2 — context for the deferred multi-node migration.

### Institutional Learnings

- `docs/solutions/developer-experience/pytest-venv-path-spaces-module-invocation-2026-06-01.md` — run backend tests with `uv run python -m pytest` (the venv-path-spaces shebang trap); applies to executing this plan's tests.
- `cold_start.py` docstring is itself the institutional record of the `mcp-client-kestrel`-doesn't-exist discovery + the HTTP remedy.

### External References

- None required — strong local precedent (`cold_start` / PR #24). The investigation workflow served as the deep research.

## Key Technical Decisions

- **Drop + flag degraded (no retry).** The earlier "retry once" idea is abandoned: retry cannot recover a nonexistent package and would only double latency under the 480s timeout. Dropping the fabricated findings while preserving Phase A is the correctness-safe default. (Reversal of an initial choice, justified by the root-cause finding.)
- **Guard/instrumentation live in `graph/sdk_utils.py` (shared), behavior change piloted on pathway_enrichment only.** Gives cross-node observability now without widening blast radius.
- **Structural signal is authoritative; phrase is a corroborator.** Degraded = expected tools were allowlisted AND zero `mcp__*` tool calls occurred (or the init tool list lacks them). The fallback phrase only raises confidence/logging. This avoids brittleness and false positives.
- **Empty allowlist never degrades.** The classifier keys off a non-empty `expected_tools`. Post-migration Phase B uses `allowed_tools=[]` (data-in-prompt), so the guard correctly never fires there and remains an inert safety net.
- **Phase B migration mirrors cold_start, including `SDK_SEMAPHORE`.** This simultaneously fixes the missing-semaphore gap (`pipeline_config.py:207-215`) without retrofitting working nodes.
- **Behind a config flag** (`drop_findings_on_degraded`, default `True`) for rollback, consistent with the repo's `pipeline_config` flag convention.
- **Single-PR delivery** (both stages together). Because the HTTP migration lands in the same PR, real Phase B data flows on merge — so `drop_findings_on_degraded=True` is a safe default with no interim window where the enrichment section renders empty. The MCP-degradation guard becomes a rarely-firing safety net; the Stage-2 emptiness guard is what actually protects the migrated node.

## Open Questions

### Resolved During Planning

- Root cause? Broken `uvx mcp-client-kestrel` stdio server (H1), not a race — confirmed by `cold_start.py:13-18` + PR #24 + an empirical `uvx` resolution check in the investigation.
- Guard action? Drop + flag degraded (user decision).
- Scope and delivery? Both work stages in a **single PR** (user decision) — guard + HTTP migration land together, so there is no interim window of degraded/empty enrichment output.

### Deferred to Implementation

- Exact SDK event/attribute that carries the "available tools at init" list — capture best-effort; degrade to `None` if the SDK version doesn't expose it. The `mcp__*` tool-use **count** is the reliable structural signal regardless.
- Whether `parse_enrichment_result` can be reused unchanged for the migrated in-prompt inference output, or needs a light prompt/parse adaptation to keep schema parity. (Note: it silently returns empty on `JSONDecodeError` — a malformed migrated response degrades to empty, which must be counted as "no data" by the emptiness guard, not as a healthy empty result.)
- Whether the LangGraph `operator.add` reducer treats an **absent** `direct_findings` key as a no-op contribution (current `except` handler omits it). This determines whether the Unit 3 independence fix must *explicitly* add `direct_findings=<staged two-hop>` to the rewritten `except` branch (assume yes — emit it explicitly).
- The concrete **interface shape** between Unit 4's emptiness signal and Unit 5's guard (per-entity `errored: bool` + a populated-entity count, vs. a sentinel return). Pin it in Unit 4's return type so Unit 5 can't misread "errored-empty" as "healthy-empty."
- Whether the emptiness guard counts populated entities **before or after** hub-filtering. Counting raw non-errored entities lets an exactly-2-entity run where one entity's neighbors are *all* hubs slip past the gate (rare at `edge_count >= 20`, and Phase A two-hop still grounds the output, so degraded-but-not-fabricated). If cheap, count post-hub-filter *usable* neighbors instead.

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation specification. The implementing agent should treat it as context, not code to reproduce.*

```
Phase B today (broken):
  options{ allowed_tools=[mcp__kestrel__*], mcp_servers={kestrel: uvx mcp-client-kestrel} }
     └─ query_with_usage ──► stdio subprocess FAILS to launch ──► 0 tools ──► model hallucinates
                                                                                  └─► fabricated shared_neighbors ─► Integration/Synthesis  ❌

Stage 1 (guard + instrumentation):
  query_with_usage now also reports: mcp_tool_calls, available_tools
     └─ classify_mcp_degradation(expected=[mcp__kestrel__*], calls=0, text, available)
            └─ degraded=true ─► drop SDK findings, KEEP Phase A two-hop, set degraded flag, log diagnostic  ✅

Stage 2 (durable fix, mirrors cold_start):
  prefetch one-hop via HTTP call_kestrel_tool(...)  ─►  options{ allowed_tools=[], SDK_SEMAPHORE, max_turns~1 }
     └─ model reasons over REAL data in-prompt ─► real shared_neighbors (expected_tools=[] ⇒ guard inert)  ✅
```

## Implementation Units

### Stage 1 — Guard + Instrumentation (implement first within the single PR)

- [ ] **Unit 1: Capture MCP tool-availability diagnostics in `query_with_usage`**

**Goal:** Make every SDK node's tool registration and MCP tool-call count observable, without changing any node's behavior.

**Requirements:** R3 (foundation), R6 (observability for deferred work)

**Dependencies:** None

**Files:**
- Modify: `backend/src/kestrel_backend/graph/sdk_utils.py` (the `query_with_usage` event loop, ~120-202)
- Modify: `backend/src/kestrel_backend/graph/state.py` (`ModelUsageRecord`, ~287 — add optional `mcp_tool_calls: int = 0`, `available_tools: list[str] | None = None`)
- Test: `backend/tests/test_sdk_utils.py`

**Approach:**
- In the `async for event` loop, count tool-use blocks using the **codebase-proven** predicate, not a string `type` check: `isinstance(block, ToolUseBlock) and block.name.startswith("mcp__")` (mirror `agent.py:429-430`). `ToolUseBlock` is a dataclass with `name`/`input`/`id` and **no `.type` attribute** — a `block.type == "tool_use"` check would always count 0 and false-flag every healthy run as degraded. Import `ToolUseBlock` into the central SDK import block in `sdk_utils.py` (it is not currently imported there).
- Best-effort capture the init tool list: the SDK delivers it as `SystemMessage(subtype="init", data={...})`, which has `.data` (not `.content`), so the current `hasattr(event, "content")` loop skips it. Add a branch `if isinstance(event, SystemMessage) and event.subtype == "init": available_tools = event.data.get(...)`. Import `SystemMessage` too. Keep the `None` fallback for SDK versions/streams that omit it — the tool-use **count** remains the authoritative signal.
- **Absent-SDK safety:** add `ToolUseBlock` and `SystemMessage` to the `except ImportError` branch as **sentinel stub classes** (a dummy `class`, exactly like the existing `_ResultMessageStub` at `sdk_utils.py:28-32`), **not** `= None`. With `= None`, the new `isinstance(...)` checks would raise `TypeError` when `HAS_SDK is False` (e.g. in tests, or any caller reaching the loop without the SDK). The existing `query_with_usage` early-returns before the loop when `HAS_SDK is False` (`sdk_utils.py:141`), so production is safe, but the stub pattern keeps the instrumentation and its tests robust.
- Because `ModelUsageRecord` is **frozen** (`ConfigDict(frozen=True)`, `state.py:294`), pass both new fields as **constructor arguments** at the single construction site (`sdk_utils.py:193-200`) — not via post-construction assignment, which would raise. Emit one diagnostic log line per node (registered? / call count). Keep the `(text, record)` return arity unchanged so existing callers are unaffected.

**Execution note:** Test-first — pure, deterministic given a fake event stream. Include a test that feeds a real-shaped `ToolUseBlock` so a healthy stream increments the count (guards against the false-positive-degraded trap above).

**Patterns to follow:** `agent.py:429-430` (`isinstance(block, ToolUseBlock)` tool detection); existing usage-accumulation loop in the same function; `ModelUsageRecord` construction at `sdk_utils.py:193-200`.

**Test scenarios:**
- Happy path: stream with three `mcp__kestrel__*` `tool_use` blocks → `record.mcp_tool_calls == 3`.
- Edge case: stream with no `tool_use` blocks → `mcp_tool_calls == 0`.
- Edge case: init/system event exposes a tool list → `available_tools` populated; init event absent → `available_tools is None` with no error.
- Integration: an existing caller that unpacks `(text, record)` still works (no arity change).

**Verification:** `ModelUsageRecord` carries accurate `mcp_tool_calls` for representative streams; a diagnostic line is logged per SDK call; no existing `query_with_usage` caller breaks.

- [ ] **Unit 2: Shared MCP-degradation classifier in `sdk_utils.py`**

**Goal:** One reusable decision function that flags the "tools unavailable → hallucinated" condition from structural signals.

**Requirements:** R1, R3

**Dependencies:** Unit 1 (consumes `mcp_tool_calls` / `available_tools`)

**Files:**
- Modify: `backend/src/kestrel_backend/graph/sdk_utils.py` (add `classify_mcp_degradation(...)` returning a small verdict object: `degraded: bool`, `reason: str`, `confidence: str`)
- Test: `backend/tests/test_sdk_utils.py`

**Approach:**
- Inputs: `expected_tools` (the allowlist), `mcp_tool_calls`, `result_text`, `available_tools` (optional).
- Degraded when `expected_tools` is non-empty AND (`available_tools` is known and lacks an expected tool → definitive) OR (`mcp_tool_calls == 0` → structural). The fallback-phrase regex (`not available in my current tool set` / `are not available`) is a corroborator that raises confidence and is logged — never the sole trigger.
- When `expected_tools` is empty → always `degraded=False` (protects data-in-prompt nodes like cold_start and post-migration Phase B).
- **Caveat for shared reuse:** the bare `mcp_tool_calls == 0` signal is only safe for nodes whose prompt *mandates* tool use (pathway_enrichment's `PATHWAY_ENRICHMENT_PROMPT` STEP 1 requires a `one_hop_query` per entity, so a healthy run always has ≥1 call). For a node where zero tool calls is a legitimate model choice, count==0 would false-positive — such nodes must require the init-list signal OR the corroborating phrase, not bare count. Carry this constraint into the deferred multi-node migration issue.

**Execution note:** Test-first — pure logic.

**Patterns to follow:** small pure helpers already in `sdk_utils.py` (e.g. `chunk`).

**Test scenarios:**
- Error path: expected non-empty + zero calls + phrase present → degraded, high confidence.
- Error path: expected non-empty + zero calls + no phrase → degraded (structural).
- Happy path: expected non-empty + ≥1 call → not degraded (honest run, even with zero shared neighbors).
- Edge case: `expected_tools == []` + zero calls → not degraded (data-in-prompt node).
- Edge case: `available_tools` provided and missing an expected tool, phrase absent → degraded (definitive, pre-first-turn signal).

**Verification:** classifier returns correct verdicts across the matrix; an honest no-results run with tools available is never flagged; an empty allowlist is never flagged.

- [ ] **Unit 3: Apply drop-on-degraded guard + `degraded` flag in pathway_enrichment Phase B**

**Goal:** Stop fabricated Phase B findings from propagating; disclose the degraded run; preserve Phase A.

**Requirements:** R1, R2, R5

**Dependencies:** Units 1, 2

**Files:**
- Modify: `backend/src/kestrel_backend/graph/nodes/pathway_enrichment.py` (Phase B, after parse, ~405-440)
- Modify: `backend/src/kestrel_backend/graph/state.py` — `DiscoveryState` has **no** degraded field today, so add one explicitly: name it `pathway_enrichment_degraded: bool` (default `False`). Use a **plain boolean, no reducer** — `pathway_enrichment` is a *serial* node (both `direct_kg`/`cold_start` branches edge **into** it; it is not a concurrent writer of its own output, `builder.py:154-155`), so single-writer last-write semantics are correct. Do **not** reach for `Annotated[list, operator.add]`. (A `degraded_nodes: list` would need that reducer; a per-node boolean avoids the question.)
- Modify: `backend/src/kestrel_backend/graph/pipeline_config.py` — add a `drop_findings_on_degraded: bool = True` field to the **existing** `PathwayEnrichmentConfig` (`pipeline_config.py:61`). It is already registered on `PipelineConfig` (`:182`) — do **not** create a new class or re-register.
- Test: `backend/tests/test_pathway_enrichment.py` (new)

**Approach:**
- After `query_with_usage` + `parse_enrichment_result`, call `classify_mcp_degradation(expected_tools=["mcp__kestrel__one_hop_query","mcp__kestrel__get_nodes"], mcp_tool_calls=record.mcp_tool_calls, result_text, available_tools=record.available_tools)`.
- If degraded and `config.drop_findings_on_degraded`: discard the **SDK-derived** `shared_neighbors`, `biological_themes`, and SDK-derived `findings`; **keep the Phase A two-hop findings** (HTTP-derived); set the degraded marker; emit a structured degraded-run log (node, tripped signal, offending tool list); add an explanatory `errors`/notice entry.
- **Discriminator (named):** SDK theme findings carry `source="pathway_enrichment"` (`pathway_enrichment.py:431`); Phase A two-hop findings carry `source="pathway_enrichment_two_hop"` (`:444`). On degrade, build findings from the two-hop set only and skip the SDK theme-findings loop (`:419-433`); return `shared_neighbors=[]` and `biological_themes=[]`. Do the filtering **inside** `run()` before assembling the result dict. The three output keys differ in reducer semantics: `direct_findings` is cross-node **additive** (`operator.add`, `state.py:348`) so SDK findings must be filtered *before* emit (additive reducers can't subtract after merge); `shared_neighbors` is additive but pathway-owned (emitting `[]` adds nothing — safe); `biological_themes` has **no** reducer (`state.py:360`) and is single-writer, so emitting `[]` is a clean overwrite — do **not** add `operator.add` to it.
- **Independence fix (must be complete):** the two-hop *neighbors* are already computed outside the `try` (`pathway_enrichment.py:342-348`), but the two-hop **Finding objects**, the `result_dict` assembly, and the `return` all sit *inside* the Phase B `try` (~`:435-465`), and the `except` handler (`:467-474`) returns a hardcoded `{shared_neighbors:[], biological_themes:[], errors:[…]}` with **no `direct_findings` key at all**. So merely "computing two-hop outside the try" does not satisfy R5 — a Phase B exception still drops the two-hop findings. The fix must also: (a) stage the two-hop `Finding` objects in an enclosing-scope variable, and (b) **rewrite the `except` handler to return `direct_findings=<staged two-hop findings>`** (and confirm whether an absent `direct_findings` key is a reducer no-op vs. required — see Deferred). Add a test that a Phase B *exception* (not just a degraded classification) preserves two-hop findings.
- **Shared degraded helper:** factor the degraded side-effects (set `pathway_enrichment_degraded=True`, keep two-hop, emit notice + structured log) into a small helper that takes an explicit `reason`. Call it here with `reason="mcp_unavailable"`; Unit 5's emptiness path reuses it with `reason="prefetch_no_data"` instead of routing through the MCP classifier (whose "zero mcp calls" reason is misleading for a no-MCP node).

**Execution note:** Characterization-first — capture the current Phase B output schema before changing it, so the drop path and the healthy path are both pinned.

**Patterns to follow:** existing config-flag pattern (`DirectKGConfig.multi_hop_enabled`, `IntegrationConfig`); the existing timeout-return shape in Phase B (`pathway_enrichment.py:399-403`).

**Test scenarios:**
- Error path: degraded transcript (zero `mcp__*` calls + fallback phrase) → output has degraded marker, no SDK `shared_neighbors`/`biological_themes`, Phase A two-hop findings retained.
- Happy path: healthy transcript (≥1 tool call, valid JSON) → not degraded, SDK findings present.
- Edge case: `drop_findings_on_degraded=False` → output retained but degraded still logged (flag regression guard).
- Error path: Phase B raises an exception → Phase A two-hop findings are still present in the output (independence fix; not only the degraded-classification path).
- Edge case: `HAS_SDK is False` → existing placeholder path unchanged.

**Verification:** with a simulated fallback transcript the pipeline carries `degraded=true` and zero fabricated SDK neighbors while retaining real two-hop findings; with a healthy transcript behavior is unchanged.

### Stage 2 — HTTP migration of Phase B (durable fix; same PR)

- [ ] **Unit 4: HTTP one-hop prefetch for Phase B entities**

**Goal:** Fetch real one-hop neighbor data via HTTP rather than MCP tools.

**Requirements:** R4

**Dependencies:** Stage 1 units (guard in place as safety net)

**Files:**
- Modify: `backend/src/kestrel_backend/graph/nodes/pathway_enrichment.py` (new prefetch helper, e.g. `prefetch_one_hop_neighbors(entities)`)
- Test: `backend/tests/test_pathway_enrichment.py`

**Approach:**
- Mirror `cold_start`'s `get_entity_connections` (`cold_start.py:148-224`): for each selected entity, `call_kestrel_tool("one_hop_query", {...})` via the HTTP client; `asyncio.gather(..., return_exceptions=True)`; optionally `get_nodes` for degree/hub flags. Build a structured neighbor summary for the prompt. Reuse hub thresholds from `pipeline_config`.
- `pathway_enrichment.py` does **not** yet import `call_kestrel_tool` (only `multi_hop_query`) — add the import. Use the **HTTP param shape** from `get_entity_connections` (`call_kestrel_tool("one_hop_query", {"start_node_ids": …})`), **not** the `one_hop_query(curie=…)` signature shown in `PATHWAY_ENRICHMENT_PROMPT` (that's the MCP-tool form; copying it would pass the wrong argument).
- **Emptiness signal (feeds Unit 5's guard) — threshold matters:** every Phase B entity is *guaranteed dense* (`edge_count >= MIN_EDGE_COUNT = 20`, filtered at `pathway_enrichment.py:299`), so an empty one-hop result for such an entity is almost never genuine sparsity — it is overwhelmingly an HTTP/parse/API failure. But `get_entity_connections` collapses *all* failure modes (query-failed, no-content, parse-error, API-error, exception) into the **same** `{"edges": [], …}` shape, indistinguishable from a real empty. Two consequences: (a) propagate a per-entity **`errored: bool`** so success is not merely inferred from non-empty `edges`; (b) the guard must fire when **fewer than 2 entities returned real (non-errored) neighbor data**, not only when *all* are empty — shared-neighbor analysis is meaningless below 2 populated entities, and with the documented 2-entity minimum a *single* silent failure already makes the prompt sparse and hallucination-prone.

**Patterns to follow:** `cold_start.py:148-224` (`get_entity_connections`: `call_kestrel_tool("one_hop_query", …)` with per-entity exception handling); `cold_start.py:440-460` (the gather site).

**Test scenarios:**
- Happy path: mocked `call_kestrel_tool` → neighbors parsed and grouped per entity.
- Error path: one entity's call raises → that entity is marked `errored=True` (distinct from a genuine empty result) + a warning; other entities still succeed.
- Error path: fewer than 2 entities return real non-errored data (e.g. 1 of 2 silently fails) → prefetch reports the "no data" emptiness signal, even though one entity succeeded.
- Edge case: hub neighbors filtered/flagged per the configured threshold.

**Verification:** given mocked HTTP responses, the helper returns per-entity one-hop neighbor data with hub handling and resilient error behavior.

- [ ] **Unit 5: Convert Phase B SDK call to data-in-prompt inference (no MCP)**

**Goal:** Replace the broken stdio MCP SDK config with cold_start-style inference over prefetched data.

**Requirements:** R4, R5

**Dependencies:** Unit 4

**Files:**
- Modify: `backend/src/kestrel_backend/graph/nodes/pathway_enrichment.py` (Phase B options + prompt; add an inference prompt constant if needed; define a module-level `SDK_SEMAPHORE`)
- Modify: `backend/src/kestrel_backend/graph/pipeline_config.py` (add an `sdk_semaphore` field to `PathwayEnrichmentConfig` — it currently has only `hub_threshold`)
- Test: `backend/tests/test_pathway_enrichment.py`

**Approach:**
- `pathway_enrichment.py` has **no** module-level `SDK_SEMAPHORE` today, and `PathwayEnrichmentConfig` has no `sdk_semaphore` field. So "acquire `SDK_SEMAPHORE`" requires first adding the config field and defining `SDK_SEMAPHORE = asyncio.Semaphore(_config.sdk_semaphore)` at module level, mirroring `cold_start.py:42`. Do **not** add `pathway_enrichment` to the `get_semaphore()` map (`pipeline_config.py:201-216`) — the module-level `SDK_SEMAPHORE` pattern is canonical (cold_start uses it directly and never calls `get_semaphore()`); leaving the map alone avoids introducing a `ValueError` surface.
- Build options with `allowed_tools=[]`, small `max_turns` (1–2), `async with SDK_SEMAPHORE` (this also closes the missing-semaphore gap), and **no** `mcp_servers` / `get_kestrel_mcp_config`. Feed the Unit 4 prefetched data into an inference prompt; parse to the same output schema (`shared_neighbors`, `biological_themes`) via `parse_enrichment_result` (adapt prompt/parse only as needed for parity).
- **Post-migration emptiness guard:** the MCP classifier is inert here (`expected_tools == []` ⇒ never degraded) **by design** — so it no longer protects this node. Add an independent check: if Unit 4's prefetch reports the "no data" emptiness signal (**fewer than 2 entities returned real non-errored neighbor data**), mark Phase B degraded via the **shared helper with `reason="prefetch_no_data"`** (Unit 3) and skip inference rather than running it on a sparse prompt and re-hallucinating. Do **not** route this through the MCP classifier. State the interaction in code comments.

**Execution note:** Characterization-first — assert output-schema parity with the pre-migration healthy path.

**Patterns to follow:** `cold_start.py:481-495` (`allowed_tools=[]`, `max_turns=1`, `SDK_SEMAPHORE`, inference prompt with embedded KG data, `HAS_SDK` graceful fallback).

**Test scenarios:**
- Happy path: mocked prefetch + mocked SDK JSON → `shared_neighbors` derived from real data; schema parity with pre-migration.
- Edge case: `HAS_SDK is False` → graceful return mirroring cold_start.
- Integration: migrated Phase B output merges with Phase A two-hop findings exactly as before.
- Error path: prefetch reports the no-data emptiness signal (fewer than 2 entities populated, e.g. 1 of 2 silently errored) → Phase B marked degraded (`reason="prefetch_no_data"`), inference is **not** run, Phase A two-hop findings retained.
- Regression: the MCP classifier (Unit 2) does not flag the migrated path (`expected_tools=[]`); the emptiness guard is what protects it now.

**Verification:** Phase B produces real one-hop-derived shared neighbors with no stdio MCP dependency; degraded never set on the healthy migrated path; combined output matches the prior schema.

- [ ] **Unit 6: Retire pathway_enrichment's stdio config usage + record the pipeline-wide finding**

**Goal:** Remove the dead stdio path from this node and surface the broader latent exposure for separate follow-up.

**Requirements:** R6

**Dependencies:** Unit 5

**Files:**
- Modify: `backend/src/kestrel_backend/graph/nodes/pathway_enrichment.py` (drop now-unused `get_kestrel_mcp_config` / `KESTREL_*` imports; add a comment)
- Optional: a short `docs/solutions/` note via `ce:compound` after merge

**Approach:**
- Ensure `pathway_enrichment` no longer imports or calls `get_kestrel_mcp_config`. Add a comment noting that `entity_resolution`, `direct_kg`, `integration`, `temporal`, and `triage` still use the same broken `uvx mcp-client-kestrel` stdio path and should be audited/migrated under a separate issue. **Do not modify those nodes.**

**Test expectation:** none — import-cleanliness only; behavior covered by Unit 5 tests.

**Verification:** `pathway_enrichment` has no remaining reference to the stdio MCP config; a follow-up issue is filed for the other five nodes.

## System-Wide Impact

- **Interaction graph:** `direct_kg ∥ cold_start → pathway_enrichment → integration → … → synthesis` (`builder.py:48,154-155`). The degraded flag set in Unit 3 must be readable by Integration/Synthesis (and ideally the UI) for disclosure. **Scope note:** emitting the flag is in scope; *rendering* it (synthesis text + a UI badge) is a follow-up — no current consumer reads it, so R2 is satisfied at the "flag emitted" level, not yet "surfaced to the user." Track the disclosure-rendering as a small follow-up.
- **Error propagation:** degraded is a *handled* state, not an exception — Phase A findings and the rest of the pipeline continue; only fabricated SDK findings are suppressed.
- **State lifecycle risks:** the `pathway_enrichment_degraded` marker is a new defaulted (`False`) single-writer boolean — no reducer, so it can't trigger a concurrent-update error and existing merges are unaffected. `ModelUsageRecord` gains optional defaulted fields only.
- **API surface parity:** the classifier + instrumentation live in `sdk_utils.py`, so the other five SDK nodes can adopt them later with no duplication — but this plan does not wire their behavior.
- **Integration coverage:** a test that drives Phase B degraded and asserts Integration/Synthesis do **not** ingest fabricated neighbors is the highest-value cross-layer check (Unit 3).
- **Unchanged invariants:** Phase A HTTP analysis, the HTTP `kestrel_client`, working nodes' SDK calls, and `builder.py` topology are explicitly unchanged.

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Dropping Phase B reduces enrichment when degraded | Phase A HTTP two-hop still contributes real neighbors; Stage 2 removes the degradation source entirely |
| `ModelUsageRecord` shape change breaks callers | Additive optional fields only; return arity of `query_with_usage` unchanged |
| Classifier false-positive on data-in-prompt nodes | Empty `expected_tools` never degrades — covered by an explicit test |
| HTTP migration changes how Phase B gets data | Mirror the proven `cold_start`/PR #24 pattern; schema-parity tests; guard ships first as safety net |
| Scope creep into the other five broken nodes | Hard scope boundary; instrumentation quantifies their exposure for a separate issue |
| SDK doesn't expose an init tool list on this version | Tool-use **count** is the reliable signal; init list is best-effort and may be `None` |

## Documentation / Operational Notes

- Run tests with `cd backend && uv run python -m pytest tests/ -v -m "not integration"` (see the venv-path-spaces learning).
- After merge, a `ce:compound` note documenting the `mcp-client-kestrel`-doesn't-exist root cause + the guard/HTTP remedy would compound well (the cold_start docstring is the only current record).
- The Stage 1 diagnostics will, in production logs, reveal whether the other five SDK nodes are also silently degraded — use that to prioritize the deferred migration issue.

## Sources & References

- Investigation: 6-agent root-cause workflow `investigate-mcp-tools-unavailable` (run `wf_d2cea02f-039`), this session — ranked hypotheses, guard design, instrumentation, scope/risk.
- Issue: #44 (Investigate: LLM SDK phase reports MCP tools unavailable)
- Precedent: PR #24 (cold_start HTTP migration); `backend/src/kestrel_backend/graph/nodes/cold_start.py:1-18,440-495`
- Broken config: `backend/src/kestrel_backend/graph/sdk_utils.py:35-50,120-202`
- Phase B: `backend/src/kestrel_backend/graph/nodes/pathway_enrichment.py:170-256,342-440`
- Semaphore gap: `backend/src/kestrel_backend/graph/pipeline_config.py:207-215`
- Related learning: `docs/solutions/developer-experience/pytest-venv-path-spaces-module-invocation-2026-06-01.md`
