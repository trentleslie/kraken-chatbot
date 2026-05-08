---
title: "feat: Add model_usages cost tracking to DiscoveryState"
type: feat
status: active
date: 2026-05-07
deepened: 2026-05-07
---

# feat: Add model_usages cost tracking to DiscoveryState

## Overview

Add a `model_usages` reducer field to `DiscoveryState` and instrument each SDK-calling graph node to emit `ModelUsageRecord` entries. This enables AstaBench cost tracking for the KRAKEN evaluation pipeline. A separate repo (`kraken-chatbot-solver`) will consume this field to report costs to Inspect AI.

**Branch workflow:** Implement on `feat/model-usages-cost-tracking` branched from `dev`. PR targets `dev` for Greptile automated review. Address any Greptile findings before merge.

## Problem Frame

AstaBench requires accurate per-model cost tracking for Pareto frontier analysis (cost vs. performance). The discovery pipeline currently has no node-level usage tracking — `agent.py` tracks turn-level costs for the chat agent, but the 8 graph nodes that call Claude Agent SDK `query()` silently discard `ResultMessage.usage` data. Without per-node usage records on the state, the solver repo cannot compute pipeline execution costs.

## Requirements Trace

- R1. Define `ModelUsageRecord` Pydantic model in `state.py` with fields: `model_name`, `node_name`, `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_creation_tokens`
- R2. Add `model_usages: Annotated[list[ModelUsageRecord], operator.add]` to `DiscoveryState`
- R3. Instrument all nodes that call SDK `query()` to emit usage records (8 nodes)
- R4. Nodes that don't call LLM (intake, literature_grounding) should not emit usage records
- R5. Existing tests must continue to pass

## Scope Boundaries

- Do not change graph structure or node logic — only add usage tracking instrumentation
- Do not change existing node return signatures beyond adding `model_usages`
- Do not add `inspect-ai` as a dependency — the solver repo handles Inspect AI conversion
- Do not build cost aggregation, reporting, or database storage — that belongs in the solver repo

## Context & Research

### Relevant Code and Patterns

- `backend/src/kestrel_backend/graph/state.py` — 12+ fields already use `Annotated[list[X], operator.add]` reducer pattern for parallel writes
- `backend/src/kestrel_backend/graph/sdk_utils.py` — centralized SDK imports, `query`, `ClaudeAgentOptions`, `get_kestrel_mcp_config()`, `chunk()`
- `backend/src/kestrel_backend/agent.py:510-534` — proven pattern for extracting usage from `ResultMessage.usage` (dict with `input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`)
- All 8 SDK-calling nodes use identical streaming pattern: `async for event in query(prompt=..., options=...)` → collect text blocks

### Nodes by SDK Usage

| Node | Uses SDK `query()` | Notes |
|------|-------------------|-------|
| intake | No | Heuristic parsing only |
| entity_resolution | Yes | Tier 2 fallback |
| triage | Yes | Fallback for edge counting |
| direct_kg | Yes | Tier 2 fallback |
| cold_start | Yes | Primary reasoning path |
| pathway_enrichment | Yes | Shared neighbor analysis |
| integration | Yes | Gap analysis reasoning |
| temporal | Yes | Classification via SDK |
| synthesis | Yes | Report + hypothesis generation |
| literature_grounding | No | HTTP APIs only (Semantic Scholar, OpenAlex, Exa, PubMed) |

### SDK Response Shape

`ResultMessage` is the final event in the `query()` stream. Its `.usage` is a dict:
```
{"input_tokens": int, "output_tokens": int, "cache_creation_input_tokens": int, "cache_read_input_tokens": int}
```

Some nodes call `query()` multiple times (e.g., entity_resolution per-entity, cold_start per-entity). Each call produces its own `ResultMessage` with independent usage.

## Key Technical Decisions

- **Shared wrapper in `sdk_utils.py`**: Add a `query_with_usage()` function that wraps the common `async for event in query(...)` streaming pattern and returns `(text, ModelUsageRecord | None)`. This centralizes usage extraction and reduces per-node changes to swapping the call site. The existing `agent.py:510-534` pattern is the reference implementation.
- **Wrapper does NOT handle timeouts**: Nodes that need timeouts (cold_start, pathway_enrichment) keep their own `asyncio.wait_for()` wrapping the `query_with_usage()` call. This is necessary because those nodes have domain-specific timeout recovery logic (cold_start constructs `Finding` objects from already-gathered analogues; pathway_enrichment returns node-specific error dicts). A generic wrapper cannot replicate this.
- **Dual-path usage extraction**: Following `agent.py:510-534`, the wrapper accumulates usage from ALL events with a `.usage` attribute, not just `ResultMessage`. This ensures accurate token counts for multi-turn nodes like pathway_enrichment (`max_turns=25`) where `ResultMessage.usage` may not contain cumulative totals.
- **Reducer for parallel safety**: `model_usages` uses `operator.add` reducer so parallel branches (direct_kg + cold_start) can safely append records in the same superstep.
- **No `to_inspect_usage()` method**: The solver repo owns the conversion from `ModelUsageRecord` → Inspect AI's `ModelUsage`. This avoids coupling this repo to `inspect_ai`'s API surface.
- **Cache token field rename is intentional**: The SDK response uses `cache_creation_input_tokens` and `cache_read_input_tokens`. `ModelUsageRecord` stores these as `cache_creation_tokens` and `cache_read_tokens` (dropping the `input_` prefix) for a cleaner domain model. The wrapper handles the mapping.
- **No model name inference**: Nodes should pass the model name explicitly. The SDK doesn't expose which model was used in `ResultMessage`. For now, a constant like `"anthropic/claude-sonnet-4-20250514"` is acceptable since all nodes use the same model.

## Open Questions

### Resolved During Planning

- **Decorator vs per-node?** Resolved: shared `query_with_usage()` wrapper. All 8 nodes use the identical streaming pattern (`async for event in query(...)`), making a wrapper natural. Per-node inline extraction would duplicate the `ResultMessage` handling 8 times.
- **Which nodes need instrumentation?** 8 of 10 — all except `intake` (heuristic) and `literature_grounding` (HTTP APIs only, no LLM).
- **How to handle nodes with multiple `query()` calls?** Each call returns its own `ModelUsageRecord`. Nodes accumulate them in a local list and return the full list in `model_usages`.

### Deferred to Implementation

- **Exact model name string**: Nodes may use different models in the future. For now, a module-level constant or parameter on `query_with_usage()` is sufficient. The implementer should check if the SDK exposes the model name anywhere in the response.

## Implementation Units

- [x] **Unit 1: ModelUsageRecord and state field**

  **Goal:** Define the Pydantic model and add the reducer field to `DiscoveryState`.

  **Requirements:** R1, R2

  **Dependencies:** None

  **Files:**
  - Modify: `backend/src/kestrel_backend/graph/state.py`
  - Test: `backend/tests/test_state_contracts.py`

  **Approach:**
  - Add `ModelUsageRecord(BaseModel)` with `model_config = ConfigDict(frozen=True)` following existing patterns (e.g., `Finding`, `NoveltyScore`)
  - Fields: `model_name: str`, `node_name: str`, `input_tokens: int = 0`, `output_tokens: int = 0`, `cache_read_tokens: int = 0`, `cache_creation_tokens: int = 0` — all token fields with `ge=0` constraint. Note: field names intentionally drop the `input_` prefix from the SDK's `cache_creation_input_tokens` / `cache_read_input_tokens` for a cleaner domain model
  - Add `model_usages: Annotated[list[ModelUsageRecord], operator.add]` to `DiscoveryState` under a `# === Cost Tracking ===` section

  **Patterns to follow:**
  - Existing Pydantic models in `state.py` (e.g., `Finding`, `NoveltyScore`, `EntityResolution`)
  - Existing reducer pattern: `Annotated[list[X], operator.add]`

  **Test scenarios:**
  - Happy path: `ModelUsageRecord` instantiates with all fields, serializes to dict correctly
  - Happy path: `ModelUsageRecord` with default token values (all 0) is valid
  - Edge case: Negative token values rejected by `ge=0` constraint
  - Happy path: Frozen model rejects mutation (assign to field raises)
  - Integration: `DiscoveryState` accepts `model_usages` field with operator.add reducer (two dicts with `model_usages` lists merge correctly)

  **Verification:**
  - `ModelUsageRecord` can be instantiated and serialized
  - State contract tests pass with new field

- [x] **Unit 2: `query_with_usage()` wrapper in sdk_utils.py**

  **Goal:** Add a shared utility that wraps the SDK streaming loop and extracts usage from `ResultMessage`.

  **Requirements:** R3 (enabling infrastructure)

  **Dependencies:** Unit 1

  **Files:**
  - Modify: `backend/src/kestrel_backend/graph/sdk_utils.py`
  - Test: `backend/tests/test_sdk_utils.py`

  **Approach:**
  - Add `async def query_with_usage(prompt: str, options: Any, node_name: str, model_name: str = "anthropic/claude-sonnet-4-20250514") -> tuple[str, ModelUsageRecord | None]`
  - Import `ResultMessage` from `claude_agent_sdk` (top-level package, matching `agent.py:20`). Add it to the existing try/except import block alongside `query`, `ClaudeAgentOptions`. For `HAS_SDK = False`, add a stub sentinel class (e.g., `class _ResultMessageStub: pass`) so `isinstance(event, ResultMessage)` returns `False` rather than raising `TypeError`
  - Internally: stream `query(prompt, options)`, collect text blocks (existing pattern). Accumulate usage from ALL events that have a `.usage` attribute (following `agent.py:510-534` dual-path pattern) — not just `ResultMessage`. This ensures accurate counts for multi-turn nodes like pathway_enrichment (`max_turns=25`)
  - Map SDK field names to model field names: `cache_creation_input_tokens` → `cache_creation_tokens`, `cache_read_input_tokens` → `cache_read_tokens`
  - Return `(joined_text, record)` where `record` is `None` if no event had usage data
  - The wrapper does NOT handle timeouts — nodes with timeout requirements (cold_start, pathway_enrichment) wrap their `query_with_usage()` call in their own `asyncio.wait_for()`
  - When `HAS_SDK = False`, raise `RuntimeError` — consistent with existing SDK-unavailable behavior where nodes check `HAS_SDK` before calling SDK functions

  **Patterns to follow:**
  - `agent.py:510-534` — `ResultMessage` usage extraction
  - Existing `sdk_utils.py` patterns for SDK availability checks

  **Test scenarios:**
  - Happy path: Mock `query()` to yield text events + `ResultMessage` with usage dict → returns correct text and `ModelUsageRecord`
  - Happy path: `ResultMessage` with cache tokens → cache fields populated correctly
  - Happy path: Multi-turn stream with usage on multiple events → accumulates tokens across all events
  - Edge case: Events with no `.usage` attribute → skipped, no error
  - Edge case: No events have usage data → returns `None` usage record
  - Error path: `HAS_SDK = False` → raises `RuntimeError`

  **Verification:**
  - `test_sdk_utils.py` passes with new tests
  - Function signature is compatible with all 8 nodes' call patterns

- [x] **Unit 3: Instrument nodes — entity_resolution, triage, direct_kg**

  **Goal:** Replace raw `query()` streaming with `query_with_usage()` in the first batch of nodes and return `model_usages` in state updates.

  **Requirements:** R3, R5

  **Dependencies:** Unit 2

  **Files:**
  - Modify: `backend/src/kestrel_backend/graph/nodes/entity_resolution.py`
  - Modify: `backend/src/kestrel_backend/graph/nodes/triage.py`
  - Modify: `backend/src/kestrel_backend/graph/nodes/direct_kg.py`

  **Approach:**
  - For each node: replace `async for event in query(...)` + text collection with `query_with_usage()` call
  - Each call returns `(text, record | None)`. Accumulate non-None records in a local list across loop iterations
  - Add `"model_usages": accumulated_records` to the return dict
  - **entity_resolution**: calls `query()` per-entity in a loop — accumulate records across all entities
  - **triage**: calls `query()` per-curie as SDK fallback — accumulate records
  - **direct_kg**: calls `query()` per-entity as Tier 2 fallback — accumulate records
  - For `operator.add` reducer fields, omitting the key from the return dict is equivalent to returning `[]`. Only add `"model_usages"` at the call sites where SDK usage is actually collected — do not modify early-return paths. This minimizes lines changed and avoids touching ~15+ early-return statements across all nodes

  **Patterns to follow:**
  - Existing node return patterns: `return {"field1": [...], "field2": [...], "errors": [...]}`
  - Keep the same error handling and fallback structure

  **Test scenarios:**
  - Happy path: Node returns `model_usages` key in output dict when SDK path is taken
  - Happy path: Node returns `model_usages: []` when primary HTTP path succeeds (no SDK fallback)
  - Edge case: Node processes multiple entities — each SDK call produces a separate `ModelUsageRecord` with correct `node_name`
  - Integration: Node output merges correctly with `DiscoveryState` reducer

  **Verification:**
  - Existing tests pass unchanged
  - Each instrumented node includes `model_usages` in its return dict

- [x] **Unit 4: Instrument nodes — cold_start, pathway_enrichment, integration**

  **Goal:** Instrument the second batch of nodes, including those with `collect_events()` inner functions.

  **Requirements:** R3, R5

  **Dependencies:** Unit 2

  **Files:**
  - Modify: `backend/src/kestrel_backend/graph/nodes/cold_start.py`
  - Modify: `backend/src/kestrel_backend/graph/nodes/pathway_enrichment.py`
  - Modify: `backend/src/kestrel_backend/graph/nodes/integration.py`

  **Approach:**
  - **cold_start**: Replace `collect_events()` inner function with `query_with_usage()`, but keep the node's own `asyncio.wait_for()` wrapping it — cold_start has domain-specific timeout recovery (constructs `Finding` from already-gathered analogues, returns specific error message). The `event_count` tracked via `nonlocal` in `collect_events()` is only used for a diagnostic `logger.info` line — accept its loss rather than complicating the wrapper interface. Each per-entity call to `query_with_usage()` returns `(text, record | None)`. Entities that exit early (no analogues, low similarity, HAS_SDK=False, timeout) produce no record — the outer loop accumulates non-None records across all entities
  - **pathway_enrichment**: Same `collect_events()` replacement but simpler — pathway_enrichment does NOT use `nonlocal event_count`. Keep the node's own `asyncio.wait_for()` for timeout handling
  - **integration**: Calls `query()` for gap analysis — straightforward replacement, no timeout needed
  - All three add `"model_usages": accumulated_records` only at the SDK call sites — early-return paths omit the key (safe for `operator.add` reducers)

  **Patterns to follow:**
  - Same pattern as Unit 3 nodes
  - Preserve existing semaphore and concurrency patterns (these are per-node for good reasons)

  **Test scenarios:**
  - Happy path: cold_start emits usage records for each entity inference
  - Happy path: pathway_enrichment emits usage records for shared-neighbor queries
  - Happy path: integration emits usage records for gap analysis
  - Edge case: cold_start skips entities (cold_start_skipped_count) — only emits records for entities actually processed

  **Verification:**
  - Existing tests pass unchanged
  - Each instrumented node includes `model_usages` in its return dict

- [x] **Unit 5: Instrument nodes — temporal, synthesis**

  **Goal:** Instrument the final two SDK-calling nodes.

  **Requirements:** R3, R5

  **Dependencies:** Unit 2

  **Files:**
  - Modify: `backend/src/kestrel_backend/graph/nodes/temporal.py`
  - Modify: `backend/src/kestrel_backend/graph/nodes/synthesis.py`

  **Approach:**
  - **temporal**: Single `query()` call for classification — straightforward replacement
  - **synthesis**: Single `query()` call for report generation — straightforward replacement. Note: synthesis has multiple `fallback_report(state)` paths for when query returns empty text. After replacing with `query_with_usage()`, check `text` for emptiness and fall back to `fallback_report()` as before
  - Both add `"model_usages": [record]` at the SDK call site (filter None)
  - Verify that `intake.py` and `literature_grounding.py` do NOT need changes (no SDK calls)

  **Patterns to follow:**
  - Same pattern as Unit 3/4 nodes

  **Test scenarios:**
  - Happy path: temporal emits single usage record for classification
  - Happy path: synthesis emits single usage record for report generation
  **Verification:**
  - All existing tests pass: `cd backend && uv run pytest tests/ -v -m "not integration"`
  - All 8 SDK-calling nodes include `model_usages` in their return dicts
  - intake and literature_grounding are unchanged (no SDK calls, no `model_usages` key needed)

## System-Wide Impact

- **Interaction graph:** No callbacks, middleware, or observers affected. The change is additive — a new field on state and usage extraction in the streaming loop.
- **Error propagation:** If usage extraction fails (e.g., unexpected `ResultMessage` shape), the wrapper should log a warning and return 0 tokens rather than failing the node. Node logic must not be disrupted by cost tracking.
- **State lifecycle risks:** The `operator.add` reducer handles parallel writes safely. No partial-write risk since records are appended atomically.
- **API surface parity:** `runner.py`'s `run_discovery()` returns the full `DiscoveryState` from `graph.ainvoke()`, so `model_usages` flows through automatically. No runner changes needed.
- **Integration coverage:** The reducer merge behavior should be tested — two parallel branches (direct_kg + cold_start) both emitting `model_usages` should produce a combined list.
- **Unchanged invariants:** All existing state fields, graph structure, node routing, and analysis logic remain unchanged. No existing return fields are modified — `model_usages` is purely additive.

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| `ResultMessage` shape changes in future SDK versions | Defensive extraction with `.get()` and fallback to 0, matching `agent.py` pattern |
| `cold_start`'s `collect_events()` uses `nonlocal event_count` (pathway_enrichment does not) | Accept loss of diagnostic log line — it has no functional effect. Wrapper focuses on text + usage only |

## Sources & References

- Task spec: `AGENT-TASK-model-usages.md`
- Usage extraction pattern: `backend/src/kestrel_backend/agent.py:510-534`
- State schema: `backend/src/kestrel_backend/graph/state.py:261-342`
- SDK utilities: `backend/src/kestrel_backend/graph/sdk_utils.py`
- Related: AstaBench evaluation strategy (referenced in task spec)
