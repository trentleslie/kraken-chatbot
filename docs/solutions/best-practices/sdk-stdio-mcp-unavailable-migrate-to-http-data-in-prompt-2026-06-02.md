---
title: "When a Claude Agent SDK node's stdio MCP server can't launch, it silently hallucinates — detect it and migrate to HTTP data-in-prompt"
date: 2026-06-02
category: docs/solutions/best-practices
module: kestrel_backend.graph
problem_type: best_practice
component: assistant
severity: high
applies_when:
  - "A discovery-pipeline node uses the Claude Agent SDK with a stdio MCP server (uvx/npx-launched) and allowed_tools requesting mcp__* tools"
  - "An SDK phase's output looks plausible but may be fabricated from training data rather than the knowledge graph"
  - "A node depends on an external MCP package that may not exist / fail to launch in the deploy environment"
tags:
  - claude-agent-sdk
  - mcp
  - stdio
  - hallucination
  - degradation-guard
  - data-in-prompt
  - kestrel
  - pathway-enrichment
---

# When a Claude Agent SDK node's stdio MCP server can't launch, it silently hallucinates — detect it and migrate to HTTP data-in-prompt

## Context

Issue #44 reported that `pathway_enrichment` Phase B intermittently announced *"The Kestrel MCP tools are not available in my current tool set"* and then produced one-hop analysis fabricated from the model's training knowledge. Investigation found the root cause is **not** a race: the stdio MCP server is configured to launch `uvx mcp-client-kestrel` (`graph/sdk_utils.py:35-36`), but **`mcp-client-kestrel` does not exist on PyPI** — already discovered and documented in `graph/nodes/cold_start.py:1-18` (PR #24 migrated cold_start off it for exactly this reason). The stdio subprocess can never start, so the tools never register; under `permission_mode="bypassPermissions"` the model fills the gap by hallucinating.

The same broken stdio path is used by **all six SDK nodes** (`pathway_enrichment`, `entity_resolution`, `direct_kg`, `integration`, `temporal`, `triage`) — each passes `mcp_servers={"kestrel": get_kestrel_mcp_config()}` and requests `mcp__kestrel__*` tools. The failure is masked to varying degrees because most nodes also have an HTTP tier (see [[discovery-pipeline-one-graph-methods-within-nodes-2026-05-28]]). `pathway_enrichment` was the most visible because its Phase B one-hop analysis was SDK-primary.

## Guidance

Three reusable practices came out of this fix (PR #60):

**1. Prefer HTTP data-in-prompt over stdio MCP for KG access.** Fetch the data deterministically via the HTTP client (`call_kestrel_tool`), then have the SDK *reason over the provided data* with `allowed_tools=[]` and no `mcp_servers`. This is the proven `cold_start` pattern — the model can't hallucinate KG facts it was handed, and you don't depend on a subprocess launching.

**2. Add a structural degradation guard at the shared SDK layer.** Don't trust SDK output that *claims* to use tools. Instrument the SDK event stream to count `mcp__*` tool-use blocks and capture the init tool list, then classify:
- **Definitive:** init tool list known and missing an expected tool.
- **Structural:** expected tools non-empty AND zero `mcp__*` calls (only safe for prompts that *mandate* tool use).
- **Corroborator only:** the fallback phrase (`"not available in my current tool set"`) — never the sole trigger; model wording is brittle.
- **Never degrade** when `expected_tools == []` (data-in-prompt nodes), or when the init list confirms all tools present and there's no phrase.

On degradation, **drop the unreliable findings and set a `degraded` flag** so synthesis/UI can disclose the gap — silent fabricated findings are worse than a visible gap.

**3. Stage independent (HTTP) findings outside the SDK `try`.** Anything not produced by the fragile SDK call (e.g. a parallel HTTP analysis) must be computed *before* the `try` and emitted on **every** exit path — happy, degraded-drop, timeout, and exception — or an SDK failure silently discards real results too.

## Why This Matters

- **The failure lies about its cause.** A `ModuleNotFoundError`-style "tools unavailable" reads like a transient MCP hiccup; the real cause is a nonexistent package that fails 100% of the time. Teams chase the wrong thing.
- **Silent hallucination is the real harm.** In a research-grade pipeline, output that *looks* KG-grounded but isn't flows into synthesis and erodes trust far more than an honest error. The guard converts a silent correctness risk into a visible, contained one.
- **Instrument at the shared layer.** Putting the tool-call/availability diagnostics in the shared `query_with_usage` makes the *same* latent failure observable across all SDK nodes for free — the logs tell you which other nodes are silently degraded before you migrate them.

## When to Apply

- Any Claude Agent SDK call that sets `mcp_servers=...` and lists `mcp__*` entries in `allowed_tools` — verify the MCP server actually launches in the deploy environment (run `uvx --from <pkg> ... --help` / check it exists).
- Any node whose authoritative output comes from the SDK tier rather than an HTTP/deterministic tier.
- When reusing the structural guard on a node whose prompt does **not** mandate tool use, require the init-list signal or the phrase — never the bare zero-calls count (it false-positives on legitimate no-tool runs).

## Examples

**Before — broken stdio MCP (model hallucinates when the server can't launch):**
```python
options = ClaudeAgentOptions(
    system_prompt=PATHWAY_ENRICHMENT_PROMPT,
    allowed_tools=["mcp__kestrel__one_hop_query", "mcp__kestrel__get_nodes"],
    mcp_servers={"kestrel": get_kestrel_mcp_config()},  # uvx mcp-client-kestrel — doesn't exist
    max_turns=25,
)
result_text, _ = await query_with_usage(prompt=user_prompt, options=options, node_name="pathway_enrichment")
```

**After — HTTP data-in-prompt (mirrors `cold_start.get_entity_connections`):**
```python
per_entity, no_data = await prefetch_one_hop_neighbors(selected_entities)  # HTTP call_kestrel_tool
if no_data:                                                                 # <2 entities returned real data
    return _degraded_phase_b_result(two_hop_findings, errors, None, "prefetch_no_data")

options = ClaudeAgentOptions(
    system_prompt=PATHWAY_INFERENCE_PROMPT,
    allowed_tools=[],                # no MCP tools — reason over the embedded data
    max_turns=2,
)
async with SDK_SEMAPHORE:
    result_text, usage = await query_with_usage(prompt=_build_inference_user_prompt(selected_entities, per_entity),
                                                options=options, node_name="pathway_enrichment")
```

**The structural guard (shared, in `sdk_utils.py`):**
```python
def classify_mcp_degradation(expected_tools, mcp_tool_calls, result_text="", available_tools=None):
    if not expected_tools:
        return McpDegradationVerdict(False, "no_tools_expected", "none")
    phrase_hit = bool(result_text) and any(p in result_text.lower() for p in _MCP_FALLBACK_PHRASES)
    if available_tools is not None:
        missing = [t for t in expected_tools if t not in available_tools]
        if missing:
            return McpDegradationVerdict(True, f"tools_missing_from_init:{','.join(missing)}", "definitive")
        if mcp_tool_calls == 0 and not phrase_hit:        # tools registered, model just didn't call → not degraded
            return McpDegradationVerdict(False, "tools_registered_zero_calls", "none")
    if mcp_tool_calls == 0:
        return McpDegradationVerdict(True, "zero_mcp_tool_calls" + ("+phrase" if phrase_hit else ""),
                                     "high" if phrase_hit else "structural")
    return McpDegradationVerdict(False, "tools_used", "none")
```

**Prevention tests** worth pinning (from `test_sdk_utils.py` / `test_pathway_enrichment.py`):
- zero calls + phrase → degraded `high`; zero calls + no phrase → `structural`; tools used → not degraded.
- `expected_tools == []` never degrades; init-list confirms-all + zero calls + no phrase → not degraded.
- a Phase B exception/timeout still preserves the HTTP findings AND sets `degraded=True`.

## Related

- Issues: #44 (root issue), #61 (follow-up: migrate the other five SDK nodes); PR #60 (this fix), PR #24 (cold_start precedent).
- `backend/src/kestrel_backend/graph/nodes/cold_start.py:1-18` — the original "package doesn't exist" record + the HTTP-not-MCP rationale.
- `docs/plans/2026-06-01-001-fix-pathway-enrichment-mcp-degradation-plan.md` — full plan + 3-round review.
- [[discovery-pipeline-one-graph-methods-within-nodes-2026-05-28]] — related pipeline-architecture learning.
- [[pytest-venv-path-spaces-module-invocation-2026-06-01]] — run these tests with `uv run python -m pytest`.
