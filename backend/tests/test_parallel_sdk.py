"""Test parallel query() calls to verify LangGraph feasibility."""

import asyncio
import time
from typing import Any

import pytest

from claude_agent_sdk import query, ClaudeAgentOptions
from claude_agent_sdk.types import McpSSEServerConfig


# Kestrel MCP server config (SSE-based, no subprocess overhead)
KESTREL_SSE_CONFIG: McpSSEServerConfig = {
    "type": "sse",
    "url": "https://kestrel.nathanpricelab.com/mcp",
}

# Minimal system prompt for fast resolution
LOOKUP_SYSTEM_PROMPT = """You are a knowledge graph lookup tool.
When asked about an entity, use hybrid_search to find it.
Return ONLY a JSON object: {"curie": "...", "name": "...", "category": "..."}
Be extremely concise. No explanations."""

# Minimal allowed tools for each test scenario
HYBRID_SEARCH_TOOLS = ["mcp__kestrel__hybrid_search"]
TEXT_SEARCH_TOOLS = ["mcp__kestrel__text_search"]
ONE_HOP_TOOLS = ["mcp__kestrel__one_hop_query", "mcp__kestrel__get_nodes"]


async def resolve_entity(entity_name: str, tools: list[str] | None = None) -> dict[str, Any]:
    """Run a single query() call to resolve an entity.

    Returns dict with:
        - result: collected text output
        - elapsed_ms: wall-clock time
        - error: error message if failed
    """
    if tools is None:
        tools = HYBRID_SEARCH_TOOLS

    options = ClaudeAgentOptions(
        system_prompt=LOOKUP_SYSTEM_PROMPT,
        allowed_tools=tools,
        mcp_servers={"kestrel": KESTREL_SSE_CONFIG},
        max_turns=2,  # Limit to avoid runaway
        permission_mode="bypassPermissions",  # No permission prompts
    )

    prompt = f"Find '{entity_name}' in the knowledge graph using the available search tool. Return JSON only."

    result_parts = []
    start = time.perf_counter()
    error = None

    try:
        async for event in query(prompt=prompt, options=options):
            # Collect text content from AssistantMessage
            if hasattr(event, 'content') and event.content:
                for block in event.content:
                    if hasattr(block, 'text'):
                        result_parts.append(block.text)
    except Exception as e:
        error = str(e)

    elapsed_ms = (time.perf_counter() - start) * 1000

    return {
        "entity": entity_name,
        "result": "\n".join(result_parts) if result_parts else None,
        "elapsed_ms": elapsed_ms,
        "error": error,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Sequential baseline
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sequential_baseline():
    """Run 3 query() calls sequentially. Establish baseline timing."""
    entities = ["glucose", "NLGN1", "baricitinib"]

    start = time.perf_counter()
    results = []
    for entity in entities:
        result = await resolve_entity(entity)
        results.append(result)
    total_ms = (time.perf_counter() - start) * 1000

    # Verify all succeeded
    for r in results:
        assert r["error"] is None, f"Entity {r['entity']} failed: {r['error']}"
        assert r["result"] is not None, f"Entity {r['entity']} returned no result"

    print(f"\n=== Sequential Baseline ===")
    for r in results:
        print(f"  {r['entity']}: {r['elapsed_ms']:.0f}ms")
    print(f"  TOTAL: {total_ms:.0f}ms")

    return {"results": results, "total_ms": total_ms}


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Parallel with asyncio.gather (same 3 entities)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_parallel_three_entities():
    """Run same 3 query() calls concurrently with asyncio.gather()."""
    entities = ["glucose", "NLGN1", "baricitinib"]

    start = time.perf_counter()
    results = await asyncio.gather(*[resolve_entity(e) for e in entities])
    total_ms = (time.perf_counter() - start) * 1000

    # Verify all succeeded
    for r in results:
        assert r["error"] is None, f"Entity {r['entity']} failed: {r['error']}"
        assert r["result"] is not None, f"Entity {r['entity']} returned no result"

    print(f"\n=== Parallel (3 entities) ===")
    for r in results:
        print(f"  {r['entity']}: {r['elapsed_ms']:.0f}ms")
    print(f"  TOTAL: {total_ms:.0f}ms")

    return {"results": results, "total_ms": total_ms}


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Parallel at scale (6 entities)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_parallel_six_entities():
    """Run 6 concurrent query() calls. Check for deadlocks, resource exhaustion."""
    entities = ["glucose", "fructose", "mannose", "KIF6", "NLGN1", "ADGRG1"]

    start = time.perf_counter()
    results = await asyncio.gather(*[resolve_entity(e) for e in entities], return_exceptions=True)
    total_ms = (time.perf_counter() - start) * 1000

    # Check for exceptions in results
    errors = []
    successes = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            errors.append({"entity": entities[i], "error": str(r)})
        elif r.get("error"):
            errors.append(r)
        else:
            successes.append(r)

    print(f"\n=== Parallel at Scale (6 entities) ===")
    print(f"  Successes: {len(successes)}/{len(entities)}")
    print(f"  Errors: {len(errors)}")
    for s in successes:
        print(f"  ✓ {s['entity']}: {s['elapsed_ms']:.0f}ms")
    for e in errors:
        print(f"  ✗ {e['entity']}: {e.get('error', 'unknown')}")
    print(f"  TOTAL: {total_ms:.0f}ms")

    # Allow partial success but report
    assert len(successes) >= 4, f"Too many failures: {errors}"

    return {"successes": successes, "errors": errors, "total_ms": total_ms}


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Parallel with different tool subsets
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_parallel_different_tools():
    """Run 3 concurrent calls with different allowed_tools. Verify no interference."""
    # Each call uses a different tool
    tasks = [
        resolve_entity("glucose", tools=HYBRID_SEARCH_TOOLS),
        resolve_entity("NLGN1", tools=TEXT_SEARCH_TOOLS),
        resolve_entity("diabetes", tools=ONE_HOP_TOOLS),  # one_hop needs a node ID, may fail gracefully
    ]

    start = time.perf_counter()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total_ms = (time.perf_counter() - start) * 1000

    print(f"\n=== Parallel with Different Tools ===")
    tool_labels = ["hybrid_search", "text_search", "one_hop_query"]
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"  {tool_labels[i]}: EXCEPTION - {r}")
        elif r.get("error"):
            print(f"  {tool_labels[i]}: ERROR - {r['error']}")
        else:
            print(f"  ✓ {tool_labels[i]} ({r['entity']}): {r['elapsed_ms']:.0f}ms")
    print(f"  TOTAL: {total_ms:.0f}ms")

    # hybrid_search and text_search should definitely work
    assert not isinstance(results[0], Exception) and results[0].get("result"), "hybrid_search failed"
    assert not isinstance(results[1], Exception) and results[1].get("result"), "text_search failed"

    return {"results": results, "total_ms": total_ms}


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint for direct execution
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    """Run all tests and produce markdown report."""
    print("=" * 60)
    print("PARALLEL QUERY() FEASIBILITY TEST")
    print("=" * 60)

    # Run tests
    seq = await test_sequential_baseline()
    par3 = await test_parallel_three_entities()
    par6 = await test_parallel_six_entities()
    par_tools = await test_parallel_different_tools()

    # Calculate speedup
    speedup = seq["total_ms"] / par3["total_ms"] if par3["total_ms"] > 0 else 0

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Sequential (3 entities):    {seq['total_ms']:.0f}ms")
    print(f"Parallel (3 entities):      {par3['total_ms']:.0f}ms")
    print(f"Speedup:                    {speedup:.2f}x")
    print(f"Parallel (6 entities):      {par6['total_ms']:.0f}ms ({len(par6['successes'])}/{len(par6['successes'])+len(par6['errors'])} success)")
    print(f"Parallel (diff tools):      {par_tools['total_ms']:.0f}ms")

    # Recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    if speedup > 1.5 and len(par6["errors"]) == 0:
        print("✓ Parallel query() is VIABLE for LangGraph. Good speedup, no errors.")
    elif speedup > 1.0 and len(par6["errors"]) <= 1:
        print("⚠ Parallel query() is CONDITIONALLY VIABLE. Some speedup, minor errors.")
    else:
        print("✗ Parallel query() NOT recommended. Speedup insufficient or too many errors.")


if __name__ == "__main__":
    asyncio.run(main())
