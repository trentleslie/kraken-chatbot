"""Iterate-loop arm: the structured-spec executor (the treatment).

Per item, the LLM emits a typed JSON query spec; the harness validates it against a
verb whitelist, dispatches to Kestrel REST, accumulates discovered paths, and feeds
results back for up to `turn_cap` self-correction turns. No temperature control exists
(plan finding), so each item is run K times and scored by majority. Grounding (R9):
any CURIE the LLM emits that was not returned by an executed call (nor an equivalent_id
of one) is a hard-fail violation. The LLM never sees the gold bridge — the harness
scores recovery, exactly like the baseline.

The LLM call is injected (`llm_fn`) so tests run without SDK cost.
"""
from __future__ import annotations

import json
import re
from typing import Awaitable, Callable

from .config import CONFIG
from .kestrel_rest import KestrelREST, parse_paths, any_path_recovers, is_grounded

# spec verbs the executor accepts (subset of the kestrel_tools whitelist, mapped to REST)
VERB_WHITELIST = {"multi_hop", "one_hop", "hybrid_search"}

LlmFn = Callable[[str, str], Awaitable[tuple[str, object]]]  # (prompt, system) -> (text, usage)

SYSTEM_PROMPT = (
    "You are finding a mechanistic path between two biomedical knowledge-graph nodes. "
    "Each turn, emit ONE JSON object and nothing else. To query, emit "
    '{\"action\":\"query\",\"verb\":\"multi_hop\",\"start_node_ids\":[\"<CURIE>\"],'
    '\"end_node_ids\":[\"<CURIE>\"],\"max_path_length\":<2-5>,\"predicate\":\"<optional>\"}. '
    "Verbs: multi_hop (paths between nodes), one_hop (neighbors of a node), hybrid_search "
    "(resolve a name to a CURIE; use \"search_text\"). Only use CURIEs that appeared in "
    "prior results or in the task. When you believe you have found the connecting path(s), "
    'emit {\"action\":\"done\"}. Do not invent CURIEs.'
)


def _extract_spec(text: str) -> dict | None:
    """Pull the first JSON object out of the LLM text (tolerates surrounding prose)."""
    for match in re.finditer(r"\{.*?\}", text, re.DOTALL):
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict) and "action" in obj:
                return obj
        except Exception:
            continue
    return None


def _spec_curies(spec: dict) -> list[str]:
    out: list[str] = []
    for key in ("start_node_ids", "end_node_ids"):
        v = spec.get(key)
        if isinstance(v, list):
            out.extend(c for c in v if isinstance(c, str))
        elif isinstance(v, str):
            out.append(v)
    return out


async def _dispatch(rest: KestrelREST, spec: dict) -> list[list[str]]:
    """Run a validated spec against REST; return discovered paths (as node-CURIE lists)."""
    verb = spec.get("verb")
    if verb == "multi_hop":
        data = await rest.multi_hop(
            spec.get("start_node_ids") or [], spec.get("end_node_ids"),
            max_path_length=min(int(spec.get("max_path_length", CONFIG.max_path_length)), CONFIG.max_path_length),
            limit=CONFIG.multi_hop_limit, mode="full")
        return parse_paths(data)
    if verb == "one_hop":
        starts = spec.get("start_node_ids") or []
        start = starts[0] if isinstance(starts, list) and starts else starts
        data = await rest._post("/one-hop", {"start_node_ids": start, "mode": "full", "limit": CONFIG.multi_hop_limit})
        # represent neighbors as 2-node "paths" [start, neighbor] for uniform handling
        nbrs = []
        results = data.get("results", []) if isinstance(data, dict) else []
        for r in results:
            nid = r.get("end_node_id") or r.get("id")
            if nid:
                nbrs.append([start, nid])
        return nbrs
    if verb == "hybrid_search":
        hits = await rest.hybrid_search(spec.get("search_text", ""), limit=3)
        return [[h["id"]] for h in hits if h.get("id")]
    return []


async def default_llm_fn(prompt: str, system: str) -> tuple[str, object]:
    """Real LLM turn via the Claude Agent SDK (allowed_tools=[] — data-in-prompt).
    Imported lazily so the harness/tests don't require the SDK unless run live."""
    from kestrel_backend.graph.sdk_utils import query_with_usage
    from claude_agent_sdk import ClaudeAgentOptions
    opts = ClaudeAgentOptions(system_prompt=system, allowed_tools=[], max_turns=1,
                              permission_mode="bypassPermissions")
    return await query_with_usage(prompt, opts, node_name="cog_spike")


async def run_iterate_loop(rest: KestrelREST, item: dict, llm_fn: LlmFn) -> dict:
    start, target = item["start_curie"], item["gold_target_curie"]
    gold = item["gold_bridge_curies"]
    rec: dict = {"trial_id": item["trial_id"], "method": "iterate", "stratum": item.get("stratum"),
                 "turns": 0, "llm_calls": 0, "grounding_violations": 0}
    returned_curies: set[str] = {start, target}
    accumulated: list[list[str]] = []
    transcript: list[dict] = []
    prompt = f"Task: find the path(s) connecting start={start} to end={target}."

    for turn in range(CONFIG.turn_cap):
        rec["turns"] = turn + 1
        text, _usage = await llm_fn(prompt, SYSTEM_PROMPT)
        rec["llm_calls"] += 1
        spec = _extract_spec(text)
        if spec is None:
            prompt = "Your last message was not a single valid JSON object. Emit one now."
            continue
        if spec.get("action") == "done":
            rec["terminal_state"] = "found"
            break
        if spec.get("verb") not in VERB_WHITELIST:
            prompt = f"verb must be one of {sorted(VERB_WHITELIST)}. Retry."
            continue
        # grounding (R9): every emitted CURIE must be grounded in prior results / the task
        for curie in _spec_curies(spec):
            if not await is_grounded(rest, curie, returned_curies):
                rec["grounding_violations"] += 1
        try:
            paths = await _dispatch(rest, spec)
        except Exception as exc:
            rec.setdefault("errors", []).append(str(exc))
            prompt = "That query errored (transport). Try a different spec."
            continue
        for p in paths:
            if p not in accumulated and len(accumulated) < CONFIG.aggregate_path_budget:
                accumulated.append(p)
                returned_curies.update(p)
        transcript.append({"turn": turn + 1, "spec": spec, "n_paths": len(paths)})
        if not paths:
            prompt = f"No results for that spec. {len(accumulated)} paths found so far. Try a broader/different spec or emit {{\"action\":\"done\"}}."
        else:
            prompt = (f"Got {len(paths)} paths (total {len(accumulated)}). Nodes seen: "
                      f"{sorted(returned_curies)[:20]}. Refine, or emit {{\"action\":\"done\"}} if the connector is found.")
    else:
        rec["terminal_state"] = "turn-cap-hit"

    rec.update(
        hit=any_path_recovers(accumulated, gold),
        n_paths=len(accumulated),
        intermediates=sorted({c for p in accumulated for c in p if c not in (start, target)}),
        kestrel_calls=rest.kestrel_calls,
        transcript=transcript,
    )
    return rec


async def run_iterate_item_k(rest: KestrelREST, item: dict, llm_fn: LlmFn, k: int | None = None) -> dict:
    """Run the loop K times; per-item hit = majority. Reports the variance band."""
    k = k or CONFIG.k_reruns
    runs = [await run_iterate_loop(rest, item, llm_fn) for _ in range(k)]
    hits = [r["hit"] for r in runs]
    majority = sum(hits) >= (len(hits) + 1) // 2
    return {
        "trial_id": item["trial_id"], "method": "iterate", "stratum": item.get("stratum"),
        "hit": majority,
        "hit_runs": hits,
        "variance": "stable" if len(set(hits)) == 1 else "flapping",
        "grounding_violations": sum(r["grounding_violations"] for r in runs),
        "llm_calls": sum(r["llm_calls"] for r in runs),
        "runs": runs,
    }
