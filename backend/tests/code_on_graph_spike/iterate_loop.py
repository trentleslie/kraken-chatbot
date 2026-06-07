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

import asyncio
import json
import re
from typing import Awaitable, Callable

from .config import CONFIG, primary_hit
from .kestrel_rest import (
    KestrelREST, any_path_recovers, is_grounded, parse_paths, recovers_any_interior,
)

# spec verbs the executor accepts (subset of the kestrel_tools whitelist, mapped to REST)
VERB_WHITELIST = {"multi_hop", "one_hop", "hybrid_search"}

LlmFn = Callable[[str, str], Awaitable[tuple[str, object]]]  # (prompt, system) -> (text, usage)

SYSTEM_PROMPT = (
    "You are a biomedical knowledge-graph explorer. The START and END node CURIEs are "
    "GIVEN to you in the task — never ask for them. Your goal is to discover INTERMEDIATE "
    "bridge nodes (genes, proteins, pathways) that mechanistically connect START to END, "
    "especially ones the initial broad query did not surface.\n"
    "Each turn, reply with EXACTLY ONE JSON object and nothing else:\n"
    '  query: {\"action\":\"query\",\"verb\":\"multi_hop\",\"start_node_ids\":[\"<CURIE>\"],'
    '\"end_node_ids\":[\"<CURIE>\"],\"max_path_length\":<2-5>,\"predicate\":\"<optional biolink predicate>\"}\n'
    '  expand a node’s neighbors: {\"action\":\"query\",\"verb\":\"one_hop\",\"start_node_ids\":[\"<CURIE>\"]}\n'
    '  finish: {\"action\":\"done\"}\n'
    "Refinement ideas: add a predicate filter; raise max_path_length; or one_hop-expand a "
    "promising intermediate then multi_hop from it to END. Only use CURIEs that appeared in "
    "the task or in prior results (never invent CURIEs). Do NOT emit 'done' until you have run "
    "at least one refining query beyond the initial results."
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


SDK_RETRIES = 3       # transient SDK/CLI hiccups (e.g. "Control request timeout: initialize")
SDK_RETRY_BACKOFF = 3.0


async def default_llm_fn(prompt: str, system: str) -> tuple[str, object]:
    """Real LLM turn via the Claude Agent SDK (allowed_tools=[] — data-in-prompt).
    Imported lazily so the harness/tests don't require the SDK unless run live.

    Retries transient SDK failures: the SDK spawns the `claude` CLI as a subprocess
    and its control-protocol handshake can intermittently time out under repeated
    invocation. A single such blip must not crash a multi-hour gate run — retry it
    (a persistent failure still raises, so a real outage fails loud rather than
    silently fabricating misses)."""
    from kestrel_backend.graph.sdk_utils import query_with_usage
    from claude_agent_sdk import ClaudeAgentOptions
    opts = ClaudeAgentOptions(system_prompt=system, allowed_tools=[], max_turns=1,
                              permission_mode="bypassPermissions")
    last_exc: Exception | None = None
    for attempt in range(SDK_RETRIES + 1):
        try:
            return await query_with_usage(prompt, opts, node_name="cog_spike")
        except Exception as exc:  # noqa: BLE001 — SDK raises varied transient errors
            last_exc = exc
            if attempt < SDK_RETRIES:
                await asyncio.sleep(SDK_RETRY_BACKOFF * (attempt + 1))
    assert last_exc is not None
    raise last_exc


def _intermediates_of(accumulated: list[list[str]], start: str, target: str) -> list[str]:
    return sorted({c for p in accumulated for c in p if c not in (start, target)})


def _turn_prompt(start: str, target: str, accumulated: list[list[str]], note: str) -> str:
    inter = _intermediates_of(accumulated, start, target)
    return (
        f"START = {start}\nEND = {target}\n"
        f"Paths found so far: {len(accumulated)} (through {len(inter)} intermediate nodes).\n"
        f"Intermediate nodes seen: {inter[:25]}\n"
        f"{note}\n"
        "Emit ONE JSON object: a refining query, or {\"action\":\"done\"} only if you have already "
        "run at least one refining query."
    )


async def run_iterate_loop(rest: KestrelREST, item: dict, llm_fn: LlmFn) -> dict:
    start, target = item["start_curie"], item["gold_target_curie"]
    gold = item["gold_bridge_curies"]
    rec: dict = {"trial_id": item["trial_id"], "method": "iterate", "stratum": item.get("stratum"),
                 "turns": 0, "llm_calls": 0, "grounding_violations": 0}
    returned_curies: set[str] = {start, target}
    accumulated: list[list[str]] = []
    # Paths returned by GROUNDED queries only (seed + specs with no grounding violation).
    # Used to detect finding-level hallucination: a win that exists only via ungrounded queries.
    grounded_accumulated: list[list[str]] = []
    transcript: list[dict] = []

    # Turn 0 (automatic): seed with the baseline query so iterate >= baseline by construction;
    # the loop can only ADD paths via refinement (isolates iteration's marginal value).
    try:
        seed = await _dispatch(rest, {"verb": "multi_hop", "start_node_ids": [start],
                                      "end_node_ids": [target], "max_path_length": CONFIG.max_path_length})
        for p in seed:
            if p not in accumulated and len(accumulated) < CONFIG.aggregate_path_budget:
                accumulated.append(p)
                grounded_accumulated.append(p)  # the seed queries the given start/target — grounded
                returned_curies.update(p)
        transcript.append({"turn": 0, "spec": "seed:multi_hop(start,end,depth=max)", "n_paths": len(seed)})
    except Exception as exc:
        rec.setdefault("errors", []).append(f"seed: {exc}")

    queried = False
    prompt = _turn_prompt(start, target, accumulated,
                          "This is the result of a broad initial query. Refine to surface ADDITIONAL "
                          "bridge nodes that connect START to END (predicate filter, deeper hops, or "
                          "one_hop-expand a promising intermediate then multi_hop to END).")

    for turn in range(CONFIG.turn_cap):
        rec["turns"] = turn + 1
        text, _usage = await llm_fn(prompt, SYSTEM_PROMPT)
        rec["llm_calls"] += 1
        spec = _extract_spec(text)
        if spec is None:
            prompt = "Your last message was not a single valid JSON object. Emit exactly one now."
            continue
        if spec.get("action") == "done":
            if not queried:  # block premature done before any refining query
                prompt = _turn_prompt(start, target, accumulated,
                                      "Run at least one REFINING query before finishing.")
                continue
            rec["terminal_state"] = "found"
            break
        if spec.get("verb") not in VERB_WHITELIST:
            prompt = f"verb must be one of {sorted(VERB_WHITELIST)}. Retry with one JSON object."
            continue
        spec_grounded = True
        for curie in _spec_curies(spec):  # grounding (R9)
            if not await is_grounded(rest, curie, returned_curies):
                rec["grounding_violations"] += 1
                spec_grounded = False  # this query referenced an ungrounded CURIE
        try:
            paths = await _dispatch(rest, spec)
        except Exception as exc:
            rec.setdefault("errors", []).append(str(exc))
            prompt = _turn_prompt(start, target, accumulated, "That query errored. Try a different spec.")
            continue
        queried = True
        added = 0
        for p in paths:
            if p not in accumulated and len(accumulated) < CONFIG.aggregate_path_budget:
                accumulated.append(p)
                if spec_grounded:
                    grounded_accumulated.append(p)
                returned_curies.update(p)
                added += 1
        transcript.append({"turn": turn + 1, "spec": spec, "n_paths": len(paths), "added": added})
        prompt = _turn_prompt(start, target, accumulated,
                              f"That query added {added} new paths. Refine further or finish.")
    else:
        rec["terminal_state"] = "turn-cap-hit"

    hit_strict = any_path_recovers(accumulated, gold)
    hit_any = recovers_any_interior(accumulated, gold)
    # Finding-level grounding (R9, corrected): does the win survive if we DROP paths from
    # ungrounded queries? If the hit exists only via an ungrounded query, it is a
    # finding-level hallucination (hard-fail). Query-arg leakage that did NOT drive the win
    # stays in grounding_violations (a reported caveat, not a kill).
    hit_full = primary_hit(hit_strict, hit_any)
    hit_grounded = primary_hit(any_path_recovers(grounded_accumulated, gold),
                               recovers_any_interior(grounded_accumulated, gold))
    rec.update(
        hit=hit_full,  # mirror of the configured primary bridge unit
        hit_strict=hit_strict,
        hit_any=hit_any,
        finding_level_hallucination=int(hit_full and not hit_grounded),
        n_paths=len(accumulated),
        intermediates=_intermediates_of(accumulated, start, target),
        kestrel_calls=rest.kestrel_calls,
        transcript=transcript,
    )
    return rec


def _majority(vals: list[bool]) -> bool:
    return sum(vals) >= (len(vals) + 1) // 2


async def run_iterate_item_k(rest: KestrelREST, item: dict, llm_fn: LlmFn, k: int | None = None) -> dict:
    """Run the loop K times; per-item hit = majority, computed for BOTH bridge units.
    Variance (the gate's flap signal) is reported on the configured PRIMARY metric."""
    k = k or CONFIG.k_reruns
    runs = [await run_iterate_loop(rest, item, llm_fn) for _ in range(k)]
    strict_runs = [r["hit_strict"] for r in runs]
    any_runs = [r["hit_any"] for r in runs]
    hit_strict, hit_any = _majority(strict_runs), _majority(any_runs)
    primary_runs = any_runs if CONFIG.primary_bridge_unit == "any_one" else strict_runs
    return {
        "trial_id": item["trial_id"], "method": "iterate", "stratum": item.get("stratum"),
        "hit": primary_hit(hit_strict, hit_any),  # primary mirror (drives the verdict)
        "hit_strict": hit_strict,
        "hit_any": hit_any,
        "hit_runs": primary_runs,  # back-compat: runs of the primary metric
        "hit_strict_runs": strict_runs,
        "hit_any_runs": any_runs,
        "variance": "stable" if len(set(primary_runs)) == 1 else "flapping",
        "grounding_violations": sum(r["grounding_violations"] for r in runs),  # query-arg leakage (caveat)
        "finding_level_hallucinations": sum(r.get("finding_level_hallucination", 0) for r in runs),
        "llm_calls": sum(r["llm_calls"] for r in runs),
        "runs": runs,
    }
