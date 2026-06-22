---
title: Accumulating LangGraph stream_mode="updates" deltas requires replicating the state reducers
date: 2026-06-22
category: best-practices
module: kraken-chatbot
problem_type: best_practice
component: service_object
severity: high
applies_when:
  - Consuming LangGraph stream_mode="updates" and manually rebuilding accumulated state from per-node deltas (WebSocket accumulator, eval harness, observability hook)
  - Adding a new DiscoveryState field whose reducer is not operator.add (e.g. a custom dict-merge)
tags: [langgraph, streaming, state-accumulation, reducer, node-timings, eval-harness, websocket]
---

# Accumulating LangGraph stream_mode="updates" deltas requires replicating the state reducers

## Context

LangGraph's `stream_mode="updates"` yields per-node **deltas** — the raw return value of each node function — not the post-reducer merged channel state. Any consumer that reads this stream and rebuilds accumulated state by hand (the WebSocket accumulator in `main.py`, an eval harness, an observability hook) is responsible for replicating every field's reducer. The graph state schema (`DiscoveryState` in `graph/state.py`) declares reducers as `Annotated[T, reducer_fn]`; two reducer types appear in this codebase — `operator.add` (list concatenation) and a custom dict-merge (`merge_node_timings`). A consumer that handles only one reducer type silently discards data for every field that uses the other, with no error and no missing key — only wrong values.

## Guidance

When accumulating `stream_mode="updates"` deltas, derive the merge behavior from each field's declared reducer rather than from a single hardcoded strategy. `DiscoveryState` fields fall into three categories:

| Category | Annotation | Correct accumulation |
|---|---|---|
| List-append | `Annotated[list[X], operator.add]` | concatenate: `existing + value` |
| Dict-merge | `Annotated[dict[K,V], custom_fn]` | merge: `{**existing, **value}` |
| Last-write-wins | no `Annotated` reducer | overwrite: `acc[k] = value` |

The two broken strategies and the fix:

```python
# BROKEN — plain dict.update() treats every field as last-write-wins:
for node_name, node_output in event.items():
    merged.update(node_output)           # operator.add lists overwritten, not appended

# STILL BROKEN — list-only concat keyed on operator.add detection:
#   _get_concat_fields() matches ONLY operator.add, so a custom dict reducer
#   (node_timings) falls through to last-write-wins and collapses each yield.
if isinstance(existing, list) and isinstance(value, list) and key in CONCAT_LIST_FIELDS:
    acc[key] = existing + value
else:
    acc[key] = value                     # node_timings overwritten every yield

# CORRECT — an explicit branch for each non-operator.add reducer, BEFORE the list check:
if key == "node_timings" and isinstance(value, dict):
    base = dict(existing) if isinstance(existing, dict) else {}
    base.update(value)
    acc[key] = base
elif isinstance(existing, list) and isinstance(value, list) and key in CONCAT_LIST_FIELDS:
    acc[key] = existing + value
else:
    acc[key] = value
```

**Better still — avoid hand-accumulation entirely.** If you do not need to stream intermediate updates, read the graph's already-reducer-merged final state instead:

```python
result = await graph.ainvoke(initial_state, config=config)   # result is post-reducer, correct
```

Or have a **terminal graph node** read the accumulated state: by the time it executes, LangGraph has applied every reducer correctly. The per-node performance report in this repo is built exactly this way — a terminal `reporting` node reads true graph state — which is why it is the trustworthy record (see the second incident below).

## Why This Matters

The failure mode is silent — wrong values, no exception — and bit this codebase twice in one session.

**Incident 1 — `node_timings` collapsed in Langfuse (`main.py`).** `direct_kg` and `cold_start` execute in the same superstep and both write `node_timings`; the graph merges them with `merge_node_timings`. But `_get_concat_fields()` detects only `operator.add` reducers, so `node_timings` was never in `CONCAT_LIST_FIELDS` and the accumulator fell through to last-write-wins, recording only the last parallel branch to complete. The Langfuse trace showed single-node timings with no error. (Note: this exact regression surfaced when `node_timings` was *changed* to a reducer field, which was itself required because LangGraph raises `InvalidUpdateError` on concurrent writes to a non-reducer field. — auto memory [claude])

**Incident 2 — eval harness undercounted findings (Brown C1 harness).** The harness accumulated the stream with `merged.update(out)`, so `direct_findings` (an `operator.add` list) was overwritten per yield instead of appended. The harness reported `errors=0, direct_findings=37`; the true reducer-merged graph state — read by the terminal `reporting` node running inside the graph — held `1 error, 1,715 direct findings`. The ~46× undercount and the masked error were invisible without the in-graph reference.

In both cases the root fact is the same: `stream_mode="updates"` gives you the delta, not the channel value. The graph applies reducers; your accumulator does not, unless you write it to.

## When to Apply

- Any code iterating `graph.astream(..., stream_mode="updates")` and building an accumulated dict from the yielded deltas.
- Eval harnesses, test scripts, or observability hooks that read pipeline output from a hand-accumulated dict rather than from `graph.ainvoke()` or a terminal node.
- Whenever a new `DiscoveryState` field is added with a custom (non-`operator.add`) reducer: every hand-accumulator (notably `main.py`) needs a matching explicit branch.
- When debugging "missing" or "too few" results from accumulated pipeline state: check whether the field has a reducer in `state.py` that the accumulator fails to replicate.

Do **not** apply the explicit-branch merge to `graph.ainvoke()` results — those are already post-reducer and correct.

## Examples

`_get_concat_fields()` introspects the schema to find the `operator.add` fields — and, by construction, **excludes** custom-reducer fields, which is the trap:

```python
def _get_concat_fields() -> set[str]:          # main.py
    hints = get_type_hints(DiscoveryState, include_extras=True)
    fields = set()
    for name, hint in hints.items():
        for arg in get_args(hint):
            if arg is operator.add:            # only operator.add — NOT merge_node_timings
                fields.add(name); break
    return fields
# includes: direct_findings, cold_start_findings, model_usages, errors, ...
# excludes: node_timings (custom dict reducer), bridges / hypotheses (no reducer)
```

The corresponding schema declaration that the accumulator must mirror:

```python
# graph/state.py
def merge_node_timings(left, right):           # custom reducer — NOT operator.add
    merged = {}
    if left: merged.update(left)
    if right: merged.update(right)
    return merged

class DiscoveryState(TypedDict, total=False):
    direct_findings: Annotated[list[Finding], operator.add]        # list-concat branch
    node_timings: Annotated[dict[str, float], merge_node_timings]  # needs explicit dict branch
```

## Related

- [Discovery pipeline: one graph, methods within nodes](discovery-pipeline-one-graph-methods-within-nodes-2026-05-28.md) — the *producer*-side rule (which fields get `operator.add` and why concurrent-write fields need a reducer). This doc is the *consumer*-side pitfall when reading those fields back from `stream_mode="updates"`.
- [LangGraph pipeline production formalization](langgraph-pipeline-production-formalization.md) — per-node state contracts / validation context.
- Source: `backend/src/kestrel_backend/main.py` (`_get_concat_fields` / `CONCAT_LIST_FIELDS`; the `stream_mode="updates"` accumulation loop in `handle_pipeline_mode`), `backend/src/kestrel_backend/graph/state.py` (`merge_node_timings`, `node_timings`), `backend/src/kestrel_backend/graph/runner.py` (`stream_discovery`), `backend/src/kestrel_backend/graph/nodes/reporting.py` (terminal node reading true state). Introduced with the per-node performance report (PR #84).
