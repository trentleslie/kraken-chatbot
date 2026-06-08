---
title: "Reliability patterns for long, expensive, flaky LLM + live-API batch runs"
date: 2026-06-07
category: docs/solutions/best-practices/
module: code_on_graph_spike
problem_type: best_practice
component: background_job
severity: high
applies_when:
  - "running a long (minutes-to-hours) eval/batch harness over a live API or LLM"
  - "aggregating fan-out results with asyncio.gather(..., return_exceptions=True) then filtering survivors"
  - "calling the Claude Agent SDK (spawns the claude CLI subprocess; flaky initialize handshake)"
  - "persisting expensive run results only at the end of the run"
  - "summing a cumulative counter from a client reused across loop iterations"
related_components:
  - testing_framework
  - tooling
tags:
  - asyncio
  - fail-loud
  - checkpointing
  - retry-backoff
  - claude-agent-sdk
  - httpx
  - eval-harness
  - cost-accounting
---

# Reliability patterns for long, expensive, flaky LLM + live-API batch runs

## Context

Long-running batch jobs that fan out over live external APIs and LLMs — recall-gate evaluations, dataset builds, eval harnesses — share a hostile failure profile: they run for hours of wall-clock, every call costs money, and the upstream services (HTTP endpoints, the Claude Agent SDK's CLI subprocess) fail *intermittently* rather than cleanly. A naive harness conflates two fundamentally different outcomes — "the API returned a legitimate empty/negative result" versus "the transport hiccuped" — and treats both as a quiet "no." Over an N-item × K-rerun run this silently biases the aggregate (a flaky timeout becomes a fabricated "miss"), shrinks datasets without warning, and discards completed work when the process dies on item 87 of 90.

These three reliability properties turn such a harness from "produces a number you can't trust" into "produces a number you can defend, and survives a crash." Each was extracted from a concrete, costly bug in the `code_on_graph_spike` Phase-0 gate (`backend/tests/code_on_graph_spike/`): a gold-set rebuild that silently collapsed ~70 → 40 survivors; an ~11-minute live run lost when a transient SDK error crashed it with no saved artifact; and a cost metric reported at ~150× reality.

## Guidance

### 1. Fail loud, never silently shrink

Classify every fan-out failure as *transient transport* or *genuine result*. Retry the transient ones; if they persist, **raise** — never let `asyncio.gather(return_exceptions=True)` plus a downstream type filter swallow an exception as if it were a negative result.

The anti-pattern — exceptions and legitimate `None`s become indistinguishable, so timeouts get filtered out exactly like genuine negatives:

```python
# ANTI-PATTERN: a timeout is filtered identically to a genuine "not reachable".
# The gold set silently collapsed ~70 → 40 survivors and the build reported success.
outcomes = await asyncio.gather(*[_evaluate(rest, r, max_hops) for r in chunk],
                                return_exceptions=True)
survivors = [o for o in outcomes if isinstance(o, dict)]   # drops exceptions AND Nones alike
```

The fix (`gold_set.py:_evaluate_chunk`, raising `GoldSetBuildError`) — share a transient-exception tuple with the transport layer, retry only those, raise a typed error if any persist, and let non-transient exceptions (real bugs) propagate immediately:

```python
async def _evaluate_chunk(rest, chunk, max_hops, retries=2) -> list[dict | None]:
    results: list[dict | None] = [None] * len(chunk)
    pending = list(range(len(chunk)))
    for attempt in range(retries + 1):
        outcomes = await asyncio.gather(
            *[_evaluate(rest, chunk[i], max_hops) for i in pending],
            return_exceptions=True)
        still = []
        for i, outcome in zip(pending, outcomes):
            if isinstance(outcome, TRANSIENT_EXC):   # transient → eligible for retry
                still.append(i)
            elif isinstance(outcome, BaseException):
                raise outcome                         # non-transient → surface the real bug now
            else:
                results[i] = outcome                  # dict survivor | None genuine filter
        if not still:
            return results
        pending = still
        if attempt < retries:
            await asyncio.sleep(_RETRY_BACKOFF_BASE * (attempt + 1))
    raise GoldSetBuildError(f"{len(pending)} record(s) still failing after {retries} retries "
                            "— refusing to silently drop them and return a short result.")
```

The shared classifier lives next to the retry loop that owns it (`kestrel_rest.py:TRANSIENT_EXC`, used in `_post`), so callers bucket failures the same way the transport does:

```python
TRANSIENT_EXC = (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError)

async def _post(self, path, body, _retries=2):
    last_exc = None
    for attempt in range(_retries + 1):
        try:
            r = await self._client.post(f"{self._base}{path}", json=body, headers=_headers())
            r.raise_for_status()
            return r.json()
        except TRANSIENT_EXC as exc:      # only transient is retried; raise_for_status (4xx/5xx) is NOT
            last_exc = exc
            await asyncio.sleep(1.0 * (attempt + 1))
    raise last_exc                         # persistent transient → raise, don't fabricate a result
```

The crucial asymmetry: a genuine empty/negative result (the record truly isn't reachable) is a *legitimate filter* and returns `None` silently — a corpus that simply lacks enough reachable items should return a short list, unchanged. Only *transport* failures are loud.

> **Drop-with-disclosure ≠ silently shrink.** A sibling doc ([sdk-stdio-mcp-unavailable…](./sdk-stdio-mcp-unavailable-migrate-to-http-data-in-prompt-2026-06-02.md)) recommends *dropping* unreliable findings on degradation — that is compatible with this rule because the drop is **visible and flagged** (`degraded`), not a silent disappearance. "Fail loud" is about never letting a failure masquerade as a real negative; an explicitly-flagged, counted drop satisfies it.

### 2. Retry transient, then checkpoint incrementally

The Claude Agent SDK spawns the `claude` CLI as a subprocess; its control-protocol handshake intermittently throws `Exception: Control request timeout: initialize`. One blip must not crash a multi-hour run, so wrap the call in a bounded retry — but a *persistent* failure must still raise, so a real outage fails loud rather than fabricating misses that bias the aggregate (`iterate_loop.py:default_llm_fn`):

```python
SDK_RETRIES, SDK_RETRY_BACKOFF = 3, 3.0

async def default_llm_fn(prompt, system) -> tuple[str, object]:
    last_exc = None
    for attempt in range(SDK_RETRIES + 1):
        try:
            return await query_with_usage(prompt, opts, node_name="cog_spike")
        except Exception as exc:          # the SDK raises varied transient errors
            last_exc = exc
            if attempt < SDK_RETRIES:
                await asyncio.sleep(SDK_RETRY_BACKOFF * (attempt + 1))
    raise last_exc                         # persistent → fail loud, do not return a fake empty turn
```

Pair retry with a checkpoint after *every* item, so a crash on item 87 keeps items 1–86. Persisting only at the very end once lost ~11 minutes; after this hardening an 8.5-hour run survived the same SDK flakiness. The orchestrator takes a `checkpoint` callback invoked per item (`run_phase0.py:run_phase0`), and the CLI writes a sibling `.partial.json` that is deleted only on clean completion (`run_phase0.py:_amain`):

```python
ckpt_path = out_path.with_suffix(".partial.json")

def checkpoint(bl, it, idx):              # crash-safety: completed items survive a mid-run death
    ckpt_path.write_text(json.dumps(
        {"_partial_through": idx, "n_total": len(items),
         "baseline_records": bl, "iterate_records": it}, indent=2, default=str))

# ... after the run finishes cleanly:
out_path.write_text(json.dumps(result, indent=2, default=str))
if ckpt_path.exists():
    ckpt_path.unlink()                     # clean exit → drop the partial
```

This *extends* the persist-by-default SOP (see Related): end-of-run persistence alone still loses everything on a mid-run crash — the per-item `.partial.json` is what makes a long run resumable/inspectable.

### 3. Measure cost with per-unit deltas, not cumulative counters

When a shared client carries a lifetime call counter, reading it per work-unit and summing across units double-counts catastrophically — a per-loop read of `rest.kestrel_calls` summed over the batch reported **291,758** calls against an actual **~1,919** (~150×). Snapshot the counter at unit start and record the *delta* (`iterate_loop.py:run_iterate_loop`):

```python
async def run_iterate_loop(rest, item, llm_fn) -> dict:
    calls_at_start = rest.kestrel_calls            # rest.kestrel_calls is cumulative/shared
    ...
    rec.update(kestrel_calls=rest.kestrel_calls - calls_at_start)   # THIS loop's calls only
    return rec
```

### Cross-cutting: persist by default, pin reproduce-inputs

*(extends the auto-memory SOP — see Related)* For runs that are expensive to reproduce, persisting is the **default, not an opt-in flag**. The orchestrator always writes a timestamped artifact to a `runs/` dir; `--out` only *overrides* the path, it is never *required* to get output. Pin everything needed to reproduce the number — frozen seed, data identity (a pinned source commit SHA), and config — into the artifact's `_meta`, so a result can be regenerated and audited weeks later.

## Why This Matters

- **A swallowed exception corrupts the science, not just the run.** In an A/B gate, a transient timeout filtered as a "miss" doesn't lose one data point — it shifts the recall estimate and can flip the verdict. The gold-set collapse (70 → 40) and the fabricated-miss risk in the LLM arm are the same bug wearing two hats.
- **Checkpointing converts a crash from "redo hours + repay the API bill" into "resume."** At hour-scale wall-clock and paid-per-call economics, losing completed work is a real dollar-and-time cost.
- **A 150× cost overcount makes the cost metric worthless** — and here cost was a gate input. Deltas over a shared counter are the difference between a number you can cite and noise.
- **Default-persist + pinned inputs is what makes an expensive result trustworthy later** — you can re-derive it, diff it, and defend it, instead of holding a stale JSON nobody can reproduce.

## When to Apply

Apply all three properties when **any** of these hold:

- The job fans out over a **live external API or an LLM/agent SDK** (especially the Claude Agent SDK, whose CLI-subprocess handshake is a known intermittent failure source).
- The run is **long** (minutes-to-hours) and/or **paid per call**, so re-running from scratch is costly.
- A failure can be **mistaken for a legitimate negative** — your pipeline filters results and an exception could look like "filtered out." This is the trigger for property 1 specifically; it is the most dangerous and easiest to miss.
- You **aggregate** results into a metric (recall, accuracy, cost) where a silently-dropped or fabricated item biases the number.
- You read a **cumulative counter on a shared/long-lived object** to attribute cost or usage per work-unit (property 3).

**Don't over-engineer** a single fast in-process call, a free/deterministic local computation, or a job cheap enough to re-run end-to-end — though the fail-loud classification (don't confuse an exception with a negative result) is cheap and worth keeping anywhere you filter.

## Examples

**Distinguishing a genuine filter from a transport failure** — the heart of property 1. Same call site, two outcomes that must NOT be merged (`gold_set.py:_evaluate`):

```python
async def _evaluate(rest, rec, max_hops) -> dict | None:
    if len(rec.interior) + 1 > max_hops:
        return None                        # GENUINE filter: gold path longer than the cap → silent None
    item = await to_gold_item(rest, rec, stratum="random")
    if not item:
        return None                        # GENUINE filter: didn't resolve → silent None
    if not await is_reachable(rest, item["start_curie"], item["gold_target_curie"], max_hops):
        return None                        # GENUINE filter: truly unreachable → silent None
    return item
    # A ReadTimeout raised anywhere above is NOT a None — it propagates to _evaluate_chunk,
    # which retries it and, if persistent, raises GoldSetBuildError. Two outcomes, never merged.
```

**Retry-then-raise as a reusable shape.** Both the HTTP layer (`_post`) and the SDK layer (`default_llm_fn`) follow the identical contract: bounded attempts with linear backoff, retry only the transient class, and on exhaustion `raise last_exc` — never return a sentinel that downstream code would read as a real (negative) result. Reuse this shape for any flaky boundary; only the exception set differs (`TRANSIENT_EXC` for HTTP, broad `Exception` for the SDK's varied transient errors).

**Per-unit delta vs. cumulative read.** The bug and fix differ by one subtraction: `rec["kestrel_calls"] = rest.kestrel_calls` (wrong — every loop reports the running total, summing to ~150× reality) vs. `rec["kestrel_calls"] = rest.kestrel_calls - calls_at_start` (right — this unit's contribution). Generalize: whenever you attribute a cumulative metric on a shared object to a sub-unit, snapshot at entry and report the delta.

## Related

- **[sdk-stdio-mcp-unavailable-migrate-to-http-data-in-prompt](./sdk-stdio-mcp-unavailable-migrate-to-http-data-in-prompt-2026-06-02.md)** — closest neighbor; same fail-loud-on-partial-results thesis ("emit findings on every exit path: happy, degraded, timeout, exception") and the shared `query_with_usage` SDK layer. Reconcile: its "drop with a `degraded` flag" is *visible* dropping, compatible with property 1's "never silently shrink."
- **[langfuse-sdk-v3-migration](./langfuse-sdk-v3-migration-2026-06-03.md)** — same swallowed-exception family ("most v2 calls fail silently under v3 rather than erroring loudly"); complementary on the broad-`except`-masks-failure theme.
- **Auto-memory SOP `persist-expensive-run-artifacts`** *(auto memory [claude])* — codified in global `~/.claude/CLAUDE.md` ("Experiment & Run Artifact Hygiene"): save results by default (no opt-in `--out` flag), pin reproduce-inputs. This doc **extends** it from end-of-run persistence to *incremental* mid-run checkpointing.
- **Issue #47** (open) — "variance computation silently truncates when run file is absent mid-sequence": a live instance of the property-1 anti-pattern; its fix is a canonical regression test for fail-loud-on-missing-data.
- **[pytest-venv-path-spaces-module-invocation](../developer-experience/pytest-venv-path-spaces-module-invocation-2026-06-01.md)** — run the prevention tests with `cd backend && uv run python -m pytest tests/code_on_graph_spike/ -q` (the venv path has spaces; bare `pytest` falls back to system Python).
