---
title: "Kestrel LMDB MDB_READERS_FULL: unbounded asyncio.gather exhausts the reader pool"
date: 2026-06-26
category: performance-issues
module: direct_kg
problem_type: performance_issue
component: background_job
symptoms:
  - "mdb_txn_begin: MDB_READERS_FULL: Environment maxreaders limit reached on every Kestrel query"
  - "Failure persists 30+ minutes after load stops (reader slots leaked, not transiently busy)"
  - "Entities silently mis-bucketed to cold_start when their edge-count reads fail (triage_measurement_failures nonzero)"
  - "A module-scale batch sweep halts with KG queries failing service-wide"
root_cause: async_timing
resolution_type: code_fix
severity: high
tags: [asyncio, concurrency, lmdb, kestrel, semaphore, direct-kg, reader-pool, thundering-herd, batch-scale]
---

# Kestrel LMDB MDB_READERS_FULL: unbounded asyncio.gather exhausts the reader pool

## Problem

A discovery-pipeline node fired an unbounded `asyncio.gather` of Kestrel `one_hop_query` calls, producing hundreds of simultaneous LMDB read transactions against the shared Kestrel knowledge-graph service at module scale. The service's reader-slot table filled and **latched**, returning `MDB_READERS_FULL` on every subsequent query for all users until the Kestrel process was restarted.

## Symptoms

- Every KG tool call returned `mdb_txn_begin: MDB_READERS_FULL: Environment maxreaders limit reached` (an LMDB error surfaced in the Kestrel response payload, not an HTTP-layer error).
- **Persistent, not transient:** a 3-query probe still failed 30+ minutes after the batch load had fully stopped, confirming reader slots were *leaked*, not merely busy.
- **Silent downstream degradation:** failed reads returned empty results, so triage entities whose edge-count probes also failed were silently classified `cold_start` (a 0-edge bucket), inflating that path and suppressing KG-grounded analysis. `triage_measurement_failures` was nonzero on affected runs.
- Degradation rose monotonically across back-to-back runs (a few absorbed errors → all-zero results → hundreds of errors in a single run).

## What Didn't Work

- **Waiting it out.** The first hypothesis was transient overload. ~30 minutes idle did not help: LMDB only reclaims stale reader slots on a `reader_check()` call or process restart, never by elapsed idle time.
- **Auth red herring.** An early diagnostic probe returned HTTP 403 "Invalid API key." This was `load_dotenv()` not finding `backend/.env` because the probe ran from a different working directory, not a real auth failure. It cost time before the real LMDB error surfaced. (Run probes from `backend/` or load the env explicitly.)
- **Blaming biomapper.** Biomapper also calls Kestrel, so it was suspected. The traceback showed all failing calls ran directly under the Kestrel traversal path (`/home/ubuntu/kestrel/clients/lmdb_client.py`); biomapper was not involved.
- **Retrying without bounding.** Re-issuing the failed calls without a concurrency cap simply re-exhausts the same reader pool.

## Solution

Cap concurrent `one_hop_query` calls with a configurable, module-level semaphore, and route every Tier-1 one-hop through it.

Config (`backend/src/kestrel_backend/graph/pipeline_config.py`, `DirectKGConfig`):

```python
one_hop_concurrency: int = Field(
    default=16,
    ge=1,
    description="Global cap on concurrent one_hop_query calls in Tier-1 analysis. "
    "Tier-1 fans out 6 one-hop calls per entity (3 categories x 2 presets) across all "
    "entities; without a cap this reached ~N_entities*6 simultaneous reads and exhausted "
    "Kestrel's LMDB reader pool at module scale. Override via KRAKEN_DIRECTKG_ONEHOP_CONCURRENCY.",
)
```

Before (`backend/src/kestrel_backend/graph/nodes/direct_kg.py`, unbounded fan-out):

```python
tasks.append(call_kestrel_tool("one_hop_query", {
    "start_node_ids": curie,
    "end_node_category": cat_filter,
    "ranking": preset,
    "mode": "slim",
    "limit": _config.preset_limit,
}))
results = await asyncio.gather(*tasks, return_exceptions=True)  # uncapped across all entities
```

After (bounded through a shared semaphore):

```python
ONE_HOP_SEMAPHORE = asyncio.Semaphore(
    int(os.getenv("KRAKEN_DIRECTKG_ONEHOP_CONCURRENCY") or _config.one_hop_concurrency)
)

async def _bounded_one_hop(args: dict) -> Any:
    """Issue one_hop_query under the shared concurrency bound (reader-pool protection)."""
    async with ONE_HOP_SEMAPHORE:
        return await call_kestrel_tool("one_hop_query", args)

# task construction:
tasks.append(_bounded_one_hop({
    "start_node_ids": curie,
    "end_node_category": cat_filter,
    "ranking": preset,
    "mode": "slim",
    "limit": _config.preset_limit,
}))
```

The client bound prevents *future* exhaustion but does not clear an *already-leaked* table. A one-time Kestrel recovery is needed to drain the stale slots before the fix can be verified: restart the Kestrel service, or run `mdb_stat -rr <env>` / call `env.reader_check()` on the server.

Verified: after the bound plus a Kestrel restart, the full 18-module Opus sweep re-ran with zero `MDB_READERS_FULL` and `triage_measurement_failures = 0` on every module. Shipped via PR #90, promoted to production via PR #91.

## Why This Works

LMDB keeps a fixed-size reader-lock table sized by `max_readers` (default 126 when unset). Each concurrent read transaction occupies one slot for its lifetime; slots are reclaimed only by `reader_check()` or environment close, so slots held by coroutines that exited uncleanly become stale and accumulate. Hundreds of simultaneous coroutines each opening a read transaction filled the table and, with no reaping, latched it. Bounding to 16 concurrent one-hop calls keeps peak simultaneous readers far below 126, leaving headroom for the other pipeline nodes and other clients of the shared service.

The server-side gaps that turned a transient client burst into a persistent outage were confirmed against the Kestrel source (`github.com/Phenome-Health/kestrel`): `clients/lmdb_client.py` opens the environment with no `max_readers` (so the default 126) and never calls `reader_check()`.

## Prevention

- **Bound every KG-query fan-out per node** with a config-driven `asyncio.Semaphore` wrapping a `_bounded_<tool>` helper, called from every `asyncio.gather` task list. Treat an uncapped `gather` over a variable-length entity list as a defect when the work is a shared-service call.
- **Audit signal:** `grep -rn "asyncio.gather" backend --include=*.py | grep -iv "bounded\|semaphore"`, then check whether any `call_kestrel_tool` / `one_hop_query` / `multi_hop_query` appears in the same file.
- **Precedent / pattern:** the triage node had the identical unbounded-gather bug, fixed earlier (PR #89) with a `TriageConfig.kestrel_concurrency` bound. The reusable rule: any pipeline node that fans out KG queries over an entity list must bound its concurrency against the shared graph service.
- **Fail loud, not empty:** a failed KG read must be distinguishable from a genuine empty result, or downstream nodes will read infrastructure failure as biology (the silent cold-start mis-bucketing here). Surface a visible error marker in state.
- **Server-side asks (Kestrel upstream):** raise `max_readers` well above 126; call `reader_check()` periodically and/or on `MDB_READERS_FULL` before erroring; return HTTP 503 with `Retry-After` under reader pressure so clients back off instead of piling on.

## Related Issues

- `docs/kestrel-mdb-readers-full-incident-2026-06-24.md` — full incident report (timeline, Kestrel source confirmation, client/server remediation). Primary source for this learning.
- `docs/plans/2026-06-23-001-fix-triage-concurrency-scale-plan.md` — the sibling triage unbounded-gather fix (PR #89), same root-cause class and same semaphore-bounding fix.
- `docs/solutions/best-practices/reliable-long-running-llm-batch-runs-2026-06-07.md` — fan-out reliability: distinguishing transient transport failures from genuine negative results.
- `docs/solutions/logic-errors/kestrel-direction-param-triage-cold-start-2026-06-11.md` — a *different* cause of triage false cold-starts (wrong MCP parameter); see both together when debugging cold-start inflation.
- PRs #90 (fix) and #91 (dev→main promotion).
