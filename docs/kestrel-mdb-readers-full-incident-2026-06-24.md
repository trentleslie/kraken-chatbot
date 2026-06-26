# Kestrel incident report: `MDB_READERS_FULL` (LMDB reader-table exhaustion)

**Service:** Kestrel MCP graph API — `https://kestrel.nathanpricelab.com/mcp` (host path `/home/ubuntu/kestrel/`)
**Date:** 2026-06-23 → 2026-06-24
**Reported by:** KRAKEN discovery-pipeline team (a downstream client of Kestrel)
**Severity:** High — graph reads fail service-wide; does not self-recover.

---

## TL;DR

During a batch of KRAKEN discovery-pipeline runs against Kestrel, the LMDB reader-lock table on the Kestrel host became exhausted. Graph queries now return:

```
mdb_txn_begin: MDB_READERS_FULL: Environment maxreaders limit reached
```

The condition is **persistent**: a single low-concurrency probe (3 sequential `one_hop_query` calls) still reproduces it more than 30 minutes after load stopped, which means stale reader slots are not being reclaimed and the LMDB environment needs a reader-table reset (a **service restart** is the simplest fix).

Two factors combined to cause it:
1. **Client side (ours):** the KRAKEN `direct_kg` node issues an *unbounded* concurrent fan-out of `one_hop_query` calls (≈ N_entities × 6), reaching several hundred simultaneous graph reads per module at module scale. We own this and are fixing it (see "Client-side remediation").
2. **Server side (Kestrel):** confirmed in source — the LMDB environment is opened with **no `max_readers`** (so the table is the LMDB default of 126) and **`reader_check()` is never called** (so stale slots are never reaped). A burst exhausts the table and the exhaustion then persists until restart.

---

## Did the client (KRAKEN) cause this?

Yes — KRAKEN supplied the load that exhausted the reader table — but the burst became a *persistent, service-wide outage* because of two confirmed Kestrel-side gaps. Both statements are true and worth separating.

**What KRAKEN did:** our `direct_kg` node issues an unbounded concurrent fan-out of `one_hop_query` calls (≈ N_entities × 6, i.e. ~600–900 simultaneous reads per module), and we ran ~30 such module passes back-to-back with no cooldown. The fingerprint is unambiguous: Kestrel was healthy beforehand, early runs showed sporadic `MDB_READERS_FULL` we absorbed with retries (i.e. we were already at the ceiling), and the failure rate then rose monotonically with our runs (Grey 10 → Lightcyan all-zero → Grey60 418). We own this and are bounding the fan-out + adding cooldowns.

**Why it should not have latched:** a transient load spike should not permanently disable a shared service. It did because (confirmed in source) the reader table is left at the default 126 and nothing reaps stale slots. A hardened LMDB service would have shed load or self-recovered within seconds. So: **KRAKEN pulled the trigger; the reader-handling configuration loaded the gun.** The fixes are split accordingly below.

---

## Symptom and impact

- All graph-read tools (`one_hop_query`, `multi_hop_query`, etc.) return an embedded error with `status` 500-class and body `mdb_txn_begin: MDB_READERS_FULL: Environment maxreaders limit reached`.
- The full traceback originates inside the Kestrel process — every frame is under `/home/ubuntu/kestrel/` (e.g. `kestrel/utils/usage_tracking.py`, Starlette middleware → anyio task group → LMDB `mdb_txn_begin`).
- In a degraded-but-not-yet-erroring window, Kestrel instead returned **empty/zero result sets** (valid HTTP 200, `results_count: 0`) for entities that are known hubs, before tipping over into hard `MDB_READERS_FULL` errors.
- Downstream impact on KRAKEN: entities with real edges were scored as 0-edge (mis-classified), and whole module runs collapsed (e.g. all 27/33 entities forced to the "cold-start" branch, 6–8 findings instead of hundreds).

This is specifically **Kestrel**, not biomapper: the failing calls go directly to `KESTREL_MCP_URL`, and the LMDB store is Kestrel's own. biomapper is not in this code path.

---

## Timeline and evidence

KRAKEN ran ~30 full-module discovery passes over 18 WGCNA modules (two model sweeps, back-to-back). Early runs were clean; degradation accumulated and then accelerated:

| Run (chronological) | Observation |
|---|---|
| Early modules | Clean. Sporadic `MDB_READERS_FULL` (≈12 per run) fully absorbed by client retries. |
| Grey | 10 triage edge-count queries failed (could not be measured). |
| Lightcyan | **All 27 entities** returned 0 edges (empty result sets); 8 findings total. |
| Grey60 | **418 `MDB_READERS_FULL` errors in a single run**; all 33 entities forced to cold-start; 6 findings. |
| +30 min, load stopped | A **3-query sequential probe still returns `MDB_READERS_FULL`.** |

The last row is the important one: with effectively zero concurrency and 30 minutes of idle time, the error persists. That rules out "transient busy" and points to **leaked/stale reader slots** in the LMDB lock table.

---

## Root cause analysis

### LMDB reader-table mechanics (why this happens and why it persists)

LMDB maintains a fixed-size **reader lock table** in the environment's lock file. Its size is set once at environment open via `mdb_env_set_maxreaders()` (default **126**). Every concurrent **read transaction** occupies one slot for its lifetime; the slot is released when the read txn ends (commit/abort). When all slots are occupied, `mdb_txn_begin()` for a new reader fails with `MDB_READERS_FULL`.

Crucially, slots held by readers that exited **without** cleanly ending their txn (process killed mid-transaction, a request cancelled mid-read, a long-lived read txn) become **stale** and are **not** reclaimed automatically. LMDB only reaps stale slots when `mdb_reader_check()` runs, or when the environment is re-opened (process restart). This is why a 30-minute wait did not help: nothing is reaping the leaked slots.

### Confirmed against Kestrel source

Verified in `github.com/Phenome-Health/kestrel` (this is from source, not inference):

- **`max_readers` is never set.** `kestrel/clients/lmdb_client.py:28`:
  ```python
  self.env = lmdb.open(str(data_dir / "lmdb"), readonly=True, max_dbs=10)
  ```
  With no `max_readers=` argument, LMDB uses its built-in default of **126**. That is the exact ceiling the load hit.
- **`reader_check()` is never called** — a repo-wide search of `kestrel/` and `tests/` finds zero references to `reader_check`, `max_readers`, or stale-slot handling. Once slots are occupied/stale, nothing reclaims them short of a process restart. This is the direct cause of the *persistence* (a single sequential probe still failing 30 min after load stopped).
- **Each query holds its read txn for the full query duration.** `kestrel/clients/lmdb_client.py:314, 370, 381, 399` all wrap an entire query in `with self.env.begin() as txn:` (the complete curie loop and edge scan run inside the txn). The transactions are correctly context-managed, so this is not a leak under normal completion — but it does mean **one occupied reader slot per in-flight query for its entire runtime**. The pressure is therefore driven by *concurrency*, not per-query size: ~700 simultaneous in-flight queries demand ~700 reader slots against a table of 126. (KRAKEN's `direct_kg` one-hop calls each use a small `limit`, but it fires hundreds of them at once; the `triage` preview path separately uses `limit=10000` to count edges, holding a slot for the duration of that count.)
- **The health endpoint shares the same failure mode.** `kestrel/api.py:485` implements `/health` as `with client.lmdb.env.begin(): pass`, so during reader exhaustion the health check itself raises and the service reports `unhealthy` — consistent with what we observed, and useful as a monitoring signal.

Net: the transactions themselves are written correctly; the gaps are the **un-raised reader ceiling (126)** and the **absence of any stale-slot reaping**, which together turn a client burst into a latched outage.

### What KRAKEN threw at it (the trigger)

KRAKEN's discovery pipeline runs per WGCNA module (≈20–200 resolved entities each). The query profile per module:

- **`direct_kg` node — the dominant burst (and unbounded):** for every well-characterized/moderate entity it calls `analyze_via_api`, and each of those issues **6 concurrent `one_hop_query` calls** (3 Biolink categories × 2 ranking presets, `mode=slim`). Both the inner per-entity fan-out *and* the outer per-entity loop use a bare `asyncio.gather` with **no concurrency cap**. Peak concurrent `one_hop_query` ≈ **N_entities × 6** (e.g. ~122 entities × 6 ≈ **730 simultaneous graph reads** for one module).
- `triage`: `one_hop_query` (`mode=preview`, `limit=10000`) per entity, bounded to 8 concurrent.
- `cold_start`: `similar_nodes` + `one_hop_query` per analogue.
- `pathway_enrichment`: `multi_hop_query` (`max_hops=2`), issued serially but reader-heavy per call (2-hop traversal).
- `integration`: doubly-pinned `multi_hop_query` per category pair + an optional `subgraph_query` (bounded to 2).

The client uses a single `httpx.AsyncClient` with **HTTP/2 multiplexing** and `max_connections=100`, so hundreds of JSON-RPC requests are multiplexed onto the connection pool and arrive at Kestrel as a near-simultaneous burst. If each Kestrel request opens its own LMDB read txn, a single `direct_kg` phase can demand far more readers than the default table holds. Running ~30 module passes back-to-back compounded any per-request slot leakage until the table was permanently full.

**Summary:** an unbounded client fan-out (ours) drove concurrent reads well past `maxreaders`, and the server's reader slots are not being reclaimed, so the exhaustion latched.

---

## Immediate remediation

To restore service now (any one of these):
1. **Restart the Kestrel service.** Re-opening the LMDB environment re-initializes the reader table and clears all stale slots. Simplest and most reliable.
2. Or run **`mdb_reader_check()`** against the environment (e.g. via `mdb_stat -rr <env>` from lmdb-utils, which calls the reader check and reaps stale slots) without a full restart.

`mdb_stat -r <env>` will show current vs. max readers and how many are stale — useful to confirm before/after.

---

## Recommendations — Kestrel side

1. **Raise `max_readers`** at `kestrel/clients/lmdb_client.py:28`, e.g.:
   ```python
   self.env = lmdb.open(str(data_dir / "lmdb"), readonly=True, max_dbs=10, max_readers=1024)
   ```
   It must be set at environment-open time (takes effect on restart). Cheap insurance against bursty clients; the current implicit 126 is far too low for a concurrent reader workload.
2. **Reap stale slots, and self-heal on exhaustion.** The Python `lmdb` binding exposes `env.reader_check()` (returns the number of stale slots cleared). Call it (a) periodically on a timer, and (b) as a recovery step when a read `txn_begin` raises `MDB_READERS_FULL`, then retry the txn once. This converts a latched outage into a transient blip. There is currently no such call anywhere in the codebase.
3. **Consider shortening slot occupancy under load.** The query txns are already correctly context-managed, so this is secondary — but since a slot is held for the full query runtime, very large scans (`limit=10000`) hold a slot a long time. Capping per-query result size, or splitting long scans, reduces peak simultaneous slot demand. Pairs well with (1) and (4).
4. **Add server-side backpressure.** A bounded semaphore or small queue around graph-read handlers would let a client burst degrade gracefully rather than exhausting readers. Returning **HTTP 503 with `Retry-After`** on overload (instead of a 500-class embedded error) lets well-behaved clients back off.
5. **Don't return empty result sets on internal failure.** During the degraded window Kestrel returned valid-looking `results_count: 0` responses; downstream clients cannot distinguish "genuinely no edges" from "backend degraded," which silently corrupts analyses. A distinct error/503 is much safer.

## Client-side remediation (KRAKEN — we own these)

1. **Bound the `direct_kg` one-hop fan-out** with a global semaphore (it is currently uncapped — the single biggest burst). This is our highest-impact fix and is queued.
2. **Cooldowns between back-to-back module runs**, and a lower global cap on concurrent Kestrel calls during multi-module sweeps, so we never present a thundering herd.
3. Revisit the per-entity 6-way preset fan-out and the `limit=10000` preview to reduce reader demand.

We will not resume the multi-module sweep until (a) Kestrel is restarted/healthy and (b) our `direct_kg` fan-out is bounded, so we don't re-trigger this.

---

## Reproduction / load profile (for testing the fix)

A single KRAKEN `direct_kg` phase at module scale is sufficient to reproduce: ~100–150 entities, each issuing 6 concurrent `one_hop_query` (`mode=slim`), with no client-side cap → ~600–900 concurrent read requests. Equivalent synthetic load: fire ~500+ concurrent `one_hop_query` calls at the `/mcp` endpoint and watch `mdb_stat -r` approach `maxreaders`. With `maxreaders` raised and `mdb_reader_check()` wired in, the same load should degrade gracefully (or 503) rather than latching into permanent `MDB_READERS_FULL`.

## Environment

- Kestrel endpoint: `https://kestrel.nathanpricelab.com/mcp`
- Kestrel host path (from traceback): `/home/ubuntu/kestrel/` (Starlette/anyio app, LMDB-backed)
- Client: KRAKEN backend, `httpx.AsyncClient` (HTTP/2, `max_connections=100`), MCP-over-HTTP/JSON-RPC with SSE responses
- Error string: `mdb_txn_begin: MDB_READERS_FULL: Environment maxreaders limit reached`
