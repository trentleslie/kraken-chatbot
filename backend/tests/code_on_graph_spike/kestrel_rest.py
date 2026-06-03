"""Thin Kestrel REST /api client + cap-free path parser + grounding helpers.

Why REST and not the kraken MCP client (plan finding #1): Kestrel's /mcp server
forwards tool calls to its own REST API but intermittently fails to pass through
X-API-Key during streamable-http calls -> spurious "Invalid API key". The REST
/api surface (what biomapper2 uses) is reliable. We reuse kraken's transport-
agnostic `_canonical_curie` for the grounding membership check.

Endpoints used:
  POST /hybrid-search  {search_text, limit, category?}  -> {name: [{id, score, neighbors_count, ...}]}
  POST /multi-hop      {start_node_ids, end_node_ids, max_path_length, min_path_length, limit, mode}
                       -> {"results": [{..., "paths": [[curie, ...], ...]}]}
  POST /one-hop        {start_node_ids, mode:"preview", limit} -> {results_count: int}
  POST /get-nodes      {curies} -> {curie: {equivalent_ids: [...], ...}}
"""
from __future__ import annotations

import os
from typing import Any

import httpx

from kestrel_backend.graph.nodes.entity_resolution import _canonical_curie

KESTREL_API_URL = os.getenv("KESTREL_API_URL", "https://kestrel.nathanpricelab.com/api")
KESTREL_API_KEY = os.getenv("KESTREL_API_KEY", "")


def _headers() -> dict[str, str]:
    return {"X-API-Key": os.getenv("KESTREL_API_KEY", "") or KESTREL_API_KEY,
            "Content-Type": "application/json"}


class KestrelREST:
    """Async REST client. Pass an httpx.AsyncClient for tests (respx/mock)."""

    def __init__(self, client: httpx.AsyncClient | None = None, base_url: str | None = None):
        self._base = (base_url or KESTREL_API_URL).rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=60.0)
        self._owns_client = client is None
        self.kestrel_calls = 0  # R10: count Kestrel calls (incl. grounding lookups)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "KestrelREST":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        self.kestrel_calls += 1
        r = await self._client.post(f"{self._base}{path}", json=body, headers=_headers())
        r.raise_for_status()
        return r.json()

    async def hybrid_search(self, text: str, limit: int = 5, category: str | None = None) -> list[dict]:
        body: dict[str, Any] = {"search_text": text, "limit": limit}
        if category:
            body["category"] = category
        data = await self._post("/hybrid-search", body)
        # Response is keyed by the search term; take the first list of hits.
        if isinstance(data, dict):
            hits = data.get(text)
            if hits is None:
                hits = next((v for v in data.values() if isinstance(v, list)), None)
            return hits or []
        return data if isinstance(data, list) else []

    async def resolve(self, text: str, category: str | None = None) -> tuple[str | None, int | None]:
        """Resolve a name to (curie, degree). Prefers the highest-degree hit to avoid
        low-degree non-human orthologs (plan finding #3)."""
        hits = await self.hybrid_search(text, limit=5, category=category)
        if not hits:
            return None, None
        top = max(hits, key=lambda h: (h.get("neighbors_count") or 0))
        return top.get("id"), top.get("neighbors_count")

    async def multi_hop(self, start: list[str], end: list[str] | None, max_path_length: int,
                        limit: int, min_path_length: int = 1, mode: str = "full") -> dict[str, Any]:
        body: dict[str, Any] = {"start_node_ids": start, "max_path_length": max_path_length,
                                "min_path_length": min_path_length, "limit": limit, "mode": mode}
        if end:
            body["end_node_ids"] = end
        return await self._post("/multi-hop", body)

    async def degree(self, curie: str) -> int | None:
        data = await self._post("/one-hop", {"start_node_ids": curie, "mode": "preview", "limit": 1000000})
        if isinstance(data, dict):
            return data.get("results_count") or data.get("count")
        return None

    async def equivalent_ids(self, curie: str) -> set[str]:
        try:
            data = await self._post("/get-nodes", {"curies": curie})
            node = data.get(curie, {}) if isinstance(data, dict) else {}
            out: set[str] = set()
            for x in node.get("equivalent_ids", []):
                c = _canonical_curie(x)
                if c:
                    out.add(c)
            return out
        except Exception:
            return set()


# ---- cap-free path parsing (no production paths[:10] cap) ----

def parse_paths(multi_hop_response: dict[str, Any]) -> list[list[str]]:
    """Extract every path as a list of node CURIEs from a /multi-hop response,
    which is shaped {"results": [{..., "paths": [[curie, ...], ...]}]}. No top-N cap."""
    out: list[list[str]] = []
    results = multi_hop_response.get("results") if isinstance(multi_hop_response, dict) else None
    if isinstance(results, list):
        for res in results:
            for path in (res.get("paths") or []) if isinstance(res, dict) else []:
                if isinstance(path, list):
                    out.append([c for c in path if isinstance(c, str)])
    return out


def path_contains_all(path: list[str], gold_bridges: list[str]) -> bool:
    """True if every gold interior CURIE is present in the path (canonical match).
    The frozen bridge unit: a hit requires ALL gold interior node(s)."""
    canon = {_canonical_curie(c) for c in path}
    return all(_canonical_curie(g) in canon for g in gold_bridges)


def any_path_recovers(paths: list[list[str]], gold_bridges: list[str]) -> bool:
    return any(path_contains_all(p, gold_bridges) for p in paths)


async def is_grounded(rest: KestrelREST, emitted_curie: str, returned_curies: set[str]) -> bool:
    """Grounding contract (R9): the emitted CURIE must be a node returned by an
    executed call — by canonical match, or as the equivalent_id of a returned node.
    Reuses the recall_gate _same_entity idea over REST (finding #1)."""
    emitted = _canonical_curie(emitted_curie)
    returned_canon = {_canonical_curie(c) for c in returned_curies}
    if emitted in returned_canon:
        return True
    # emitted may be the equivalent_id of a returned node, or vice versa
    if emitted_curie in {c for c in returned_curies}:
        return True
    eq = await rest.equivalent_ids(emitted_curie)
    return bool(eq & returned_canon)
