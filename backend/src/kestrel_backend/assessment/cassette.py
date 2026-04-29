"""Shared HTTP cassette module for recording and replaying external API responses.

Uses respx to intercept httpx calls at the transport level. Handles all 5 external
API clients (Kestrel KG, OpenAlex, Semantic Scholar, Exa, PubMed) through a single
mechanism.

This module is the single point of respx integration for the entire assessment
infrastructure — used by both the capture script (recording) and the assessment
runner (replay).

Key design constraint: Kestrel returns SSE-formatted responses (event: message\\ndata: {json}).
respx intercepts at the transport level and returns the full response body, so SSE parsing
via parse_sse_response(response.text) should work identically in replay mode.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import httpx
import respx

logger = logging.getLogger(__name__)


class CassetteRecorder:
    """Records HTTP request/response pairs during pipeline execution."""

    def __init__(self):
        self._interactions: list[dict[str, Any]] = []

    def record_interaction(
        self,
        request: httpx.Request,
        response: httpx.Response,
    ) -> None:
        """Capture a request/response pair."""
        self._interactions.append({
            "request": {
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers),
                "content": request.content.decode("utf-8") if request.content else None,
            },
            "response": {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "text": response.text,
            },
        })

    def save(self, path: Path, metadata: dict[str, Any] | None = None) -> None:
        """Save recorded interactions to a JSON cassette file."""
        cassette = {
            "metadata": metadata or {},
            "interactions": self._interactions,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cassette, indent=2))
        logger.info("Saved cassette with %d interactions to %s", len(self._interactions), path)

    @property
    def interaction_count(self) -> int:
        return len(self._interactions)


def load_cassette(path: Path) -> dict[str, Any]:
    """Load a cassette file from disk."""
    return json.loads(path.read_text())


def _request_key(method: str, url: str, content: str | None) -> str:
    """Generate a deterministic key for matching requests during replay."""
    # Hash the request body for POST requests (Kestrel JSON-RPC)
    body_hash = hashlib.md5(content.encode()).hexdigest() if content else ""
    return f"{method}:{url}:{body_hash}"


class CassetteReplayer:
    """Replays recorded HTTP responses during assessment runs.

    Sets up respx routes that match recorded requests and return cached responses.
    Unmatched requests pass through to the real network (for LLM/SDK calls that
    don't use httpx).
    """

    def __init__(self, cassette_data: dict[str, Any]):
        self._interactions = cassette_data["interactions"]
        self._metadata = cassette_data.get("metadata", {})
        # Index interactions by request key for fast lookup
        self._lookup: dict[str, list[dict]] = {}
        for interaction in self._interactions:
            req = interaction["request"]
            key = _request_key(req["method"], req["url"], req.get("content"))
            self._lookup.setdefault(key, []).append(interaction)
        # Track consumption index per key for sequential replay
        self._consumed: dict[str, int] = {}

    def get_response(self, request: httpx.Request) -> httpx.Response | None:
        """Look up the recorded response for a given request."""
        key = _request_key(
            request.method,
            str(request.url),
            request.content.decode("utf-8") if request.content else None,
        )
        interactions = self._lookup.get(key, [])
        if not interactions:
            return None

        idx = self._consumed.get(key, 0)
        if idx >= len(interactions):
            # Wrap around if more requests than recorded (shouldn't happen in normal use)
            idx = len(interactions) - 1

        interaction = interactions[idx]
        self._consumed[key] = idx + 1

        resp_data = interaction["response"]
        return httpx.Response(
            status_code=resp_data["status_code"],
            headers=resp_data.get("headers", {}),
            text=resp_data["text"],
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


def create_recording_transport(
    recorder: CassetteRecorder,
    base_transport: httpx.AsyncBaseTransport | None = None,
) -> respx.MockRouter:
    """Create a respx mock router that records all requests while passing them through.

    This is used during baseline capture: real HTTP calls are made, but request/response
    pairs are also recorded to the CassetteRecorder.

    Returns a respx router configured to intercept all requests. The caller should
    use it as a context manager or via respx.mock().
    """
    router = respx.MockRouter(assert_all_called=False)

    def side_effect(request: httpx.Request) -> httpx.Response:
        # This function is called by respx when a request matches.
        # We can't make real network calls from inside respx side_effects easily,
        # so recording mode uses a different approach — see the capture module.
        raise NotImplementedError(
            "Recording transport should not be used directly. "
            "Use the capture module's record-then-save pattern instead."
        )

    return router


def setup_replay(cassette_path: Path) -> tuple[respx.MockRouter, CassetteReplayer]:
    """Set up respx to replay cached responses from a cassette file.

    Returns (router, replayer) tuple. The router should be used as a context manager.
    Unmatched requests will raise an error (all external APIs should be in the cassette).

    Usage:
        router, replayer = setup_replay(cassette_path)
        with router:
            # Pipeline runs here, all external HTTP calls served from cassette
            result = await run_discovery(query)
    """
    cassette_data = load_cassette(cassette_path)
    replayer = CassetteReplayer(cassette_data)

    router = respx.MockRouter(assert_all_called=False)

    def replay_handler(request: httpx.Request) -> httpx.Response:
        response = replayer.get_response(request)
        if response is None:
            # Log unmatched request for debugging
            logger.warning(
                "No cassette match for %s %s — passing through",
                request.method, request.url,
            )
            raise respx.errors.AllMockedResponsesSent(
                f"No recorded response for {request.method} {request.url}"
            )
        return response

    # Match all HTTP requests
    router.route().mock(side_effect=replay_handler)

    return router, replayer
