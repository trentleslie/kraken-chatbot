"""Spike test: Validate respx can intercept and replay Kestrel SSE responses.

This is the Unit 0 gating test. If Kestrel SSE round-trips correctly through
respx record/replay, we proceed with transport-level recording. If it fails,
we fall back to function-level mocking.

Decision gate: Binary on Kestrel SSE — parsed JSON must be byte-identical
after round-trip through respx.
"""

import json
from pathlib import Path

import httpx
import pytest
import respx

from kestrel_backend.assessment.cassette import (
    CassetteRecorder,
    CassetteReplayer,
    load_cassette,
    setup_replay,
)
from kestrel_backend.kestrel_client import parse_sse_response


# --- Synthetic SSE responses for testing ---

KESTREL_SSE_RESPONSE = (
    'event: message\n'
    'data: {"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text",'
    '"text":"{\\"entities\\":[{\\"name\\":\\"NAD+\\",\\"curie\\":\\"CHEBI:15422\\"}]}"}]}}\n'
)

KESTREL_SSE_PARSED = {
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
        "content": [
            {
                "type": "text",
                "text": '{"entities":[{"name":"NAD+","curie":"CHEBI:15422"}]}',
            }
        ]
    },
}

OPENALEX_JSON_RESPONSE = {
    "results": [
        {
            "id": "https://openalex.org/W12345",
            "title": "NAD+ metabolism in aging",
            "publication_year": 2023,
            "cited_by_count": 42,
        }
    ]
}


class TestRespxSSECompat:
    """Validate that respx can record and replay SSE-formatted responses."""

    @respx.mock
    async def test_kestrel_sse_roundtrip_via_respx(self):
        """Core spike test: SSE response round-trips through respx with identical parsed output."""
        # Mock a Kestrel endpoint returning SSE-formatted text
        respx.post("https://kestrel.nathanpricelab.com/mcp").mock(
            return_value=httpx.Response(
                status_code=200,
                text=KESTREL_SSE_RESPONSE,
                headers={"content-type": "text/event-stream"},
            )
        )

        # Make request through httpx (as KestrelClient does)
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://kestrel.nathanpricelab.com/mcp",
                json={"jsonrpc": "2.0", "method": "tools/call", "id": 1},
            )

        # Parse SSE response (same as kestrel_client.py:151)
        parsed = parse_sse_response(response.text)

        # Verify parsed JSON matches expected output
        assert parsed == KESTREL_SSE_PARSED
        assert response.text == KESTREL_SSE_RESPONSE

    @respx.mock
    async def test_openalex_json_roundtrip_via_respx(self):
        """Verify stateless JSON API round-trips through respx."""
        respx.get("https://api.openalex.org/works").mock(
            return_value=httpx.Response(
                status_code=200,
                json=OPENALEX_JSON_RESPONSE,
            )
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                "https://api.openalex.org/works",
                params={"search": "NAD+ aging", "per_page": 5},
            )

        data = response.json()
        assert data == OPENALEX_JSON_RESPONSE
        assert len(data["results"]) == 1


class TestCassetteRecordReplay:
    """Test the full record -> save -> load -> replay cycle."""

    async def test_record_and_replay_sse_response(self, tmp_path: Path):
        """Full round-trip: record an SSE response, save to cassette, replay, verify identical parse."""
        recorder = CassetteRecorder()

        # Simulate recording an SSE interaction
        request = httpx.Request(
            "POST",
            "https://kestrel.nathanpricelab.com/mcp",
            json={"jsonrpc": "2.0", "method": "tools/call", "id": 1},
        )
        response = httpx.Response(
            status_code=200,
            text=KESTREL_SSE_RESPONSE,
            headers={"content-type": "text/event-stream", "mcp-session-id": "test-session"},
        )
        recorder.record_interaction(request, response)

        # Save cassette
        cassette_path = tmp_path / "test_cassette.json"
        recorder.save(cassette_path, metadata={"kestrel_api_version": "v1.16.0"})

        assert cassette_path.exists()
        assert recorder.interaction_count == 1

        # Load and replay
        cassette_data = load_cassette(cassette_path)
        assert cassette_data["metadata"]["kestrel_api_version"] == "v1.16.0"
        assert len(cassette_data["interactions"]) == 1

        replayer = CassetteReplayer(cassette_data)

        # Replay the same request
        replay_request = httpx.Request(
            "POST",
            "https://kestrel.nathanpricelab.com/mcp",
            json={"jsonrpc": "2.0", "method": "tools/call", "id": 1},
        )
        replayed_response = replayer.get_response(replay_request)

        assert replayed_response is not None
        assert replayed_response.status_code == 200
        assert replayed_response.text == KESTREL_SSE_RESPONSE

        # Parse the replayed SSE response — this is the critical test
        parsed_from_replay = parse_sse_response(replayed_response.text)
        parsed_from_original = parse_sse_response(KESTREL_SSE_RESPONSE)

        assert parsed_from_replay == parsed_from_original
        # Verify byte-identical JSON serialization
        assert json.dumps(parsed_from_replay, sort_keys=True) == json.dumps(parsed_from_original, sort_keys=True)

    async def test_record_and_replay_json_api(self, tmp_path: Path):
        """Record and replay a standard JSON API response (OpenAlex-style)."""
        recorder = CassetteRecorder()

        request = httpx.Request(
            "GET",
            "https://api.openalex.org/works?search=NAD%2B+aging&per_page=5",
        )
        response = httpx.Response(
            status_code=200,
            json=OPENALEX_JSON_RESPONSE,
        )
        recorder.record_interaction(request, response)

        cassette_path = tmp_path / "openalex_cassette.json"
        recorder.save(cassette_path)

        cassette_data = load_cassette(cassette_path)
        replayer = CassetteReplayer(cassette_data)

        replayed = replayer.get_response(
            httpx.Request("GET", "https://api.openalex.org/works?search=NAD%2B+aging&per_page=5")
        )

        assert replayed is not None
        assert replayed.json() == OPENALEX_JSON_RESPONSE

    async def test_replay_unmatched_request_returns_none(self, tmp_path: Path):
        """Unmatched requests should return None from the replayer."""
        recorder = CassetteRecorder()
        cassette_path = tmp_path / "empty_cassette.json"
        recorder.save(cassette_path)

        cassette_data = load_cassette(cassette_path)
        replayer = CassetteReplayer(cassette_data)

        result = replayer.get_response(
            httpx.Request("GET", "https://unknown-api.com/endpoint")
        )
        assert result is None

    async def test_setup_replay_creates_working_mock(self, tmp_path: Path):
        """setup_replay() should create a respx router that serves cached responses."""
        # Create a cassette with one interaction
        recorder = CassetteRecorder()
        request = httpx.Request(
            "POST",
            "https://kestrel.nathanpricelab.com/mcp",
            json={"jsonrpc": "2.0", "method": "tools/call", "id": 1},
        )
        response = httpx.Response(status_code=200, text=KESTREL_SSE_RESPONSE)
        recorder.record_interaction(request, response)

        cassette_path = tmp_path / "replay_test.json"
        recorder.save(cassette_path)

        # Set up replay
        router, replayer = setup_replay(cassette_path)

        with router:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://kestrel.nathanpricelab.com/mcp",
                    json={"jsonrpc": "2.0", "method": "tools/call", "id": 1},
                )

        assert resp.status_code == 200
        parsed = parse_sse_response(resp.text)
        assert parsed == KESTREL_SSE_PARSED

    async def test_multiple_interactions_same_endpoint(self, tmp_path: Path):
        """Multiple calls to the same endpoint replay in order."""
        recorder = CassetteRecorder()

        for i in range(3):
            request = httpx.Request(
                "POST",
                "https://kestrel.nathanpricelab.com/mcp",
                json={"jsonrpc": "2.0", "method": "tools/call", "id": i + 1},
            )
            response = httpx.Response(
                status_code=200,
                text=f'event: message\ndata: {{"jsonrpc":"2.0","id":{i + 1},"result":{{"index":{i}}}}}\n',
            )
            recorder.record_interaction(request, response)

        cassette_path = tmp_path / "multi_cassette.json"
        recorder.save(cassette_path)

        cassette_data = load_cassette(cassette_path)
        replayer = CassetteReplayer(cassette_data)

        # Each replay should return the next interaction in sequence
        for i in range(3):
            req = httpx.Request(
                "POST",
                "https://kestrel.nathanpricelab.com/mcp",
                json={"jsonrpc": "2.0", "method": "tools/call", "id": i + 1},
            )
            resp = replayer.get_response(req)
            assert resp is not None
            parsed = parse_sse_response(resp.text)
            assert parsed["id"] == i + 1
            assert parsed["result"]["index"] == i
