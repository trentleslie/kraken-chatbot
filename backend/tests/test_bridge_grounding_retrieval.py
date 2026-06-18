"""U2 — per-leg co-occurrence retrieval (search_pmids wrapper + cooccurrence/dedup helpers).

Network is mocked: search_pmids (the retrieval module's imported alias) is monkeypatched.

Run with: uv run python -m pytest tests/test_bridge_grounding_retrieval.py -v
"""

import pytest

from src.kestrel_backend.bridge_grounding import retrieval
from src.kestrel_backend.bridge_grounding.retrieval import (
    cooccurrence_pmids,
    cooccurrence_query,
    dedupe_co_mention,
    is_curie_like,
)


class _Recorder:
    """Async stand-in for search_pmids that records its call args."""

    def __init__(self, result):
        self.result = result
        self.calls = []

    async def __call__(self, query, retmax=50):
        self.calls.append({"query": query, "retmax": retmax})
        return list(self.result)


# --- query construction ----------------------------------------------------------------

def test_cooccurrence_query_quotes_and_ands():
    assert cooccurrence_query("HPV", "cervical cancer") == '"HPV" AND "cervical cancer"'


# --- is_curie_like ---------------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("NCBIGene:7132", True),
    ("CHEBI:4167", True),
    ("UMLS:C0022818", True),
    ("glucose", False),
    ("type 2 diabetes", False),
    ("E. coli", False),       # dot + space, no CURIE colon
    ("", False),
])
def test_is_curie_like(name, expected):
    assert is_curie_like(name) is expected


# --- cooccurrence_pmids ----------------------------------------------------------------

async def test_cooccurrence_pmids_happy_path(monkeypatch):
    rec = _Recorder(["1", "2", "3"])
    monkeypatch.setattr(retrieval, "search_pmids", rec)
    monkeypatch.setattr(retrieval, "ncbi_api_key_present", lambda: True)
    out = await cooccurrence_pmids("HPV", "cervical cancer")
    assert out == ["1", "2", "3"]
    assert rec.calls[0]["query"] == '"HPV" AND "cervical cancer"'


async def test_cooccurrence_pmids_caps_retmax_at_50(monkeypatch):
    rec = _Recorder([])
    monkeypatch.setattr(retrieval, "search_pmids", rec)
    monkeypatch.setattr(retrieval, "ncbi_api_key_present", lambda: True)
    await cooccurrence_pmids("a", "b", limit=200)
    assert rec.calls[0]["retmax"] == 50


async def test_cooccurrence_pmids_missing_name_returns_empty_no_search(monkeypatch):
    rec = _Recorder(["should-not-be-returned"])
    monkeypatch.setattr(retrieval, "search_pmids", rec)
    assert await cooccurrence_pmids("", "b") == []
    assert await cooccurrence_pmids("a", "") == []
    assert rec.calls == []  # never hit the network


async def test_cooccurrence_pmids_curie_like_name_returns_empty(monkeypatch):
    rec = _Recorder(["x"])
    monkeypatch.setattr(retrieval, "search_pmids", rec)
    # A CURIE-shaped "name" (KG omitted a label) must not be PubMed-searched (O4).
    assert await cooccurrence_pmids("NCBIGene:7132", "cervical cancer") == []
    assert rec.calls == []


async def test_cooccurrence_pmids_warns_when_no_api_key(monkeypatch, caplog):
    rec = _Recorder(["1"])
    monkeypatch.setattr(retrieval, "search_pmids", rec)
    monkeypatch.setattr(retrieval, "ncbi_api_key_present", lambda: False)  # no key
    with caplog.at_level("WARNING"):
        await cooccurrence_pmids("a", "b")
    assert any("NCBI_API_KEY unset" in r.message for r in caplog.records)


# --- dedupe_co_mention -----------------------------------------------------------------

def test_dedupe_co_mention_keep_first_into_a():
    a_kept, b_kept, dropped = dedupe_co_mention(["1", "2", "3"], ["3", "4", "5"])
    assert a_kept == ["1", "2", "3"]   # A unchanged (keeps the co-mention)
    assert b_kept == ["4", "5"]        # overlap removed from B
    assert dropped == 1


def test_dedupe_co_mention_no_overlap():
    a_kept, b_kept, dropped = dedupe_co_mention(["1", "2"], ["3", "4"])
    assert (a_kept, b_kept, dropped) == (["1", "2"], ["3", "4"], 0)


def test_dedupe_co_mention_full_overlap_empties_b():
    a_kept, b_kept, dropped = dedupe_co_mention(["1", "2"], ["1", "2"])
    assert b_kept == [] and dropped == 2
