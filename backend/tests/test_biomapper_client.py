"""Tests for the Biomapper pre-resolver wrapper (Unit 2).

The wrapper lazy-imports ``biomapper`` inside ``resolve_entity``; we patch
``biomapper.BioMapperClient`` with a fake async context manager and use REAL
``MappingResult`` objects so the ``confidence_tier`` @property (dropped by ``model_dump()``)
is exercised exactly as in production.
"""

import logging

import pytest
from biomapper import (
    BioMapperAuthError,
    BioMapperError,
    BioMapperRateLimitError,
    MappingResult,
)

from kestrel_backend import biomapper_client
from kestrel_backend.biomapper_client import biolink_class_for, resolve_entity


class _FakeClient:
    """Async-context-manager stand-in for BioMapperClient; records the last map_entity call."""

    last_call: dict = {}

    def __init__(self, result=None, exc=None):
        self._result = result
        self._exc = exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def map_entity(self, name, entity_type, **kwargs):
        _FakeClient.last_call = {"name": name, "entity_type": entity_type}
        if self._exc is not None:
            raise self._exc
        return self._result


def _install(monkeypatch, *, result=None, exc=None):
    """Patch biomapper.BioMapperClient to yield a fake returning `result` or raising `exc`."""
    _FakeClient.last_call = {}
    monkeypatch.setattr(
        "biomapper.BioMapperClient",
        lambda **kwargs: _FakeClient(result=result, exc=exc),
    )


def _result(curie="NCBIGene:7132", score=2.5, resolved=True, xrefs=None):
    return MappingResult(
        query_name="TNFRSF1A",
        resolved=resolved,
        primary_curie=curie,
        confidence_score=score,
        identifiers={},
        kg_equivalent_ids=xrefs or {"HGNC": ["HGNC:11916"], "NCBIGene": ["NCBIGene:7132"]},
    )


class TestBiolinkMapping:
    def test_known_hints(self):
        assert biolink_class_for("gene") == "biolink:Gene"
        assert biolink_class_for("protein") == "biolink:Protein"
        assert biolink_class_for("metabolite") == "biolink:SmallMolecule"
        assert biolink_class_for("GENE") == "biolink:Gene"  # case-insensitive

    def test_unknown_or_missing_hint(self):
        assert biolink_class_for(None) is None
        assert biolink_class_for("") is None
        assert biolink_class_for("disease") is None


class TestResolveHappyPath:
    async def test_high_tier_returns_dict(self, monkeypatch):
        _install(monkeypatch, result=_result(score=2.5))
        out = await resolve_entity("TNFRSF1A", "gene")
        assert out is not None
        assert out["curie"] == "NCBIGene:7132"
        assert out["tier"] == "high"
        assert out["xrefs"]["HGNC"] == ["HGNC:11916"]
        # The correct Biolink class was passed to Biomapper.
        assert _FakeClient.last_call["entity_type"] == "biolink:Gene"

    async def test_medium_tier_accepted(self, monkeypatch):
        _install(monkeypatch, result=_result(score=1.5))  # >=1.0 -> medium
        out = await resolve_entity("TNFRSF1A", "gene")
        assert out is not None and out["tier"] == "medium"

    async def test_confidence_tier_read_via_attribute_not_model_dump(self, monkeypatch):
        # Guards the load-bearing gotcha: model_dump() drops the computed tier.
        res = _result(score=2.5)
        assert "confidence_tier" not in res.model_dump()
        assert res.confidence_tier == "high"
        _install(monkeypatch, result=res)
        out = await resolve_entity("TNFRSF1A", "gene")
        assert out is not None and out["tier"] == "high"


class TestResolveGateAndSkips:
    async def test_low_tier_returns_none(self, monkeypatch):
        _install(monkeypatch, result=_result(score=0.5))  # <1.0 -> low, below medium gate
        assert await resolve_entity("TNFRSF1A", "gene") is None

    async def test_unknown_tier_none_score_returns_none(self, monkeypatch):
        _install(monkeypatch, result=_result(score=None))
        assert await resolve_entity("TNFRSF1A", "gene") is None

    async def test_unknown_hint_skips_biomapper(self, monkeypatch):
        _install(monkeypatch, result=_result())
        assert await resolve_entity("whatever", None) is None
        # Biomapper must not have been called at all.
        assert _FakeClient.last_call == {}

    async def test_not_resolved_returns_none(self, monkeypatch):
        _install(monkeypatch, result=_result(resolved=False))
        assert await resolve_entity("TNFRSF1A", "gene") is None

    async def test_malformed_curie_returns_none(self, monkeypatch):
        _install(monkeypatch, result=_result(curie="not a curie!!", score=2.5))
        assert await resolve_entity("TNFRSF1A", "gene") is None


class TestErrorTaxonomy:
    async def test_rate_limit_returns_none(self, monkeypatch):
        _install(monkeypatch, exc=BioMapperRateLimitError("slow down"))
        assert await resolve_entity("TNFRSF1A", "gene") is None

    async def test_base_error_returns_none(self, monkeypatch):
        _install(monkeypatch, exc=BioMapperError("boom"))
        assert await resolve_entity("TNFRSF1A", "gene") is None

    async def test_transport_error_returns_none(self, monkeypatch):
        _install(monkeypatch, exc=RuntimeError("connection reset"))
        assert await resolve_entity("TNFRSF1A", "gene") is None

    async def test_auth_error_propagates(self, monkeypatch):
        _install(monkeypatch, exc=BioMapperAuthError("bad key"))
        with pytest.raises(BioMapperAuthError):
            await resolve_entity("TNFRSF1A", "gene")


class TestSecretHygiene:
    async def test_api_key_never_logged(self, monkeypatch, caplog):
        sentinel = "SENTINEL_BIOMAPPER_KEY_should_never_appear"
        monkeypatch.setenv("BIOMAPPER_API_KEY", sentinel)
        from kestrel_backend.config import get_settings

        get_settings.cache_clear()
        try:
            _install(monkeypatch, exc=BioMapperError("boom"))  # forces a fallback log line
            with caplog.at_level(logging.DEBUG, logger=biomapper_client.logger.name):
                await resolve_entity("TNFRSF1A", "gene")
            assert sentinel not in caplog.text
        finally:
            get_settings.cache_clear()
