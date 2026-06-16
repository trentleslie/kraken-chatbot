"""Tests for the biomapper resolution gold-set eval (Unit 5)."""

import json

from kestrel_backend.evals.biomapper_resolution import run_eval

_GOLD = [
    {"name": "TNFRSF1A", "entity_type_hint": "gene", "expected_curie": "NCBIGene:7132"},
    {"name": "LDLR", "entity_type_hint": "gene", "expected_curie": "NCBIGene:3949"},
    {"name": "GH1", "entity_type_hint": "gene", "expected_curie": "NCBIGene:2688", "known_residual": True},
]
_EXPECT = {e["name"]: e["expected_curie"] for e in _GOLD}


def _resolve_correct():
    async def fn(name, hint, base_url=None):
        return {"curie": _EXPECT[name], "tier": "high"}
    return fn


async def _reconcile_echo(r, hint):
    return (r["curie"], "biolink:Gene")


class TestEvaluate:
    async def test_all_correct(self):
        rows = await run_eval.evaluate(_GOLD, base_url=None,
                                       resolve_fn=_resolve_correct(), reconcile_fn=_reconcile_echo)
        s = run_eval.summarize(rows)
        assert s["accuracy"] == 1.0
        assert s["biomapper_hit_rate"] == 1.0
        assert s["mismatched"] == 0
        assert all(r["method"] == "biomapper" for r in rows)

    async def test_residual_excluded_from_adjusted_accuracy(self):
        # GH1 falls back (unresolved); accuracy drops but accuracy_excl_residual stays 1.0.
        async def resolve_miss_gh1(name, hint, base_url=None):
            if name == "GH1":
                return None
            return {"curie": _EXPECT[name], "tier": "high"}
        rows = await run_eval.evaluate(_GOLD, base_url=None,
                                       resolve_fn=resolve_miss_gh1, reconcile_fn=_reconcile_echo)
        s = run_eval.summarize(rows)
        assert s["correct"] == 2 and s["total"] == 3
        assert s["accuracy_excl_residual"] == 1.0  # 2/2 non-residual correct
        assert s["known_residuals"] == 1

    async def test_wrong_species_counts_as_mismatch(self):
        async def resolve_ortholog(name, hint, base_url=None):
            return {"curie": "NCBIGene:397020", "tier": "high"}  # ortholog for everything
        rows = await run_eval.evaluate(_GOLD, base_url=None,
                                       resolve_fn=resolve_ortholog, reconcile_fn=_reconcile_echo)
        s = run_eval.summarize(rows)
        assert s["accuracy"] == 0.0
        assert s["mismatched"] == 3  # resolved but != expected human CURIE

    async def test_degraded_backend_detected(self):
        async def resolve_none(name, hint, base_url=None):
            return None  # throttle signature: nothing resolves
        rows = await run_eval.evaluate(_GOLD, base_url=None,
                                       resolve_fn=resolve_none, reconcile_fn=_reconcile_echo)
        s = run_eval.summarize(rows)
        assert s["degraded_backend_suspected"] is True
        assert s["biomapper_hit_rate"] == 0.0


class TestArtifact:
    def test_artifact_has_pinned_inputs_and_no_key(self):
        rows = [{"name": "X", "entity_type_hint": "gene", "expected_curie": "NCBIGene:1",
                 "resolved_curie": "NCBIGene:1", "method": "biomapper", "correct": True,
                 "known_residual": False}]
        art = run_eval.build_artifact(rows, base_url="https://dev.example/api/v1", env="dev",
                                      ts="20260616T000000Z")
        repro = art["reproduce_inputs"]
        assert repro["biomapper_env"] == "dev"
        assert repro["biomapper_base_url"] == "https://dev.example/api/v1"
        assert "gold_set_sha" in repro and "biomapper_version" in repro
        # The API key must never appear anywhere in the artifact.
        blob = json.dumps(art)
        assert "api_key" not in blob.lower()
        assert "BIOMAPPER_API_KEY" not in blob

    def test_default_artifact_path_is_timestamped(self):
        p = run_eval.default_artifact_path("20260616T010203Z")
        assert p.name == "biomapper_resolution_20260616T010203Z.json"
        assert p.parent.name == "runs"

    def test_gold_set_loads_with_expected_curies(self):
        gold = run_eval.load_gold()
        names = {e["name"] for e in gold}
        assert {"TNFRSF1A", "LDLR", "GH1"} <= names
        for e in gold:
            assert e["expected_curie"].startswith("NCBIGene:")
            assert e["entity_type_hint"] in {"gene", "protein", "metabolite"}
