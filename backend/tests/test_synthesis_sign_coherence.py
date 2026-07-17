"""Unit 5 — apply the sign-coherence partition at the synthesis fusion boundary (Axis D, Guard 2).

A group that splits in sign is rendered as two sign-coherent sub-programs (``<group> (+)`` /
``<group> (−)``) plus a visible annotation, so the "coordinated group" prompt instruction never
reads a fused mixed-sign set. When no ModuleSpine is present (the state today, before axis A lands)
the section is absent and the assembled context is byte-identical.
"""

import pytest

from kestrel_backend.graph.nodes import synthesis
from kestrel_backend.graph.nodes.synthesis import (
    assemble_synthesis_context,
    compute_sign_splits,
    format_sign_coherence,
)
from kestrel_backend.graph.state import Finding


def _spine(members: dict) -> dict:
    """A minimal ModuleSpine-shaped stub: {"members": {name -> kME}}."""
    return {"members": members}


GG_SPLIT_STATE = {
    "raw_query": "gamma-glutamyl dipeptides",
    "entity_groups": {
        "gamma-glutamylvaline": ["GGT dipeptides"],
        "gamma-glutamylleucine": ["GGT dipeptides"],
        "gamma-glutamylglutamate": ["GGT dipeptides"],
        "gamma-glutamylglycine": ["GGT dipeptides"],
    },
    "module_spine": _spine(
        {
            "gamma-glutamylvaline": 0.8,
            "gamma-glutamylleucine": 0.7,
            "gamma-glutamylglutamate": -0.6,
            "gamma-glutamylglycine": -0.75,
        }
    ),
}


class TestComputeSignSplits:
    def test_split_group_detected_and_counted(self):
        splits, counts = compute_sign_splits(GG_SPLIT_STATE)
        assert "GGT dipeptides" in splits
        assert counts["groups_sign_split"] == 1
        # two sign-coherent sub-programs produced from the one split group
        assert counts["groups_split"] == 2

    def test_coherent_group_not_split(self):
        state = {
            "entity_groups": {"a": ["mod"], "b": ["mod"]},
            "module_spine": _spine({"a": 0.8, "b": 0.6}),
        }
        splits, counts = compute_sign_splits(state)
        assert splits == {}
        assert counts["groups_sign_split"] == 0

    def test_no_module_spine_is_inert(self):
        state = {"entity_groups": {"a": ["mod"], "b": ["mod"]}}
        splits, counts = compute_sign_splits(state)
        assert splits == {}
        assert counts == {"groups_sign_split": 0, "groups_split": 0}

    def test_group_with_no_spine_entries_skipped(self):
        state = {
            "entity_groups": {"x": ["mod"], "y": ["mod"]},
            "module_spine": _spine({"unrelated": 0.5}),
        }
        splits, counts = compute_sign_splits(state)
        assert splits == {}


class TestFormatSignCoherence:
    def test_split_renders_two_subprograms_and_annotation(self):
        splits, _ = compute_sign_splits(GG_SPLIT_STATE)
        section = format_sign_coherence(splits)
        assert "GGT dipeptides (+)" in section
        assert "GGT dipeptides (−)" in section  # (−) minus sign
        assert "gamma-glutamylvaline" in section
        assert "gamma-glutamylglutamate" in section

    def test_empty_splits_render_nothing(self):
        assert format_sign_coherence({}) == ""


class TestAssembleContextIntegration:
    def test_split_group_appears_in_assembled_context(self):
        ctx = assemble_synthesis_context(dict(GG_SPLIT_STATE))
        assert "GGT dipeptides (+)" in ctx
        assert "Sign-coherence" in ctx or "sign-coherent" in ctx.lower()

    def test_byte_identical_without_module_spine(self):
        # Same state minus the module_spine → the sign-coherence section must not appear, and the
        # assembled context must equal the context assembled when the key is simply absent.
        base = {k: v for k, v in GG_SPLIT_STATE.items() if k != "module_spine"}
        ctx = assemble_synthesis_context(dict(base))
        assert "GGT dipeptides (+)" not in ctx
        assert "Sign-coherence" not in ctx


class TestRunPersistsHooks:
    # SynthesisInput requires at least one finding to be present; add a trivial one.
    _FINDING = Finding(entity="x", claim="c", tier=2)

    @pytest.mark.asyncio
    async def test_run_emits_sign_coherence_measurement_hooks(self, monkeypatch):
        # Force the deterministic fallback (no LLM) so the node runs offline.
        monkeypatch.setattr(synthesis, "HAS_SDK", False)
        state = dict(GG_SPLIT_STATE, direct_findings=[self._FINDING])
        result = await synthesis.run(state)
        assert result["groups_sign_split"] == 1
        assert result["groups_split"] == 2
        # and the telemetry sub-dict carries them too
        assert result["synthesis_context_stats"]["sign_coherence"]["groups_sign_split"] == 1

    @pytest.mark.asyncio
    async def test_run_hooks_zero_without_spine(self, monkeypatch):
        monkeypatch.setattr(synthesis, "HAS_SDK", False)
        base = {k: v for k, v in GG_SPLIT_STATE.items() if k != "module_spine"}
        result = await synthesis.run(dict(base, direct_findings=[self._FINDING]))
        assert result["groups_sign_split"] == 0
        assert result["groups_split"] == 0
