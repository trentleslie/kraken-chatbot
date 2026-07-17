"""Tests for the Intake node building + emitting module_spine (Axis A, Unit 3)."""

import pytest

from kestrel_backend.graph.nodes import intake
from kestrel_backend.graph.state import ModuleSpine

pytestmark = pytest.mark.asyncio


async def test_structured_panel_with_kme_emits_module_spine():
    state = {
        "raw_query": "",
        "structured_analytes": [
            {"name": "glucose", "group": "Brown", "kme": 0.82, "kim": 12},
            {"name": "IL6", "group": "Brown", "kme": -0.4},
        ],
        "selected_groups": [],
        "module_directions": [
            {"group": "Brown", "eigengene_trait_correlation": 0.6, "trait_label": "frailty"}
        ],
    }
    out = await intake.run(state)
    spine = out["module_spine"]
    assert set(spine) == {"brown"}
    assert isinstance(spine["brown"], ModuleSpine)
    assert spine["brown"].group == "Brown"  # display label preserved
    assert spine["brown"].members["glucose"].kme == 0.82
    assert spine["brown"].members["glucose"].kim == 12.0
    assert spine["brown"].members["IL6"].kim is None
    assert spine["brown"].direction is not None
    assert spine["brown"].direction.eigengene_trait_correlation == 0.6
    assert spine["brown"].direction.trait_label == "frailty"


async def test_panel_without_kme_has_no_module_spine_key():
    state = {
        "raw_query": "",
        "structured_analytes": [
            {"name": "glucose", "group": "Brown"},
            {"name": "IL6", "group": "Blue"},
        ],
        "selected_groups": [],
    }
    out = await intake.run(state)
    assert "module_spine" not in out


async def test_classic_query_has_no_module_spine():
    state = {"raw_query": "what connects glucose and frailty?"}
    out = await intake.run(state)
    assert "module_spine" not in out


async def test_rejected_panel_short_circuits_without_module_spine():
    # Out-of-range kME rejects the whole panel (gate #2, Studio/harness path).
    state = {
        "raw_query": "",
        "structured_analytes": [{"name": "glucose", "group": "Brown", "kme": 5.0}],
        "selected_groups": [],
    }
    out = await intake.run(state)
    assert out["upload_rejected"] is True
    assert "module_spine" not in out


async def test_coverage_summary_counts_and_metric_computable():
    state = {
        "raw_query": "",
        "structured_analytes": [
            {"name": "glucose", "group": "Brown", "kme": 0.82, "kim": 12},
            {"name": "IL6", "group": "Brown", "kme": -0.4},
            {"name": "KIF6", "group": "Blue", "kme": 0.2},
        ],
        "selected_groups": [],
        "module_directions": [
            {"group": "Brown", "eigengene_trait_correlation": 0.6, "trait_label": "frailty"}
        ],
    }
    out = await intake.run(state)
    cov = out["module_spine_coverage"]
    assert cov["total_members_with_kme"] == 3
    assert cov["total_members_with_kim"] == 1
    assert cov["directions_supplied"] == 1
    assert cov["metric_computable"] is True  # Brown has kME members AND a direction
    assert cov["groups"]["brown"]["members_with_kme"] == 2
    assert cov["groups"]["brown"]["direction_supplied"] is True
    assert cov["groups"]["blue"]["direction_supplied"] is False


async def test_coverage_flags_uncomputable_metric_when_no_direction():
    state = {
        "raw_query": "",
        "structured_analytes": [{"name": "glucose", "group": "Brown", "kme": 0.5}],
        "selected_groups": [],
    }
    out = await intake.run(state)
    assert "module_spine" in out
    cov = out["module_spine_coverage"]
    assert cov["metric_computable"] is False
    assert cov["directions_supplied"] == 0
