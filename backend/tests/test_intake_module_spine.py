"""Tests for the Intake node building `module_spine` from the signed-weight upload path.

Axis A Unit 3: after gate-#2 validation, intake assembles a per-group ``ModuleSpine`` from the
normalized member weights + validated directions, emits it only when kME material is present
(classic/no-kME parity), and emits a coverage summary whose ``metric_computable`` flag is honest
about whether a direction was supplied.
"""

import pytest

from kestrel_backend.graph.nodes import intake
from kestrel_backend.graph.state import ModuleSpine, MemberWeight, ModuleDirection


pytestmark = pytest.mark.asyncio


async def test_structured_panel_with_kme_builds_module_spine():
    state = {
        "raw_query": "",
        "structured_analytes": [
            {"name": "glucose", "group": "Brown", "type": "metabolite", "kme": 0.8, "kim": 12.0},
            {"name": "IL6", "group": "Brown", "type": "protein", "kme": -0.6},
        ],
        "selected_groups": [],
    }
    out = await intake.run(state)

    assert out["raw_entities"] == ["glucose", "IL6"]  # run set unchanged
    spine = out["module_spine"]
    assert set(spine) == {"brown"}
    module = spine["brown"]
    assert isinstance(module, ModuleSpine)
    assert module.group == "Brown"  # display label preserved
    assert set(module.members) == {"glucose", "IL6"}
    assert isinstance(module.members["glucose"], MemberWeight)
    assert module.members["glucose"].kme == 0.8
    assert module.members["glucose"].kim == 12.0
    assert module.members["IL6"].kme == -0.6
    assert module.members["IL6"].kim is None
    assert module.direction is None


async def test_module_spine_carries_direction_and_metric_computable():
    state = {
        "raw_query": "",
        "structured_analytes": [
            {"name": "glucose", "group": "Brown", "kme": 0.8},
        ],
        "selected_groups": [],
        "module_directions": [
            {"group": "Brown", "eigengene_trait_correlation": -0.5, "trait_label": "frailty"},
        ],
    }
    out = await intake.run(state)

    module = out["module_spine"]["brown"]
    assert isinstance(module.direction, ModuleDirection)
    assert module.direction.eigengene_trait_correlation == -0.5
    assert module.direction.trait_label == "frailty"

    cov = out["module_spine_coverage"]
    assert cov["metric_computable"] is True
    assert cov["members_with_kme"] == 1
    assert cov["groups_with_direction"] == 1


async def test_module_spine_without_direction_flags_metric_uncomputable():
    state = {
        "raw_query": "",
        "structured_analytes": [
            {"name": "glucose", "group": "Brown", "kme": 0.8, "kim": 3.0},
            {"name": "IL6", "group": "Brown", "kme": -0.6},
        ],
        "selected_groups": [],
    }
    out = await intake.run(state)

    cov = out["module_spine_coverage"]
    assert cov["metric_computable"] is False
    assert cov["members_with_kme"] == 2
    assert cov["members_with_kim"] == 1
    assert cov["groups_with_direction"] == 0


async def test_member_in_two_groups_gets_distinct_weights():
    state = {
        "raw_query": "",
        "structured_analytes": [
            {"name": "glucose", "group": "Brown", "kme": 0.8},
            {"name": "glucose", "group": "Blue", "kme": 0.2},
        ],
        "selected_groups": [],
    }
    out = await intake.run(state)

    spine = out["module_spine"]
    assert spine["brown"].members["glucose"].kme == 0.8
    assert spine["blue"].members["glucose"].kme == 0.2


async def test_panel_without_kme_has_no_module_spine():
    state = {
        "raw_query": "",
        "structured_analytes": [
            {"name": "glucose", "group": "Brown", "type": "metabolite"},
        ],
        "selected_groups": [],
    }
    out = await intake.run(state)

    assert out["raw_entities"] == ["glucose"]
    assert "module_spine" not in out
    assert "module_spine_coverage" not in out


async def test_classic_query_has_no_module_spine():
    state = {
        "raw_query": "analyze: glucose, fructose, mannose",
        "structured_analytes": [],
        "selected_groups": [],
    }
    out = await intake.run(state)

    assert "module_spine" not in out
    assert "module_spine_coverage" not in out


async def test_rejected_panel_emits_no_partial_module_spine():
    state = {
        "raw_query": "",
        # kME out of range → the shared gate rejects the whole panel.
        "structured_analytes": [
            {"name": "glucose", "group": "Brown", "kme": 1.5},
        ],
        "selected_groups": [],
    }
    out = await intake.run(state)

    assert out.get("upload_rejected") is True
    assert "module_spine" not in out
    assert any("rejected" in e.lower() for e in out["errors"])


async def test_partial_kme_only_weighted_members_in_spine():
    state = {
        "raw_query": "",
        "structured_analytes": [
            {"name": "glucose", "group": "Brown", "kme": 0.8},
            {"name": "IL6", "group": "Brown"},  # no kME → excluded from members
        ],
        "selected_groups": [],
    }
    out = await intake.run(state)

    module = out["module_spine"]["brown"]
    assert set(module.members) == {"glucose"}
    # IL6 still in the run set even though it is not weighted.
    assert "IL6" in out["raw_entities"]
