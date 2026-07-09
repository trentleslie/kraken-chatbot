"""Tests for the Intake node's structured (file-upload) branch (Unit 3, R13/R14/R19)."""

import pytest

from kestrel_backend.graph.nodes import intake


pytestmark = pytest.mark.asyncio


async def test_structured_panel_populates_run_set_and_groups():
    state = {
        "raw_query": "",
        "structured_analytes": [
            {"name": "glucose", "group": "Brown", "type": "metabolite"},
            {"name": "IL6", "group": "Blue", "type": "protein"},
        ],
        "selected_groups": [],
    }
    out = await intake.run(state)
    assert out["query_type"] == "discovery"
    assert out["raw_entities"] == ["glucose", "IL6"]
    assert out["entity_type_hints"] == {"glucose": "metabolite", "IL6": "protein"}
    assert out["entity_groups"] == {"glucose": ["Brown"], "IL6": ["Blue"]}


async def test_selection_filters_run_set():
    state = {
        "raw_query": "",
        "structured_analytes": [
            {"name": "glucose", "group": "Brown"},
            {"name": "IL6", "group": "Blue"},
        ],
        "selected_groups": ["Brown"],
    }
    out = await intake.run(state)
    assert out["raw_entities"] == ["glucose"]


async def test_name_in_two_selected_groups_single_entity_both_groups():
    state = {
        "raw_query": "",
        "structured_analytes": [
            {"name": "glucose", "group": "Brown"},
            {"name": "glucose", "group": "Blue"},
        ],
        "selected_groups": ["Brown", "Blue"],
    }
    out = await intake.run(state)
    assert out["raw_entities"] == ["glucose"]
    assert out["entity_groups"]["glucose"] == ["Brown", "Blue"]


async def test_unrecognized_type_omitted_still_in_entities():
    state = {
        "raw_query": "",
        "structured_analytes": [{"name": "glucose", "type": "lipid"}],
        "selected_groups": [],
    }
    out = await intake.run(state)
    assert out["raw_entities"] == ["glucose"]
    assert out["entity_type_hints"] == {}


async def test_study_context_derived_from_query_not_bypassed():
    state = {
        "raw_query": "Longitudinal 5-year study of type 2 diabetes progression",
        "structured_analytes": [{"name": "glucose", "group": "Brown"}],
        "selected_groups": [],
    }
    out = await intake.run(state)
    assert out["raw_entities"] == ["glucose"]  # from panel, not prose
    assert out["is_longitudinal"] is True
    assert out["duration_years"] == 5
    assert out["study_context"].get("disease_focus") == "type 2 diabetes"


async def test_no_structured_panel_uses_free_text_path():
    state = {
        "raw_query": "analyze: glucose, fructose, mannose",
        "structured_analytes": [],
        "selected_groups": [],
    }
    out = await intake.run(state)
    # Free-text heuristic path: entities parsed from prose, entity_groups absent.
    assert "glucose" in [e.lower() for e in out["raw_entities"]]
    assert "entity_groups" not in out


async def test_over_ceiling_panel_rejected_with_error_channel():
    state = {
        "raw_query": "",
        "structured_analytes": [{"name": f"a{i}"} for i in range(500)],
        "selected_groups": [],
    }
    out = await intake.run(state)
    # Guard clears the run set and surfaces the reason on the errors channel.
    assert out["raw_entities"] == []
    assert out["query_type"] == "discovery"
    assert out.get("errors")
    assert any("rejected" in e.lower() for e in out["errors"])
