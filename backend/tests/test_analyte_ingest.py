"""Tests for the shared analyte validate/normalize helper (R19, Unit 1)."""

from dataclasses import dataclass

import pytest

from kestrel_backend.analyte_ingest import validate_and_normalize


@dataclass
class _FakeSettings:
    analyte_run_ceiling: int = 200
    analyte_panel_row_cap: int = 10000
    analyte_field_max_len: int = 512


def _settings(**kw) -> _FakeSettings:
    return _FakeSettings(**kw)


def test_none_or_empty_panel_returns_empty():
    for panel_input in (None, []):
        result = validate_and_normalize(panel_input, None, _settings())
        assert result.run_analytes == []
        assert result.entity_groups == {}
        assert result.errors == []


def test_happy_path_mixed_panel():
    panel = [
        {"name": "glucose", "group": "Brown", "type": "metabolite"},
        {"name": "IL6", "group": "Blue", "type": "protein"},
        {"name": "KIF6", "group": "Blue", "type": "gene"},
    ]
    result = validate_and_normalize(panel, None, _settings())
    assert result.errors == []
    assert result.run_analytes == ["glucose", "IL6", "KIF6"]
    assert result.entity_groups == {
        "glucose": ["Brown"],
        "IL6": ["Blue"],
        "KIF6": ["Blue"],
    }
    assert result.entity_type_hints == {
        "glucose": "metabolite",
        "IL6": "protein",
        "KIF6": "gene",
    }
    assert result.rows_read == 3


def test_selection_filter_keeps_only_chosen_groups():
    panel = [
        {"name": "glucose", "group": "Brown"},
        {"name": "IL6", "group": "Blue"},
    ]
    result = validate_and_normalize(panel, ["Brown"], _settings())
    assert result.errors == []
    assert result.run_analytes == ["glucose"]
    assert "IL6" not in result.entity_groups


def test_selection_is_case_insensitive():
    panel = [{"name": "glucose", "group": "Brown"}]
    result = validate_and_normalize(panel, ["brown"], _settings())
    assert result.run_analytes == ["glucose"]


def test_same_name_two_selected_groups_kept_as_single_entity_both_groups():
    panel = [
        {"name": "glucose", "group": "Brown"},
        {"name": "glucose", "group": "Blue"},
    ]
    result = validate_and_normalize(panel, ["Brown", "Blue"], _settings())
    assert result.errors == []
    assert result.run_analytes == ["glucose"]  # name-unique
    assert result.entity_groups["glucose"] == ["Brown", "Blue"]


def test_duplicate_name_same_group_collapses():
    panel = [
        {"name": "glucose", "group": "Brown"},
        {"name": "GLUCOSE", "group": "Brown"},  # case-insensitive dup
    ]
    result = validate_and_normalize(panel, None, _settings())
    assert result.run_analytes == ["glucose"]  # first-seen casing preserved
    assert result.entity_groups["glucose"] == ["Brown"]
    assert result.rows_read == 2  # both rows read, one kept


def test_empty_name_rows_skipped():
    panel = [
        {"name": "  ", "group": "Brown"},
        {"name": "glucose", "group": "Brown"},
        {"group": "Brown"},  # no name key at all
    ]
    result = validate_and_normalize(panel, None, _settings())
    assert result.run_analytes == ["glucose"]
    assert result.rows_read == 1  # only the non-empty-name row counts


def test_values_are_trimmed():
    panel = [{"name": "  glucose  ", "group": "  Brown  ", "type": "  metabolite  "}]
    result = validate_and_normalize(panel, None, _settings())
    assert result.run_analytes == ["glucose"]
    assert result.entity_groups == {"glucose": ["Brown"]}
    assert result.entity_type_hints == {"glucose": "metabolite"}


def test_unrecognized_type_omitted_but_row_kept():
    panel = [{"name": "glucose", "type": "lipid"}]
    result = validate_and_normalize(panel, None, _settings())
    assert result.run_analytes == ["glucose"]
    assert result.entity_type_hints == {}  # unrecognized → omitted (R11)


def test_disease_type_not_recognized():
    # disease is deliberately excluded from analyte types (origin decision).
    panel = [{"name": "T2D", "type": "disease"}]
    result = validate_and_normalize(panel, None, _settings())
    assert result.entity_type_hints == {}


def test_row_cap_rejects_oversized_panel():
    panel = [{"name": f"a{i}"} for i in range(11)]
    result = validate_and_normalize(panel, None, _settings(analyte_panel_row_cap=10))
    assert result.errors
    assert result.run_analytes == []


def test_run_ceiling_rejects_and_clears_run_set():
    panel = [{"name": f"a{i}"} for i in range(11)]
    result = validate_and_normalize(panel, None, _settings(analyte_run_ceiling=10))
    assert result.errors
    # Run set cleared so an error-ignoring caller cannot launch an over-ceiling run.
    assert result.run_analytes == []
    assert result.entity_groups == {}


def test_field_over_length_fails_fast():
    panel = [
        {"name": "ok"},
        {"name": "x" * 600},
    ]
    result = validate_and_normalize(panel, None, _settings(analyte_field_max_len=512))
    assert result.errors
    assert result.run_analytes == []


def test_control_char_in_field_rejected():
    panel = [{"name": "gluc\x00ose"}]
    result = validate_and_normalize(panel, None, _settings())
    assert result.errors
    assert result.run_analytes == []


def test_control_char_in_group_rejected():
    panel = [{"name": "glucose", "group": "Br\nown"}]
    result = validate_and_normalize(panel, None, _settings())
    assert result.errors


def test_non_dict_row_rejected():
    panel = ["glucose"]  # type: ignore[list-item]
    result = validate_and_normalize(panel, None, _settings())
    assert result.errors
