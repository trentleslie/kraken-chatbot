"""Tests for signed-weight (kME/kIM) + module-direction ingest in the shared R19 helper.

Axis A, Unit 2. Extends ``validate_and_normalize`` / ``NormalizedPanel`` to parse the new
numeric per-row columns and the optional per-module direction list, with fail-fast
range/type validation and per-(name, group) accretion. The helper NEVER raises — errors
are RETURNED on ``NormalizedPanel.errors``.
"""

from dataclasses import dataclass

from kestrel_backend.analyte_ingest import validate_and_normalize


@dataclass
class _FakeSettings:
    analyte_run_ceiling: int = 200
    analyte_panel_row_cap: int = 10000
    analyte_field_max_len: int = 512


def _settings(**kw) -> _FakeSettings:
    return _FakeSettings(**kw)


# --- happy paths -----------------------------------------------------------


def test_kme_and_kim_populate_per_group_member_weights():
    panel = [
        {"name": "glucose", "group": "Brown", "kme": 0.82, "kim": 40},
        {"name": "IL6", "group": "Brown", "kme": -0.4},
    ]
    result = validate_and_normalize(panel, None, _settings())
    assert result.errors == []
    assert result.run_analytes == ["glucose", "IL6"]  # run set unchanged
    members = result.module_members["brown"]
    assert members["glucose"]["kme"] == 0.82
    assert members["glucose"]["kim"] == 40.0  # kIM > 1 accepted (unbounded)
    assert members["IL6"]["kme"] == -0.4
    assert members["IL6"]["kim"] is None
    assert result.module_group_labels["brown"] == "Brown"


def test_member_in_two_groups_gets_distinct_weight_per_group():
    panel = [
        {"name": "glucose", "group": "Brown", "kme": 0.5},
        {"name": "glucose", "group": "Blue", "kme": -0.7},
    ]
    result = validate_and_normalize(panel, None, _settings())
    assert result.errors == []
    assert result.module_members["brown"]["glucose"]["kme"] == 0.5
    assert result.module_members["blue"]["glucose"]["kme"] == -0.7


def test_case_variant_name_attaches_under_canonical_run_set_name():
    panel = [
        {"name": "Glucose", "group": "Brown", "kme": 0.5},
        {"name": "glucose", "group": "Brown", "kme": 0.5},  # idempotent duplicate
    ]
    result = validate_and_normalize(panel, None, _settings())
    assert result.errors == []
    # Canonical first-seen name "Glucose" (byte-identical to entity_groups key), not "glucose".
    assert "Glucose" in result.entity_groups
    assert set(result.module_members["brown"]) == {"Glucose"}


def test_no_kme_column_yields_no_module_material():
    panel = [{"name": "glucose", "group": "Brown"}, {"name": "IL6", "group": "Brown"}]
    result = validate_and_normalize(panel, None, _settings())
    assert result.errors == []
    assert result.module_members == {}
    assert result.run_analytes == ["glucose", "IL6"]


def test_partial_panel_only_weighted_members_appear():
    panel = [
        {"name": "glucose", "group": "Brown", "kme": 0.5},
        {"name": "IL6", "group": "Brown"},  # blank kME
    ]
    result = validate_and_normalize(panel, None, _settings())
    assert result.errors == []
    assert set(result.module_members["brown"]) == {"glucose"}
    assert result.run_analytes == ["glucose", "IL6"]  # both still in run set


def test_kme_boundaries_and_epsilon_clamp():
    panel = [
        {"name": "a", "group": "G", "kme": -1.0},
        {"name": "b", "group": "G", "kme": 1.0},
        {"name": "c", "group": "G", "kme": 1.0000002},  # within epsilon → clamps to 1.0
    ]
    result = validate_and_normalize(panel, None, _settings())
    assert result.errors == []
    members = result.module_members["g"]
    assert members["a"]["kme"] == -1.0
    assert members["b"]["kme"] == 1.0
    assert members["c"]["kme"] == 1.0


# --- error paths -----------------------------------------------------------


def test_kme_out_of_range_rejected():
    panel = [{"name": "a", "group": "G", "kme": 1.5}]
    result = validate_and_normalize(panel, None, _settings())
    assert result.errors  # rejected
    assert result.run_analytes == []  # run set cleared, no partial launch


def test_non_numeric_kme_rejected():
    panel = [{"name": "a", "group": "G", "kme": "high"}]
    result = validate_and_normalize(panel, None, _settings())
    assert result.errors
    assert result.run_analytes == []


def test_negative_kim_rejected():
    panel = [{"name": "a", "group": "G", "kme": 0.1, "kim": -1}]
    result = validate_and_normalize(panel, None, _settings())
    assert result.errors
    assert result.run_analytes == []


def test_non_finite_kme_rejected():
    panel = [{"name": "a", "group": "G", "kme": float("inf")}]
    result = validate_and_normalize(panel, None, _settings())
    assert result.errors


def test_conflicting_kme_for_same_name_group_rejected():
    panel = [
        {"name": "a", "group": "G", "kme": 0.5},
        {"name": "a", "group": "G", "kme": 0.6},  # non-equal → conflict
    ]
    result = validate_and_normalize(panel, None, _settings())
    assert result.errors
    assert result.run_analytes == []


def test_groupless_kme_dropped_with_warning():
    panel = [{"name": "a", "kme": 0.5}]  # no group
    result = validate_and_normalize(panel, None, _settings())
    assert result.errors == []  # run proceeds
    assert result.run_analytes == ["a"]
    assert result.module_members == {}
    assert any("group" in w.lower() for w in result.warnings)


# --- module directions -----------------------------------------------------


def test_direction_joins_group_despite_casing_drift():
    panel = [{"name": "a", "group": "Brown Module", "kme": 0.5}]
    directions = [
        {"group": "brown module", "eigengene_trait_correlation": 0.6, "trait_label": "frailty"}
    ]
    result = validate_and_normalize(panel, None, _settings(), directions)
    assert result.errors == []
    d = result.module_directions["brown module"]
    assert d["eigengene_trait_correlation"] == 0.6
    assert d["trait_label"] == "frailty"


def test_direction_out_of_range_correlation_rejected():
    panel = [{"name": "a", "group": "G", "kme": 0.5}]
    directions = [{"group": "G", "eigengene_trait_correlation": 1.4, "trait_label": "t"}]
    result = validate_and_normalize(panel, None, _settings(), directions)
    assert result.errors


def test_direction_overlength_trait_label_rejected():
    panel = [{"name": "a", "group": "G", "kme": 0.5}]
    directions = [{"group": "G", "eigengene_trait_correlation": 0.5, "trait_label": "x" * 5}]
    result = validate_and_normalize(panel, None, _settings(analyte_field_max_len=4), directions)
    assert result.errors


def test_direction_blank_trait_label_rejected():
    panel = [{"name": "a", "group": "G", "kme": 0.5}]
    directions = [{"group": "G", "eigengene_trait_correlation": 0.5, "trait_label": "  "}]
    result = validate_and_normalize(panel, None, _settings(), directions)
    assert result.errors


def test_direction_control_char_trait_label_rejected():
    panel = [{"name": "a", "group": "G", "kme": 0.5}]
    directions = [{"group": "G", "eigengene_trait_correlation": 0.5, "trait_label": "bad\x00label"}]
    result = validate_and_normalize(panel, None, _settings(), directions)
    assert result.errors


def test_direction_for_unknown_group_warn_and_dropped():
    panel = [{"name": "a", "group": "G", "kme": 0.5}]
    directions = [{"group": "NOPE", "eigengene_trait_correlation": 0.5, "trait_label": "t"}]
    result = validate_and_normalize(panel, None, _settings(), directions)
    assert result.errors == []  # run proceeds
    assert result.module_directions == {}
    assert any("nope" in w.lower() for w in result.warnings)


def test_direction_count_cap_rejects_before_per_item():
    panel = [{"name": "a", "group": "G", "kme": 0.5}]
    # Far more directions than distinct groups AND beyond the constant floor.
    directions = [
        {"group": f"G{i}", "eigengene_trait_correlation": 0.5, "trait_label": "t"}
        for i in range(500)
    ]
    result = validate_and_normalize(panel, None, _settings(), directions)
    assert result.errors


# --- idempotence (dual-gate) -----------------------------------------------


def test_helper_is_idempotent_on_weights():
    panel = [
        {"name": "glucose", "group": "Brown", "kme": 0.82, "kim": 12},
        {"name": "IL6", "group": "Blue", "kme": -0.4},
    ]
    directions = [{"group": "Brown", "eigengene_trait_correlation": 0.6, "trait_label": "t"}]
    r1 = validate_and_normalize(panel, None, _settings(), directions)
    r2 = validate_and_normalize(panel, None, _settings(), directions)
    assert r1.module_members == r2.module_members
    assert r1.module_directions == r2.module_directions
    assert r1.errors == r2.errors == []
