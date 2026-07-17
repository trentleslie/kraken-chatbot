"""Axis A, Unit 6: the pinned Brown fixture reproduces a stable module_spine.

Reads the pinned measurement-substrate CSVs through the SAME shared R19 gate + intake spine
builder the live upload path uses, and asserts a deterministic module_spine + coverage. This is
the reproducibility guarantee for the sign-inversion-rate metric (the metric eval itself is
Axis E).
"""

import csv
from dataclasses import dataclass
from pathlib import Path

from kestrel_backend.analyte_ingest import validate_and_normalize
from kestrel_backend.graph.nodes.intake import _build_module_spine

_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "assessment_data" / "module_spine"


@dataclass
class _FakeSettings:
    analyte_run_ceiling: int = 200
    analyte_panel_row_cap: int = 10000
    analyte_field_max_len: int = 512


def _load_panel() -> list[dict]:
    with open(_FIXTURE_DIR / "brown_signed_weights.csv", newline="") as fh:
        return [
            {"name": r["analyte"], "group": r["module"], "type": r["type"],
             "kme": r["kME"], "kim": r["kIM"]}
            for r in csv.DictReader(fh)
        ]


def _load_directions() -> list[dict]:
    with open(_FIXTURE_DIR / "brown_module_directions.csv", newline="") as fh:
        return [
            {"group": r["module"],
             "eigengene_trait_correlation": r["eigengene_trait_correlation"],
             "trait_label": r["trait_label"]}
            for r in csv.DictReader(fh)
        ]


def test_pinned_fixture_reproduces_module_spine():
    panel = _load_panel()
    directions = _load_directions()
    normalized = validate_and_normalize(panel, None, _FakeSettings(), directions)
    assert normalized.errors == []

    module_spine, coverage = _build_module_spine(normalized)

    assert set(module_spine) == {"brown", "blue"}

    brown = module_spine["brown"]
    assert brown.group == "Brown"
    assert set(brown.members) == {"glucose", "1-methylnicotinamide", "IL6"}
    assert brown.members["glucose"].kme == 0.82
    assert brown.members["glucose"].kim == 40.1  # kIM > 1 ingests (unbounded kWithin)
    assert brown.direction is not None
    assert brown.direction.eigengene_trait_correlation == 0.61

    blue = module_spine["blue"]
    assert blue.members["KIF6"].kim is None  # blank kIM cell → None
    assert blue.direction is not None
    assert blue.direction.eigengene_trait_correlation == -0.44

    # Coverage: both modules have kME members AND a direction → metric is computable.
    assert coverage["total_members_with_kme"] == 6
    assert coverage["total_members_with_kim"] == 5
    assert coverage["directions_supplied"] == 2
    assert coverage["metric_computable"] is True


def test_pinned_fixture_is_deterministic():
    panel, directions, settings = _load_panel(), _load_directions(), _FakeSettings()
    s1, c1 = _build_module_spine(validate_and_normalize(panel, None, settings, directions))
    s2, c2 = _build_module_spine(validate_and_normalize(panel, None, settings, directions))
    assert s1 == s2  # frozen pydantic models compare by value
    assert c1 == c2
