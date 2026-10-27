"""Axis A Unit 6: the pinned Brown module upload reproduces module_spine deterministically and the
coverage summary reports metric_computable = true (the sign-inversion metric substrate exists).

Guards the measurement substrate in ``backend/assessment_data/module_spine/`` so a regression that
silently drops kME/kIM ingestion or the direction join is caught here (not only in the eval axis).
"""

import csv
from pathlib import Path

from kestrel_backend.analyte_ingest import validate_and_normalize
from kestrel_backend.config import get_settings
from kestrel_backend.graph.nodes.intake import _build_module_spine

_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "assessment_data" / "module_spine"


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _load_panel():
    members = [
        {
            "name": r["analyte"],
            "group": r["module"],
            "type": r["type"],
            "kme": r["kME"] or None,
            "kim": r["kIM"] or None,
        }
        for r in _read_rows(_FIXTURE_DIR / "brown_module_members.csv")
    ]
    directions = _read_rows(_FIXTURE_DIR / "brown_module_directions.csv")
    return validate_and_normalize(members, [], get_settings(), module_directions=directions)


def test_pinned_upload_builds_expected_module_spine():
    panel = _load_panel()
    assert not panel.errors

    spine, coverage = _build_module_spine(panel)

    # Two weighted modules (Brown, Blue); Grey has a blank-kME member only, so it is absent.
    assert set(spine) == {"brown", "blue"}
    assert spine["brown"].group == "Brown"
    assert set(spine["brown"].members) == {"Glucose", "Insulin", "IL6", "CRP", "Leptin"}
    assert spine["brown"].members["IL6"].kme == -0.61
    assert spine["brown"].members["Glucose"].kim == 14.3
    # Direction joined across the separately-exported ME-trait table.
    assert spine["brown"].direction is not None
    assert spine["brown"].direction.trait_label == "frailty"
    assert spine["brown"].direction.eigengene_trait_correlation == -0.52


def test_pinned_upload_metric_is_computable():
    panel = _load_panel()
    _, coverage = _build_module_spine(panel)

    assert coverage["metric_computable"] is True
    assert coverage["members_with_kme"] == 9  # 5 Brown + 4 Blue
    assert coverage["groups_with_direction"] == 2


def test_pinned_upload_is_deterministic():
    """Running the pinned input twice yields identical spine + coverage (reproducibility SOP)."""
    spine_a, cov_a = _build_module_spine(_load_panel())
    spine_b, cov_b = _build_module_spine(_load_panel())

    assert cov_a == cov_b
    assert spine_a.keys() == spine_b.keys()
    for key in spine_a:
        assert spine_a[key].model_dump() == spine_b[key].model_dump()
