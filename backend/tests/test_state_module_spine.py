"""Tests for the signed-weight data-spine state models (Axis A, Unit 1).

Covers ``MemberWeight``, ``ModuleDirection``, ``ModuleSpine`` and the optional
``module_spine`` field on ``DiscoveryState``.
"""

import pytest
from pydantic import ValidationError

from kestrel_backend.graph.state import (
    DiscoveryState,
    MemberWeight,
    ModuleDirection,
    ModuleSpine,
)


# --- MemberWeight ----------------------------------------------------------


def test_member_weight_happy_path_with_and_without_kim():
    with_kim = MemberWeight(name="glucose", kme=0.82, kim=12.5)
    without_kim = MemberWeight(name="IL6", kme=-0.4)
    assert with_kim.name == "glucose"
    assert with_kim.kme == 0.82
    assert with_kim.kim == 12.5
    assert without_kim.kim is None


def test_member_weight_is_frozen():
    mw = MemberWeight(name="glucose", kme=0.5)
    with pytest.raises(ValidationError):
        mw.kme = 0.9  # type: ignore[misc]


def test_member_weight_kme_boundaries_accepted():
    assert MemberWeight(name="a", kme=-1.0).kme == -1.0
    assert MemberWeight(name="b", kme=1.0).kme == 1.0


def test_member_weight_kim_is_unbounded_nonnegative():
    # kIM is raw kWithin — unbounded, so values well beyond 1 are valid.
    for kim in (0.0, 3.0, 40.0, 999.0):
        assert MemberWeight(name="x", kme=0.1, kim=kim).kim == kim


def test_member_weight_out_of_range_kme_rejected():
    with pytest.raises(ValidationError):
        MemberWeight(name="a", kme=1.5)
    with pytest.raises(ValidationError):
        MemberWeight(name="a", kme=-2.0)


def test_member_weight_negative_kim_rejected():
    with pytest.raises(ValidationError):
        MemberWeight(name="a", kme=0.1, kim=-1.0)


def test_member_weight_requires_kme():
    with pytest.raises(ValidationError):
        MemberWeight(name="a")  # type: ignore[call-arg]


# --- ModuleDirection -------------------------------------------------------


def test_module_direction_happy_path_and_frozen():
    d = ModuleDirection(eigengene_trait_correlation=0.6, trait_label="frailty index")
    assert d.eigengene_trait_correlation == 0.6
    assert d.trait_label == "frailty index"
    with pytest.raises(ValidationError):
        d.trait_label = "other"  # type: ignore[misc]


def test_module_direction_correlation_boundaries_and_out_of_range():
    assert ModuleDirection(eigengene_trait_correlation=-1.0, trait_label="t").eigengene_trait_correlation == -1.0
    assert ModuleDirection(eigengene_trait_correlation=1.0, trait_label="t").eigengene_trait_correlation == 1.0
    with pytest.raises(ValidationError):
        ModuleDirection(eigengene_trait_correlation=1.5, trait_label="t")


# --- ModuleSpine -----------------------------------------------------------


def test_module_spine_happy_path():
    spine = ModuleSpine(
        group="Brown",
        members={
            "glucose": MemberWeight(name="glucose", kme=0.82, kim=12.5),
            "IL6": MemberWeight(name="IL6", kme=-0.4),
        },
        direction=ModuleDirection(eigengene_trait_correlation=0.6, trait_label="frailty"),
    )
    assert spine.group == "Brown"
    assert set(spine.members) == {"glucose", "IL6"}
    assert spine.members["glucose"].kim == 12.5
    assert spine.direction is not None
    # Reserved v2 correlation slot is absent by default.
    assert spine.correlation is None


def test_module_spine_direction_optional():
    spine = ModuleSpine(group="Blue", members={"a": MemberWeight(name="a", kme=0.3)})
    assert spine.direction is None
    assert spine.correlation is None


def test_module_spine_is_frozen():
    spine = ModuleSpine(group="Blue", members={"a": MemberWeight(name="a", kme=0.3)})
    with pytest.raises(ValidationError):
        spine.group = "Red"  # type: ignore[misc]


def test_module_spine_correlation_reserved_must_be_none():
    # The reserved v2 slot only accepts None until the v2 task defines its shape.
    with pytest.raises(ValidationError):
        ModuleSpine(
            group="Blue",
            members={"a": MemberWeight(name="a", kme=0.3)},
            correlation=[[1.0]],  # type: ignore[arg-type]
        )


# --- DiscoveryState.module_spine -------------------------------------------


def test_discovery_state_module_spine_absent_is_valid():
    # total=False TypedDict — absence must be legal (classic / no-kME runs).
    state: DiscoveryState = {"raw_query": "hello"}
    assert "module_spine" not in state


def test_discovery_state_module_spine_present():
    spine = ModuleSpine(group="Brown", members={"g": MemberWeight(name="g", kme=0.5)})
    state: DiscoveryState = {"raw_query": "q", "module_spine": {"Brown": spine}}
    assert state["module_spine"]["Brown"].members["g"].kme == 0.5
