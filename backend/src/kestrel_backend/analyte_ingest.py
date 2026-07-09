"""Shared validate-and-normalize helper for structured analyte panels (R19).

This module is the single source of R19 truth. It is invoked at BOTH the WebSocket
handler (``main.py``) and the runner/intake boundary (``graph/nodes/intake.py``) so that
every entry path — including LangGraph Studio and the assessment harnesses that call the
runner directly, bypassing ``main.py`` — is bounded identically.

Given a raw structured-analyte panel (``[{name, group?, type?}, ...]``) plus a selected-group
set, it:
  - enforces the full-panel row cap (before any per-row work);
  - trims + fail-fast validates each row's ``name``/``group``/``type`` for length and control
    characters (R19), skipping rows with an empty analyte name (R9);
  - forms the run set = analytes whose group is in the selection (or all if no selection);
  - de-duplicates the run set to distinct names (case-insensitive), preserving cross-group
    membership in ``entity_groups`` (name -> [groups]) (R10/R14);
  - enforces the distinct-name run ceiling.

Errors are RETURNED (not raised) so callers surface them as an ``ErrorMessage`` / state error
channel rather than crashing the socket or the graph. Design mirrors the "surface degradation,
never except-swallow" learning.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Any

# Recognized analyte-type values. Lowercase to match ``biolink_class_for``; ``disease`` is
# deliberately excluded (origin decision — a disease type inverts Direct-KG analysis).
RECOGNIZED_TYPES = {"metabolite", "protein", "gene"}


@dataclass
class NormalizedPanel:
    """Result of validating + normalizing a structured analyte panel.

    Attributes:
        run_analytes: Distinct-name run set (verbatim post-trim names, order-preserving).
        entity_groups: name -> [group, ...] membership map for the run set (R14 seam).
        entity_type_hints: name -> recognized type, only for rows whose type maps (R11).
        rows_read: Count of input rows after the row cap (pre-dedup, pre-empty-skip).
        errors: Fatal validation errors; when non-empty the panel is rejected.
    """

    run_analytes: list[str] = field(default_factory=list)
    entity_groups: dict[str, list[str]] = field(default_factory=dict)
    entity_type_hints: dict[str, str] = field(default_factory=dict)
    rows_read: int = 0
    errors: list[str] = field(default_factory=list)


def _has_control_chars(value: str) -> bool:
    """True if the string contains disallowed control characters.

    Tabs/newlines and other C0/C1 control chars have no place in an analyte name/group/type and
    are a classic vector for smuggling structure into a downstream prompt or query. Category "Cc"
    covers the C0/C1 control range; ordinary printable text (incl. unicode letters) passes.
    """
    return any(unicodedata.category(ch) == "Cc" for ch in value)


def _clean_field(value: Any) -> str | None:
    """Trim a mapped value to a string, or None if it is absent/blank after trimming."""
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def validate_and_normalize(
    structured_analytes: list[dict] | None,
    selected_groups: list[str] | None,
    settings: Any,
) -> NormalizedPanel:
    """Validate + normalize a structured analyte panel into a bounded run set (R19).

    Args:
        structured_analytes: Full parsed panel — list of ``{name, group?, type?}`` dicts.
        selected_groups: Group values to include; empty/None means "all groups".
        settings: A ``Settings`` instance carrying the R19 caps.

    Returns:
        A ``NormalizedPanel``. When ``errors`` is non-empty the panel is rejected and the run
        set is empty — callers MUST check ``errors`` before using ``run_analytes``.
    """
    panel = NormalizedPanel()

    if not structured_analytes:
        return panel

    def _reject(message: str) -> NormalizedPanel:
        """Record a fatal error and clear the run set so no partial panel can launch a run."""
        panel.errors.append(message)
        panel.run_analytes = []
        panel.entity_groups = {}
        panel.entity_type_hints = {}
        return panel

    row_cap = getattr(settings, "analyte_panel_row_cap", 10000)
    run_ceiling = getattr(settings, "analyte_run_ceiling", 200)
    field_max = getattr(settings, "analyte_field_max_len", 512)

    # Full-panel row cap BEFORE per-row work (bounds worst-case iteration).
    if len(structured_analytes) > row_cap:
        return _reject(
            f"Uploaded panel has {len(structured_analytes)} rows, exceeding the "
            f"{row_cap}-row limit. Please reduce the file."
        )

    # Defensive: this shared R19 gate is also called off the runner/intake path (not just the
    # validated WS handler), so tolerate a malformed selection (non-list, or non-string members)
    # rather than raising TypeError and killing the caller. A bad selection degrades to "all".
    _selected = selected_groups if isinstance(selected_groups, list) else []
    selection = {g.strip().lower() for g in _selected if isinstance(g, str) and g.strip()}

    # key (lowercased name) -> canonical first-seen name, so cross-group membership accretes
    # under one stable key without an O(n) scan per row.
    key_to_canonical: dict[str, str] = {}

    for idx, row in enumerate(structured_analytes):
        if not isinstance(row, dict):
            return _reject(f"Analyte row {idx} is not an object.")

        name = _clean_field(row.get("name"))
        if name is None:
            # R9: skip rows with an empty analyte-name cell (not an error).
            continue

        group = _clean_field(row.get("group"))
        type_ = _clean_field(row.get("type"))

        # Fail-fast per-field length + control-char validation (R19).
        for label, val in (("name", name), ("group", group), ("type", type_)):
            if val is None:
                continue
            if len(val) > field_max:
                return _reject(
                    f"Analyte {label!r} value at row {idx} exceeds {field_max} characters."
                )
            if _has_control_chars(val):
                return _reject(
                    f"Analyte {label!r} value at row {idx} contains control characters."
                )

        panel.rows_read += 1

        # Group selection filter: keep the row only if it matches the selection (or no selection).
        if selection and (group is None or group.lower() not in selection):
            continue

        key = name.lower()
        if key not in key_to_canonical:
            # First sighting of this distinct name → run-set entry (verbatim post-trim name).
            key_to_canonical[key] = name
            panel.run_analytes.append(name)
            panel.entity_groups[name] = []
            # Type hint keyed by the verbatim run-set name (byte-identical to raw_entities).
            if type_ is not None and type_.lower() in RECOGNIZED_TYPES:
                panel.entity_type_hints[name] = type_.lower()

        # Preserve cross-group membership under the FIRST-seen canonical name (R10/R14).
        canonical = key_to_canonical[key]
        if group is not None and group not in panel.entity_groups[canonical]:
            panel.entity_groups[canonical].append(group)

    # Distinct-name run ceiling (R19).
    if len(panel.run_analytes) > run_ceiling:
        return _reject(
            f"Selected analyte set has {len(panel.run_analytes)} distinct analytes, exceeding "
            f"the {run_ceiling}-analyte run ceiling. Narrow the selected group(s) or reduce the panel."
        )

    return panel
