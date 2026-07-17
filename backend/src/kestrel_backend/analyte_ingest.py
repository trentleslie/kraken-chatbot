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

import math
import unicodedata
from dataclasses import dataclass, field
from typing import Any

# Recognized analyte-type values. Lowercase to match ``biolink_class_for``; ``disease`` is
# deliberately excluded (origin decision — a disease type inverts Direct-KG analysis).
RECOGNIZED_TYPES = {"metabolite", "protein", "gene"}

# kME (module-eigengene correlation) is bounded to [-1, 1]; a serialized export can round to
# e.g. 1.0000002, so tolerate ±epsilon on the bound and clamp back inside (Axis A).
_KME_EPSILON = 1e-6

# Floor on the module_directions count cap so a panel whose groups were all filtered out (0
# distinct groups) still admits a reasonable number of directions before the per-item loop,
# while remaining bounded for the Studio/harness gate that bypasses the WS byte cap.
_DIRECTIONS_MIN_CAP = 64


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
    # === Signed-weight data spine (Axis A) — raw material for intake to build ModuleSpine ===
    # Keyed by CANONICAL group key (group.strip().lower(), same as the selection filter) so the
    # separately-exported ME-trait direction table joins reliably. ``module_members`` holds only
    # weighted run-set members: group_key -> {canonical_name -> {"name","kme","kim"}}.
    module_members: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    # group_key -> first-seen verbatim group label (ModuleSpine display label).
    module_group_labels: dict[str, str] = field(default_factory=dict)
    # group_key -> {"eigengene_trait_correlation": float, "trait_label": str} (validated).
    module_directions: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Non-fatal surfaced warnings (group-less weights dropped, unmatched directions).
    warnings: list[str] = field(default_factory=list)


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


def _coerce_weight(value: Any) -> tuple[str, float | None]:
    """Coerce a raw kME/kIM/correlation cell to a float.

    Returns a (status, value) pair where status is one of:
      - ``"absent"``  — the cell is missing/blank (member simply carries no weight);
      - ``"value"``   — a finite float was parsed (in ``value``);
      - ``"invalid"`` — present but non-numeric or non-finite (caller rejects the panel).

    A ``bool`` is treated as invalid: ``True``/``False`` are almost certainly a mis-mapped
    column, not a legitimate correlation, and silently coercing them to 1.0/0.0 is exactly the
    kind of silent misdirection this axis exists to prevent.
    """
    if value is None:
        return ("absent", None)
    if isinstance(value, bool):
        return ("invalid", None)
    if isinstance(value, (int, float)):
        v = float(value)
        return ("value", v) if math.isfinite(v) else ("invalid", None)
    s = str(value).strip()
    if s == "":
        return ("absent", None)
    try:
        v = float(s)
    except (TypeError, ValueError):
        return ("invalid", None)
    return ("value", v) if math.isfinite(v) else ("invalid", None)


def validate_and_normalize(
    structured_analytes: list[dict] | None,
    selected_groups: list[str] | None,
    settings: Any,
    module_directions: list[dict] | None = None,
) -> NormalizedPanel:
    """Validate + normalize a structured analyte panel into a bounded run set (R19).

    Args:
        structured_analytes: Full parsed panel — list of ``{name, group?, type?, kme?, kim?}`` dicts.
        selected_groups: Group values to include; empty/None means "all groups".
        settings: A ``Settings`` instance carrying the R19 caps.
        module_directions: Optional per-module eigengene→outcome direction rows
            (``{group, eigengene_trait_correlation, trait_label}``). Validated + count-bounded
            here so both entry-point gates are guarded identically (Axis A).

    Returns:
        A ``NormalizedPanel``. When ``errors`` is non-empty the panel is rejected and the run
        set is empty — callers MUST check ``errors`` before using ``run_analytes``. Signed-weight
        material (``module_members``/``module_group_labels``/``module_directions``) is populated
        only when kME cells / directions are present.
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
        # Clear signed-weight material too — a rejected panel launches nothing.
        panel.module_members = {}
        panel.module_group_labels = {}
        panel.module_directions = {}
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

        # --- Signed-weight ingest (Axis A) ---------------------------------------------------
        # kME (correlation, [-1,1]) is required to admit a member to the spine; kIM (raw kWithin,
        # unbounded/non-negative) rides along optionally. Missing kME = member simply not weighted.
        kme_status, kme_val = _coerce_weight(row.get("kme"))
        if kme_status == "invalid":
            return _reject(f"Analyte 'kME' value at row {idx} is not a finite number.")
        if kme_status == "value":
            assert kme_val is not None  # narrow for type-checkers
            # Epsilon-tolerant [-1, 1] bound: clamp a value within ±epsilon of a bound; reject beyond.
            if kme_val < -1.0 - _KME_EPSILON or kme_val > 1.0 + _KME_EPSILON:
                return _reject(
                    f"Analyte 'kME' value at row {idx} ({kme_val}) is outside the [-1, 1] range."
                )
            kme_val = max(-1.0, min(1.0, kme_val))

            kim_status, kim_val = _coerce_weight(row.get("kim"))
            if kim_status == "invalid":
                return _reject(f"Analyte 'kIM' value at row {idx} is not a finite number.")
            if kim_status == "value":
                assert kim_val is not None
                # kIM is raw intramodular connectivity (kWithin): unbounded, non-negative. Do NOT
                # apply the [-1, 1] bound — only reject genuinely-negative values.
                if kim_val < 0.0:
                    return _reject(
                        f"Analyte 'kIM' value at row {idx} ({kim_val}) is negative; kIM (kWithin) "
                        "must be >= 0."
                    )
            kim_out = kim_val if kim_status == "value" else None

            if group is None:
                # No module bucket exists for a group-less member → surface, don't silently drop.
                panel.warnings.append(
                    f"Analyte {canonical!r} at row {idx} has a kME but no group; weight dropped."
                )
            else:
                group_key = group.strip().lower()
                panel.module_group_labels.setdefault(group_key, group)
                members = panel.module_members.setdefault(group_key, {})
                existing = members.get(canonical)
                if existing is not None:
                    # Idempotent when identical; a non-equal repeat is a mis-map/duplicate footgun.
                    if existing["kme"] != kme_val or existing["kim"] != kim_out:
                        return _reject(
                            f"Conflicting kME/kIM for analyte {canonical!r} in group {group!r}: "
                            f"({existing['kme']}, {existing['kim']}) vs ({kme_val}, {kim_out})."
                        )
                else:
                    members[canonical] = {"name": canonical, "kme": kme_val, "kim": kim_out}

    # --- Module-direction validation (Axis A) ------------------------------------------------
    # Bound the count BEFORE per-item work so the Studio/harness gate (no WS byte cap) is bounded
    # identically. Cap = distinct panel groups (or a small constant floor).
    if module_directions:
        if not isinstance(module_directions, list):
            return _reject("module_directions must be a list.")
        direction_cap = max(len(panel.module_group_labels), _DIRECTIONS_MIN_CAP)
        if len(module_directions) > direction_cap:
            return _reject(
                f"module_directions has {len(module_directions)} rows, exceeding the "
                f"{direction_cap}-direction limit for this panel."
            )
        for d_idx, entry in enumerate(module_directions):
            if not isinstance(entry, dict):
                return _reject(f"module_directions row {d_idx} is not an object.")

            corr_status, corr_val = _coerce_weight(entry.get("eigengene_trait_correlation"))
            if corr_status != "value" or corr_val is None:
                return _reject(
                    f"module_directions row {d_idx} has a missing/non-numeric "
                    "eigengene_trait_correlation."
                )
            if corr_val < -1.0 - _KME_EPSILON or corr_val > 1.0 + _KME_EPSILON:
                return _reject(
                    f"module_directions row {d_idx} eigengene_trait_correlation ({corr_val}) is "
                    "outside the [-1, 1] range."
                )
            corr_val = max(-1.0, min(1.0, corr_val))

            trait_label = _clean_field(entry.get("trait_label"))
            if trait_label is None:
                return _reject(f"module_directions row {d_idx} has a blank trait_label.")
            if len(trait_label) > field_max:
                return _reject(
                    f"module_directions row {d_idx} trait_label exceeds {field_max} characters."
                )
            if _has_control_chars(trait_label):
                return _reject(
                    f"module_directions row {d_idx} trait_label contains control characters."
                )

            d_group = _clean_field(entry.get("group"))
            if d_group is None:
                panel.warnings.append(
                    f"module_directions row {d_idx} has no group; direction dropped."
                )
                continue
            d_key = d_group.strip().lower()
            if d_key not in panel.module_group_labels:
                available = sorted(panel.module_group_labels.values())
                panel.warnings.append(
                    f"module_directions row {d_idx} group {d_group!r} matches no panel group "
                    f"(available: {available}); direction dropped."
                )
                continue

            new_direction = {
                "eigengene_trait_correlation": corr_val,
                "trait_label": trait_label,
            }
            existing_dir = panel.module_directions.get(d_key)
            if existing_dir is not None and existing_dir != new_direction:
                return _reject(
                    f"Conflicting module direction for group {d_group!r}: "
                    f"{existing_dir} vs {new_direction}."
                )
            panel.module_directions[d_key] = new_direction

    # Distinct-name run ceiling (R19).
    if len(panel.run_analytes) > run_ceiling:
        return _reject(
            f"Selected analyte set has {len(panel.run_analytes)} distinct analytes, exceeding "
            f"the {run_ceiling}-analyte run ceiling. Narrow the selected group(s) or reduce the panel."
        )

    return panel
