"""Per-bridge evidence-provenance labeling (A–B–C mediated chains).

Deterministic KG-edge provenance classification (no score, no LLM): see
``provenance.py`` and docs/plans/2026-06-17-002-feat-bridge-evidence-provenance-labeler-plan.md.

The superseded v1 literature co-occurrence scorer and v2 KG-provenance scorer were removed
(neither could support a trustworthy per-bridge mechanism-confidence number — see
backend/assessment_data/BRIDGE_GROUNDING_V2_FINDINGS.md / BRIDGE_GROUNDING_TIER_A_FINDINGS.md
and the retained kg_bridge_* probes).
"""
