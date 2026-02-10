"""
KRAKEN LangGraph workflow nodes.

Each node is a pure async function: (state: DiscoveryState) -> dict
Nodes should return only the fields they modify - LangGraph handles merging.

Phase 1: intake, entity_resolution, synthesis
Phase 2: + triage, direct_kg, cold_start
Phase 4a: + pathway_enrichment
Phase 4b: + integration, temporal
"""

from . import intake
from . import entity_resolution
from . import triage
from . import direct_kg
from . import cold_start
from . import pathway_enrichment
from . import integration
from . import temporal
from . import synthesis

__all__ = [
    "intake",
    "entity_resolution",
    "triage",
    "direct_kg",
    "cold_start",
    "pathway_enrichment",
    "integration",
    "temporal",
    "synthesis",
]
