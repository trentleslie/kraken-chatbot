"""
LangGraph-based KRAKEN Discovery Workflow

This module implements a multi-node graph workflow for knowledge graph exploration,
replacing the monolithic single-turn agent with an explicit state machine.

Architecture: Intake → Entity Resolution → Triage & Route → [Direct KG | Cold-Start] 
              → Pathway Enrichment → Integration → [Temporal] → Synthesis
"""

from .builder import build_discovery_graph
from .state import DiscoveryState, EntityResolution

__all__ = ["build_discovery_graph", "DiscoveryState", "EntityResolution"]
