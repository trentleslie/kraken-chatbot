"""
Graph Builder: Construct the LangGraph workflow.

This module defines the graph structure and edge connections for the
KRAKEN discovery workflow.

Phase 1 (Prototype): Intake -> Entity Resolution -> Synthesis
Phase 2: + Triage -> [Direct KG | Cold-Start] -> Synthesis
Phase 4a: + Pathway Enrichment (after analysis branches converge)
Phase 4b: + Integration (bridges + gaps) + Temporal (conditional for longitudinal)

Full 11-node architecture (ground-before-synthesis):
    Intake -> Entity Resolution -> Triage -> [Direct KG | Cold-Start]
           -> Pathway Enrichment -> Integration -> [Temporal]
           -> Hypothesis Extraction -> Literature Grounding -> Synthesis
"""

from typing import Literal
from langgraph.graph import StateGraph, END
from .state import DiscoveryState
from .nodes import (
    intake, entity_resolution, triage, direct_kg, cold_start,
    pathway_enrichment, integration, temporal, hypothesis_extraction,
    synthesis, literature_grounding
)


def route_after_triage(state: DiscoveryState) -> list[str] | str:
    """
    Deterministic routing based on novelty classifications.

    Returns:
        - ["direct_kg", "cold_start"] for parallel execution when both branches needed
        - "direct_kg" when only well-characterized/moderate entities exist
        - "cold_start" when only sparse/cold-start entities exist
        - "pathway_enrichment" when no entities were resolved (skip analysis)
    """
    has_well_char = bool(
        state.get("well_characterized_curies") or
        state.get("moderate_curies")
    )
    has_sparse = bool(
        state.get("sparse_curies") or
        state.get("cold_start_curies")
    )

    if has_well_char and has_sparse:
        # Both branches execute in parallel (same superstep)
        return ["direct_kg", "cold_start"]
    elif has_well_char:
        return "direct_kg"
    elif has_sparse:
        return "cold_start"
    else:
        # No entities to analyze - skip to pathway_enrichment
        # (which will handle empty input gracefully)
        return "pathway_enrichment"


def route_after_integration(state: DiscoveryState) -> str:
    """
    Conditional routing after integration node.

    The temporal node only runs for longitudinal studies. Both the temporal branch and the
    non-temporal branch lead into hypothesis_extraction (then literature_grounding, then
    synthesis) in the ground-before-synthesis topology.

    Returns:
        "temporal" if is_longitudinal is True
        "hypothesis_extraction" otherwise
    """
    if state.get("is_longitudinal", False):
        return "temporal"
    return "hypothesis_extraction"


def build_discovery_graph() -> StateGraph:
    """
    Build the KRAKEN discovery workflow graph.

    Phase 5b: 11-node pipeline with conditional parallel branches, temporal routing,
    and ground-before-synthesis ordering (hypotheses are grounded with literature BEFORE
    synthesis writes the report).

    Nodes:
    - Intake: Parse query, extract entities, detect mode
    - Entity Resolution: Resolve names to CURIEs via Kestrel
    - Triage: Count edges, classify entities by KG connectivity
    - Direct KG: Analyze well-characterized entities (parallel branch)
    - Cold-Start: Analyze sparse/unknown entities (parallel branch)
    - Pathway Enrichment: Find shared neighbors across all entities
    - Integration: Cross-type bridge detection + gap analysis
    - Temporal: Classify findings by temporal relationship (conditional)
    - Hypothesis Extraction: Validate bridges, build structured hypotheses from findings
    - Literature Grounding: Add Semantic Scholar / OpenAlex / PubMed / Exa citations to hypotheses
    - Synthesis: Generate the final report from all findings + grounded hypotheses

    Graph structure:
        intake -> entity_resolution -> triage -+-> direct_kg ------+-> pathway_enrichment
                                               |                   |
                                               +-> cold_start -----+
                                                                   |
                                                                   v
                                                             integration
                                                                   |
                                               +-------------------+-------------------+
                                               |                                       |
                                               v (if longitudinal)                     v (else)
                                           temporal                                    |
                                               |                                       |
                                               +------> hypothesis_extraction <--------+
                                                                   |
                                                          literature_grounding
                                                                   |
                                                                synthesis
                                                                   |
                                                                  END

    When both triage branches are needed, LangGraph executes them in the same superstep
    (parallel). The operator.add reducers on findings fields ensure safe state
    merging from concurrent branches. Both branches converge to pathway_enrichment.

    The temporal node only executes for longitudinal studies (is_longitudinal=True);
    both routes converge on hypothesis_extraction.

    Returns:
        Compiled LangGraph StateGraph ready for execution
    """
    # Create graph with our state schema
    workflow = StateGraph(DiscoveryState)

    # Add all 11 nodes
    workflow.add_node("intake", intake.run)
    workflow.add_node("entity_resolution", entity_resolution.run)
    workflow.add_node("triage", triage.run)
    workflow.add_node("direct_kg", direct_kg.run)
    workflow.add_node("cold_start", cold_start.run)
    workflow.add_node("pathway_enrichment", pathway_enrichment.run)
    workflow.add_node("integration", integration.run)
    workflow.add_node("temporal", temporal.run)
    workflow.add_node("hypothesis_extraction", hypothesis_extraction.run)
    workflow.add_node("synthesis", synthesis.run)
    workflow.add_node("literature_grounding", literature_grounding.run)

    # Linear edges: intake -> entity_resolution -> triage
    workflow.set_entry_point("intake")
    workflow.add_edge("intake", "entity_resolution")
    workflow.add_edge("entity_resolution", "triage")

    # Conditional routing after triage
    # When route_after_triage returns a list, both branches execute in parallel
    workflow.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "direct_kg": "direct_kg",
            "cold_start": "cold_start",
            "pathway_enrichment": "pathway_enrichment",
        }
    )

    # Both analysis branches converge to pathway_enrichment
    workflow.add_edge("direct_kg", "pathway_enrichment")
    workflow.add_edge("cold_start", "pathway_enrichment")

    # Pathway enrichment flows to integration
    workflow.add_edge("pathway_enrichment", "integration")

    # Conditional routing after integration: temporal (if longitudinal) or hypothesis_extraction
    workflow.add_conditional_edges(
        "integration",
        route_after_integration,
        {
            "temporal": "temporal",
            "hypothesis_extraction": "hypothesis_extraction",
        }
    )

    # Temporal flows to hypothesis_extraction
    workflow.add_edge("temporal", "hypothesis_extraction")

    # Ground-before-synthesis: hypothesis_extraction -> literature_grounding -> synthesis -> END.
    # Hypotheses are produced, then grounded with literature, BEFORE synthesis reads them to
    # write the report (so the report can cite the grounded evidence — PR 2).
    workflow.add_edge("hypothesis_extraction", "literature_grounding")
    workflow.add_edge("literature_grounding", "synthesis")
    workflow.add_edge("synthesis", END)

    # Compile and return
    return workflow.compile()


def build_discovery_graph_v1() -> StateGraph:
    """
    Legacy Phase 1 graph for backward compatibility.

    Linear 3-node pipeline without triage or parallel branches.
    Use build_discovery_graph() for the latest implementation.
    """
    workflow = StateGraph(DiscoveryState)

    workflow.add_node("intake", intake.run)
    workflow.add_node("entity_resolution", entity_resolution.run)
    workflow.add_node("synthesis", synthesis.run)

    workflow.set_entry_point("intake")
    workflow.add_edge("intake", "entity_resolution")
    workflow.add_edge("entity_resolution", "synthesis")
    workflow.add_edge("synthesis", END)

    return workflow.compile()
