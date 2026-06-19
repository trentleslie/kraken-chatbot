"""
Hypothesis Extraction Node: validate cross-type bridges, then extract structured hypotheses.

Runs after integration (and the conditional temporal node) and BEFORE literature_grounding
and synthesis in the ground-before-synthesis topology. It produces the `hypotheses` that
grounding enriches with literature and that synthesis then reports on — so hypothesis
production must happen here, upstream of where the grounded evidence is needed, rather than
inside synthesis.run() as it did historically.

Two pieces were relocated VERBATIM out of synthesis.py (validate_bridge_hypotheses and
extract_hypotheses); their behavior is unchanged. The node adds a run()-level failure
boundary (R13): because main.py converts any uncaught node exception into PIPELINE_ERROR
with no report, an unguarded crash here — now upstream of synthesis — would abort the whole
run before any report is produced. The boundary degrades to a contract-valid payload instead.
"""

import logging
import time
from typing import Any

from ..state import DiscoveryState, Bridge, Hypothesis, InferredAssociation
from ...kestrel_client import multi_hop_query, parse_kestrel_response
from ..state_contracts import (
    validate_state,
    HypothesisExtractionInput,
    HypothesisExtractionOutput,
)

logger = logging.getLogger(__name__)


# Validation gap calibration constant (moved with extract_hypotheses from synthesis.py)
VALIDATION_GAP_NOTE = "~18% of computational predictions reach clinical investigation (systematic review calibration)"


async def validate_bridge_hypotheses(bridges: list[Bridge]) -> list[Bridge]:
    """
    Validate Tier 3 bridge hypotheses using doubly-pinned multi_hop_query.

    For each Tier 3 bridge, attempt to verify the path exists in the KG.
    If verified, upgrade to Tier 2. If not verified, keep as Tier 3.

    Args:
        bridges: List of Bridge objects to validate

    Returns:
        Updated list of bridges with validated ones upgraded to Tier 2
    """
    if not bridges:
        return bridges

    validated_bridges: list[Bridge] = []
    tier3_bridges = [b for b in bridges if b.tier == 3]
    other_bridges = [b for b in bridges if b.tier != 3]

    logger.info("validate_bridge_hypotheses: validating %d Tier 3 bridges", len(tier3_bridges))

    for bridge in tier3_bridges:
        if len(bridge.entities) < 2:
            # Can't validate without start/end
            validated_bridges.append(bridge)
            continue

        start_curie = bridge.entities[0]
        end_curie = bridge.entities[-1]

        try:
            # Use doubly-pinned search with max_hops based on path length
            expected_hops = len(bridge.entities) - 1
            result = await multi_hop_query(
                start_node_ids=[start_curie],
                end_node_ids=[end_curie],
                max_hops=min(expected_hops + 1, 5),  # Allow 1 extra hop, cap at 5
                limit=1,  # We just need to know if ANY path exists
            )

            if result.get("isError"):
                # Validation failed, keep as Tier 3
                validated_bridges.append(bridge)
                continue

            # Check if we got any paths back. Use the shared helper — the real response is
            # {"results":[{"paths":[[curie,...]]}], ...} with NO top-level "paths" key. The old
            # data.get("paths", data) fell back to the whole dict (always truthy) and upgraded
            # EVERY bridge to Tier 2 on garbage; the helper returns n_paths=0 on a no-path result.
            parsed = parse_kestrel_response(result)

            if parsed["n_paths"] > 0:
                # Path verified! Upgrade to Tier 2
                logger.info(
                    "validate_bridge_hypotheses: VALIDATED %s -> %s, upgrading to Tier 2",
                    start_curie, end_curie
                )
                validated_bridges.append(Bridge(
                    path_description=bridge.path_description,
                    entities=bridge.entities,
                    entity_names=bridge.entity_names,
                    predicates=bridge.predicates,
                    tier=2,  # UPGRADED
                    novelty="known",  # Now verified in KG
                    significance=bridge.significance + " [KG-validated]",
                ))
            else:
                # No path found, keep as Tier 3
                validated_bridges.append(bridge)

        except Exception as e:
            logger.warning("Error validating bridge %s -> %s: %s", start_curie, end_curie, str(e))
            # Keep as Tier 3 on error
            validated_bridges.append(bridge)

    # Combine validated bridges with other tiers
    all_bridges = other_bridges + validated_bridges
    logger.info(
        "validate_bridge_hypotheses: %d bridges after validation (%d upgraded to Tier 2)",
        len(all_bridges),
        sum(1 for b in validated_bridges if b.tier == 2)
    )

    return all_bridges


def extract_hypotheses(state: DiscoveryState) -> list[Hypothesis]:
    """
    Extract structured Hypothesis objects from accumulated state.

    Builds hypotheses programmatically from:
    - cold_start_findings (Tier 3 inferences)
    - bridges (Tier 2-3 cross-type connections)

    All hypotheses include the ~18% validation gap note.
    """
    hypotheses: list[Hypothesis] = []

    # From cold-start findings (Tier 3 inferences)
    cold_start_findings = state.get("cold_start_findings", [])
    for finding in cold_start_findings:
        # Skip placeholder or pending findings
        if not finding.claim or "pending" in finding.claim.lower():
            continue

        # Only process Tier 3 findings (speculative)
        if finding.tier == 3:
            hypotheses.append(Hypothesis(
                title=f"Inferred role of {finding.entity}",
                tier=3,
                claim=finding.claim,
                supporting_entities=[finding.entity],
                contradicting_entities=[],
                structural_logic=finding.logic_chain or "Based on analogue inference",
                confidence=finding.confidence,
                validation_steps=[
                    f"Search literature for {finding.entity} associations",
                    f"Validate in independent cohort",
                ],
                validation_gap_note=VALIDATION_GAP_NOTE,
            ))

    # From cross-type bridges
    bridges = state.get("bridges", [])
    for bridge in bridges:
        if not isinstance(bridge, Bridge):
            continue

        # Only create hypothesis if bridge has significance
        if not bridge.significance:
            continue

        # Build the logic chain from path
        if bridge.entity_names and bridge.predicates:
            logic = f"{' -> '.join(bridge.entity_names)} via {', '.join(bridge.predicates)}"
        else:
            logic = bridge.path_description

        # Get target entity name for validation step
        target_name = bridge.entity_names[-1] if bridge.entity_names else "target"

        hypotheses.append(Hypothesis(
            title=f"Bridge: {bridge.path_description}",
            tier=bridge.tier,
            claim=bridge.significance,
            supporting_entities=bridge.entities,
            contradicting_entities=[],
            structural_logic=logic,
            confidence="moderate",
            validation_steps=[
                f"Verify path in literature",
                f"Check {target_name} in GWAS Catalog",
            ],
            validation_gap_note=VALIDATION_GAP_NOTE,
        ))

    # From inferred associations (cold-start analogues)
    inferred_associations = state.get("inferred_associations", [])
    for inference in inferred_associations:
        if not isinstance(inference, InferredAssociation):
            continue

        hypotheses.append(Hypothesis(
            title=f"Inferred: {inference.source_entity} -> {inference.target_name}",
            tier=3,
            claim=f"{inference.source_entity} may be associated with {inference.target_name} via {inference.predicate}",
            supporting_entities=[inference.source_entity, inference.target_curie],
            contradicting_entities=[],
            structural_logic=inference.logic_chain,
            confidence=inference.confidence,
            validation_steps=[inference.validation_step],
            validation_gap_note=VALIDATION_GAP_NOTE,
        ))

    return hypotheses


@validate_state(HypothesisExtractionInput, HypothesisExtractionOutput)
async def run(state: DiscoveryState) -> dict[str, Any]:
    """
    Validate cross-type bridges then extract structured hypotheses from accumulated state.

    Returns:
        bridges: the validated bridge list (Tier-3 paths verified in the KG upgraded to Tier 2),
            re-emitted so downstream synthesis reads the validated list (last-write-wins channel).
        hypotheses: structured Hypothesis objects (may be empty for a well-characterized-only run,
            which is a valid output, not a failure).

    Failure boundary (R13): this node is now the unguarded upstream gate feeding both grounding
    and synthesis. validate_bridge_hypotheses is called unguarded at this level and
    extract_hypotheses iterates typed state objects that can AttributeError on a degraded state.
    Because main.py turns any uncaught exception into PIPELINE_ERROR with no report, the body is
    wrapped so a crash degrades to a contract-valid payload — which MUST include `bridges` (a
    required output field); a bare {"hypotheses": []} would itself fail output validation and
    become the abort it is meant to prevent.
    """
    logger.info("Starting hypothesis extraction")
    start = time.time()

    bridges = state.get("bridges", [])
    try:
        validated_bridges = await validate_bridge_hypotheses(bridges)

        # Extract hypotheses from a state view carrying the validated bridges.
        state_with_validated = dict(state)
        state_with_validated["bridges"] = validated_bridges
        hypotheses = extract_hypotheses(state_with_validated)
    except Exception as e:
        logger.warning("hypothesis_extraction degraded after exception: %s", e)
        # Degrade payload satisfies HypothesisExtractionOutput (bridges required): pass the
        # upstream bridges through unvalidated and emit no hypotheses, so the run still reaches
        # synthesis with a report instead of aborting upstream.
        return {"bridges": bridges, "hypotheses": []}

    duration = time.time() - start
    logger.info(
        "Completed hypothesis_extraction in %.1fs — validated_bridges=%d, hypotheses=%d",
        duration, len(validated_bridges), len(hypotheses),
    )

    return {
        "bridges": validated_bridges,
        "hypotheses": hypotheses,
    }
