"""Per-node state validation contracts for the discovery pipeline.

Defines input/output Pydantic models for each pipeline node and a @validate_state
decorator that validates state at node entry and exit. Models serve dual purpose:
runtime validation (R6) and machine-checkable documentation (R7).

Field categories:
- Always-required: populated by all upstream paths (validated as non-None)
- Path-conditional: populated only on certain routes (Optional, with model_validator
  for OR-semantics where at least one branch must provide data)
- Output: what the node must return

The @validate_state decorator wraps async run() functions. On entry, it constructs
the InputModel from the state dict. On exit, it validates the returned dict against
the OutputModel. Validation errors include node name, missing/invalid fields, and
actual vs. expected types.
"""

import functools
import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

logger = logging.getLogger(__name__)


class StateValidationError(Exception):
    """Raised when node state fails validation."""

    def __init__(self, node_name: str, direction: str, errors: list[str]):
        self.node_name = node_name
        self.direction = direction  # "input" or "output"
        self.validation_errors = errors
        msg = (
            f"State validation failed for node '{node_name}' ({direction}):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )
        super().__init__(msg)


# =============================================================================
# Base configuration for all contract models
# =============================================================================

class _ContractBase(BaseModel):
    """Base for all contract models. Uses extra='ignore' to accept the full state dict."""
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)


# =============================================================================
# Input Models
# =============================================================================

class IntakeInput(_ContractBase):
    """Intake node reads the raw query from the user."""
    raw_query: str


class EntityResolutionInput(_ContractBase):
    """Entity resolution reads extracted entity names from intake."""
    raw_entities: list[str]


class TriageInput(_ContractBase):
    """Triage reads resolved entities from entity resolution."""
    resolved_entities: list[Any]  # list[EntityResolution] but using Any to avoid circular import


class DirectKGInput(_ContractBase):
    """Direct KG reads well-characterized and moderate CURIEs from triage."""
    well_characterized_curies: list[str] | None = None
    moderate_curies: list[str] | None = None
    novelty_scores: list[Any] | None = None
    resolved_entities: list[Any] | None = None

    @model_validator(mode="after")
    def at_least_one_curie_list(self) -> "DirectKGInput":
        wc = self.well_characterized_curies or []
        mod = self.moderate_curies or []
        if not wc and not mod:
            raise ValueError(
                "DirectKGInput requires at least one of well_characterized_curies or "
                "moderate_curies to be non-empty. Both are empty/None — "
                "this suggests triage found no well-characterized or moderate entities."
            )
        return self


class ColdStartInput(_ContractBase):
    """Cold start reads sparse and cold-start CURIEs from triage."""
    sparse_curies: list[str] | None = None
    cold_start_curies: list[str] | None = None
    novelty_scores: list[Any] | None = None
    resolved_entities: list[Any] | None = None

    @model_validator(mode="after")
    def at_least_one_curie_list(self) -> "ColdStartInput":
        sparse = self.sparse_curies or []
        cold = self.cold_start_curies or []
        if not sparse and not cold:
            raise ValueError(
                "ColdStartInput requires at least one of sparse_curies or "
                "cold_start_curies to be non-empty. Both are empty/None — "
                "this suggests triage found no sparse or cold-start entities."
            )
        return self


class PathwayEnrichmentInput(_ContractBase):
    """Pathway enrichment reads all resolved entities after both branches converge."""
    resolved_entities: list[Any]
    novelty_scores: list[Any] | None = None


class IntegrationInput(_ContractBase):
    """Integration reads findings from both branches — at least one must be non-empty.

    Uses @model_validator(mode='after') for OR-semantics: Pydantic doesn't express
    'at least one of X or Y must be non-empty' natively, so Optional alone would
    silently accept both-None/both-empty states.
    """
    resolved_entities: list[Any] | None = None
    direct_findings: list[Any] | None = None
    cold_start_findings: list[Any] | None = None
    disease_associations: list[Any] | None = None
    pathway_memberships: list[Any] | None = None
    inferred_associations: list[Any] | None = None
    biological_themes: list[Any] | None = None
    raw_query: str | None = None

    @model_validator(mode="after")
    def at_least_one_findings_branch(self) -> "IntegrationInput":
        direct = self.direct_findings or []
        cold = self.cold_start_findings or []
        if not direct and not cold:
            raise ValueError(
                "IntegrationInput requires at least one of direct_findings or "
                "cold_start_findings to be non-empty. Both are empty/None — "
                "this suggests neither direct_kg nor cold_start produced results."
            )
        return self


class TemporalInput(_ContractBase):
    """Temporal reads findings and longitudinal context — only runs if is_longitudinal."""
    is_longitudinal: bool | None = None
    direct_findings: list[Any] | None = None
    cold_start_findings: list[Any] | None = None


class SynthesisInput(_ContractBase):
    """Synthesis reads all accumulated state. Path-conditional fields are Optional."""
    resolved_entities: list[Any] | None = None
    novelty_scores: list[Any] | None = None
    direct_findings: list[Any] | None = None
    cold_start_findings: list[Any] | None = None
    disease_associations: list[Any] | None = None
    pathway_memberships: list[Any] | None = None
    inferred_associations: list[Any] | None = None
    analogues_found: list[Any] | None = None
    shared_neighbors: list[Any] | None = None
    biological_themes: list[Any] | None = None
    bridges: list[Any] | None = None
    gap_entities: list[Any] | None = None
    temporal_classifications: list[Any] | None = None

    @model_validator(mode="after")
    def at_least_one_findings_branch(self) -> "SynthesisInput":
        """Same OR-semantics as IntegrationInput."""
        direct = self.direct_findings or []
        cold = self.cold_start_findings or []
        if not direct and not cold:
            raise ValueError(
                "SynthesisInput requires at least one of direct_findings or "
                "cold_start_findings to be non-empty."
            )
        return self


class LiteratureGroundingInput(_ContractBase):
    """Literature grounding reads hypotheses from synthesis."""
    hypotheses: list[Any]


# =============================================================================
# Output Models
# =============================================================================

class IntakeOutput(_ContractBase):
    """Intake must produce raw_entities and query_type."""
    raw_entities: list[str]
    query_type: str


class EntityResolutionOutput(_ContractBase):
    """Entity resolution must produce resolved_entities."""
    resolved_entities: list[Any]


class TriageOutput(_ContractBase):
    """Triage must produce novelty_scores and classification buckets."""
    novelty_scores: list[Any]
    well_characterized_curies: list[str]
    moderate_curies: list[str]
    sparse_curies: list[str]
    cold_start_curies: list[str]


class DirectKGOutput(_ContractBase):
    """Direct KG must produce direct_findings."""
    direct_findings: list[Any]


class ColdStartOutput(_ContractBase):
    """Cold start must produce cold_start_findings."""
    cold_start_findings: list[Any]


class PathwayEnrichmentOutput(_ContractBase):
    """Pathway enrichment must produce shared_neighbors."""
    shared_neighbors: list[Any]


class IntegrationOutput(_ContractBase):
    """Integration must produce bridges and gap_entities."""
    bridges: list[Any]
    gap_entities: list[Any]


class TemporalOutput(_ContractBase):
    """Temporal must produce temporal_classifications (may be empty if not longitudinal)."""
    temporal_classifications: list[Any] | None = None


class SynthesisOutput(_ContractBase):
    """Synthesis must produce synthesis_report and hypotheses."""
    synthesis_report: str
    hypotheses: list[Any]


class LiteratureGroundingOutput(_ContractBase):
    """Literature grounding must produce updated hypotheses."""
    hypotheses: list[Any]


# =============================================================================
# Node Contract Registry
# =============================================================================

NODE_CONTRACTS: dict[str, tuple[type[_ContractBase], type[_ContractBase]]] = {
    "intake": (IntakeInput, IntakeOutput),
    "entity_resolution": (EntityResolutionInput, EntityResolutionOutput),
    "triage": (TriageInput, TriageOutput),
    "direct_kg": (DirectKGInput, DirectKGOutput),
    "cold_start": (ColdStartInput, ColdStartOutput),
    "pathway_enrichment": (PathwayEnrichmentInput, PathwayEnrichmentOutput),
    "integration": (IntegrationInput, IntegrationOutput),
    "temporal": (TemporalInput, TemporalOutput),
    "synthesis": (SynthesisInput, SynthesisOutput),
    "literature_grounding": (LiteratureGroundingInput, LiteratureGroundingOutput),
}


# =============================================================================
# Decorator
# =============================================================================

def validate_state(input_model: type[_ContractBase], output_model: type[_ContractBase]):
    """Decorator that validates node state at entry and exit.

    Usage:
        @validate_state(IntakeInput, IntakeOutput)
        async def run(state: DiscoveryState) -> dict[str, Any]:
            ...

    On entry: constructs input_model from state dict, raises StateValidationError on failure.
    On exit: constructs output_model from returned dict, raises StateValidationError on failure.
    """
    def decorator(func):
        # Extract node name from the function's module path
        node_name = func.__module__.rsplit(".", 1)[-1] if func.__module__ else func.__qualname__

        @functools.wraps(func)
        async def wrapper(state, *args, **kwargs):
            # Validate input — catch only ValidationError, let other exceptions propagate
            try:
                input_model.model_validate(dict(state))
            except ValidationError as e:
                errors = _extract_validation_errors(e)
                raise StateValidationError(node_name, "input", errors) from e

            # Execute the node
            result = await func(state, *args, **kwargs)

            # Validate output
            if result is not None:
                try:
                    output_model.model_validate(result)
                except ValidationError as e:
                    errors = _extract_validation_errors(e)
                    raise StateValidationError(node_name, "output", errors) from e
            else:
                # Log warning when None is returned from nodes with required output fields
                required_fields = [
                    name for name, field in output_model.model_fields.items()
                    if field.is_required()
                ]
                if required_fields:
                    logger.warning(
                        "Node '%s' returned None but output contract requires: %s",
                        node_name, ", ".join(required_fields),
                    )

            return result

        return wrapper
    return decorator


def _extract_validation_errors(exc: Exception) -> list[str]:
    """Extract human-readable error messages from a Pydantic validation exception."""
    if isinstance(exc, ValidationError):
        return [
            f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}"
            for err in exc.errors()
        ]
    return [str(exc)]
