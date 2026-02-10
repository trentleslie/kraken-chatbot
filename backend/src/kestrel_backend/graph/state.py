"""
State definitions for the KRAKEN LangGraph workflow.

Uses TypedDict for LangGraph compatibility with Pydantic models for validation.
The Annotated[list[X], operator.add] pattern enables parallel writes to list fields.
"""

from typing import TypedDict, Literal, Annotated
from pydantic import BaseModel, Field, ConfigDict
import operator


class EntityResolution(BaseModel):
    """Result of resolving a raw entity name to a knowledge graph identifier."""

    model_config = ConfigDict(frozen=True)  # Immutable for safe state passing

    raw_name: str = Field(..., description="Original entity name from user input")
    curie: str | None = Field(None, description="Compact URI (e.g., CHEBI:17234)")
    resolved_name: str | None = Field(None, description="Canonical name from KG")
    category: str | None = Field(None, description="Biolink category (e.g., biolink:ChemicalEntity)")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Resolution confidence score")
    method: Literal["exact", "fuzzy", "semantic", "failed"] = Field(
        "failed", description="Resolution method used"
    )


class NoveltyScore(BaseModel):
    """Entity characterization based on KG edge density."""

    model_config = ConfigDict(frozen=True)

    curie: str = Field(..., description="Resolved CURIE")
    raw_name: str = Field(..., description="Original entity name")
    edge_count: int = Field(..., ge=0, description="Number of edges in KG")
    classification: Literal["cold_start", "sparse", "moderate", "well_characterized"] = Field(
        ..., description="Classification based on edge count thresholds"
    )


class Finding(BaseModel):
    """Universal evidence container for analysis nodes."""

    model_config = ConfigDict(frozen=True)

    entity: str = Field(..., description="Entity CURIE or name this finding relates to")
    claim: str = Field(..., description="The finding or hypothesis text")
    tier: Literal[1, 2, 3] = Field(..., description="1=high confidence, 2=moderate, 3=speculative")
    predicate: str | None = Field(None, description="Relationship predicate if applicable")
    source: str | None = Field(None, description="Source node or analysis that generated this")
    pmids: list[str] = Field(default_factory=list, description="Supporting PubMed IDs")
    confidence: Literal["high", "moderate", "low"] = Field(
        "moderate", description="Confidence level in this finding"
    )
    logic_chain: str | None = Field(None, description="Structural reasoning chain for inferred findings")


# =============================================================================
# Phase 3: Structured Analysis Models
# =============================================================================

class DiseaseAssociation(BaseModel):
    """Structured disease association from direct KG query."""

    model_config = ConfigDict(frozen=True)

    entity_curie: str = Field(..., description="Source entity CURIE")
    disease_curie: str = Field(..., description="Disease CURIE")
    disease_name: str = Field(..., description="Disease canonical name")
    predicate: str = Field(..., description="Relationship predicate (e.g., biolink:gene_associated_with_condition)")
    source: str = Field(..., description="Source database (e.g., GWAS Catalog, DisGeNET)")
    pmids: list[str] = Field(default_factory=list, description="Supporting PubMed IDs")
    evidence_type: Literal["gwas", "curated", "text_mined", "predicted"] = Field(
        "curated", description="Type of evidence supporting this association"
    )


class PathwayMembership(BaseModel):
    """Entity's pathway/biological process associations."""

    model_config = ConfigDict(frozen=True)

    entity_curie: str = Field(..., description="Source entity CURIE")
    pathway_curie: str = Field(..., description="Pathway or process CURIE")
    pathway_name: str = Field(..., description="Pathway canonical name")
    predicate: str = Field(..., description="Relationship predicate")
    source: str = Field(..., description="Source database (e.g., Reactome, KEGG)")


class InferredAssociation(BaseModel):
    """Association inferred via cold-start semantic reasoning."""

    model_config = ConfigDict(frozen=True)

    source_entity: str = Field(..., description="The sparse entity being analyzed")
    target_curie: str = Field(..., description="Inferred target CURIE")
    target_name: str = Field(..., description="Inferred target name")
    predicate: str = Field(..., description="Inferred relationship predicate")
    logic_chain: str = Field(..., description="Structural reasoning: 'X similar to Y, Y connected to Z'")
    supporting_analogues: int = Field(..., ge=0, description="Number of analogues supporting this inference")
    confidence: Literal["high", "moderate", "low"] = Field("low", description="Inference confidence")
    validation_step: str = Field(..., description="Suggested experimental validation")


class AnalogueEntity(BaseModel):
    """A similar entity found via vector similarity for cold-start analysis."""

    model_config = ConfigDict(frozen=True)

    curie: str = Field(..., description="Analogue CURIE")
    name: str = Field(..., description="Analogue canonical name")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Semantic similarity score")
    category: str | None = Field(None, description="Biolink category")


# =============================================================================
# Phase 4a: Pathway Enrichment Models
# =============================================================================

class SharedNeighbor(BaseModel):
    """A neighbor shared by 2+ input entities."""

    model_config = ConfigDict(frozen=True)

    curie: str = Field(..., description="Shared neighbor CURIE")
    name: str = Field(..., description="Canonical name")
    category: str = Field(..., description="Biolink category")
    degree: int = Field(..., ge=0, description="Total edge count in KG")
    is_hub: bool = Field(False, description="True if degree > 1000")
    connected_inputs: list[str] = Field(..., description="Input CURIEs sharing this neighbor")
    predicates: list[str] = Field(default_factory=list, description="Predicate types in connections")


class BiologicalTheme(BaseModel):
    """A biological theme identified from shared neighbors grouped by category."""

    model_config = ConfigDict(frozen=True)

    category: str = Field(..., description="Biolink category (e.g., biolink:BiologicalProcess)")
    members: list[str] = Field(..., description="CURIEs of shared neighbors in this theme")
    member_names: list[str] = Field(default_factory=list, description="Names of shared neighbors")
    input_coverage: int = Field(..., ge=0, description="Number of input entities connected to this theme")
    top_non_hub: str | None = Field(None, description="CURIE of most specific (lowest degree) non-hub member")


# =============================================================================
# Phase 4b: Integration Models (Bridges + Gap Analysis)
# =============================================================================

class Bridge(BaseModel):
    """Cross-entity-type connection discovered through multi-hop analysis."""

    model_config = ConfigDict(frozen=True)

    path_description: str = Field(..., description="e.g., 'metabolite → gene → disease'")
    entities: list[str] = Field(..., description="CURIEs along the path")
    entity_names: list[str] = Field(default_factory=list, description="Names along the path")
    predicates: list[str] = Field(default_factory=list, description="Predicate at each hop")
    tier: Literal[2, 3] = Field(3, description="Evidence tier (2=moderate, 3=speculative)")
    novelty: Literal["known", "inferred"] = Field("inferred", description="Known from KG or inferred")
    significance: str = Field(..., description="Why this bridge matters for the study")


class GapEntity(BaseModel):
    """Expected-but-absent entity in the analysis (Open World Assumption)."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="e.g., 'BCAAs', 'HbA1c'")
    category: str = Field(..., description="Biolink category")
    curie: str | None = Field(None, description="CURIE if known")
    expected_reason: str = Field(..., description="Why expected in study context")
    absence_interpretation: str = Field(
        ..., description="Open World Assumption framing (unstudied, not nonexistent)"
    )
    is_informative: bool = Field(
        False, description="True if absence is itself informative"
    )


# =============================================================================
# Phase 4b: Temporal Models (for Longitudinal Studies)
# =============================================================================

class TemporalClassification(BaseModel):
    """Temporal classification for longitudinal study findings."""

    model_config = ConfigDict(frozen=True)

    entity: str = Field(..., description="Entity CURIE or name")
    finding_claim: str = Field(..., description="The finding being classified")
    classification: Literal["upstream_cause", "downstream_consequence", "parallel_effect"] = Field(
        ..., description="Temporal relationship to disease progression"
    )
    reasoning: str = Field(..., description="Why this classification was assigned")
    confidence: Literal["high", "moderate", "low"] = Field(
        "moderate", description="Classification confidence"
    )


# =============================================================================
# Phase 5: Hypothesis Model
# =============================================================================

class Hypothesis(BaseModel):
    """A biological hypothesis generated from accumulated evidence."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(..., description="Short descriptive title")
    tier: Literal[1, 2, 3] = Field(..., description="Evidence tier (1=high, 2=moderate, 3=speculative)")
    claim: str = Field(..., description="The hypothesis statement")
    supporting_entities: list[str] = Field(..., description="CURIEs that support this hypothesis")
    contradicting_entities: list[str] = Field(default_factory=list, description="CURIEs that don't fit")
    structural_logic: str = Field(..., description="The reasoning chain")
    confidence: Literal["high", "moderate", "low"] = Field(
        "moderate", description="Confidence level in this hypothesis"
    )
    validation_steps: list[str] = Field(..., description="Concrete experiments/analyses to validate")
    validation_gap_note: str = Field(
        "",
        description="Calibration note: ~18% of computational predictions reach clinical investigation"
    )


# Legacy alias for backward compatibility
NoveltyTriage = NoveltyScore


class DiscoveryState(TypedDict, total=False):
    """
    State schema for the KRAKEN discovery workflow.

    Uses total=False to make all fields optional, enabling incremental state building
    as the workflow progresses through nodes.

    Fields with Annotated[list[X], operator.add] use LangGraph reducers to merge
    parallel writes (e.g., from asyncio.gather batches or parallel branches).
    """

    # === Phase 1: Input Processing ===
    raw_query: str  # Original user query
    query_type: Literal["retrieval", "discovery", "hybrid"]
    raw_entities: list[str]  # Extracted entity names before resolution
    conversation_history: list[tuple[str, str]]  # (role, content) pairs

    # === Study Context (for longitudinal analysis) ===
    is_longitudinal: bool
    duration_years: int | None
    fdr_entities: list[str]  # First-degree relatives
    marginal_entities: list[str]  # Entities with marginal significance

    # === Phase 2: Entity Resolution ===
    # Uses operator.add reducer for parallel batch writes
    resolved_entities: Annotated[list[EntityResolution], operator.add]

    # === Phase 3: Triage & Classification ===
    # Uses operator.add reducer for parallel novelty scoring
    novelty_scores: Annotated[list[NoveltyScore], operator.add]
    # Classification buckets (populated by triage node)
    well_characterized_curies: list[str]  # >=200 edges
    moderate_curies: list[str]  # 20-199 edges
    sparse_curies: list[str]  # 1-19 edges
    cold_start_curies: list[str]  # 0 edges

    # === Phase 4: Analysis Results (parallel branches) ===
    # Both use operator.add to enable safe parallel writes from concurrent branches
    direct_findings: Annotated[list[Finding], operator.add]
    cold_start_findings: Annotated[list[Finding], operator.add]

    # === Phase 4: Structured Analysis Data ===
    disease_associations: Annotated[list[DiseaseAssociation], operator.add]
    pathway_memberships: Annotated[list[PathwayMembership], operator.add]
    inferred_associations: Annotated[list[InferredAssociation], operator.add]
    analogues_found: Annotated[list[AnalogueEntity], operator.add]
    hub_flags: Annotated[list[str], operator.add]  # CURIEs flagged as high-degree hubs

    # === Phase 4a: Pathway Enrichment ===
    shared_neighbors: Annotated[list[SharedNeighbor], operator.add]
    biological_themes: list[BiologicalTheme]

    # === Phase 4b: Integration (Bridges + Gap Analysis) ===
    bridges: Annotated[list[Bridge], operator.add]
    gap_entities: Annotated[list[GapEntity], operator.add]

    # === Phase 4b: Temporal Analysis (Conditional) ===
    temporal_classifications: Annotated[list[TemporalClassification], operator.add]

    # Legacy fields for compatibility
    kg_results: dict  # Direct KG query results
    predictions: dict  # Cold-start predictions
    pathway_enrichment: dict  # Pathway analysis
    legacy_bridges: list[dict]  # Cross-type bridge paths (legacy)
    gap_analysis: dict  # Missing edge analysis (legacy)
    novelty_triage: list[NoveltyScore]  # Legacy alias
    well_characterized_ids: list[str]  # Legacy alias
    sparse_ids: list[str]  # Legacy alias

    # === Phase 5: Temporal Analysis ===
    temporal_patterns: dict | None

    # === Phase 5: Output (Updated) ===
    synthesis_report: str
    # Updated to use Hypothesis model with operator.add reducer
    hypotheses: Annotated[list[Hypothesis], operator.add]

    # === Metadata ===
    # Uses operator.add reducer to accumulate errors from all nodes
    errors: Annotated[list[str], operator.add]
    node_timings: dict[str, float]  # Performance tracking
