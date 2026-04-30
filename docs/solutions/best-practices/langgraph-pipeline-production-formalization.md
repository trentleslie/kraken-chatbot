---
title: "Production-Grade Formalization of a LangGraph Biomedical Discovery Pipeline"
date: 2026-04-29
category: best-practices
module: discovery-pipeline
problem_type: best_practice
component: tooling
severity: high
applies_when:
  - "Refactoring a large LLM pipeline with no regression detection"
  - "Building assessment infrastructure for non-deterministic AI pipelines"
  - "Formalizing state contracts across conditional pipeline branches"
tags:
  - langgraph
  - pipeline-refactoring
  - assessment-infrastructure
  - respx-cassettes
  - state-validation
  - pydantic
  - llm-as-judge
  - config-flags
---

# Production-Grade Formalization of a LangGraph Biomedical Discovery Pipeline

## Context

The KRAKEN discovery pipeline (~8,100 lines, 10 LangGraph nodes) was functionally complete and producing research-grade biomedical output, but had accumulated technical debt that made it risky to modify: SDK setup code duplicated across 8 nodes, hardcoded thresholds with no documented rationale, 9 legacy state fields, and zero evaluation infrastructure. Changes were risky (no regression detection) and output quality was unmeasured.

A 5-PR effort formalized the pipeline across four workstreams: shared SDK utilities, parameterized configuration, per-node state validation, and a two-tier assessment framework (structural regression + LLM-as-judge quality scoring). The work produced 13 implementation units, 165 new tests, and established the measurement foundation for confident iteration.

Key constraint: the pipeline makes LLM calls via Claude Agent SDK in 8 of 10 nodes, making output inherently non-deterministic. Assessment infrastructure had to distinguish pipeline regressions from normal LLM variance.

## Guidance

### 1. Gate transport-level mocking on a wire-format spike

Before committing to respx for HTTP recording, run a focused spike: can respx round-trip the target service's wire format? For Kestrel MCP, that meant SSE-formatted responses (`event: message\ndata: {json}`). The spike was binary — pass means proceed with cassettes, fail means fall back to function-level mocking. The threshold was pre-committed before the spike ran so it couldn't drift under deadline pressure.

Key prerequisite: all API clients must use the same HTTP library (httpx in this case). If any client uses requests or aiohttp, those calls escape the transport-level intercept and the cassette is incomplete.

### 2. Build a shared cassette module with explicit boundary documentation

A single `assessment/cassette.py` handles both recording (`CassetteRecorder`) and replay (`CassetteReplayer`). The replayer indexes by request key for O(1) lookup. Critically: replay eliminates external API dependency but does NOT eliminate LLM non-determinism — Claude SDK `query()` calls still execute live. Documenting this boundary explicitly prevents future confusion about what "cassette replay" actually controls.

### 3. Use variance-based tolerance bands, not fixed percentages

Run 5 baseline captures per query. Compute mean +/- 2 standard deviations per metric. Coefficient of variation (CV) > 0.5 produces a warning, not a hard failure. Canonical run selected by median finding count for quality scoring.

Fixed percentages (e.g., "within 20%") mask whether variance is intrinsic to the pipeline or an artifact of test conditions. Empirically calibrated bands catch genuine regressions without noise-driven failures.

### 4. Extract boilerplate without homogenizing intentional differences

Pull identical code — imports, MCP config factory, chunk utility — into a shared module. Leave semaphores and fallback orchestration per-node, because these differ for documented reasons:

- Semaphore values: 1 for entity_resolution/triage (prevents CLI spawn conflicts), 6-8 for batch analysis nodes
- Fallback patterns: per-entity, batch, or no-fallback depending on node semantics

The rule: extract when the code is identical AND the configuration should be identical. Leave it in place when values differ for documented reasons.

### 5. Document config rationale per-field with Field(description=...)

Each node gets its own Pydantic config sub-model. Every field uses `Field(description=...)` to record *why* the default exists, not just its value. Hub thresholds are the canonical example — `pathway_enrichment` uses 1000 (shared-neighbor filtering) while `direct_kg` uses 5000 (entity hub bias). Without per-field documentation, these look like inconsistencies rather than deliberate design.

Config-flagged behavioral changes (e.g., `use_llm_classifier=False`) ship in the same PR as the feature code. The flag flip is a separate PR after quality measurement confirms improvement.

### 6. Use model_validator OR-semantics for path-conditional state validation

Pydantic's `Optional[X]` cannot express "at least one of X or Y must be non-empty." For nodes that accept input from multiple upstream paths (e.g., `IntegrationInput` accepts either `direct_findings` or `cold_start_findings`), use `@model_validator(mode='after')` to enforce the constraint. Apply a `@validate_state` decorator consistently across all pipeline nodes.

Catch `ValidationError` specifically — not broad `Exception` — to avoid masking non-validation errors as state contract violations. (session history)

### 7. Use a 1-10 scale with explicit anchors for LLM-as-judge scoring

Five-point scales produce excessive ties, compressing variance and making stability testing unreliable. The judge prompt must include anchors for each band (1-2, 3-4, 5-6, 7-8, 9-10) per dimension.

Stability criterion: per-dimension pairwise Spearman correlation across 5 runs on the full query set, mean >= 0.80. For degenerate cases where all runs produce identical scores, fall back to Krippendorff's alpha (which correctly returns 1.0 for perfect agreement).

Parse LLM score responses with `int()` coercion — LLMs commonly return scores as strings (`"8"` instead of `8`), which causes `TypeError` in `min()`/`max()` comparisons. (session history)

### 8. Structure Greptile reviews around single-concern PRs

Each PR should have one logical concern. This project used 5 PRs: baseline capture, code cleanup, state formalization, structural checks, quality scoring. Greptile reviews were consistently more focused and caught real issues — including an invalid exception class (`respx.errors.AllMockedResponsesSent` doesn't exist), unsafe class-level monkey-patching, missing output contract fields, and string-score type coercion failures. (session history)

## Why This Matters

**SSE spike**: Transport-level mocking is brittle if the wire format assumption is wrong. A 30-minute spike prevents building an entire cassette infrastructure on a false premise. The binary threshold makes the decision auditable.

**Cassette boundary clarity**: Cassettes that appear to eliminate non-determinism but actually don't (because LLM calls still run live) produce misleading stability numbers. Making the boundary explicit is a correctness issue.

**Variance bands**: A pipeline that is inherently noisy will fail fixed-percentage gates on every run regardless of correctness. Empirically calibrated bands catch genuine regressions without noise-driven failures.

**Intentional semaphore differences**: Entity resolution and triage spawn CLI processes; unbounded parallelism causes spawn conflicts. Batch analysis nodes have no CLI interaction and benefit from higher concurrency. Homogenizing these hides meaningful operational differences.

**OR-semantics validation**: If validation requires both branch fields, cold-start-only queries fail. If validation requires neither, upstream routing bugs produce silent empty-input failures. `model_validator` with OR-semantics is the correct expression of the routing contract.

**Config flags with deferred flip**: Shipping new behavior behind a config flag means the feature is observable in production logs before it affects output. The flip PR has its own review and rollback surface.

## When to Apply

- **SSE spike pattern**: Any time the target service uses a non-standard response format (SSE, chunked JSON, multipart) and you are considering transport-level mocking
- **Cassette module**: Assessment runners or regression suites for pipelines with external HTTP dependencies
- **Variance bands**: Pipelines with LLM steps, network variability, or any legitimate non-determinism source
- **Boilerplate extraction**: When 3+ nodes share identical infrastructure code but have different operational parameters
- **Per-node PipelineConfig**: Whenever thresholds, concurrency limits, or behavioral flags differ per node and someone needs to understand why
- **model_validator OR-semantics**: Pydantic models receiving input from branching upstream paths where different branches populate different fields
- **LLM-as-judge 1-10 with anchors**: Any automated quality scoring where you need stability measurement across runs
- **Config-flagged rollout**: New behaviors that change output quality (classifiers, re-rankers, prompt variants)

## Examples

### Pre-committed spike threshold

```
Spike: Verify respx can round-trip Kestrel SSE format
Pass criterion: parsed JSON is byte-identical after record/replay
Fail criterion: SSE event structure deviation causes silent parse failure
Decision gate: If fail -> revert to function-level mocking in all assessment nodes
```

### OR-semantics model_validator

```python
class IntegrationInput(BaseModel):
    direct_findings: Optional[list[Finding]] = None
    cold_start_findings: Optional[list[Finding]] = None

    @model_validator(mode='after')
    def require_at_least_one_source(self) -> 'IntegrationInput':
        if not self.direct_findings and not self.cold_start_findings:
            raise ValueError(
                "IntegrationInput requires direct_findings or cold_start_findings "
                "(or both). Neither was provided — check upstream routing."
            )
        return self
```

### Config flag with deferred flip

```python
class LiteratureGroundingConfig(BaseModel):
    use_llm_classifier: bool = Field(
        default=False,
        description="Enable LLM-based literature classifier. "
        "Shipped False pending quality measurement. "
        "Flip in a separate PR after assessment baselines are established."
    )
```

### Variance band with CV warning

```python
if cv > _CV_WARNING_THRESHOLD:  # 0.5
    status = "warning"  # not "fail"
    logger.warning(
        "Finding count %d outside band [%.1f, %.1f] but CV=%.3f > %.1f — warning only",
        actual, lower, upper, cv, _CV_WARNING_THRESHOLD,
    )
```

## Related

- Origin requirements: `docs/brainstorms/pipeline-cleanup-evals-requirements.md`
- Implementation plan: `docs/plans/2026-04-28-001-refactor-pipeline-cleanup-evals-plan.md`
- Pipeline architecture: `docs/discovery-pipeline.md`
- PRs: #45 (baseline capture), #48 (code cleanup), #49 (state formalization), #50 (assessment Tier 1), #51 (assessment Tier 2)
- GitHub issues: #46 (set/frozenset serializer), #47 (variance truncation on missing runs)
