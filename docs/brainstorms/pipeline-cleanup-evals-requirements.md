---
date: 2026-04-28
topic: pipeline-cleanup-formalization-evals
---

# Discovery Pipeline: Cleanup, Formalization, and Evaluation Infrastructure

## Problem Frame

The LangGraph discovery pipeline (~8,100 lines across 10 nodes) is functionally complete and produces research-grade output, but has accumulated technical debt that makes it harder to maintain, extend, and validate. Repeated SDK boilerplate across 8 nodes, legacy state fields, undocumented threshold choices, and zero evaluation infrastructure mean that changes are risky (no regression detection) and output quality is unmeasured (no scoring framework). Additionally, the literature grounding node hardcodes all relationship classifications to "supporting" (see TODO in `state.py:223`), producing inaccurate metadata that degrades hypothesis quality. This effort brings the pipeline to production-grade engineering standards and establishes the measurement foundation needed for confident iteration.

## Requirements

**Baseline Capture**

- R1. Capture pipeline outputs from at least 10 representative queries (minimum 2 per pipeline path: well-characterized, sparse, cold-start, longitudinal, multi-entity) before any code changes. Run each query 5 times to establish variance baselines. Store the pipeline outputs, all external API responses (Kestrel KG, OpenAlex, Semantic Scholar, Exa, PubMed), and per-run variance data. Cached API responses enable eval runs without any live API dependency; LLM non-determinism across the 8 nodes that use Claude SDK calls remains and is handled by tolerance bands in R9. After R1 completes, compute tolerance bands (e.g., mean +/- 2 standard deviations per metric) and persist them alongside the baseline for R9 consumption.

**Code Cleanup**

- R2. Extract shared SDK setup (try/except imports, `HAS_SDK`, `McpStdioServerConfig`) into a single shared module, eliminating the 8-node duplication.
- R3. Consolidate repeated SDK boilerplate and common patterns across nodes into shared utilities. The primary (HTTP API) / fallback (SDK) pattern varies meaningfully across nodes (entity_resolution does per-entity fallback, triage does batch fallback, cold_start has no SDK fallback at all), so consolidation should target the genuinely duplicated parts (SDK imports, MCP config, semaphore setup) rather than forcing a single abstraction over divergent fallback logic.
- R4. Parameterize currently hardcoded values into a central configuration with documented rationale for each default. Hub edge thresholds must remain per-node (pathway_enrichment uses 1000 for shared-neighbor filtering; direct_kg uses 5000 for entity-level hub bias — these serve different purposes). Semaphore concurrency values must remain per-node with documented constraints (entity_resolution/triage use `Semaphore(1)` to prevent CLI spawn conflicts; direct_kg/cold_start use higher values for batch parallelism). Entity limits (top 5 sparse + top 3 cold-start) should be configurable.
- R5. Remove legacy/deprecated state fields (`NoveltyTriage`, `well_characterized_ids`, `sparse_ids`, `kg_results`, `predictions`, `pathway_enrichment`, `legacy_bridges`, `gap_analysis`) after verifying they are unused in active code paths. Verification must cover: (a) no node reads the field, (b) no node writes to the field, (c) `node_detail_extractors.py` — the `_extract_pathway_enrichment` function (line 197) and its registration (line 394) must be updated or removed when the legacy `pathway_enrichment` dict field is removed. Note: `pathway_enrichment` as a state field (legacy dict) is distinct from the active `pathway_enrichment` node. Also remove `.bak` and `.phase4b` files from the graph directory.

**State Formalization**

- R6. Add inter-node state validation that verifies required fields are present and correctly typed after each node executes, catching contract violations at the node boundary rather than downstream.
- R7. Document the state contract for each node (reads/writes/requires) in a format that is both human-readable and machine-checkable.

**Eval Infrastructure — Regression (Eval Tier 1)**

- R8. Build an eval runner that: (a) accepts a dataset of queries with optional cached API responses, (b) executes the pipeline in either live or replay mode — replay intercepts all external HTTP calls (Kestrel, OpenAlex, Semantic Scholar, Exa, PubMed) and serves cached responses, but LLM calls (Claude SDK `query()`) still execute normally — this eliminates external API dependency, not LLM non-determinism, (c) runs the structural checks from R9, and (d) produces a JSON report per R10. Must be invocable with a single command.
- R9. Implement structural checks using tolerance bands computed from R1 baseline variance data (R1 must complete before R9 tolerance bands can be finalized): pipeline completion (all expected nodes executed), output schema conformance, entity resolution recall against known CURIEs, finding count stability (tolerance band from R1 variance, not a fixed percentage), hypothesis structural completeness (all required fields present).
- R10. Eval results must be serializable to JSON for storage, diffing across runs, and future ingestion by expert-in-the-loop tooling.

**Eval Infrastructure — Quality (Eval Tier 2)**

- R11. Implement LLM-as-judge scoring for hypothesis biological plausibility, finding relevance to query intent, and hypothesis novelty (non-obviousness given input). Scorer must use temperature 0 and fixed prompts. Scorer operates on frozen pipeline outputs (one selected baseline run per query), not re-executed pipeline runs, to isolate scorer variance from pipeline variance. Stability criterion: run the scorer 5 times on frozen outputs from the full baseline query set and compute Spearman correlation between each pair of score vectors. Mean pairwise correlation must be >= 0.80 before the scorer is considered calibrated.
- R12. Design a preliminary eval dataset format (query + pipeline output + scores) documented with JSON schema and at least one example. Format is forward-looking for expert-in-the-loop consumption and should be versioned to allow breaking changes if expert-in-the-loop requirements diverge from initial assumptions.
- R13. The eval dataset format must include a `human_judgment` field per hypothesis that is initially null and can be populated when expert-in-the-loop is ready. No ingestion pipeline needs to be built now — the format compatibility is sufficient.

**Refinement (Measured)**

- R14. Improve the literature relationship classification beyond the current hardcoded "supporting" default (see `state.py:223` TODO), validated by eval quality scores before and after. This addresses a known accuracy deficiency in pipeline output, not a new feature. Changes must be scoped to the literature grounding node; if implementation requires new nodes or cross-node architectural changes, escalate to scope review.
- R15. Add observability for primary (HTTP) → fallback (SDK) events so it is clear which entities required SDK fallback and why, without disrupting the user-facing output.
- R16. Any refinement to node logic must demonstrate non-regression on Eval Tier 1 checks and improvement (or neutrality) on Eval Tier 2 quality scores. For the first refinement (R14), treat eval results as informational rather than gating, since the eval infrastructure itself is new and uncalibrated. After the first refinement validates that evals produce sensible signals, subsequent refinements should treat eval results as gating.

## Success Criteria

- All existing unit tests continue to pass after cleanup/formalization (zero behavior change in refactor phases).
- State validation (R6) catches contract violations at node boundaries; state contracts (R7) are documented for all 10 nodes.
- Eval Tier 1 (structural/regression) runs locally with a single command and produces a pass/fail report. Supports both live and replay modes. Tolerance bands derived from empirical baseline variance.
- Eval Tier 2 (quality) produces numeric scores per hypothesis and per query on frozen baseline outputs. Stability validated: mean pairwise Spearman correlation >= 0.80 across 5 scorer runs on the full baseline query set.
- Baseline snapshot exists for at least 10 representative queries (minimum 2 per pipeline path), each run 5 times for variance measurement, with cached API responses (Kestrel + literature sources) for replay mode.
- Eval output format is documented with JSON schema, includes `human_judgment` field for future expert-in-the-loop use, and has at least one complete example.
- Hub threshold and concurrency parameters are configurable per-node with documented rationale for each default value and its constraints.

## Scope Boundaries

- **In scope**: Pipeline code cleanup, state formalization, eval framework, quality scoring, measured refinements (including R14 literature classification fix as a known deficiency, scoped to the literature grounding node).
- **Out of scope**: Expert-in-the-loop UI development or ingestion pipeline (eval format compatibility is sufficient). Frontend changes. New pipeline nodes. Kestrel API changes.
- **Out of scope**: Performance optimization beyond what falls out naturally from cleanup. Dedicated performance benchmarking infrastructure.

## Key Decisions

- **Sequence: Snapshot → Cleanup → Evals → Refine**: Capturing baseline before any changes ensures regression detection. Cleaning up before building evals means instrumentation targets stable code. Refinements come last because they require evals to validate.
- **Five isolated PRs**: Each phase is a standalone PR for Greptile review. Pure refactors are separated from new infrastructure to keep reviews focused.
- **Cached API replay for external-API-independent evals**: Baseline capture records all external HTTP responses (Kestrel KG, OpenAlex, Semantic Scholar, Exa, PubMed) alongside pipeline outputs. Eval runner replays these to eliminate external API dependency and data churn. LLM non-determinism (8/10 nodes use Claude SDK `query()`) is inherent and handled by empirically calibrated tolerance bands, not by replay.
- **Scorer operates on frozen outputs**: Eval Tier 2 quality scoring runs on fixed pipeline outputs (one selected baseline run), not re-executed pipeline runs, to isolate scorer variance from pipeline variance.
- **LLM-as-judge for quality**: No human-labeled ground truth exists yet. LLM scoring bootstraps the quality signal. Expert-in-the-loop will progressively upgrade this as it comes online.
- **Eval dataset is append-only**: New queries and expert judgments add to the dataset; existing entries are never deleted, only supplemented.
- **Consolidate boilerplate, not fallback logic**: R3 targets genuinely duplicated code (SDK imports, MCP config) rather than forcing a single abstraction over the primary/fallback pattern, which varies meaningfully across nodes.
- **First refinement is informational, not gating**: R16 eval gating bootstraps gradually — the first refinement (R14) validates that evals produce sensible signals before subsequent refinements are gated by them.
- **Terminology**: "Primary (HTTP) / fallback (SDK)" refers to the API resilience pattern in pipeline nodes. "Eval Tier 1 / Eval Tier 2" refers to eval categories (structural vs. quality). These are unrelated concepts.

## Dependencies / Assumptions

- Kestrel API and literature APIs (OpenAlex, Semantic Scholar, Exa, PubMed) must be accessible for baseline capture (one-time, 5 runs per query). After baseline is captured, eval runs can use replay mode without any external API access.
- Claude Agent SDK must be available for pipeline execution during eval runs (LLM calls cannot be replayed).
- Expert-in-the-loop project (`projects/expert-in-the-loop/`) is not yet ready — eval format includes `human_judgment` placeholder field but no ingestion pipeline is needed now.
- LLM-as-judge scoring requires API access to a capable model (Claude or equivalent) with temperature 0 support.

## Outstanding Questions

### Resolve Before Planning

(None — all product decisions are captured above.)

### Deferred to Planning

- [Affects R3][Needs research] Which parts of the SDK boilerplate are truly identical across nodes vs. which have meaningful per-node variation? Determines what gets extracted vs. left in place.
- [Affects R4][Technical] Document the rationale for each per-node hub threshold and semaphore value by reading the code comments and commit history.
- [Affects R5][Technical] Verify whether the pathway_enrichment node writes to the `pathway_enrichment` state field or only to `shared_neighbors` and `biological_themes`. Also verify that `_extract_pathway_enrichment` in `node_detail_extractors.py` reads from the legacy dict field vs. the node's active output fields.
- [Affects R6][Technical] Should state validation be a LangGraph middleware/callback or explicit checks at node entry/exit?
- [Affects R7][Technical] What format should state contracts use? Options: Python docstrings with structured fields, separate YAML/JSON schema files, or inline type annotations with runtime validation.
- [Affects R8][Technical] Should cached API responses use VCR-style HTTP-layer recording (captures all external calls in one mechanism) or custom per-client serialization? VCR-style is recommended given 5 distinct external APIs.
- [Affects R11][Needs research] Which LLM and prompt strategy produces the most stable/calibrated quality scores for biomedical hypothesis evaluation? Should a different model judge to avoid self-evaluation bias (pipeline uses Claude, judge could use a different model)?

## Next Steps

-> `/ce:plan` for structured implementation planning across the five PR phases.
