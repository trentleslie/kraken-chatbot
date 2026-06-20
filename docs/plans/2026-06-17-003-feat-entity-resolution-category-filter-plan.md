---
title: "feat: Category-constrained entity resolution (Tier 1 wrong-namespace fix)"
type: feat
status: active
date: 2026-06-17
origin: docs/wiki/entity-resolution-namespace-fix.outline.md
---

# feat: Category-constrained entity resolution (Tier 1 wrong-namespace fix)

## Overview

Kraken's Tier 1 entity resolution (`resolve_via_api`) calls Kestrel `hybrid_search` with `limit=1`
and **no category constraint**, so it returns whatever same-text node scores highest across all
namespaces. Disease names resolve to metabolic-pathway CURIEs (e.g. `chronic myeloid leukemia` →
`KEGG:05220` instead of `MONDO:0011996`) and silently acquire zero edges downstream. This plan passes
the Biolink **category** that kraken can infer at intake into the `hybrid_search` call, recovering the
correct-namespace node, with a safe fallback to today's behavior whenever no usable hint exists. It is
the kraken-side ("Tier 1") half of the entity-resolution findings; the within-category canonical
ranking ("Tier 2") is owned by biomapper2 and is explicitly out of scope here.

## Problem Frame

Resolution is the load-bearing input to triage, direct-KG, multi-hop, and the (parked) bridge
evidence-provenance labeler. A mis-resolved entity returns zero edges, which triage reads as a false
cold-start, so a resolution error surfaces downstream as "missing biology" rather than a lookup
failure. The dominant error mode is **cross-namespace substitution**: `hybrid_search` ranks by combined
text/vector similarity over names and synonyms across all namespaces at once, and a pathway / ortholog /
consumer-vocabulary entry frequently shares a name string with the intended disease, gene, or chemical
and outranks it. The fix is available client-side: `hybrid_search` accepts a `category` argument, and
kraken's intake already infers an entity type for many entities — but that signal is never passed to
resolution, and intake does not yet infer the **disease** type, which is exactly the most damaging case.
See origin: `docs/wiki/entity-resolution-namespace-fix.outline.md`.

## Requirements Trace

- **R1.** When intake can infer an entity's type, `resolve_via_api` passes the mapped Biolink category to
  `hybrid_search`, so a same-text node from another category cannot win (disease → Disease node, not a
  Pathway node).
- **R2.** Intake infers a **disease** type hint (it currently emits only gene/protein/metabolite), so the
  disease substitution cases — the most damaging and most frequent — are actually covered.
- **R3.** Entities with no hint behave **byte-identically** to today (unconstrained `limit=1` call).
- **R4.** No regression for both error classes. A category-confirmed in-category node **above**
  `tier1_min_score` is accepted (fixes the cross-namespace substitution). An in-category node **below**
  threshold, or an **empty** in-category result, returns `None` and routes to the existing Tier 2 LLM
  resolver — so an ambiguous or token-sharing same-category candidate is adjudicated by the LLM with full
  context rather than silently accepted, and the wrong cross-namespace node is never returned in its place.
- **R5.** If Kestrel **rejects or errors** the category-constrained call (`isError`), resolution degrades
  gracefully to today's unconstrained behavior (does not hard-fail, does not cascade into blanket
  cold-starts) — this is the only path that can still return today's result, and only when the filter
  mechanism itself fails.

## Scope Boundaries

- Not changing the existing score→confidence bands or `tier1_min_score` itself. The only score-logic change
  is *routing*: on the constrained path, a below-threshold or empty in-category result returns `None` (to
  Tier 2) instead of resolving, and an `isError` falls back to the unconstrained call at reduced confidence.
- Not changing the biomapper pre-resolver toggle, its default (off), or its code path behavior.
- Not wiring or enabling the bridge evidence-provenance labeler (separate work, gated on this fix).
- Intake type inference stays heuristic (no LLM added to intake). Routing a low-score/empty in-category
  result to the **existing** Tier 2 LLM resolver is not a new call path — it reuses the pipeline's current
  Tier 1→1.5→2 fall-through; it does mean somewhat more entities can reach Tier 2 than before.

### Deferred to Separate Tasks

- **Tier 2 — within-category canonical / species ranking** (prefer human/HGNC genes; MONDO over ICD/UMLS
  for disease; CHEBI/HMDB over UMLS for metabolite): owned by **biomapper2**, kickoff prompt
  `../biomapper2/AGENT-TASK-canonical-namespace-preference.md`. The category filter here does **not** fix
  same-category wrong-identity (e.g. `VKORC1` → non-canonical `NCBIGene`), and must not try to.
- **Enabling the bridge evidence-provenance labeler** (L2/L3 of
  `docs/plans/2026-06-17-002-feat-bridge-evidence-provenance-labeler-plan.md`): proceeds after this lands.

## Context & Research

### Relevant Code and Patterns

- `backend/src/kestrel_backend/graph/nodes/entity_resolution.py`
  - `resolve_via_api(entity)` (line ~94): the call to harden — `hybrid_search` with `{"search_text", "limit": 1}`,
    no `category`; parses `{search_text: [rows]}`, takes `rows[0]`, maps `score`→confidence, reads
    `categories[0]`.
  - Node body reads `entity_type_hints` from state (line ~604); Tier 1 calls `resolve_via_api(e)` at
    line ~682; the Tier 1.5 **alias** path calls `resolve_via_api(alias)` at line ~723 (second call site).
- `backend/src/kestrel_backend/biomapper_client.py`
  - `_CLASS_TO_BIOLINK` (line ~33): `{gene, protein, metabolite}` → Biolink classes — **no `disease`**.
  - `biolink_class_for(hint)` (line ~49): the shared hint→category mapper to reuse; already used by the
    (default-off) biomapper pre-resolver path in `entity_resolution.py:621/628`.
- `backend/src/kestrel_backend/graph/nodes/intake.py`
  - `detect_entity_types(query, entities)` (line ~368): emits `metabolite` / `protein` / `gene` via
    section headers, gene-symbol casing (`^[A-Z]{2,6}\d?$`), and chemical suffixes. Stored to state as
    `entity_type_hints` (line ~628/646). No disease branch today.
- `backend/src/kestrel_backend/graph/state.py:396`: `entity_type_hints` typed
  `# entity_name -> "metabolite"|"protein"|"gene"` (update the comment when disease is added).
- `.claude/skills/kestrel-api/` and the Kestrel response-shape rules (parse by the real key, fall back to
  empty `[]`/`{}`, never to the container).

### Institutional Learnings

- `docs/solutions/logic-errors/kestrel-direction-param-triage-cold-start-2026-06-11.md` — a stale/unknown
  kwarg to a Kestrel **MCP** tool returns `isError=True` ("Unexpected keyword argument"); a caller that
  treats it as transient cold-started 25/25 entities. **Implication:** `category` must be a verified-accepted
  MCP parameter (confirmed live 2026-06-17: `category` passes MCP schema validation; `node_categories` is
  rejected), and an `isError` on the constrained call must fall back, not silently drop or cascade.
- `docs/solutions/logic-errors/kestrel-multi-hop-response-shape-paths-key-2026-06-11.md` — container-fallback
  parsing (`.get(key, data)`) masked a shape mismatch for 3.5 months; wrong-shape mocks certified the bug.
  **Implication:** new tests mock the **real** `hybrid_search` shape (`{search_text: [rows]}`, rows carrying
  `id`/`name`/`categories`/`score`); never fall back to the container.
- `docs/solutions/runtime-errors/triage-tier2-undefined-variable-2026-05-06.md` — fallback branches in
  resolution/triage have historically shipped untested because the happy path always succeeds.
  **Implication:** the new fallback branch (constrained-miss → unconstrained) must have a test that
  actually exercises it.

### External References

- **Reconfirmed live against the MCP endpoint 2026-06-17** (raw `call_kestrel_tool("hybrid_search", …)`):
  - Unconstrained `chronic myeloid leukemia` → `KEGG:05220` (`biolink:Pathway`, **score 4.86**) outranks
    `MONDO:0011996` (**score 2.495**) — the wiki report's numbers exactly.
  - `category=biolink:Disease` → `MONDO:0011996` at **rank 1** (score 2.495); the KEGG pathway is **absent**
    from the filtered results (filter genuinely filters — it does not return best-effort non-matching rows).
- **Resolved — `category` matching is list-membership, not exact-single-string (subclass concern is moot).**
  `MONDO:0011996`'s `categories` are `["biolink:DiseaseOrPhenotypicFeature", "biolink:Disease",
  "biolink:PhenotypicFeature"]`, and `category=biolink:Disease` still recovered it at rank 1. So a node
  matches when the requested class appears **anywhere** in its categories array; passing `biolink:Disease`
  is safe even for nodes that also carry Disease subclasses. (The earlier "exact-string vs subclass" open
  premise is closed.) Within `biolink:Disease`, non-MONDO disease nodes (`ICD9:205.12` @0.85) also appear —
  confirming the MONDO-vs-ICD within-category ranking is a real **Tier 2 / biomapper2** concern, as scoped.

## Key Technical Decisions

- **Reuse `biolink_class_for` + `_CLASS_TO_BIOLINK`; add one `disease` entry.** Single source of truth for
  the hint→category mapping, already shared with the biomapper path — do not introduce a second mapping.
- **Server-side `category` filter, `limit` stays 1.** Pass `category` to `hybrid_search` (Kestrel performs
  the demotion server-side — verified to promote MONDO to rank 1 for CML/Parkinson) and keep `limit=1`: the
  filtered rank-1 row *is* the best in-category candidate, so no client re-rank or raised limit is needed.
  (This deviates from the origin report's "raise `limit` so a re-rank is possible" — that was premised on
  client-side re-ranking; the server-side filter obviates it, and within-category re-ranking is Tier 2's
  job, owned by biomapper2.)
- **Routing, not blind acceptance — the regression-critical decision (revised after review).** Category-
  confirmation is *not* identity-confirmation: a `category=biolink:Disease` filter can still return a
  token-sharing **wrong** disease (CML vs CLL vs AML). And resolution confidence is **display-only** — it
  does not gate Tier 2 (a non-`None` Tier-1 result simply skips the LLM tier; verified in
  `entity_resolution.py:693-696`, `synthesis.py:135`). So "accept any in-category row at modest confidence"
  would *silently* resolve a wrong same-category node and suppress the LLM that could correct it. Instead,
  branch on score: an in-category hit **above** `tier1_min_score` is accepted via the existing bands (the
  win — for CML/Parkinson the MONDO node scores ~2.49, well above); an in-category hit **below** threshold,
  or an **empty** in-category result, returns `None` so the entity routes to the **existing Tier 2 LLM
  resolver**, which adjudicates with full context. This fixes the cross-namespace bug without trading a
  visible failure (cold-start) for a silent one (wrong same-category node), and needs **no** new confidence
  constant for acceptance.
- **`isError` → unconstrained fallback (the only degradation path).** If the constrained call errors
  (`isError`) — e.g. a future `category` arg-drift — fall back to the unconstrained `limit=1` call (today's
  behavior) at a reduced confidence and a fallback `method` marker, and log it. This is the only branch that
  can still return today's result, and only when the filter mechanism itself fails; empty/low-score do **not**
  fall back (they route to Tier 2). No hint → behave exactly as today (R3).
- **Constrain both call sites (primary and alias), sharing the same logic (revised after review).** The
  Tier 1.5 alias call reuses the identical branch: pass the parent entity's category; `isError`→unconstrained
  fallback, empty/low-score→`None` (Tier 2). This recovers **type-preserving** aliases (a synonym of the same
  disease, the common case) instead of leaving them on the wrong-namespace bug, while a **type-mismatched**
  alias (e.g. a gene-name alias of a disease) yields an empty in-category result and routes to Tier 2 exactly
  as an over-fired hint does — no special-casing needed.
- **`category` is an optional `resolve_via_api` parameter defaulting to `None`.** Mirrors the existing
  "flag-off -> byte-identical" property already honored by the biomapper block; keeps the unhinted path
  unchanged and the diff reviewable.

## Open Questions

### Resolved During Planning

- *Does the MCP `hybrid_search` accept `category`?* Yes — verified live 2026-06-17 (`category` passes schema
  validation; `node_categories` is rejected with `unexpected_keyword_argument`).
- *Does a hint→category mapping already exist?* Yes — `biolink_class_for` / `_CLASS_TO_BIOLINK`; it only
  lacks a `disease` entry.
- *Is `entity_type_hints` available where `resolve_via_api` is called?* Yes — read into the node body at
  `entity_resolution.py:604`, in scope for both the Tier 1 loop and the alias loop.
- *Does `category` match exact-string or Biolink-subclass?* **List-membership** — verified live 2026-06-17:
  `category=biolink:Disease` recovers `MONDO:0011996` at rank 1 even though that node also lists
  `biolink:DiseaseOrPhenotypicFeature` and `biolink:PhenotypicFeature`. Passing `biolink:Disease` is safe.
- *Does a `category` filter return empty when no in-category node matches, or best-effort rows?* It
  **genuinely filters** — verified live: under `category=biolink:Disease` the KEGG pathway (rank 1
  unconstrained) is absent; only Disease-category nodes are returned.

### Deferred to Implementation

- The single fixed reduced-confidence value for the `isError`→unconstrained fallback path. Pick one small
  constant (no new runtime config); name it next to the existing `tier1_min_score`. (No acceptance constant
  is needed any more — below-threshold hits route to Tier 2, not accepted.)
- The exact `disease_patterns` reuse mechanism. **Re-apply the regex list against each entity token**
  (`re.search(pattern, entity.lower())` per entity), **not** against the whole query, and **not** via
  equality with `extract_study_context`'s canonical `disease_focus` label — that label is a normalized string
  (e.g. `"Parkinson's disease"`) that need not match the entity token text, so `token == disease_focus`
  would frequently miss. The lexicon (regex list) is the reusable asset; the token-level application is new.
- Metabolite hint mapping stays `biolink:SmallMolecule` (current `_CLASS_TO_BIOLINK` value). Not in scope —
  the shared mapping is not re-tuned here.
  (The `category` matching semantics and empty-vs-best-effort behavior are now **resolved** live — see
  Open Questions › Resolved During Planning — so no pre-merge Kestrel probe remains outstanding.)

## Implementation Units

- [ ] **Unit 1: Infer a disease type hint at intake**

**Goal:** `detect_entity_types` emits a `"disease"` hint so disease entities carry a type signal into
resolution (R2).

**Requirements:** R2.

**Dependencies:** None.

**Files:**
- Modify: `backend/src/kestrel_backend/graph/nodes/intake.py` (`detect_entity_types`)
- Modify: `backend/src/kestrel_backend/graph/state.py` (update the `entity_type_hints` value-comment to include `disease`)
- Test: `backend/tests/test_intake.py` (create; or extend the existing `detect_entity_types` coverage in `backend/tests/test_entity_resolution_biomapper.py`)

**Approach:**
- Two complementary signals, because disease names arrive both ways:
  1. **Section headers** (`diseases:`, `disease:`, `conditions:`, `significant diseases:`) mapped to
     `"disease"`, parallel to the existing metabolite/protein/gene header lists — covers the labeled
     multi-omics input shape.
  2. **Reuse the existing curated disease-regex list, applied per token.** `extract_study_context` maintains
     `disease_patterns` (intake.py ~line 468: diabetes/t2d, alzheimer, parkinson, cancer/tumor/carcinoma,
     cardiovascular, obesity, …) — but that function runs the regexes over the **whole query** and returns a
     normalized `disease_focus` **label**, not a per-token tag. So **re-apply the same regex list against
     each candidate entity token** (`re.search(pattern, entity.lower())` per entity); do **not** compare the
     token to `disease_focus` (the canonical label need not equal the token text). This reuses the maintained
     lexicon (the regex list) while adding only the token-level application, and gives real recall on prose
     queries — where headers fire almost nowhere and the most-frequent disease mentions appear.
- **Precedence:** the gene-symbol-casing (`^[A-Z]{2,6}\d?$`) and chemical-suffix heuristics take priority for
  clear gene/metabolite tokens; the disease lexicon must not override them. A genuine over-fire is still
  absorbed by Unit 3's empty→fallback (no regression), but precision is preferred so the category benefit is
  not lost on a mislabeled gene/metabolite.

**Patterns to follow:** the existing header-list + `rfind`-before-entity blocks in `detect_entity_types`;
the `disease_patterns` regex list in `extract_study_context`.

**Test scenarios:**
- Happy path (header): a query with a `Diseases:` / `Conditions:` header followed by
  `chronic myeloid leukemia` / `Parkinson disease` → those entities get hint `"disease"`.
- Happy path (prose lexicon): a prose query mentioning `Parkinson's disease` / `type 2 diabetes` with no
  header → the disease token gets hint `"disease"` via the reused `disease_patterns`.
- Edge case: a gene token (`VKORC1`, matches `^[A-Z]{2,6}\d?$`) → stays `"gene"`, not disease (precedence).
- Edge case: a metabolite token (suffix `-ose`/`-ate`) with no disease signal → stays `"metabolite"`.
- Edge case: entity not found in the query string → no hint emitted (unchanged behavior).
- Regression: existing metabolite/protein/gene header and heuristic cases still produce their current hints;
  `extract_study_context`'s own `disease_focus` output is unchanged (read-only reuse, not a mutation).

**Verification:** `detect_entity_types` returns `"disease"` for both header-tagged and prose disease names
(via the reused lexicon) while all existing gene/protein/metabolite outcomes are unchanged.

- [ ] **Unit 2: Add `disease` to the shared hint→Biolink mapping**

**Goal:** `biolink_class_for("disease")` returns `biolink:Disease` so the disease hint maps to a category
(R1, R2).

**Requirements:** R1, R2.

**Dependencies:** None (independent of Unit 1; both feed Unit 3).

**Files:**
- Modify: `backend/src/kestrel_backend/biomapper_client.py` (`_CLASS_TO_BIOLINK`)
- Test: `backend/tests/test_entity_resolution_biomapper.py` (existing `biolink_class_for` coverage)

**Approach:**
- Add `"disease": "biolink:Disease"` to `_CLASS_TO_BIOLINK`. No signature change.
- **Cross-effect to note:** this also lets the default-off biomapper pre-resolver attempt disease entities
  when it is enabled. That is consistent and desirable (biomapper2 Tier 2 handles disease MONDO ranking),
  but call it out in the PR; it changes no behavior while the flag is off.

**Patterns to follow:** the existing `_CLASS_TO_BIOLINK` entries.

**Test scenarios:**
- Happy path: `biolink_class_for("disease")` → `"biolink:Disease"`; case/space-insensitive (`" Disease "`).
- Regression: `gene`/`protein`/`metabolite` still map as before; unknown/`None` hint → `None`.

**Verification:** `biolink_class_for` maps disease and the three existing types correctly; unknown → `None`.

- [ ] **Unit 3: Category-constrained `resolve_via_api` with safe fallback**

**Goal:** `resolve_via_api` accepts an optional Biolink `category`, passes it (with `limit=1`) to
`hybrid_search`, **accepts an in-category hit above `tier1_min_score`**, **routes a below-threshold or empty
in-category result to the existing Tier 2 LLM** (returns `None`), and falls back to today's unconstrained
behavior only on `isError`; the category is threaded from intake hints at **both** the Tier 1 primary and
the Tier 1.5 alias call sites (R1, R3, R4, R5).

**Requirements:** R1, R3, R4, R5.

**Dependencies:** Unit 2 (mapping). Unit 1 (disease hint) is the hard prerequisite for any *disease* effect —
without it, disease entities carry no hint and this unit is a no-op for them. The signature/logic can be
built and unit-tested against Unit 2 alone (with explicit category arguments); the disease end-to-end
scenario requires Unit 1.

**Files:**
- Modify: `backend/src/kestrel_backend/graph/nodes/entity_resolution.py` (`resolve_via_api`; **both** call
  sites threaded — Tier 1 primary ~line 682 and Tier 1.5 alias ~line 723)
- Test: `backend/tests/test_entity_resolution_category.py` (create)

**Approach:**
- **No `build_resolution` helper exists today** — the score→confidence mapping and `EntityResolution`
  construction are inlined in `resolve_via_api` (~lines 164-201), and that inline block contains the exact
  `score <= tier1_min_score -> return None` branch (~lines 171-179). The constrained path reuses this block
  **as-is**: an in-category hit above threshold maps through the existing bands; an in-category hit at/below
  threshold returns `None` (today's reject) which now routes the entity to Tier 2 — exactly the desired
  behavior, no bypass needed. So the only structural change is adding the `category` argument and the
  isError-fallback branch; the existing reject branch is *kept*, not bypassed.
- Add `category: str | None = None` to `resolve_via_api`. When `None`: current call (`limit=1`, no category) —
  byte-identical path (R3).
- When `category` is set: call `hybrid_search` with `category` and `limit=1`. Parse the real
  `{search_text: [rows]}` shape (parse by the real key; never `.get(key, data)` container fallback). Then:
  - **`isError`** → log a fallback event and re-issue the unconstrained `limit=1` call (today's behavior); on
    success return it at a **reduced** confidence (named constant) with the existing `method` field marked as
    an unconstrained fallback (reuse the `method` convention — no new field). (R5)
  - **empty rows** → return `None` (entity routes to Tier 1.5/Tier 2; the LLM adjudicates rather than
    accepting the wrong cross-namespace node). (R4)
  - **in-category row present** → run the existing score→confidence block unchanged: above `tier1_min_score`
    → accept; at/below → `None` → Tier 2. (R1, R4)
- **Recorded category** stays the node's own `categories[0]` (as today) — the node's true category, not
  necessarily the query string. Assert on the resolved **CURIE/namespace** (MONDO), not a category-string
  equality, in tests.
- Thread the category at **both** sites via `biolink_class_for(entity_type_hints.get(entity))` where `entity`
  is the original entity name (the hint key — at the alias site too, the parent entity's name, **not** the
  alias string). `None` when no/unknown hint.

**Execution note:** Implement test-first — the isError-fallback and below-threshold→Tier 2 branches are the
historically untested paths (see `triage-tier2-undefined-variable`), so write the below-threshold→`None` and
the isError→unconstrained-fallback tests before the logic, and assert each executes the intended call.

**Technical design:** *(directional guidance for review, not implementation specification)*

    resolve_via_api(entity, category=None):
        if category is None:
            return current_behavior(entity)            # limit=1, no category — unchanged (R3)
        rows = hybrid_search(entity, category=category, limit=1)    # server-side filter, ranked
        if rows.isError:
            log FALLBACK_EVENT(reason=error)
            res = current_behavior(entity)             # unconstrained fallback, reduced confidence (R5)
            return res with reduced confidence if res else None
        if rows is empty:
            return None                                # -> Tier 1.5/Tier 2 LLM (R4)
        # in-category hit: existing score->confidence block UNCHANGED
        #   score > tier1_min_score -> accept; score <= tier1_min_score -> None -> Tier 2 (R1/R4)
        return existing_inline_resolution(rows[0])

**Patterns to follow:** the existing `resolve_via_api` parse/confidence block (kept intact); the
`FALLBACK_EVENT node=entity_resolution reason=...` log style used by the biomapper block (lines ~638/644/650);
the existing `method` values on `EntityResolution` (`"biomapper"`, `"alias:..."`); the "flag-off ->
byte-identical" discipline in the biomapper block.

**Test scenarios:**
- Happy path (constrained hit, above threshold): hint→`biolink:Disease`; mocked constrained `hybrid_search`
  returns a MONDO Disease row at score 2.49 → resolves to the MONDO **CURIE** (assert CURIE/namespace, not a
  category string).
- **Regression-critical (low-score in-category routes to Tier 2, not the wrong node):** constrained returns
  one MONDO row at score 0.55 (below `tier1_min_score`) → `resolve_via_api` returns `None` (entity reaches
  Tier 2); assert it does **not** return a node, and specifically does **not** fall back to and accept the
  unconstrained KEGG pathway. This is the core no-regression proof for R4 (the ambiguous case defers to the
  LLM rather than silently accepting a wrong cross- or same-category node).
- Happy path (no hint): `category=None` → exactly one unconstrained `limit=1` call, result identical to today.
- Error path (constrained `isError`): returns `isError=True` → falls back to the unconstrained call; resolves
  with reduced confidence and the fallback `method` marker; no exception (R5).
- Edge case (constrained empty): zero in-category rows (e.g. over-fired disease hint on a gene, or a
  type-mismatched alias) → returns `None` → Tier 2; assert no unconstrained fallback fired.
- Alias path (constrained): the Tier 1.5 alias call is invoked **with** the parent entity's category;
  a type-preserving alias returning an in-category MONDO row resolves to it; a type-mismatched alias returns
  empty → `None`. Assert the alias call carries the parent's category and keys the hint on the parent name.
- Real-shape guard: all mocks use the real `{search_text: [rows]}` envelope with `id`/`name`/`categories`/
  `score`; a wrong-shape mock is not used to assert success.

**Verification:** above-threshold in-category hits resolve to in-category CURIEs; below-threshold/empty
in-category results return `None` and reach Tier 2; `isError` falls back to today's behavior; unhinted
entities are byte-identical to today; both call sites are threaded and the alias keys on the parent name; the
below-threshold→Tier 2 and isError-fallback branches are each covered by a test that proves they execute.

## System-Wide Impact

- **Interaction graph:** `intake.detect_entity_types` → state `entity_type_hints` → `entity_resolution`
  node → `resolve_via_api` (both the Tier 1 primary and the Tier 1.5 alias call sites). Downstream consumers
  (triage, direct-KG, integration) are unchanged in interface; they receive better CURIEs.
- **Tier 2 load:** routing below-threshold/empty in-category results to `None` means somewhat **more entities
  reach the Tier 2 LLM** than today (previously a high-score wrong node would have resolved at Tier 1). This
  is the intended cost of correctness; it is bounded to hinted entities whose in-category match is weak/absent.
- **Error propagation:** a constrained `hybrid_search` `isError` must fall back, not raise and not return a
  blanket `None` that could be read as cold-start; mirror the documented direction-param lesson.
- **State lifecycle risks:** none new; resolution remains stateless per entity. The `entity_type_hints`
  state key gains a `disease` value (update its type comment).
- **API surface parity:** the biomapper pre-resolver path already category-maps via `biolink_class_for`;
  this change brings the **default Kestrel path** to parity. **Both** call sites of `resolve_via_api` are
  threaded with the same logic; a test asserts the alias call carries the parent's category and keys the hint
  on the parent entity name (not the alias string).
- **Integration coverage:** a near-end-to-end test from a disease-mentioning query (header **and** prose)
  through `detect_entity_types` → `biolink_class_for` → a mocked constrained `hybrid_search` to an
  in-category CURIE proves the chain that unit mocks alone will not. **Owner:** Unit 3's test file
  (`backend/tests/test_entity_resolution_category.py`), added once Unit 1 lands so the disease hint is real.
- **Unchanged invariants:** the unhinted resolution path is byte-identical to today, the existing
  score→confidence bands and `tier1_min_score` are unchanged (the constrained path *reuses* them), and the
  biomapper toggle/default is unchanged. No new acceptance threshold is introduced — the only behavior change
  is routing (below-threshold/empty in-category → Tier 2; `isError` → unconstrained fallback).

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| ~~Kestrel matches `category` exact-string, excluding Disease-subclass nodes~~ — **RESOLVED** | Verified live 2026-06-17: matching is list-membership; `category=biolink:Disease` recovers `MONDO:0011996` at rank 1 despite its subclass categories. No longer a risk. |
| Correct in-category node scores below `tier1_min_score` → would be discarded and fall back to the wrong node | Below-threshold in-category results return `None` and route to the **Tier 2 LLM** (not to the unconstrained wrong node); the LLM adjudicates with full context. Covered by the low-score→Tier 2 test. |
| **Category-confirmation ≠ identity-confirmation: a token-sharing wrong same-category node accepted silently** | Only above-threshold in-category hits are accepted; below-threshold ones route to Tier 2 rather than being blindly accepted, so an ambiguous CML-vs-CLL-vs-AML case reaches the LLM. A high-score *wrong same-category* node is the residual Tier-2/biomapper2 concern (within-category ranking), explicitly out of scope here. |
| Disease heuristic over-fires and mislabels a gene/metabolite as disease | Empty in-category result → `None` → Tier 2 LLM (no worse than today); gene/metabolite heuristics take precedence over the disease lexicon; test the not-disease cases. |
| Disease heuristic under-fires (headers absent in prose) → fix never triggers | Reuse the existing `extract_study_context` `disease_patterns` regex list **applied per token** for prose recall, not headers-only; test a prose disease scenario. |
| Kestrel `category` semantics undocumented in the repo reference / could drift | Re-confirm live against the MCP endpoint before merge (a `@pytest.mark.integration` probe asserting `category` is accepted and recovers MONDO for CML); already verified 2026-06-17. |
| Wrong-shape test mocks certify a broken filter (historical failure) | Mock the real `{search_text: [rows]}` shape; no container-fallback parsing. |
| Routing / fallback branches ship untested (historical failure) | Execution-note test-first; assert each branch executes (above-threshold accept, below-threshold→Tier 2, empty→Tier 2, isError→unconstrained fallback). |
| More entities reach Tier 2 LLM (latency/cost) than today | Bounded to hinted entities with weak/absent in-category matches; the common case (MONDO scores ~2.49, well above threshold) still resolves at Tier 1. Accepted tradeoff for correctness. |

## Documentation / Operational Notes

- Update the `entity_type_hints` comment in `state.py` to include `disease`.
- The published wiki report (`docs/wiki/entity-resolution-namespace-fix.outline.md`) already describes this
  as Tier 1 and names the disease-hint prerequisite; no wiki change required, but link the PR to it.
- No migration, no env var, no rollout flag (the change is on by virtue of hints existing; unhinted entities
  are unaffected).

## Sources & References

- **Origin document:** `docs/wiki/entity-resolution-namespace-fix.outline.md` (published BioMapper-collection report)
- Related (distinct) effort: `docs/plans/2026-06-11-001-feat-biomapper-entity-resolution-plan.md` and
  `docs/brainstorms/2026-06-04-biomapper-entity-resolution-prefilter.md` — the default-off biomapper
  pre-resolver, a different approach to the same root cause.
- Tier 2 follow-on (biomapper2): `../biomapper2/AGENT-TASK-canonical-namespace-preference.md`.
- Gated downstream: `docs/plans/2026-06-17-002-feat-bridge-evidence-provenance-labeler-plan.md`.
- Code: `entity_resolution.py` (`resolve_via_api`), `biomapper_client.py` (`biolink_class_for` /
  `_CLASS_TO_BIOLINK`), `intake.py` (`detect_entity_types`).
- Learnings: `docs/solutions/logic-errors/kestrel-direction-param-triage-cold-start-2026-06-11.md`,
  `docs/solutions/logic-errors/kestrel-multi-hop-response-shape-paths-key-2026-06-11.md`,
  `docs/solutions/runtime-errors/triage-tier2-undefined-variable-2026-05-06.md`.
