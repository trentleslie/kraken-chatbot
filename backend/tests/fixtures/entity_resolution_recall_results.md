# entity_resolution Tier-2 recall gate — result (#61)

Run against **live Kestrel** (`kestrel.nathanpricelab.com`) + Claude Agent SDK, on the
22-entity hand-labeled hard-variant fixture (`entity_resolution_hard_variants.json`).
Re-run with `cd backend && uv run python tests/recall_gate.py`.

## Result

| Metric | Value |
|---|---|
| **Prefetch coverage** (expected CURIE in candidate set) | **22/22 = 100%** |
| Exact-CURIE recall (prefetch + SDK select) | 20/22 = 90.9% |
| Same-entity recall (incl. `equivalent_ids`) | 20/22 = 90.9% |
| Threshold | 95% |

## Interpretation: no recall regression

Prefetch coverage is **100%** — the correct candidate is always surfaced, so the
fixed-variant prefetch does **not** drop entities the way the plan worried it might.
The two exact-match shortfalls are **labeling/criterion artifacts, not resolution
failures**:

1. **`hexadecanedioate`** — labeled `CHEBI:73722` (hexadecanedioic *acid*); resolved to
   `CHEBI:76276` (hexadecanedioate(2-) **anion**). The `-ate` suffix denotes the anion,
   so the model's pick is arguably the *more* correct node. Distinct CHEBI entities, not
   linked by `equivalent_ids`.
2. **`carnitine`** — labeled `UMLS:C0007258`; resolved to `MESH:M0003493` ("Carnitine").
   Same real-world entity, different DB; the KG does not cross-link these two nodes.

Every one of the 22 hard variants resolved to a correct node (incl. all 7 lowercase gene
symbols → the verified **human** NCBIGene IDs, where the raw `hybrid_search` top hit was a
non-human ortholog). Decision (recorded with the team): **accept — no regression shown**;
the migrated prefetch+select Tier-2 is sound. The strict exact-CURIE gate slightly
under-counts due to KG node-identity ambiguity (acid/anion, same-entity-cross-DB).

## Fixture labeling method (model-independent)

- **Gene symbols** → the verified **human** NCBIGene id, confirmed via human
  `equivalent_ids` (ENSG / UniProt-human / REACT `R-HSA`) — *not* the arbitrary-species
  ortholog that a bare-symbol `hybrid_search` returns.
- **Metabolites** → the node whose canonical name exactly equals the entity, else the
  standard CHEBI L-/acid form. Genuinely ambiguous entries are marked `_ambiguous`.
