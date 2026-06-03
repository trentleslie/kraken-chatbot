# Agent Task — Time-Sliced (Temporal Holdout) Eval

Single working doc for the temporal-eval workstream until it's branched and an agent writes a proper brainstorm/plan. Stays **within the kraken-chatbot project**, in its **own git worktree**.

## Status: DEFERRED — blocked on the one/multi/subgraph reasoning

**Do not set up the worktree yet.** The temporal eval measures **gap identification**, and gap identification *is* the one/multi/subgraph reasoning. Evaluating it before that reasoning is finalized = evaluating the wrong thing. So:

1. **First (now, deck):** finish/finalize the **one/multi/subgraph reasoning** on its **own feature branch — no worktree** (it's the kestrel-api-depth plan, and the 6/4 deck's Studio + AstaBench demo material). e.g. `feat/kestrel-api-depth` off `dev`.
2. **Then (after #1 is sorted):** fork the temporal-eval **worktree off that finalized reasoning branch**, so it inherits the **exact same harness + reasoning model**. See STEP 1.

## Architecture: same pipeline + harness, swapped KG backend via a parallel MCP server

The eval must run the **same discovery pipeline and the same reasoning model** as production — only the **KG backend** changes.

- The pipeline talks to Kestrel over MCP (`KESTREL_MCP_URL` → `kestrel.nathanpricelab.com/mcp`).
- **Approach to explore:** stand up a **parallel MCP server for a biomedical *dated* KG** that implements the **same tool contract** as Kestrel (`one_hop_query`, `multi_hop_query`, `subgraph`, `get_edges`, search…) **plus** per-edge publication dates and a **date-cutoff** parameter. Then point the pipeline at the new MCP URL — the harness/nodes/model run unchanged.
- The MCP server enforces cutoff **T** (only edges + literature dated ≤ T are visible); edges published after T are the held-out test set. Run the pipeline "as of T" → score its gap predictions against the post-T edges.

This is why it stays in kraken-chatbot: it reuses this repo's pipeline verbatim; only the MCP backend is swapped.

## The dated biomedical KG (must be real + dated)

- **SemMedDB** — PubMed semantic predications, **per-PMID dates**; the native Swanson-ABC substrate. Heavier (UMLS license + ETL), but real and dated.
- **Pharmacogenomic KG** (medRxiv 2025.09.24) — already implements chronological splits + publication-date verification; **verify the data is downloadable** (PDF 403'd 2026-05-29).
- *(ogbl-collab / TGB 2.0 — only as an optional unit-test of the scoring **math**; they're non-biomedical/homogeneous so the pipeline's biomedical nodes can't run on them. Not the main path.)*

## Requirements / open questions (for the future brainstorm)

- **Same harness + same reasoning model** as the finalized one/multi/subgraph branch — inherit it, don't fork the logic.
- **MCP parity:** the dated-KG MCP server must mirror Kestrel's tool contract, add publication-date fields, and accept a cutoff.
- **Which dated KG:** SemMedDB vs pharmacogenomic KG (availability check).
- **Cutoff scope:** apply T to edges **and** the literature corpus the `literature_grounding` node retrieves (no future leakage / circular evidence).
- **Scoring:** AUC-PR primary (sparse graph; ROC misleads), + MRR / Hits@k. Open World Assumption — recall against held-out post-T edges is the trustworthy metric; an unpublished prediction is OWA-uncertain, not a false positive. Leakage-safe negative sampling.
- **Reference design:** pharmacogenomic KG paper's leakage-free chronological-split pipeline.

## STEP 1 — worktree setup (RUN ONLY AFTER the one/multi/subgraph branch is finalized)

```bash
cd "/home/trentleslie/trentleslie@gmail.com/Google Drive/projects/kraken-chatbot"

# branch the eval worktree OFF the finalized reasoning branch (inherits the same harness + model)
git worktree add ../kraken-chatbot-temporal-eval -b feat/time-sliced-eval feat/kestrel-api-depth

cd ../kraken-chatbot-temporal-eval && claude
```

> The context docs (`docs/kestrel-api-reference.md`, the kestrel-api-depth plan, the solution doc, `docs/references/`) should be committed on the one/multi/subgraph reasoning branch — the eval worktree inherits them by branching off it, and they reach `dev` when that branch PRs. (`docs/references/` has a ~10 MB PDF — consider `.gitignore`/LFS.)

## STEP 2 — agent starting prompt (inside the worktree)

> You are in an isolated worktree (`kraken-chatbot-temporal-eval`, branch `feat/time-sliced-eval`), forked from the finalized one/multi/subgraph reasoning branch. Build a time-sliced (temporal-holdout) eval that runs **this repo's existing discovery pipeline, unchanged**, against a **biomedical dated KG served over a parallel MCP server** (same tool contract as Kestrel + publication dates + a cutoff). This is independent of the 6/4 deck — don't touch Studio/AstaBench/deck work.
>
> Read first: `docs/kestrel-api-reference.md` (current MCP backend + its no-date limitation) and `docs/solutions/best-practices/discovery-pipeline-one-graph-methods-within-nodes-2026-05-28.md` (the pipeline you're pointing at a new backend). Then use the brainstorming skill to resolve the open questions above (esp. which dated KG, and the MCP-server tool-parity surface) before planning.
>
> Build order: (1) stand up / spec the dated-KG MCP server mirroring Kestrel's tool contract with date fields + cutoff; (2) point the pipeline at it via the MCP URL; (3) run at cutoff T and score gap predictions vs post-T edges (AUC-PR/MRR/Hits@k, OWA recall). Optional first: unit-test the scoring math on ogbl-collab.
>
> Guardrails: PR workflow; validation-first; brainstorming → writing-plans → executing-plans; same harness + model as the inherited reasoning branch — don't re-implement the pipeline.

## Meanwhile: the one/multi/subgraph branch (prerequisite, NO worktree)

Just a feature branch off `dev` (e.g. `feat/kestrel-api-depth`) executing the 8-unit `docs/plans/2026-05-25-001-feat-kestrel-api-depth-plan.md`. This is the deck's reasoning-depth + Studio + AstaBench substance. Commit the context docs here. No worktree needed.
