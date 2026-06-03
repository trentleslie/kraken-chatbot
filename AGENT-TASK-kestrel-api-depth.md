# Agent Task — Kestrel API-Depth Reasoning (one/multi/subgraph) + Studio + AstaBench

**Workstream owner:** the main local agent on a **feature branch — NO worktree** (this is the active deck work; keep it on `dev`'s lineage).

**This IS the 6/4 data-science-meeting deck work.** Goal: flesh out the discovery pipeline's **one-hop / multi-hop / subgraph reasoning depth** (the kestrel-api-depth plan), then **demo it in LangGraph Studio** and wire **AstaBench**. This is the gap-identification capability the (separate, deferred) temporal eval later evaluates — so it must be finalized first. The temporal-eval worktree forks off *this* branch (see `AGENT-TASK-temporal-eval-port.md`).

## STEP 1 — Branch + commit context docs (no worktree)

```bash
cd "/home/trentleslie/trentleslie@gmail.com/Google Drive/projects/kraken-chatbot"

# feature branch off dev (no worktree needed — this is the main line of deck work)
git checkout -b feat/kestrel-api-depth

# commit the context docs here (they serve this work AND the temporal eval that forks off this branch);
# open a PR to dev per workflow
git add docs/
git commit -m "docs: kestrel API reference + temporal-eval feasibility learning + discovery-pipeline deep-dive artifacts"
git push -u origin feat/kestrel-api-depth   # then open the PR to dev

# launch the local agent in the main checkout (no worktree)
claude
```

> `docs/references/` has a ~10 MB PDF — `git rm --cached docs/references/*.pdf` + `.gitignore` if you want git lean.

## STEP 2 — Paste as the agent's starting prompt

> You are executing the **Kestrel API-depth (one/multi/subgraph) reasoning** plan on branch `feat/kestrel-api-depth`, then preparing a LangGraph Studio demo + AstaBench wiring for the 6/4 data-science meeting. This is the deck's substance.
>
> **Read first, in order:**
> 1. `docs/plans/2026-05-25-001-feat-kestrel-api-depth-plan.md` — the 8-unit plan + the D1–D6 deepening note (note D4's feasibility correction re: temporal eval — out of scope here).
> 2. `docs/solutions/best-practices/discovery-pipeline-one-graph-methods-within-nodes-2026-05-28.md` — the architecture: ONE graph, methods within nodes; `multi_hop_query` is a real endpoint; Tier-1 vs Tier-2; the `direct_kg` multi-hop gap.
> 3. `docs/kestrel-api-reference.md` — the live API surface (22 endpoints, 6-preset ladder `established→long_shot`, `mode: slim|full|preview`, the 17 constraint fields, edge schema).
>
> **Core build (from the plan):**
> - **R5 — multi-hop in `direct_kg`:** add a `multi_hop_query` call within `direct_kg` for mechanistic chains (Drug→Gene→Pathway→Disease). It currently uses only `one_hop_query` across 2 presets. Rebuild clean from the plan — `feat/issue-27-multi-hop-api` is a confirmed throwaway branch.
> - **R6 — `/subgraph` in integration:** connecting subgraphs between entities feed Synthesis.
> - **R7 — `/canonicalize`** after CURIE resolution in entity_resolution.
> - **R8 — query `/traversal-options` + `/metagraph` once at init and cache** (the API reference already confirms the constraint fields).
> - **Triage `tool_strategies`** (per-entity: ranking preset, use_multi_hop, search_mode) as a plain dict field (written once before the parallel fork — no reducer; see the architecture doc).
>
> **Then for the deck:**
> - **Studio demo:** Studio is already wired (`backend/langgraph.json` registers `kraken_discovery`). Launch `cd backend && langgraph dev` → visualize the one/multi/subgraph reasoning live. Build a clean demo query that exercises one-hop, multi-hop, and subgraph paths so the node-level reasoning is visible on screen.
> - **AstaBench:** integrate the finalized discovery pipeline with AstaBench (deck deliverable).
>
> **Guardrails:** PR workflow (branch → PR, no direct merges). Validation-first — benchmark against the existing eval (PRs #45–51) before claiming improvement. Additive throughout (new endpoints augment, never replace). brainstorming → writing-plans → executing-plans. Keep scope to reasoning depth + Studio + AstaBench; the temporal eval is a separate deferred worktree.

## Relationship to the other launchpad

- **This file** = the deck work (reasoning depth + Studio + AstaBench), feature branch, now.
- `AGENT-TASK-temporal-eval-port.md` = the deferred, isolated temporal-eval worktree that **forks off this branch once it's finalized** (because this reasoning is the gap-identifier it evaluates).
