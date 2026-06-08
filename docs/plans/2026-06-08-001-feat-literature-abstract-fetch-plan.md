# Spike: Abstract-body fetch + cache in `literature_grounding`

**Date:** 2026-06-08
**Branch:** `feat/literature-abstract-fetch` (off `dev`)
**Type:** Spike (narrow, single-gap)
**Status:** in progress

## Problem

Every KM-GPT-DCH-style mechanism we plan to port — the A-B-C mediated-chain grounding
scorer and the Beta-Binomial resampling credible interval (see
`docs/references/km-gpt-dch-competing-hypotheses.md`, code-level mining section) — reads
**abstract text**. The `literature_grounding` node currently attaches only paper
**metadata** to each hypothesis (`LiteratureSupport` carries a ≤300-char `key_passage`
snippet, never the full body). That data gap, not the scoring math, is the real porting
blocker, and it is step 0 of the planned synthesis-literature swap.

This spike closes **only** that gap: make abstract bodies fetchable and attached
("cached" in-state) for the papers a hypothesis already surfaces.

## Current state (verified)

- Sources in `graph/nodes/literature_grounding.py`: KG PMIDs, OpenAlex, Exa, PubMed, S2.
  None persist an abstract body.
- `pubmed_client.py` uses ESearch → **ESummary** → metadata only (no abstract).
- `semantic_scholar.py:40-62` already requests the `abstract` field and uses it for
  relevance scoring + `key_passage`, then **discards** it.
- `LiteratureSupport` (`graph/state.py:221-240`, `frozen=True`) has no abstract field.
- **No persistent literature cache exists.** DB models (`models.py`) are
  conversation/turn/tool-call observability only. The de-facto cache is the in-memory
  `hypothesis.literature_support` list on LangGraph state.

## Minimal change

1. **`LiteratureSupport.abstract: str | None = None`** — new optional frozen field.
   Backward-compatible: existing constructors omit it (default `None`); the references
   table and frontend ignore it (additive serialization).

2. **S2 (free win):** in `create_literature_from_s2`, set `abstract=paper.get("abstract")`.
   No new API calls — the body is already fetched and currently dropped.

3. **PubMed EFetch path:** add `fetch_abstracts(pmids) -> dict[str, str]` to
   `pubmed_client.py` using `efetch.fcgi` (`retmode=xml`, `rettype=abstract`), parsed with
   stdlib `xml.etree.ElementTree` (no Biopython dep). Reuse the existing
   `PUBMED_SEMAPHORE`, `PUBMED_DELAY`, and `NCBI_API_KEY`. Batch PMIDs (≤200/request).
   Concatenate multiple `<AbstractText>` segments (structured abstracts) with their
   `Label`s.

4. **Backfill step in the node:** after grounding, collect every grounded
   `LiteratureSupport` that has a resolvable PMID but no `abstract`, dedupe PMIDs across
   all hypotheses, EFetch once, and rewrite those entries with the body filled in.

## Idempotency & rate limits

- Skip any paper that already has `abstract` (S2 papers are never re-fetched).
- Dedupe PMIDs across the whole run → each PMID hits EFetch at most once per run.
- Batch requests; reuse the repo's existing semaphore + inter-request delay + API key.
- Network failure is non-fatal: on EFetch error the paper keeps `abstract=None` and the
  pipeline proceeds (grounding is best-effort today and stays best-effort).

## Out of scope (non-goals)

- No grounding-confidence scorer, A-B-C chain scorer, or credible interval.
- No persistent (disk/DB) cross-run cache, no Alembic migration. In-state + in-run dedup
  only. (Cross-run cache noted as a follow-up below.)
- No change to the temporal-eval workstream (`AGENT-TASK-temporal-eval-port.md` stays
  blocked on one/multi/subgraph reasoning).
- No change to the reasoning model or harness.

## Verification

- Unit: XML parsing (single + structured multi-segment abstract; empty/no-abstract PMID),
  S2 abstract passthrough, idempotent skip when `abstract` already set, cross-hypothesis
  PMID dedup.
- End-to-end (one real hypothesis): run the backfill against live NCBI EFetch for a known
  PMID set and show a cached abstract body on a `LiteratureSupport`.

## What this unblocks

With abstract bodies attached to each hypothesis's papers, the **A-B-C mediated-chain
grounding scorer** can be built next: it reads `LiteratureSupport.abstract` for the
papers on a bridge hypothesis, splits them into AB/BC pools, and runs the
composition-scoring prompt mined from skimgpt — no further data plumbing required. The
Beta-Binomial resampling CI likewise becomes feasible because it can now sample over real
abstract sets. A cross-run persistent abstract cache (keyed by PMID) is the natural
follow-up optimization once the scorer's fetch volume is known.
