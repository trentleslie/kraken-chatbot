# Kestrel/Translator gene-drug conflation: biologic genes unreachable by symbol (GH1, CTLA4, GBA1)

> Updated 2026-06-16 after live investigation. This ticket originally described `GH1` as a hybrid-search *recall gap* (the human node "absent from the candidate set"). Direct node inspection shows the cause is more specific and more actionable: the node is not missing from the graph, it has been **conflated** into a multi-type drug node, which is why no gene-symbol query retrieves it.

## Summary

We report an entity-conflation defect in the knowledge graph served by Kestrel `hybrid-search`: the canonical human `GH1` identifier `NCBIGene:2688` exists, but it resolves to an over-merged node whose preferred name is `SOMATROPIN` (recombinant human growth hormone, a drug) and whose synonyms are exclusively pharmaceutical product names. The node carries no `GH1` or `growth hormone 1` gene-symbol text, so a bare-symbol query cannot retrieve it in any search mode (text, vector, or hybrid), even at a request limit of 50. This is not a retrieval coverage gap and not a ranking artifact; it is a normalization decision in the upstream Translator/Babel clique-building layer that collapsed the GH1 gene, its growth-hormone protein product, and the somatropin drug into a single concept and selected the drug label as canonical. No downstream re-ranking can recover a gene identity that the node no longer exposes. Because we operate the Kestrel and kraken stack, this is ours to correct rather than to escalate. We therefore characterized the full extent of the pattern across the human protein-coding genome (see Characterization below) and find it rare and predictable: six genes whose protein product is itself a marketed therapeutic are absorbed into the drug concept and become unreachable by their gene symbol. The defect is real but narrow, and a targeted remediation is tractable.

## Evidence

We inspected the node directly on 2026-06-16 (live, production REST) rather than inferring its state from search results alone. A `get-nodes` lookup of `NCBIGene:2688` returns a node named `SOMATROPIN` that is simultaneously categorized as `biolink:ChemicalEntity`, `biolink:SmallMolecule`, `biolink:Gene`, `biolink:Protein`, and `biolink:Drug`, with 4 073 neighbors and a synonym list composed entirely of drug formulations (`Norditropin`, `Genotropin`, `Saizen`, `Omnitrope`, `somatropin Cartridge`, and similar). The gene symbol `GH1` and the descriptor `growth hormone 1` are absent from both the name and the synonyms. Consequently a `hybrid-search` for `GH1` (limit 50, `biolink:Gene`) returns fifty candidates, none of which is `NCBIGene:2688`; the top hits are the bovine and rat orthologs `NCBIGene:403795` and `NCBIGene:396884`, a cell-line node, a zebrafish ortholog, and human `GH1` transcript nodes (`ENSEMBL:…GH1-205`). The same absence holds for `text-search` and `vector-search`, which confirms the node is unreachable by gene symbol across every retrieval modality rather than filtered by a scoring threshold.

This refines, and partly corrects, the original framing. `GH1` is genuinely distinct from the mis-ranking cases (`TNFRSF1A`, `LDLR`), where the human node is present in the candidate set at approximately rank #4 and is recoverable by an HGNC-preference re-rank. For `GH1`, however, the human gene node as such does not exist in a retrievable form: its identifier points at a drug-conflated concept, and only species orthologs and transcript fragments remain searchable under the symbol. The practical outcome (no `GH1` human gene resolution) is unchanged, but the remedy is not "index the missing node"; it is "do not conflate the gene with its drug product, or retain the gene symbol on the merged concept."

## Mechanism (Translator/Babel conflation)

The behavior is consistent with the documented conflation mechanism in the NCATS Translator ecosystem, on which Kestrel's identifier model is built. Babel constructs cliques of equivalent identifiers across biomedical vocabularies, and the NodeNormalization service applies two controlled conflations over those cliques: `GeneProtein` conflation, which merges a gene with the protein it encodes, and `DrugChemical` conflation, which merges a drug with the chemical of its single active ingredient. Growth hormone is a worst case for these rules acting in sequence: the `GH1` gene conflates with the growth-hormone (somatotropin) protein, and the somatropin drug conflates with that same growth-hormone chemical, so gene, protein, and drug collapse into one concept whose canonical label is the drug. The `GeneProtein` conflation is also why our protein-category queries for these symbols correctly return `NCBIGene` gene nodes, which confirms the conflation layer is active and behaving as designed; `GH1` is the case where the design over-merges across the gene/drug boundary. The Biolink Model team has an open issue (`biolink/biolink-model#633`) on selective conflation, which acknowledges that some consumers require gene and protein (and by extension gene and drug) to remain distinct while others want them merged, and proposes context-dependent control rather than a single universal policy.

The broader literature corroborates that cross-type merging of this kind is a recognized hazard rather than an accepted norm. Surveys of biomedical knowledge-graph construction note that conservative pipelines retain multiple node types and map identifiers onto them rather than collapsing aligned nodes of different semantic types, precisely to avoid losing entity identity in the merge (Know2BIO benchmark, arXiv:2310.03221). The underlying difficulty is named-entity ambiguity: the myriad acronyms and synonyms of the biomedical domain make entity recognition and normalization error-prone, and shared surface forms across a gene, its protein product, and a drug derived from that product (here `GH1`, growth hormone, and somatropin) are exactly the conditions under which an automated clique builder over-merges ("Natural Language Processing for Drug Discovery Knowledge Graphs: promises and pitfalls", arXiv:2310.15572).

## Characterization (genome-wide)

We quantified the prevalence of this defect across the entire human protein-coding genome on 2026-06-16. We took the 19 295 protein-coding genes with NCBI Gene identifiers from the HGNC complete set, retrieved each gene node from Kestrel `get-nodes`, and classified a node as chemically conflated when its `categories` include a drug or chemical type (`biolink:Drug`, `biolink:SmallMolecule`, `biolink:ChemicalEntity`, or a chemical mixture) that a clean gene node would not carry. We then tested whether each gene is reachable by its own symbol, defining reachability as the appearance of the gene's own `NCBIGene` identifier among the `hybrid-search` candidates for that symbol at a request limit of 100.

The pattern is rare and concentrated. Of the 19 295 genes, 2 794 (14.5%) carry some chemical category, but in nearly all cases the tag is a benign cross-reference: the node retains the gene symbol as its name and remains reachable. The damaging cases are those in which the conflation also overwrote the canonical name with a drug label and removed the symbol from the searchable surface. Six genes meet both conditions, and in every case the gene encodes a protein that is itself a marketed biologic or peptide therapeutic, which is the precise condition under which `GeneProtein` and `DrugChemical` conflation chain together.

| Gene | Identifier | Conflated node name (canonical) | Therapeutic product |
|------|-----------|----------------------------------|---------------------|
| GH1 | NCBIGene:2688 | SOMATROPIN | recombinant human growth hormone |
| CALCA | NCBIGene:796 | calcitonin | salmon/human calcitonin |
| POMC | NCBIGene:5443 | corticotropin | ACTH (corticotropin) |
| CRH | NCBIGene:1392 | corticotropin-releasing hormone (ovine) | corticorelin |
| CTLA4 | NCBIGene:1493 | BELATACEPT | belatacept (CTLA4-Ig) |
| GBA1 | NCBIGene:2629 | IMIGLUCERASE | imiglucerase (enzyme replacement) |

Each of the six is unreachable by its gene symbol: a `hybrid-search` for the symbol returns only species orthologs and transcript fragments, never the gene's own identifier, exactly as for `GH1`. `CTLA4` and `GBA1` are notable because they are widely studied genes (an immune checkpoint and the Gaucher-disease locus, respectively), which demonstrates that the defect is not confined to obscure loci; it tracks the druggability of the protein product rather than the prominence of the gene. The remediation footprint is therefore six identifiers, with one adjacent case (`CYP2A6`, conflated with the phenotype "coumarin resistance") representing a separate gene/disease over-merge that the same fix strategy would address. The sweep also surfaced a distinct and larger class of stale gene-symbol mismatches, in which Kestrel retains a superseded symbol (for example `NAT8L` where HGNC now approves `ASPNAT`); these are a nomenclature-currency issue rather than conflation, and we track them separately.

## Impact

This defect propagates to two consumers, with the same observable symptom as the original recall-gap framing. In biomapper2, the `prefer_human` re-ranking (PR #69, now in production) fixes wrong-species resolution for every gene and protein whose human node is present and HGNC-marked in the candidate set, but `GH1`-type symbols remain unresolvable because no HGNC-bearing gene node is returned; the project captures this as a strict-tolerant `xfail` that will pass automatically if the upstream node is corrected. In kraken-chatbot, the discovery pre-resolver's HGNC human-marker gate correctly declines to emit the non-human ortholog, the entity falls back to the existing Kestrel path, the same ortholog is returned, and the gene surfaces as an under-characterized or false cold-start result. The HGNC gate behaves correctly in both systems; it declines a confident wrong-species identifier, but it cannot manufacture a human gene node that the conflation layer has dissolved into a drug concept.

## Reproduction

All queries run against the production Kestrel REST API with a valid `X-API-Key`. Substitute `$KESTREL_API_KEY` (do not paste a key into the wiki). The decisive query is the direct node lookup, which shows the conflation without reference to search ranking.

**1. Inspect the node `NCBIGene:2688` directly — it is the SOMATROPIN drug, not the GH1 gene:**

```bash
curl -s -X POST https://kestrel.nathanpricelab.com/api/get-nodes \
  -H "X-API-Key: $KESTREL_API_KEY" -H "Content-Type: application/json" \
  -d '{"curies":["NCBIGene:2688"],"slim":false,"truncate_long_fields":false}' \
  | python3 -c "import sys,json; n=json.load(sys.stdin)['NCBIGene:2688']; print('name:', n['name']); print('categories:', n['categories']); print('neighbors_count:', n['neighbors_count']); print('synonyms[:6]:', n.get('synonyms',[])[:6])"
```

Observed:

```
name: SOMATROPIN
categories: ['biolink:ChemicalEntity', 'biolink:SmallMolecule', 'biolink:Gene', 'biolink:Protein', 'biolink:Drug']
neighbors_count: 4073
synonyms[:6]: ['somatropin Cartridge', 'GHN', 'somatropin Prefilled Syringe [Genotropin]', 'somatropin 8 MG/ML [Genotropin]', 'somatropin 7.2 MG/ML', 'Norditropin Injectable Product']
```

**2. Search for the bare symbol `GH1` — the human gene node is never returned (orthologs and transcripts only):**

```bash
curl -s -X POST https://kestrel.nathanpricelab.com/api/hybrid-search \
  -H "X-API-Key: $KESTREL_API_KEY" -H "Content-Type: application/json" \
  -d '{"search_text":["GH1"],"limit":50,"category_filter":"biolink:Gene"}' \
  | python3 -c "import sys,json; rows=json.load(sys.stdin)['GH1']; print('rows:', len(rows), '| NCBIGene:2688 present:', any(r['id']=='NCBIGene:2688' for r in rows)); [print(' ', r['id'], round(r['score'],3), repr(r.get('name')), 'HGNC=' + str('HGNC' in (r.get('prefixes') or []))) for r in rows[:5]]"
```

Observed:

```
rows: 50 | NCBIGene:2688 present: False
  NCBIGene:403795 4.837 'GH1' HGNC=False      # bovine ortholog
  NCBIGene:396884 4.831 'GH1' HGNC=False      # rat ortholog
  CLO:0003486 3.309 'GH1' HGNC=False          # cell line
  NCBIGene:407639 3.292 'gh1' HGNC=False       # zebrafish ortholog
  ENSEMBL:ENST00000579711 1.36 'GH1-205' HGNC=False  # human transcript, not the gene node
```

**3. Confirm the absence is not modality-specific (substitute `text-search` / `vector-search` for the endpoint):**

```bash
for EP in text-search vector-search hybrid-search; do
  echo -n "$EP: "
  curl -s -X POST "https://kestrel.nathanpricelab.com/api/$EP" \
    -H "X-API-Key: $KESTREL_API_KEY" -H "Content-Type: application/json" \
    -d '{"search_text":["GH1"],"limit":50,"category_filter":"biolink:Gene"}' \
    | python3 -c "import sys,json; rows=json.load(sys.stdin)['GH1']; print('NCBIGene:2688 present:', any(r['id']=='NCBIGene:2688' for r in rows))"
done
```

Observed: `NCBIGene:2688 present: False` for all three modalities.

**4. Downstream confirmation via the biomapper2 dev API** (the gene resolves to an ortholog under the default `prefer_human=true`, because no human gene node is retrievable; contrast with `TNFRSF1A`, which resolves correctly):

```bash
curl -s -X POST https://dev-biomapper.expertintheloop.io/api/v1/map/entity \
  -H "X-API-Key: $BIOMAPPER2_API_KEY" -H "Content-Type: application/json" \
  -d '{"name":"GH1","entity_type":"gene","identifiers":{},"options":{"prefer_human":true}}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['result']['chosen_kg_id'])"
# -> NCBIGene:403795  (bovine ortholog; graceful fallback, no human gene node available)
```

## Remediation

We own the Kestrel and kraken stack, so this is an internal correction rather than an external request, and the characterization above bounds the work to a small, enumerated set. We propose remediation at two levels.

At the data level, the durable fix is to retain the gene symbol on the conflated clique: each of the six identifiers should carry its HGNC gene symbol (`GH1`, `CALCA`, `POMC`, `CRH`, `CTLA4`, `GBA1`) and gene name among the node's searchable synonyms, even if the canonical preferred name remains the drug, so that a symbol query can reach the node. Where retaining a single merged concept is undesirable, the preferable correction is to apply selective conflation and keep the gene node distinct from the drug node, which is the strategy under discussion in `biolink/biolink-model#633`. Because the conflation originates in the Babel and NodeNormalization clique-building layer that we ingest, the fix is either a configuration change to that ingestion or a post-ingestion repair pass that re-attaches gene symbols to drug-conflated gene identifiers; we recommend the latter as an immediate measure because it is fully under our control and scoped to six known identifiers.

At the consumer level, both biomapper2 and kraken-chatbot can mitigate immediately with a small static override that maps each affected gene symbol to its canonical human `NCBIGene` identifier, applied before the Kestrel search so that the six known cases bypass the conflated node entirely. This is a stopgap that removes the user-visible false cold-start for the enumerated genes while the data-level fix is implemented; it is not a substitute for the data fix, since the override does not generalize to genes we have not yet enumerated. We will re-run the genome-wide sweep after any data-level change to confirm the count returns to zero and to detect regressions, and we will extend the sweep to the gene/disease over-merge exemplified by `CYP2A6` and to the stale-symbol class surfaced alongside it.

## References

| Item | Location |
|------|----------|
| biomapper2 HGNC human-preference fix | Phenome-Health/biomapper2 PR #69 (dev), #70 (dev → main) |
| biomapper2 `GH1` xfail + conflation note | `tests/test_human_gene_gold_set.py` (biomapper2) |
| biomapper2 solution write-up | `docs/solutions/integration-issues/human-gene-symbols-resolve-to-wrong-species-orthologs-2026-06-15.md` |
| kraken-chatbot pre-resolver plan | `docs/plans/2026-06-11-001-feat-biomapper-entity-resolution-plan.md` |
| Genome-wide conflation sweep (data, 2026-06-16) | `docs/tickets/gh1-conflation-genome-sweep.json` (kraken-chatbot) |
| Gene list source (HGNC complete set, protein-coding) | https://www.genenames.org/download/archive/ |
| Babel (clique construction) | https://github.com/NCATSTranslator/Babel |
| NodeNormalization (conflation: GeneProtein, DrugChemical) | https://github.com/TranslatorSRI/NodeNormalization |
| Selective conflation discussion | https://github.com/biolink/biolink-model/issues/633 |
| Know2BIO benchmark (multi-type nodes; conservative merging) | https://arxiv.org/abs/2310.03221 |
| NLP for Drug Discovery KGs: promises and pitfalls (NER/synonym ambiguity) | https://arxiv.org/abs/2310.15572 |
