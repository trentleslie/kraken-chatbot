# Kestrel/Translator gene-drug conflation: biologic genes unreachable by symbol (GH1, CTLA4, GBA1)

## Summary

Kestrel inherits an entity-conflation behavior from the Translator/Babel normalization layer that, for genes whose protein product is itself a marketed therapeutic, collapses the gene, its protein, and the drug into a single node and labels that node with the drug. The node retains the gene symbol and its HGNC identifier, but only as low-salience synonyms beneath a drug canonical name, so no search mode ranks it for the gene's own symbol and the node is never returned as a candidate. The exemplar is `GH1`: its canonical identifier `NCBIGene:2688` exists and even carries `GH1` among its 255 synonyms and `HGNC:4261` among its 535 equivalent identifiers, yet because the node is named `SOMATROPIN` (recombinant human growth hormone) and is semantically a drug, a `GH1` query never retrieves it (not even at limit 200). The proximate failure is therefore in retrieval, and its root cause is the normalization decision to collapse gene, protein, and drug into one drug-labeled node; downstream re-ranking cannot help, because the node carrying the human identity is never a candidate to re-rank. We characterized the full extent across the human protein-coding genome and find it rare and predictable: of 19,295 protein-coding genes, six are affected (`GH1`, `CALCA`, `POMC`, `CRH`, `CTLA4`, `GBA1`), each a gene whose product is a biologic or peptide drug. The defect is real but narrow, and a targeted remediation is tractable.

## Evidence

We established the mechanism by inspecting the exemplar node directly on 2026-06-16 (live, production REST) rather than inferring its state from search results. A `get-nodes` lookup of `NCBIGene:2688` returns a node named `SOMATROPIN`, simultaneously categorized as `biolink:Gene`, `biolink:Protein`, `biolink:Drug`, `biolink:SmallMolecule`, and `biolink:ChemicalEntity`, with 4,073 neighbors, 255 synonyms, and 535 equivalent identifiers. The node does carry its gene identity: `GH1` is present among the synonyms (alongside drug formulations such as `Norditropin`, `Genotropin`, and `Saizen`), and `HGNC:4261`, the human-only gene marker, is present among the equivalent identifiers. What the node lacks is the gene symbol as its canonical name and a gene-dominated semantic profile. Consequently a `hybrid-search` for `GH1` (limit 50, `biolink:Gene`) returns fifty candidates, none of which is `NCBIGene:2688`, and raising the limit to 200 does not surface it: the top hits are the bovine and rat orthologs `NCBIGene:403795` and `NCBIGene:396884`, a cell line, a zebrafish ortholog, and human `GH1` transcript fragments (`ENSEMBL:…GH1-205`). The same absence holds under `text-search` and `vector-search`. Because the node demonstrably carries both the symbol and the HGNC marker, this is not a data gap; it is a ranking failure in which a drug-named, drug-embedded node loses to the species orthologs whose primary name is the queried symbol, so the human node never becomes a candidate at all.

The signature is therefore structural and reproducible: a gene whose own `NCBIGene` node is categorized as a drug or chemical and whose canonical name is the drug, with the gene symbol retained only as a low-salience synonym rather than the node's primary name, so retrieval never surfaces it for the symbol.

## Mechanism (Translator/Babel conflation)

The behavior is consistent with the documented conflation mechanism in the NCATS Translator ecosystem, on which Kestrel's identifier model is built. Babel constructs cliques of equivalent identifiers across biomedical vocabularies, and the NodeNormalization service applies two controlled conflations over those cliques: `GeneProtein` conflation, which merges a gene with the protein it encodes, and `DrugChemical` conflation, which merges a drug with the chemical of its single active ingredient. Growth hormone is a worst case for these rules acting in sequence: the `GH1` gene conflates with the growth-hormone (somatotropin) protein, and the somatropin drug conflates with that same growth-hormone chemical, so gene, protein, and drug collapse into one concept whose canonical label is the drug. The `GeneProtein` conflation is also why our protein-category queries for these symbols correctly return `NCBIGene` gene nodes, which confirms the conflation layer is active and behaving as designed; `GH1` is the case where the design over-merges across the gene/drug boundary. The Biolink Model team has an open issue (`biolink/biolink-model#633`) on selective conflation, which acknowledges that some consumers require gene and protein (and by extension gene and drug) to remain distinct while others want them merged, and proposes context-dependent control rather than a single universal policy. The retrieval failure follows directly from the over-merge: `hybrid-search` scores a node on its canonical name and overall semantic embedding, both of which are dominated by the drug, so the node does not rank for a gene symbol that survives only as one synonym among hundreds, even though that symbol and the HGNC marker remain attached to it. The species orthologs, whose primary name is exactly the queried symbol, therefore outrank the human node and fill the candidate list.

The broader literature corroborates that cross-type merging of this kind is a recognized hazard rather than an accepted norm. Surveys of biomedical knowledge-graph construction note that conservative pipelines retain multiple node types and map identifiers onto them rather than collapsing aligned nodes of different semantic types, precisely to avoid losing entity identity in the merge (Know2BIO benchmark, arXiv:2310.03221). The underlying difficulty is named-entity ambiguity: the myriad acronyms and synonyms of the biomedical domain make entity recognition and normalization error-prone, and shared surface forms across a gene, its protein product, and a drug derived from that product (here `GH1`, growth hormone, and somatropin) are exactly the conditions under which an automated clique builder over-merges ("Natural Language Processing for Drug Discovery Knowledge Graphs: promises and pitfalls", arXiv:2310.15572).

## Characterization (genome-wide)

Applying this structural signature genome-wide, we took the 19,295 protein-coding genes with NCBI Gene identifiers from the HGNC complete set (2026-06-16), retrieved each gene node from Kestrel `get-nodes`, and classified a node as chemically conflated when its `categories` include a drug or chemical type (`biolink:Drug`, `biolink:SmallMolecule`, `biolink:ChemicalEntity`, or a chemical mixture) that a clean gene node would not carry. We then tested whether each gene is reachable by its own symbol, defining reachability as the appearance of the gene's own `NCBIGene` identifier among the `hybrid-search` candidates for that symbol at a request limit of 100.

The pattern is rare and concentrated. Of the 19,295 genes, 2,794 (14.5%) carry some chemical category, but in nearly all cases the tag is a benign cross-reference: the node retains the gene symbol as its name and remains reachable. The damaging cases are those in which the conflation also overwrote the canonical name with a drug label, demoting the gene symbol to a buried synonym that retrieval no longer surfaces. Six genes meet both conditions, and in every case the gene encodes a protein that is itself a marketed biologic or peptide therapeutic, which is the precise condition under which `GeneProtein` and `DrugChemical` conflation chain together.

| Gene | Identifier | Conflated node name (canonical) | Therapeutic product |
|------|------------|---------------------------------|---------------------|
| GH1  | NCBIGene:2688 | SOMATROPIN                      | recombinant human growth hormone |
| CALCA | NCBIGene:796 | calcitonin                      | salmon/human calcitonin |
| POMC | NCBIGene:5443 | corticotropin                   | ACTH (corticotropin) |
| CRH  | NCBIGene:1392 | corticotropin-releasing hormone (ovine) | corticorelin        |
| CTLA4 | NCBIGene:1493 | BELATACEPT                      | belatacept (CTLA4-Ig) |
| GBA1 | NCBIGene:2629 | IMIGLUCERASE                    | imiglucerase (enzyme replacement) |

Each of the six is unreachable by its gene symbol: a `hybrid-search` for the symbol returns only species orthologs and transcript fragments, never the gene's own identifier, exactly as for `GH1`. We verified by direct node inspection that all six behave identically to `GH1` in the decisive respect: every node carries its own gene symbol as a synonym and an HGNC equivalent identifier (`HGNC:4261`, `1437`, `9201`, `2355`, `2505`, and `4177` for `GH1`, `CALCA`, `POMC`, `CRH`, `CTLA4`, and `GBA1` respectively), yet none appears among the top 200 `hybrid-search` candidates for its symbol. In every case the failure is therefore retrieval, not missing data. `CTLA4` and `GBA1` are notable because they are widely studied genes (an immune checkpoint and the Gaucher-disease locus, respectively), which demonstrates that the defect is not confined to obscure loci; it tracks the druggability of the protein product rather than the prominence of the gene. The remediation footprint is therefore six identifiers, with one adjacent case (`CYP2A6`, conflated with the phenotype "coumarin resistance") representing a separate gene/disease over-merge that the same fix strategy would address. The sweep also surfaced a distinct and larger class of stale gene-symbol mismatches, in which Kestrel retains a superseded symbol (for example `NAT8L` where HGNC now approves `ASPNAT`); these are a nomenclature-currency issue rather than conflation, and we track them separately.

## Impact

This defect propagates to two consumers, with the same observable symptom as the original recall-gap framing. In biomapper2, the `prefer_human` re-ranking (PR #69, now in dev deployment) fixes wrong-species resolution for every gene and protein whose human node is present and HGNC-marked in the candidate set, but `GH1`-type symbols remain unresolvable because no HGNC-bearing gene node is returned; the project captures this as a strict-tolerant `xfail` that will pass automatically if the upstream node is corrected. In kraken-chatbot, the discovery pre-resolver's HGNC human-marker gate correctly declines to emit the non-human ortholog, the entity falls back to the existing Kestrel path, the same ortholog is returned, and the gene surfaces as an under-characterized or false cold-start result. The HGNC gate behaves correctly in both systems; it declines a confident wrong-species identifier, but it cannot select a human gene node that retrieval never surfaces because the conflation buried it inside a drug-labeled concept.

## Reproduction

All queries run against the production Kestrel REST API with a valid `X-API-Key`. Substitute `$KESTREL_API_KEY` (do not paste a key into the wiki). The decisive query is the direct node lookup, which shows the conflation without reference to search ranking.

**1. Inspect the node** `**NCBIGene:2688**` **directly — its canonical identity is the SOMATROPIN drug, yet it still carries the** `**GH1**` **symbol and** `**HGNC:4261**`**:**

```bash
curl -s -X POST https://kestrel.nathanpricelab.com/api/get-nodes \
  -H "X-API-Key: $KESTREL_API_KEY" -H "Content-Type: application/json" \
  -d '{"curies":["NCBIGene:2688"],"slim":false,"truncate_long_fields":false}' \
  | python3 -c "import sys,json; n=json.load(sys.stdin)['NCBIGene:2688']; s=n.get('synonyms',[]); e=n.get('equivalent_ids',[]); print('name:', n['name']); print('categories:', n['categories']); print('neighbors:', n['neighbors_count']); print('synonyms:', len(s), \"| 'GH1' present:\", 'GH1' in s); print('equivalent_ids:', len(e), '| HGNC:', [x for x in e if x.startswith('HGNC:')])"
```

Observed — the gene symbol and the human marker are present on the node; only the canonical name and semantic type are the drug:

```
name: SOMATROPIN
categories: ['biolink:ChemicalEntity', 'biolink:SmallMolecule', 'biolink:Gene', 'biolink:Protein', 'biolink:Drug']
neighbors: 4073
synonyms: 255 | 'GH1' present: True
equivalent_ids: 535 | HGNC: ['HGNC:4261']
```

**2. Search for the bare symbol** `**GH1**` **— the human gene node is never returned (orthologs and transcripts only):**

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

**3. Confirm the absence is not modality-specific (substitute** `**text-search**` **/** `**vector-search**` **for the endpoint):**

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

**4. Reproduce the genome-wide Characterization — classify every protein-coding gene and recover the six GH1-type cases.** The sweep needs only the public HGNC gene list plus the Kestrel `get-nodes` and `hybrid-search` endpoints:

```python
# pip install requests; download the gene list first:
#   curl -sO https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt
import csv, requests

KEY  = "$KESTREL_API_KEY"                       # do not paste a real key into the wiki
BASE = "https://kestrel.nathanpricelab.com/api"
H    = {"X-API-Key": KEY, "Content-Type": "application/json"}
CHEM = {"biolink:Drug", "biolink:SmallMolecule", "biolink:ChemicalEntity",
        "biolink:MolecularMixture", "biolink:ChemicalMixture"}

# 1) protein-coding genes with NCBI Gene IDs from the HGNC complete set (19,295)
rows  = list(csv.DictReader(open("hgnc_complete_set.txt"), delimiter="\t"))
genes = [(r["symbol"], "NCBIGene:" + r["entrez_id"]) for r in rows
         if r["locus_group"] == "protein-coding gene" and r["entrez_id"]]

# 2) get-nodes every gene -> (canonical name, categories), batched
meta = {}
for i in range(0, len(genes), 1000):
    curies = [c for _, c in genes[i:i + 1000]]
    r = requests.post(f"{BASE}/get-nodes", headers=H,
                      json={"curies": curies, "slim": True, "truncate_long_fields": True}).json()
    for c in curies:
        n = r.get(c)
        if n:
            meta[c] = (n.get("name", ""), set(n.get("categories") or []))

# 3) chemically conflated = node carries a drug/chemical category a clean gene would not
conflated = [(s, c) for s, c in genes if c in meta and (meta[c][1] & CHEM)]

# 4) GH1-type = drug-named (name != symbol) AND carries biolink:Drug AND unreachable by symbol
def reachable(sym, curie):
    r = requests.post(f"{BASE}/hybrid-search", headers=H,
                      json={"search_text": [sym], "limit": 100, "category_filter": "biolink:Gene"}).json()
    return any(x.get("id") == curie for x in r.get(sym, []))

hits = [(s, c, meta[c][0]) for s, c in conflated
        if meta[c][0].lower() != s.lower() and "biolink:Drug" in meta[c][1] and not reachable(s, c)]

print(f"protein-coding genes:               {len(genes)}")
print(f"carry a drug/chemical category:     {len(conflated)}")
print(f"GH1-type (drug-named + unreachable): {len(hits)}")
for s, c, name in sorted(hits):
    print(f"  {s:<7} {c:<15} {name}")
```

Observed:

```
protein-coding genes:               19295
carry a drug/chemical category:     2794
GH1-type (drug-named + unreachable): 6
  CALCA   NCBIGene:796    calcitonin
  CRH     NCBIGene:1392   corticotropin-releasing hormone (ovine)
  CTLA4   NCBIGene:1493   BELATACEPT
  GBA1    NCBIGene:2629   IMIGLUCERASE
  GH1     NCBIGene:2688   SOMATROPIN
  POMC    NCBIGene:5443   corticotropin
```

**5. Downstream confirmation via the biomapper2 dev API** (the gene resolves to an ortholog under the default `prefer_human=true`, because no human gene node is retrievable; contrast with `TNFRSF1A`, which resolves correctly):

```bash
curl -s -X POST https://dev-biomapper.expertintheloop.io/api/v1/map/entity \
  -H "X-API-Key: $BIOMAPPER2_API_KEY" -H "Content-Type: application/json" \
  -d '{"name":"GH1","entity_type":"gene","identifiers":{},"options":{"prefer_human":true}}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['result']['chosen_kg_id'])"
# -> NCBIGene:403795  (bovine ortholog; graceful fallback, no human gene node available)
```

## Remediation

All six affected nodes already carry the exact gene symbol as a synonym and an HGNC equivalent identifier, so no data needs to be added; the work is to make that latent identity reachable. We propose three measures, ordered by how soon each can land and how broadly each generalizes.

**Immediate, consumer-side.** The genome-wide sweep already emits, from Kestrel's own data, the exact set of affected symbols paired with their canonical HGNC-bearing `NCBIGene` identifiers. biomapper2 and kraken-chatbot can consume that output directly as a symbol-to-identifier override, applied ahead of the Kestrel search so that an affected symbol resolves to its correct human identifier without depending on ranking. Because the map is generated and refreshed by re-running the sweep rather than hand-maintained, it stays correct as the knowledge graph changes; it removes the user-visible false cold-start for the enumerated genes today, but it is a stopgap because it covers only genes the sweep has already enumerated.

**Durable, retrieval-side, under our control.** The general fix is a deterministic exact-symbol retrieval path for gene-category queries: a node whose synonyms contain the exact queried symbol and that carries an HGNC equivalent identifier should be surfaced as a candidate regardless of its semantic score, rather than being outranked by species orthologs whose only advantage is that the symbol is their primary name. This requires no new data, since the symbol and the HGNC marker are already attached to the node; it requires only that retrieval weight an exact gene-symbol match on a human-marked node appropriately. Once the node is a candidate, biomapper2's existing HGNC re-ranking selects it over the orthologs, so the change composes with the shipped fix and generalizes to every such gene, including ones we have not yet enumerated.

**Cleanest, upstream.** The most durable correction stops the over-merge at its source by applying selective conflation so the gene node remains distinct from the drug node, the strategy under discussion in `biolink/biolink-model#633`. This is the preferable long-term outcome but depends on the Babel and NodeNormalization clique-building layer that we ingest, so we pursue it in parallel rather than as the immediate measure.

We will re-run the genome-wide sweep after any change to confirm the affected count returns to zero and to detect regressions, and we will extend it to the gene/disease over-merge exemplified by `CYP2A6` and to the stale-symbol class surfaced alongside it.

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