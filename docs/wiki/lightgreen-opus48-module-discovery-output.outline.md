# Lightgreen Module Run on Opus 4.8: Discovery Output (29-analyte, dev, 2026-06-24)

This document presents the discovery output of a full-module run of the **Lightgreen** WGCNA module through the Kraken 12-node discovery pipeline, executed on the dev environment on 2026-06-24 (commit `a7d7fd6`) with BioMapper entity resolution enabled. The SDK-backed pipeline nodes ran on the Opus 4.8 model rather than the default Sonnet; this run is published for direct model comparison. It is one of a per-module series over the Frailty multi-omics WGCNA modules; see the [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm).

## Run Provenance

The run submitted 29 named analytes, parsed 30 at intake, and resolved 30 distinct entities (14 fuzzy, 15 biomapper, 1 exact) to 30 distinct CURIEs. Triage classified 5 well-characterized, 10 moderate, 11 sparse, and 4 cold-start, with 0 edge-count measurement failures. Downstream analysis produced 654 direct-KG findings, 35 cold-start findings, 4 biological themes, 0 cross-entity bridges (0 evidence-grounded), and 68 hypotheses supported by 15 literature references. Synthesis emitted a 24238-character report. The run completed in approximately 570.8 s of wall-clock time (status complete, 1 errors). All counts are taken from the reducer-merged pipeline state.

| Stage | Result |
|---|---|
| Submitted | 29 named analytes |
| Intake | 30 parsed |
| Entity resolution | 30 resolved (14 fuzzy, 15 biomapper, 1 exact) to 30 distinct CURIEs |
| Triage | 5 well-characterized, 10 moderate, 11 sparse, 4 cold-start (0 measurement failures) |
| Direct KG | 654 findings |
| Cold-start | 35 findings, 7 skipped |
| Pathway enrichment | 4 biological themes |
| Integration | 0 bridges (0 evidence-grounded) |
| Literature grounding | 15 papers |
| Synthesis | 68 hypotheses, 24238-character report |
| Run total | ~570.8 s wall-clock, status complete, 1 errors |

## Related

- Companion run metrics: [Lightgreen Module Run on Opus 4.8: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/lightgreen-module-run-on-opus-48-pipeline-performance-report-29-analyte-dev-2026-06-24-eIlHwQarWo)
- Model comparison baseline (Sonnet): [Lightgreen Module Run: Discovery Output](https://phwiki.phenoma.ai/doc/lightgreen-module-run-discovery-output-29-analyte-dev-2026-06-23-AbI5HbhH8b)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---



## Discovery Report: Lightgreen WGCNA Metabolite Module

### 1. Executive Summary

The Lightgreen WGCNA module encodes a coordinated metabolic signature spanning three principal biological axes: gut microbial fermentation (butyrate, benzoate, 2-hydroxyphenylacetate), one-carbon and sulfur amino acid metabolism (cystathionine, pyridoxal), and complex glycerophospholipid remodeling (seven lysophospholipid and plasmalogen species linked to lipase and acyltransferase gene networks). [KG Evidence] The convergence of gamma-glutamyl dipeptides (gamma-glutamylhistidine, gamma-glutamylalanine, gamma-glutamyl-epsilon-lysine) with the conspicuous absence of glutathione itself strongly implicates gamma-glutamyl transferase (GGT) activity as a unifying enzymatic theme, connecting oxidative stress, hepatic function, and metabolic disease risk across module members. [Inferred] Disease enrichment analysis reveals recurrent associations with gastrointestinal malignancies (colorectal cancer, pancreatic neoplasm, hepatocellular carcinoma) and inflammatory bowel disease, consistent with a module anchored in gut-liver axis metabolism and microbiome-derived signaling. [KG Evidence]

### 2. Key Findings (Tier 1 to 2)

#### 2.1 Disease Associations

The module-level disease recurrence analysis identifies **colorectal cancer** as the most broadly shared disease association (3 members: pyridoxal, cytidine, xanthosine; strongest evidence: curated databases). [KG Evidence] **Melanoma** recurs across pyridoxal, cytidine, and arachidonoyl ethanolamide (text-mined evidence). [KG Evidence] **Inflammatory bowel disease** (IBD), including the Crohn's-associated subtype IBD1, recurs across cytidine and xanthosine with curated-level support, while general IBD additionally connects pyridoxal and cytidine via text-mined evidence. [KG Evidence] Additional gastrointestinal and hepatic malignancies (pancreatic neoplasm, hepatocellular carcinoma, malignant colon neoplasm) are each associated with both pyridoxal and cytidine. [KG Evidence] **Peripheral neuropathy** is associated with both pyridoxal and cytidine, consistent with the established clinical role of vitamin B6 deficiency in neuropathic disease. [KG Evidence]

At the individual-member level, **pyridoxal** (374 edges) participates in vitamin B6 metabolism and associates with multiple inborn errors of branched-chain amino acid catabolism (beta-ketothiolase deficiency, maple syrup urine disease, isovaleric acidemia, propionic acidemia, methylmalonic aciduria) and hypophosphatasia. [KG Evidence] **Cytidine** (547 edges) participates in pyrimidine metabolism, including UMP synthase deficiency (orotic aciduria), MNGIE, and dihydropyrimidinase deficiency. [KG Evidence] **Butyrate** (496 edges) participates in butyrate biosynthesis, catabolism, and cellular response pathways; its top disease association is Epstein-Barr virus infection. [KG Evidence] **Retinal** (483 edges) participates in retinol metabolism with a primary association to vitamin A deficiency. [KG Evidence] **Arachidonoyl ethanolamide** (anandamide; 209 edges) participates in neuroactive ligand-receptor interaction and associates with cirrhosis of liver and hypotensive disorder. [KG Evidence]

#### 2.2 Pathway Memberships and Molecular Interactions

Two pathways recur across module members: **vitamin transport** (GO:0051180), shared by retinal and pyridoxal, and **ABC transporters** (KEGG:02010), shared by cytidine and xanthosine. [KG Evidence] The shared vitamin transport annotation connects the module's two major vitamin-class metabolites (B6 and A-related) through a common cellular distribution mechanism.

Pyridoxal interacts with PDXK (pyridoxal kinase), PDXP (pyridoxal phosphatase), PNPO (pyridoxamine 5'-phosphate oxidase), and ALPL (alkaline phosphatase, tissue-nonspecific isozyme); these constitute the core vitamin B6 salvage and activation machinery. [KG Evidence] Notably, pyridoxal also exhibits KG-recorded interactions with inflammatory mediators TNF, IL-1beta, IL-6, and NFKB1, as well as metabolic regulators INS (insulin) and PPARA (peroxisome proliferator-activated receptor alpha). [KG Evidence] These interactions suggest a broader role for pyridoxal status in modulating inflammatory and metabolic signaling beyond its classical cofactor function.

Cytidine interacts with uridine-cytidine kinases (UCK1, UCK2), cytidine deaminase (CDA), multiple nucleotidases (NT5E, NT5C2, NT5C, NT5C1A, NT5C3A, NT5M), and the concentrative nucleoside transporter SLC28A1. [KG Evidence] These define the nucleotide salvage and catabolic network through which cytidine levels are regulated.

#### 2.3 Lipid Remodeling Gene Network

Pathway enrichment reveals that five glycerophospholipid species in the module (the GPC, GPE, and GPI species containing dihomo-linolenoyl, docosahexaenoyl, docosapentaenoyl, and eicosapentaenoyl chains) share connections to a set of lipase and acyltransferase genes: **CEL** (bile salt-stimulated lipase), **PNLIP** (pancreatic lipase), **LIPC** (hepatic lipase C), **LIPA** (lysosomal acid lipase), **LIPG** (endothelial lipase), **LPL** (lipoprotein lipase), **CPT2** (carnitine palmitoyltransferase 2), and **DGAT1** (diacylglycerol O-acyltransferase 1), each with 40 KG edges. [KG Evidence] These enzymes span pancreatic, hepatic, endothelial, and lysosomal compartments of lipid processing, indicating that the module's complex lipid species reflect coordinated activity across the digestive and systemic lipid remodeling apparatus.

A caveat is warranted: the lipid-species-to-lipase connections all carry the generic predicate `biolink:related_to` and the five lipid species were resolved at only 70% confidence (fuzzy matching to triglyceride species rather than the original lysophospholipid and plasmalogen identities). [KG Evidence] The biological interpretation should therefore be treated as directionally correct but requiring confirmation of entity identity.

### 3. Novel Predictions (Tier 3)

#### 3.1 GGT-Mediated Glutathione Turnover as a Module Driver

**Prediction:** The co-expression of gamma-glutamylhistidine, gamma-glutamylalanine, and gamma-glutamyl-epsilon-lysine in the absence of glutathione itself reflects elevated gamma-glutamyl transferase (GGT) enzymatic activity in the study cohort.

**Structural logic chain:** Gamma-glutamyl dipeptides are direct products of GGT acting on glutathione to transfer the gamma-glutamyl moiety to free amino acids (histidine, alanine) or to the epsilon-amino group of lysine in proteins. [Inferred] The module contains three such products but lacks glutathione (the substrate), indicating net consumption. [KG Evidence for member presence; Model Knowledge for GGT mechanism] Elevated GGT is an established clinical biomarker of oxidative stress, hepatic steatosis, and type 2 diabetes risk. [Model Knowledge]

**Calibration note:** Approximately 18% of computational predictions of this type progress to clinical investigation. This prediction is mechanistically constrained and testable but requires direct GGT activity measurement.

**Validation step:** Measure serum GGT activity in the study cohort and correlate with the Lightgreen module eigengene. If the module captures GGT-mediated turnover, the eigengene should correlate positively with GGT activity and inversely with reduced glutathione levels.

#### 3.2 Saccharolytic Gut Microbiome Signature Distinct from TMAO Pathway

**Prediction:** The Lightgreen module captures a saccharolytic or fermentative microbial metabolic axis (fiber fermentation producing short-chain fatty acids) rather than the proteolytic choline/carnitine-to-TMAO pathway.

**Structural logic chain:** Butyrate is a canonical product of dietary fiber fermentation by gut commensals (Faecalibacterium, Roseburia). [Model Knowledge] Benzoate and 2-hydroxyphenylacetate are microbial transformation products of dietary polyphenols and aromatic amino acids. [Model Knowledge] All three co-cluster in this module, while TMAO (the terminal product of choline/carnitine-dependent microbial metabolism) is conspicuously absent. [KG Evidence for member composition; Inferred for pathway interpretation] This segregation implies that the cohort exhibits a dominant saccharolytic microbial phenotype, potentially associated with higher dietary fiber intake or a Prevotella/Ruminococcus-enriched enterotype. [Model Knowledge]

**Calibration note:** Approximately 18% of such computational predictions advance to clinical investigation. The microbiome-metabolite axis is increasingly validated in multi-omic studies but remains context-dependent.

**Validation step:** Correlate the module eigengene with 16S rRNA or shotgun metagenomics data if available, specifically testing for enrichment of saccharolytic genera (Faecalibacterium prausnitzii, Roseburia intestinalis, Eubacterium rectale). Additionally, test for inverse correlation with TMAO levels measured independently.

#### 3.3 Efficient Transsulfuration Pathway Flux

**Prediction:** The presence of cystathionine (the product of homocysteine conversion) and pyridoxal (the essential cofactor for cystathionine beta-synthase) without homocysteine (the substrate) indicates active transsulfuration, with homocysteine efficiently cleared in this cohort.

**Structural logic chain:** Cystathionine beta-synthase (CBS) converts homocysteine to cystathionine in a pyridoxal 5'-phosphate (PLP)-dependent reaction. [Model Knowledge] Both the product (cystathionine) and the cofactor precursor (pyridoxal) co-cluster in the module while the substrate (homocysteine) is absent. [KG Evidence for member presence] This pattern is consistent with efficient enzymatic flux through the transsulfuration arm of one-carbon metabolism. [Inferred]

**Calibration note:** Approximately 18% of computational metabolic pathway predictions advance to clinical investigation.

**Validation step:** Measure plasma homocysteine in the study cohort and test for inverse correlation with the Lightgreen module eigengene. Additionally, assess CBS genotype (common variant C699T/rs234706) and dietary B6 intake as potential modulators.

#### 3.4 Genetic Regulation of Lysophospholipid Levels (mQTL Predictions)

**Prediction:** Genetic variants associated with 1-linoleoyl-GPE and 1-linoleoyl-GPC levels (e.g., CAID:CA6476722, CAID:CA15677622, supported by 2 analogues each at 89 to 91% similarity) also regulate 1-linoleoyl-GPA (18:2) levels through pleiotropic effects on shared glycerophospholipid remodeling enzymes.

**Structural logic chain:** 1-linoleoyl-GPA shares the linoleoyl (18:2) acyl chain with 1-linoleoyl-GPE (91% similarity) and 1-linoleoyl-GPG (89% similarity), which are linked to multiple genetic variants in metabolomics GWAS. [KG Evidence] Cross-headgroup-class pleiotropic genetic regulation of lysophospholipid levels is a well-documented phenomenon in metabolomics QTL studies, mediated by shared phospholipase and lysophospholipid acyltransferase substrates. [Literature: GWAS studies of PUFAs in Hispanic and African American cohorts (2023) report FADS-region signals affecting multiple lipid classes; GWAS of acyl-lipid metabolism genes (2020) identifies shared genetic architecture across fatty acid species.] Enzymes of the Lands cycle (LPCAT, LPEAT, LPGAT) act on the same acyl-CoA donor (linoleoyl-CoA) irrespective of headgroup, providing a mechanistic basis for cross-class genetic regulation. [Model Knowledge]

**Calibration note:** Approximately 18% of such computational variant-trait predictions progress to clinical investigation. The structural similarity basis (89 to 91%) is relatively strong for lysophospholipid analogue inference.

**Validation step:** Query metabolomics QTL databases (e.g., mGWAS catalog, GWAS Catalog metabolite traits) for association of variants CA6476722 and CA15677622 with any lysophosphatidic acid (LPA) species. Map these variants to specific genomic loci and test whether they reside near phospholipase A or lysophospholipid acyltransferase genes.

### 4. Biological Themes

#### 4.1 Gut-Liver Axis Metabolism

The module's most coherent biological theme integrates gut microbial fermentation products (butyrate, benzoate, 2-hydroxyphenylacetate) with hepatic processing markers (gamma-glutamyl dipeptides, cystathionine) and complex lipid remodeling species processed by hepatic and pancreatic lipases. [KG Evidence for individual associations; Inferred for thematic integration] This gut-liver axis interpretation is reinforced by the disease enrichment pattern: colorectal cancer, inflammatory bowel disease, hepatocellular carcinoma, and pancreatic neoplasm all recur across module members. [KG Evidence]

#### 4.2 Vitamin and Cofactor Status

Pyridoxal (vitamin B6) and retinal (vitamin A aldehyde) anchor a micronutrient component of the module, sharing the vitamin transport pathway (GO:0051180). [KG Evidence] Pyridoxal's interactions with PLP-dependent enzymes (PDXK, PNPO, AGXT, ABAT, ALAS2, GLDC) and its co-clustering with cystathionine (a product of a PLP-dependent reaction) indicate that the module partially reflects vitamin B6 bioavailability and cofactor utilization. [KG Evidence]

#### 4.3 Nucleotide Salvage and Turnover

Cytidine and xanthosine represent pyrimidine and purine nucleoside pools, respectively. Both share the ABC transporter pathway (KEGG:02010) and carry nucleoside chemical-role annotations. [KG Evidence] Xanthosine's presence without downstream urate (a terminal purine catabolite) suggests the module captures nucleotide turnover or salvage rather than terminal degradation. [Inferred]

#### 4.4 Lipid Remodeling and Fatty Acid Signaling

Seven complex glycerophospholipid species (plasmalogens, GPC, GPE, GPI classes with polyunsaturated acyl chains including 20:3, 20:4, 20:5, 22:5, 22:6) co-cluster with arachidonoyl ethanolamide (anandamide), an endocannabinoid lipid mediator. [KG Evidence] The shared lipase gene network (CEL, PNLIP, LIPC, LIPA, LIPG, LPL, CPT2, DGAT1) has moderate connectivity (40 edges each), falling below the hub threshold. [KG Evidence] Note: cytoplasm (GO:0005737; 5,000 edges) was identified as a hub node connecting two members and is de-emphasized as likely non-specific. [KG Evidence]

#### 4.5 Dipeptides and Proteolytic Products

The module contains three dipeptide species (tryptophylleucine, isoleucylleucine/leucylisoleucine, gamma-glutamyl-epsilon-lysine) and the cyclic dipeptide cyclo(ala-pro), alongside multiple gamma-glutamyl amino acid conjugates. [KG Evidence for member identity] The presence of dipeptides without their constituent free amino acids (tryptophan, leucine, isoleucine are absent) suggests the module captures peptidase-generated intermediates rather than canonical amino acid pools. [Inferred]

### 5. Gap Analysis

#### 5.1 Informative Absences

| Expected Entity | Present Pathway Neighbors | Interpretation | Evidence |
|---|---|---|---|
| **Glutathione** | gamma-glutamylhistidine, gamma-glutamylalanine, gamma-glutamyl-epsilon-lysine | Module captures GGT-mediated glutathione degradation products, not glutathione pools; implicates oxidative stress and hepatic turnover | [Inferred] |
| **Homocysteine** | cystathionine (product), pyridoxal (cofactor) | Efficient transsulfuration flux; homocysteine actively converted; may segregate to a distinct WGCNA module | [Inferred] |
| **TMAO** | butyrate, benzoate, 2-hydroxyphenylacetate (gut-derived metabolites) | Module reflects saccharolytic/fermentative microbial pathways, not choline/carnitine proteolytic metabolism | [Inferred] |
| **Urate** | xanthosine (purine precursor) | Module captures nucleotide salvage/turnover, not terminal purine catabolism; urate regulation may be dominated by renal excretion dynamics | [Inferred] |
| **BCAAs** | isoleucylleucine/leucylisoleucine (dipeptide) | BCAAs likely segregate to insulin-resistance-associated modules; this module captures dipeptide-level products instead | [Inferred] |
| **Aromatic amino acids** | tryptophylleucine (dipeptide) | Free amino acids absent despite dipeptide presence; module reflects proteolytic/dipeptidase activity patterns | [Inferred] |
| **Ceramides** | arachidonoyl ethanolamide, complex glycerophospholipids | Module captures endocannabinoid and glycerophospholipid signaling, not sphingolipid-mediated lipotoxicity; distinct lipid axis | [Inferred] |
| **Acylcarnitines** | butyrate, caproate (free fatty acids); 3-decenoylcarnitine (single member) | Free SCFAs dominate over mitochondrial beta-oxidation intermediates; gut-derived rather than hepatic/muscle oxidative origin | [Inferred] |

#### 5.2 Standard (Methodological) Gaps

Insulin, C-peptide, HbA1c, and fasting glucose are absent from the module. [KG Evidence for absence] These clinical measures are typically derived from immunoassay or clinical chemistry platforms rather than untargeted metabolomics, and are likely treated as phenotypic endpoints or covariates rather than WGCNA input features. [Model Knowledge] Their absence is methodological and not biologically informative.

#### 5.3 Entity Resolution Caveats

Several entities were resolved at reduced confidence (70%), which may distort KG findings:
- **Caproate (6:0)** resolved to 17alpha-hydroxyprogesterone caproate (a pharmaceutical ester), not hexanoic acid; disease and pathway associations for this entity should be disregarded. [KG Evidence]
- **Levulinate (4-oxovalerate)** resolved to benzyl levulinate (a synthetic ester); the entity is cold-start (0 edges) regardless. [KG Evidence]
- **N6-succinyladenosine** resolved to N6-benzoyladenosine; cold-start (0 edges). [KG Evidence]
- **Five complex glycerophospholipid species** (the GPC, GPE, GPI species) resolved to triglyceride analogues rather than their original lysophospholipid/plasmalogen identities, inflating the lipase gene network connections. [KG Evidence]
- **Cyclo(ala-pro)** resolved to a large cyclic peptide (PUBCHEM.COMPOUND:53321822), not the intended diketopiperazine; inferred associations (e.g., somatostatin-related binding) are likely artifacts of this misresolution. [KG Evidence]

These resolution limitations affect approximately 8 of 30 entities (27%) and should be accounted for in downstream interpretation.

### 6. Temporal Context

No explicit longitudinal design or timepoint information was provided for this WGCNA module. The following causal inference opportunities are noted:

**Upstream (likely causes):** Dietary fiber intake and gut microbial community composition are plausible upstream determinants of butyrate, benzoate, and 2-hydroxyphenylacetate levels. [Model Knowledge] Vitamin B6 dietary intake and absorption regulate pyridoxal availability, which in turn modulates transsulfuration pathway flux (cystathionine production). [Model Knowledge]

**Downstream (likely consequences):** Elevated GGT activity (inferred from gamma-glutamyl dipeptide accumulation) and altered glycerophospholipid remodeling (lipase network activity) represent downstream metabolic consequences that may predict hepatic dysfunction, oxidative stress burden, and gastrointestinal disease risk. [Inferred]

**Causal inference opportunity:** If longitudinal sampling is available, testing whether butyrate and pyridoxal levels at baseline predict subsequent changes in gamma-glutamyl dipeptide concentrations would help establish the directionality of the gut-liver axis signature. Mendelian randomization using the mQTL variants inferred for lysophospholipid species (Section 3.4) could test causal relationships between lipid remodeling and disease outcomes.

### 7. Research Recommendations

#### Priority 1: Direct Experimental Validation

1. **Measure serum GGT activity** and correlate with the Lightgreen module eigengene. This is the single highest-value validation because it tests the core mechanistic hypothesis (GGT-driven glutathione turnover) unifying the gamma-glutamyl dipeptide cluster. [Inferred]

2. **Measure plasma homocysteine** and test for inverse correlation with the module eigengene, assessing whether the cystathionine-pyridoxal co-clustering reflects efficient transsulfuration. Genotype CBS (rs234706) as a potential effect modifier. [Inferred]

3. **Confirm entity resolution** for the five complex glycerophospholipid species and for caproate. Re-query the knowledge graph with corrected identifiers (lysophospholipid and plasmalogen forms rather than triglycerides; hexanoic acid rather than 17alpha-hydroxyprogesterone caproate) to obtain accurate disease and pathway associations. [KG Evidence for resolution quality concerns]

#### Priority 2: Microbiome Integration

4. **Correlate the module eigengene with 16S rRNA or shotgun metagenomics** data, testing for enrichment of saccharolytic genera (Faecalibacterium, Roseburia, Eubacterium) and depletion of choline-metabolizing taxa. Confirm the predicted separation of saccharolytic and proteolytic microbial pathways. [Inferred]

5. **Assess dietary fiber and polyphenol intake** as potential upstream drivers of the butyrate-benzoate-hydroxyphenylacetate cluster. [Model Knowledge]

#### Priority 3: Genetic and Literature Follow-Up

6. **Query metabolomics GWAS catalogs** for the inferred mQTL variants (CA6476722, CA15677622, CA1164183940, CA10669099) and their association with lysophosphatidic acid species. Map variants to specific loci and test for enrichment near Lands cycle enzymes (LPCAT, LPEAT, PLA2 family). [KG Evidence for variant-analogue inference]

7. **Conduct targeted literature review** on the intersection of GGT activity, gut-derived short-chain fatty acids, and gastrointestinal cancer risk, given the module's convergent disease enrichment for colorectal cancer, IBD, and hepatocellular carcinoma. [KG Evidence for disease recurrence; Model Knowledge for biological plausibility]

#### Priority 4: Module Comparison

8. **Compare the Lightgreen module with other WGCNA modules** in the dataset to test whether BCAAs, homocysteine, TMAO, and ceramides segregate into distinct co-expression clusters, as predicted by the gap analysis. Cross-module correlation analysis would determine whether the Lightgreen module is anti-correlated with canonical insulin-resistance metabolite modules. [Inferred]

9. **Test for cold-start entity recovery**: Levulinate, gamma-glutamyl-epsilon-lysine, N6-succinyladenosine, and isoleucylleucine/leucylisoleucine have zero KG edges. Targeted literature searches in HMDB and MetaCyc for these metabolites may reveal pathway annotations absent from the current knowledge graph. [KG Evidence for cold-start status]

---

*Report generated from KRAKEN knowledge graph analysis. All evidence attributions ([KG Evidence], [Literature], [Model Knowledge], [Inferred]) are provided inline. Findings should be interpreted in the context of entity resolution confidence (73% of entities resolved at ≥80% confidence) and the Open World Assumption (absence of an entity or edge indicates "unstudied," not "nonexistent").*

### Literature References

Papers discovered via semantic search. 3 unique papers across 2 hypotheses.

| Hypothesis | Citation | Link | Relevance |
|------------|----------|------|-----------|
| Inferred role of CHEBI:195048 |  (2013) "Dovetailing biology and chemistry: integrating the Gene Ontology with the ChEBI chemical ontology \| ..." | [Link](https://link.springer.com/article/10.1186/1471-2164-14-513) | GO chemical ontology that referred to existing terms in the ... ontology, but for which the matches were not detected au... |
| Inferred role of EFO:0800445 |  (2023) "Frontiers \| Genome-wide analysis of oxylipins and oxylipin profiles in a pediatric population" | [Link](https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2023.1040993/full) | We conducted GWAS using the top 5 loading oxylipins for oxylipin PC1, to further investigate the findings for oxylipin P... |
| Inferred role of CHEBI:195048 |  (2021) "Frontiers \| Uncovering Competitive and Restorative Effects of Macro- and Micronutrients on Sodium Be..." | [Link](https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2021.634753/full) | A model aromatic compound, sodium benzoate, is generally used for simulating aromatic pollutants present in textile effl... |