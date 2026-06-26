# Lightgreen Module Run: Discovery Output (29-analyte, dev, 2026-06-23)

This document presents the discovery output of a full-module run of the **Lightgreen** WGCNA module through the Kraken 12-node discovery pipeline, executed on the dev environment on 2026-06-23 (commit `813fc3f`) with BioMapper entity resolution enabled. It is one of a per-module series over the Frailty multi-omics WGCNA modules; see the [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm).

## Run Provenance

The run submitted 29 named analytes, parsed 30 at intake, and resolved 30 distinct entities (14 fuzzy, 15 biomapper, 1 exact) to 30 distinct CURIEs. Triage classified 5 well-characterized, 10 moderate, 11 sparse, and 4 cold-start, with 0 edge-count measurement failures. Downstream analysis produced 654 direct-KG findings, 36 cold-start findings, 4 biological themes, 0 cross-entity bridges (0 evidence-grounded), and 70 hypotheses supported by 19 literature references. Synthesis emitted a 25649-character report. The run completed in approximately 752.8 s of wall-clock time (status complete, 30 errors). All counts are taken from the reducer-merged pipeline state.

| Stage | Result |
|---|---|
| Submitted | 29 named analytes |
| Intake | 30 parsed |
| Entity resolution | 30 resolved (14 fuzzy, 15 biomapper, 1 exact) to 30 distinct CURIEs |
| Triage | 5 well-characterized, 10 moderate, 11 sparse, 4 cold-start (0 measurement failures) |
| Direct KG | 654 findings |
| Cold-start | 36 findings, 7 skipped |
| Pathway enrichment | 4 biological themes |
| Integration | 0 bridges (0 evidence-grounded) |
| Literature grounding | 19 papers |
| Synthesis | 70 hypotheses, 25649-character report |
| Run total | ~752.8 s wall-clock, status complete, 30 errors |

## Related

- Companion run metrics: [Lightgreen Module Run: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/lightgreen-module-run-pipeline-performance-report-29-analyte-dev-2026-06-23-dYmjwNiqBg)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---



## Lightgreen WGCNA Module: Discovery Report

### Gut Microbial Fermentation, Transsulfuration, and Endocannabinoid Signaling as a Coordinated Metabolic Axis

---

### 1. Executive Summary

The Lightgreen WGCNA module encodes a coherent metabolic program uniting three biological axes: gut microbial fermentation (butyrate, benzoate, 2-hydroxyphenylacetate), transsulfuration and vitamin B6 cofactor metabolism (cystathionine, pyridoxal, gamma-glutamyl peptides), and endocannabinoid/polyunsaturated fatty acid lipid signaling (arachidonoyl ethanolamide, complex glycerophospholipids, retinal). [Inferred] The co-expression of these biochemically distinct metabolite classes indicates a shared upstream regulatory program, most plausibly reflecting the intersection of Firmicutes-dominated gut fermentation with hepatic B6-dependent disposal pathways and membrane phospholipid remodeling. [Inferred] Disease enrichment analysis converges on gastrointestinal malignancies (colorectal cancer, pancreatic neoplasm, hepatocellular carcinoma) and inflammatory bowel disease, supporting a gut-liver axis interpretation with clinical relevance to cardiometabolic and oncologic outcomes. [KG Evidence]

---

### 2. Key Findings (Tier 1 to 2)

#### 2.1 Disease Associations

The module's disease landscape is dominated by gastrointestinal and hepatic pathology. Colorectal cancer recurs across three module members (pyridoxal, cytidine, xanthosine; strongest evidence: curated), establishing it as the most robustly supported disease association. [KG Evidence] Inflammatory bowel disease (IBD) recurs via two independent associations: cytidine and xanthosine share a curated link to IBD type 1 (MONDO:0009960), while pyridoxal and cytidine share text-mined associations with general IBD (MONDO:0005265). [KG Evidence] Melanoma (MONDO:0005105) is linked to pyridoxal, cytidine, and arachidonoyl ethanolamide (text-mined), suggesting a possible role for endocannabinoid signaling in this context. [KG Evidence] Hepatocellular carcinoma and malignant pancreatic neoplasm each associate with pyridoxal and cytidine (text-mined). [KG Evidence] Peripheral neuropathy is linked to pyridoxal and cytidine (text-mined), consistent with established vitamin B6 neurotoxicity at supraphysiologic concentrations. [KG Evidence]

Hypotensive disorder associates with cytidine and arachidonoyl ethanolamide (curated), a connection with mechanistic plausibility given that anandamide is a potent vasodilator acting via CB1 receptors and vanilloid channels. [KG Evidence; Model Knowledge]

Among individual high-priority members, pyridoxal participates in 18 inborn errors of branched-chain amino acid metabolism (beta-ketothiolase deficiency, maple syrup urine disease, isovaleric acidemia, propionic acidemia, methylmalonic aciduria, among others), reflecting its role as a cofactor for aminotransferases in these pathways. [KG Evidence] Butyrate's top disease association is Epstein-Barr virus infection, consistent with histone deacetylase inhibition by short-chain fatty acids modulating viral latency. [KG Evidence] Retinal associates with vitamin A deficiency (curated). [KG Evidence] Arachidonoyl ethanolamide associates with cirrhosis of liver (curated), aligning with the established role of the endocannabinoid system in hepatic fibrogenesis. [KG Evidence]

#### 2.2 Pathway Memberships

Two pathways recur across module members. Vitamin transport (GO:0051180) is shared by retinal and pyridoxal, reflecting their status as fat-soluble and water-soluble vitamin forms, respectively. [KG Evidence] ABC transporters (KEGG:02010) is shared by cytidine and xanthosine, both nucleosides requiring concentrative or equilibrative transporters for cellular uptake and efflux. [KG Evidence]

Pyridoxal participates directly in vitamin B6 metabolism. [KG Evidence] Cytidine participates in pyrimidine metabolism, pyrimidine ribonucleoside degradation, and nucleotide catabolism pathways. [KG Evidence] Butyrate participates in multiple biosynthetic and catabolic processes including glucose catabolic process to butyrate and glutamate catabolic process to butyrate. [KG Evidence] Arachidonoyl ethanolamide participates in neuroactive ligand-receptor interaction. [KG Evidence] Retinal participates in retinol metabolism. [KG Evidence]

#### 2.3 Key Molecular Interactions (Tier 2)

Pyridoxal interacts with 10 established enzymatic partners including PDXK (pyridoxal kinase), AOX1 (aldehyde oxidase), PDXP (pyridoxal phosphatase), PNPO (pyridoxamine 5'-phosphate oxidase), ALPL (tissue-nonspecific alkaline phosphatase), and AGXT (alanine:glyoxylate aminotransferase). [KG Evidence] Novel (hidden gem) interactions were detected between pyridoxal and TNF, IL1B, IL6, INS, MAPK1, MAPK3, HMOX1, and PPARA, implicating pyridoxal in inflammatory cytokine signaling, insulin signaling, and peroxisome proliferator-activated receptor alpha pathways. [KG Evidence] These novel interactions, if validated, would position pyridoxal as a node linking B6 cofactor status to systemic inflammation and metabolic regulation.

Cytidine interacts with nucleoside kinases (UCK1, UCK2), cytidine deaminase (CDA), nucleotidases (NT5E, NT5C2, NT5C, NT5M, NT5C1A, NT5C3A), nucleoside transporters (SLC28A1, SLC28A3), and APOBEC3B (cytidine deaminase involved in innate immunity). [KG Evidence] The diversity of cytidine-processing enzymes in the knowledge graph highlights cytidine as a metabolic hub at the intersection of pyrimidine salvage and innate antiviral defense.

#### 2.4 Member Prioritization

The five well-characterized members (cytidine: 547 edges; butyrate: 496 edges; retinal: 483 edges; pyridoxal: 374 edges; arachidonoyl ethanolamide: 209 edges) anchor the module's biological interpretation. [KG Evidence] Pyridoxal and cytidine jointly drive the majority of recurrent disease associations, making them the highest-leverage members for follow-up investigation. Four entities (levulinate, gamma-glutamyl-epsilon-lysine, N6-succinyladenosine, isoleucylleucine/leucylisoleucine) are cold-start in the knowledge graph (zero edges), representing genuinely novel analytes whose biology cannot be assessed through current KG resources. [KG Evidence]

---

### 3. Novel Predictions (Tier 3)

#### 3.1 Cyclo(ala-pro) as a Potential Cholecystokinin Receptor Ligand

The cyclic dipeptide cyclo(ala-pro), resolved to PUBCHEM.COMPOUND:53321822, exhibits 89% structural similarity to cyclo(CYIQNCPLG) (PUBCHEM.COMPOUND:53321821), which binds cholecystokinin (CCK; NCBIGene:5021) in the knowledge graph. [KG Evidence] This structural similarity raises the hypothesis that cyclo(ala-pro) may interact with CCK or its receptors. The grounded literature confirms that natural cyclic peptides exhibit diverse receptor-binding capabilities (Natural Cyclic Peptides: Synthetic Strategies, 2025) [Literature], and cyclo(His-Pro), a structurally analogous endogenous cyclic dipeptide, demonstrates established neuromodulatory activity via dopaminergic mechanisms (Focus on cyclo(His-Pro), 2007) [Literature]. The CCK connection is mechanistically intriguing because CCK regulates gastric acid secretion, pancreatic enzyme release, and gut motility, processes directly relevant to the module's gut-derived metabolite signature. Approximately 18% of computational predictions of this type progress to clinical investigation; experimental validation through competitive binding assays against CCK receptors (CCK-A/CCK-B) is warranted. [Inferred]

**Logic chain:** cyclo(ala-pro) → 89% similarity to cyclo(CYIQNCPLG) → binds CCK (NCBIGene:5021) → potential gut hormone signaling connection to butyrate and microbial metabolite co-expression.

**Validation step:** Perform radioligand displacement assays using synthetic cyclo(ala-pro) against [125I]-CCK-8 at CCK-A and CCK-B receptors; query ChEMBL for existing bioactivity data on PUBCHEM.COMPOUND:53321822.

#### 3.2 Genetic Variants Governing 1-Linoleoyl-GPA Levels

The lysophosphatidic acid species 1-linoleoyl-GPA (EFO:0800445) shares 91% semantic similarity with 1-linoleoyl-GPE (18:2) and 89% with 1-linoleoyl-GPG (18:2). [KG Evidence] Both GPE and GPG are associated with multiple mQTL variants (CA6476722, CA15677622, CA1164183940, CA10669099) via has_phenotype edges. [KG Evidence] The inference that these variants also influence GPA levels rests on the shared linoleoyl (18:2) acyl chain, which is subject to common enzymatic processing by phospholipases and lysophospholipid acyltransferases. [Inferred] The grounded literature demonstrates that GWAS approaches can detect genetic loci governing oxylipin and phospholipid profiles (Genome-wide analysis of oxylipins, 2023) [Literature], supporting the feasibility of mQTL discovery for LPA species. However, none of the retrieved abstracts directly report GPA-specific mQTL associations, leaving this prediction unvalidated. The ~18% validation rate for such computational inferences applies.

**Logic chain:** 1-linoleoyl-GPA → 91% similarity to 1-linoleoyl-GPE → GPE has mQTL hits (CA6476722, CA15677622, CA1164183940, CA10669099) → shared 18:2 acyl chain metabolism → plausible shared genetic regulation.

**Validation step:** Query the GWAS Catalog and Metabolon/Nightingale mQTL databases for lysophosphatidic acid 18:2 associations; genotype the priority variants (CA6476722, CA15677622) in a cohort with lipidomic coverage including LPA species.

#### 3.3 Pyridoxal-Inflammatory Cytokine Axis

The knowledge graph reveals novel (hidden gem) interactions between pyridoxal and TNF, IL1B, IL6, MAPK1/3, and INS. [KG Evidence] No direct KG evidence was found for the mechanism underlying these interactions. The following is based on [Model Knowledge]: vitamin B6 deficiency is associated with elevated C-reactive protein and pro-inflammatory cytokine production; pyridoxal 5'-phosphate (the active cofactor form) modulates NF-kB signaling, and pyridoxal's interaction with NFKB1 (NCBIGene:4790) is established in the KG. [KG Evidence for NFKB1; Model Knowledge for mechanism] The co-expression of pyridoxal with gut-derived fermentation products (butyrate, a known HDAC inhibitor with anti-inflammatory properties) in this module suggests a coordinated anti-inflammatory metabolic program. [Inferred] The ~18% validation calibration note applies to these novel interaction predictions.

**Logic chain:** Pyridoxal → established NFKB1 interaction [KG Evidence] → novel TNF/IL1B/IL6 interactions [KG Evidence, hidden gem] → anti-inflammatory synergy with butyrate (HDAC inhibition) [Inferred] → module as coordinated gut-inflammation regulatory unit.

**Validation step:** Measure plasma pyridoxal alongside TNF, IL1B, and IL6 in the study cohort; perform Mendelian randomization using PDXK variants as instruments for pyridoxal levels on inflammatory cytokine outcomes.

---

### 4. Biological Themes

#### 4.1 Gut Microbial Fermentation Guild Signature

The module contains butyrate, 2-hydroxyphenylacetate (a product of phenylalanine/tyrosine microbial catabolism), benzoate (aromatic ring degradation product), and caproate (6:0; a medium-chain fatty acid produced by chain-elongating Clostridiales). [KG Evidence for entity presence; Model Knowledge for microbial attribution] This metabolite constellation is characteristic of Firmicutes-associated fermentative metabolism, specifically Faecalibacterium, Roseburia, and related butyrate-producing Clostridiales. [Model Knowledge] The absence of propionate, acetate, TMAO, and indole derivatives (discussed in Section 5) sharpens this interpretation: the module does not capture a generic "gut microbiome" signal but rather the output of a specific fermentative guild.

#### 4.2 Transsulfuration and Gamma-Glutamyl Transferase Activity

Cystathionine (the intermediate of the transsulfuration pathway) co-expresses with pyridoxal (the essential CBS cofactor), gamma-glutamylhistidine, gamma-glutamylalanine, and gamma-glutamyl-epsilon-lysine. [KG Evidence for entity presence] The gamma-glutamyl peptides are products of gamma-glutamyl transferase (GGT) activity, which catalyzes the transfer of the gamma-glutamyl moiety from glutathione to amino acid acceptors. [Model Knowledge] This pattern captures GSH turnover dynamics rather than GSH steady-state, as further elaborated in the gap analysis.

#### 4.3 Endocannabinoid and Membrane Phospholipid Remodeling

Arachidonoyl ethanolamide (anandamide) is the prototypical endocannabinoid, and its co-expression with multiple glycerophospholipid species (GPE, GPC, GPI, and GPA headgroup classes with polyunsaturated acyl chains including 20:3, 20:4, 20:5, 22:5, and 22:6) indicates coordinated phospholipid membrane remodeling. [KG Evidence for entity presence; Inferred for coordination] Retinal, a fat-soluble vitamin A aldehyde, participates in vitamin transport alongside pyridoxal (GO:0051180). [KG Evidence] The lipase gene enrichment (LIPC, LIPA, LIPF, PNPLA3, CEL) connecting seven input entities through the pathway analysis further supports active phospholipid processing as a unifying theme. [KG Evidence]

#### 4.4 Nucleoside Metabolism

Cytidine and xanthosine, a pyrimidine and purine nucleoside respectively, share ABC transporter pathway membership and connect through the hub nodes "nucleoside role" (CHEBI:76971), "pyrimidine nucleoside role" (CHEBI:75771), and "purine nucleoside role" (CHEBI:77746). [KG Evidence] These hub nodes connect three input entities but, being high-connectivity chemical class definitions, should be interpreted as ontological groupings rather than specific mechanistic links. [KG Evidence, hub-flagged; de-emphasized accordingly] The co-expression of N6-succinyladenosine (a modified purine nucleoside, cold-start in the KG) with cytidine and xanthosine may reflect shared nucleotide salvage or catabolism dynamics. [Inferred]

#### 4.5 Hub-Flagged Entities (De-emphasized)

The cytoplasm (GO:0005737) connects two input entities but represents the most generic cellular compartment annotation in gene ontology; this connection carries no discriminative biological information. [KG Evidence, hub-flagged] The nucleoside role chemical class nodes (CHEBI:76971, CHEBI:75771, CHEBI:77746, CHEBI:75772) are similarly ontological hubs. [KG Evidence, hub-flagged] These associations are reported for completeness but are uninformative for module interpretation.

---

### 5. Gap Analysis

Using the Open World Assumption, absence of an entity from the module means "not co-expressed with this group," not "biologically irrelevant."

#### 5.1 Informative Absences

**Homocysteine** is the direct upstream substrate of cystathionine in the transsulfuration pathway, and pyridoxal (as PLP) is the essential cofactor for cystathionine beta-synthase (CBS) that catalyzes this conversion. [Model Knowledge] The absence of homocysteine from this module, despite the presence of both its immediate product (cystathionine) and its required cofactor (pyridoxal), is among the most informative gaps. [Inferred] Homocysteine variance is governed by remethylation (folate/B12-dependent methionine synthase) as well as transsulfuration; its co-expression likely resides in a separate one-carbon metabolism module. This decoupling implies that the Lightgreen module captures transsulfuration disposal capacity independent of methionine cycle flux.

**Propionate and acetate** are typically co-produced with butyrate as gut microbial SCFAs, yet their absence indicates that the module captures Firmicutes-specific butyrate production (via butyryl-CoA:acetate CoA transferase in Faecalibacterium and Roseburia) rather than a generic fermentation signal. [Model Knowledge; Inferred]

**Glutathione (GSH)** is conspicuously absent despite the presence of its precursor (cystathionine supplies cysteine for GSH synthesis) and its degradation products (gamma-glutamylhistidine and gamma-glutamylalanine, generated by GGT-mediated GSH catabolism). [Model Knowledge; Inferred] This dissociation indicates that the module captures the kinetics of GSH turnover (GGT-dependent catabolism rate) rather than the thermodynamic steady-state of the GSH pool, which is governed independently by gamma-glutamylcysteine synthetase activity and oxidative stress burden.

**TMAO** absence separates "fermentative gut metabolites" (butyrate, 2-hydroxyphenylacetate, benzoate; present) from "choline-pathway gut metabolites" (TMAO; absent), reinforcing the guild-specific microbial interpretation. [Model Knowledge; Inferred]

**Indole derivatives (indoxyl sulfate, indole-3-propionic acid)** absence, despite the presence of a tryptophan-containing dipeptide (tryptophylleucine), further refines the microbial guild assignment: indole production via tryptophanase is dominated by Escherichia and Bacteroides, taxonomically distinct from the Firmicutes-associated fermenters implicated here. [Model Knowledge; Inferred]

**Bile acids** absence, despite the lipase gene enrichment (LIPC, LIPA, LIPF, PNPLA3, CEL) in the pathway analysis, clarifies that these lipase connections relate to phospholipid and endocannabinoid processing, not to bile acid metabolism governed by the FXR-FGF19 axis. [Model Knowledge; Inferred]

**Lysophosphatidylcholines (LysoPC)** absence, despite the presence of arachidonoyl ethanolamide and multiple glycerophospholipid species, distinguishes NAPE-PLD-mediated endocannabinoid synthesis from PLA1/PLA2/LIPC-mediated lysophospholipid generation. [Model Knowledge; Inferred] LysoPC species may reside in an adjacent WGCNA module; cross-module correlation analysis is recommended.

#### 5.2 Standard Gaps

**BCAAs (leucine, isoleucine, valine)** typically form a tight co-expression cluster driven by BCKDH activity and insulin signaling, a regulatory axis distinct from the proteolytic/peptidase activity captured by the dipeptides in this module. [Model Knowledge] **Ceramides** are governed by de novo sphingolipid synthesis and sphingomyelinase activity, separate from the endocannabinoid system (FAAH/NAPE-PLD). [Model Knowledge] **Medium- and long-chain acylcarnitines** reflect mitochondrial beta-oxidation flux, fundamentally different from endocannabinoid or SCFA pathways. [Model Knowledge] These absences are structurally expected in WGCNA and do not represent anomalies.

---

### 6. Temporal Context

No explicit longitudinal design or time-course metadata accompanies this analysis. However, the module's composition permits provisional causal ordering. [Inferred]

**Upstream causes (earlier in causal chain):** Gut microbial fermentation (producing butyrate, benzoate, 2-hydroxyphenylacetate, caproate) and dietary vitamin B6 intake (supplying pyridoxal) represent environmental and dietary inputs. Membrane phospholipid composition (the complex glycerophospholipid species) reflects chronic dietary fatty acid intake and hepatic desaturase/elongase activity.

**Downstream consequences (later in causal chain):** GGT-mediated gamma-glutamyl peptide generation (gamma-glutamylhistidine, gamma-glutamylalanine) is a consequence of GSH turnover. Arachidonoyl ethanolamide levels reflect on-demand synthesis via NAPE-PLD in response to cellular signals. Cystathionine accumulation reflects the balance between CBS-mediated production and cystathionine gamma-lyase (CSE)-mediated cleavage.

**Causal inference opportunity:** If longitudinal samples are available, Granger causality or vector autoregression models could test whether changes in butyrate precede or follow changes in gamma-glutamyl peptides, discriminating between "gut fermentation drives GSH turnover" and "GSH turnover products are independently co-regulated with SCFA production." [Inferred]

---

### 7. Research Recommendations

#### 7.1 High Priority: Experimental Validations

1. **Cyclo(ala-pro) and CCK receptor binding:** Synthesize cyclo(ala-pro) and test binding affinity at CCK-A and CCK-B receptors via radioligand displacement assay. Query ChEMBL and BindingDB for existing bioactivity data on PUBCHEM.COMPOUND:53321822. This is the module's most actionable novel prediction linking a co-expressed metabolite to gut hormone signaling. [Inferred; ~18% validation rate applies]

2. **Pyridoxal-cytokine interactions:** Measure plasma pyridoxal, PLP, TNF, IL-1beta, and IL-6 in the study cohort. Test whether pyridoxal levels inversely correlate with pro-inflammatory cytokines after adjusting for butyrate, testing the inferred anti-inflammatory synergy. [KG Evidence for interactions; Inferred for synergy hypothesis]

3. **mQTL validation for 1-linoleoyl-GPA:** Genotype priority variants (CA6476722, CA15677622, CA1164183940, CA10669099) in a cohort with lipidomic coverage including lysophosphatidic acid species; test whether known GPE/GPC-associated variants also influence GPA levels. [KG Evidence for variant-GPE associations; Inferred for GPA extension; ~18% validation rate applies]

#### 7.2 Moderate Priority: Targeted Literature Searches

4. **Butyrate-cystathionine co-regulation:** Search for studies examining whether gut-derived butyrate (via HDAC inhibition) modulates CBS or CSE expression, directly connecting the fermentation and transsulfuration axes.

5. **Anandamide-phospholipid membrane dynamics:** Search for lipidomic studies examining co-variance of anandamide with glycerophospholipid species containing 20:3, 20:4, 20:5, 22:5, and 22:6 acyl chains, to determine whether their co-expression reflects shared membrane remodeling.

6. **N6-succinyladenosine biology:** This modified nucleoside is cold-start in the KG. Targeted literature search in HMDB and recent metabolomics publications may reveal pathway assignments and disease associations unavailable through the knowledge graph.

#### 7.3 Follow-Up Computational Analyses

7. **Cross-module bridge analysis:** Correlate the Lightgreen module eigengene with eigengenes of adjacent modules, specifically testing for partial overlap with modules containing homocysteine, LysoPC species, and BCAAs. Shared genetic regulation (via LIPC, PNPLA3) may bridge the Lightgreen phospholipid submodule to a lysophospholipid-enriched neighbor.

8. **Microbial guild assignment:** Perform shotgun metagenomics or 16S rRNA sequencing on matched samples. Correlate butyrate, 2-hydroxyphenylacetate, and benzoate levels with relative abundances of Faecalibacterium prausnitzii, Roseburia intestinalis, and Clostridium cluster XIVa to confirm the Firmicutes-specific fermentation guild hypothesis.

9. **Entity resolution refinement:** Four entities with fuzzy resolution at 70% confidence (caproate resolved to 17alpha-hydroxyprogesterone caproate; tartarate, N6-succinyladenosine, and several complex glycerophospholipids) require manual curation. The caproate mapping to a pharmaceutical steroid ester (hydroxyprogesterone caproate) is likely incorrect; the intended analyte is hexanoic acid (CHEBI:30776). This misresolution inflates KG edge counts for caproate (111 edges may reflect the steroid, not the fatty acid) and should be corrected before downstream analyses.

10. **Cold-start entity characterization:** Levulinate, gamma-glutamyl-epsilon-lysine, N6-succinyladenosine, and isoleucylleucine/leucylisoleucine have zero KG edges. Experimental characterization of these analytes through targeted MS/MS fragmentation, enzymatic synthesis assays, or metabolic flux analysis would be maximally informative for expanding knowledge graph coverage.

---

#### Methodological Caveats

Several entity resolutions carry 70% (fuzzy) confidence scores, most notably caproate (resolved to 17alpha-hydroxyprogesterone caproate rather than hexanoic acid), tartarate, N6-succinyladenosine (resolved to N6-benzoyladenosine), and multiple complex glycerophospholipids resolved to triglyceride rather than phospholipid structures. These misresolutions may have introduced spurious KG connections or missed genuine ones. All findings involving these entities should be interpreted with this caveat. The "Metabolites:" header string was erroneously resolved to substance P-metabolite 5-11 (UMLS:C1698199) and should be excluded from biological interpretation. The cyclo(ala-pro) entity was resolved to a 15-residue cyclic peptide (PUBCHEM.COMPOUND:53321822), which is structurally distinct from the intended dipeptide; the CCK-binding inference thus requires re-evaluation against the correct cyclo(Ala-Pro) structure (CHEBI:59033 or equivalent).

### Literature References

Papers discovered via semantic search. 3 unique papers across 2 hypotheses.

| Hypothesis | Citation | Link | Relevance |
|------------|----------|------|-----------|
| Inferred role of CHEBI:195048 |  (2013) "Dovetailing biology and chemistry: integrating the Gene Ontology with the ChEBI chemical ontology \| ..." | [Link](https://link.springer.com/article/10.1186/1471-2164-14-513) | GO chemical ontology that referred to existing terms in the ... ontology, but for which the matches were not detected au... |
| Inferred role of EFO:0800445 |  (2023) "Frontiers \| Genome-wide analysis of oxylipins and oxylipin profiles in a pediatric population" | [Link](https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2023.1040993/full) | We conducted GWAS using the top 5 loading oxylipins for oxylipin PC1, to further investigate the findings for oxylipin P... |
| Inferred role of CHEBI:195048 |  (2021) "Frontiers \| Uncovering Competitive and Restorative Effects of Macro- and Micronutrients on Sodium Be..." | [Link](https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2021.634753/full) | A model aromatic compound, sodium benzoate, is generally used for simulating aromatic pollutants present in textile effl... |