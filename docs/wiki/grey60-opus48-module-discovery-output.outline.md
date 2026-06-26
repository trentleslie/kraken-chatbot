# Grey60 Module Run on Opus 4.8: Discovery Output (32-analyte, dev, 2026-06-24)

This document presents the discovery output of a full-module run of the **Grey60** WGCNA module through the Kraken 12-node discovery pipeline, executed on the dev environment on 2026-06-24 (commit `a7d7fd6`) with BioMapper entity resolution enabled. The SDK-backed pipeline nodes ran on the Opus 4.8 model rather than the default Sonnet; this run is published for direct model comparison. It is one of a per-module series over the Frailty multi-omics WGCNA modules; see the [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm).

## Run Provenance

The run submitted 32 named analytes, parsed 33 at intake, and resolved 33 distinct entities (7 fuzzy, 25 biomapper, 1 exact) to 33 distinct CURIEs. Triage classified 3 well-characterized, 3 moderate, 21 sparse, and 6 cold-start, with 0 edge-count measurement failures. Downstream analysis produced 263 direct-KG findings, 24 cold-start findings, 3 biological themes, 0 cross-entity bridges (0 evidence-grounded), and 44 hypotheses supported by 13 literature references. Synthesis emitted a 24340-character report. The run completed in approximately 474.6 s of wall-clock time (status complete, 6 errors). All counts are taken from the reducer-merged pipeline state.

| Stage | Result |
|---|---|
| Submitted | 32 named analytes |
| Intake | 33 parsed |
| Entity resolution | 33 resolved (7 fuzzy, 25 biomapper, 1 exact) to 33 distinct CURIEs |
| Triage | 3 well-characterized, 3 moderate, 21 sparse, 6 cold-start (0 measurement failures) |
| Direct KG | 263 findings |
| Cold-start | 24 findings, 19 skipped |
| Pathway enrichment | 3 biological themes |
| Integration | 0 bridges (0 evidence-grounded) |
| Literature grounding | 13 papers |
| Synthesis | 44 hypotheses, 24340-character report |
| Run total | ~474.6 s wall-clock, status complete, 6 errors |

## Related

- Companion run metrics: [Grey60 Module Run on Opus 4.8: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/grey60-module-run-on-opus-48-pipeline-performance-report-32-analyte-dev-2026-06-24-u5b4eDZxlq)
- Model comparison baseline (Sonnet): [Grey60 Module Run: Discovery Output](https://phwiki.phenoma.ai/doc/grey60-module-run-discovery-output-32-analyte-dev-2026-06-23-GPq51xU6sG)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---



## Grey60 WGCNA Module Discovery Report: A Gut Microbiome to Hepatic Detoxification Metabolite Network

---

### 1. Executive Summary

The Grey60 WGCNA module encodes a coherent metabolic axis spanning gut microbial fermentation of dietary polyphenols and aromatic amino acids, hepatic phase II sulfation and glycine conjugation, and secondary bile acid processing. [Inferred] Thirty-two small molecules co-express in this module; the majority are sulfated phenolic compounds, hippurate-family glycine conjugates, and unconjugated secondary bile acids, collectively fingerprinting the microbiome to liver detoxification corridor. [KG Evidence] This module's composition implies that inter-individual variation in colonic microbial ecology and hepatic conjugation capacity drives coordinated metabolite abundance, with direct relevance to type 2 diabetes mellitus (T2D) and hepatobiliary disease risk stratification. [Inferred]

---

### 2. Key Findings (Tier 1 to 2)

#### 2.1 Unifying Biological Theme: Microbiome-Derived Phenolics and Their Hepatic Conjugates

The module is dominated by three biochemically linked metabolite classes whose co-expression reveals a single production pipeline:

1. **Sulfated phenolics** (catechol sulfate, hydroquinone sulfate, 4-ethylphenylsulfate, 4-vinylphenol sulfate, guaiacol sulfate, eugenol sulfate, 4-allylphenol sulfate, 4-acetylphenol sulfate, 1,2,3-benzenetriol sulfate, 3-methoxycatechol sulfate, 4-ethylcatechol sulfate, 4-allylcatechol sulfate): these compounds arise when colonic bacteria degrade dietary polyphenols and lignans into simple phenols, which the liver then sulfates via sulfotransferases (SULTs). [Inferred] Their tight co-expression indicates a shared rate-limiting step, likely colonic microbial phenol production or hepatic SULT capacity.

2. **Glycine conjugates** (hippurate, 4-hydroxyhippurate, 3-hydroxyhippurate, cinnamoylglycine): hippurate participates in excretion pathways [KG Evidence] and is the canonical product of microbial benzoate production followed by mitochondrial glycine N-acyltransferase (GLYAT) conjugation. [Model Knowledge] The hydroxylated hippurate variants and cinnamoylglycine extend this axis to hydroxylated and unsaturated cinnamate precursors.

3. **Unconjugated secondary bile acids** (ursodeoxycholate, isoursodeoxycholate, glycoursodeoxycholate): ursodeoxycholate participates in bile acid secretion, cholesterol biosynthetic processes, and hepatocyte apoptotic processes [KG Evidence]; it interacts with the bile salt export pump ABCB11, the sodium/bile acid cotransporter SLC10A1, the farnesoid X receptor NR1H4, and multiple OATP transporters (SLCO1B1, SLCO1B3, SLCO1A2) [KG Evidence]. Isoursodeoxycholate, a 3-beta epimer of ursodeoxycholate, shares an association with primary biliary cholangitis [KG Evidence].

#### 2.2 Disease Associations with Direct KG Support

| Disease | Members Implicated | Evidence Type | Source |
|---|---|---|---|
| Primary biliary cholangitis | Ursodeoxycholate (treats), isoursodeoxycholate | Curated | [KG Evidence] |
| Type 2 diabetes mellitus | Ursodeoxycholate (studied to treat) | Curated | [KG Evidence] |
| Colorectal cancer | Hippurate, 4-hydroxycoumarin | Curated | [KG Evidence] |
| Cholestasis / intrahepatic cholestasis | Ursodeoxycholate (treats) | Curated | [KG Evidence] |
| Metabolic dysfunction-associated steatotic liver disease | Ursodeoxycholate (studied to treat) | Curated | [KG Evidence] |
| Phenylketonuria | Hippurate | Curated | [KG Evidence] |
| Acute myeloid leukemia | Ursodeoxycholate, 4-hydroxycoumarin | Curated | [KG Evidence] |
| Liver cancer | Ursodeoxycholate, 4-hydroxycoumarin | Curated | [KG Evidence] |
| Obesity disorder | Ursodeoxycholate (studied to treat) | Novel connection | [KG Evidence] |
| Diabetes mellitus (broad) | Ursodeoxycholate (studied to treat) | Novel connection | [KG Evidence] |

Ursodeoxycholate accounts for the vast majority of curated disease associations (1,824 edges), and its KG profile encompasses hepatobiliary, metabolic, and even oncologic contexts. [KG Evidence] This hub status warrants cautious interpretation: disease associations mediated solely through ursodeoxycholate may reflect its pharmacological use rather than endogenous biology. However, the T2D and MASLD connections are biologically convergent with the module's microbial phenolic signature, lending these associations greater credibility in this context. [Inferred]

#### 2.3 Key Molecular Interactions

Ursodeoxycholate engages nuclear receptors and transporters central to bile acid homeostasis and metabolic regulation [KG Evidence]:

- **NR1H4 (FXR)**: the master bile acid sensor; ursodeoxycholate interaction suggests FXR-mediated transcriptional regulation of bile acid synthesis and glucose metabolism. [KG Evidence; Model Knowledge]
- **GPBAR1 (TGR5)**: a membrane bile acid receptor linked to GLP-1 secretion and energy expenditure; interaction with ursodeoxycholate provides a mechanistic link between this module and T2D risk. [KG Evidence; Model Knowledge]
- **AHR (aryl hydrocarbon receptor)**: a novel KG-identified interaction partner for ursodeoxycholate [KG Evidence, hidden_gems], and notably, AHR is also activated by indole derivatives (indolepropionate is a module member) and by microbial tryptophan metabolites. [Model Knowledge] This convergence suggests AHR signaling as a point of integration between the bile acid and phenolic arms of this module.
- **HMGCR**: ursodeoxycholate interaction with HMG-CoA reductase [KG Evidence, hidden_gems] aligns with the established role of ursodeoxycholate in cholesterol metabolism and the KG-confirmed participation in cholesterol biosynthetic processes. [KG Evidence]

Gentisate participates in tyrosine metabolism [KG Evidence], connecting another module member to aromatic amino acid catabolism by gut microbiota. 3-(3-Hydroxyphenyl)propanoate participates in its own catabolic, metabolic, and biosynthetic processes [KG Evidence], consistent with its role as a microbial degradation product of caffeic acid and chlorogenic acid. [Model Knowledge]

#### 2.4 Pathway Enrichment

The pathway enrichment analysis identified bile acid transport genes SLC10A1 (NTCP) and ABCB11 (BSEP) as shared connector nodes linking module members [KG Evidence]. Both are established ursodeoxycholate interactors [KG Evidence], reinforcing the bile acid transport axis. CA8 (carbonic anhydrase 8) also appeared as a connector [KG Evidence]; its functional relevance to this module is unclear and may reflect hub-driven noise given that 4-hydroxycoumarin (a known carbonic anhydrase inhibitor scaffold, 1,281 edges) likely drives this connection. [Inferred]

---

### 3. Novel Predictions (Tier 3)

#### 3.1 L-Urobilin as a Biomarker of Microbial Heme Catabolism in This Module

**Prediction**: L-urobilin co-varies with gut microbial phenolics because colonic bacteria that produce urobilin from bilirubin overlap taxonomically with those fermenting polyphenols and aromatic amino acids.

**Structural logic chain**: L-urobilin (RM:0136260) is the terminal oxidation product of urobilinogen, itself produced by gut bacterial reduction of bilirubin [Model Knowledge]. By semantic similarity (0.86 to urobilin, 0.82 to D-urobilinogen), L-urobilin is inferred to participate in heme degradation pathways (SMPDB:SMP0000024, SMP0000346) and to correlate with Dubin-Johnson syndrome (MONDO:0009960) [KG Evidence, inferred]. Grounded literature confirms that urobilin binds albumin and is an active metabolite of bilirubin with implications for cardiovascular health, stroke, and diabetes [Literature: "Urobilin Derived from Bilirubin Bioconversion Binds Albumin," 2025; "Metabolic Engineering of Escherichia coli for Production of a Bioactive Metabolite of Bilirubin," 2024]. The 2024 study explicitly notes health benefits of bilirubin-derived metabolites in diabetes and metabolic syndrome [Literature], aligning L-urobilin with the module's cardiometabolic theme.

**Validation step**: Correlate L-urobilin levels with fecal microbiome composition (particularly Clostridium, Ruminococcus, and Faecalibacterium species) and with other module members in participant-level data. Approximately 18% of such computational predictions progress to clinical investigation; this prediction merits priority given the mechanistic coherence.

#### 3.2 Indolepropionate as a Tryptophan-Derived Arm of This Microbial Module

**Prediction**: Indolepropionate, a cold-start entity (0 KG edges under its resolved identifier), is indole-3-propionic acid (IPA), a potent microbial tryptophan metabolite with established roles in gut barrier integrity and T2D risk reduction.

**Structural logic chain**: The entity resolved to PUBCHEM.COMPOUND:157009863 (Indosterol) at 70% confidence, likely a mapping artifact [Inferred]. By semantic similarity (0.77 to Indole Compound, NCIT:C54677), indolepropionate is inferred to belong to the indole chemical class (CHEBI:24828) [KG Evidence, inferred]. Its co-expression with sulfated phenolics, hippurates, and secondary bile acids is consistent with production by Clostridium sporogenes and related commensals that also generate phenylpropionate and hippurate precursors [Model Knowledge]. No grounded literature was fetched for this entity's inferred role.

**Validation step**: Re-map indolepropionate to the correct ChEBI identifier (CHEBI:43580 for indole-3-propionate) and re-query the KG. Correlate with Clostridium sporogenes abundance and GLP-1 levels. Calibration note: approximately 18% of computational predictions advance to clinical testing.

#### 3.3 Ursodeoxycholate to AHR Signaling as a Module Integrator

**Prediction**: The novel KG interaction between ursodeoxycholate and AHR [KG Evidence, hidden_gems] may represent a signaling node where bile acid and microbial phenolic signals converge, as AHR is activated by multiple indole and polyphenol metabolites present in this module.

**Structural logic chain**: Ursodeoxycholate interacts with AHR [KG Evidence]; indolepropionate and sulfated phenolics (catechol sulfate, 4-ethylphenylsulfate) are known or suspected AHR ligands [Model Knowledge]. AHR activation regulates intestinal barrier function, IL-22 production, and hepatic xenobiotic metabolism [Model Knowledge]. This creates a potential feed-forward loop: microbial metabolites activate AHR, which upregulates phase II conjugation enzymes (including SULTs), which in turn sulfate the very metabolites produced by the microbiome. No grounded literature was fetched for this specific interaction.

**Validation step**: Test ursodeoxycholate, catechol sulfate, and indolepropionate individually and in combination in AHR reporter assays. Approximately 18% of such predictions advance to clinical investigation.

#### 3.4 Absence of p-Cresol Sulfate Indicates Pathway Specificity

**Prediction**: The absence of p-cresol sulfate from this module, despite the presence of structurally analogous sulfated phenolics, suggests that p-cresol production (via bacterial tyrosine decarboxylation by Clostridioides difficile and related organisms) is taxonomically and kinetically distinct from the polyphenol/cinnamate fermentation pathways dominating this module.

**Structural logic chain**: p-Cresol sulfate was identified as an informative absence [KG Evidence, gap analysis]. It shares the sulfated phenol structure with 4-ethylphenylsulfate, catechol sulfate, and hydroquinone sulfate [Model Knowledge], yet segregates into a different co-expression module. This implies that the microbial communities producing p-cresol (from tyrosine) differ from those producing catechol (from polyphenols) and 4-ethylphenol (from lignans), or that renal clearance kinetics differ [Inferred].

**Validation step**: Examine cross-module correlations between Grey60 members and p-cresol sulfate. If negatively correlated, this would support competing microbial pathway dynamics. Approximately 18% of computational predictions advance to clinical investigation.

---

### 4. Biological Themes

#### 4.1 Microbial Polyphenol Fermentation and Hepatic Phase II Conjugation

The dominant theme unifying this module is the two-step production pipeline: (i) colonic microbial degradation of dietary polyphenols, lignans, and aromatic amino acids into simple phenols, cinnamates, and benzoates; followed by (ii) hepatic phase II conjugation via sulfotransferases (producing the ~15 sulfated metabolites) and glycine N-acyltransferase GLYAT (producing hippurate, hydroxyhippurates, and cinnamoylglycine). [Inferred] This pipeline has been consistently associated with polyphenol-rich diets (fruits, vegetables, coffee, tea) and with favorable cardiometabolic profiles. [Model Knowledge]

#### 4.2 Bile Acid Processing as a Co-regulated Arm

The inclusion of ursodeoxycholate, isoursodeoxycholate, and glycoursodeoxycholate within a phenolic metabolite module is notable. [Inferred] Ursodeoxycholate is a secondary bile acid produced by 7-alpha/beta-epimerization of chenodeoxycholate by gut bacteria [Model Knowledge]; its co-expression with microbial phenolics suggests that the bacterial communities mediating bile acid biotransformation overlap with those fermenting polyphenols. [Inferred] The exclusive presence of unconjugated or minimally conjugated secondary bile acids (absence of taurine- and glycine-conjugated primary bile acids) indicates that this module specifically captures the microbiome-processed bile acid fraction. [KG Evidence, gap analysis]

#### 4.3 Xenobiotic and Environmental Chemical Signatures

Several module members indicate environmental or xenobiotic exposure: 4-hydroxychlorothalonil (a chlorothalonil fungicide metabolite), 3,5-dichloro-2,6-dihydroxybenzoic acid and 3-bromo-5-chloro-2,6-dihydroxybenzoic acid (halogenated benzoic acids likely of environmental origin). [Model Knowledge] Their co-expression with microbial phenolics suggests shared hepatic clearance pathways (sulfation, glucuronidation) or shared dietary/environmental exposure sources. [Inferred]

#### 4.4 Hub Filtering

4-Hydroxycoumarin (1,281 edges) and ursodeoxycholate (1,824 edges) are hub entities. [KG Evidence] Associations mediated exclusively through these entities (e.g., melanoma for 4-hydroxycoumarin, many rare disease associations for ursodeoxycholate as a pharmaceutical agent) should be de-emphasized. Disease associations that converge across multiple module members (colorectal cancer via hippurate and 4-hydroxycoumarin; T2D via ursodeoxycholate) carry greater credibility. [Inferred]

---

### 5. Gap Analysis

#### 5.1 Informative Absences

| Absent Entity | Interpretation | Evidence |
|---|---|---|
| p-Cresol sulfate | Distinct microbial tyrosine decarboxylation pathway; taxonomically separated from polyphenol fermentation | [KG Evidence, gap analysis] |
| Conjugated primary bile acids (glycochenodeoxycholate, taurocholate) | Module captures deconjugated, microbiome-processed bile acid fraction only; conjugated bile acids likely cluster in a hepatic/enterohepatic module | [KG Evidence, gap analysis] |
| Phenylacetylglutamine (PAGln) | Preferential glycine conjugation (GLYAT) over glutamine conjugation in this module; informative for conjugation enzyme phenotyping | [KG Evidence, gap analysis] |
| TMAO | Choline/carnitine pathway is mechanistically distinct from polyphenol fermentation; genuine pathway compartmentalization | [KG Evidence, gap analysis] |
| Indole-3-propionate (canonical form) | Mapping artifact: indolepropionate is present but resolved to incorrect PubChem identifier (Indosterol); likely partial coverage | [KG Evidence, gap analysis] |

#### 5.2 Standard Gaps (Expected and Non-informative)

Short-chain fatty acids, branched-chain amino acids, ceramides, insulin/C-peptide, and HbA1c are absent as expected: they belong to different metabolic pathways, analytical platforms, or data modalities and would not co-express with this metabolite class in WGCNA network construction. [KG Evidence, gap analysis]

#### 5.3 Cold-Start Entities

Six entities had zero KG edges: indolepropionate (mapping artifact), 3-(3-hydroxyphenyl)propionate sulfate, 1,2,3-benzenetriol sulfate, 3-methoxycatechol sulfate, 3-bromo-5-chloro-2,6-dihydroxybenzoic acid, and 4-allylcatechol sulfate. [KG Evidence] Under the Open World Assumption, these absences indicate insufficient curation in current knowledge graphs rather than biological irrelevance. Semantic similarity analysis placed several of these within the sulfated polyphenol class (1,2,3-benzenetriol sulfate similar to pyrogallol 1-sulfate at 0.85 [KG Evidence, inferred]), supporting their assignment to the microbial polyphenol fermentation theme despite absent direct KG evidence.

---

### 6. Temporal Context

No longitudinal time-point data were provided with this analysis. However, the module's composition suggests a causal ordering amenable to testing in longitudinal cohort studies [Inferred]:

- **Upstream causes**: Dietary polyphenol intake and gut microbiome composition (specifically, polyphenol-fermenting and bile acid-transforming taxa) are the likely upstream determinants of module member abundance.
- **Downstream consequences**: Elevated microbial phenolic production and secondary bile acid levels may influence insulin sensitivity (via FXR/TGR5 signaling [KG Evidence]), intestinal barrier function (via AHR activation [KG Evidence, hidden_gems; Model Knowledge]), and hepatic lipid metabolism (via HMGCR interaction [KG Evidence, hidden_gems]).
- **Causal inference opportunity**: Mendelian randomization using genetic variants in SLC10A1, ABCB11, NR1H4, or GLYAT as instruments could test whether the bile acid or glycine conjugation arm of this module causally influences T2D risk. [Inferred]

---

### 7. Research Recommendations

#### Priority 1: High-Value Experimental Validations

1. **Re-map indolepropionate**: The current entity resolution (70% confidence to Indosterol) is likely incorrect. Re-query the KG with CHEBI:43580 (indole-3-propionic acid) to recover KG relationships and confirm its module membership biochemistry.

2. **AHR reporter assay panel**: Test ursodeoxycholate, catechol sulfate, 4-ethylphenylsulfate, and indolepropionate for AHR agonist/antagonist activity individually and in combination, given the novel KG-identified ursodeoxycholate to AHR interaction. [KG Evidence, hidden_gems]

3. **Microbiome correlation analysis**: Correlate participant-level abundances of all module members with 16S/shotgun metagenomic profiles to identify the microbial taxa driving this co-expression pattern. Prioritize Clostridium, Eggerthella, Adlercreutzia, and Gordonibacter genera (known polyphenol fermenters). [Model Knowledge]

#### Priority 2: Literature and Database Searches

4. **L-urobilin and metabolic disease**: Conduct a targeted literature search for L-urobilin (and D-urobilin) associations with T2D, metabolic syndrome, and cardiovascular disease, building on the grounded literature indicating bilirubin-derived metabolite benefits in diabetes [Literature: "Metabolic Engineering of Escherichia coli for Production of a Bioactive Metabolite of Bilirubin," 2024].

5. **GLYAT phenotyping**: Review literature on GLYAT activity variation and its relationship to hippurate production; the exclusive presence of glycine conjugates (no glutamine conjugates) in this module may reflect GLYAT genotype or expression differences in the cohort. [Inferred]

#### Priority 3: Follow-up Computational Analyses

6. **Cross-module correlation**: Examine correlations between Grey60 members and (i) p-cresol sulfate, (ii) conjugated primary bile acids, and (iii) PAGln in other WGCNA modules to test the pathway compartmentalization hypotheses from the gap analysis.

7. **Dietary covariate analysis**: Test associations between module eigengene values and dietary intake variables (polyphenol-rich foods, fiber, coffee, tea) to confirm the dietary origin hypothesis.

8. **Mendelian randomization**: Use cis-pQTLs or cis-mQTLs for hippurate (GLYAT locus) and ursodeoxycholate (bile acid pathway loci) as genetic instruments to test causal effects on T2D conversion.

9. **Xenobiotic source investigation**: Determine whether 4-hydroxychlorothalonil and the halogenated benzoic acids (3,5-dichloro-2,6-dihydroxybenzoic acid; 3-bromo-5-chloro-2,6-dihydroxybenzoic acid) originate from pesticide exposure, water treatment byproducts, or dietary sources, as their co-expression with microbial metabolites may reflect shared environmental exposures in this cohort.

---

#### Member Prioritization Summary

The highest-leverage individual members for follow-up investigation, ranked by information content and biological novelty:

| Rank | Member | Rationale |
|---|---|---|
| 1 | Ursodeoxycholate | Hub entity with 1,824 KG edges; direct T2D and MASLD treatment associations; novel AHR and HMGCR interactions [KG Evidence] |
| 2 | Hippurate | Canonical GLYAT product; colorectal cancer association; excretion pathway validated [KG Evidence]; strongest standalone biomarker candidate |
| 3 | L-urobilin | Cold-start but literature-grounded link to metabolic disease; uniquely connects microbiome heme catabolism to this module [Inferred; Literature] |
| 4 | Indolepropionate | Mapping artifact requiring correction; if confirmed as IPA, provides the tryptophan-derived arm of microbial metabolism [Inferred] |
| 5 | Catechol sulfate / 4-ethylphenylsulfate | Representative sulfated phenolics with moderate KG coverage (9 and 13 edges); suitable for AHR testing [KG Evidence] |
| 6 | 4-Hydroxychlorothalonil | Xenobiotic sentinel; co-expression with microbial metabolites requires mechanistic explanation [Model Knowledge] |

---

*Report generated from KRAKEN knowledge graph analysis. All Tier 3 predictions are speculative and require experimental validation. Approximately 18% of computational predictions advance to clinical investigation (empirical calibration). Evidence attribution tags ([KG Evidence], [Literature], [Model Knowledge], [Inferred]) are provided throughout to distinguish data sources and epistemic confidence levels.*

### Literature References

Papers discovered via semantic search. 6 unique papers across 4 hypotheses.

| Hypothesis | Citation | Link | Relevance |
|------------|----------|------|-----------|
| Inferred role of PUBCHEM.COMPOUND:24721632 |  (2017) "4,5-dichloro-2-[6-hydroxy-2,7-dimethyl-3-oxo-5-(2-propen-1-yl)-3H-xanthen-9-yl]benzoic acid" | [Link](https://www.nature.com/articles/nchem.2729/compounds/4b) | 4,5-dichloro-2-[6-hydroxy-2,7-dimethyl-3-oxo-5-(2-propen-1-yl)-3H-xanthen-9-yl]benzoic acid View in PubChem\| 1H NMR\| 13... |
| Inferred role of CHEBI:232804 |  (2021) "Biosynthesis of ethyl caffeate via caffeoyl-CoA acyltransferase expression in Escherichia coli \| App..." | [Link](https://link.springer.com/article/10.1186/s13765-021-00643-0) | Hydroxycinnamic acids (HCs) are natural compounds that form conjugates with diverse compounds in nature. Ethyl caffeate... |
| Inferred role of CHEBI:141079 |  (2023) "Identification and quantification of branched-chain fatty acids and odd-chain fatty acids of mammali..." | [Link](https://www.sciencedirect.com/science/article/abs/pii/S0958694623000067) | Branched-chain fatty acids (BCFAs) are primarily saturated fatty acids with one or more methyl branches. Fatty acids (FA... |
| Inferred role of RM:0136260 |  (2020) "Mutation spectrum and biochemical features in infants with neonatal Dubin-Johnson syndrome \| BMC Ped..." | [Link](https://link.springer.com/article/10.1186/s12887-020-02260-0) | Dubin-Johnson syndrome (DJS) is an autosomal recessive disorder presenting as isolated direct hyperbilirubinemia.DJS is... |
| Inferred role of CHEBI:232804 |  (2022) "Sulfated Phenolic Substances: Preparation and Optimized HPLC Analysis" | [Link](https://www.mdpi.com/1422-0067/23/10/5743) | 2,3 ... due to the low flow rate and the absence of phosphate buffer and/or ion-pairing reagents ... Sulfation of 4-meth... |
| Inferred role of CHEBI:232804 |  (2017) "Table 1 Human bioavailable (poly)phenol metabolites. (Poly)phenol metabolites nomenclature, abbrevia..." | [Link](https://www.nature.com/articles/s41598-017-11512-6/tables/1) | Table 1 Human bioavailable (poly)phenol metabolites. (Poly)phenol metabolites nomenclature, abbreviation, chemical struc... |