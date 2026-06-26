# Grey60 Module Run: Discovery Output (32-analyte, dev, 2026-06-23)

This document presents the discovery output of a full-module run of the **Grey60** WGCNA module through the Kraken 12-node discovery pipeline, executed on the dev environment on 2026-06-23 (commit `813fc3f`) with BioMapper entity resolution enabled. It is one of a per-module series over the Frailty multi-omics WGCNA modules; see the [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm).

## Run Provenance

The run submitted 32 named analytes, parsed 33 at intake, and resolved 33 distinct entities (7 fuzzy, 25 biomapper, 1 exact) to 33 distinct CURIEs. Triage classified 3 well-characterized, 3 moderate, 21 sparse, and 6 cold-start, with 0 edge-count measurement failures. Downstream analysis produced 265 direct-KG findings, 27 cold-start findings, 3 biological themes, 0 cross-entity bridges (0 evidence-grounded), and 51 hypotheses supported by 19 literature references. Synthesis emitted a 25842-character report. The run completed in approximately 513.7 s of wall-clock time (status complete, 6 errors). All counts are taken from the reducer-merged pipeline state.

| Stage | Result |
|---|---|
| Submitted | 32 named analytes |
| Intake | 33 parsed |
| Entity resolution | 33 resolved (7 fuzzy, 25 biomapper, 1 exact) to 33 distinct CURIEs |
| Triage | 3 well-characterized, 3 moderate, 21 sparse, 6 cold-start (0 measurement failures) |
| Direct KG | 265 findings |
| Cold-start | 27 findings, 19 skipped |
| Pathway enrichment | 3 biological themes |
| Integration | 0 bridges (0 evidence-grounded) |
| Literature grounding | 19 papers |
| Synthesis | 51 hypotheses, 25842-character report |
| Run total | ~513.7 s wall-clock, status complete, 6 errors |

## Related

- Companion run metrics: [Grey60 Module Run: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/grey60-module-run-pipeline-performance-report-32-analyte-dev-2026-06-23-xpx5udQQOy)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---



## Grey60 WGCNA Module: A Gut Microbial Co-Metabolism Signature Spanning Phenylpropanoid Catabolism, Secondary Bile Acid Transformation, and Phase II Sulfate Conjugation

---

### 1. Executive Summary

The Grey60 module encodes a coherent gut microbiome-to-host co-metabolism signature dominated by three convergent axes: (i) microbial catabolism of dietary phenylpropanoids and aromatic amino acids, (ii) bacterial transformation of primary into secondary and tertiary bile acids, and (iii) hepatic Phase II sulfate conjugation of microbially derived phenolic substrates. [KG Evidence] [Inferred] The module's composition, notably its exclusive representation of secondary bile acids, deconjugated bile species, and sulfated (rather than glucuronidated) phenolics, points to a phenylalanine/tyrosine-dominant microbial metabolic axis with active bile salt hydrolase (BSH) and 7α/β-dehydroxylase activity, while its informative absences (no primary bile acids, no p-cresol sulfate, no indoxyl sulfate, no TMAO) distinguish it from a generic microbiome metabolite cluster and suggest cohort-specific or platform-specific selectivity. [KG Evidence] [Inferred] Ursodeoxycholate emerges as the highest-leverage member, bridging bile acid transport, hepatoprotection, and exploratory therapeutic associations with type 2 diabetes and metabolic dysfunction-associated steatotic liver disease. [KG Evidence]

---

### 2. Key Findings (Tier 1 to 2)

#### 2.1 Unifying Biological Theme: Gut Microbial Co-Metabolism

The module comprises 32 small-molecule metabolites and no proteins (the single "Metabolites:" entity resolved to a spurious UMLS protein entry at 70% confidence and should be disregarded). [KG Evidence] Twenty of the 32 metabolites are sulfate conjugates of phenolic or catechol scaffolds (catechol sulfate, hydroquinone sulfate, 4-ethylphenylsulfate, 4-vinylphenol sulfate, guaiacol sulfate, eugenol sulfate, 4-allylphenol sulfate, 4-allylcatechol sulfate, 4-ethylcatechol sulfate, 3-methoxycatechol sulfate, 1,2,3-benzenetriol sulfate, 4-acetylphenol sulfate, 3-(3-hydroxyphenyl)propionate sulfate, among others), establishing Phase II sulfation as the dominant conjugation pathway represented. [KG Evidence] [Model Knowledge] The remaining members comprise glycine conjugates of aromatic acids (hippurate, cinnamoylglycine, 3-hydroxyhippurate, 4-hydroxyhippurate), unconjugated phenylpropanoid catabolites (3-phenylpropionate, 3-(3-hydroxyphenyl)propionate, gentisate), bile acids and their conjugates (ursodeoxycholate, isoursodeoxycholate, glycoursodeoxycholate), a heme catabolite (L-urobilin), and minor species including a branched-chain dicarboxylic acid, 5-hydroxyhexanoate, methyl glucopyranoside, 4-hydroxycoumarin, and two halogenated benzoic acids likely of xenobiotic or environmental origin. [KG Evidence]

#### 2.2 Bile Acid Axis

Ursodeoxycholate (1,824 edges; well-characterized) is the most densely connected module member and anchors the bile acid sub-network. [KG Evidence] It interacts with bile acid transporters SLC10A1 (NTCP) and ABCB11 (BSEP), both of which were independently identified in the pathway enrichment analysis as shared connectors across module members. [KG Evidence] The KG records curated therapeutic associations between ursodeoxycholate and primary biliary cholangitis, cholestasis, intrahepatic cholestasis, primary sclerosing cholangitis, cholelithiasis, and autoimmune hepatitis, all via "treats_or_applied_or_studied_to_treat" predicates. [KG Evidence] Isoursodeoxycholate (353 edges) shares a top disease association with primary biliary cholangitis, reinforcing the hepatobiliary disease relevance of this sub-module. [KG Evidence] Glycoursodeoxycholate (10 edges) represents the glycine-conjugated form and bridges the bile acid and glycine conjugation sub-networks. [KG Evidence]

Ursodeoxycholate also participates in bile acid secretion, decreased cholesterol absorption, cholesterol biosynthetic process, and hepatocyte apoptotic process. [KG Evidence] Novel (hidden-gem) disease associations from the KG include type 2 diabetes mellitus (treats_or_applied_or_studied_to_treat), obesity disorder, metabolic dysfunction-associated steatotic liver disease, kidney disorder, and severe acute respiratory syndrome. [KG Evidence]

#### 2.3 Phenylpropanoid and Aromatic Amino Acid Catabolism Axis

Hippurate (141 edges) participates in excretion pathways and is associated with phenylketonuria, reflecting its origin from hepatic glycine conjugation of benzoate, itself a product of gut microbial catabolism of dietary polyphenols and phenylalanine. [KG Evidence] Gentisate (93 edges) participates in tyrosine metabolism and is associated with cardiomyopathy and colorectal cancer (the latter shared with hippurate at the module level). [KG Evidence] 3-(3-Hydroxyphenyl)propionate (26 edges) has dedicated catabolic, metabolic, and biosynthetic process annotations in the KG, placing it squarely in the microbial transformation of caffeic acid and chlorogenic acid. [KG Evidence]

#### 2.4 Module-Level Disease Recurrence

Two diseases recur across two or more module members with curated evidence: acute myeloid leukemia (4-hydroxycoumarin, ursodeoxycholate) and liver cancer (4-hydroxycoumarin, ursodeoxycholate). [KG Evidence] Colorectal cancer is shared between gentisate and hippurate with curated evidence. [KG Evidence] Cancer (broad) and glioblastoma are shared between gentisate and 4-hydroxycoumarin via text-mined evidence. [KG Evidence] The ursodeoxycholate and 4-hydroxycoumarin disease associations should be interpreted cautiously: both are well-characterized compounds (1,824 and 1,281 edges, respectively), and their cancer associations may partly reflect hub-driven connectivity rather than module-specific biology.

#### 2.5 Molecular Interaction Highlights (Tier 2)

Ursodeoxycholate interacts with the nuclear bile acid receptor FXR (NR1H4), the glucocorticoid receptor (NR3C1), and the membrane bile acid receptor GPBAR1/TGR5, consistent with its role as a signaling bile acid. [KG Evidence] Novel interaction targets include the aryl hydrocarbon receptor AHR, the pregnane X receptor NR1I2, the constitutive androstane receptor co-factor NR0B2 (SHP), and the transcription factor HNF1A, all of which are master regulators of xenobiotic and bile acid metabolism gene programs. [KG Evidence] 4-Hydroxycoumarin interacts with multiple carbonic anhydrase isoforms (CA1, CA2, CA9, CA12) and VKORC1L1, consistent with its pharmacological class as a coumarin derivative. [KG Evidence]

---

### 3. Novel Predictions (Tier 3)

All Tier 3 predictions derive from semantic similarity inference for cold-start or sparse entities. Approximately 18% of computational predictions of this nature progress to clinical or experimental investigation; accordingly, each prediction below should be regarded as a hypothesis-generating lead, not an established finding.

#### 3.1 L-Urobilin: Inferred Placement in Heme Degradation and Bilirubin Metabolism

**Prediction.** L-urobilin (RM:0136260; 2 edges) is inferred to participate in heme degradation and bilirubin metabolism pathways (SMPDB:SMP0000346, SMP0000344, SMP0000342, SMP0000024) and to correlate with Dubin-Johnson syndrome (MONDO:0009960). [Inferred]

**Structural logic chain.** L-urobilin is semantically similar to urobilin (CHEBI:36378; similarity 0.86) and D-urobilinogen (CHEBI:4260; similarity 0.82). Both analogues participate in SMPDB bilirubin metabolism pathways, and both correlate with Dubin-Johnson syndrome in the KG. L-urobilin, as the oxidized product of urobilinogen and a terminal metabolite in the heme→biliverdin→bilirubin→urobilinogen→urobilin cascade, is biochemically expected in these pathways. [Inferred] [KG Evidence]

**Literature support.** Grounded literature confirms that urobilin is derived from bilirubin bioconversion via gut microbiota and binds albumin, potentially interfering with bilirubin-albumin interactions (Urobilin Derived from Bilirubin Bioconversion, 2025). [Literature] A second grounded study reports elevated urinary urobilinogen in acute hepatic porphyria patients (A high urinary urobilinogen/serum total bilirubin ratio, 2023), supporting the clinical relevance of urobilinoid measurements. [Literature] Open-chain tetrapyrrole classification of L-urobilin is supported by spectroscopic characterization literature (Absorption and fluorescence spectra of open-chain tetrapyrrole pigments, 2023). [Literature] A study on neonatal Dubin-Johnson syndrome confirms the disorder involves direct hyperbilirubinemia and ATP-binding cassette transporter mutations (Mutation spectrum and biochemical features in infants with neonatal Dubin-Johnson syndrome, 2020), which is mechanistically upstream of urobilin production but does not directly measure L-urobilin levels. [Literature]

**Validation step.** Measure L-urobilin concentrations in urine and feces of Dubin-Johnson syndrome patients and matched controls. Query SMPDB pathway entries to confirm whether L-urobilin is explicitly annotated. Calibration note: ~18% of such inferences advance to clinical investigation.

#### 3.2 Branched-Chain 14:0 Dicarboxylic Acid: Inferred Classification Under Methyl-Branched Fatty Acids

**Prediction.** Branched-chain 14:0 dicarboxylic acid (CHEBI:141079; 1 edge) is inferred to be a subclass of methyl-branched fatty acid (CHEBI:62499). [Inferred]

**Structural logic chain.** Three analogues (methyl-branched fatty acid 22:0, 15:0, and 21:0; similarity 0.94 each) are all classified as subclass_of CHEBI:62499 in the KG. The target compound shares a branched-chain structural motif but differs in possessing two carboxylate termini (dicarboxylic acid). [Inferred] [KG Evidence]

**Literature support.** Grounded literature on branched-chain fatty acid identification in mammalian milk and dairy products (Identification and quantification of branched-chain fatty acids, 2023) confirms that BCFAs are primarily saturated fatty acids with methyl branches, but does not specifically address dicarboxylic acid variants. [Literature]

**Validation step.** Verify whether CHEBI:62499 (methyl-branched fatty acid) ontology definition admits dicarboxylic acid members; if not, the correct parent class may be a branched-chain dicarboxylic acid superclass. Calibration note: ~18% of such inferences advance to experimental validation.

#### 3.3 1,2,3-Benzenetriol Sulfate: Inferred Urinary Localization and Sulfate Ester Classification

**Prediction.** 1,2,3-Benzenetriol sulfate (RM:0186652; cold-start) is inferred to be a sulfate ester (subclass_of CHEBI:38757) detected in urine (located_in UBERON:0001088). [Inferred]

**Structural logic chain.** The closest analogue is pyrogallol 1-sulfate (similarity 0.85), which is classified under sulfate ester and localized to urine in the KG. Since 1,2,3-benzenetriol and pyrogallol are identical compounds (pyrogallol is the common name for 1,2,3-benzenetriol), the sulfate conjugate of this scaffold is expected to share both chemical classification and biofluid localization. [Inferred] [KG Evidence]

**Literature support.** Grounded polyphenol metabolite data reports pyrogallol-O-sulfate at Cmax of 11.4 ± 6.7 µM in human bioavailability studies (Table 1, Human bioavailable (poly)phenol metabolites, 2017), confirming that sulfate conjugates of this scaffold reach quantifiable plasma concentrations. [Literature]

**Validation step.** Search HMDB and MetaboLights for 1,2,3-benzenetriol sulfate detection in urine samples. Calibration note: ~18% validation rate applies.

#### 3.4 Indolepropionate: Inferred Indole Class Membership

**Prediction.** Indolepropionate (PUBCHEM.COMPOUND:157009863; cold-start, 0 edges) is inferred to belong to the indole chemical class (CHEBI:24828/24829). [Inferred]

**Structural logic chain.** Indolepropionate is semantically similar to the NCIT Indole Compound class (similarity 0.77) and to indolinone (CHEBI:51625; similarity 0.77), the latter being a subclass of CHEBI:24829 (indoles). As an indole-ring-containing propionate, indolepropionate is expected to share this classification. [Inferred] [KG Evidence]

**Literature support.** No grounded abstracts were fetched for this prediction. The classification is based on structural logic and KG analogy. [Model Knowledge]

**Validation step.** Confirm the canonical ChEBI or HMDB entry for indole-3-propionic acid (IPA; CHEBI:43580 or equivalent) and verify its ontology placement. Note that entity resolution mapped "indolepropionate" to PUBCHEM.COMPOUND:157009863 (Indosterol) at only 70% confidence; the intended analyte is almost certainly indole-3-propionic acid, a well-characterized gut microbial tryptophan catabolite. This resolution artifact should be corrected before downstream analysis. Calibration note: ~18% validation rate applies.

---

### 4. Biological Themes

#### 4.1 Theme 1: Microbial Phenylpropanoid and Polyphenol Catabolism

The largest cluster within the module comprises metabolites arising from gut bacterial degradation of dietary polyphenols and aromatic amino acids (phenylalanine, tyrosine). [Inferred] [Model Knowledge] The core pathway proceeds from complex dietary polyphenols through microbial ring fission and side-chain shortening to yield simple phenolic acids (3-phenylpropionate, 3-(3-hydroxyphenyl)propionate), which are subsequently conjugated in the liver via glycine (yielding hippurate, cinnamoylglycine, 3-hydroxyhippurate, 4-hydroxyhippurate) or sulfation (yielding catechol sulfate, hydroquinone sulfate, 4-ethylphenylsulfate, 4-vinylphenol sulfate, guaiacol sulfate, eugenol sulfate, and the allyl- and ethyl-catechol sulfates). [Model Knowledge] [KG Evidence] Gentisate (2,5-dihydroxybenzoate) connects this theme to tyrosine metabolism via direct KG annotation. [KG Evidence]

#### 4.2 Theme 2: Secondary Bile Acid Metabolism and Enterohepatic Cycling

Ursodeoxycholate, isoursodeoxycholate, and glycoursodeoxycholate form a bile acid sub-cluster exclusively comprising secondary and tertiary bile acid species. [KG Evidence] The pathway enrichment analysis identified SLC10A1 (NTCP) and ABCB11 (BSEP) as shared gene connectors, both of which are hepatic bile acid transporters central to enterohepatic circulation. [KG Evidence] The exclusive presence of secondary bile acids without their primary precursors (cholate, chenodeoxycholate) is a highly informative structural feature indicating that this module captures post-microbial transformation products. [KG Evidence] [Inferred]

#### 4.3 Theme 3: Phase II Sulfate Conjugation Dominance

The module contains at least 20 sulfate conjugates and zero glucuronide conjugates. [KG Evidence] This asymmetry may reflect: (a) genuine cohort-level predominance of sulfotransferase (SULT) over UDP-glucuronosyltransferase (UGT) activity; (b) tighter co-variance among sulfated species due to shared SULT enzyme kinetics; or (c) analytical platform bias favoring sulfate detection. [Inferred] [Model Knowledge] If biologically authentic, this pattern identifies sulfation capacity as a potential stratifying variable for this cohort.

#### 4.4 Theme 4: Heme Catabolism (Minor)

L-urobilin, the terminal oxidation product of bilirubin degradation by gut bacteria, represents a minor but distinct metabolic axis. [Model Knowledge] Its presence without upstream bilirubin reinforces the module's gut-microbial processing orientation. [Inferred]

#### 4.5 Hub Filtering Note

4-Hydroxycoumarin (1,281 edges) and ursodeoxycholate (1,824 edges) are both well-characterized hub entities. [KG Evidence] Disease associations unique to these hubs (e.g., melanoma for 4-hydroxycoumarin, retinoblastoma and sarcoma for ursodeoxycholate) should be interpreted with caution, as they may reflect the compounds' broad pharmacological profiles rather than module-specific biology. The colorectal cancer recurrence (shared between gentisate and hippurate, both moderate-coverage entities with 93 and 141 edges, respectively) is more likely to reflect genuine module biology. [KG Evidence] [Inferred]

---

### 5. Gap Analysis

#### 5.1 Highly Informative Absences

**Primary bile acids (cholate, chenodeoxycholate).** The exclusive presence of secondary/tertiary bile acids without their hepatically synthesized precursors strongly indicates that this module captures post-deconjugation, post-dehydroxylation metabolites processed by gut bacteria with BSH and 7α-dehydroxylase activity. [Inferred] [Model Knowledge]

**Glycine-conjugated bile acids (glycocholate, glycochenodeoxycholate).** Their absence, combined with the presence of unconjugated secondary bile acids, indicates the module reflects metabolites that have undergone bacterial deconjugation. [Inferred]

**p-Cresol sulfate.** This abundant tyrosine-derived microbial sulfate conjugate does not co-cluster with the module's other phenolic sulfates, suggesting distinct temporal or inter-individual variance patterns that place it in a different WGCNA eigengene trajectory. [Inferred]

**Indole/tryptophan-derived metabolites (indoxyl sulfate).** The module captures a phenylalanine/tyrosine-dominant axis rather than a tryptophan-dominant one, potentially reflecting microbial community composition or substrate availability. [Inferred]

#### 5.2 Moderately Informative Absences

**Glucuronide conjugates.** The exclusive sulfation pattern warrants investigation into whether this reflects genuine SULT/UGT balance or analytical artifact. [Inferred]

**TMAO.** Its absence confirms the module is pathway-specific (phenylpropanoid and bile acid pathways) rather than a generic microbial metabolite cluster. [Inferred]

**Bilirubin.** The presence of L-urobilin without bilirubin reflects distinct tissue-compartment kinetics (hepatic bilirubin versus gut-derived urobilin). [Inferred]

#### 5.3 Non-Informative Absences

Short-chain fatty acids (butyrate, propionate, acetate) are likely absent due to analytical platform limitations (volatile compounds requiring GC-MS). [Model Knowledge] Branched-chain amino acids and equol operate through distinct metabolic and dietary axes and would not be expected to co-cluster. [Inferred]

---

### 6. Temporal Context

No explicit longitudinal metadata was provided with this module. However, the module's composition permits temporal inference. The upstream causes of the observed metabolite signature include: (i) dietary polyphenol and aromatic amino acid intake; (ii) gut microbial community composition, particularly taxa with BSH, 7α-dehydroxylase, and phenylpropanoid reductase activity (e.g., Clostridium scindens, Eggerthella lenta, Clostridiales); and (iii) hepatic Phase II sulfotransferase capacity. [Model Knowledge] [Inferred] The downstream consequences detectable in this module include: (i) circulating levels of sulfated phenolics and secondary bile acids, which may influence systemic inflammation, FXR/TGR5 signaling, and cholesterol metabolism via ursodeoxycholate's documented pathways; and (ii) urinary excretion of hippurate, hydroxyhippurates, and phenolic sulfates. [KG Evidence] [Inferred]

If the study design includes time-series data, this module's eigengene trajectory could be tested for Granger-causal relationships with clinical endpoints (e.g., hepatic function markers, insulin sensitivity) to distinguish whether changes in microbial co-metabolism precede or follow metabolic deterioration.

---

### 7. Research Recommendations

#### Priority 1: Entity Resolution Correction

1. **Indolepropionate resolution.** The mapping of "indolepropionate" to PUBCHEM.COMPOUND:157009863 (Indosterol) at 70% confidence is almost certainly incorrect. The intended analyte is indole-3-propionic acid (IPA), a well-characterized gut microbial tryptophan metabolite with neuroprotective and anti-inflammatory properties. Re-resolution to the correct ChEBI/HMDB identifier (e.g., CHEBI:43580) would likely yield substantial KG connectivity and disease associations.

2. **"Metabolites:" spurious entity.** The header string "Metabolites:" resolved to UMLS:C1698199 (substance P-metabolite 5-11) at 70% confidence. This entry should be excluded from downstream analyses.

#### Priority 2: Experimental Validations

3. **Sulfation versus glucuronidation balance.** Measure both sulfate and glucuronide conjugates of shared phenolic substrates (e.g., catechol, hydroquinone, 4-ethylphenol) in cohort samples to determine whether the module's sulfate exclusivity reflects genuine metabolic phenotype or platform detection bias.

4. **p-Cresol sulfate and indoxyl sulfate co-variance.** Examine the WGCNA module assignments for p-cresol sulfate and indoxyl sulfate. If they cluster in a separate module, compare eigengene trajectories with Grey60 to assess whether they reflect distinct microbial community dynamics or merely distinct variance structures.

5. **L-urobilin in Dubin-Johnson syndrome.** Quantify L-urobilin in urine of Dubin-Johnson syndrome patients as suggested by the Tier 3 inference; the grounded literature on urobilinogen and bilirubin pathway enzymes supports biological plausibility but direct measurement data for L-urobilin specifically are lacking.

#### Priority 3: Pathway and Disease Follow-Up

6. **Ursodeoxycholate and type 2 diabetes.** The KG records an exploratory therapeutic association (treats_or_applied_or_studied_to_treat) between ursodeoxycholate and T2D (MONDO:0005148). [KG Evidence] Conduct a targeted literature review of clinical trials investigating UDCA for glycemic control and bile acid-mediated FXR/TGR5 signaling in insulin sensitivity.

7. **Colorectal cancer association (hippurate and gentisate).** This module-level disease recurrence (shared by two non-hub members with curated evidence) merits investigation via case-control metabolomics comparing Grey60 module metabolites in colorectal cancer patients versus healthy controls. [KG Evidence]

8. **Halogenated benzoic acids.** 3,5-Dichloro-2,6-dihydroxybenzoic acid, 3-bromo-5-chloro-2,6-dihydroxybenzoic acid, and 4-hydroxychlorothalonil are likely of xenobiotic or environmental origin (chlorothalonil is a fungicide). [Model Knowledge] Investigate whether their co-clustering with microbial metabolites reflects shared renal or hepatic clearance kinetics or a common environmental exposure source in the study cohort.

#### Priority 4: Computational Follow-Up

9. **Cold-start entity enrichment.** Six module members (indolepropionate [if correctly resolved], 3-(3-hydroxyphenyl)propionate sulfate, 1,2,3-benzenetriol sulfate, 3-methoxycatechol sulfate, 3-bromo-5-chloro-2,6-dihydroxybenzoic acid, 4-allylcatechol sulfate) have zero KG edges. Targeted curation of these entities in ChEBI/HMDB with cross-references to SMPDB pathways would substantially improve future module-level analyses.

10. **Microbial taxa deconvolution.** If paired 16S rRNA or metagenomic sequencing data are available, correlate Grey60 eigengene values with relative abundance of taxa known to perform the enzymatic transformations encoded by this module (BSH-positive taxa, phenylpropanoid-metabolizing Clostridiales, bilirubin-reducing Clostridioides).

---

*Report generated from KRAKEN knowledge graph analysis. All [KG Evidence] claims derive from direct Kestrel query results. [Literature] claims are supported by grounded abstracts cited in situ. [Model Knowledge] claims draw on general biomedical knowledge not backed by KG or grounded literature evidence in this analysis. [Inferred] claims combine multiple evidence sources. Tier 3 predictions carry an ~18% historical validation rate and require independent experimental confirmation.*

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