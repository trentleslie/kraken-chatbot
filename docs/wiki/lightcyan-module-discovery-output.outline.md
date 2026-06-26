# Lightcyan Module Run: Discovery Output (27-analyte, dev, 2026-06-23)

This document presents the discovery output of a full-module run of the **Lightcyan** WGCNA module through the Kraken 12-node discovery pipeline, executed on the dev environment on 2026-06-23 (commit `813fc3f`) with BioMapper entity resolution enabled. It is one of a per-module series over the Frailty multi-omics WGCNA modules; see the [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm).

## Run Provenance

The run submitted 27 named analytes, parsed 27 at intake, and resolved 27 distinct entities (11 biomapper, 16 fuzzy) to 24 distinct CURIEs. Triage classified 5 well-characterized, 12 moderate, 8 sparse, and 2 cold-start, with 0 edge-count measurement failures. Downstream analysis produced 756 direct-KG findings, 27 cold-start findings, 6 biological themes, 10 cross-entity bridges (2 evidence-grounded), and 61 hypotheses supported by 35 literature references. Synthesis emitted a 24344-character report. The run completed in approximately 638.6 s of wall-clock time (status complete, 26 errors). All counts are taken from the reducer-merged pipeline state.

| Stage | Result |
|---|---|
| Submitted | 27 named analytes |
| Intake | 27 parsed |
| Entity resolution | 27 resolved (11 biomapper, 16 fuzzy) to 24 distinct CURIEs |
| Triage | 5 well-characterized, 12 moderate, 8 sparse, 2 cold-start (0 measurement failures) |
| Direct KG | 756 findings |
| Cold-start | 27 findings, 3 skipped |
| Pathway enrichment | 6 biological themes |
| Integration | 10 bridges (2 evidence-grounded) |
| Literature grounding | 35 papers |
| Synthesis | 61 hypotheses, 24344-character report |
| Run total | ~638.6 s wall-clock, status complete, 26 errors |

## Related

- Companion run metrics: [Lightcyan Module Run: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/lightcyan-module-run-pipeline-performance-report-27-analyte-dev-2026-06-23-nakWIM0Nkc)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---



## Lightcyan WGCNA Module: An Omega-3 PUFA and Immune Signaling Axis with Protective Cardiometabolic Character

---

### 1. Executive Summary

The Lightcyan WGCNA module encodes a coordinated biological signature in which omega-3 polyunsaturated fatty acid (PUFA) enriched phospholipids, plasmalogens, and acylcarnitines co-express with four immune and extracellular matrix proteins (IL4R, THBS2, FABP6, CCL23) and a suite of antioxidant metabolites (ergothioneine, ascorbic acid 3-sulfate, threonate). [KG Evidence; Inferred] This convergence indicates that the module captures a protective, anti-inflammatory lipid and immune remodeling program rather than a metabolic dysfunction axis, as evidenced by the systematic exclusion of pro-inflammatory omega-6 species, ceramides, branched-chain amino acids, and short-chain acylcarnitines. [KG Evidence; Inferred] The module's disease recurrence profile implicates shared vulnerability across allergic/atopic, gastrointestinal, and cardiovascular conditions, with psoriasis (5 of 4 protein members plus phosphatidylcholine) and asthma (4 members) representing the strongest convergences. [KG Evidence]

---

### 2. Key Findings (Tier 1 to 2)

#### 2.1 Module-Level Disease Convergence

The four protein members and phosphatidylcholine converge on a distinctive spectrum of disease associations. [KG Evidence]

| Disease | Members Sharing Association | Strongest Evidence |
|---|---|---|
| Psoriasis | 5 (all 4 genes + phosphatidylcholine) | Curated |
| Asthma | 4 (all 4 genes) | Curated |
| Coronary artery disorder | 4 (all 4 genes) | Curated |
| Essential hypertension | 4 (all 4 genes) | Curated |
| Depressive disorder | 4 (all 4 genes) | Curated |
| Irritable bowel syndrome | 4 (all 4 genes) | Curated |
| Urothelial carcinoma | 4 (all 4 genes) | Curated |
| Diabetes mellitus | 3 (FABP6, IL4R, THBS2 + phosphatidylcholine) | Curated |
| Atopic eczema | 2 (IL4R, CCL23) | Curated |
| Metabolic dysfunction-associated steatotic liver disease | 2 (threonate, phosphatidylcholine) | Text-mined |

The coexistence of allergic/atopic conditions (psoriasis, asthma, atopic eczema, vasomotor and allergic rhinitis) with cardiovascular and metabolic diseases (coronary artery disorder, essential hypertension, diabetes mellitus) is consistent with the module's dual immune and lipid character. [KG Evidence; Inferred] Psoriasis exhibited the broadest cross-member recurrence (5 members), indicating that this module may capture a systemic inflammatory state in which Th2 immune polarization (driven by IL4R) intersects with lipid membrane remodeling. [KG Evidence; Model Knowledge]

#### 2.2 Pathway Architecture

The module's protein members converge on the following validated biological processes. [KG Evidence]

**Immune and cytokine signaling.** IL4R participates in IL-4/IL-13 signaling, Th1/Th2 differentiation, cytokine receptor interaction, and positive regulation of immunoglobulin production. CCL23 participates in immune response, cell-cell signaling, and signal transduction. Both IL4R and CCL23 participate in the cytokine-cytokine receptor interaction pathway (WikiPathways WP5473). [KG Evidence]

**Extracellular matrix and adhesion.** THBS2 participates in extracellular matrix structural constitution, cell adhesion, focal adhesion, negative regulation of angiogenesis, and heparin binding. THBS2 and IL4R share membership in the PI3K-Akt signaling pathway (WikiPathways WP4172) and the focal adhesion PI3K-Akt-mTOR signaling pathway (WikiPathways WP3932). [KG Evidence]

**Stress response and proliferation control.** All four protein members participate in the response to stress (GO:0006950). FABP6 and CCL23 share negative regulation of cell population proliferation (GO:0008285). [KG Evidence]

**Smoking exposure.** All four protein members are annotated to smoking (UMLS:C0037369), indicating that this module's expression may be modulated by tobacco exposure or that its constituent genes are responsive to smoking-induced inflammation. [KG Evidence]

#### 2.3 Protein Member Highlights

**IL4R** (2,735 edges) is the most extensively connected member. Its established interactions include IL2RG, IL4, IL13, JAK1, JAK3, STAT6, IRS1, and IRS2, confirming canonical IL-4/IL-13 signaling through the type I and type II receptor complexes. [KG Evidence] The interaction with IRS1 and IRS2 is noteworthy: it links IL4R to insulin signaling, providing a mechanistic bridge between immune activation and metabolic regulation that is consistent with the module's co-expression of metabolic lipid species. [KG Evidence; Model Knowledge]

**THBS2** (2,570 edges) encodes an extracellular matrix glycoprotein with validated roles in angiogenesis suppression, synapse assembly, and ECM organization. Its interactions with integrins (ITGA1, ITGA8, ITGA9, ITGB5), CD47, and collagen (COL4A4) indicate active ECM remodeling within this module's biology. [KG Evidence]

**FABP6** (1,468 edges) is the ileal bile acid binding protein. Its top disease association is hypophysitis, but its expected primary ligands (bile acids) are absent from the module (see Gap Analysis, Section 5). [KG Evidence; Inferred] This absence suggests that FABP6's role in this module relates to its capacity as a lipid-binding protein that may interact with the omega-3 fatty acid species dominating the metabolite component.

**CCL23** (1,425 edges) is a CC-chemokine whose top disease association is inflammatory response. Its co-expression with IL4R is consistent with a Th2-polarized or alternatively activated macrophage milieu. [KG Evidence; Model Knowledge]

#### 2.4 Metabolite Landscape

The metabolite composition is overwhelmingly dominated by DHA-containing (22:6) and EPA-containing (20:5) phospholipids. Of 23 metabolites in the module, at least 17 carry a DHA or EPA acyl chain esterified to glycerophosphocholine (GPC) or glycerophosphoethanolamine (GPE) backbones. [KG Evidence] Several of these species are plasmalogens (ether-linked phospholipids denoted by "P-16:0" or "P-18:0" prefixes), which function as endogenous antioxidants by scavenging reactive oxygen species at vinyl-ether bonds. [Model Knowledge]

The remaining metabolites include:
- **Phosphatidylcholine** (CHEBI:64482; 410 edges): associated with hypothyroidism, diabetes mellitus, and metabolic dysfunction-associated steatotic liver disease. [KG Evidence]
- **Ergothioneine** (CHEBI:4828; 94 edges): a diet-derived antioxidant amino acid associated with learning disability in the KG. [KG Evidence]
- **Threonate** (CHEBI:49059; 103 edges): a vitamin C metabolite associated with anxiety disorder. [KG Evidence]
- **CMPF and related furanoid fatty acids** (CHEBI:82986, CHEBI:194524): fish-oil-derived uremic toxins; CMPF is associated with macular dystrophy. [KG Evidence]
- **Docosahexaenoylcarnitine** (PUBCHEM.COMPOUND:127055; 2 edges): a DHA-conjugated acylcarnitine. [KG Evidence]
- **Ascorbic acid 3-sulfate** (CHEBI:176456; 2 edges): a sulfated vitamin C conjugate. [KG Evidence]

---

### 3. Novel Predictions (Tier 3)

#### 3.1 Docosahexaenoylcarnitine as a Cardiovascular Disease Biomarker

**Logic chain:** Docosahexaenoylcarnitine (PUBCHEM.COMPOUND:127055) shares 0.74 structural similarity with methylthiopropionylcarnitine, which is associated with cardiovascular diseases (UMLS:C0007258) in the KG. Docosahexaenoylcarnitine is a DHA-derived long-chain acylcarnitine generated during fatty acid beta-oxidation of DHA. [KG Evidence; Inferred]

**Literature support:** Liepinsh et al. (2025) demonstrated that EPA- and DHA-derived acylcarnitines (EPAC, DHAC) are significantly less cardiotoxic than saturated and monounsaturated long-chain acylcarnitines; DHAC did not impair heart functionality, mitochondrial OXPHOS, or Akt phosphorylation at concentrations that were toxic for palmitoylcarnitine and elaidoylcarnitine. [Literature: Liepinsh et al. 2025] Leung et al. (2020) reported that docosahexaenoylcarnitine was among the metabolites most strongly correlated with serum 25(OH)D levels, linking it to vitamin D status and cardiometabolic health. [Literature: Leung et al. 2020] An LC-MS/MS method paper (2025) confirmed that PUFA-derived acylcarnitines including docosahexaenoylcarnitine are quantifiable in human plasma and represent potential biomarkers of omega-3 status. [Literature: LC-MS/MS acylcarnitines 2025]

**Calibration note:** Approximately 18% of computational predictions of this type progress to clinical investigation. The convergence of structural analogy with direct experimental evidence from Liepinsh et al. raises the prior for this particular prediction.

**Validation step:** Measure docosahexaenoylcarnitine in case-control cardiovascular cohorts, stratified by omega-3 supplementation status, to determine whether it serves as a protective biomarker inversely correlated with cardiovascular events.

#### 3.2 FABP6 as an Omega-3 PUFA Carrier Beyond Classical Bile Acid Transport

**Logic chain:** FABP6 is classically the ileal bile acid binding protein, yet bile acids are absent from this module. FABP6 co-expresses with 17+ DHA/EPA-containing lipid species. Fatty acid binding proteins possess hydrophobic binding pockets capable of accommodating diverse lipophilic ligands. FABP6 participates in negative regulation of cell population proliferation (shared with CCL23) and response to stress (shared with all four proteins). [KG Evidence; Inferred; Model Knowledge]

**Calibration note:** Approximately 18% of computational predictions of this type progress to clinical investigation. No direct literature evidence was retrieved for FABP6-omega-3 interaction.

**Validation step:** Perform competitive binding assays (isothermal titration calorimetry or fluorescent displacement) of DHA, EPA, and their lysophospholipid conjugates against recombinant FABP6 to determine whether omega-3 PUFAs are bona fide FABP6 ligands.

#### 3.3 IL4R Signaling as a Driver of Omega-3 Phospholipid Membrane Remodeling via Alternative Macrophage Activation

**Logic chain:** IL4R activates STAT6 to promote alternative (M2) macrophage polarization. M2 macrophages are known to exhibit enhanced fatty acid oxidation and lipid uptake. IL4R interacts with IRS1 and IRS2, linking it to insulin signaling and metabolic regulation. The module's omega-3 phospholipid enrichment may reflect IL4R-driven macrophage membrane remodeling toward anti-inflammatory DHA/EPA incorporation. The co-expression of CCL23 (a monocyte/macrophage chemokine) supports macrophage involvement. [KG Evidence; Model Knowledge; Inferred]

**Calibration note:** Approximately 18% of computational predictions of this type progress to clinical investigation. The absence of IL-4 and IL-13 ligands (see Section 5) and STAT6 from the module limits confidence in classical Th2 signaling and raises the possibility of a non-canonical metabolic role.

**Validation step:** Perform lipidomic profiling of IL-4-stimulated versus unstimulated macrophages to determine whether IL4R activation preferentially incorporates DHA and EPA into membrane phospholipids and plasmalogens.

#### 3.4 CMPF and Hydroxy-CMPF as Fish Oil-Derived Beta-Cell Modulators

**Logic chain:** CMPF (3-carboxy-4-methyl-5-propyl-2-furanpropanoate) is a furanoid fatty acid metabolite produced from omega-3 PUFA oxidation. It is a known uremic toxin and has been implicated as a beta-cell toxin. Its presence alongside 3-CMPFP and hydroxy-CMPF in an omega-3-enriched module is consistent with the metabolic origin of these compounds from the DHA/EPA precursors that dominate this module. [KG Evidence; Model Knowledge; Inferred]

**Calibration note:** Approximately 18% of computational predictions of this type progress to clinical investigation.

**Validation step:** Correlate circulating CMPF and hydroxy-CMPF concentrations with omega-3 PUFA intake and beta-cell function markers (HOMA-B, C-peptide) in a longitudinal cohort to distinguish protective fish-oil exposure from harmful accumulation.

---

### 4. Biological Themes

#### 4.1 Unifying Theme: Anti-Inflammatory Lipid Membrane Remodeling Coupled to Type 2 Immune Polarization

The module is unified by the convergence of two biological programs.

**Omega-3 PUFA membrane enrichment.** The metabolite component encodes a coordinated pattern of DHA and EPA incorporation into diverse phospholipid classes: diacyl phosphatidylcholines (PC 16:0/22:6, PC 18:0/22:6, PC 18:1/22:6, PC 16:0/20:5), lysophosphatidylcholines (LPC 22:6), lysophosphatidylethanolamines (LPE 22:6, LPE 20:5), plasmalogen phosphatidylethanolamines (PE P-16:0/22:6, PE P-18:0/22:6, PE P-18:1/22:6), and plasmalogen phosphatidylcholines (PC P-16:0/22:6, PC P-18:0/22:6). [KG Evidence; Model Knowledge] The plasmalogen species are particularly notable: they serve dual functions as membrane structural components and sacrificial antioxidants. [Model Knowledge]

**Immune modulation.** IL4R and CCL23 provide the immune signaling axis, with IL4R driving Th2 polarization and alternative macrophage activation, and CCL23 functioning as a monocyte-attracting chemokine. THBS2 contributes ECM remodeling and anti-angiogenic activity. [KG Evidence]

#### 4.2 Antioxidant Defense Network

Three module members contribute to redox homeostasis: ergothioneine (a histidine-derived thiol antioxidant concentrated in mitochondria), ascorbic acid 3-sulfate (a sulfated vitamin C conjugate), and threonate (a vitamin C catabolite). [KG Evidence; Model Knowledge] The co-expression of these antioxidants with plasmalogen species reinforces the module's protective character; plasmalogens themselves act as antioxidants through preferential oxidation at their vinyl-ether bond. [Model Knowledge]

#### 4.3 Hub-Filtered Considerations

Homo sapiens (NCBITaxon:9606) connects 4 input entities but is flagged as a hub node with no informative specificity. [KG Evidence] The "Gene" and "Protein" biological themes (connecting 6 input entities) are similarly generic hub categories and are de-emphasized. The pathway enrichment for "protein binding" (GO:0005515; 3 members) is likewise a high-connectivity annotation with limited biological specificity. [KG Evidence]

---

### 5. Gap Analysis

Under the Open World Assumption, absence of an entity means "unstudied" or "not co-expressed," not "nonexistent."

#### 5.1 Informative Absences

**Arachidonic acid (AA, 20:4 n-6) and AA-containing phospholipids.** The complete exclusion of omega-6 arachidonate species from a module dominated by omega-3 (EPA, DHA) species is the most informative absence. [KG Evidence; Inferred] This pattern indicates that the module captures a specifically anti-inflammatory omega-3 enrichment axis that is either inversely correlated with or independent of pro-inflammatory omega-6 pathways. The absence may reflect dietary omega-3 intake, a Lands-cycle remodeling program favoring omega-3 acyl chain incorporation, or both.

**Ceramides.** The absence of ceramide species (Cer d18:1/16:0, Cer d18:1/24:1, etc.) from a module containing complex glycerophospholipids confirms that this module does not capture lipotoxic or pro-atherogenic lipid signatures. [KG Evidence; Inferred] Ceramides likely segregate into a separate module associated with insulin resistance and cardiovascular risk.

**BCAAs (leucine, isoleucine, valine).** These canonical insulin resistance markers are absent, reinforcing the interpretation that this module represents a protective rather than a metabolic dysfunction axis. [Inferred]

**IL-4, IL-13 (Th2 cytokines).** The absence of the canonical IL4R ligands is notable. Possible explanations include: (i) circulating cytokine concentrations below the assay limit of detection; (ii) the proteomics platform did not include these analytes; or (iii) IL4R's role in this module is non-canonical (metabolic rather than immunological). [KG Evidence; Inferred]

**Bile acids.** The absence of bile acid species despite FABP6 membership suggests FABP6 functions here in a non-classical capacity, possibly as a carrier for omega-3 PUFA species. [KG Evidence; Inferred]

#### 5.2 Standard Gaps (Likely Platform Limitations)

**Specialized pro-resolving mediators (resolvins, protectins, maresins)** are the bioactive downstream products of the DHA/EPA precursors in this module but require specialized picomolar-sensitivity assays not included on standard metabolomics platforms. [Model Knowledge]

**STAT6** is the primary intracellular transducer downstream of IL4R but is typically not measured on circulating proteomics panels (e.g., SomaScan, Olink). [KG Evidence; Model Knowledge]

**Glutathione (GSH/GSSG)** is labile and prone to ex vivo oxidation, likely explaining its absence despite the module's antioxidant theme. [Model Knowledge]

**Short/medium-chain acylcarnitines (C3, C4, C5, C8, C10)** are absent despite the presence of docosahexaenoylcarnitine (C22:6). Their absence indicates that this module does not capture mitochondrial dysfunction or incomplete beta-oxidation. [KG Evidence; Inferred] DHA-carnitine likely reflects DHA mobilization and transport rather than impaired oxidation.

---

### 6. Temporal Context

No explicit longitudinal timepoints were provided for this WGCNA analysis. The following considerations apply if longitudinal data are available.

**Upstream causes (candidate drivers).** Dietary omega-3 PUFA intake and IL-4/IL-13 cytokine signaling are plausible upstream drivers of this module. Omega-3 consumption would increase DHA/EPA availability for phospholipid remodeling, while IL4R activation would promote M2 macrophage polarization and associated lipid metabolic reprogramming. [Model Knowledge; Inferred]

**Downstream consequences.** The module's output may include: (i) reduced pro-inflammatory eicosanoid production (via competitive displacement of arachidonate from membrane phospholipids); (ii) enhanced plasmalogen-mediated antioxidant defense; and (iii) modulation of cardiovascular and allergic disease risk. [Model Knowledge; Inferred]

**Causal inference opportunities.** If longitudinal sampling is available, Granger causality or mediation analysis could test whether changes in omega-3 phospholipid levels precede or follow changes in IL4R expression, thereby distinguishing whether immune activation drives lipid remodeling or vice versa.

---

### 7. Research Recommendations

#### Priority 1: High-Value Experimental Validations

1. **FABP6 binding specificity for omega-3 PUFAs.** Conduct competitive binding assays (ITC, fluorescent displacement) of DHA, EPA, LPC-DHA, and LPC-EPA against recombinant FABP6 to test the prediction that FABP6 serves as an omega-3 carrier in this module. [Inferred]

2. **IL4R-driven macrophage lipidomics.** Profile membrane phospholipids (especially DHA/EPA-containing PC, PE, and plasmalogens) in IL-4-stimulated versus unstimulated human monocyte-derived macrophages to test whether IL4R signaling promotes omega-3 membrane enrichment. [Inferred]

3. **Docosahexaenoylcarnitine as a cardiovascular biomarker.** Measure docosahexaenoylcarnitine in cardiovascular disease case-control cohorts, stratified by omega-3 intake. The evidence from Liepinsh et al. (2025) that DHAC is non-cardiotoxic (unlike saturated acylcarnitines) suggests it may serve as a protective biomarker. [Literature: Liepinsh et al. 2025; Inferred]

#### Priority 2: Targeted Literature Investigations

4. **CMPF and beta-cell toxicity in omega-3 supplementation trials.** Systematically review clinical trials of fish oil supplementation for reported CMPF elevations and concurrent changes in beta-cell function, to determine whether CMPF accumulation represents a risk boundary for omega-3 supplementation. [Inferred]

5. **Plasmalogen deficiency in psoriasis and atopic disease.** The module's convergence on psoriasis (5 members) and atopic eczema (2 members) alongside plasmalogen species warrants a literature review of plasmalogen levels in these conditions. [KG Evidence; Model Knowledge]

#### Priority 3: Follow-Up Computational Analyses

6. **Module preservation analysis across tissues.** Test whether the Lightcyan module is preserved in adipose tissue, liver, and immune cell transcriptomic datasets (using WGCNA modulePreservation statistics) to identify the tissue of origin for this circulating signature. [Inferred]

7. **Omega-3/omega-6 ratio as a module eigengene predictor.** Correlate the module eigengene with dietary omega-6/omega-3 ratios (from food frequency questionnaires if available) to test the hypothesis that dietary intake drives this module. [Inferred]

8. **Mediation analysis.** If clinical phenotype data are available, test whether the Lightcyan module eigengene mediates the association between omega-3 intake and cardiovascular or allergic disease outcomes. [Inferred]

9. **Cross-module contrast with ceramide-enriched modules.** Formally compare the Lightcyan (omega-3 protective) module with any ceramide- or BCAA-enriched modules in the same WGCNA analysis to define the opposing metabolic axes captured by this cohort. [Inferred]

---

*Report generated from KRAKEN knowledge graph analysis of 27 resolved entities (4 genes, 23 metabolites/small molecules). Evidence tiers: Tier 1 (direct KG evidence, 287+ findings), Tier 2 (derived associations, 353+ findings), Tier 3 (semantic inference, 42 predictions). Two entities (1-eicosapentaenoyl-GPC, hydroxy-CMPF) had zero KG edges (cold start). Hub-flagged nodes (Homo sapiens) were de-emphasized throughout.*

### Literature References

Papers discovered via semantic search. 8 unique papers across 4 hypotheses.

| Hypothesis | Citation | Link | Relevance |
|------------|----------|------|-----------|
| Inferred role of PUBCHEM.COMPOUND:127055 | Leung RYH et al. (2020) "Serum metabolomic profiling and its association with 25-hydroxyvitamin D." | [DOI](https://doi.org/10.1016/j.clnu.2019.04.035) | — |
| Bridge: Gene → SmallMolecule (2 hops) |  (2020) "Frontiers \| Thrombospondin 2/Toll-Like Receptor 4 Axis Contributes to HIF-1α-Derived Glycolysis in C..." | [Link](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2020.557730/full) | Subsequent experiments also showed that THBS ... . In terms of mechanism, THBS ... interacted with Toll-like receptor 4... |
| Inferred role of PUBCHEM.COMPOUND:127055 |  (2025) "LC–MS/MS-based simultaneous quantification of acylcarnitines, eicosapentaenoic acid, and docosahexae..." | [Link](https://link.springer.com/article/10.1007/s00216-025-05943-8) | Acylcarnitines have emerged as valuable markers of the intracellular fatty acid content, mitochondrial functionality, an... |
| Bridge: Gene → SmallMolecule (2 hops) |  (2015) "Metabolic fate of unsaturated glucuronic/iduronic acids from glycosaminoglycans: molecular identific..." | [Link](https://pubmed.ncbi.nlm.nih.gov/25605731/) | Glycosaminoglycans in mammalian extracellular matrices are degraded to their constituents, unsaturated uronic (glucuroni... |
| Bridge: Gene → SmallMolecule (2 hops); Bridge: Gene → SmallMolecule (3 hops) |  (2021) "NICEpath: Finding metabolic pathways in large networks through atom-conserving substrate-product pai..." | [Link](https://pubmed.ncbi.nlm.nih.gov/34003971/) | Motivation: Finding biosynthetic pathways is essential for metabolic engineering of organisms to produce chemicals, biod... |
| Bridge: Gene → SmallMolecule (3 hops) |  (2019) "PathMe: merging and exploring mechanistic pathway knowledge \| BMC Bioinformatics \| Springer Nature L..." | [Link](https://link.springer.com/article/10.1186/s12859-019-2863-9) | Integrating pathway knowledge from multiple databases first requires transforming the content of each database into a co... |
| Inferred role of PUBCHEM.COMPOUND:127055 |  (2013) "Quantification of plasma carnitine and acylcarnitines by high-performance liquid chromatography-tand..." | [Link](https://link.springer.com/article/10.1007/s00216-013-7309-z) | Carnitine is an amino acid derivative that plays a key role in energy metabolism. Endogenous carnitine is found in its f... |
| Inferred role of CHEBI:133432 |  (2020) "The Nootropic Drug Α-Glyceryl-Phosphoryl-Ethanolamine Exerts Neuroprotective Effects in Human Hippoc..." | [Link](https://www.mdpi.com/1422-0067/21/3/941) | plasma membrane composition [1,2,3]. Specifically, brain cells undergo a decline of polyunsaturated n-3 fatty acids (3-P... |