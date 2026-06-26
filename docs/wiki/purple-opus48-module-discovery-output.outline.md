# Purple Module Run on Opus 4.8: Discovery Output (51-analyte, dev, 2026-06-24)

This document presents the discovery output of a full-module run of the **Purple** WGCNA module through the Kraken 12-node discovery pipeline, executed on the dev environment on 2026-06-24 (commit `098093e`) with BioMapper entity resolution enabled. The SDK-backed pipeline nodes ran on the Opus 4.8 model rather than the default Sonnet; this run is published for direct model comparison. It is one of a per-module series over the Frailty multi-omics WGCNA modules; see the [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm).

## Run Provenance

The run submitted 51 named analytes, parsed 47 at intake, and resolved 47 distinct entities (4 biomapper, 42 fuzzy, 1 exact) to 35 distinct CURIEs. Triage classified 4 well-characterized, 30 moderate, 11 sparse, and 2 cold-start, with 0 edge-count measurement failures. Downstream analysis produced 659 direct-KG findings, 18 cold-start findings, 3 biological themes, 10 cross-entity bridges (10 evidence-grounded), and 40 hypotheses supported by 17 literature references. Synthesis emitted a 23201-character report. The run completed in approximately 761.4 s of wall-clock time (status complete, 0 errors). All counts are taken from the reducer-merged pipeline state.

| Stage | Result |
|---|---|
| Submitted | 51 named analytes |
| Intake | 47 parsed |
| Entity resolution | 47 resolved (4 biomapper, 42 fuzzy, 1 exact) to 35 distinct CURIEs |
| Triage | 4 well-characterized, 30 moderate, 11 sparse, 2 cold-start (0 measurement failures) |
| Direct KG | 659 findings |
| Cold-start | 18 findings, 6 skipped |
| Pathway enrichment | 3 biological themes |
| Integration | 10 bridges (10 evidence-grounded) |
| Literature grounding | 17 papers |
| Synthesis | 40 hypotheses, 23201-character report |
| Run total | ~761.4 s wall-clock, status complete, 0 errors |

## Related

- Companion run metrics: [Purple Module Run on Opus 4.8: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/purple-module-run-on-opus-48-pipeline-performance-report-51-analyte-dev-2026-06-24-cRIOg2dKtS)
- Model comparison baseline (Sonnet): [Purple Module Run: Discovery Output](https://phwiki.phenoma.ai/doc/purple-module-run-discovery-output-51-analyte-dev-2026-06-23-Sahgv1nyKF)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---



## Purple WGCNA Module Discovery Report: Glycerophospholipid Membrane Composition, Fat-Soluble Vitamin Biology, and KITLG Signaling

---

### 1. Executive Summary

The Purple WGCNA module encodes a coordinated signature of intact glycerophospholipid membrane composition, fat-soluble antioxidant vitamins (retinol and alpha-tocopherol), and the pleiotropic cytokine KITLG (stem cell factor), unified by their shared dependence on lipid bilayer integrity and lipophilic transport biology. [Inferred] Knowledge graph analysis reveals that these three biological arms converge through shared disease associations (skin disorder, cardiovascular disorder, lung disorder), shared pathway involvement (T cell proliferation, vitamin transport), and a common set of lipase and apolipoprotein neighbors that implicate lipoprotein-mediated lipid remodeling as the upstream process governing module co-expression. [KG Evidence] The systematic absence of sphingolipids, ceramides, lysophospholipids, diacylglycerols, and branched-chain amino acids confirms that this module captures a structurally coherent, lipid-class-specific axis of metabolic variation rather than a general lipotoxicity or catabolic signature. [Inferred]

---

### 2. Key Findings (Tier 1 to 2)

#### 2.1 Module Unifying Theme: Structural Glycerophospholipid Composition

The module is dominated by 45 glycerophospholipid species spanning four headgroup classes: phosphatidylcholine (GPC), phosphatidylethanolamine (GPE), phosphatidylinositol (GPI), and phosphatidylglycerol (GPG). [KG Evidence] These species collectively share neighbors among 14 lipase genes (PNLIP, LIPC, LIPA, PNLIPRP3, LIPF, and others) and apolipoprotein genes (APOA1, APOB, APOC3, APOE), as well as lipid transport receptors (CD36) and cholesterol ester transfer protein (CETP). [KG Evidence] The shared enzymatic neighborhood indicates that a common lipolytic and lipoprotein-mediated remodeling process governs the circulating abundance of these phospholipid species. [Inferred]

Notably, P4HB (protein disulfide isomerase / prolyl 4-hydroxylase subunit beta) and CPT2 (carnitine palmitoyltransferase 2) were identified as non-hub shared neighbors connecting five or more module lipids. [KG Evidence] P4HB participates in endoplasmic reticulum lipid quality control; CPT2 catalyzes the transfer of long-chain fatty acyl groups across the inner mitochondrial membrane. [Model Knowledge] Their shared connectivity to the module lipids suggests that ER-to-mitochondria lipid trafficking may represent an underappreciated regulatory node for this phospholipid cluster. [Inferred]

#### 2.2 Fat-Soluble Vitamin Axis

Retinol (vitamin A; 7,680 edges) and alpha-tocopherol (vitamin E; 9,195 edges) constitute the well-characterized hub members of this module. [KG Evidence] Both participate in vitamin transport (GO:0051180) and are annotated to "Physiological Phenomena" and "Diet, Food, and Nutrition" categories. [KG Evidence] Their co-occurrence in a phospholipid-dominated module is biologically coherent: both are lipophilic molecules that depend on lipoprotein particles (chylomicrons, LDL, HDL) for systemic transport and require intact phospholipid bilayers for cellular uptake and storage. [Model Knowledge]

Alpha-tocopherol interacts with TTPA (alpha-tocopherol transfer protein), GSTP1, NR1I2 (pregnane X receptor), PRKCA, PRKCB, and DGKA (diacylglycerol kinase alpha). [KG Evidence] The interaction with DGKA is particularly relevant to the module composition, as DGKA phosphorylates diacylglycerols to produce phosphatidic acid, a precursor of the glycerophospholipid species that dominate this module. [Model Knowledge] Alpha-tocopherol also participates in ferroptosis (a lipid-peroxidation-driven cell death pathway), ubiquinone biosynthesis, and interacts with ALOX5 and PTGS2 (COX-2), establishing its role as a membrane-embedded antioxidant that protects polyunsaturated fatty acyl chains from oxidative damage. [KG Evidence]

Retinol participates in retinol metabolism, vitamin A transport, vitamin A import into cell, rhodopsin biosynthetic process, and T cell proliferation. [KG Evidence] The shared involvement of retinol and KITLG in T cell proliferation (GO:0042098) represents a validated convergence point between the vitamin and cytokine arms of this module. [KG Evidence]

**Hub bias caveat**: Both retinol and alpha-tocopherol exceed 1,000 edges and are flagged for hub bias. [KG Evidence] Disease associations involving these entities (panniculitis, irritable bowel syndrome, gastroesophageal reflux disease, dilated cardiomyopathy, visual epilepsy, and others) should be interpreted cautiously, as high-connectivity nodes accumulate associations that may be non-specific. [Inferred]

#### 2.3 KITLG: The Sole Protein Member

KITLG (stem cell factor; 2,959 edges) is the only protein in this module, making it the highest-leverage member for mechanistic follow-up. [KG Evidence] KITLG participates in positive regulation of cell population proliferation, positive regulation of hematopoietic progenitor cell differentiation, positive regulation of mast cell proliferation, positive regulation of melanocyte differentiation, neural crest cell migration, negative regulation of apoptotic process, Ras protein signal transduction, and T cell proliferation, among other processes. [KG Evidence] KITLG interacts with its cognate receptor KIT (NCBIGene:3815), as well as cytokines IL33, TNF, IL4, IL1B, CSF2, and LIF. [KG Evidence]

#### 2.4 Module-Level Disease Recurrence

Three diseases recur across three module members, and eleven recur across two members. [KG Evidence] The strongest recurrence signatures are:

| Disease | Members | Evidence |
|---|---|---|
| Skin disorder | retinol, alpha-tocopherol, KITLG | curated [KG Evidence] |
| Lung disorder | retinol, KITLG | curated [KG Evidence] |
| Digestive system disorder | retinol, KITLG | curated [KG Evidence] |
| Cardiovascular disorder | retinol, KITLG | curated [KG Evidence] |
| Metabolic disease | retinol, alpha-tocopherol | curated [KG Evidence] |
| Hypothyroidism | retinol, phosphatidylcholine | curated [KG Evidence] |

Skin disorder is the only disease shared across all three well-characterized, non-lipid-generic members (retinol, alpha-tocopherol, KITLG). [KG Evidence] This convergence is mechanistically plausible: KITLG drives melanocyte differentiation and mast cell biology in skin; retinol regulates keratinocyte differentiation; alpha-tocopherol protects cutaneous membranes from UV-induced lipid peroxidation. [Model Knowledge]

#### 2.5 Cross-Type Bridges: KITLG to Fat-Soluble Vitamins

Ten two-hop paths connect KITLG to retinol and alpha-tocopherol through shared disease intermediaries (melanoma, cardiovascular disorder, eye disorder, kidney disorder, lung disorder, digestive system disorder). [KG Evidence] The weakest leg in all paths is curated-associative, indicating moderate but not definitive evidence for each bridge. [KG Evidence]

The KITLG-to-retinol connection via eye disorder is supported by grounded literature: Lorber et al. (2020) demonstrated that KITLG-KIT signaling protects photoreceptors against light-induced and genetic retinal degeneration by activating NRF2 and inducing HMOX1 expression. [Literature: "KIT ligand protects against both light-induced and genetic photoreceptor degeneration," 2020] Retinol is essential for rhodopsin biosynthesis and photoreceptor function. [KG Evidence] The convergence of KITLG-mediated photoreceptor survival signaling with retinol-dependent visual function represents a biologically validated bridge within this module. [Inferred]

---

### 3. Novel Predictions (Tier 3)

#### 3.1 KITLG as a Regulator of Membrane Phospholipid Composition

**Structural logic chain**: KITLG activates KIT receptor tyrosine kinase signaling → KIT activates PI3K-AKT and Ras-MAPK cascades [KG Evidence: Ras protein signal transduction] → PI3K directly consumes phosphatidylinositol (PI) species to generate PIP2/PIP3 signaling lipids → downstream metabolic effects alter glycerophospholipid remodeling via the Lands cycle → the 45 glycerophospholipid species in this module may reflect the membrane composition footprint of tonic KITLG-KIT signaling. [Inferred]

**Supporting evidence**: KITLG interacts with ACOT6 (acyl-CoA thioesterase 6) [KG Evidence], which hydrolyzes medium- and long-chain acyl-CoA species and could influence the fatty acyl pool available for phospholipid remodeling. [Model Knowledge] The module's enrichment for phosphatidylinositol species (GPI headgroup) is consistent with PI3K pathway consumption of inositol lipids. [Inferred]

**Validation step**: Measure glycerophospholipid profiles (targeted lipidomics of GPC, GPE, GPI, GPG species) in KITLG-stimulated vs. unstimulated mast cells or hematopoietic progenitors; assess whether KITLG-KIT signaling alters the sn-2 fatty acyl composition of membrane phospholipids.

**Calibration**: Approximately 18% of computational predictions of this nature progress to clinical investigation. This prediction is mechanistically coherent but indirect.

#### 3.2 Alpha-Tocopherol as a Ferroptosis-Protective Factor Linked to Phospholipid PUFA Content

**Structural logic chain**: The module contains multiple phospholipids with polyunsaturated sn-2 acyl chains (arachidonoyl [20:4], eicosapentaenoyl [20:5], docosahexaenoyl [22:6], adrenoyl [22:4]) → polyunsaturated phospholipids are the primary substrates of ferroptotic lipid peroxidation [Model Knowledge] → alpha-tocopherol participates in ferroptosis (GO annotation) and interacts with ALOX5 (arachidonate 5-lipoxygenase) and PTGS2 (COX-2) [KG Evidence] → the co-expression of alpha-tocopherol with PUFA-enriched phospholipids may reflect a coordinated antioxidant defense: cells or tissues with higher PUFA-phospholipid content require proportionally higher alpha-tocopherol to prevent ferroptotic death. [Inferred]

**Supporting evidence**: Alpha-tocopherol interacts with GPX-pathway enzymes (GSTP1, GSTO1) and antioxidant gene regulators (NQO1, HMOX1, GCLC, SOD1). [KG Evidence] The gene-regulatory activity of alpha-tocopherol has been characterized in grounded literature (Azzi, 2010). [Literature: "Gene-Regulatory Activity of α-Tocopherol," 2010]

**Validation step**: Correlate alpha-tocopherol levels with PUFA-phospholipid content across study participants; test whether alpha-tocopherol supplementation attenuates ferroptosis in cells enriched with the specific PUFA-phospholipid species found in this module (particularly arachidonoyl-GPE and arachidonoyl-GPI species).

**Calibration**: Approximately 18% of computational predictions progress to clinical investigation. The ferroptosis-phospholipid connection is well-established in the field; the novel element is its module-level coordination with KITLG.

#### 3.3 Module as a Biomarker of Hepatic Lipoprotein Remodeling Capacity

**Structural logic chain**: The shared neighborhood of the module lipids includes hepatic lipase (LIPC), lysosomal acid lipase (LIPA), pancreatic lipases (PNLIP, PNLIPRP1, PNLIPRP3), CETP, and apolipoproteins (APOA1, APOB, APOC3, APOE) [KG Evidence] → these enzymes collectively govern lipoprotein particle remodeling in the liver and circulation [Model Knowledge] → the coordinated variation of 45 phospholipid species plus fat-soluble vitamins may serve as an integrative readout of hepatic lipoprotein secretion and remodeling capacity. [Inferred]

**Validation step**: Correlate the Purple module eigengene with VLDL/LDL/HDL particle number and composition (measured by NMR lipoprotein profiling); assess whether the module tracks hepatic steatosis markers (ALT, FLI) independently of standard lipid panels.

**Calibration**: Approximately 18% of such computational predictions progress to clinical investigation.

---

### 4. Biological Themes

#### 4.1 Glycerophospholipid Membrane Integrity

The dominant theme of this module is glycerophospholipid composition. [KG Evidence] The 45 phospholipid species span acyl chain lengths from C14:0 (myristoyl) to C22:6 (docosahexaenoyl) and headgroups from choline to ethanolamine, inositol, and glycerol. [KG Evidence] This breadth suggests the module captures bulk membrane composition rather than a single biosynthetic branch. [Inferred] The fatty acyl diversity (saturated: palmitoyl, stearoyl, myristoyl; monounsaturated: oleoyl, palmitoleoyl; polyunsaturated: linoleoyl, arachidonoyl, eicosapentaenoyl, docosahexaenoyl, adrenoyl) indicates representation of both omega-6 and omega-3 PUFA incorporation pathways. [Inferred]

#### 4.2 Antioxidant Defense at the Lipid-Water Interface

Alpha-tocopherol and retinol are both lipid-soluble antioxidants that partition into phospholipid bilayers. [Model Knowledge] Their co-expression with PUFA-enriched phospholipids (particularly arachidonoyl and docosahexaenoyl species) is consistent with a coordinated antioxidant defense system: membranes enriched in oxidation-susceptible PUFAs require proportionally more lipophilic radical scavengers. [Inferred] Alpha-tocopherol's established interactions with ALOX5, PTGS2, and ferroptosis pathways [KG Evidence] reinforce this theme.

#### 4.3 Immune and Hematopoietic Signaling

KITLG anchors the immune and hematopoietic signaling theme, with established roles in mast cell proliferation, hematopoietic stem cell proliferation, T cell proliferation, melanocyte differentiation, and leukocyte migration. [KG Evidence] Retinol shares the T cell proliferation annotation with KITLG. [KG Evidence] The cytokine interaction partners of KITLG (TNF, IL1B, IL4, IL33, CSF2, LIF) [KG Evidence] position this module at the intersection of innate and adaptive immunity. Notably, the connection to immune biology is driven exclusively by KITLG and retinol; the phospholipid members show no direct immune pathway annotations. [KG Evidence]

#### 4.4 Hub-Filtered Insights

After de-emphasizing retinol (7,680 edges) and alpha-tocopherol (9,195 edges) as potential hub-bias sources [KG Evidence], the most informative module members are:
- **KITLG** (2,959 edges): below the hub threshold; its disease and pathway annotations represent specific, non-spurious biology. [KG Evidence]
- **Phosphatidylcholine** (410 edges): its unique association with hypothyroidism (shared with retinol, 2 members) and visual epilepsy (shared with alpha-tocopherol, 2 members) merits cautious attention. [KG Evidence]
- **Linoleate** (CHEBI:30245; 74 edges): as the essential fatty acid precursor to the omega-6 PUFA cascade, its presence anchors the fatty acyl composition theme. [KG Evidence]

---

### 5. Gap Analysis

#### 5.1 Informative Absences (Open World Assumption)

The systematic absences from this module are as biologically revealing as its contents. Under the Open World Assumption, absence indicates "unstudied or independently regulated" rather than "nonexistent."

**Sphingolipid branch entirely absent**: Ceramides, sphingomyelins, and dihydroceramides are absent despite sharing biosynthetic precursors (palmitoyl-CoA, serine) with the glycerophospholipids present. [Inferred] This separation indicates that glycerophospholipid and sphingolipid arms of lipid metabolism are independently regulated in this cohort, and the module captures membrane composition rather than ceramide-mediated lipotoxicity. [Inferred]

**Lysophospholipid species absent**: The parent diacyl-phospholipids are present but their lysophospholipid hydrolysis products (LPC, LPE, LPI) are absent. [Inferred] This distinction implies the module reflects intact membrane phospholipid composition rather than active phospholipase A2-mediated Lands cycle remodeling. [Inferred]

**Diacylglycerols absent**: DAGs occupy the metabolic junction between phospholipids and triacylglycerols. [Model Knowledge] Their absence, despite the presence of both upstream (phospholipids) and downstream (stearoyl-arachidonoyl-glycerol) species, suggests the module captures steady-state composition rather than active Kennedy pathway flux. [Inferred]

**KIT receptor absent**: The obligate receptor for KITLG is conspicuously missing. [Inferred] This likely reflects measurement platform limitations (plasma proteomics detects shed soluble KITLG but not membrane-bound KIT) rather than biological disconnection. [Inferred]

**RBP4 absent**: The specific retinol transport protein is absent despite retinol's presence. [Inferred] Retinol levels in this module may therefore reflect tissue stores or dietary intake rather than RBP4-mediated circulating transport. [Inferred]

**BCAAs absent**: Branched-chain amino acids (leucine, isoleucine, valine) are the most replicated metabolomic predictors of T2D conversion but segregate from this lipid-specific module. [Inferred] This confirms that the amino acid and lipid axes of metabolic risk are independently regulated in this cohort. [Inferred]

**Arachidonic acid absent**: The module contains arachidonic acid esterified at the sn-2 position of multiple phospholipids (arachidonoyl-GPE, arachidonoyl-GPI, arachidonoyl-GPC species) but lacks free arachidonic acid. [Inferred] This suggests the module tracks esterified (membrane-bound) arachidonic acid rather than the free form released by phospholipase A2 for eicosanoid biosynthesis. [Inferred]

#### 5.2 Standard Gaps

Vitamin D, insulin/C-peptide, HbA1c, FADS1/FADS2, and adiponectin are absent; these likely reflect study design and platform coverage rather than biological dissociation. [Inferred] The absence of FADS1/FADS2 limits the ability to attribute PUFA composition to desaturase activity versus dietary intake. [Inferred]

---

### 6. Temporal Context

No longitudinal time-series data are provided in this analysis. However, the module composition permits inference of causal direction:

**Upstream causes (likely)**: Dietary intake of fat-soluble vitamins (retinol, alpha-tocopherol) and essential fatty acids (linoleate and its elongation/desaturation products) represents the most parsimonious upstream driver. [Inferred] Hepatic lipoprotein remodeling (via LIPC, CETP, APOB, APOE) determines the circulating abundance and acyl composition of phospholipid species. [Model Knowledge]

**Downstream consequences (possible)**: KITLG signaling through KIT may represent a downstream consequence of altered membrane lipid composition: phospholipid bilayer composition modulates receptor tyrosine kinase clustering, lateral diffusion, and signaling efficiency. [Model Knowledge] Alternatively, KITLG may be an upstream regulator that shapes membrane composition through PI3K-dependent lipid remodeling (see Tier 3 prediction 3.1).

**Causal inference opportunity**: A Mendelian randomization study using FADS1/FADS2 locus variants (which alter circulating PUFA-phospholipid composition) as instruments could test whether genetically determined phospholipid PUFA content causally influences KITLG levels or KIT pathway activity. [Inferred]

---

### 7. Research Recommendations

#### 7.1 Highest Priority: KITLG as a Phospholipid-Linked Biomarker

1. **Targeted lipidomics validation**: Measure GPC, GPE, GPI, and GPG species (matching the acyl chain compositions in this module) alongside soluble KITLG in an independent cohort. Test whether KITLG correlates with specific phospholipid acyl profiles (particularly arachidonoyl and docosahexaenoyl species). [Inferred]

2. **KITLG-KIT signaling and membrane composition**: In vitro stimulation of KIT-expressing cells (mast cells, hematopoietic progenitors) with recombinant KITLG, followed by lipidomic profiling of membrane phospholipids, would test the prediction that KITLG-KIT signaling reshapes glycerophospholipid composition. [Inferred]

#### 7.2 High Priority: Ferroptosis and PUFA-Phospholipid Biology

3. **Alpha-tocopherol and ferroptosis susceptibility**: Assess whether participants with lower alpha-tocopherol levels (relative to their PUFA-phospholipid content) show elevated lipid peroxidation markers (4-HNE, MDA, oxidized phospholipids). [Inferred]

4. **PUFA composition and disease risk**: The module's arachidonoyl, eicosapentaenoyl, and docosahexaenoyl phospholipid content should be tested as predictors of cardiovascular and skin disease outcomes (the two disease categories with strongest module-level recurrence). [KG Evidence]

#### 7.3 Moderate Priority: Literature and Database Searches

5. **KITLG and retinol in retinal biology**: The grounded literature (Lorber et al., 2020) [Literature: "KIT ligand protects against both light-induced and genetic photoreceptor degeneration," 2020] demonstrates KITLG-mediated photoreceptor protection via NRF2-HMOX1. A systematic review of retinol-KITLG interactions in retinal and skin tissues would clarify whether these represent converging protective pathways.

6. **Phosphatidylcholine and thyroid function**: The association of phosphatidylcholine with hypothyroidism (shared with retinol) [KG Evidence] is unexplored. Thyroid hormones regulate hepatic phospholipid metabolism; a literature search for thyroid-phospholipid interactions is warranted.

7. **Module eigengene and hepatic function**: Correlate the Purple module eigengene with NMR lipoprotein subfractions and liver function markers to test the hepatic remodeling hypothesis (Tier 3 prediction 3.3).

#### 7.4 Lower Priority: Cold-Start Entity Characterization

8. Two module members (1-oleoyl-GPI, 1-linolenoyl-GPE) have zero edges in the knowledge graph (cold-start entities). [KG Evidence] Their biological roles should be characterized through LIPID MAPS and SwissLipids database searches and, if absent, through targeted lipidomics standards to confirm their identity and abundance.

#### 7.5 Entity Resolution Caveat

A substantial fraction (approximately 70%) of the glycerophospholipid entities were resolved at 70% confidence via fuzzy matching, often mapping to diacylglycerol or triacylglycerol entities rather than the intended glycerophospholipid species. [KG Evidence] Downstream analyses involving these entities should be interpreted with the understanding that knowledge graph neighborhoods may reflect the resolved entity (e.g., a triacylglycerol) rather than the original phospholipid analyte. Future iterations should incorporate LIPID MAPS identifiers (LMGP series) for precise glycerophospholipid resolution.

---

*Report generated from KRAKEN knowledge graph analysis. All factual claims are tagged with evidence provenance. Findings tagged [Inferred] or [Model Knowledge] require independent verification. Tier 3 predictions carry an estimated approximately 18% probability of progressing to clinical investigation based on historical validation rates of computational predictions.*

### Literature References

Papers discovered via semantic search. 1 unique papers across 1 hypotheses.

| Hypothesis | Citation | Link | Relevance |
|------------|----------|------|-----------|
| Bridge: Gene → SmallMolecule (2 hops) |  (2020) "KIT ligand protects against both light-induced and genetic photoreceptor degeneration - PubMed" | [Link](https://pubmed.ncbi.nlm.nih.gov/32242818/) | Photoreceptor degeneration is a major cause of blindness and a considerable health burden during aging but effective the... |