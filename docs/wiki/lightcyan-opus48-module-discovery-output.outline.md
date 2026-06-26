# Lightcyan Module Run on Opus 4.8: Discovery Output (27-analyte, dev, 2026-06-24)

This document presents the discovery output of a full-module run of the **Lightcyan** WGCNA module through the Kraken 12-node discovery pipeline, executed on the dev environment on 2026-06-24 (commit `a7d7fd6`) with BioMapper entity resolution enabled. The SDK-backed pipeline nodes ran on the Opus 4.8 model rather than the default Sonnet; this run is published for direct model comparison. It is one of a per-module series over the Frailty multi-omics WGCNA modules; see the [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm).

## Run Provenance

The run submitted 27 named analytes, parsed 27 at intake, and resolved 27 distinct entities (11 biomapper, 16 fuzzy) to 24 distinct CURIEs. Triage classified 5 well-characterized, 12 moderate, 8 sparse, and 2 cold-start, with 0 edge-count measurement failures. Downstream analysis produced 889 direct-KG findings, 26 cold-start findings, 6 biological themes, 10 cross-entity bridges (2 evidence-grounded), and 59 hypotheses supported by 32 literature references. Synthesis emitted a 26272-character report. The run completed in approximately 709.9 s of wall-clock time (status complete, 0 errors). All counts are taken from the reducer-merged pipeline state.

| Stage | Result |
|---|---|
| Submitted | 27 named analytes |
| Intake | 27 parsed |
| Entity resolution | 27 resolved (11 biomapper, 16 fuzzy) to 24 distinct CURIEs |
| Triage | 5 well-characterized, 12 moderate, 8 sparse, 2 cold-start (0 measurement failures) |
| Direct KG | 889 findings |
| Cold-start | 26 findings, 3 skipped |
| Pathway enrichment | 6 biological themes |
| Integration | 10 bridges (2 evidence-grounded) |
| Literature grounding | 32 papers |
| Synthesis | 59 hypotheses, 26272-character report |
| Run total | ~709.9 s wall-clock, status complete, 0 errors |

## Related

- Companion run metrics: [Lightcyan Module Run on Opus 4.8: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/lightcyan-module-run-on-opus-48-pipeline-performance-report-27-analyte-dev-2026-06-24-hjke25MdQM)
- Model comparison baseline (Sonnet): [Lightcyan Module Run: Discovery Output](https://phwiki.phenoma.ai/doc/lightcyan-module-run-discovery-output-27-analyte-dev-2026-06-23-Uwabc4tJ77)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---



## Lightcyan WGCNA Module: Omega-3 Phospholipid and Th2 Immune Axis

### 1. Executive Summary

The Lightcyan WGCNA module encodes a coordinated biological program that unites omega-3 polyunsaturated fatty acid (PUFA) phospholipid metabolism with a Th2-polarized immune signaling axis and dietary antioxidant/xenobiotic markers. [KG Evidence] The module's 27 members converge on three interlocking themes: (i) DHA- and EPA-esterified glycerophospholipids that dominate the metabolomic compartment (18 of 23 metabolites), (ii) four proteins (IL4R, CCL23, THBS2, FABP6) that share curated associations with allergic, inflammatory, and cardiometabolic diseases, and (iii) dietary-origin small molecules (ergothioneine, ascorbic acid 3-sulfate, CMPF derivatives) whose co-expression with the PUFA lipids suggests a shared absorptive or dietary-intake driver. [KG Evidence; Inferred] The systematic absence of pro-inflammatory cytokines (TNF-alpha, IL-6, CRP), ceramides, and branched-chain amino acids confirms that this module captures a biologically distinct regulatory program: one that is anti-inflammatory, omega-3 enriched, and likely diet-responsive, rather than reflective of the classical metaflammation axis of insulin resistance.

### 2. Key Findings (Tier 1 to 2)

#### 2.1 Disease Convergence Across the Protein Members

All four protein-coding genes share curated disease associations that cluster into three domains. [KG Evidence]

| Disease Domain | Representative Conditions | Members Sharing Association | Strength |
|---|---|---|---|
| Allergic/Atopic | Psoriasis (5 members), asthma (4), atopic eczema (2), allergic rhinitis (4) | IL4R, CCL23, THBS2, FABP6 (+ phosphatidylcholine for psoriasis) | Curated |
| Cardiometabolic | Coronary artery disorder (4), essential hypertension (4), diabetes mellitus (3) | IL4R, THBS2, FABP6, CCL23 | Curated |
| Gastrointestinal/Mucosal | Irritable bowel syndrome (4), gastroesophageal reflux (4), gastroduodenitis (4) | IL4R, THBS2, FABP6, CCL23 | Curated |

Psoriasis achieves the broadest recurrence (5 of 27 members, including phosphatidylcholine), making it the strongest module-level disease signature. [KG Evidence] Asthma and coronary artery disorder each recur across four protein members, consistent with a Th2 immune/cardiometabolic interface. [KG Evidence]

Metabolic dysfunction-associated steatotic liver disease (MASLD) was associated with both threonate and phosphatidylcholine (text-mined evidence), suggesting a hepatic dimension to the module's metabolic footprint. [KG Evidence]

#### 2.2 Pathway Architecture of the Protein Members

The four genes share membership in biologically coherent pathways. [KG Evidence]

- **Th2 immune signaling**: IL4R participates in IL4 signaling, Th1/Th2 differentiation, positive regulation of Th2 cell differentiation, positive regulation of immunoglobulin production, and cytokine-cytokine receptor interaction (shared with CCL23). [KG Evidence] CCL23 participates in immune response, signal transduction, and cell-cell signaling. [KG Evidence]
- **PI3K-Akt-mTOR axis**: IL4R and THBS2 co-participate in PI3K-Akt signaling and focal adhesion PI3K-Akt-mTOR signaling. [KG Evidence] This pathway provides a mechanistic bridge between immune activation (via IL4R) and extracellular matrix remodeling (via THBS2).
- **Extracellular matrix and angiogenesis**: THBS2 participates in negative regulation of angiogenesis, extracellular matrix structural constitution, cell adhesion, and focal adhesion. [KG Evidence]
- **Stress response and smoking exposure**: All four proteins share "response to stress" (GO:0006950) and "Smoking" (UMLS:C0037369) annotations. [KG Evidence] The latter is notable, as smoking status represents a modifiable exposure that may influence module expression.

#### 2.3 Lipase Network Connecting the Phospholipid Members

Pathway enrichment analysis reveals that six lipase genes (PNLIP, LIPC, LIPA, PNLIPRP2, LIPF, LIPG) plus the bile salt-stimulated lipase CEL, diacylglycerol acyltransferase DGAT1, lipoprotein lipase LPL, and carnitine palmitoyltransferase CPT2 serve as shared neighbors connecting five or more of the DHA/EPA-containing glycerolipid module members. [KG Evidence] These enzymes collectively encode the lipolysis, remodeling, and beta-oxidation machinery for long-chain polyunsaturated fatty acids, reinforcing the interpretation that this module captures omega-3 PUFA processing from dietary absorption through membrane incorporation to mitochondrial catabolism. [Inferred]

LIPC (hepatic lipase; 80 edges), LPL (lipoprotein lipase; 120 edges), and LIPA (lysosomal acid lipase; 80 edges) have the highest connectivity among these shared neighbors; LPL approaches the hub threshold and should be interpreted with appropriate caution. [KG Evidence]

#### 2.4 Heparin Binding as a Molecular Activity Bridge

Heparin binding (GO:0008201) connects two input entities, likely IL4R and THBS2. [KG Evidence] THBS2 is a well-characterized heparin-binding matricellular protein, and IL4R signaling can be modulated by heparan sulfate proteoglycans. This shared molecular activity provides a potential extracellular matrix-dependent mechanism for coordinated regulation. [Inferred]

#### 2.5 Member Prioritization: Highest-Leverage Entities

From the Member Prioritization Table, the following entities merit particular attention: [KG Evidence]

- **IL4R** (2,735 edges): The most connected member and the canonical Th2 immune receptor. Its established interactions with STAT6, JAK1, JAK2, JAK3, IRS1, IRS2, and IL13RA1 position it as the primary immune signaling node of the module. [KG Evidence]
- **THBS2** (2,570 edges): An extracellular matrix glycoprotein whose roles in angiogenesis inhibition, synapse assembly, and focal adhesion provide the module's tissue-remodeling component. [KG Evidence]
- **FABP6** (1,468 edges): The ileal bile acid binding protein, whose presence is unexpected in an omega-3 lipid module and represents one of the most intriguing connections (see Section 3). [KG Evidence]
- **Phosphatidylcholine** (410 edges): The only metabolite with substantial KG coverage; associated with hypothyroidism and contributing to psoriasis and diabetes mellitus recurrence. [KG Evidence]
- **Ergothioneine** (94 edges): A diet-derived thiol antioxidant exclusively obtained from fungal sources (mushrooms), associated with learning disability in the KG. [KG Evidence]
- **CMPF (3-carboxy-4-methyl-5-propyl-2-furanpropanoate)** (26 edges): A uremic toxin and fish-oil metabolite associated with macular dystrophy. [KG Evidence]

### 3. Novel Predictions (Tier 3)

#### 3.1 FABP6 Co-Expression with Omega-3 Lipids Independent of Bile Acid Transport

**Logic chain**: FABP6 is the ileal bile acid binding protein, yet its cargo molecules (bile acids: taurocholate, glycocholate, chenodeoxycholate, deoxycholate) are absent from this module. [KG Evidence; Inferred] Simultaneously, FABP6 co-varies with 18 omega-3 PUFA-containing glycerophospholipids in a module whose shared neighbors are lipases (PNLIP, LIPC, LPL, CEL). [KG Evidence] This dissociation between FABP6 expression and bile acid levels suggests that FABP6 may possess an underappreciated role in intestinal lipid absorption or PUFA trafficking that is independent of its canonical bile acid transport function. The structural similarity between bile acid amphiphiles and long-chain fatty acid micelles provides plausible biophysical grounds for such a dual function. [Inferred]

**Calibration**: Approximately 18% of computational predictions of this nature progress to clinical investigation. This prediction is strengthened by the module's enrichment for intestinal lipases (CEL, PNLIP, PNLIPRP2) that specifically operate in the intestinal absorptive context where FABP6 is expressed.

**Validation step**: (i) Test whether FABP6 protein binds DHA or EPA in a fluorescence displacement assay; (ii) examine whether FABP6 expression in intestinal organoids correlates with DHA-phospholipid incorporation; (iii) check FABP6 knockout mouse data for altered circulating omega-3 phospholipid profiles.

#### 3.2 Docosahexaenoylcarnitine as a Cardiovascular Disease Biomarker

**Logic chain**: Docosahexaenoylcarnitine (C22:6 acylcarnitine) has only 2 KG edges (cold-start entity). Semantic analogy to methylthiopropionylcarnitine (0.74 similarity) suggests a relationship with cardiovascular diseases (UMLS:C0007258). [KG Evidence] This inference is corroborated by grounded literature: Leung et al. (2020) demonstrated that "docosahexaenoylcarnitine and eicosapentaenoylcholine had the h[ighest]" correlations with serum 25(OH)D levels among 835 measured metabolites, and DHA-derived acylcarnitines have been quantified in human plasma as markers of fatty acid beta-oxidation. [Literature: Leung et al. 2020; LC-MS/MS acylcarnitine quantification, 2025] Separately, the inferred classification of docosahexaenoylcarnitine as an O-acylcarnitine (subclass of CHEBI:17387) and as a blood-detectable metabolite is supported by the structural logic that all acylcarnitines share this parent class and are routinely measured in plasma. [KG Evidence; Literature: acylcarnitine quantification, 2013]

**Calibration**: ~18% validation-rate caveat applies. The grounded literature substantially increases confidence relative to a purely computational prediction, as docosahexaenoylcarnitine has been directly measured in human serum and its levels correlate with vitamin D, a known cardiometabolic modifier.

**Validation step**: (i) Test association between plasma docosahexaenoylcarnitine and incident cardiovascular events in existing metabolomics cohorts; (ii) assess whether docosahexaenoylcarnitine adds predictive value beyond DHA and standard acylcarnitine panels.

#### 3.3 Ergothioneine and CMPF as Dietary-Intake Biomarkers Linking the Module to a Fish/Mushroom-Rich Diet

**Logic chain**: Ergothioneine is obtained exclusively from dietary sources (mushrooms, organ meats); CMPF is a furanoid fatty acid metabolite elevated following fish oil consumption; and the module's lipid compartment is overwhelmingly composed of DHA (22:6) and EPA (20:5) esterified phospholipids characteristic of marine omega-3 intake. [Model Knowledge] The co-expression of these three chemically distinct marker classes in a single WGCNA module suggests that a shared dietary pattern (fish and/or mushroom consumption) drives their coordinated variation. Ascorbic acid 3-sulfate, a vitamin C conjugate, further supports a diet-quality axis. [Inferred]

**Calibration**: ~18% validation-rate caveat applies to the prediction that dietary pattern is the causal driver. Alternative explanations include shared hepatic or renal clearance mechanisms.

**Validation step**: (i) Correlate module eigengene with dietary fish/mushroom intake from food-frequency questionnaires; (ii) test whether omega-3 supplementation in an intervention trial shifts the module eigengene; (iii) assess CMPF/ergothioneine ratios as candidate biomarkers of adherence to an anti-inflammatory diet.

#### 3.4 IL4R-IRS1/IRS2 Axis as a Mechanistic Bridge Between Th2 Immunity and Insulin Sensitivity

**Logic chain**: IL4R physically interacts with IRS1 (NCBIGene:3667) and IRS2 (NCBIGene:8660) in the KG. [KG Evidence] IRS1 and IRS2 are the canonical insulin receptor substrates, and IL4 signaling through IRS proteins activates PI3K-Akt. [Model Knowledge] The module's simultaneous encoding of Th2 immune signaling (IL4R, CCL23) and omega-3 PUFA lipids (which enhance insulin sensitivity) suggests a functional circuit in which dietary omega-3 PUFAs modulate Th2 immune tone via IL4R-IRS-PI3K-Akt, with downstream effects on metabolic homeostasis. [Inferred]

**Calibration**: ~18% validation-rate caveat applies. The IL4R-IRS interaction is KG-validated, but the functional integration with omega-3 lipids within this module is speculative.

**Validation step**: (i) Assess whether omega-3 supplementation alters IL4R-dependent IRS1/IRS2 phosphorylation in monocyte or macrophage models; (ii) test mediation of IL4R-IRS signaling in the association between omega-3 intake and Th2 cytokine profiles.

### 4. Biological Themes

#### 4.1 Unifying Theme: A Diet-Responsive Omega-3 Phospholipid and Anti-Inflammatory Immune Module

The module's dominant biological signal is the coordinated metabolism of omega-3 PUFAs (DHA and EPA) across multiple glycerophospholipid species, coupled with a Th2-skewed immune signature and dietary antioxidant markers. [KG Evidence; Inferred]

Three convergent lines of evidence establish this theme:

1. **Lipid biochemistry**: 18 of 23 metabolites contain DHA (22:6) or EPA (20:5) acyl chains esterified to glycerophosphocholine (GPC), glycerophosphoethanolamine (GPE), or carnitine backbones. Shared enzymatic neighbors include lipases (PNLIP, LIPC, LIPG, LIPF), acyltransferases (DGAT1), and beta-oxidation enzymes (CPT2, LPL), covering the full trajectory from dietary absorption through membrane remodeling to mitochondrial catabolism. [KG Evidence]

2. **Immune polarization**: IL4R and CCL23 define a Th2/allergic immune axis. IL4R drives Th2 differentiation, immunoglobulin class-switching, and mast cell degranulation, while simultaneously suppressing Th1 responses. [KG Evidence] CCL23 is a CC-chemokine involved in immune cell recruitment. The notable absence of Th1/metaflammatory mediators (TNF-alpha, IL-6, IL-1beta, CRP) confirms the module's immunological specificity. [KG Evidence; Inferred]

3. **Diet-derived small molecules**: Ergothioneine (fungal-dietary origin), CMPF (fish-oil furanoid metabolite), ascorbic acid 3-sulfate (vitamin C conjugate), and threonate (ascorbate catabolite) are all markers of dietary quality or specific food-group consumption. [Model Knowledge] Their co-clustering with omega-3 lipids is consistent with a dietary-pattern driver.

#### 4.2 Hub-Filtered Insights

The following hub nodes were identified and de-emphasized in this analysis: [KG Evidence]

- *Homo sapiens* (NCBITaxon:9606; ~50,000 edges): a trivially shared taxonomic node.
- *Protein binding* (GO:0005515; ~15,000 edges): annotated to 3 members (IL4R, CCL23, THBS2); this is a non-specific molecular function annotation.
- *Plant metabolite* (CHEBI:83056; ~2,000 edges) and *mammalian metabolite* (CHEBI:84735; ~3,000 edges): broad chemical classification nodes.

Cross-type bridges passing through these hubs (e.g., the 3-hop path THBS2 → Homo sapiens → blood → threonate) carry low specificity and should not be interpreted as evidence of a direct functional relationship. [KG Evidence; Inferred]

#### 4.3 Plasmalogen Subgroup

Notably, the module contains four plasmalogen species (1-enyl-palmitoyl and 1-enyl-stearoyl ether-linked DHA-containing GPE and GPC species). [Inferred from entity names] Plasmalogens are enriched in neural and cardiac membranes and serve as endogenous antioxidants; their co-expression with ester-linked DHA phospholipids suggests coordinate regulation of both ester and ether branches of the Lands cycle for DHA membrane incorporation. [Model Knowledge]

### 5. Gap Analysis

#### 5.1 Informative Absences (Open World Assumption)

The following expected-but-absent entities reveal defining characteristics of the Lightcyan module:

| Absent Entity | Expected Because | Interpretation |
|---|---|---|
| **FADS1/FADS2** | Rate-limiting desaturases for endogenous DHA/EPA synthesis; strongest GWAS hits for circulating PUFA levels | Most notable absence. Likely absent from the proteomics panel (membrane-bound hepatic enzymes). Alternatively, indicates that module-level PUFA variation is driven by dietary intake rather than endogenous biosynthesis. [Inferred] |
| **TNF-alpha, IL-6, IL-1beta** | Canonical metaflammation cytokines | Confirms Th2/anti-inflammatory polarization of the module's immune compartment, distinct from the Th1/metaflammation axis. [Inferred] |
| **CRP** | Most common clinical inflammatory biomarker | Consistent with Th2 (not Th1) immune signaling; CRP is driven by IL-6/Th1 pathways. [Inferred] |
| **Ceramides** | Strongest validated lipid biomarkers for T2D/CVD; biologically antagonistic to omega-3 PUFAs | Likely in a negatively correlated module. Confirms biological separation of omega-3 protective axis from ceramide/lipotoxicity axis. [Inferred] |
| **BCAAs** | Most replicated early predictors of T2D | Likely in a separate insulin-resistance module. Confirms the Lightcyan module captures a dietary/absorptive axis, not a BCAA-driven metabolic dysfunction axis. [Inferred] |
| **Arachidonic acid (AA, 20:4n-6) phospholipids** | Omega-6 counterparts competing for sn-2 positions via the Lands cycle | Absence is biologically coherent; confirms exclusive omega-3 enrichment and likely negative correlation with an omega-6 module. [Inferred] |
| **Bile acid species** | FABP6 cargo molecules | Suggests FABP6 co-varies with omega-3 lipids independently of its bile acid transport function (see Section 3.1). [Inferred] |
| **Adiponectin** | Anti-inflammatory adipokine; often co-varies with omega-3 levels | May cluster with a broader adipokine/metabolic module; absence suggests the module is driven by dietary absorption rather than adipose tissue signaling. [Inferred] |

#### 5.2 Standard (Platform-Related) Gaps

STAT6/GATA3 (intracellular Th2 transcription factors), PPARs (nuclear receptors), insulin/C-peptide, HbA1c, and HOMA-IR are absent, most likely due to platform limitations: intracellular transcription factors and nuclear receptors are not detectable in plasma proteomics, and glycemic indices are clinical variables rather than omics-platform analytes. [Inferred] SLC22A4/OCTN1 (the sole ergothioneine transporter) is also absent; if this protein is measured on the proteomics panel but clusters elsewhere, it would indicate that ergothioneine levels in this module are driven by dietary intake patterns rather than transporter expression. [Inferred]

### 6. Temporal Context

No longitudinal metadata were provided for this WGCNA analysis; therefore, formal causal inference regarding temporal ordering is not possible. [Model Knowledge] The following upstream/downstream framework is proposed on biological grounds:

- **Upstream (probable causes)**: Dietary omega-3 PUFA and antioxidant intake represents the most parsimonious upstream driver. The coordinate variation of DHA/EPA phospholipids, ergothioneine, CMPF, and ascorbic acid 3-sulfate is most easily explained by dietary pattern as the common cause. [Inferred]
- **Intermediate processing**: Hepatic and intestinal lipase activity (LIPC, PNLIP, CEL, LPL) and phospholipid remodeling via the Lands cycle generate the diversity of DHA/EPA-containing GPC and GPE species observed. FABP6 may facilitate intestinal lipid absorption. [Inferred]
- **Downstream consequences**: Th2 immune polarization (IL4R, CCL23) and extracellular matrix remodeling (THBS2) may represent downstream physiological responses to omega-3 membrane enrichment. Omega-3 PUFAs are known to modulate immune cell membrane composition, receptor signaling, and inflammatory tone. [Model Knowledge]

If longitudinal data become available, mediation analysis testing whether baseline omega-3 lipid levels predict subsequent changes in IL4R-dependent immune markers would be of particular value.

### 7. Research Recommendations

#### Priority 1: Experimental Validations

1. **FABP6 omega-3 binding assay**: Test direct binding of DHA and EPA to recombinant FABP6 protein using fluorescence displacement (e.g., ANS displacement assay). This addresses the novel prediction that FABP6 has an uncharacterized role in omega-3 lipid transport. [Addresses Section 3.1]

2. **Dietary-pattern correlation**: Correlate the Lightcyan module eigengene with food-frequency questionnaire data (specifically fish, shellfish, and mushroom intake). If dietary data are available in this cohort, this analysis can be performed immediately. [Addresses Section 3.3]

3. **Docosahexaenoylcarnitine cardiovascular association**: Query existing metabolomics cohorts with cardiovascular endpoints (e.g., Framingham, MESA) for docosahexaenoylcarnitine levels and incident CVD events. [Addresses Section 3.2]

#### Priority 2: Bioinformatic Follow-Up

4. **Cross-module correlation**: Compute module-module correlations to test the prediction that ceramides, BCAAs, and AA-containing phospholipids occupy negatively correlated modules. A significant negative correlation between the Lightcyan eigengene and a ceramide-enriched module would confirm biological antagonism. [Addresses Section 5.1]

5. **Module-trait association**: Correlate the Lightcyan eigengene with clinical traits (HOMA-IR, HbA1c, CRP, BMI, triglycerides) to establish the module's cardiometabolic signature. The gap analysis predicts that this module will associate with insulin sensitivity and anti-inflammatory markers rather than with insulin resistance and pro-inflammatory markers.

6. **FADS1/FADS2 genotype interaction**: Test whether FADS1/FADS2 genotype (the strongest GWAS determinant of circulating PUFA levels) modifies the Lightcyan module eigengene. A significant interaction would clarify whether the module is driven by dietary intake (genotype-independent) or endogenous biosynthesis (genotype-dependent). [Addresses Section 5.1]

#### Priority 3: Literature and Database Searches

7. **FABP6-PUFA literature review**: Conduct a systematic search for evidence of FABP6 binding to fatty acids other than bile acids. Examine structural studies (crystal structures with non-bile-acid ligands) and expression-QTL data linking FABP6 variants to circulating lipid species.

8. **Ergothioneine-immune axis**: Search for evidence connecting ergothioneine to Th2 immune modulation. Ergothioneine has documented cytoprotective and antioxidant properties; whether it influences Th2/Th1 polarization is poorly characterized but would provide an additional mechanistic link within the module.

9. **Ascorbic acid 3-sulfate characterization**: This metabolite has only 2 KG edges and warrants ontological classification. Tier 3 inference strongly supports its classification as a sulfuric ester (CHEBI:63395) and ascorbic acid derivative (CHEBI:22652) based on analogy to ascorbic acid 2-sulfate (94% similarity). [KG Evidence] Verification in ChEBI and exploration of its biological role as a vitamin C storage form or catabolite would be valuable.

#### Priority 4: Entity Resolution Quality Check

10. **Lipid entity resolution audit**: Several DHA/EPA-containing phospholipids resolved to PUBCHEM.COMPOUND identifiers at 70% confidence (fuzzy match), and some mapped to triacylglycerol rather than diacyl-phospholipid entries (e.g., 1-palmitoyl-2-docosahexaenoyl-GPC mapped to "1-Palmitoyl-2-docosahexaenoyl-3-stearoyl-glycerol," a triacylglycerol). These mismatches may affect KG traversal accuracy. Manual curation of lipid identifiers against LIPID MAPS or SwissLipids is recommended before downstream computational analyses.

---

*Report generated from KRAKEN knowledge graph analysis of the Lightcyan WGCNA module (4 proteins, 23 metabolites; 27 entities resolved). Evidence tiers are labeled throughout; all factual claims carry provenance tags. Tier 3 predictions carry an approximate 18% historical progression rate to clinical investigation and require independent experimental validation.*

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
| Inferred role of CHEBI:133432 |  (2021) "Structurally different lysophosphatidylethanolamine species stimulate neurite outgrowth in cultured ..." | [Link](https://www.sciencedirect.com/science/article/abs/pii/S0006291X2032163X) | Phospholipids are amphiphilic molecules with a hydrophilic head esterified by a hydrophobic fatty acid chain. They play... |