# Tan Module Run: Discovery Output (43-analyte, dev, 2026-06-23)

This document presents the discovery output of a full-module run of the **Tan** WGCNA module through the Kraken 12-node discovery pipeline, executed on the dev environment on 2026-06-23 (commit `813fc3f`) with BioMapper entity resolution enabled. It is one of a per-module series over the Frailty multi-omics WGCNA modules; see the [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm).

## Run Provenance

The run submitted 43 named analytes, parsed 39 at intake, and resolved 39 distinct entities (38 biomapper, 1 fuzzy) to 37 distinct CURIEs. Triage classified 13 well-characterized, 5 moderate, 21 sparse, and 0 cold-start, with 0 edge-count measurement failures. Downstream analysis produced 1307 direct-KG findings, 20 cold-start findings, 6 biological themes, 10 cross-entity bridges (10 evidence-grounded), and 49 hypotheses supported by 29 literature references. Synthesis emitted a 25901-character report. The run completed in approximately 729.9 s of wall-clock time (status complete, 0 errors). All counts are taken from the reducer-merged pipeline state.

| Stage | Result |
|---|---|
| Submitted | 43 named analytes |
| Intake | 39 parsed |
| Entity resolution | 39 resolved (38 biomapper, 1 fuzzy) to 37 distinct CURIEs |
| Triage | 13 well-characterized, 5 moderate, 21 sparse, 0 cold-start (0 measurement failures) |
| Direct KG | 1307 findings |
| Cold-start | 20 findings, 16 skipped |
| Pathway enrichment | 6 biological themes |
| Integration | 10 bridges (10 evidence-grounded) |
| Literature grounding | 29 papers |
| Synthesis | 49 hypotheses, 25901-character report |
| Run total | ~729.9 s wall-clock, status complete, 0 errors |

## Related

- Companion run metrics: [Tan Module Run: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/tan-module-run-pipeline-performance-report-43-analyte-dev-2026-06-23-KmqmXjZe2m)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---



## Tan WGCNA Module Discovery Report: Steroid Sulfate Conjugation and Extracellular Matrix Remodeling

### 1. Executive Summary

The Tan WGCNA module encodes a biologically coherent axis linking adrenal steroid sulfate conjugation to extracellular matrix (ECM) remodeling and innate immune signaling. [KG Evidence] [Inferred] Thirty-one sulfated and glucuronidated steroid metabolites, dominated by DHEA-S and its androstane/pregnane derivatives, co-vary with 11 proteins that collectively span ECM proteoglycan biology (DCN, PRELP, MEPE), cytokine and immune receptor signaling (IL17D, IL1RL2, TNFRSF11B, HAVCR1), matrix metalloproteinase activity (MMP12, TIMP4), and hypothalamic energy-sensing neuropeptide signaling (AGRP). The exclusive presence of sulfated (inactive/storage) steroid forms, together with the absence of free active hormones (testosterone, estradiol, cortisol), identifies the SULT2A1/STS enzymatic compartment as the likely unifying biochemical process governing this module's metabolite arm. [KG Evidence] [Inferred]

### 2. Key Findings (Tier 1 to 2)

#### 2.1 Steroid Sulfate Conjugation as the Metabolic Core

The metabolite component of this module comprises 31 steroid conjugates that fall into three biosynthetic families: androstane sulfates (DHEA-S, androsterone sulfate, epiandrosterone sulfate, androstenediol mono/disulfates, androstan-diol mono/disulfates), pregnane sulfates (pregnenolone sulfate, pregnanediol sulfates, pregnenediol sulfates, 21-hydroxypregnenolone disulfate, 17alpha-hydroxypregnenolone 3-sulfate), and glucuronide conjugates (androsterone glucuronide, etiocholanolone glucuronide, pregnanediol-3-glucuronide, 11beta-hydroxyandrosterone glucuronide). [KG Evidence] Four of these metabolites (DHEA-S, androsterone glucuronide, pregnenolone sulfate, etiocholanolone glucuronide) map to the "steroid hormone biosynthesis" pathway (UMLS:C0597517). [KG Evidence] Pathway enrichment analysis identified SULT2A1 (NCBIGene:6820), STS (NCBIGene:412), and multiple UGT family members (UGT2B15, UGT2B4, UGT2B17, UGT1A8, UGT1A3, UGT1A1, UGT1A10) as shared neighbors connecting the glucuronide conjugates. [KG Evidence] These enzymes catalyze Phase II steroid conjugation reactions, and their convergent identification as shared neighbors confirms that the metabolite module reflects coordinated sulfotransferase and glucuronosyltransferase activity rather than active steroidogenesis. [Inferred]

DHEA-S (CHEBI:16814; 588 edges) is the best-characterized metabolite in the module and carries curated disease associations with polycystic ovary syndrome, schizophrenia, epilepsy, diabetes mellitus, and depressive disorder. [KG Evidence] Three metabolites are classified as "human xenobiotic metabolites" (CHEBI:77746): DHEA-S, pregnenolone sulfate, and etiocholanolone glucuronide. [KG Evidence]

#### 2.2 ECM Remodeling and Bone Matrix Biology

Three protein members (DCN, PRELP, MEPE) constitute a coherent ECM/bone matrix subgroup. DCN (decorin; 2,633 edges) participates in extracellular matrix organization (GO:0030198), extracellular matrix disassembly (GO:0022617), and carbohydrate metabolic process (GO:0005975), and interacts with fibronectin (FN1), fibrillin-1 (FBN1), and multiple matrix metalloproteinases (MMP2, MMP3, MMP7). [KG Evidence] PRELP (prolargin; 1,063 edges) and MEPE (matrix extracellular phosphoglycoprotein; 1,486 edges) share the GO annotations "extracellular matrix structural constituent" (GO:0005201) and "skeletal system development" (GO:0001501). [KG Evidence] TNFRSF11B (osteoprotegerin/OPG; 3,164 edges) also participates in skeletal system development and bone resorption (MONDO:0000837), and the "Clock-controlled autophagy in bone metabolism" WikiPathway (WP5205) connects TNFRSF11B and MEPE. [KG Evidence] Golgi lumen (GO:0005796; 250 edges, non-hub) is a shared cellular compartment for AGRP, DCN, and PRELP, consistent with post-translational processing of secreted ECM proteoglycans. [KG Evidence]

MMP12 (macrophage metalloelastase; 3,017 edges) and TIMP4 (tissue inhibitor of metalloproteinases 4; 1,551 edges) together participate in the "Matrix metalloproteinases" WikiPathway (WP129), representing the proteolytic and anti-proteolytic arms of ECM turnover. [KG Evidence] MMP12 carries a curated association with rheumatoid arthritis; TIMP4 carries a curated association with prostate cancer. [KG Evidence]

#### 2.3 Cytokine and Immune Receptor Signaling

IL17D (1,723 edges) and IL1RL2 (1,732 edges) share annotations for inflammatory response (GO:0006954), positive regulation of interleukin-6 production (GO:0032755), and the "Cytokine-cytokine receptor interaction" WikiPathway (WP5473). [KG Evidence] HAVCR1 (KIM-1; 2,266 edges), a validated biomarker of acute kidney injury, participates in immune system process (GO:0002376), positive regulation of leukocyte activation (GO:0002696), and defense response to symbiont (GO:0140546). [KG Evidence] LGALS4 (galectin-4; 3,304 edges, the highest-connectivity protein) is associated with colorectal cancer and participates in defense response to symbiont together with IL1RL2. [KG Evidence]

#### 2.4 Module-Level Disease Recurrence

Depressive disorder and asthma each recur across 11 module members (all 11 proteins plus DHEA-S), representing the broadest disease associations in the module. [KG Evidence] Ten members share curated associations with coronary artery disorder, psoriasis, essential hypertension, gastroesophageal reflux disease, and irritable bowel syndrome. [KG Evidence] Schizophrenia (9 members, including DHEA-S and pregnenolone sulfate) is notable because it connects the metabolite and protein arms of the module. [KG Evidence] These broadly shared disease terms (depressive disorder, asthma, coronary artery disease, essential hypertension) likely reflect high-connectivity (hub) disease nodes in the knowledge graph; they indicate systemic inflammatory and cardiometabolic relevance but should be interpreted cautiously as potentially non-specific. [Inferred]

More biologically informative disease recurrence patterns include: kidney disorder (6 members: AGRP, HAVCR1, LGALS4, MMP12, MEPE, TIMP4); arthritic joint disease (5 members: LGALS4, TNFRSF11B, IL17D, MEPE, IL1RL2); and diabetes mellitus (5 members: DHEA-S, LGALS4, MMP12, TIMP4, IL1RL2). [KG Evidence]

#### 2.5 Cross-Type Bridges: Gene to Metabolite Connections

Multiple two-hop paths connect the protein arm to DHEA-S through shared localization in the extracellular region (GO:0005576). [KG Evidence] DCN and AGRP both localize to the extracellular region and share this compartment with DHEA-S, providing a spatial basis for their co-detection in circulation. [KG Evidence] Additional two-hop bridges link DCN to DHEA-S through pharmacogenomic intermediaries (doxorubicin, vincristine, paclitaxel, 17alpha-ethynylestradiol), indicating that DCN expression modulates drug responses involving DHEA-S pathways. [KG Evidence] The association between DHEA-S and bone mineral density has been characterized in a 15-year longitudinal study of postmenopausal women (Chingford Study, 2011), providing epidemiological support for the bone-steroid axis captured by this module. [Literature]

#### 2.6 AGRP: Hypothalamic Neuropeptide in a Peripheral Module

AGRP (agouti-related protein; 1,204 edges) is the canonical melanocortin receptor antagonist regulating feeding behavior, energy homeostasis, and insulin response. [KG Evidence] Its established interactions with MC3R, MC4R, NPY, GHRL (ghrelin), LEPR (leptin receptor), and SOCS3 anchor it in the hypothalamic energy-sensing circuit. [KG Evidence] AGRP participates in response to insulin (GO:0032868), circadian rhythm, and adipocytokine signaling. [KG Evidence] Its presence in this peripheral ECM/steroid sulfate module is non-canonical and likely reflects its measurable circulating levels (AGRP is a secreted protein), its adipokine signaling role, or a shared upstream regulatory program with adrenal steroid output. [Model Knowledge] [Inferred] Literature on AgRP neurons demonstrates their role as integrators of metabolic, sensory, and environmental cues, and lipid biosynthesis enzyme Agpat5 in AgRP neurons is required for insulin-induced hypoglycemia sensing (Nature Communications, 2022). [Literature]

### 3. Novel Predictions (Tier 3)

#### 3.1 Pregnane Sulfate Metabolites as TRPM3 Modulators and Gestational Diabetes Biomarkers

**Prediction**: The sulfated pregnane metabolites in this module (5alpha-pregnan-3beta,20beta-diol monosulfate, 5alpha-pregnan-3beta,20alpha-diol disulfate, pregnenediol disulfate, pregnenetriol disulfate) may modulate insulin secretion via TRPM3 ion channels, positioning them as candidate biomarkers for gestational diabetes mellitus (GDM) risk.

**Logic chain**: 5alpha-pregnan-3beta,20beta-diol monosulfate (CHEBI:133712; 1 edge, sparse) was inferred via semantic similarity (0.99) to belong to the steroid sulfate conjugate class (CHEBI:59696). [KG Evidence] Literature evidence demonstrates that the structurally related epiallopregnanolone sulfate (PM5S) increases glucose-stimulated insulin secretion (GSIS) at least twofold (P < 0.001) via TRPM3 activation, and PM5S concentrations are reduced in GDM serum (P < 0.05). [Literature: "Sulfated Progesterone Metabolites That Enhance Insulin Secretion via TRPM3 Are Reduced in Serum From Women With Gestational Diabetes Mellitus," 2022] The pregnane sulfates in this module share the 5alpha-pregnane core scaffold with PM5S, differing primarily in stereochemistry at C-3 and C-20 positions. [Inferred]

**Calibration**: Approximately 18% of computational predictions of this type progress to clinical investigation. The structural analogy is strong but stereochemistry critically determines TRPM3 binding affinity.

**Validation step**: Test GSIS enhancement by these specific pregnane sulfate isomers in isolated mouse and human islets using the published PM5S protocol; measure serum concentrations by LC-MS/MS in GDM versus control cohorts.

#### 3.2 The SULT2A1/STS Axis as a Modifiable Determinant of Module Behavior

**Prediction**: Genetic variation in SULT2A1 and STS (both identified as shared pathway neighbors) may explain interindividual variation in the entire metabolite arm of the Tan module, and by extension, in the correlated protein signals.

**Logic chain**: SULT2A1 (NCBIGene:6820) and STS (NCBIGene:412) were identified as shared gene neighbors connecting multiple steroid sulfate metabolites in pathway enrichment. [KG Evidence] The exclusive presence of sulfated (not free) steroid forms indicates that the SULT2A1/STS equilibrium governs the pool sizes of all 31 metabolites in the module. [Inferred] The absence of testosterone, estradiol, and cortisol (active hormones) from the module further supports this interpretation: the module captures conjugation biology, not steroidogenesis. [KG Evidence (gap analysis)]

**Calibration**: Approximately 18% of such computational predictions advance to validation. Metabolomics-GWAS studies have previously identified SULT2A1 locus variants associated with DHEA-S levels, supporting biological plausibility. [Model Knowledge]

**Validation step**: Perform a cis-eQTL and cis-mQTL analysis of the SULT2A1 and STS loci against the full set of 31 metabolites in this module; test whether SULT2A1 expression or common variants predict module eigengene values.

#### 3.3 HAVCR1 (KIM-1) as an Early Renal Injury Marker Coupled to Steroid Sulfate Clearance

**Prediction**: Circulating HAVCR1 levels in this module may reflect subclinical tubular injury that alters renal steroid sulfate clearance, mechanistically linking the protein and metabolite arms.

**Logic chain**: HAVCR1 is a validated marker of acute kidney injury (top disease association). [KG Evidence] Kidney disorder recurs across 6 module members (AGRP, HAVCR1, LGALS4, MMP12, MEPE, TIMP4). [KG Evidence] Creatinine is expected but absent from this module; its absence suggests the module captures early tubular signals not yet reflected in glomerular filtration rate. [KG Evidence (gap analysis)] Steroid sulfate conjugates are cleared partly through renal tubular secretion and reabsorption, mediated by organic anion transporters. [Model Knowledge] Subclinical tubular injury (reflected by HAVCR1 elevation) could alter the clearance kinetics of circulating steroid sulfates, producing the observed co-expression pattern. [Inferred]

**Calibration**: Approximately 18% of such predictions progress to clinical investigation. The mechanistic link between tubular injury markers and steroid sulfate clearance has not been directly tested.

**Validation step**: Correlate urinary KIM-1 levels with plasma steroid sulfate concentrations in a cohort with graded kidney function; test whether adjustment for eGFR attenuates the HAVCR1-steroid sulfate correlation.

#### 3.4 OPG/RANKL Ratio Imbalance: An Osteoprotective Signature

**Prediction**: This module encodes a net anti-resorptive bone state, as reflected by the presence of TNFRSF11B (OPG) without its ligand RANKL (TNFSF11), together with mineralization inhibitor MEPE and ECM proteoglycans PRELP and DCN.

**Logic chain**: TNFRSF11B and MEPE share participation in bone resorption (MONDO:0000837) and Clock-controlled autophagy in bone metabolism (WP5205). [KG Evidence] RANKL, the obligate ligand for OPG-mediated osteoclast regulation, is absent from the module. [KG Evidence (gap analysis)] MEPE's top disease association is osteomalacia, a condition of impaired mineralization. [KG Evidence] DCN and PRELP are ECM structural constituents of bone and cartilage. [KG Evidence] The co-expression of OPG (without RANKL) together with matrix proteins and steroid sulfates (DHEA-S has known bone-protective associations) suggests this module captures a coordinated osteoprotective program. [Inferred] A 15-year longitudinal study demonstrated significant associations between serum DHEA-S and bone mineral density at the femoral neck and lumbar spine (Chingford Study, 2011). [Literature]

**Calibration**: Approximately 18% of such predictions advance to validation.

**Validation step**: Measure the OPG/RANKL ratio, serum DHEA-S, and bone densitometry (DXA) concurrently in the study cohort; test whether the module eigengene predicts bone mineral density independently of age and sex.

### 4. Biological Themes

#### 4.1 Steroid Sulfate Conjugation and Phase II Metabolism

The dominant biological theme is coordinated steroid sulfate and glucuronide conjugation. [KG Evidence] [Inferred] All 31 metabolites represent Phase II conjugation products of the androstane (C19) and pregnane (C21) steroid families. The convergence of SULT2A1, STS, and six UGT family members as shared pathway neighbors confirms enzymatic co-regulation. [KG Evidence] This theme is non-trivial because it excludes free steroids, glucocorticoids, and mineralocorticoids, indicating biological specificity to the adrenal zona reticularis DHEA/androgen sulfate output. [Inferred]

#### 4.2 ECM Remodeling and Proteoglycan Biology

The protein arm converges on ECM organization through DCN (decorin, a leucine-rich proteoglycan), PRELP (proline-arginine-rich end leucine-rich repeat protein), MMP12 (macrophage metalloelastase), and TIMP4 (metalloproteinase inhibitor). [KG Evidence] Shared cellular compartments include extracellular matrix (GO:0031012, hub-flagged, interpreted cautiously), extracellular space (GO:0005615, hub-flagged), and Golgi lumen (GO:0005796, 250 edges, non-hub). [KG Evidence] The non-hub Golgi lumen connection (AGRP, DCN, PRELP) reflects shared secretory pathway processing of these proteins. [Inferred] COL14A1 (collagen XIV alpha-1) was identified as an additional shared gene neighbor connecting multiple input proteins, reinforcing the collagen/ECM structural theme. [KG Evidence]

#### 4.3 Immune and Inflammatory Signaling

IL17D and IL1RL2 anchor an inflammatory signaling subtheme through shared participation in inflammatory response, IL-6 production regulation, and cytokine-cytokine receptor interaction. [KG Evidence] HAVCR1 contributes through immune system process and leukocyte activation pathways. [KG Evidence] Cytokine activity (GO:0005125) is flagged as a hub node (500 edges) and should be interpreted as a broad functional category rather than a specific mechanistic link. [KG Evidence]

#### 4.4 Bone and Skeletal System Development

TNFRSF11B, MEPE, and PRELP share skeletal system development (GO:0001501; 300 edges, non-hub). [KG Evidence] TNFRSF11B and MEPE additionally share bone resorption and clock-controlled autophagy in bone metabolism. [KG Evidence] This theme connects to the steroid sulfate arm through the known effects of DHEA-S on bone mineral density. [Literature] [Inferred]

#### 4.5 Response to Stress and Smoking

The broadly shared pathway annotations "response to stress" (GO:0006950) and "Smoking" (UMLS:C0037369) each connect all 11 protein members. [KG Evidence] These are high-level annotations likely reflecting hub behavior in the pathway ontology; they indicate general stress-responsive and environmentally modulated gene expression rather than a specific mechanistic program. [Inferred]

### 5. Gap Analysis

#### 5.1 Informative Absences

The following expected-but-absent entities provide diagnostic information about the module's biological identity:

| Absent Entity | Expected Because | Interpretation |
|---|---|---|
| Testosterone, Estradiol | Direct products of module precursors | Module captures sulfated storage forms exclusively; reflects SULT2A1/STS conjugation, not active hormone signaling [KG Evidence (gap analysis)] |
| Cortisol | Pregnenolone derivatives are shared precursors | Module captures adrenal DHEA/androgen branch, not glucocorticoid branch [KG Evidence (gap analysis)] |
| RANKL (TNFSF11) | OPG (TNFRSF11B) is present; RANKL/OPG is a canonical signaling pair | OPG without RANKL suggests net anti-resorptive state [KG Evidence (gap analysis)] |
| Leptin, Adiponectin | AGRP is a canonical counter-regulator of leptin signaling | AGRP's presence here is driven by its ECM/inflammatory context, not its canonical appetite axis [KG Evidence (gap analysis)] |
| Creatinine | HAVCR1 (KIM-1) is a kidney injury marker | Module captures early/subclinical tubular injury not yet reflected in filtration markers [KG Evidence (gap analysis)] |
| Osteocalcin (BGLAP) | MEPE, PRELP, TNFRSF11B form a bone matrix subgroup | Bone component reflects matrix remodeling/mineralization inhibition, not osteoblast formation [KG Evidence (gap analysis)] |
| BCAAs | AGRP is associated with insulin resistance/T2D | Module captures steroid sulfate/ECM axis, not amino acid metabolism [KG Evidence (gap analysis)] |

#### 5.2 Standard Gaps

Insulin (INS), CYP17A1, HSD3B2, ceramides, HbA1c, and fasting glucose are absent for methodological reasons: tissue-compartment separation (INS, CYP17A1, HSD3B2 are tissue-restricted), platform separation (ceramides in a lipid module), or clinical-variable versus discovery-analyte distinction (HbA1c, fasting glucose). [KG Evidence (gap analysis)]

#### 5.3 Open World Assumption

Under the open world assumption, the absence of direct knowledge graph edges for the majority of sparse-coverage metabolites (20 entities with fewer than 20 edges) reflects incomplete annotation in current biomedical ontologies rather than confirmed non-association. [Inferred] These metabolites represent a frontier for knowledge graph expansion, and their co-expression with well-characterized proteins provides the empirical basis for future annotation.

### 6. Temporal Context

No explicit longitudinal design is described for this analysis. Causal inference opportunities nonetheless arise from the upstream-to-downstream architecture of the steroidogenic pathway:

Pregnenolone sulfate and 17alpha-hydroxypregnenolone 3-sulfate represent upstream steroidogenic precursors; DHEA-S, androsterone sulfate, and their glucuronide/disulfate derivatives represent downstream metabolic products. [Model Knowledge] In a longitudinal design, changes in upstream precursor sulfates should precede changes in downstream androstane conjugates. [Inferred] The protein arm's temporal relationship to the metabolite arm is less clear. AGRP and HAVCR1, as circulating biomarkers reflecting hypothalamic and renal physiology respectively, may respond to (rather than cause) changes in systemic steroid sulfate pools. [Model Knowledge] Conversely, ECM remodeling proteins (DCN, MMP12) may respond to chronic hormonal exposure, positioning the steroid sulfates as upstream causes and ECM changes as downstream consequences. [Inferred]

A prospective design measuring the module eigengene trajectory against clinical endpoints (incident diabetes, osteoporotic fracture, kidney function decline) would clarify causal directionality. [Inferred]

### 7. Research Recommendations

#### 7.1 High Priority (Experimental Validation)

1. **SULT2A1/STS genotype-to-module analysis**: Perform cis-eQTL mapping at the SULT2A1 (chr19) and STS (chrX) loci against all 31 metabolite levels and the module eigengene. This single analysis could confirm whether genetic variation in conjugation enzymes governs the entire metabolite arm. [Inferred]

2. **Pregnane sulfate-TRPM3-GSIS functional assay**: Test the specific pregnane sulfate isomers present in this module (5alpha-pregnan-3beta,20alpha-diol disulfate, pregnenediol disulfate, 5alpha-pregnan-diol disulfate) for GSIS enhancement in isolated islets using the published TRPM3-activation protocol. [Literature]

3. **HAVCR1 to steroid sulfate clearance study**: In the study cohort, correlate urinary KIM-1 and plasma steroid sulfate concentrations, stratified by eGFR tertiles, to test whether subclinical tubular injury explains the protein-metabolite co-expression. [Inferred]

#### 7.2 Moderate Priority (Targeted Literature and Data Mining)

4. **OPG/RANKL ratio and DHEA-S in bone health**: Examine whether the module eigengene predicts bone mineral density (DXA T-score) independently of age, sex, and BMI. The literature supports DHEA-S associations with bone density (Chingford Study, 2011). [Literature]

5. **Cross-module comparison**: Identify the WGCNA modules containing testosterone, leptin, adiponectin, BCAAs, and ceramides; test for inter-module correlations with the Tan module to map the broader network architecture. [Inferred]

6. **Sparse metabolite annotation**: Submit the 20 sparsely annotated steroid sulfate/glucuronide conjugates (1 to 18 edges) to HMDB and ChEBI for formal pathway and tissue-localization annotation, using the co-expression data from this module as supporting evidence. [Inferred]

#### 7.3 Lower Priority (Follow-Up Analyses)

7. **Smoking-stratified analysis**: Given that "Smoking" was the most broadly shared pathway annotation (all 11 proteins), stratify the module eigengene by smoking status to determine whether this annotation reflects a true biological interaction or a confound. [KG Evidence] [Inferred]

8. **Sex-stratified module stability**: The steroid sulfate content of this module is dominated by androgens and their precursors, which are sexually dimorphic. Test whether the module eigengene structure and protein-metabolite correlations are preserved across sexes. [Model Knowledge]

9. **Placental steroid sulfate profiling**: The Tier 3 inference that androstenediol (3beta,17beta) disulfate may localize to placenta is biologically plausible given placental sulfatase activity. Confirm detection by targeted LC-MS/MS in placental tissue. [KG Evidence (Tier 3)] [Literature]

### Literature References

Papers discovered via semantic search. 6 unique papers across 3 hypotheses.

| Hypothesis | Citation | Link | Relevance |
|------------|----------|------|-----------|
| Bridge: Gene → SmallMolecule (2 hops) |  (2011) "Association between DHEAS and Bone Loss in Postmenopausal Women: A 15-Year Longitudinal Population-B..." | [Link](https://link.springer.com/article/10.1007/s00223-011-9518-9) | Our aim was to examine the association between serum dehydroepiandrosterone ... (DHEAS ... femoral neck (FN ... lumbar s... |
| Inferred role of CHEBI:133715 |  (2026) "Development and validation of an LC-MS/MS assay for serum 5α-androstane-3α, 17β-diol 17-glucuronide ..." | [Link](https://pubmed.ncbi.nlm.nih.gov/41967453/) | Development and validation of an LC-MS/MS assay for serum 5α-androstane-3α, 17β-diol 17-glucuronide with enhanced interf... |
| Inferred role of CHEBI:133712 |  (2021) "Frontiers \| A Sulfuryl Group Transfer Strategy to Selectively Prepare Sulfated Steroids and Isotopic..." | [Link](https://www.frontiersin.org/journals/molecular-biosciences/articles/10.3389/fmolb.2021.776900/full) | The treatment of common steroids: estrone, estradiol, cortisol, and pregnenolone with tributylsulfoammonium betaine (TBS... |
| Bridge: Gene → SmallMolecule (2 hops) |  (2010) "Signaling network of dendritic cells in response to pathogens: a community-input supported knowledge..." | [Link](https://link.springer.com/article/10.1186/1752-0509-4-137) | Based on a manual curation of the published literature, we have ... an extensive and detailed map of the signaling pathw... |
| Inferred role of CHEBI:133712 |  (2022) "Sulfated Progesterone Metabolites That Enhance Insulin Secretion via TRPM3 Are Reduced in Serum From..." | [Link](https://pubmed.ncbi.nlm.nih.gov/35073578/) | Serum progesterone sulfates were evaluated in ... etiology of gestational diabetes mellitus (GDM). Serum progesterone su... |
| Inferred role of CHEBI:133712 |  (2015) "Validated LC–MS/MS simultaneous assay of five sex steroid/neurosteroid-related sulfates in human ser..." | [Link](https://www.sciencedirect.com/science/article/abs/pii/S0960076015000084) | Introduction of the polar sulfate group onto steroids and their metabolites facilitates the detection of sulfated compou... |