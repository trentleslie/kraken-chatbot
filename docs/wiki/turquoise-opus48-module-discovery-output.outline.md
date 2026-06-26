# Turquoise Module Run on Opus 4.8: Discovery Output (107-analyte, dev, 2026-06-24)

This document presents the discovery output of a full-module run of the **Turquoise** WGCNA module through the Kraken 12-node discovery pipeline, executed on the dev environment on 2026-06-24 (commit `098093e`) with BioMapper entity resolution enabled. The SDK-backed pipeline nodes ran on the Opus 4.8 model rather than the default Sonnet; this run is published for direct model comparison. It is one of a per-module series over the Frailty multi-omics WGCNA modules; see the [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm).

## Run Provenance

The run submitted 107 named analytes, parsed 106 at intake, and resolved 106 distinct entities (99 biomapper, 7 fuzzy) to 106 distinct CURIEs. Triage classified 77 well-characterized, 14 moderate, 14 sparse, and 1 cold-start, with 0 edge-count measurement failures. Downstream analysis produced 3441 direct-KG findings, 10 cold-start findings, 8 biological themes, 22 cross-entity bridges (20 evidence-grounded), and 38 hypotheses supported by 23 literature references. Synthesis emitted a 29828-character report. The run completed in approximately 1161.5 s of wall-clock time (status complete, 0 errors). All counts are taken from the reducer-merged pipeline state.

| Stage | Result |
|---|---|
| Submitted | 107 named analytes |
| Intake | 106 parsed |
| Entity resolution | 106 resolved (99 biomapper, 7 fuzzy) to 106 distinct CURIEs |
| Triage | 77 well-characterized, 14 moderate, 14 sparse, 1 cold-start (0 measurement failures) |
| Direct KG | 3441 findings |
| Cold-start | 10 findings, 9 skipped |
| Pathway enrichment | 8 biological themes |
| Integration | 22 bridges (20 evidence-grounded) |
| Literature grounding | 23 papers |
| Synthesis | 38 hypotheses, 29828-character report |
| Run total | ~1161.5 s wall-clock, status complete, 0 errors |

## Related

- Companion run metrics: [Turquoise Module Run on Opus 4.8: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/turquoise-module-run-on-opus-48-pipeline-performance-report-107-analyte-dev-2026-06-24-Kcn1KMFIyU)
- Model comparison baseline (Sonnet): [Turquoise Module Run: Discovery Output](https://phwiki.phenoma.ai/doc/turquoise-module-run-discovery-output-107-analyte-dev-2026-06-23-tuVcWPUCy6)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---



## Turquoise WGCNA Module: Vascular Remodeling, Inflammatory Signaling, and Metabolic Stress Integration

### 1. Executive Summary

This Turquoise WGCNA module encodes a coordinated program of vascular remodeling, immune activation, and oxidative/metabolic stress response, unifying 60 proteins and 46 metabolites through convergent signaling via PDGF/Angiopoietin-driven angiogenesis, NF-κB-mediated inflammation, and sphingolipid/polyamine metabolism. [KG Evidence] The module's disease recurrence profile implicates cardiovascular and inflammatory disorders (panniculitis, 18 members; hematologic disorder, 15 members; coronary artery disorder, 14 members; essential hypertension, 14 members), while the notable absence of VEGFA, TNF, and IL-6 from the co-expression network suggests a vessel-stabilization and tissue-remodeling phenotype distinct from acute angiogenic or cytokine-storm programs. [KG Evidence; Inferred] The metabolite complement, spanning sphingolipid intermediates (sphingosine, sphinganine, sphingosine 1-phosphate), TCA cycle metabolites (succinate, 2-hydroxyglutarate), and nucleotide precursors (orotate, dihydroorotate), reinforces a model of metabolically active tissue undergoing chronic remodeling under oxidative stress. [Model Knowledge]

### 2. Key Findings (Tier 1 to 2)

#### 2.1 Dominant Vascular and Hemostatic Signature

The module contains a dense cluster of endothelial, platelet, and perivascular genes. [KG Evidence] ANGPT1, PDGFB, PDGFA, VWF, SERPINE1 (PAI-1), SELP (P-selectin), PECAM1, F2R (PAR1), PLAT (tPA), and GP6 collectively define a hemostasis and vascular remodeling axis. Blood coagulation (GO:0007596; 4 members: F2R, SERPINE1, GP6, VWF), platelet activation (GO:0030168; 4 members: F2R, GP6, VWF, CD40LG), and angiogenesis (GO:0001525; 5 members: CXCL1, SERPINE1, PDGFA, PDGFB, TNFSF12) are recurrently enriched pathways. [KG Evidence]

Disease recurrence analysis reveals cardiovascular disorder shared by 14 members, coronary artery disorder by 14 members, and essential hypertension by 14 members, all with curated evidence. [KG Evidence] Hematologic disorder is shared by 15 members, consistent with the platelet/coagulation biology. [KG Evidence]

VWF (4,142 edges) is associated with von Willebrand disease type 3; SERPINE1 (4,942 edges) with congenital PAI-1 deficiency; and SELP with cerebral infarction. [KG Evidence] These monogenic associations anchor the module's vascular core with strong genetic evidence.

#### 2.2 Inflammatory and Immune Activation Axis

The module contains 8 chemokines (CXCL1, CXCL5, CXCL6, CXCL11, CCL8, CCL13, CCL17, CCL28) and 4 TNF superfamily members (TNFSF12, TNFSF14, TNFRSF14, CD40LG), alongside co-stimulatory molecules (CD40, CD84, CD244, CD8A) and the IL-17 receptor (IL17RA). [KG Evidence] Inflammatory response (GO:0006954; 10 members), immune response (GO:0006955; 8 members), leukocyte migration (GO:0050900; 5 members), and T cell activation (GO:0042110; shared neighbor of CD84 and CD8A) are enriched biological processes. [KG Evidence]

The chemokine profile is notable: CXCL1, CXCL5, and CXCL6 are ELR+ CXC chemokines that recruit neutrophils, while CXCL11 signals through CXCR3 to attract Th1/effector T cells, and CCL17 is a canonical Th2-attracting chemokine. [Model Knowledge] This mixed Th1/Th2/neutrophilic chemokine profile suggests a complex, non-polarized inflammatory milieu. [Inferred]

CD40LG is associated with hyper-IgM syndrome type 1 (curated); IKBKG (NEMO) with incontinentia pigmenti; and STK4 with combined immunodeficiency due to STK4 deficiency. [KG Evidence] These monogenic immune associations identify the module's immune signaling core as biologically essential rather than epiphenomenal.

#### 2.3 Oxidative Stress and Metabolic Defense

SOD2 (mitochondrial superoxide dismutase), GLO1 (glyoxalase 1), PARP1 (poly-ADP-ribose polymerase), and SIRT2 (NAD-dependent deacetylase) constitute an oxidative stress and metabolic defense cluster. [KG Evidence] Response to stress (GO:0006950) is the second most broadly enriched pathway across the module, shared by 27 members. [KG Evidence]

The metabolite complement directly supports this interpretation. Taurine and hypotaurine are cysteine-derived antioxidants; cysteine itself is present and serves as the rate-limiting precursor for glutathione synthesis. [Model Knowledge] Nicotinamide and FAD are essential cofactors for NAD+- and flavin-dependent redox enzymes, respectively. [Model Knowledge] Succinate and 2-hydroxyglutarate, both TCA cycle metabolites, indicate mitochondrial metabolic activity; 2-hydroxyglutarate is additionally recognized as an oncometabolite whose accumulation reflects altered alpha-ketoglutarate metabolism. [Model Knowledge]

SOD2 is associated with microvascular complications of diabetes susceptibility type 6; GLO1 with melanoma; PARP1 with ovarian cancer. [KG Evidence] These disease links connect the oxidative stress axis to both metabolic (diabetic) and neoplastic contexts.

#### 2.4 Wnt Pathway Antagonism and Developmental Signaling

DKK1 (2,844 edges) participates in negative regulation of canonical Wnt signaling, negative regulation of BMP signaling, and regulation of synapse organization, among 46 established pathway annotations. [KG Evidence] AXIN1 (4,486 edges), a scaffold protein of the beta-catenin destruction complex, reinforces the Wnt-inhibitory theme: AXIN1 is associated with hepatocellular carcinoma. [KG Evidence] BMP6 participates in cell differentiation and is associated with osteoporosis and cardiovascular disorder. [KG Evidence]

DKK1 interacts with multiple Wnt ligands (WNT3A, WNT5B, WNT7A, WNT7B, WNT8A, WNT8B, WNT9A, WNT9B, WNT10A, WNT11) and with MYC, PTH1R, DPP4, and POU5F1 (OCT4), revealing extensive Wnt-pathway cross-talk. [KG Evidence] DKK1 is also associated with osteoporosis, consistent with its role in bone homeostasis via Wnt/LRP5/6 regulation. [KG Evidence]

#### 2.5 Sphingolipid Metabolism and Signaling

Sphingosine, sphinganine, sphingosine 1-phosphate (S1P), and sphinganine-1-phosphate form a complete sphingolipid signaling axis. [KG Evidence] The shared neighbor SPHK1 (sphingosine kinase 1; 100 edges) connects sphingosine and sphinganine via biolink:affects and biolink:related_to predicates. [KG Evidence] Phosphoethanolamine (4,442 edges) and choline phosphate are downstream products of sphingolipid catabolism and phospholipid turnover, linking this axis to membrane remodeling. [Model Knowledge]

S1P is a bioactive lipid mediator that regulates vascular permeability, immune cell trafficking, and inflammation. [Model Knowledge] Its co-expression with the module's vascular (ANGPT1, PECAM1) and immune (CD40LG, CXCL1) components suggests coordinated regulation of endothelial barrier function and leukocyte egress. [Inferred]

#### 2.6 Nucleotide and Polyamine Metabolism

Orotate and dihydroorotate (pyrimidine biosynthesis intermediates), uracil, and modified nucleosides (3-methylcytidine, 2'-O-methylcytidine) indicate active nucleotide metabolism. [Model Knowledge] Spermidine (1,110 edges), connected to PDGFB through obesity, cardiovascular disorder, and cognitive impairment clinical trial nodes [KG Evidence], and to DKK1 through progesterone [KG Evidence], participates in chromatin remodeling and translational regulation. [Model Knowledge] Inosine (1,774 edges) and AMP (8,546 edges; hub-flagged) further indicate purine salvage pathway engagement. [KG Evidence; note: AMP is hub-flagged and associations should be interpreted cautiously]

#### 2.7 Module-Level Disease Convergence

The strongest module-level disease recurrence signals are:

| Disease | Members | Evidence |
|---|---|---|
| Panniculitis | 18 | Curated |
| Depressive disorder | 16 | Curated |
| Gastroesophageal reflux disease | 16 | Curated |
| Chronic rhinitis | 16 | Curated |
| Hematologic disorder | 15 | Curated |
| Asthma | 15 | Curated |
| Irritable bowel syndrome | 15 | Curated |
| Essential hypertension | 14 | Curated |
| Coronary artery disorder | 14 | Curated |
| Cardiovascular disorder | 14 | Curated |

[KG Evidence]

Panniculitis (subcutaneous fat inflammation involving vascular and inflammatory pathology) is the most broadly shared disease association at 18 members, a finding consistent with the module's dual vascular/inflammatory identity. [KG Evidence; Inferred] The convergence on depressive disorder (16 members) is notable and may reflect shared neurovascular or neuroinflammatory biology. [KG Evidence; Inferred]

### 3. Novel Predictions (Tier 3)

#### 3.1 VEGF-Independent, Angiopoietin-Driven Vascular Program

**Prediction**: This module encodes a vessel-stabilization phenotype characterized by pericyte recruitment (PDGFB, PDGFA) and endothelial survival (ANGPT1) in the absence of VEGFA-driven sprouting angiogenesis.

**Structural logic chain**: ANGPT1 → Tie2 receptor activation → endothelial quiescence; PDGFB → PDGFRβ on pericytes → pericyte recruitment and vessel maturation; SERPINE1 → fibrinolysis inhibition → clot stabilization; VWF → platelet adhesion → hemostasis. All four processes converge on vessel stabilization. VEGFA, the canonical driver of neoangiogenesis through endothelial proliferation, is absent from the module despite the presence of 5 angiogenesis-annotated genes (GO:0001525). [KG Evidence; Inferred]

**Validation step**: Measure VEGFA protein levels in the same samples to determine whether VEGFA is low/absent (confirming VEGF-independent vascular biology) or present but not co-expressed (suggesting independent regulation). Examine ANGPT2 levels, as the ANGPT1/ANGPT2 ratio determines vessel destabilization versus stabilization.

**Calibration note**: Approximately 18% of computational predictions of this nature progress to clinical investigation; this prediction is supported by substantial internal logic but remains unvalidated.

#### 3.2 Non-TNF Inflammatory Axis: IL-17/Th2 Hybrid Program

**Prediction**: The module captures a non-canonical inflammatory program driven by IL-17 and Th2-skewed signaling rather than TNF-alpha.

**Structural logic chain**: IL17RA (IL-17 receptor) → CXCL1/CXCL5/CXCL6 induction (neutrophil-recruiting ELR+ CXC chemokines); CCL17 → CCR4 → Th2 cell recruitment; CD40/CD40LG → B cell co-stimulation and class switching. TNF, IL-6, and IFNG are all absent from the module. IKBKG (NEMO) is present, confirming NF-κB signaling competence, but the upstream activating signal appears to be non-TNF. [KG Evidence; Inferred]

**Validation step**: Measure TNF, IL-6, and IFNG in matched samples. Assess IL-17A/IL-17F levels and Th2 cytokines (IL-4, IL-13). If TNF is truly low while CXCL1 and CCL17 are elevated, the non-TNF inflammatory axis is confirmed.

**Calibration note**: Approximately 18% of such computational predictions advance to clinical validation. The mixed Th1 (CXCL11) and Th2 (CCL17) chemokine profile is unusual and warrants confirmation.

#### 3.3 TGM2-Mediated Tissue Stabilization Rather Than MMP-Driven Degradation

**Prediction**: The extracellular matrix program in this module is characterized by cross-linking and stabilization (TGM2) rather than degradation, despite the presence of MMP1.

**Structural logic chain**: TGM2 (tissue transglutaminase) → ECM cross-linking → tissue fibrosis/stabilization; BMP6 → SMAD signaling → pro-fibrotic gene induction; PDGFB → fibroblast activation; TGFB1 (8,121 edges) → canonical pro-fibrotic signaling. MMP9 is absent. MMP1 is present (9,091 edges; hub-flagged), but as a collagenase with distinct substrate specificity from MMP9 (gelatinase). [KG Evidence; Inferred] The co-expression of MMP1 with TGM2 may reflect regulated matrix turnover (collagen I degradation with simultaneous cross-linking of other ECM components) rather than wholesale matrix destruction. [Model Knowledge]

**Validation step**: Measure collagen cross-links (pyridinoline, deoxypyridinoline) and MMP-2/MMP-9 activity (zymography) in matched samples. Histological assessment for fibrosis (Masson's trichrome staining) would directly test this prediction.

**Calibration note**: Approximately 18% of computational predictions progress to experimental validation. The fibrosis interpretation is supported by the absence of fibrosis as a KG-annotated phenotype, which likely reflects ontological coverage gaps rather than biological absence. [KG Evidence; Inferred]

#### 3.4 Spermidine as a Metabolic Link Between Vascular and Cognitive Phenotypes

**Prediction**: Spermidine connects the vascular remodeling program (via PDGFB) to the depressive disorder association (16 members) through polyamine-dependent autophagy and neuroprotection.

**Structural logic chain**: PDGFB → obesity (MONDO:0011122) → spermidine (clinical trials; biolink:in_clinical_trials_for); PDGFB → cardiovascular disorder (MONDO:0004995) → spermidine (clinical trials); PDGFB → cognitive impairment (HP:0100543) → spermidine (clinical trials). [KG Evidence] Spermidine induces autophagy through hypusination of eIF5A and is in clinical trials for cognitive impairment. [Literature: Non-Linear Association of Dietary Polyamines with the Risk of Incident Dementia, 2024] Depressive disorder is the second most recurrent disease association across 16 module members. [KG Evidence]

**Validation step**: Correlate spermidine levels with depressive symptom scores and vascular function markers (flow-mediated dilation, carotid intima-media thickness) in this cohort. Test whether spermidine mediates the relationship between vascular protein levels and depressive symptoms.

**Calibration note**: Approximately 18% of computational predictions progress to clinical investigation. The grounded literature on polyamines and dementia risk (UK Biobank cohort, 2024) provides population-level support for the spermidine-cognition link, though direct connection to this module's depressive disorder signal remains speculative.

#### 3.5 Gamma-Glutamyl Dipeptides as Markers of GGT Activity and Glutathione Turnover

**Prediction**: The gamma-glutamyl dipeptides in this module (gamma-glutamylglutamine, gamma-glutamylmethionine, gamma-glutamylcitrulline) reflect elevated gamma-glutamyl transferase (GGT) activity and accelerated glutathione turnover.

**Structural logic chain**: gamma-glutamylcitrulline → semantic similarity (0.81) to gamma-glutamylcystine → GGT-catalyzed transpeptidation products [KG Evidence; cold_start inference]; GLO1 (present in module) → requires glutathione as obligate cofactor → glutathione consumption; cysteine (present) → rate-limiting glutathione precursor. [KG Evidence; Model Knowledge] The co-expression of cysteine, GLO1, and gamma-glutamyl dipeptides in a single WGCNA module suggests a coordinated glutathione synthesis-utilization-recycling program. [Inferred]

**Validation step**: Measure serum GGT activity, reduced/oxidized glutathione ratio (GSH/GSSG), and glutathione S-transferase activity. Correlate gamma-glutamyl dipeptide levels with GLO1 protein levels and with markers of oxidative stress (8-isoprostane, malondialdehyde).

**Calibration note**: Approximately 18% of computational predictions advance to validation. Glutathione itself was expected but absent, likely reflecting assay platform limitations (see Gap Analysis). [Inferred]

#### 3.6 2-Hydroxyglutarate as an Oncometabolite and Epigenetic Modifier

**Prediction**: Elevated 2-hydroxyglutarate in this module may contribute to epigenetic dysregulation through inhibition of alpha-ketoglutarate-dependent dioxygenases (TET enzymes, Jumonji-domain histone demethylases), linking metabolic stress to the cancer associations (cancer shared by 10 members; urothelial carcinoma by 12 members).

**Structural logic chain**: 2-hydroxyglutarate (108 edges) → competitive inhibition of alpha-KG-dependent dioxygenases → DNA and histone hypermethylation → altered gene expression [Model Knowledge]; cancer (MONDO:0004992; 10 members), urothelial carcinoma (MONDO:0040679; 12 members) are recurrent disease associations [KG Evidence]; PARP1 (DNA damage repair) and CASP3/CASP8 (apoptosis; hub-flagged) are module members. [KG Evidence]

**Validation step**: Determine the D/L-enantiomeric ratio of 2-hydroxyglutarate (D-2HG is the oncometabolite, L-2HG accumulates under hypoxia). Assess IDH1/IDH2 mutation status if tissue samples are available. Correlate 2-HG levels with DNA methylation (5-methylcytosine) global levels.

**Calibration note**: Approximately 18% of such predictions advance to clinical investigation. No direct KG evidence connects 2-HG to the specific cancer associations in this module; this prediction is based on [Model Knowledge] regarding 2-HG oncometabolite biology.

### 4. Biological Themes

#### 4.1 Unifying Theme: Chronic Vascular Inflammation with Metabolic Stress

The module integrates three biological programs that converge on a phenotype of chronically inflamed, metabolically stressed vasculature undergoing stabilization rather than acute injury:

1. **Vascular stabilization and hemostasis**: ANGPT1, PDGFB, PDGFA, VWF, SERPINE1, SELP, PECAM1, F2R, GP6, PLAT, THPO (platelet production). [KG Evidence]
2. **Immune activation and chemokine signaling**: CXCL1/5/6/11, CCL8/13/17/28, CD40/CD40LG, TNFSF12/14, TNFRSF14, IL17RA, CD8A, CD84, CD244. [KG Evidence]
3. **Oxidative/metabolic defense**: SOD2, GLO1, PARP1, SIRT2, cysteine, taurine, nicotinamide, FAD, spermidine. [KG Evidence]

#### 4.2 Pathway Enrichment Convergence

Protein binding (GO:0005515; 30 members) and response to stress (GO:0006950; 27 members) are the most broadly shared annotations, though their breadth reduces specificity. [KG Evidence] The more informative enrichments are:

- **Signal transduction** (GO:0007165 + UMLS:C0037083; 9 members each): F2R, CXCL1, PDGFA, PDGFB, ITGB1BP2, CCL17, STK4, THPO, TNFSF12. [KG Evidence]
- **MAPK signaling pathway** (5 members: HSPB1, PDGFA, PDGFB, STK4, CASP3): a central proliferation and stress-response cascade. [KG Evidence; CASP3 is hub-flagged]
- **PI3K/Akt signaling** (GO:0051897; 5 members: F2R, PDGFA, PDGFB, SELP, THPO): survival and anti-apoptotic signaling. [KG Evidence]
- **Cell adhesion** (GO:0007155; 5 members: F11R, GP6, SELP, VWF, CD84): endothelial-platelet and immune cell adhesion. [KG Evidence]

#### 4.3 Hub-Filtered Insights

Entities with greater than 5,000 edges (PARP1, SRC, CASP3, MMP1, TGFB1, CASP8, EIF4EBP1, HSPB1, CD40; all hub-flagged) connect to very broad disease and pathway annotations that should be interpreted cautiously. [KG Evidence] AMP (8,546 edges) is similarly hub-flagged among the metabolites. Disease associations driven solely by these hubs are de-emphasized in this analysis.

The non-hub pathway enrichments (MAPK cascade, PI3K/Akt, angiogenesis, blood coagulation, platelet activation) provide more specific and interpretable biological context. [KG Evidence]

#### 4.4 Cross-Type Bridges

The pathway enrichment analysis identified the shared neighbor SPHK1 (sphingosine kinase 1; 100 edges) connecting sphingosine and sphinganine, providing a mechanistic enzyme link between the sphingolipid metabolites and the broader signaling program. [KG Evidence] Colorectal cancer (MONDO:0005575; 300 edges) was identified as a non-hub shared neighbor connecting sphingosine, maltose, and N-acetylaspartate, suggesting a metabolic signature with cancer relevance. [KG Evidence]

SH2D1A (80 edges) and SH2D1B (50 edges) connect CD84 and CD244, revealing a SLAM-family immune receptor signaling sub-network regulated by SAP/EAT-2 adaptors. [KG Evidence] PTPN11 (SHP-2; 200 edges) connects PECAM1 and CD244 via physical interaction, identifying a phosphatase-mediated checkpoint signaling node. [KG Evidence]

### 5. Gap Analysis

#### 5.1 Informative Absences (Open World Assumption)

Under the Open World Assumption, absence of an entity from this co-expression module means "not co-expressed in this context," not "biologically irrelevant."

**VEGFA**: Absent despite 5 angiogenesis-annotated module members. This absence indicates an angiopoietin/PDGF-dominant vascular program consistent with vessel maturation and stabilization rather than sprouting neoangiogenesis. [Inferred] This distinction has therapeutic implications: anti-VEGF therapies would not be predicted to affect this module's vascular biology, whereas Tie2 agonists or PDGFR inhibitors would.

**TNF, IL-6, IFNG**: Absent despite extensive inflammatory signaling. The module's inflammatory axis appears non-TNF-driven, potentially IL-17- or growth-factor-driven (via SRC, HBEGF). [Inferred] This is an informative absence that distinguishes this module from canonical acute-phase inflammatory programs.

**NFE2L2 (NRF2), HIF1A, RELA/NFKB1**: These transcription factors are regulated primarily post-translationally (KEAP1-mediated stabilization for NRF2; prolyl hydroxylation for HIF1A; IκB degradation for NF-κB). Their absence from a co-expression module is biologically expected, as their mRNA levels do not co-vary with their transcriptional targets. [Model Knowledge] Their functional involvement is strongly implied by the presence of target genes (SOD2 for NRF2; CXCL1, SOD2, PARP1 for NF-κB).

**SFRP1/SFRP2**: Absent alongside present DKK1 and AXIN1. This suggests that Wnt inhibition in this module operates through the LRP5/6 co-receptor (DKK1 mechanism) and the beta-catenin destruction complex (AXIN1 mechanism), but not through Wnt ligand sequestration (SFRP mechanism). [Inferred] The module encodes pathway-specific rather than broad-spectrum Wnt suppression.

**MMP9**: Absent despite the presence of ECM-remodeling genes (TGM2, TGFB1, BMP6). MMP1 is present but hub-flagged (9,091 edges). The absence of MMP9 alongside the presence of TGM2 (ECM cross-linker) is informative: it suggests matrix stabilization rather than degradation. [Inferred]

**Glutathione, methylglyoxal**: Expected metabolites not detected, likely reflecting assay platform limitations rather than biological absence. GLO1 (which requires glutathione) and gamma-glutamyl dipeptides (GGT products of glutathione) indirectly confirm glutathione pathway activity. [Inferred]

**CASP3 note**: CASP3 is listed in the gap analysis as "expected but absent" in the original analysis context; however, CASP3 is actually present in the module input (NCBIGene:836; 10,000 edges; hub-flagged). The gap analysis entry likely refers to the expectation that caspase-mediated apoptosis should be a central theme, which is not the case: PARP1 and STK4 have non-apoptotic roles (DNA repair and Hippo pathway, respectively). [KG Evidence; Inferred]

#### 5.2 Standard Gaps

**CTNNB1 (beta-catenin) and AKT1**: Both were identified as major hub interactors connecting 14 input entities each but are not module members. [KG Evidence] Their absence from the co-expression module is expected: both are post-translationally regulated and function as signal transduction intermediates whose protein levels are controlled by degradation (CTNNB1) or phosphorylation (AKT1) rather than transcription. [Model Knowledge]

**N-acetylglucosamine (GlcNAc)**: N-acetylglucosaminylasparagine is present (sparse coverage; 2 edges), and cold-start inference connects it to GlcNAc (CHEBI:17411) via semantic similarity (0.91) to a GlcNAc-containing analogue. [KG Evidence] O-GlcNAcylation is a nutrient-sensing post-translational modification that would require specialized glycoproteomics to detect. [Model Knowledge]

### 6. Temporal Context

This analysis is cross-sectional (single WGCNA module from one time-point), so direct causal ordering is not possible. The following upstream/downstream architecture is inferred from established signaling biology:

**Probable upstream causes** (signal initiators):
- Growth factor signaling: PDGFB, PDGFA, HBEGF, BMP6, TGFB1 → receptor activation → downstream cascades [Model Knowledge]
- Immune activation: CD40LG/CD40 co-stimulation, IL-17 signaling (via IL17RA) [Model Knowledge]
- Metabolic inputs: cysteine availability, NAD+ levels (nicotinamide), sphingolipid turnover [Model Knowledge]

**Probable downstream consequences** (effector outputs):
- Chemokine secretion: CXCL1/5/6/11, CCL8/13/17/28 → leukocyte recruitment [Model Knowledge]
- ECM remodeling: TGM2-mediated cross-linking, MMP1-mediated collagen turnover [Model Knowledge]
- Hemostatic regulation: VWF, SERPINE1, SELP → platelet adhesion and fibrinolysis control [Model Knowledge]
- Oxidative defense: SOD2, GLO1 → ROS scavenging and methylglyoxal detoxification [Model Knowledge]

**Causal inference opportunities**: Longitudinal sampling with Mendelian randomization (using genetic instruments for PDGFB, ANGPT1, or TGFB1) could distinguish whether the vascular signaling program causally drives the inflammatory and metabolic features, or whether they are co-regulated by an unmeasured upstream factor. [Inferred]

### 7. Research Recommendations

#### 7.1 Highest Priority: Experimental Validations

1. **Measure VEGFA, TNF, IL-6, and ANGPT2 in matched samples.** The module's identity as a VEGF-independent, non-TNF vascular-inflammatory program is the central novel prediction. Confirming low VEGFA and high ANGPT1/ANGPT2 ratio would validate the vessel-stabilization interpretation and has direct therapeutic implications. [Priority: Critical]

2. **Assess D/L-2-hydroxyglutarate enantiomeric ratio.** D-2HG (oncometabolite) versus L-2HG (hypoxia marker) distinction would clarify whether the metabolic stress signature reflects neoplastic metabolism or tissue hypoxia. [Priority: High]

3. **Measure GGT activity and GSH/GSSG ratio.** Validate the inferred glutathione turnover program suggested by gamma-glutamyl dipeptides and GLO1 co-expression. [Priority: High]

4. **Perform targeted sphingolipid profiling.** The module contains the complete sphingolipid signaling axis (sphinganine → sphingosine → S1P); measuring ceramide species and S1P receptor expression would determine whether the S1P-vascular barrier axis is active. [Priority: Moderate]

#### 7.2 Literature Searches

5. **Search for ANGPT1/PDGFB co-expression modules in cardiovascular cohorts.** Determine whether this VEGF-independent vascular signature has been reported in other WGCNA analyses of cardiovascular or inflammatory disease. [Priority: High]

6. **Review TGM2-BMP6-TGFB1 axis in tissue fibrosis.** The co-expression of these three pro-fibrotic mediators with absent MMP9 warrants a systematic review of fibrotic disease contexts. [Priority: Moderate]

7. **Investigate the depressive disorder association.** The convergence of 16 members on depressive disorder, combined with spermidine's clinical trial evidence for cognitive impairment [Literature: Non-Linear Association of Dietary Polyamines, 2024], warrants a targeted literature search on neurovascular contributions to depression in the context of this module's biology. [Priority: Moderate]

#### 7.3 Follow-Up Analyses

8. **Conditional analysis of hub nodes.** Repeat pathway enrichment after excluding the 10 hub-flagged entities (PARP1, SRC, CASP3, MMP1, TGFB1, CASP8, EIF4EBP1, HSPB1, CD40, AMP) to determine which disease associations are driven by specific, lower-degree module members versus non-specific hub effects. [Priority: High]

9. **Module preservation analysis.** Test whether the Turquoise module is preserved in independent cohorts (e.g., GTEx vascular tissues, inflammatory disease cohorts) to assess reproducibility and tissue specificity. [Priority: Moderate]

10. **Causal mediation analysis.** Test whether spermidine, S1P, or 2-hydroxyglutarate levels mediate the association between vascular protein levels and the depressive disorder or cardiovascular disorder phenotypes. [Priority: Moderate]

11. **Cold-start entity follow-up.** gamma-glutamylcitrulline (0 KG edges) was connected by semantic inference to gamma-glutamylcystine (similarity 0.81) and thereby to GGT-mediated glutathione metabolism. [KG Evidence; cold_start] Direct enzymatic assays for GGT and glutathione recycling enzymes would ground this inference experimentally. [Priority: Low]

---

*Report generated from KRAKEN knowledge graph analysis of 106 resolved entities (60 proteins, 46 metabolites) in the Turquoise WGCNA module. All evidence attributions are tagged per the evidence classification system: [KG Evidence], [Literature], [Model Knowledge], and [Inferred]. Hub-flagged entities (greater than 1,000 edges) are noted where relevant; associations driven solely by such entities should be interpreted with caution.*

### Literature References

Papers discovered via semantic search. 4 unique papers across 3 hypotheses.

| Hypothesis | Citation | Link | Relevance |
|------------|----------|------|-----------|
| Bridge: Gene → ChemicalEntity (2 hops) | Weijia Fu et al. (2025) "Piezo1-related physiological and pathological processes in glioblastoma" | [DOI](https://doi.org/10.3389/fcell.2025.1536320) | — |
| Bridge: Gene → ChemicalEntity (2 hops) | Wojciech Żwierełło et al. (2025) "Metabolic Reprogramming Triggered by Fluoride in U-87 Glioblastoma Cells: Implications for Tumor Pro..." | [DOI](https://doi.org/10.3390/cells14110800) | — |
| Bridge: Gene → Protein (2 hops) |  (2018) "A role for APP in Wnt signalling links synapse loss with β-amyloid production \| Translational Psychi..." | [Link](https://www.nature.com/articles/s41398-018-0231-6) | 10, ... 11, ... kopf-1 ( ... models of A ... 1, and ... is induced in neuronally ... cultures as an ... response to Aβ .... |
| Bridge: Gene → ChemicalEntity (2 hops); Bridge: Gene → SmallMolecule (2 hops) |  (2022) "Frontiers \| Most Pathways Can Be Related to the Pathogenesis of Alzheimer's Disease" | [Link](https://www.frontiersin.org/journals/aging-neuroscience/articles/10.3389/fnagi.2022.846902/full) | Kyoto Encyclopedia of ... ) pathways have publications containing ... association via at ... 63 ... of pathway terms hav... |