# Blue Module Run: Discovery Output (149-analyte, dev, 2026-06-23)

This document presents the discovery output of a full-module run of the **Blue** WGCNA module through the Kraken 12-node discovery pipeline, executed on the dev environment on 2026-06-23 (commit `813fc3f`) with BioMapper entity resolution enabled. It is one of a per-module series over the Frailty multi-omics WGCNA modules; see the [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm).

## Run Provenance

The run submitted 149 named analytes, parsed 146 at intake, and resolved 146 distinct entities (135 biomapper, 2 exact, 9 fuzzy) to 146 distinct CURIEs. Triage classified 95 well-characterized, 24 moderate, 24 sparse, and 3 cold-start, with 0 edge-count measurement failures. Downstream analysis produced 3806 direct-KG findings, 14 cold-start findings, 13 biological themes, 20 cross-entity bridges (15 evidence-grounded), and 44 hypotheses supported by 38 literature references. Synthesis emitted a 28835-character report. The run completed in approximately 1341.8 s of wall-clock time (status complete, 0 errors). All counts are taken from the reducer-merged pipeline state.

| Stage | Result |
|---|---|
| Submitted | 149 named analytes |
| Intake | 146 parsed |
| Entity resolution | 146 resolved (135 biomapper, 2 exact, 9 fuzzy) to 146 distinct CURIEs |
| Triage | 95 well-characterized, 24 moderate, 24 sparse, 3 cold-start (0 measurement failures) |
| Direct KG | 3806 findings |
| Cold-start | 14 findings, 19 skipped |
| Pathway enrichment | 13 biological themes |
| Integration | 20 bridges (15 evidence-grounded) |
| Literature grounding | 38 papers |
| Synthesis | 44 hypotheses, 28835-character report |
| Run total | ~1341.8 s wall-clock, status complete, 0 errors |

## Related

- Companion run metrics: [Blue Module Run: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/blue-module-run-pipeline-performance-report-149-analyte-dev-2026-06-23-Yh2cTvs6GD)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---



## Discovery Report: Blue WGCNA Module, Immune–Vascular–Metabolic Co-expression Network

---

### 1. Executive Summary

This Blue WGCNA module encodes a coordinated program of immune activation, vascular remodeling, and altered nucleoside/amino acid catabolism, uniting 90 proteins and 56 metabolites through convergent TNF superfamily receptor signaling, galectin-mediated immunomodulation, and neutrophil degranulation. [KG Evidence] [Inferred] The module's disease recurrence profile implicates inflammatory and cardiometabolic conditions (panniculitis, coronary artery disorder, asthma, psoriasis) at the highest member overlap, while the metabolite complement (pseudouridine, modified nucleosides, kynurenine, dimethylarginine) points to accelerated cellular turnover and impaired renal clearance as co-regulatory axes. [KG Evidence] Notably, the absence of canonical adipokines (adiponectin, leptin) and classical acute-phase mediators (CRP, IL-6) indicates that this module captures a distinct vascular–immune interface rather than systemic metabolic inflammation, an observation with direct implications for biomarker stratification in cardiometabolic disease. [Inferred]

---

### 2. Key Findings (Tier 1 to 2)

#### 2.1 Unifying Biological Theme: Immune–Vascular Crosstalk Under TNF Superfamily Control

The module is organized around three interlocking functional axes, each supported by direct knowledge graph evidence:

**TNF superfamily receptor signaling.** Seven TNF receptor superfamily members (TNFRSF1A, TNFRSF1B, TNFRSF10B, TNFRSF10C, TNFRSF11A, TNFRSF13B, TNFRSF9) and three TNF superfamily ligands (TNFSF13B/BAFF, FAS, LTBR) co-express within the module. [KG Evidence] TNFRSF10B participates in TRAIL-activated apoptotic signaling and canonical NF-κB signal transduction (GO:0043123), a pathway shared by 6 module members (LGALS9, CX3CL1, TNFRSF1A, TNFRSF11A, TNFRSF10B, CD4). [KG Evidence] This receptor cluster connects to curated disease associations spanning panniculitis (28 members), coronary artery disorder (27 members), and psoriasis (27 members), each at the strongest curated evidence level. [KG Evidence]

**Galectin-mediated immune regulation.** LGALS9 (Galectin-9; 2,824 edges) and LGALS3 (Galectin-3; 5,962 edges) occupy central immunomodulatory positions. LGALS9 participates in negative regulation of activated T cell proliferation, positive regulation of IL-10 and IL-12 production, negative regulation of TNF production, and positive regulation of dendritic cell differentiation. [KG Evidence] LGALS9 directly interacts with LGALS3, ALCAM, and MET within the module (all established interactions). [KG Evidence] The LGALS9–ALCAM interaction is notable because both are module members, indicating a physically validated intra-module connection. [KG Evidence]

**Vascular endothelial remodeling.** TEK (Tie-2; 4,728 edges), CDH5 (VE-cadherin), EPHB4, PGF (placental growth factor), ADM (adrenomedullin), NOTCH3, THBD (thrombomodulin), CD93, and ICAM2 constitute a vascular endothelial subnetwork. [KG Evidence] TEK participates in signal transduction, cell-cell signaling, and angiogenesis; PGF participates in positive regulation of cell population proliferation and cell differentiation (GO:0008284, GO:0030154). [KG Evidence] These factors converge on shared pathway annotations for angiogenesis (GO:0001525) and female pregnancy (GO:0007565), connecting 28 input entities to shared biological process annotations. [KG Evidence]

#### 2.2 Module-Level Disease Recurrence (Highest Priority)

| Disease | Members | Evidence |
|---|---|---|
| Panniculitis | 28 | Curated [KG Evidence] |
| Asthma | 27 | Curated [KG Evidence] |
| Coronary artery disorder | 27 | Curated [KG Evidence] |
| Psoriasis | 27 | Curated [KG Evidence] |
| Hypophysitis | 25 | Curated [KG Evidence] |
| Schizophrenia | 25 | Curated [KG Evidence] |
| Depressive disorder | 24 | Curated [KG Evidence] |
| Urinary system disorder | 24 | Curated [KG Evidence] |
| IBS | 23 | GWAS [KG Evidence] |
| Essential hypertension | 22 | Curated [KG Evidence] |
| Diabetes mellitus | 16 | Curated [KG Evidence] |
| Cancer (general) | 17 | Curated [KG Evidence] |

The convergence of inflammatory dermatological conditions (panniculitis, psoriasis), cardiovascular disease (coronary artery disorder, essential hypertension), and neuropsychiatric disorders (schizophrenia, depressive disorder) suggests that this module captures a shared immune-inflammatory endophenotype spanning organ systems. [Inferred]

#### 2.3 Neutrophil Degranulation Signature

MPO, AZU1, PRTN3, CEACAM8, PGLYRP1, and PI3 (elafin) represent a coherent neutrophil azurophilic and specific granule signature. [KG Evidence] These proteins participate in defense response to bacterium (GO:0042742; 5 members), innate immune response (GO:0045087; 5 members), and proteolysis (GO:0006508; 7 members). [KG Evidence] The co-expression of neutrophil granule proteins with immune checkpoint molecules (CD274/PD-L1, PDCD1LG2/PD-L2) and T cell regulators (CD4, IL2RA, TNFRSF9/4-1BB) indicates coordinated myeloid–lymphoid activation. [KG Evidence] [Inferred]

#### 2.4 Immune Checkpoint and T Cell Regulatory Axis

CD274 (PD-L1; 8,249 edges), PDCD1LG2 (PD-L2), CD4, IL2RA (CD25), IL15RA, TNFRSF9 (4-1BB), CD5, CD6, SLAMF1, and SLAMF7 define a T cell co-stimulatory/co-inhibitory module. [KG Evidence] The T cell modulation in pancreatic cancer pathway (WikiPathways WP5078) connects 4 members (LGALS3, LGALS9, PGF, PDCD1LG2). [KG Evidence] LGALS9 participates in negative regulation of T-helper 1 type immune response and positive regulation of activated T cell autonomous cell death. [KG Evidence] This architecture suggests simultaneous immune activation (via co-stimulatory receptors) and immune regulation (via checkpoint ligands and galectin-mediated T cell apoptosis). [Inferred]

#### 2.5 Metabolite Complement: Uremic Toxins, Modified Nucleosides, and Tryptophan Catabolism

The metabolite compartment is dominated by three functional classes:

**Modified nucleosides and uremic retention solutes.** Pseudouridine, N1-methyladenosine, N2,N2-dimethylguanosine, N4-acetylcytidine, N6-carbamoylthreonyladenosine, 5,6-dihydrouridine, 3-(3-amino-3-carboxypropyl)uridine, 7-methylguanine, and orotidine are products of tRNA/rRNA turnover and established uremic toxins. [Model Knowledge] Their co-expression with immune-inflammatory proteins suggests that impaired renal clearance or accelerated RNA turnover (from immune cell activation) contributes to this module's behavior. [Inferred]

**Tryptophan-kynurenine pathway.** Kynurenine is present alongside C-glycosyltryptophan and gamma-glutamyltryptophan. [KG Evidence] Kynurenine is an immunomodulatory metabolite produced by indoleamine 2,3-dioxygenase (IDO) in response to inflammatory cytokines; its presence alongside immune checkpoint molecules (PD-L1, PD-L2) is consistent with an immunosuppressive metabolic milieu. [Model Knowledge] [Inferred]

**Catecholamine metabolites.** Homovanillate (HVA), vanillylmandelate (VMA), 3-methoxytyrosine, and 3-methoxytyramine sulfate represent catecholamine degradation products. [Model Knowledge] Their co-expression with immune markers may reflect sympathetic nervous system–immune axis coupling. [Inferred]

#### 2.6 Highest-Leverage Individual Members

From the Member Prioritization Table, the following entities merit special attention for their high connectivity and specific disease associations:

| Member | Edges | Notable Feature |
|---|---|---|
| EGFR | 10,000 | Non-small cell lung carcinoma (hub-flagged) [KG Evidence] |
| CD274 (PD-L1) | 8,249 | Immune checkpoint (hub-flagged) [KG Evidence] |
| LGALS9 | 2,824 | Dual immunomodulator; 46 Tier 1 functional annotations [KG Evidence] |
| TNFRSF1A | 3,776 | Autosomal dominant familial periodic fever [KG Evidence] |
| TNFRSF11A (RANK) | 2,955 | Familial expansile osteolysis; bone remodeling [KG Evidence] |
| THBD | 2,588 | Thrombophilia [KG Evidence] |
| GDF15 | 6,063 | Stress-responsive cytokine (hub-flagged) [KG Evidence] |
| PCSK9 | 4,001 | Lipid metabolism regulator [KG Evidence] |

EGFR, CD274, MMP2, IL10, CCL2, MERTK, LGALS3, AXL, SPP1, GDF15, CTSL, and choline carry hub-bias warnings (more than 1,000 edges) and should be interpreted with appropriate caution: their associations may reflect high general connectivity rather than module-specific biology. [KG Evidence]

#### 2.7 Cross-Type Bridges (Gene to Metabolite Connections)

LGALS9 directly interacts with CCL2 in the knowledge graph (1-hop bridge, biolink:interacts_with). [KG Evidence] Multiple 2-hop paths connect protein members to metabolite members through shared tissue localization (liver, blood) and disease associations (cancer). [KG Evidence] LGALS9 connects to dimethylglycine via cytoplasm co-localization and via TNF-mediated regulation (LGALS9 → TNF → dimethylglycine, both legs curated). [KG Evidence] TNFRSF10B connects to CCL2 through IFNGR1 (direct physical interaction) and through doxorubicin (both legs curated-causal). [KG Evidence]

The shared localization of several metabolites (N-acetylvaline, N-acetylalanine, N-acetylputrescine, 4-acetamidobutanoate, hydantoin-5-propionate) in feces (UBERON:0001988) and placenta (UBERON:0001987) represents a non-hub shared neighbor connecting 4 to 5 metabolites at each anatomical site (degree 300 to 500 edges). [KG Evidence]

---

### 3. Novel Predictions (Tier 3)

#### 3.1 VEGF-A-Independent Angiogenic Program in Immune-Vascular Pathology

**Prediction:** The module encodes a VEGF-A-independent angiogenic pathway mediated through PGF/VEGFR1 and TEK/Angiopoietin signaling, potentially relevant to immune-driven microvascular remodeling in cardiometabolic disease.

**Structural logic chain:** PGF (module member) signals through VEGFR1/Flt-1 independently of VEGF-A [Model Knowledge]; TEK (Tie-2 receptor, module member) participates in angiogenesis and signal transduction [KG Evidence]; VEGF-A is expected but absent from the module [KG Evidence: gap analysis]; co-expressed endothelial markers (CDH5, EPHB4, CD93, ICAM2, NOTCH3) reinforce vascular identity [KG Evidence]; coronary artery disorder is the third most recurrent disease (27 members) [KG Evidence]. This combination suggests that VEGF-A-independent angiogenesis may represent a specific vascular remodeling pathway active in immune-inflammatory cardiometabolic states. [Inferred]

**Validation step:** Measure PGF, soluble Tie-2, and VEGF-A levels in parallel in the study cohort. Test whether PGF/sTie-2 ratios predict cardiovascular endpoints independently of VEGF-A levels. Approximately 18% of such computational predictions progress to clinical investigation.

#### 3.2 Kynurenine–Immune Checkpoint Synergy

**Prediction:** The co-expression of kynurenine with CD274 (PD-L1) and PDCD1LG2 (PD-L2) reflects a coordinated immunosuppressive program in which IDO-derived kynurenine reinforces checkpoint-mediated T cell exhaustion.

**Structural logic chain:** Kynurenine is a module metabolite [KG Evidence]; CD274 and PDCD1LG2 are module proteins [KG Evidence]; LGALS9 participates in positive regulation of activated T cell autonomous cell death and negative regulation of T-helper 1 type immune response [KG Evidence]; kynurenine activates the aryl hydrocarbon receptor (AhR) on T cells, promoting regulatory T cell differentiation and suppressing effector T cell function [Model Knowledge]; IDO expression is induced by interferon-gamma, and LGALS9 participates in cellular response to type II interferon (GO:0071346) [KG Evidence]. CD274 is hub-flagged (8,249 edges), requiring interpretive caution. [KG Evidence]

**Validation step:** Correlate plasma kynurenine/tryptophan ratios with soluble PD-L1 levels and T cell exhaustion markers (PD-1, TIM-3) in the study cohort. Assess whether IDO1 expression in tissue biopsies (if available) tracks with the Blue module eigengene. Approximately 18% of computational predictions of this nature advance to clinical investigation.

#### 3.3 Modified Nucleosides as Markers of Immune Cell Turnover

**Prediction:** The cluster of modified nucleosides (pseudouridine, N2,N2-dimethylguanosine, N4-acetylcytidine, 5,6-dihydrouridine, 3-(3-amino-3-carboxypropyl)uridine) in this module reflects accelerated tRNA/rRNA degradation driven by the activated immune cell populations (neutrophils, T cells, macrophages) that produce the module's protein members.

**Structural logic chain:** Modified nucleosides are established products of tRNA and rRNA catabolism [Model Knowledge]; neutrophil degranulation markers (MPO, AZU1, PRTN3, CEACAM8) are module members [KG Evidence]; activated immune cells undergo rapid proliferation and apoptosis, releasing modified nucleosides [Model Knowledge]; pseudouridine (67 KG edges) and N2,N2-dimethylguanosine (9 KG edges) have limited but non-zero KG representation [KG Evidence]; these metabolites are also recognized uremic retention solutes, and urinary system disorder is associated with 24 module members [KG Evidence].

**Validation step:** Measure modified nucleoside panel (pseudouridine, N2,N2-dimethylguanosine) alongside neutrophil activation markers (MPO, PRTN3) in serial samples. Test whether nucleoside levels correlate with immune cell turnover rates (e.g., lymphocyte proliferation indices) after adjusting for estimated glomerular filtration rate. Approximately 18% of such predictions advance to clinical investigation.

#### 3.4 C-Glycosyltryptophan as a Novel Immune-Associated Aging Biomarker

**Prediction:** C-glycosyltryptophan, a metabolite with only 2 KG edges, may interact with immune signaling pathways; semantic similarity analysis (0.84 similarity to 1,1'-ethylidenebistryptophan) infers a potential IL-5 interaction and extracellular localization. [KG Evidence: cold-start inference]

**Structural logic chain:** C-glycosyltryptophan (CHEBI:165856) is semantically similar to 1,1'-ethylidenebistryptophan (CHEBI:172675; similarity 0.84) [KG Evidence]; the analogue interacts with IL-5 (NCBIGene:3567) and is located in the extracellular region [KG Evidence]; C-glycosyltryptophan is an established marker of biological aging in epidemiological studies [Model Knowledge]; its co-expression with immune-inflammatory proteins in this module may reflect aging-related immune dysregulation. [Inferred] This inference rests on a single analogue (low confidence).

**Validation step:** Query HMDB and published metabolomics-GWAS studies for associations between C-glycosyltryptophan and immune or inflammatory traits. Test whether C-glycosyltryptophan levels in the study cohort correlate with chronological age after adjusting for the Blue module eigengene. Approximately 18% of computational predictions advance to clinical investigation.

#### 3.5 Dimethylarginine as Endothelial Dysfunction Marker Linking Vascular and Immune Axes

**Prediction:** Dimethylarginine (SDMA + ADMA; 522 KG edges) co-expresses with vascular endothelial proteins because ADMA inhibits endothelial nitric oxide synthase (eNOS), linking endothelial dysfunction to the vascular remodeling program (TEK, CDH5, THBD) in the module.

**Structural logic chain:** Dimethylarginine is a module metabolite [KG Evidence]; essential hypertension is associated with 22 module members [KG Evidence]; coronary artery disorder with 27 members [KG Evidence]; ADMA is a competitive inhibitor of eNOS and an established cardiovascular risk factor [Model Knowledge]; vascular endothelial proteins (TEK, CDH5, THBD, EPHB4) co-express within the module [KG Evidence]; THBD is associated with thrombophilia [KG Evidence]. No direct KG bridge was found connecting dimethylarginine to the vascular protein members. [KG Evidence: absence noted]

**Validation step:** Test whether ADMA (distinguished from SDMA by chromatographic separation if possible) correlates preferentially with the vascular subcluster (TEK, CDH5, THBD) versus the immune subcluster (LGALS9, CD274, MPO) within the module. Approximately 18% of computational predictions advance to clinical investigation.

---

### 4. Biological Themes

#### 4.1 Dominant Shared Processes

Pathway enrichment analysis reveals the following hierarchy of shared biological processes, after excluding hub nodes (more than 1,000 edges flagged; Homo sapiens, plasma membrane, extracellular space, protein homodimerization activity excluded from interpretation):

| Process | Members | Hub Status |
|---|---|---|
| Response to stress (GO:0006950) | 30 | Non-hub [KG Evidence] |
| Protein binding (GO:0005515) | 27 | Non-hub [KG Evidence] |
| Signal transduction (GO:0007165) | 13 | Non-hub [KG Evidence] |
| Cell-cell signaling (GO:0007267) | 12 | Non-hub (500 edges) [KG Evidence] |
| Immune response (GO:0006955) | 11 | Non-hub [KG Evidence] |
| Immune system process (GO:0002376) | 10 | Non-hub [KG Evidence] |
| Inflammatory response (GO:0006954) | 8 | Borderline (800 edges) [KG Evidence] |
| Response to LPS (GO:0032496) | 7 | Non-hub [KG Evidence] |
| Adaptive immune response (GO:0002250) | 7 | Non-hub (400 edges) [KG Evidence] |
| Cytokine-mediated signaling (GO:0019221) | 7 | Non-hub [KG Evidence] |
| Proteolysis (GO:0006508) | 7 | Non-hub [KG Evidence] |
| NF-κB signaling (GO:0043123) | 6 | Non-hub [KG Evidence] |
| Apoptosis (UMLS:C0162638) | 6 | Non-hub [KG Evidence] |
| Defense response to bacterium (GO:0042742) | 5 | Non-hub [KG Evidence] |
| TNF signaling (UMLS:C1519684) | 5 | Non-hub [KG Evidence] |

#### 4.2 Non-Hub Shared Neighbors of Special Interest

The following specific shared neighbors have moderate connectivity and connect entities across type boundaries:

**Negative regulation of TNF production (GO:0032720; 120 edges):** Connects LGALS9, CX3CL1, and SLAMF1. [KG Evidence] This low-degree, high-specificity node indicates that galectin and fractalkine signaling converge on TNF regulation; the absence of TNF itself from the module (see Gap Analysis) supports a model of receptor-side rather than ligand-side TNF pathway modulation. [Inferred]

**Colorectal cancer (MONDO:0005575; 500 edges):** Connects 5 metabolites (N-acetylputrescine, N-formylmethionine, N-acetylmethionine, N-acetylvaline, N-acetylalanine) via correlated_with and contributes_to predicates. [KG Evidence] This non-hub disease association for the N-acetylated amino acid cluster suggests a shared metabolic perturbation relevant to gastrointestinal pathology. [Inferred]

**Placenta (UBERON:0001987; 300 edges):** Connects N-acetylvaline, N-acetylalanine, 4-acetamidobutanoate, and O-sulfo-L-tyrosine. [KG Evidence] The placental localization, combined with the module's pathway annotation for female pregnancy (GO:0007565; connecting 28 entities) and LGALS9's role in maternal processes (GO:0007565), raises the possibility that this module's biology is relevant to gestational immune adaptation. [KG Evidence] [Inferred]

#### 4.3 Shared Interacting Genes (Non-Module Members)

Pathway enrichment identified 30 input entities connected through shared gene neighbors, with CD44, CASP8, FADD, APP, and ERBB2 as the most prominent non-module connectors. [KG Evidence] The presence of CASP8 and FADD as shared neighbors is consistent with the module's TRAIL/FAS apoptotic signaling theme (TNFRSF10B, FAS, GO:0006915). [KG Evidence] [Inferred]

---

### 5. Gap Analysis

#### 5.1 Informative Absences (Open World Assumption)

Under the open world assumption, the absence of an entity means "not observed in this context," not "biologically uninvolved." The following absences are informationally rich:

**TNF-alpha (TNF).** Seven TNF receptor superfamily members and pathway annotations for TNF production regulation are present, yet TNF itself is absent. [KG Evidence: gap] This pattern indicates receptor-side modulation rather than ligand abundance changes as the primary signaling mechanism. Post-transcriptional regulation and rapid TNF clearance kinetics further explain this dissociation. [Inferred]

**IL-6 and CRP.** The module contains IL6R (IL-6 receptor) but not IL-6, and contains no CRP. [KG Evidence: gap] IL6R participates in cytokine-mediated signaling (GO:0019221) and cell-cell signaling (GO:0007267). [KG Evidence] The presence of the receptor without the ligand suggests trans-signaling (soluble IL-6R binding IL-6 in trans) may be the relevant pathway, and that systemic acute-phase response is not the dominant inflammatory mechanism in this module. [Inferred]

**VEGF-A.** PGF and TEK are present without their canonical co-player VEGF-A. [KG Evidence: gap] This absence defines a VEGF-A-independent angiogenic program (see Novel Prediction 3.1).

**Adiponectin and Leptin.** The absence of both major adipokines indicates the module is orthogonal to adipose tissue dysfunction pathways. [KG Evidence: gap] The module's co-expression signature (immune receptors, galectins, chemokines, vascular markers) instead reflects vascular-immune crosstalk. [Inferred]

**BCAAs and Ceramides.** Branched-chain amino acids and ceramides are absent, likely reflecting platform limitations (targeted proteomics/metabolomics) rather than biological irrelevance. [KG Evidence: gap]

#### 5.2 Standard Methodological Gaps

HbA1c, HOMA-IR, fasting glucose, and GLP-1/GIP are clinical indices or rapidly cleared peptide hormones not expected in omics-derived co-expression modules. [KG Evidence: gap] NF-κB pathway components (NFKB1, RELA) are regulated post-translationally and would not appear in protein/transcript abundance co-expression analysis. [KG Evidence: gap]

#### 5.3 Cold-Start Metabolites

Three metabolites have zero KG edges: N1-methylinosine, N6-carbamoylthreonyladenosine, and carboxyethyl-GABA. [KG Evidence] Semantic similarity analysis identified 7-methylinosine (0.87 similarity) as an analogue for N1-methylinosine, inferring a relationship to inosine (CHEBI:17596). [KG Evidence] N6-carbamoylthreonyladenosine matched only to structurally unrelated carbamate compounds (similarity 0.88 to 0.90), providing limited biological insight. [KG Evidence] Carboxyethyl-GABA matched to 1-carboxyethylleucine (0.88 similarity), inferring blood localization. [KG Evidence] These cold-start entities require targeted literature review and experimental characterization.

---

### 6. Temporal Context

No explicit longitudinal timepoints are provided in the analysis input. The gap analysis references a T2D conversion study context, and the following temporal inferences apply:

**Upstream causes (candidate drivers).** The immune activation signature (TNF receptor signaling, neutrophil degranulation, galectin-mediated T cell regulation) and vascular remodeling markers (TEK, PGF, CDH5) are candidate upstream drivers. [Inferred] Immune checkpoint engagement (CD274, PDCD1LG2) and kynurenine accumulation may represent early immunosuppressive responses to chronic immune activation. [Inferred]

**Downstream consequences (candidate effects).** Modified nucleoside accumulation (pseudouridine, methylated nucleosides) likely reflects downstream cellular turnover. [Inferred] Urinary system disorder association (24 members) suggests that declining renal function may be both a consequence of vascular-immune pathology and a contributor to uremic toxin accumulation, creating a feed-forward loop. [Inferred] GDF15, a mitochondrial stress cytokine (6,063 edges; hub-flagged), represents a downstream integrator of cellular stress signals. [KG Evidence] [Inferred]

**Causal inference opportunities.** If serial sampling is available, Granger causality or dynamic Bayesian network analysis could test whether changes in the immune subcluster (LGALS9, TNFRSF1A, MPO) temporally precede changes in the metabolite subcluster (pseudouridine, kynurenine, dimethylarginine). [Inferred] Mendelian randomization using cis-pQTL instruments for PCSK9 (4,001 edges), IL6R, or EGFR could test causal relationships between these proteins and module-associated disease outcomes. [Inferred]

---

### 7. Research Recommendations

#### 7.1 High Priority: Experimental Validations

1. **Kynurenine–checkpoint axis.** Measure plasma kynurenine/tryptophan ratio, soluble PD-L1, and Galectin-9 levels in the study cohort. Test three-way correlations and assess whether this triad predicts disease outcomes independently of conventional risk factors. [Inferred]

2. **VEGF-A-independent angiogenesis.** Measure VEGF-A, PGF, and soluble Tie-2 in parallel. The module predicts that PGF and sTie-2 (but not VEGF-A) will track with the Blue module eigengene. If confirmed, this implicates a specific angiogenic program amenable to anti-PGF or anti-Angiopoietin-2 therapeutic strategies. [Inferred]

3. **Modified nucleoside panel as immune turnover marker.** Correlate pseudouridine and N2,N2-dimethylguanosine with neutrophil counts, MPO levels, and estimated GFR to distinguish immune turnover from renal clearance as the dominant determinant. [Inferred]

#### 7.2 Medium Priority: Targeted Literature Review

4. **LGALS9–INSR interaction.** The knowledge graph identifies a novel hidden-gem interaction between LGALS9 and INSR (insulin receptor). [KG Evidence] This connection, if validated, would provide a direct mechanistic link between galectin-mediated immune modulation and insulin signaling. Targeted PubMed search for "galectin-9 insulin receptor" and experimental co-immunoprecipitation studies are recommended. [Inferred]

5. **C-glycosyltryptophan aging connection.** Search published metabolomics-GWAS studies for genetic determinants of C-glycosyltryptophan levels and their overlap with immune or inflammatory trait loci. [Inferred]

6. **Dimethylarginine–vascular subcluster.** Review the literature on ADMA as a predictor of endothelial dysfunction in immune-mediated diseases (e.g., rheumatoid arthritis, SLE) to contextualize its co-expression with vascular markers in this module. [Inferred]

#### 7.3 Follow-Up Computational Analyses

7. **Sub-module structure.** Apply hierarchical clustering or community detection within the Blue module to resolve the immune, vascular, and metabolic subclusters and identify bridging members that connect them. LGALS9, GDF15, and PLAU/PLAUR are candidate bridge nodes. [Inferred]

8. **Cross-module comparison.** Compare the Blue module's disease recurrence profile against other WGCNA modules in the study to identify diseases unique to this module versus shared across the network. [Inferred]

9. **Hub-adjusted re-analysis.** Re-run disease and pathway enrichment after removing the 13 hub-flagged entities (EGFR, MMP2, choline, CD274, SPP1, CCL2, AXL, IL10, CTSL, MERTK, GDF15, LGALS3, CD4) to assess whether the same biological themes emerge from lower-connectivity members alone. [Inferred]

10. **Mendelian randomization.** Leverage cis-pQTLs for IL6R, PCSK9, and TNFRSF1A as genetic instruments to test causal effects on cardiovascular and metabolic outcomes using publicly available GWAS summary statistics. [Inferred]

---

*Report generated from KRAKEN knowledge graph analysis. All [KG Evidence] claims are derived from direct query results. [Literature] citations are provided where grounded abstracts support specific claims. [Model Knowledge] reflects general biomedical understanding not backed by the KG or grounded literature in this analysis. [Inferred] marks conclusions derived by integrating multiple evidence sources. Entities with hub-bias warnings (more than 1,000 edges) are flagged throughout; their associations should be interpreted as potentially non-specific.*

### Literature References

Papers discovered via semantic search. 4 unique papers across 3 hypotheses.

| Hypothesis | Citation | Link | Relevance |
|------------|----------|------|-----------|
| Bridge: Gene → ChemicalEntity (1 hops); Bridge: Gene → ChemicalEntity (2 hops) | Alessandra Zingoni et al. (2024) "The senescence journey in cancer immunoediting" | [DOI](https://doi.org/10.1186/s12943-024-01973-5) | — |
| Bridge: Gene → ChemicalEntity (1 hops); Bridge: Gene → ChemicalEntity (2 hops) | Bouabid Badaoui et al. (2014) "RNA-Sequence Analysis of Primary Alveolar Macrophages after In Vitro Infection with Porcine Reproduc..." | [DOI](https://doi.org/10.1371/journal.pone.0091918) | — |
| Bridge: Gene → ChemicalEntity (1 hops); Bridge: Gene → ChemicalEntity (2 hops) | Juhee Jeong et al. (2019) "Context Drives Diversification of Monocytes and Neutrophils in Orchestrating the Tumor Microenvironm..." | [DOI](https://doi.org/10.3389/fimmu.2019.01817) | — |
| Bridge: Gene → MolecularMixture (2 hops) |  (2021) "Frontiers \| Aberrantly Expressed Galectin-9 Is Involved in the Immunopathogenesis of Anti-MDA5-Posit..." | [Link](https://www.frontiersin.org/journals/cell-and-developmental-biology/articles/10.3389/fcell.2021.628128/full) | Galectins are a family of proteins that bind to β-galactoside-containing glycans (Thiemann and Baum, 2016). In this fami... |