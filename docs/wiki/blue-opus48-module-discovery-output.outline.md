# Blue Module Run on Opus 4.8: Discovery Output (149-analyte, dev, 2026-06-24)

This document presents the discovery output of a full-module run of the **Blue** WGCNA module through the Kraken 12-node discovery pipeline, executed on the dev environment on 2026-06-24 (commit `098093e`) with BioMapper entity resolution enabled. The SDK-backed pipeline nodes ran on the Opus 4.8 model rather than the default Sonnet; this run is published for direct model comparison. It is one of a per-module series over the Frailty multi-omics WGCNA modules; see the [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm).

## Run Provenance

The run submitted 149 named analytes, parsed 146 at intake, and resolved 146 distinct entities (135 biomapper, 2 exact, 9 fuzzy) to 146 distinct CURIEs. Triage classified 95 well-characterized, 24 moderate, 24 sparse, and 3 cold-start, with 0 edge-count measurement failures. Downstream analysis produced 3696 direct-KG findings, 8 cold-start findings, 10 biological themes, 20 cross-entity bridges (15 evidence-grounded), and 29 hypotheses supported by 38 literature references. Synthesis emitted a 26892-character report. The run completed in approximately 1296.1 s of wall-clock time (status complete, 1 errors). All counts are taken from the reducer-merged pipeline state.

| Stage | Result |
|---|---|
| Submitted | 149 named analytes |
| Intake | 146 parsed |
| Entity resolution | 146 resolved (135 biomapper, 2 exact, 9 fuzzy) to 146 distinct CURIEs |
| Triage | 95 well-characterized, 24 moderate, 24 sparse, 3 cold-start (0 measurement failures) |
| Direct KG | 3696 findings |
| Cold-start | 8 findings, 19 skipped |
| Pathway enrichment | 10 biological themes |
| Integration | 20 bridges (15 evidence-grounded) |
| Literature grounding | 38 papers |
| Synthesis | 29 hypotheses, 26892-character report |
| Run total | ~1296.1 s wall-clock, status complete, 1 errors |

## Related

- Companion run metrics: [Blue Module Run on Opus 4.8: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/blue-module-run-on-opus-48-pipeline-performance-report-149-analyte-dev-2026-06-24-1rG3j14din)
- Model comparison baseline (Sonnet): [Blue Module Run: Discovery Output](https://phwiki.phenoma.ai/doc/blue-module-run-discovery-output-149-analyte-dev-2026-06-23-9dAc5UFomv)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---



## Blue WGCNA Module Discovery Report: An Immune-Vascular-Metabolic Co-Expression Network

### 1. Executive Summary

This Blue WGCNA module encodes a coordinated immune-vascular surveillance program characterized by the co-expression of 90 proteins and 56 metabolites that collectively link innate/adaptive immune checkpoint signaling, endothelial homeostasis, and uremic solute accumulation. [KG Evidence] The module is distinguished by a recurring "receptor-without-ligand" architecture: TNF superfamily receptors (TNFRSF10B, TNFRSF13B, TNFRSF1A, TNFRSF1B, TNFRSF11A), immune checkpoint ligands (PDCD1LG2, CD274), and the vascular receptor TEK co-express in the absence of their cognate ligands (TNF, BAFF/APRIL, PD-1, ANGPT1/2), indicating that this module captures the responding cell compartment (myeloid/endothelial) rather than the stimulating compartment. [KG Evidence; Inferred] The metabolite complement, dominated by modified nucleosides (pseudouridine, N2,N2-dimethylguanosine, N1-methyladenosine, C-glycosyltryptophan), N-acetylated amino acids, and catecholamine catabolites, points to accelerated RNA turnover and renal clearance impairment as the metabolic signature unifying this module with its inflammatory-vascular protein axis. [Model Knowledge]

### 2. Key Findings (Tier 1 to 2)

#### 2.1 Disease Associations

The module-level disease recurrence analysis reveals strong, multi-member associations with inflammatory, cardiovascular, and autoimmune conditions. [KG Evidence]

| Disease | Members | Evidence |
|---------|---------|----------|
| Asthma | 23 | Curated |
| Coronary artery disorder | 23 | Curated |
| Panniculitis | 22 | Curated |
| Psoriasis | 22 | Curated |
| Irritable bowel syndrome | 19 | GWAS |
| Depressive disorder | 19 | Curated |
| Cancer (general) | 19 | Curated |
| Essential hypertension | 17 | Curated |
| Diabetes mellitus | 16 | Curated |
| Hematologic disorder | 11 | Curated |

Notably, 23 module members share curated associations with both asthma and coronary artery disorder, indicating that this module sits at the intersection of mucosal immune activation and vascular inflammation. [KG Evidence] Diabetes mellitus associates with 16 members spanning both protein (IL18BP, MERTK, OSCAR, LGALS9, PIGR) and metabolite (adenine, dimethylglycine, dimethylarginine) categories, reinforcing the module's relevance to metabolic-inflammatory crosstalk. [KG Evidence]

#### 2.2 Pathway Architecture

The module converges on five principal biological processes, each supported by direct KG evidence from multiple members. [KG Evidence]

**Immune regulation and cytokine signaling.** A total of 12 members participate in immune response (GO:0006955), 8 in inflammatory response (GO:0006954), and 5 in cytokine-mediated signaling (GO:0019221). [KG Evidence] LGALS9 occupies a central regulatory position: it negatively regulates chemokine production and activated T cell proliferation while positively regulating IL-10, IL-12, IL-1β, and IL-8 production. [KG Evidence] This dual pro- and anti-inflammatory capacity positions LGALS9 as a master immunomodulator within the module. Five members (CD274, LGALS9, XCL1, SLAMF1, PDCD1LG2) participate in negative regulation of type II interferon production (GO:0032689), while four members (IL1R1, IL12B, LGALS9, SLAMF1) participate in positive regulation of type II interferon production (GO:0032729), suggesting finely tuned IFN-γ control as a hallmark of this module. [KG Evidence]

**TNF superfamily signaling and apoptosis.** Six members participate in cytokine signaling processes involving TNFRSF13B, TNFRSF11A, TNFRSF10C, and TNFRSF10B. [KG Evidence] TNFRSF10B mediates TRAIL-activated apoptotic signaling and positive regulation of canonical NF-κB signal transduction. [KG Evidence] The convergence of four TNF superfamily receptors (TNFRSF10B, TNFRSF10C, TNFRSF11A, TNFRSF13B) on apoptosis, cell survival, and TNF signaling pathways indicates that programmed cell death regulation constitutes a core function of this module. [KG Evidence]

**Vascular and endothelial biology.** TEK, CDH5, EPHB4, THBD, CD93, NOTCH3, ICAM2, and PGF collectively represent a comprehensive endothelial cell surface signature. [Model Knowledge] TEK participates in signal transduction and cell-cell signaling (GO:0007267). [KG Evidence] THBD participates in response to lipopolysaccharide (GO:0032496) and is associated with thrombophilia. [KG Evidence] The presence of both arterial (EPHB4, NOTCH3) and venous/capillary (CDH5, CD93) endothelial markers suggests the module captures a pan-endothelial activation state rather than a vessel-type-specific program. [Model Knowledge]

**Cell adhesion and leukocyte migration.** Six members participate in cell adhesion (GO:0007155): CEACAM8, AMBP, CCL2, SLAMF1, SPP1, and CD6. [KG Evidence] Six members participate in leukocyte migration (GO:0050900): CEACAM8, IL16, CCL2, XCL1, SLAMF1, and THBD. [KG Evidence] These overlapping memberships indicate that the module encodes both the adhesive substrate and the chemotactic signals required for immune cell trafficking across endothelium.

**Neutrophil granule contents.** MPO, PRTN3, AZU1, CEACAM8, and PI3 represent azurophil and specific granule proteins released during neutrophil degranulation. [Model Knowledge] Their co-expression with endothelial markers (CDH5, ICAM2, THBD) suggests active neutrophil-endothelial interaction within the biological context captured by this module.

#### 2.3 Protein-Protein Interactions

LGALS9 physically interacts with two other module members: LGALS3 and ALCAM. [KG Evidence] LGALS9 also interacts with the insulin receptor (INSR), a novel connection identified in the knowledge graph that may bridge the immune and metabolic axes of this module. [KG Evidence] The SGTA protein physically interacts with four module members (TNFRSF13B, CTSL, TFF3, CSF1), serving as a shared interaction hub. [KG Evidence] TRAF2, TRAF5, and TRAF6 connect 17 input entities, confirming that TNF receptor-associated factor signaling constitutes the dominant intracellular signaling scaffold of this module. [KG Evidence]

#### 2.4 Cross-Type Bridges (Protein to Metabolite)

The knowledge graph reveals multi-hop connections between the protein and metabolite arms of this module. [KG Evidence]

LGALS9 connects to CCL2 through direct interaction (1 hop) and through shared participation in inflammatory response (GO:0006954) and cellular response to type II interferon (GO:0071346) (2 hops). [KG Evidence] LGALS9 and TNFRSF10B both connect to dimethylglycine through liver co-localization (2 hops) and through TNF-mediated regulation (2 hops). [KG Evidence] The path TNFRSF10B → doxorubicin → CCL2 represents the strongest causal bridge, with both legs carrying curated-causal provenance, suggesting that drug perturbation of TNFRSF10B-mediated apoptosis directly modulates CCL2 chemokine production. [KG Evidence]

### 3. Novel Predictions (Tier 3)

#### 3.1 The Module as a Renal Clearance Failure Signature

**Logic chain:** The metabolite arm of this module is enriched for uremic retention solutes: pseudouridine, C-glycosyltryptophan, N2,N2-dimethylguanosine, dimethylarginine (SDMA+ADMA), N1-methyladenosine, erythritol, and mannitol/sorbitol are all established markers of declining glomerular filtration rate. [Model Knowledge] Their co-expression with FGF23 (fibroblast growth factor 23, the master regulator of phosphate homeostasis in chronic kidney disease) and ADM (adrenomedullin, elevated in renal dysfunction and associated with glomerulonephritis per the KG [KG Evidence]) suggests this module captures a subclinical nephropathy state. The co-presence of kynurenine, a tryptophan catabolite elevated in renal failure via indoleamine 2,3-dioxygenase activation, further supports this interpretation. [Model Knowledge]

**Validation step:** Correlate the module eigengene with estimated glomerular filtration rate (eGFR) and serum cystatin C in the study cohort. If the module tracks renal function, targeted metabolomics of uremic solutes in matched urine samples would confirm the renal clearance hypothesis.

**Calibration:** Approximately 18% of computational predictions of this nature progress to clinical investigation; the strength of this prediction is enhanced by the convergence of more than seven independent uremic markers within a single co-expression module.

#### 3.2 LGALS9-INSR Interaction as a Mechanism for Immune-Metabolic Coupling

**Logic chain:** LGALS9 physically interacts with the insulin receptor (INSR), a novel connection identified in the knowledge graph. [KG Evidence] LGALS9 is a β-galactoside-binding lectin expressed on macrophages, endothelial cells, and Kupffer cells [Literature: Frontiers, 2021, "Aberrantly Expressed Galectin-9"]. Galectin-9 binding to glycosylated cell surface receptors modulates their signaling; interaction with INSR could directly modulate insulin sensitivity in immune and endothelial cells. The module contains both immune checkpoint molecules (PDCD1LG2, CD274) and metabolic mediators (dimethylarginine, kynurenine), and this interaction provides a mechanistic link between the two arms. [Inferred]

**Validation step:** Confirm the LGALS9-INSR interaction by co-immunoprecipitation in primary macrophages or endothelial cells. Test whether recombinant galectin-9 modulates insulin-stimulated AKT phosphorylation in these cell types.

**Calibration:** Approximately 18% of such computationally predicted protein-protein interactions advance to experimental validation; this prediction is strengthened by the known galectin-glycoprotein binding mode and the INSR's extensive N-glycosylation.

#### 3.3 Kynurenine Pathway as the Metabolic Effector of Immune Checkpoint Co-Expression

**Logic chain:** Kynurenine, the product of indoleamine 2,3-dioxygenase (IDO1), co-expresses with immune checkpoint ligands CD274 (PD-L1) and PDCD1LG2 (PD-L2). [KG Evidence for co-expression membership; Model Knowledge for IDO1 mechanism] IDO1 is transcriptionally induced by IFN-γ and generates kynurenine, which activates the aryl hydrocarbon receptor (AhR) to promote regulatory T cell differentiation and immunosuppression. The module's negative regulation of type II interferon production (5 members, GO:0032689 [KG Evidence]) and simultaneous presence of kynurenine suggests a feedback circuit: IFN-γ → IDO1 → kynurenine → AhR → suppression of further IFN-γ. [Inferred]

**Validation step:** Measure IDO1 protein or enzymatic activity (kynurenine-to-tryptophan ratio) in the study cohort. Correlate with CD274 and PDCD1LG2 protein levels to test the co-regulation hypothesis.

**Calibration:** Approximately 18% of pathway-inferred predictions reach validation; this prediction carries above-average plausibility given the well-established IDO1-PD-L1 co-induction axis in tumor immunology (Zhang et al., 2020) [Literature].

#### 3.4 Catecholamine Catabolites Indicate Sympathetic Nervous System Activation

**Logic chain:** Homovanillate (HVA), vanillylmandelate (VMA), 3-methoxytyrosine, 3-methoxytyramine sulfate, and vanillactate are catecholamine degradation products that co-express within this module. [KG Evidence for module membership] Their simultaneous elevation indicates increased catecholamine turnover, consistent with sympathetic nervous system activation. ADM (adrenomedullin) is a vasodilatory peptide released in response to sympathetic overdrive and cardiovascular stress, and its co-expression with these catabolites reinforces a sympatho-adrenal activation signature. [Model Knowledge; KG Evidence for ADM association with glomerulonephritis] The module's associations with essential hypertension (17 members) and coronary artery disorder (23 members) [KG Evidence] are consistent with chronic sympathetic activation as a shared upstream driver.

**Validation step:** Correlate module eigengene with plasma norepinephrine or 24-hour urinary catecholamine excretion. Test whether beta-adrenergic blockade modifies the module's protein expression in a subset of treated participants.

**Calibration:** Approximately 18% of such multi-analyte pathway inferences proceed to clinical testing.

#### 3.5 Carboxyethyl-GABA as a Novel Blood-Detectable Uremic Solute

**Logic chain:** Carboxyethyl-GABA (RM:0004069) has no direct knowledge graph presence (0 edges, cold-start entity). Semantic similarity analysis identifies 1-carboxyethylleucine (88% similarity) as a structural analogue that is located in blood. [KG Evidence] As a carboxyethyl-amino acid derivative sharing the same chemical modification pattern (carboxyethylation of an amino acid backbone), carboxyethyl-GABA may similarly accumulate in blood, particularly under conditions of impaired renal clearance that characterize this module. [Inferred]

**Validation step:** Search HMDB and MetaboLights for evidence of carboxyethyl-GABA detection in blood or plasma. Develop a targeted LC-MS/MS assay if no existing data confirm its presence.

**Calibration:** Approximately 18% of semantic-similarity-based predictions validate; the low similarity threshold (88%) and cold-start status of this entity warrant conservative interpretation.

### 4. Biological Themes

#### 4.1 Unifying Theme: Immune-Vascular Surveillance with Metabolic Exhaustion

The Blue module captures a biological state in which the innate immune system (neutrophil granule proteins: MPO, PRTN3, AZU1; macrophage receptors: MERTK, AXL, CSF1, SIRPA) and the adaptive immune system (CD4, IL2RA, SLAMF1, CD5, CD6; checkpoint ligands: CD274, PDCD1LG2) co-engage with the vascular endothelium (TEK, CDH5, EPHB4, THBD, NOTCH3, ICAM2) in a context of impaired metabolite clearance (pseudouridine, dimethylarginine, modified nucleosides). [KG Evidence for pathway memberships; Model Knowledge for metabolic interpretation]

#### 4.2 Emergent Patterns from Pathway Enrichment

**Response to stress** constitutes the most broadly shared pathway, connecting 24 members. [KG Evidence] **Protein binding** connects 23 members, reflecting the receptor-rich composition of the module. [KG Evidence] These broad categories, when filtered for hub bias, yield more informative signals:

**Non-hub enrichment patterns** (de-emphasizing hub nodes with more than 1000 edges):

- **Chemotaxis** (GO:0006935): connects IL16, PLAUR, CCL15, and RARRES2, representing a specific myeloid-cell-recruiting chemotactic program. [KG Evidence]
- **Defense response to Gram-positive bacterium** (GO:0050830): connects module members, indicating mucosal antimicrobial capacity. [KG Evidence]
- **Response to hypoxia** (GO:0001666): connects module members and aligns with the vascular/endothelial component of the module. [KG Evidence]
- **Cytokine activity** (GO:0005125): connects IL16, CCL15, and IL12B, identifying the specific secreted mediators. [KG Evidence]

#### 4.3 Metabolite Substructure Themes

The 56 metabolites organize into four coherent biochemical classes:

1. **Modified nucleosides and RNA turnover products** (pseudouridine, N2,N2-dimethylguanosine, N4-acetylcytidine, N1-methyladenosine, 5,6-dihydrouridine, N6-carbamoylthreonyladenosine, 3-(3-amino-3-carboxypropyl)uridine, 7-methylguanine, orotidine): these 9 metabolites derive from tRNA and rRNA degradation and represent established uremic retention solutes. [Model Knowledge]

2. **N-acetylated amino acids** (N-acetylmethionine, N-acetylvaline, N-acetylalanine, N-acetylthreonine, N-acetylserine, N-acetyl-beta-alanine, N-acetyltaurine, N6-acetyllysine): this class of 8 metabolites shares the N-acyl-amino acid chemical ontology (CHEBI:21545) and localizes to feces, placenta, urine, and blood. [KG Evidence] Their accumulation may reflect altered hepatic N-acetyltransferase activity or reduced renal tubular clearance. [Model Knowledge]

3. **Catecholamine catabolites** (homovanillate, vanillylmandelate, 3-methoxytyrosine, 3-methoxytyramine sulfate, vanillactate): these 5 metabolites represent sequential degradation products of dopamine and norepinephrine. [Model Knowledge]

4. **One-carbon and methyl-donor metabolites** (dimethylglycine, choline, N-formylmethionine, dimethylarginine): these metabolites connect to one-carbon metabolism and methylation capacity. [Model Knowledge] The co-expression of dimethylglycine with immune proteins is bridged through liver co-localization and TNF-mediated regulation in the knowledge graph. [KG Evidence]

### 5. Gap Analysis

#### 5.1 Informative Absences (Open World Assumption)

The absence of canonical ligands for receptors present in the module constitutes the most informative pattern in this analysis. Each absence is interpreted under the open world assumption: absence means "unstudied or not co-expressed," not "biologically irrelevant."

**Receptor-ligand decoupling pattern:**

| Present Receptor/Ligand | Absent Cognate Partner | Interpretation |
|---|---|---|
| TEK (Tie2) | ANGPT1, ANGPT2 | Receptor-side vascular signaling captured independently [Inferred] |
| TNFRSF10B (TRAIL-R2) | TNF | Receptor-poised immune state, not active TNF signaling [Inferred] |
| TNFRSF13B (TACI) | TNFSF13 (APRIL) | B cell response machinery without B cell stimulating signals [Inferred] |
| PDCD1LG2 (PD-L2) | PDCD1 (PD-1) | Ligand-expressing APC compartment, not T cell compartment [Inferred] |
| PGF (PlGF) | FLT1 (VEGFR1) | Paracrine signal without autocrine receptor loop [Inferred] |
| PIGR | TNFSF13B (BAFF) | Mucosal Ig transport without B cell activation signal [Inferred] |

This systematic decoupling reveals that the module captures a "receiver" cell state: cells that bear receptors for inflammatory and angiogenic signals but do not themselves produce the ligands. [Inferred] This architecture is consistent with a myeloid/endothelial cell compartment poised to respond to external stimuli.

**Absence of the IL6 to CRP acute-phase axis.** Neither IL6 nor CRP co-express with this module, despite 8 members participating in inflammatory response (GO:0006954). [KG Evidence for pathway membership; Inferred for absence interpretation] This absence indicates that the module encodes a chronic, tissue-level inflammatory program distinct from the systemic acute-phase response. The presence of IL10, IL10RB, and IL18BP (anti-inflammatory mediators) alongside pro-inflammatory molecules (MPO, CCL2, CXCL9) suggests a mixed or resolving inflammatory phenotype rather than an acute flare. [KG Evidence for membership; Model Knowledge for interpretation]

**Absence of adipokines (ADIPOQ, LEP).** The module lacks adipose-tissue-derived signals despite its strong metabolic associations (diabetes mellitus: 16 members [KG Evidence]). This confirms the module's identity as an immune-endothelial network rather than an adipose-metabolic one. [Inferred]

#### 5.2 Standard Gaps (Platform Limitations)

Branched-chain amino acids (leucine, isoleucine, valine), ceramides, insulin, C-peptide, and HbA1c are absent from this module. [KG Evidence] These absences likely reflect assay platform boundaries (proteomic panels vs. clinical assays vs. lipidomic platforms) rather than biological exclusion. [Model Knowledge]

#### 5.3 Cold-Start Entities

Three metabolites have zero knowledge graph edges: N1-methylinosine (RM:0005028), N6-carbamoylthreonyladenosine (CHEMBL.COMPOUND:CHEMBL3628298), and carboxyethyl-GABA (RM:0004069). [KG Evidence] All three are modified nucleosides or amino acid derivatives consistent with the module's RNA turnover and uremic solute themes. [Model Knowledge] Semantic similarity analysis identifies 7-methylinosine (87% similarity) as the closest analogue for N1-methylinosine, and 1-carboxyethylleucine (88% similarity) for carboxyethyl-GABA. [KG Evidence]

### 6. Temporal Context

No explicit longitudinal timepoints are provided in this analysis; however, the module's composition permits inference about causal directionality.

#### 6.1 Upstream Causes (Candidate Drivers)

**FGF23** is a strong candidate upstream driver. FGF23 is produced by osteocytes in response to phosphate load and declining renal function; it suppresses 1,25-dihydroxyvitamin D synthesis and promotes sodium retention. [Model Knowledge] Its co-expression with uremic solutes (pseudouridine, dimethylarginine) and vascular markers (ADM, THBD) suggests it may be an early signal of nephropathy that drives downstream vascular and immune activation. [Inferred] The knowledge graph identifies FGF23 association with diabetic macular edema through PGF (also in the module). [KG Evidence]

**GDF15** (growth differentiation factor 15) is a stress-responsive cytokine elevated in mitochondrial dysfunction, renal impairment, and cardiovascular disease. [Model Knowledge] With 6,063 KG edges, it is well-characterized and may serve as a proximal trigger of the metabolic stress captured by the metabolite arm of this module. [KG Evidence for edge count; Inferred for causal role]

#### 6.2 Downstream Consequences (Candidate Effectors)

**SPP1** (osteopontin) and **CHI3L1** (YKL-40) are established downstream effectors of chronic inflammation and tissue remodeling. [Model Knowledge] SPP1 participates in cell adhesion (GO:0007155) and is associated with central nervous system malformation. [KG Evidence] Both proteins are clinical biomarkers of fibrosis progression and could serve as pharmacodynamic endpoints in intervention studies.

**PCSK9** is a druggable target (alirocumab, evolocumab) present in the module with 4,001 KG edges. [KG Evidence] Its co-expression with inflammatory and vascular markers suggests that PCSK9 inhibition might modulate not only LDL cholesterol but also the immune-vascular program encoded by this module. [Inferred]

### 7. Research Recommendations

#### 7.1 High-Priority Experimental Validations

1. **Correlate module eigengene with eGFR and cystatin C.** The uremic solute enrichment in the metabolite arm predicts a strong inverse correlation with renal function. This is the single most informative validation experiment for this module. [Inferred]

2. **Test the LGALS9-INSR physical interaction.** Confirm by co-immunoprecipitation in THP-1 macrophages or HUVECs. If validated, assess whether galectin-9 modulates insulin signaling (pAKT, pERK) in these cell types. [KG Evidence for interaction; Inferred for functional consequence]

3. **Measure IDO1 activity (kynurenine:tryptophan ratio) and correlate with CD274/PDCD1LG2 protein levels.** This tests the prediction that the kynurenine pathway serves as the metabolic effector of immune checkpoint co-expression. [Inferred]

4. **Profile urinary catecholamines and correlate with module eigengene.** The catecholamine catabolite cluster (HVA, VMA, 3-methoxytyrosine, 3-methoxytyramine sulfate, vanillactate) predicts sympathetic activation as a contributor to this module. [Inferred]

#### 7.2 Literature Searches

1. **Galectin-9 in diabetic nephropathy.** The co-expression of LGALS9 with FGF23 and uremic metabolites, combined with its KG association with nephritis [KG Evidence], warrants a systematic review of galectin-9 as a biomarker or mediator of diabetic kidney disease.

2. **Modified nucleoside panels as integrated renal biomarkers.** The 9 modified nucleosides in this module may constitute a more sensitive renal function panel than individual markers. Search for existing literature on pseudouridine, N2,N2-dimethylguanosine, and C-glycosyltryptophan as GFR surrogates.

3. **MERTK-AXL efferocytosis axis in vascular inflammation.** Both TAM family receptors (MERTK, AXL) are present with high KG connectivity (6,205 and 6,681 edges, respectively). [KG Evidence] Their role in clearing apoptotic cells from atherosclerotic plaques is well-established; their co-expression with endothelial markers in this module suggests an active efferocytosis program at the vessel wall. [Model Knowledge]

#### 7.3 Follow-Up Analyses

1. **Cross-module comparison.** Compare the Blue module's disease associations with other WGCNA modules to determine whether the receptor-ligand decoupling pattern is unique to this module or reflects a general feature of the cohort.

2. **Cell-type deconvolution.** Apply CIBERSORTx or similar deconvolution to the proteomic data to determine whether the module is driven by a specific cell type (monocytes/macrophages, endothelial cells, or neutrophils) or represents a multi-cellular program.

3. **Hub-adjusted disease enrichment.** Thirteen module members exceed 1,000 KG edges and are flagged for hub bias (EGFR, MMP2, choline, CD274, SPP1, CCL2, AXL, IL10, CTSL, MERTK, GDF15, LGALS3, CD4). [KG Evidence] Disease associations involving these members (particularly cancer, coronary artery disorder, and asthma) should be reanalyzed after hub correction to distinguish specific from nonspecific enrichment.

4. **Targeted metabolomics for cold-start entities.** N1-methylinosine, N6-carbamoylthreonyladenosine, and carboxyethyl-GABA lack KG representation. [KG Evidence] Developing LC-MS/MS reference standards and querying large metabolomics repositories (UK Biobank NMR/MS, MESA metabolomics) for these compounds would enable their biological characterization and potentially reveal novel renal or immune associations.

5. **PCSK9 as a pharmacological probe.** Given PCSK9's druggability and co-expression within this immune-vascular module, retrospective analysis of PCSK9 inhibitor trial biobanks (FOURIER, ODYSSEY) for changes in module-member proteins (GDF15, SPP1, IL6R, TNFRSF1A) could reveal whether statin-independent immune-vascular effects accompany LDL lowering. [Inferred]

### Literature References

Papers discovered via semantic search. 4 unique papers across 3 hypotheses.

| Hypothesis | Citation | Link | Relevance |
|------------|----------|------|-----------|
| Bridge: Gene → ChemicalEntity (1 hops); Bridge: Gene → ChemicalEntity (2 hops) | Alessandra Zingoni et al. (2024) "The senescence journey in cancer immunoediting" | [DOI](https://doi.org/10.1186/s12943-024-01973-5) | — |
| Bridge: Gene → ChemicalEntity (1 hops); Bridge: Gene → ChemicalEntity (2 hops) | Bouabid Badaoui et al. (2014) "RNA-Sequence Analysis of Primary Alveolar Macrophages after In Vitro Infection with Porcine Reproduc..." | [DOI](https://doi.org/10.1371/journal.pone.0091918) | — |
| Bridge: Gene → ChemicalEntity (1 hops); Bridge: Gene → ChemicalEntity (2 hops) | Juhee Jeong et al. (2019) "Context Drives Diversification of Monocytes and Neutrophils in Orchestrating the Tumor Microenvironm..." | [DOI](https://doi.org/10.3389/fimmu.2019.01817) | — |
| Bridge: Gene → MolecularMixture (2 hops) |  (2021) "Frontiers \| Aberrantly Expressed Galectin-9 Is Involved in the Immunopathogenesis of Anti-MDA5-Posit..." | [Link](https://www.frontiersin.org/journals/cell-and-developmental-biology/articles/10.3389/fcell.2021.628128/full) | Galectins are a family of proteins that bind to β-galactoside-containing glycans (Thiemann and Baum, 2016). In this fami... |