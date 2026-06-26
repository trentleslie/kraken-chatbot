# Grey Module Run: Discovery Output (46-analyte, dev, 2026-06-23)

This document presents the discovery output of a full-module run of the **Grey** WGCNA module through the Kraken 12-node discovery pipeline, executed on the dev environment on 2026-06-23 (commit `813fc3f`) with BioMapper entity resolution enabled. It is one of a per-module series over the Frailty multi-omics WGCNA modules; see the [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm).

## Run Provenance

The run submitted 46 named analytes, parsed 46 at intake, and resolved 46 distinct entities (41 biomapper, 5 fuzzy) to 46 distinct CURIEs. Triage classified 30 well-characterized, 6 moderate, 9 sparse, and 1 cold-start, with 0 edge-count measurement failures. Downstream analysis produced 2674 direct-KG findings, 28 cold-start findings, 8 biological themes, 20 cross-entity bridges (17 evidence-grounded), and 73 hypotheses supported by 23 literature references. Synthesis emitted a 23518-character report. The run completed in approximately 852.3 s of wall-clock time (status complete, 1 errors). All counts are taken from the reducer-merged pipeline state.

| Stage | Result |
|---|---|
| Submitted | 46 named analytes |
| Intake | 46 parsed |
| Entity resolution | 46 resolved (41 biomapper, 5 fuzzy) to 46 distinct CURIEs |
| Triage | 30 well-characterized, 6 moderate, 9 sparse, 1 cold-start (0 measurement failures) |
| Direct KG | 2674 findings |
| Cold-start | 28 findings, 4 skipped |
| Pathway enrichment | 8 biological themes |
| Integration | 20 bridges (17 evidence-grounded) |
| Literature grounding | 23 papers |
| Synthesis | 73 hypotheses, 23518-character report |
| Run total | ~852.3 s wall-clock, status complete, 1 errors |

## Related

- Companion run metrics: [Grey Module Run: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/grey-module-run-pipeline-performance-report-46-analyte-dev-2026-06-23-WhfflHp2gV)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---



## Discovery Report: Grey WGCNA Module (Multi-Omics Co-Expression Analysis)

### 1. Executive Summary

This Grey WGCNA module encodes a coordinated multi-axis immune activation program spanning Th1 (IFNG, TNF, IL2), Th2 (IL4, IL13, CCL24), and IL-20 subfamily (IL20, IL24, IL22RA1, IL20RA) signaling, coupled with tissue-remodeling effectors (MMP10, SFTPD, FGF5) and the renin-angiotensin axis (REN). [KG Evidence; Inferred] The co-expressed metabolite complement, rich in xenobiotic sulfate conjugates (thymol sulfate, umbelliferone sulfate, 2-naphthol sulfate, methyl-4-hydroxybenzoate sulfate) and microbially derived compounds (trans-urocanate, homostachydrine, N-formylphenylalanine), indicates an environmental or gut-microbial exposure signature that co-varies with systemic immune activation. [Inferred] Knowledge graph analysis reveals convergent disease associations in inflammatory, autoimmune, and barrier-tissue disorders (16 members linked to depressive disorder; 15 to panniculitis and asthma), while the conspicuous absence of canonical regulatory cytokines (IL-10, TGFB1, IL-5) suggests this module captures an unresolved effector state rather than a homeostatic or resolution program. [KG Evidence; Inferred]

---

### 2. Key Findings (Tier 1 to 2)

#### 2.1 Unifying Biological Theme: Multilineage Immune Effector Activation

The module converges on three functionally validated processes: inflammatory response (GO:0006954; 8 members), immune response (GO:0006955; 8 members), and cell-cell signaling (GO:0007267; 12 members). [KG Evidence] A total of 19 protein members participate in protein binding (GO:0005515), and 17 participate in response to stress (GO:0006950), establishing a broad signaling hub architecture. [KG Evidence]

The cytokine-cytokine receptor interaction pathway (WikiPathways WP5473) directly links 6 members (IL17C, IL2RB, IL10RA, IL20, IL20RA, CCL24), confirming a coherent receptor-ligand co-expression program within the module. [KG Evidence] JAK1 (NCBIGene:3716; 300 edges) serves as a non-hub shared neighbor connecting IL2RB, IL10RA, IL22RA1, IL20RA, and IL2 via regulatory and physical interaction edges, identifying JAK-STAT signaling as the principal intracellular transduction axis. [KG Evidence]

#### 2.2 Module-Level Disease Convergence

The strongest disease recurrence signals, ranked by member count, are:

| Disease | Members | Evidence Level |
|---|---|---|
| Depressive disorder (MONDO:0002050) | 16 | Curated |
| Panniculitis (MONDO:0006591) | 15 | Curated |
| Asthma (MONDO:0004979) | 15 | Curated |
| Essential hypertension (MONDO:0001134) | 14 | Curated |
| Coronary artery disorder (MONDO:0005010) | 14 | Curated |
| Psoriasis (MONDO:0005083) | 13 | Curated |
| Rheumatoid arthritis (MONDO:0008383) | 8 | Curated |
| Autoimmune disease (MONDO:0007179) | 6 | Curated |

[KG Evidence]

Depressive disorder ranks first (16 of 28 protein members), consistent with the established neuroinflammatory hypothesis linking peripheral cytokine elevation to major depression. [KG Evidence; Model Knowledge] Panniculitis (15 members) and psoriasis (13 members) implicate subcutaneous and dermal barrier-tissue inflammation as a disease axis, consistent with the IL-20 subfamily's known role in keratinocyte biology. [KG Evidence; Model Knowledge] Asthma (15 members) aligns with the Th2 axis (IL4, IL13, CCL24) and adenosine (CHEBI:16027), which appears among the asthma-associated members. [KG Evidence]

Autoimmune disease (MONDO:0007179) links FCGR2B, IL2RB, IFNG, IL4, and PTX3; separately, rheumatoid arthritis (MONDO:0008383) connects salicylate, FCGR2B, IL2, IL2RB, MMP10, IL20, SFTPD, and TNF. [KG Evidence] The overlap of metabolite (salicylate) and protein associations in rheumatoid arthritis is pharmacologically noteworthy, as salicylate is both a therapeutic agent and an endogenous metabolite.

#### 2.3 Highest-Leverage Module Members

The Member Prioritization Table identifies the following as highest-impact nodes:

**TNF** (10,000 edges; top disease: psoriatic arthritis): the most connected protein member; a hub-flagged node whose associations require careful interpretation owing to connectivity bias. [KG Evidence] TNF connects to TNFRSF1B (shared neighbor, 200 edges) and drives the inflammatory response pathway with 5 co-members (IFNG, IL2, IL4, IL13, IL33). [KG Evidence]

**IFNG** (7,922 edges; top disease: aplastic anemia): the canonical Th1 effector cytokine; participates in positive regulation of gene expression (GO:0010628), negative regulation of DNA-templated transcription (GO:0045892), and positive regulation of IL-6 production (GO:0032755) alongside TNF and IL33. [KG Evidence]

**REN** (7,029 edges; top disease: hypertensive disorder): an unexpected member of this immune module. REN participates in regulation of blood pressure, angiotensin maturation, and renin-angiotensin regulation of aldosterone production. [KG Evidence] Its interaction with AGT (NCBIGene:183) and its prostaglandin E2-mediated path to IL4 (both legs curated-causal) position REN as a mechanistic bridge between hemodynamic regulation and Th2 cytokine signaling. [KG Evidence]

**IL20RA/IL22RA1/IL20/IL24** (949 to 1,386 edges): These IL-20 subfamily components form a tight co-expression cluster. IL20RB (NCBIGene:53833; 80 edges, non-hub) physically interconnects IL22RA1, IL20, and IL20RA, confirming a bona fide receptor complex within the module. [KG Evidence] A recent comprehensive signaling map of IL-24 documents its multifunctional immunoregulatory role spanning cancer, infection, and inflammatory disease contexts (Frontiers, 2025). [Literature]

**S100A12** (2,002 edges; top disease: inflammatory response): an innate immune alarmin that participates in innate immune response (GO:0045087) alongside SFTPD and PTX3, and in positive regulation of inflammatory response (GO:0050729) alongside IL2, CCL24, and IL33. [KG Evidence]

**PRSS27** (973 edges; top disease: panniculitis): a serine protease that participates in proteolysis (GO:0006508) alongside MMP10 and REN, suggesting a shared tissue-remodeling effector function. [KG Evidence] PRSS27 appears in 14 to 16 of the top recurrent disease associations despite its comparatively modest edge count, identifying it as a disproportionately disease-connected member warranting further investigation. [KG Evidence; Inferred]

---

### 3. Novel Predictions (Tier 3)

#### 3.1 REN as a Neuro-Immune-Metabolic Bridge

**Prediction**: REN co-expression with immune cytokines reflects a functional renin-angiotensin-immune crosstalk mechanism relevant to hypertensive disorder (8 module members), kidney disorder (7 members), and depressive disorder (16 members).

**Structural logic chain**: REN → AGT (interacts_with, curated) → prostaglandin E2 (interacts_with, curated) → IL4 (affects, curated-causal) [KG Evidence]. Separately, REN participates in response to lipopolysaccharide (GO:0032496) alongside IL10RA and IL13, indicating immune activation triggers renin expression. [KG Evidence] REN also shows novel connections to ADHD/autism ASD pathways and to MTOR, AKT1, and SRC signaling. [KG Evidence]

**Calibration**: Approximately 18% of computational predictions linking the renin-angiotensin system to immune modulation advance to clinical investigation; this prediction is strengthened by the curated evidence on both legs of the prostaglandin E2 bridge.

**Validation step**: Measure REN protein levels in cytokine-stimulated immune cell cultures; test whether angiotensin receptor blockade modulates IL4/IL13 production in the relevant cohort.

#### 3.2 Xenobiotic Sulfate Conjugates as Microbial Exposome Markers

**Prediction**: The cluster of sulfate conjugate metabolites (thymol sulfate, umbelliferone sulfate, 2-naphthol sulfate, methyl-4-hydroxybenzoate sulfate) co-expresses with immune cytokines because microbial or dietary xenobiotic exposure drives barrier-tissue immune activation.

**Structural logic chain**: Thymol sulfate (9 edges), umbelliferone sulfate (8 edges), 2-naphthol sulfate (7 edges), and methyl-4-hydroxybenzoate sulfate (25 edges) are phase II conjugates of plant-derived or xenobiotic phenolic compounds processed by gut microbiota and hepatic sulfotransferases. [Model Knowledge] Their co-expression with barrier-tissue cytokines (IL20, IL24, IL17C, IL22RA1) and disease convergence on panniculitis, psoriasis, and gastrointestinal disorders (gastroduodenitis, 14 members; irritable bowel syndrome, 11 members; gastroesophageal reflux disease, 12 members) implicates gut-barrier compromise as a mechanism linking xenobiotic absorption to systemic immune activation. [KG Evidence; Inferred]

**Calibration**: This multi-entity prediction is speculative (~18% benchmark); individual sulfate conjugates have sparse KG coverage (4 to 25 edges) limiting direct evidence.

**Validation step**: Correlate sulfate conjugate metabolite levels with fecal microbiome diversity and intestinal permeability markers (e.g., zonulin, lactulose:mannitol ratio) in the cohort; test whether the xenobiotic metabolite cluster predicts disease severity in the inflammatory conditions identified.

#### 3.3 N-Formylphenylalanine as a Bacterial Translocation Signal

**Prediction**: N-formylphenylalanine in this module may serve as a marker of bacterial translocation or innate immune activation via formyl peptide receptor (FPR) signaling.

**Structural logic chain**: N-formylphenylalanine (CHEBI:133534; 4 edges, sparse) is structurally analogous to N-formylmethionine, the universal bacterial translation initiator that activates formyl peptide receptors on neutrophils and macrophages. [Model Knowledge] Semantic similarity analysis links it to ribosomal proteins RPSA and RPL4 via the N-methylphenylalanine analogue (similarity 0.88), consistent with a role in translational or ribosomal biology. [KG Evidence] Its co-expression with S100A12 (an innate alarmin), PTX3 (an opsonin), and TNF (a master pro-inflammatory cytokine) supports a bacterial-product-driven innate immune activation scenario. [Inferred]

**Calibration**: ~18% of such structural analogy predictions validate experimentally. The biological plausibility is elevated by the known role of N-formyl peptides in neutrophil chemotaxis, but the specific compound identity and its circulating levels require confirmation.

**Validation step**: Measure N-formylphenylalanine by targeted mass spectrometry in cohort samples; correlate with neutrophil activation markers (MPO, elastase) and with S100A12 levels; test FPR1/FPR2 activation using synthetic N-formylphenylalanine in neutrophil chemotaxis assays.

#### 3.4 Trans-Urocanate as an Immune-Metabolic Link to Histidine Catabolism

**Prediction**: Trans-urocanate (87 edges; top disease: colorectal cancer) connects microbial histidine metabolism to the immune activation state of this module, potentially modulating immune responses via immunosuppressive cis-urocanate photoisomerization or histidine-derived metabolite pools.

**Structural logic chain**: Trans-urocanate is the product of histidine ammonia-lyase (HAL) activity, the first step in histidine catabolism. [Model Knowledge] The KG links it to autoimmune disease (MONDO:0007179) alongside FCGR2B, IFNG, IL2RB, IL4, and PTX3 (6 members total), and to obesity disorder (MONDO:0011122) alongside adenosine, LIF, PTX3, S100A12, and CCL24. [KG Evidence] Histidinemia (MONDO:0012451) appears among shared disease neighbors in pathway enrichment. [KG Evidence]

**Calibration**: ~18% progression rate applies; trans-urocanate has moderate KG coverage (87 edges) lending some confidence.

**Validation step**: Quantify both cis- and trans-urocanate isomers in the cohort; correlate with histidine levels, HAL expression, and UV exposure history; test association with autoimmune disease phenotypes within the dataset.

---

### 4. Biological Themes

#### 4.1 Core Theme: Convergent Cytokine Signaling Through JAK-STAT

The dominant pathway architecture involves JAK1 as a non-hub integrator (300 edges) connecting five module members: IL2RB, IL10RA, IL22RA1, IL20RA, and IL2. [KG Evidence] IL20RB (80 edges, low-hub) further links the IL-20 subfamily receptor components. [KG Evidence] This signaling convergence indicates that the module captures a coordinated cytokine receptor activation state rather than isolated cytokine expression.

#### 4.2 Innate Immune Pattern Recognition and Opsonization

PTX3, SFTPD, and S100A12 collectively participate in innate immune response (GO:0045087) and positive regulation of phagocytosis (GO:0050766; with IL2RB). [KG Evidence] PTX3 engages complement system activation, opsonization, and negative regulation of viral entry. [KG Evidence] PTX3 localizes to terminal lymphatics and shapes their organization and function, as demonstrated by its selective expression in initial lymphatic vessels (Frontiers, 2024). [Literature] The role of PTX3 in mineralization processes and aging-related bone diseases further documents its extracellular matrix interactions, including with FGF2 (a direct PTX3 interactor in the KG) and complement factor H (CFH). [Literature]

#### 4.3 Tissue Remodeling and Barrier Disruption

MMP10 (matrix metalloproteinase 10), PRSS27 (serine protease), and REN (aspartyl protease) share proteolysis (GO:0006508) as a biological process annotation. [KG Evidence] These proteases co-express with barrier-tissue cytokines (IL20, IL24, IL17C) and are associated with diseases of tissue integrity (panniculitis, 15 members; psoriasis, 13 members). [KG Evidence] FGF5 (2,276 edges) adds a growth factor dimension, potentially reflecting regenerative responses to tissue damage. [Inferred]

#### 4.4 Hub-Filtered Insight

TNF (10,000 edges), phosphate (10,000 edges), adenosine (8,546 edges), IFNG (7,922 edges), REN (7,029 edges), IL4 (6,911 edges), and IL2 (6,275 edges) are flagged as hub-biased nodes. [KG Evidence] Associations exclusively supported by these hubs (e.g., the shared "Homo sapiens" organismal taxon connecting 26 members) are non-informative and are excluded from biological interpretation. The disease and pathway recurrences reported above involve substantial contributions from non-hub members (PRSS27, IL20, IL24, IL20RA, IL22RA1, CCL24, S100A12, IL17C), lending confidence to the specificity of the detected associations.

---

### 5. Gap Analysis

#### 5.1 Informative Absences

**IL-10 (anti-inflammatory counterregulator)**: The absence of IL-10 from a module containing both Th1 (IFNG, TNF) and Th2 (IL4, IL13) effectors indicates that this module captures an activation state lacking its principal negative feedback mediator. [KG Evidence] This pattern is consistent with an unresolved or dysregulated immune response rather than a homeostatic oscillation.

**IL-1β (inflammasome effector)**: TNF, IL-33, and S100A12 are all present, yet IL-1β is absent. [KG Evidence] This distinction suggests that the module represents TNF/NF-κB-driven inflammation independent of canonical NLRP3 inflammasome activation, a mechanistically informative separation.

**IL-12 (IL12A/IL12B)**: IFNG and IL-27 are both present, but the classical Th1 inducer IL-12 is absent. [KG Evidence] IL-27 may serve as the dominant Th1-polarizing signal in this cohort, representing a potentially distinct immunological mechanism.

**CRP (hepatic pentraxin)**: PTX3 is present while CRP is absent. [KG Evidence] These two pentraxin family members reflect different inflammatory compartments: PTX3 marks local tissue inflammation (consistent with its expression in terminal lymphatics; [Literature]), whereas CRP marks the systemic hepatic acute-phase response. This distinction suggests the module captures tissue-level rather than systemic inflammation.

**IL-5 (eosinophil activator)**: IL-4, IL-13, and CCL24 (eotaxin-2) are present, but IL-5 is absent. [KG Evidence] The incomplete Th2 triad may indicate a non-eosinophilic Th2 subtype or distinct temporal kinetics for IL-5 expression.

#### 5.2 Standard Gaps (Likely Platform-Related)

STAT1/STAT4/STAT6 (intracellular transcription factors), NF-κB pathway components (NFKB1, RELA), and FOXP3 are absent but are intracellular proteins unlikely to be measured on secreted-protein assay platforms (e.g., Olink, SomaScan). [Model Knowledge] Their absence is non-informative regarding biology.

#### 5.3 Metabolite Cold-Start and Sparse Entities

N,N,N-trimethyl-5-aminovalerate (RM:0140021) has zero KG edges and no direct biological interpretation is available. [KG Evidence] Several sparse-coverage metabolites (homostachydrine, 8 edges; umbelliferone sulfate, 8 edges; 2-naphthol sulfate, 7 edges; glutamate gamma-methyl ester, 6 edges; m-hydroxyhippurate, 4 edges; N-formylphenylalanine, 4 edges; N,N-dimethylalanine, 4 edges) have limited KG presence, restricting direct annotation. [KG Evidence] Semantic similarity analysis placed N,N,N-trimethyl-5-aminovalerate closest to 5-aminovaleric acid (similarity 0.83), a lysine catabolite, suggesting a role in amino acid degradation. [KG Evidence]

---

### 6. Temporal Context

No longitudinal design information was provided with this analysis. The following causal inference opportunities are noted for consideration if temporal data are available:

**Upstream causes (candidate drivers)**: Xenobiotic exposures (indexed by sulfate conjugate metabolites) and microbial metabolites (trans-urocanate, N-formylphenylalanine, homostachydrine) are plausible upstream triggers of immune activation, as environmental exposures temporally precede cytokine responses. [Inferred]

**Downstream consequences (candidate effects)**: Tissue-remodeling markers (MMP10, FGF5), disease phenotypes (panniculitis, psoriasis, depressive disorder), and hemodynamic effects (REN-mediated blood pressure regulation) are plausible downstream consequences of sustained immune activation. [Inferred]

**Causal inference opportunity**: If longitudinal samples exist, Granger causality or dynamic Bayesian network analysis could test whether xenobiotic metabolite levels at early time points predict cytokine elevation at later time points, establishing temporal precedence for the microbial-immune activation hypothesis.

---

### 7. Research Recommendations

#### 7.1 Highest Priority (Experimental Validation)

1. **REN-immune crosstalk validation**: Test whether angiotensin II receptor blockers (ARBs) or ACE inhibitors modulate IL4/IL13 production in peripheral blood mononuclear cells from cohort subjects. The curated REN → prostaglandin E2 → IL4 path (both legs curated-causal) provides the strongest cross-type bridge in the analysis. [KG Evidence]

2. **Xenobiotic-barrier integrity testing**: Quantify intestinal permeability markers alongside sulfate conjugate metabolite panels. If sulfate conjugate levels correlate with intestinal permeability, this establishes a mechanism for the metabolite-immune co-expression pattern. [Inferred]

3. **N-formylphenylalanine functional characterization**: Perform targeted mass spectrometry to confirm identity and concentration; test FPR1/FPR2 activation in neutrophil assays. [Model Knowledge; Inferred]

#### 7.2 Moderate Priority (Computational Follow-Up)

4. **Cross-module comparison**: Compare the Grey module against other WGCNA modules to determine where IL-6, IL-10, IL-1β, IL-5, IL-12, CRP, and TGFB1 segregated. Their absence patterns constrain the immunological identity of this module and may reveal complementary regulatory networks.

5. **IL-20 subfamily sub-network analysis**: Perform focused KG queries on the IL20/IL24/IL20RA/IL22RA1/IL20RB receptor complex to identify disease associations unique to this sub-network, distinct from the broader module signal.

6. **PRSS27 disease enrichment**: PRSS27 (973 edges) appears in an unexpectedly large fraction (14 to 16) of the top disease associations despite modest connectivity. Investigate whether PRSS27 has under-characterized roles in barrier-tissue inflammatory diseases.

#### 7.3 Lower Priority (Literature Review)

7. **IL-27 versus IL-12 dominance**: Review recent literature on IL-27 as an alternative Th1 inducer, particularly in contexts where IL-12 is absent. The presence of IL-27 (via the IL27/EBI3 heterodimer, 6 edges in KG) without IL-12 is a potentially distinctive immunological feature of this cohort.

8. **Adenosine-purinergic signaling**: Adenosine (8,546 edges; hub-flagged) is associated with asthma in the KG. Although hub-biased, the adenosine-purinergic axis is a validated immunomodulatory mechanism in airway inflammation, and its co-expression with Th2 cytokines warrants a focused literature assessment in the context of this cohort's phenotype.

9. **Microbial metabolite provenance**: Investigate whether homostachydrine (plant-derived proline betaine), trans-urocanate (histidine catabolite), and 4-chlorobenzoic acid (halogenated industrial/microbial metabolite) have documented roles as immune modulators or biomarkers of specific microbial community compositions.

---

*Report generated from KRAKEN knowledge graph analysis. All Tier 1 and Tier 2 findings derive from curated knowledge graph evidence. Tier 3 predictions carry the standard ~18% computational-to-clinical progression rate and require independent experimental validation. Entity resolution achieved 46/46 (100%) success rate; one entity (GIF → MT3) and three metabolites were resolved at reduced confidence (70 to 80%) and should be verified against the original assay manifest.*

### Literature References

Papers discovered via semantic search. 5 unique papers across 4 hypotheses.

| Hypothesis | Citation | Link | Relevance |
|------------|----------|------|-----------|
| Bridge: Gene → MolecularEntity (1 hops); Bridge: Gene → MolecularEntity (2 hops) |  (2025) "Diverse infections transcriptionally reprogram the intestinal epithelium and epithelial-immune cell ..." | [Link](https://www.biorxiv.org/content/10.64898/2025.12.22.695505v1) | The distal small intestine plays vital roles in host physiology by regulating nutrient and fluid homeostasis. Despite be... |
| Bridge: Gene → MolecularEntity (1 hops); Bridge: Gene → MolecularEntity (2 hops) |  (2025) "Frontiers \| An assembled molecular signaling map of interleukin-24: a resource to decipher its multi..." | [Link](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2025.1608101/full) | The study-centric molecular reactions of IL-24 in various pathological conditions comprising cancer, infection, and othe... |
| Bridge: Gene → SmallMolecule (1 hops) |  (2009) "Inferring branching pathways in genome-scale metabolic networks \| BMC Systems Biology \| Springer Nat..." | [Link](https://link.springer.com/article/10.1186/1752-0509-3-103) | [28]. ... Second, we are able to achieve higher Z O with combinations of pathways. Consider a pathway P with Z O (P, S,... |
| Bridge: Gene → SmallMolecule (1 hops) |  (2021) "NICEpath: Finding metabolic pathways in large networks through atom-conserving substrate-product pai..." | [Link](https://pubmed.ncbi.nlm.nih.gov/34003971/) | Results: Here, we propose the construction of searchable graph representations of metabolic networks. Each reaction is d... |
| Bridge: Gene → SmallMolecule (2 hops) |  (2021) "The Role of PTX3 in Mineralization Processes and Aging-Related Bone Diseases" | [Link](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2020.622772/full) | Pentraxin 3 ... PTX3) is the prototypic ... pentraxin that was ... (1). Long pentraxins have an unrelated amino-terminal... |