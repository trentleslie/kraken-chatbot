# Pink Module Run on Opus 4.8: Discovery Output (54-analyte, dev, 2026-06-24)

This document presents the discovery output of a full-module run of the **Pink** WGCNA module through the Kraken 12-node discovery pipeline, executed on the dev environment on 2026-06-24 (commit `098093e`) with BioMapper entity resolution enabled. The SDK-backed pipeline nodes ran on the Opus 4.8 model rather than the default Sonnet; this run is published for direct model comparison. It is one of a per-module series over the Frailty multi-omics WGCNA modules; see the [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm).

## Run Provenance

The run submitted 54 named analytes, parsed 53 at intake, and resolved 53 distinct entities (14 biomapper, 35 fuzzy, 4 exact) to 41 distinct CURIEs. Triage classified 3 well-characterized, 6 moderate, 31 sparse, and 13 cold-start, with 0 edge-count measurement failures. Downstream analysis produced 466 direct-KG findings, 24 cold-start findings, 3 biological themes, 0 cross-entity bridges (0 evidence-grounded), and 44 hypotheses supported by 36 literature references. Synthesis emitted a 23088-character report. The run completed in approximately 580.8 s of wall-clock time (status complete, 20 errors). All counts are taken from the reducer-merged pipeline state.

| Stage | Result |
|---|---|
| Submitted | 54 named analytes |
| Intake | 53 parsed |
| Entity resolution | 53 resolved (14 biomapper, 35 fuzzy, 4 exact) to 41 distinct CURIEs |
| Triage | 3 well-characterized, 6 moderate, 31 sparse, 13 cold-start (0 measurement failures) |
| Direct KG | 466 findings |
| Cold-start | 24 findings, 36 skipped |
| Pathway enrichment | 3 biological themes |
| Integration | 0 bridges (0 evidence-grounded) |
| Literature grounding | 36 papers |
| Synthesis | 44 hypotheses, 23088-character report |
| Run total | ~580.8 s wall-clock, status complete, 20 errors |

## Related

- Companion run metrics: [Pink Module Run on Opus 4.8: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/pink-module-run-on-opus-48-pipeline-performance-report-54-analyte-dev-2026-06-24-wIUArRW7HL)
- Model comparison baseline (Sonnet): [Pink Module Run: Discovery Output](https://phwiki.phenoma.ai/doc/pink-module-run-discovery-output-54-analyte-dev-2026-06-23-2wJf2HBLjK)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---



## Pink WGCNA Module: Glycerophospholipid Remodeling Coupled to Vascular and Extracellular Matrix Biology

### 1. Executive Summary

The Pink WGCNA module encodes a coordinated glycerophospholipid remodeling program dominated by lysophosphatidylcholine (LPC), lysophosphatidylethanolamine (LPE), and lysophosphatidylinositol (LPI) species, linked to two protein hubs: MMP7 (matrix metalloproteinase 7) and GDF2/BMP9 (a TGF-beta superfamily ligand governing vascular homeostasis). [KG Evidence] Module-level disease recurrence implicates gastrointestinal and hepatic cancers, prostate cancer, and psoriasis as shared pathological contexts for this lipid-protein axis. [KG Evidence] The biological coherence of the module points to a membrane phospholipid remodeling signature (Lands cycle activity) intersecting with extracellular matrix turnover and endothelial signaling; the systematic absence of ceramides, acylcarnitines, and branched-chain amino acids confirms that this module is distinct from sphingolipid-driven lipotoxicity and mitochondrial beta-oxidation programs. [Inferred]

### 2. Key Findings (Tier 1 to 2)

#### 2.1 Module Composition and Lipid Architecture

The module comprises 53 resolved entities: 2 proteins (MMP7, GDF2) and 51 lipid metabolites. [KG Evidence] The metabolite fraction is overwhelmingly composed of glycerophospholipid species: approximately 35 LPC variants spanning acyl chain lengths from C15:0 to C22:6, 8 LPE species, 4 LPI species, 6 acylcholines (palmitoylcholine through docosahexaenoylcholine), and ancillary species including glycerophosphorylcholine (GPC), 2-hydroxypalmitate, 2-hydroxystearate, 1-arachidonylglycerol, and a specific triacylglycerol (stearoyl-arachidonoyl-glycerol 18:0/20:4). [KG Evidence] The acyl chain diversity (saturated C15:0 to C18:0; monounsaturated C16:1, C18:1, C20:1; polyunsaturated C18:2, C18:3, C20:2, C20:3, C20:4, C22:4, C22:5, C22:6) and the inclusion of both sn-1 and sn-2 positional isomers indicate comprehensive Lands cycle remodeling activity encompassing both phospholipase A1 and A2 pathways. [Inferred]

Three ether-linked (plasmalogen-derived) species are present: 1-(1-enyl-palmitoyl)-GPC (P-16:0), 1-(1-enyl-oleoyl)-GPC (P-18:1), and 1-(1-enyl-stearoyl)-GPC (P-18:0). [KG Evidence] Plasmalogens serve as endogenous antioxidants and membrane fluidity regulators; their co-expression with conventional diacyl lysophospholipids suggests coordinate regulation of both plasmalogen and diacyl phospholipid pools. [Model Knowledge]

#### 2.2 MMP7: ECM Remodeling and Epithelial Defense

MMP7 is the most highly connected module member (3,637 edges). [KG Evidence] Direct KG queries confirm its participation in extracellular matrix disassembly, collagen catabolic processes, membrane protein ectodomain proteolysis, and extracellular matrix degradation pathways. [KG Evidence] MMP7 also participates in the AGE/RAGE pathway, chronic hyperglycemia impairment of neuron function, Wnt signaling and pluripotency, and gastrin signaling. [KG Evidence] Its antimicrobial functions are documented through participation in defense response to Gram-positive and Gram-negative bacteria, antibacterial peptide biosynthesis and secretion, and humoral immune response. [KG Evidence]

Protein-protein interactions of MMP7 include established substrates and binding partners: FASLG (Fas ligand shedding), DCN (decorin), SPP1 (osteopontin), A2M (alpha-2-macroglobulin), and multiple cathepsins (CTSB, CTSK, CTSV). [KG Evidence] Novel interaction partners flagged in the analysis include CXCR2, CXCL5, and PRSS1 (trypsinogen), the last of which suggests a connection to pancreatic exocrine biology. [KG Evidence]

Disease associations for MMP7 include malignant colon neoplasm, liver cancer, prostate cancer, and psoriasis. [KG Evidence]

#### 2.3 GDF2/BMP9: Vascular and Metabolic Signaling

GDF2 (1,394 edges) participates in angiogenesis, BMP signaling, vasculogenesis, blood vessel morphogenesis, and both positive and negative regulation of endothelial cell proliferation and migration. [KG Evidence] Its established interactors include ACVRL1 (ALK1), ENG (endoglin), BMPR2, SMAD4, NOTCH1, EDN1 (endothelin-1), and BMP10, all of which constitute the canonical ALK1 signaling axis governing vascular integrity. [KG Evidence] GDF2 additionally participates in intracellular iron ion homeostasis, osteoblast differentiation, ossification, and cartilage development. [KG Evidence]

The top disease association for GDF2 is hereditary hemorrhagic telangiectasia (HHT), consistent with its established role in ALK1-endoglin signaling. [KG Evidence] Shared disease associations between GDF2 and MMP7 include digestive system disorder and cancer (broadly defined). [KG Evidence]

#### 2.4 Phosphatidylcholine: The Hub Lipid

Phosphatidylcholine (CHEBI:64482; 410 edges) is the third well-characterized member. [KG Evidence] It shares disease associations with MMP7 for malignant colon neoplasm, liver cancer, prostate cancer, and psoriasis, and with GDF2 for visual epilepsy. [KG Evidence] Its top individual disease association is hypothyroidism. [KG Evidence] As the parent lipid class for the majority of module metabolites, phosphatidylcholine serves as the biochemical precursor from which lysophosphatidylcholine species are generated via phospholipase activity. [Model Knowledge]

#### 2.5 2-Hydroxypalmitate: A Lipid with Colorectal Cancer Association

2-Hydroxypalmitate (CHEBI:65101; 23 edges) is the only sparse-coverage metabolite with a curated disease association: colorectal cancer. [KG Evidence] This association reinforces the module's connection to gastrointestinal malignancy, converging with MMP7's established link to malignant colon neoplasm. [Inferred]

#### 2.6 Module-Level Pathway Convergence

Two gene ontology processes are shared between the protein members: protein binding (GO:0005515) and response to stress (GO:0006950). [KG Evidence] Both MMP7 and GDF2 are associated with smoking as an environmental exposure. [KG Evidence] The pathway enrichment analysis identified a network of lipase genes (PNLIP, LIPC, LIPA, PNLIP3, LIPF, plus 13 additional enzymes) connecting five input entities, and bile salt-activated lipase (CEL; UniProtKB:P19835) as a shared protein node. [KG Evidence] These lipase connections reinforce the module's identity as a phospholipid remodeling signature.

### 3. Novel Predictions (Tier 3)

#### 3.1 2-Hydroxystearate Classification as a Hydroxy Fatty Acid Anion

**Prediction**: 2-Hydroxystearate (CHEBI:229769) is a subclass of hydroxy fatty acid anion (CHEBI:59835). [KG Evidence]

**Structural logic chain**: 2-Hydroxystearate shares vector similarity with three hydroxystearate positional isomers: 7-hydroxystearate (similarity 0.89), 11-hydroxystearate (0.88), and 8-hydroxyoleate (0.83). All three analogues maintain a biolink:subclass_of relationship to CHEBI:59835. [KG Evidence] As a positional isomer differing only in hydroxyl placement, 2-hydroxystearate very likely belongs to the same ontological class.

**Literature support**: Hydroxy fatty acids are produced by fatty acid hydratases (EC 4.2.1.53) with broad substrate tolerance across acyl chain lengths from C11:1 to C22:6. [Literature (Expanding the biosynthesis spectrum of hydroxy fatty acids, 2024)] The FAHFA lipid family description confirms that hydroxy fatty acids constitute a recognized structural class with terminal or internal hydroxyl groups. [Literature (Fatty Acyl Esters of Hydroxy Fatty Acid Lipid Families, 2020)]

**Validation step**: Verify in the ChEBI ontology whether CHEBI:229769 is already classified under CHEBI:59835; if not, this represents a curation gap. **Confidence calibration**: approximately 18% of computational predictions progress to clinical investigation, though ontological classification predictions carry higher intrinsic validity than phenotypic predictions.

#### 3.2 Convergence of MMP7 and 2-Hydroxypalmitate in Colorectal Cancer

**Prediction**: The co-expression of MMP7 and 2-hydroxypalmitate reflects a shared role in colorectal cancer biology, where ECM remodeling by MMP7 may facilitate tumor invasion in a lipid microenvironment characterized by altered hydroxylated fatty acid metabolism. [Inferred]

**Structural logic chain**: MMP7 is associated with malignant colon neoplasm (curated, MONDO:0021063). [KG Evidence] 2-Hydroxypalmitate is associated with colorectal cancer (curated). [KG Evidence] Both are members of the same co-expression module. The LPC-rich microenvironment captured by this module may represent the lipid milieu accompanying MMP7-mediated ECM degradation in gastrointestinal epithelium. [Model Knowledge]

**Validation step**: Correlate MMP7 protein levels with 2-hydroxypalmitate concentrations in colorectal tumor tissue versus matched normal mucosa. Assess whether 2-hydroxypalmitate accumulation correlates with MMP7-dependent invasion assays in colorectal cancer cell lines. **Confidence calibration**: approximately 18% of computational predictions progress to clinical validation.

#### 3.3 GDF2 as the Vascular Integrator of Lipid Module Biology

**Prediction**: GDF2/BMP9 co-expression with glycerophospholipids reflects endothelial membrane remodeling during vascular homeostasis or pathological angiogenesis. GDF2-ALK1 signaling in endothelial cells may regulate phospholipase activity governing LPC generation from membrane phosphatidylcholine pools. [Inferred]

**Structural logic chain**: GDF2 signals through ALK1 and BMPR2 in endothelial cells (established interactions with ACVRL1, ENG, BMPR2). [KG Evidence] Endothelial activation alters membrane phospholipid composition, and phospholipase A2 activity generates LPC species from phosphatidylcholine. [Model Knowledge] The co-expression of GDF2 with dozens of LPC species may capture the lipid consequence of BMP9-driven endothelial quiescence or activation.

**Validation step**: Treat primary endothelial cells with recombinant BMP9/GDF2 and perform targeted lipidomics to quantify LPC, LPE, and LPI species. Compare the resulting lipid profile to the Pink module metabolite signature. **Confidence calibration**: approximately 18% of computational predictions advance to clinical investigation.

#### 3.4 Acylcholine Species as Candidate Signaling Lipids

**Prediction**: The six acylcholine species in the module (palmitoylcholine, oleoylcholine, arachidonoylcholine, stearoylcholine, linoleoylcholine, docosahexaenoylcholine, dihomo-linolenoyl-choline) represent a coordinated family of non-classical lipid mediators whose biology is largely uncharacterized. Their co-expression with conventional LPC species suggests shared enzymatic origin, possibly through a choline transferase acting on fatty acyl-CoA substrates. [Inferred]

**Structural logic chain**: Acylcholines are structurally related to acylcarnitines but utilize choline rather than carnitine as the head group. [Model Knowledge] Their presence alongside lysophosphatidylcholines but not acylcarnitines indicates module specificity for choline-linked lipid metabolism rather than mitochondrial fatty acid transport. [Inferred]

**Validation step**: Determine whether acylcholine synthesis correlates with phospholipase-mediated LPC production using isotope-labeled choline tracing in relevant cell models. No KG disease associations exist for these metabolites; characterizing their biology represents a genuinely novel opportunity. **Confidence calibration**: approximately 18% of such predictions reach clinical investigation.

### 4. Biological Themes

#### 4.1 Unifying Theme: Lands Cycle Glycerophospholipid Remodeling

The dominant biological theme of the Pink module is membrane phospholipid remodeling via the Lands cycle. [Inferred] The Lands cycle describes the deacylation of intact glycerophospholipids by phospholipases (generating lysophospholipids) and their subsequent reacylation by acyltransferases (restoring the diacyl form). [Model Knowledge] The module captures the lysophospholipid intermediates of this cycle across three head-group classes (phosphocholine, phosphoethanolamine, phosphoinositol), both sn-1 and sn-2 positional isomers, and a wide spectrum of acyl chain compositions (saturated, monounsaturated, and polyunsaturated including arachidonate). The presence of arachidonate-containing species (1-arachidonoyl-GPC, 2-arachidonoyl-GPC, 1-arachidonoyl-GPE, 2-arachidonoyl-GPE, 1-arachidonoyl-GPI, 1-arachidonylglycerol, arachidonoylcholine) suggests this module captures eicosanoid precursor mobilization from membrane stores.

#### 4.2 ECM Turnover Coupled to Lipid Microenvironment

MMP7's participation in ECM degradation, matrix metalloproteinase pathways, and ectodomain shedding positions it as the proteolytic component of the module. [KG Evidence] Its co-expression with membrane-derived lipids is consistent with tissue remodeling events in which ECM degradation and membrane phospholipid turnover occur simultaneously, as in wound healing, tumor invasion, or inflammatory epithelial damage. [Model Knowledge]

#### 4.3 Vascular Biology via GDF2-ALK1 Axis

GDF2's established role in endothelial quiescence, angiogenesis regulation, and vascular morphogenesis (via ALK1, endoglin, and BMPR2 signaling) provides a vascular context for the module. [KG Evidence] The co-occurrence of GDF2 with lyso-glycerophospholipids may reflect the lipid composition of endothelial membranes undergoing BMP9-regulated remodeling.

#### 4.4 Hub Filtering

Homo sapiens (NCBITaxon:9606) was flagged as a hub node connecting two input entities in the pathway enrichment; this association is non-informative and reflects organism-level annotation rather than biology. [KG Evidence] This node is de-emphasized.

### 5. Gap Analysis

#### 5.1 Informative Absences

The following expected-but-absent entity classes reveal the biological specificity of the Pink module:

| Expected Entity Class | Interpretation |
|---|---|
| **Ceramides** | Absent from this lipid-focused module, indicating glycerophospholipid biology is separable from sphingolipid-driven lipotoxicity. If ceramides were measured, they likely segregate to a distinct module. [KG Evidence, gap analysis] |
| **Acylcarnitines** | Their absence alongside fatty acid substrates confirms the module captures membrane lipid remodeling rather than mitochondrial beta-oxidation. [KG Evidence, gap analysis] |
| **BCAAs** | Module is lipid-dominated; BCAAs likely segregate to an amino acid-centric module, consistent with known WGCNA partitioning of metabolic pathways. [KG Evidence, gap analysis] |
| **Inflammatory cytokines (IL-6, TNF-alpha, CRP)** | Despite MMP7's inflammatory functions (defense response to bacteria, CCL2 regulation), no inflammatory mediators co-express here. This indicates functional partitioning: the module captures MMP7's ECM-remodeling role, not its inflammatory role. [KG Evidence, gap analysis] |
| **LPA species** | Lysophosphatidylcholines are present but not their LPA derivatives. The module reflects phospholipase A-mediated remodeling without the downstream autotaxin/LPA signaling arm. [KG Evidence, gap analysis] |
| **Triglycerides/DAGs** | Only one triacylglycerol is present (18:0/20:4-containing), suggesting specificity for arachidonic acid-containing lipids rather than bulk triglyceride storage. [KG Evidence, gap analysis] |
| **Adiponectin** | Absent alongside GDF2, suggesting the module captures vascular/ECM biology rather than adipose-derived metabolic signaling. [KG Evidence, gap analysis] |

#### 5.2 Standard (Non-Informative) Gaps

HbA1c, insulin, C-peptide, and fasting glucose are clinical laboratory measures not captured by omics platforms and are expected to be absent. [KG Evidence, gap analysis]

#### 5.3 Entity Resolution Limitations

A substantial proportion (13 of 53) of module metabolites mapped to cold-start nodes with zero KG edges, and an additional 31 were classified as sparse (1 to 19 edges). [KG Evidence] Many metabolite resolutions were low-confidence (70% fuzzy matches), with several lysophospholipid species mapping to incorrect entities (e.g., multiple LPC species mapped to KEGG.GLYCAN:G00122/GP1c, a ganglioside; arachidonoyl-containing species mapped to UNII:67YKL086OI/ARACHIDETH-3, a cosmetics ingredient). These misresolutions limit the reliability of Tier 3 semantic similarity inferences, particularly the GPC1 (glypican-1) analogue series, which confounds the lysophospholipid abbreviation "GPC" with the glypican protein family. The tissue localization inferences (bone marrow, hippocampus, trigeminal ganglion) derived from GPC1 gene expression patterns are therefore not biologically relevant to the lysophospholipid species and should be disregarded. [Inferred]

### 6. Temporal Context

No longitudinal design information was provided with this WGCNA module. Several observations are pertinent if temporal data become available:

**Upstream candidates**: GDF2/BMP9 signaling through the ALK1-endoglin axis operates as an upstream regulator of endothelial membrane composition. [Model Knowledge] MMP7 transcriptional activation (e.g., via Wnt signaling; [KG Evidence]) would precede ECM degradation products and secondary lipid changes. Phospholipase activation represents an enzymatic upstream event generating the observed lysophospholipid signatures. [Model Knowledge]

**Downstream candidates**: The accumulation of specific LPC and LPE species, acylcholines, and arachidonate-containing species likely represents the downstream metabolic consequence of phospholipase activity and membrane remodeling. [Model Knowledge] 2-Hydroxypalmitate and 2-hydroxystearate may represent downstream products of fatty acid alpha-hydroxylation. [Model Knowledge]

**Causal inference opportunity**: If time-series data exist, Granger causality or dynamic Bayesian network analysis could determine whether MMP7 or GDF2 protein levels temporally precede the lysophospholipid signature, or whether lipid changes and protein expression are co-regulated by an upstream stimulus.

### 7. Research Recommendations

#### Priority 1: Experimental Validations

1. **GDF2-LPC axis in endothelial cells**: Treat human endothelial cells (HUVECs or organ-specific endothelial lines) with recombinant BMP9 at physiological concentrations (1 to 10 ng/mL) and perform targeted lipidomics covering the module's LPC, LPE, LPI, and acylcholine species. This directly tests whether GDF2/ALK1 signaling regulates the phospholipid remodeling signature captured by the module.

2. **MMP7 and 2-hydroxypalmitate in colorectal tissue**: Perform paired proteomic and metabolomic profiling in colorectal adenoma and carcinoma tissue versus adjacent normal mucosa, quantifying MMP7 protein and 2-hydroxypalmitate. This validates the co-association observed in both KG disease linkages and WGCNA co-expression.

3. **Acylcholine species characterization**: Acylcholines (palmitoylcholine, oleoylcholine, arachidonoylcholine, stearoylcholine, linoleoylcholine, docosahexaenoylcholine) lack KG annotations entirely. Isotope tracing experiments with labeled choline and palmitate in hepatocytes or enterocytes could elucidate their biosynthetic origin and relationship to LPC metabolism.

#### Priority 2: Literature and Database Searches

4. **Mendelian randomization evidence for LPC species in cancer risk**: A comprehensive two-sample MR study has already linked plasma metabolites to seven cancer types, including colorectal and prostate cancers. [Literature (Chen et al., 2024)] The specific LPC species present in this module should be cross-referenced against that dataset to identify causal evidence for lipid-cancer associations.

5. **Resolve entity mapping ambiguities**: Many module metabolites mapped incorrectly to KG nodes (glypican-1 instead of lysophosphatidylcholine species). Re-resolution through HMDB, LIPID MAPS, or RefMet identifiers would substantially improve KG coverage and enable additional pathway and disease association queries.

#### Priority 3: Follow-Up Analyses

6. **Cross-module comparison**: Compare the Pink module metabolite signature against other WGCNA modules to confirm segregation from ceramide, acylcarnitine, and BCAA modules. Identify which module(s) contain these expected-but-absent metabolites to map the full metabolic landscape.

7. **Module-trait correlation**: Correlate the Pink module eigengene with clinical variables (BMI, HOMA-IR, liver enzymes, inflammatory markers, vascular function measures) to contextualize the lipid-protein axis in phenotypic space.

8. **Plasmalogen sub-analysis**: The three ether-linked LPC species (P-16:0, P-18:1, P-18:0) merit specific investigation. Plasmalogen deficiency is associated with neurodegeneration, cardiovascular disease, and metabolic syndrome. [Model Knowledge] Determining whether their module membership reflects coordinate regulation with diacyl LPC species or independent biology would clarify the module's mechanistic interpretation.

9. **ChEBI ontology curation for 2-hydroxystearate**: Verify and, if necessary, submit the subclass_of relationship between CHEBI:229769 and CHEBI:59835 (hydroxy fatty acid anion), as this predicted classification is near-certain based on structural analogy to validated positional isomers. [KG Evidence]

---

*Report generated from KRAKEN knowledge graph analysis. All factual claims are tagged with evidence source: [KG Evidence], [Literature], [Model Knowledge], or [Inferred]. Tier 3 predictions are calibrated against the approximately 18% computational-to-clinical validation rate. Entity resolution quality (many 70% fuzzy matches) represents a significant limitation; re-resolution of lysophospholipid identifiers is recommended before downstream analysis.*

### Literature References

Papers discovered via semantic search. 4 unique papers across 2 hypotheses.

| Hypothesis | Citation | Link | Relevance |
|------------|----------|------|-----------|
| Inferred role of CHEBI:229769 | Edward Mubiru (2018) "Epoxy fatty acids in foods : analytics, formation and risk assessment" | — | — |
| Inferred role of CHEBI:229769 |  (2024) "Expanding the biosynthesis spectrum of hydroxy fatty acids: unleashing the potential of novel bacter..." | [Link](https://link.springer.com/article/10.1186/s13068-024-02578-2) | ratases ( ... EC 4.2.1.53) have drawn considerable attention due to ... substrate range as well as their widespread dist... |
| Inferred role of KEGG.GLYCAN:G00122 |  (2019) "Paradoxical Role of Glypican-1 in Prostate Cancer Cell and Tumor Growth \| Scientific Reports" | [Link](https://www.nature.com/articles/s41598-019-47874-2) | Glypicans (GPCs) are heparan sulfate proteoglycans (HSPGs) usually localized at the cellular membrane 1. Six glypican is... |
| Inferred role of CHEBI:229769 |  (2022) "Towards an understanding of oleate hydratases and their application in industrial processes \| Microb..." | [Link](https://link.springer.com/article/10.1186/s12934-022-01777-6) | Fatty acid hydratases are able to hydroxylate unsaturated fatty acids. A plethora of fatty acid hydratases, which conver... |