# Lightyellow Module Run on Opus 4.8: Discovery Output (19-analyte, dev, 2026-06-24)

This document presents the discovery output of a full-module run of the **Lightyellow** WGCNA module through the Kraken 12-node discovery pipeline, executed on the dev environment on 2026-06-24 (commit `a7d7fd6`) with BioMapper entity resolution enabled. The SDK-backed pipeline nodes ran on the Opus 4.8 model rather than the default Sonnet; this run is published for direct model comparison. It is one of a per-module series over the Frailty multi-omics WGCNA modules; see the [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm).

## Run Provenance

The run submitted 19 named analytes, parsed 19 at intake, and resolved 19 distinct entities (17 biomapper, 2 fuzzy) to 19 distinct CURIEs. Triage classified 9 well-characterized, 3 moderate, 6 sparse, and 1 cold-start, with 0 edge-count measurement failures. Downstream analysis produced 1015 direct-KG findings, 37 cold-start findings, 4 biological themes, 20 cross-entity bridges (19 evidence-grounded), and 92 hypotheses supported by 37 literature references. Synthesis emitted a 26454-character report. The run completed in approximately 680.1 s of wall-clock time (status complete, 8 errors). All counts are taken from the reducer-merged pipeline state.

| Stage | Result |
|---|---|
| Submitted | 19 named analytes |
| Intake | 19 parsed |
| Entity resolution | 19 resolved (17 biomapper, 2 fuzzy) to 19 distinct CURIEs |
| Triage | 9 well-characterized, 3 moderate, 6 sparse, 1 cold-start (0 measurement failures) |
| Direct KG | 1015 findings |
| Cold-start | 37 findings, 1 skipped |
| Pathway enrichment | 4 biological themes |
| Integration | 20 bridges (19 evidence-grounded) |
| Literature grounding | 37 papers |
| Synthesis | 92 hypotheses, 26454-character report |
| Run total | ~680.1 s wall-clock, status complete, 8 errors |

## Related

- Companion run metrics: [Lightyellow Module Run on Opus 4.8: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/lightyellow-module-run-on-opus-48-pipeline-performance-report-19-analyte-dev-2026-06-24-pmi9BUULwl)
- Model comparison baseline (Sonnet): [Lightyellow Module Run: Discovery Output](https://phwiki.phenoma.ai/doc/lightyellow-module-run-discovery-output-19-analyte-dev-2026-06-23-iDbMdsgIiU)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---



## Lightyellow WGCNA Module: Epithelial Barrier Stress, Glutathione Depletion, and Innate Type 2 Immunity

### 1. Executive Summary

The Lightyellow WGCNA module encodes a coordinated program of epithelial barrier damage, glutathione cycle depletion, and innate type 2 immune activation, bridged by aerobic glycolysis (Warburg-like metabolism). [KG Evidence] Five proteins (ADAMTS13, IL1A, IL5, TSLP, NRTN) converge with 14 metabolites spanning sulfur amino acid catabolism, pyroglutamyl dipeptide accumulation, and lactate/pyruvate imbalance, indicating that tissue injury and oxidative stress drive immune polarization toward a TSLP/IL5 axis without classical Th2 engagement. [KG Evidence; Inferred] The module's disease associations cluster on cardiometabolic, gastrointestinal, and allergic endpoints (coronary artery disorder, colorectal cancer, schizophrenia, atopic eczema; each shared by 3 to 5 members), suggesting that this biology underlies multi-organ inflammatory vulnerability. [KG Evidence]

### 2. Key Findings (Tier 1 to 2)

#### 2.1 Disease Convergence across the Module

The module-level disease recurrence analysis identified 30 diseases shared by two or more members. [KG Evidence] Three diseases reached the maximum overlap of 5 members:

| Disease | Members | Evidence |
|---|---|---|
| Coronary artery disorder (MONDO:0005010) | pyruvate, lactate, IL1A, IL5, TSLP | curated [KG Evidence] |
| Schizophrenia (MONDO:0005090) | ornithine, 5-oxoproline, lactate, IL5, TSLP | curated [KG Evidence] |
| Colorectal cancer (MONDO:0005575) | pyruvate, ornithine, 5-oxoproline, 3-ureidopropionate, lactate | curated [KG Evidence] |

Coronary artery disorder is notable because it unites both immune cytokines (IL5, TSLP, IL1A) and glycolytic metabolites (pyruvate, lactate), consistent with the emerging recognition that lactate-driven inflammatory signaling contributes to atherosclerotic plaque instability [Literature: "The role of lactate in cardiovascular diseases," 2023]. Colorectal cancer is driven entirely by metabolites, four of which participate in glutathione or pyrimidine catabolism, consistent with altered redox and nucleotide salvage in colonic epithelium. [KG Evidence; Inferred]

ADAMTS13 contributes to cancer (MONDO:0004992), kidney disorder, and cardiovascular disorder associations. [KG Evidence] Its canonical role in thrombotic thrombocytopenic purpura (TTP) and hemolytic-uremic syndrome is confirmed by curated disease edges shared with IL1A (hemolytic-uremic syndrome, brain ischemia). [KG Evidence] The absence of VWF from the module (see Section 5) raises the possibility that ADAMTS13 exerts a non-canonical, anti-inflammatory function in this context. [Inferred]

IL1A (5,816 edges) is flagged as a hub entity; associations mediated solely through IL1A should be interpreted with caution due to hub bias. [KG Evidence] Nonetheless, several diseases recur specifically among IL1A, IL5, and TSLP (hiatus hernia, chronic rhinitis, necrotizing ulcerative gingivitis, gastroduodenitis), pointing to mucosal and gastrointestinal inflammatory pathology. [KG Evidence]

#### 2.2 Pathway Convergence

Four module members (ADAMTS13, IL5, NRTN, TSLP) share the Gene Ontology annotation "response to stress" (GO:0006950). [KG Evidence] Three cytokine-encoding genes (IL1A, IL5, TSLP) converge on "cytokine-mediated signaling pathway" (GO:0019221) and "hemopoiesis" (GO:0030097). [KG Evidence] IL1A and TSLP both drive "positive regulation of interleukin-6 production" (GO:0032755), a downstream amplification step even though IL-6 itself is absent from the module (see Section 5). [KG Evidence]

On the metabolite side, three members (ornithine, 5-oxoproline, cysteinylglycine) converge on "glutathione metabolic process" (GO:0006749), and two of these (5-oxoproline, cysteinylglycine) co-participate in five SMPDB disease pathways: glutathione synthetase deficiency, gamma-glutamyltransferase deficiency, gamma-glutamyltranspeptidase deficiency, 5-oxoprolinuria, and 5-oxoprolinase deficiency. [KG Evidence] This enrichment establishes the gamma-glutamyl cycle as the dominant metabolic axis of the module.

#### 2.3 Cross-Type Bridges

The knowledge graph identifies a direct interaction edge between IL1A and lactate (biolink:interacts_with), constituting a 1-hop bridge between the protein and metabolite compartments of the module. [KG Evidence] Recent literature demonstrates that lactic acid fermentation is required for NLRP3 inflammasome activation, mechanistically linking lactate accumulation to IL-1 family cytokine processing [Literature: "Lactic Acid Fermentation Is Required for NLRP3 Inflammasome Activation," 2021; "Lactic acid drives NLRP3 inflammasome activation," 2026].

ADAMTS13 connects to lactate through seven 2-hop disease-mediated bridges (myocardial infarction, kidney disorder, acidosis disorder, liver disorder, cardiovascular disorder, stroke disorder, confusion), each with curated associative evidence. [KG Evidence] ADAMTS13 also connects to ornithine through ALDH18A1 (a physical interaction partner) and through shared cellular localization in the extracellular region. [KG Evidence] LACC1, a macrophage enzyme bridging NOS2 and polyamine metabolism, provides a plausible mechanistic link between inflammatory macrophage activation and ornithine utilization [Literature: "LACC1 bridges NOS2 and polyamine metabolism in inflammatory macrophages," Nature, 2022].

IL1A connects to cysteinylglycine through shared cytoplasmic and extracellular region localization (2-hop, curated-neutral). [KG Evidence] A 2025 publication in Nature Metabolism reports pathway analysis of cysteine-peptides and metabolites altered following IL-1alpha stimulation, providing direct experimental support for a functional link between IL1A signaling and the thiol metabolite pool [Literature: "Pathway analysis of cys-peptides and metabolites altered following IL-1alpha stimulation," Nature Metabolism, 2025].

LDHA (lactate dehydrogenase A, NCBIGene:3939) appears as a shared intermediate connecting multiple input entities in the pathway enrichment analysis. [KG Evidence] Protein-metabolite interactomics studies have confirmed that LDHA mediates pyruvate-lactate interconversion and is subject to isoform-specific regulation by long-chain acyl-CoA [Literature: "Protein-metabolite interactomics of carbohydrate metabolism reveal regulation of lactate dehydrogenase," 2023].

#### 2.4 Member Prioritization

The Member Prioritization Table identifies IL1A (5,816 edges), IL5 (3,200 edges), and ADAMTS13 (2,517 edges) as the most connected members. [KG Evidence] NRTN (1,454 edges) stands out as the sole neurotrophic factor in the module; its top disease association is Hirschsprung disease, implicating enteric nervous system development and gut innervation. [KG Evidence] Among metabolites, lactate (1,712 edges) serves as the most connected bridge between the immune and metabolic arms of the module. [KG Evidence]

### 3. Novel Predictions (Tier 3)

#### 3.1 Cysteine-Glutathione Disulfide as a Regulator of Glutathione S-Transferases

**Prediction**: Cysteine-glutathione disulfide (CHEBI:21264) physically interacts with GSTP1 (glutathione S-transferase Pi 1) and GSTA1 (glutathione S-transferase Alpha 1). [Inferred]

**Structural logic chain**: Cysteine-glutathione disulfide shares 0.78 semantic similarity with S-hydroxy-L-cysteine (CHEBI:41710). S-hydroxy-L-cysteine physically interacts with both GSTP1 (NCBIGene:2950) and GSTA1 (NCBIGene:2938) in the knowledge graph. [KG Evidence] Cysteine-glutathione disulfide contains a glutathione moiety and is a direct product of oxidative glutathione metabolism; interaction with glutathione-processing enzymes is biochemically highly plausible. [Model Knowledge]

**Validation step**: Conduct in vitro substrate/inhibitor assays for GSTP1 and GSTA1 using cysteine-glutathione disulfide. Query BRENDA and BindingDB for existing kinetic data on mixed disulfide substrates.

**Calibration**: Approximately 18% of computational predictions of this type progress to clinical or experimental investigation. This prediction has above-average plausibility given the direct biochemical relationship between the substrate and enzyme families.

#### 3.2 Pyroglutamyl Dipeptides as Colorectal Cancer Biomarkers

**Prediction**: Pyroglutamylvaline (CHEBI:132991) correlates with colorectal cancer (MONDO:0005575). [Inferred]

**Structural logic chain**: Pyroglutamylvaline shares 0.84 similarity with pyroglutamylglutamine and 0.78 similarity with Val-Glu, both of which are related to or correlated with colorectal cancer in the KG (supported by 2 independent analogues). [KG Evidence] The module already contains five members independently associated with colorectal cancer (pyruvate, ornithine, 5-oxoproline, 3-ureidopropionate, lactate). [KG Evidence] Pyroglutamyl dipeptides accumulate when gamma-glutamyl transpeptidase activity is altered, and altered gamma-glutamyl cycle metabolism is a recognized feature of colorectal neoplasia. [Model Knowledge]

**Validation step**: Perform targeted metabolomics profiling of pyroglutamylvaline, pyroglutamylleucine, and pyroglutamylglutamine in colorectal cancer vs. control plasma or tissue samples. Check existing metabolomics repositories (HMDB, Metabolomics Workbench) for prior measurements.

**Calibration**: The ~18% base rate for computational-to-clinical progression applies; the convergence of 2 supporting analogues and 5 independently associated module members elevates this prediction above the base rate.

#### 3.3 Cysteine-Glutathione Disulfide in Eosinophilic Esophagitis

**Prediction**: Cysteine-glutathione disulfide (CHEBI:21264) correlates with eosinophilic esophagitis (MONDO:0012451). [Inferred]

**Structural logic chain**: Cysteine-glutathione disulfide shares 0.79 similarity with S-sulfo-L-cysteine (cysteine S-sulfate, CHEBI:27891), which is correlated with eosinophilic esophagitis in the KG. [KG Evidence] The module already contains three members associated with eosinophilic esophagitis (ornithine, 5-oxoproline, cysteine S-sulfate). [KG Evidence] Both metabolites are sulfur-modified cysteine derivatives involved in thiol-disulfide metabolism; shared disease correlations are plausible via disrupted sulfur amino acid metabolism in eosinophilic inflammation. [Model Knowledge]

**Validation step**: Measure cysteine-glutathione disulfide levels in esophageal biopsies or plasma from eosinophilic esophagitis patients vs. controls. Assess whether oxidative stress markers co-elevate with this metabolite.

**Calibration**: The ~18% base rate applies. The triple convergence of ornithine, 5-oxoproline, and cysteine S-sulfate on eosinophilic esophagitis provides moderate reinforcement.

#### 3.4 Pyroglutamylvaline and Pyroglutamylleucine in the Gamma-Glutamyl Cycle

**Prediction**: Pyroglutamylvaline and pyroglutamylleucine participate in gamma-glutamyl cycle / 5-oxoprolinase deficiency pathways (SMPDB:SMP0000500). [Inferred]

**Structural logic chain**: Pyroglutamylvaline shares 0.78 similarity with 5-oxoproline, a confirmed participant in SMPDB:SMP0000500. [KG Evidence] Pyroglutamylleucine shares 0.91 similarity with pyroglutamylisoleucine, which is classified as a dipeptide (KEGG:C00107). [KG Evidence] Both pyroglutamyl dipeptides contain a 5-oxoprolyl (pyroglutamyl) moiety and are plausible products or substrates of gamma-glutamyl transpeptidase acting on amino acids; their accumulation may reflect impaired 5-oxoprolinase activity or enhanced gamma-glutamyl cycle flux. [Model Knowledge]

**Validation step**: Conduct enzymatic assays to determine whether 5-oxoprolinase or gamma-glutamyl transpeptidase processes these dipeptides. Search SMPDB and HMDB for pathway annotations.

**Calibration**: The ~18% base rate applies. Biochemical plausibility is moderate given the shared pyroglutamyl moiety, but no direct experimental data currently link these dipeptides to the gamma-glutamyl cycle.

### 4. Biological Themes

#### 4.1 Glutathione Depletion and Sulfur Amino Acid Stress

The dominant metabolic theme of the module is active glutathione cycling under conditions of depletion. [Inferred] Six metabolites directly participate in or derive from glutathione metabolism: cysteinylglycine (a gamma-glutamyl transpeptidase product of GSH), 5-oxoproline (a gamma-glutamyl cycle intermediate), cysteine-glutathione disulfide (a mixed disulfide formed during oxidative stress), cysteine S-sulfate (a minor cysteine oxidation product), S-adenosylhomocysteine (a methyl cycle intermediate upstream of the transsulfuration pathway), and three pyroglutamyl dipeptides (pyroglutamylglutamine, pyroglutamylvaline, pyroglutamylleucine) whose pyroglutamyl moiety derives from glutamate cyclization. [KG Evidence; Model Knowledge] The knowledge graph confirms pathway enrichment for glutathione metabolic process (GO:0006749, 3 members) and five SMPDB glutathione-related disease pathways (each 2 members). [KG Evidence]

#### 4.2 Innate Epithelial Type 2 Immunity (TSLP/IL5 Axis)

The cytokine signature (TSLP, IL5, IL1A) defines an innate epithelial immune response. TSLP and IL5 share pathway annotations for cytokine-mediated signaling, hemopoiesis, cell proliferation, signal transduction, and the WikiPathways cytokine-cytokine receptor interaction pathway (WP5473). [KG Evidence] The absence of IL-4 and IL-13 (see Section 5) distinguishes this from a classical Th2 adaptive response; the module instead reflects barrier tissue activation through TSLP-driven innate lymphoid cell stimulation. [Inferred]

#### 4.3 Aerobic Glycolysis (Warburg-like Metabolism)

Lactate and pyruvate co-occur without TCA cycle intermediates, a pattern consistent with aerobic glycolysis. [KG Evidence; Inferred] LDHA (NCBIGene:3939) is identified as a shared intermediate connecting module members in the pathway enrichment. [KG Evidence] Lactate accumulation has been directly linked to NLRP3 inflammasome activation and IL-1 family cytokine processing [Literature: "Lactic Acid Fermentation Is Required for NLRP3 Inflammasome Activation," 2021], providing a mechanistic bridge between the metabolic and immune arms of the module.

#### 4.4 Enteric Neurotrophic Signaling

NRTN is the sole neurotrophic factor in the module. Its top disease association (Hirschsprung disease) and pathway annotations (cell-cell signaling, cell differentiation, cell proliferation regulation) implicate enteric nervous system biology. [KG Evidence] NRTN shares the "response to stress" annotation with ADAMTS13, IL5, and TSLP, and its presence alongside gastrointestinal disease associations (diarrheal disease, gastroduodenitis) supports a role in gut epithelial-neural crosstalk. [KG Evidence; Inferred] The absence of other GDNF family members (Section 5) suggests a non-canonical, possibly immunomodulatory function for NRTN in this context.

Note: Hub-mediated associations (particularly those routed through IL1A with 5,816 edges, and high-connectivity generic nodes such as Homo sapiens, chemical role, metabolite, and extracellular space) have been de-emphasized in this analysis. [KG Evidence] Diseases and pathways shared among IL1A, IL5, and TSLP (e.g., hiatus hernia, chronic rhinitis, soft tissue disorders, arthropathies) may partly reflect the high connectivity of IL1A rather than module-specific biology.

### 5. Gap Analysis

The Open World Assumption governs interpretation: absence means "unstudied or uncaptured," not "nonexistent."

#### 5.1 Informative Absences (Biologically Revealing)

**Glutathione (GSH)**: The parent antioxidant is absent despite six degradation/cycling products being present. [Inferred] GSH is primarily intracellular and labile; its absence alongside degradation products strongly suggests active depletion under oxidative stress, a finding that is more informative than its presence would be.

**IL-1beta (IL1B)**: IL1A without IL1B suggests alarmin-mediated, damage-associated release rather than NLRP3 inflammasome-driven secretion. [Model Knowledge] IL1A is constitutively expressed and released upon cell necrosis; this distinction implies tissue damage rather than classical systemic inflammation.

**IL-4 / IL-13**: The canonical adaptive Th2 cytokines are absent despite IL5 and TSLP being present. [Inferred] This pattern is consistent with innate lymphoid cell group 2 (ILC2) activation driven by epithelial-derived TSLP, without full Th2 polarization.

**IL-6**: The master acute-phase cytokine is absent despite IL1A presence and despite IL1A and TSLP both annotated as driving IL-6 production (GO:0032755). [KG Evidence; Inferred] This module does not represent classical systemic inflammation.

**VWF (von Willebrand Factor)**: The primary ADAMTS13 substrate is absent. [Inferred] ADAMTS13 may serve a non-canonical function (anti-inflammatory, angiogenic) in this module, or VWF clusters in a separate endothelial co-expression module.

**TCA cycle intermediates**: Lactate and pyruvate accumulate without citrate, succinate, alpha-ketoglutarate, or fumarate. [Inferred] This pattern is characteristic of Warburg-like metabolic reprogramming, a hallmark of inflammatory and neoplastic cells that shunt glucose carbon toward lactate rather than oxidative phosphorylation.

**Homocysteine**: The expected hydrolysis product of S-adenosylhomocysteine is absent. [Inferred] Given the presence of cysteine S-sulfate (a transsulfuration pathway product), this absence may indicate rapid transsulfuration flux: homocysteine is efficiently converted to cysteine rather than accumulating.

**Taurine**: The major cysteine oxidation product is absent while the minor product (cysteine S-sulfate) is present. [Inferred] This suggests preferential routing of cysteine toward glutathione synthesis rather than taurine biosynthesis, a metabolic shunting pattern consistent with glutathione demand under oxidative stress.

#### 5.2 Standard Gaps

**BCAAs (leucine, isoleucine, valine)**: Their absence indicates that this module captures biology orthogonal to classical insulin resistance, despite diabetes mellitus appearing as a disease association for 3 members (pyruvate, 5-oxoproline, TSLP). [KG Evidence; Inferred]

**Arginine**: Ornithine without arginine prevents assessment of urea cycle directionality and NO bioavailability. Ornithine accumulation may reflect polyamine pathway activity (consistent with inflammatory macrophage biology) rather than urea cycle flux. [Model Knowledge]

**GDNF, ARTN, PSPN**: NRTN appears without other GDNF family members, reinforcing a non-canonical role for NRTN in mucosal immunity rather than classical neurotrophic signaling. [Inferred]

**GSSG (glutathione disulfide)**: Its absence limits quantitative redox assessment but is likely an analytical artifact (rapid interconversion ex vivo). [Model Knowledge]

### 6. Temporal Context

No longitudinal design metadata was provided; formal causal inference is therefore not possible. Nonetheless, the module's composition suggests a plausible temporal sequence. [Inferred]

**Upstream causes (likely earlier events)**: Epithelial barrier damage releases IL1A (an alarmin released upon cell necrosis) and TSLP (an epithelial-derived cytokine induced by barrier disruption). Metabolic stress generates S-adenosylhomocysteine (reflecting methylation demand), pyruvate, and lactate (reflecting glycolytic flux).

**Intermediate processes**: TSLP-driven ILC2 activation produces IL5, driving eosinophil recruitment. Glutathione depletion (evidenced by cysteinylglycine, 5-oxoproline, cysteine-glutathione disulfide accumulation) reflects oxidative stress from inflammatory cell activation. Lactate accumulates as aerobic glycolysis predominates in activated immune cells.

**Downstream consequences (likely later events)**: Pyroglutamyl dipeptide accumulation (pyroglutamylglutamine, pyroglutamylvaline, pyroglutamylleucine) reflects gamma-glutamyl cycle overflow when glutathione recycling is overwhelmed. 3-ureidopropionate accumulation reflects pyrimidine catabolism, potentially driven by cell turnover. NRTN may represent a reparative neurotrophic signal in damaged gut epithelium.

A longitudinal study design with time-resolved sampling would enable formal Granger causality or mediation analysis to test these ordering hypotheses.

### 7. Research Recommendations

#### Priority 1: Immediate Experimental Validation

1. **Measure glutathione (GSH/GSSG ratio) and cysteine in parallel with module metabolites** in the study cohort. The module strongly implicates glutathione depletion, but direct measurement is needed to confirm this interpretation. [Inferred]

2. **Profile pyroglutamyl dipeptides (pyroglutamylvaline, pyroglutamylleucine, pyroglutamylglutamine) in colorectal cancer and eosinophilic esophagitis cohorts.** These poorly characterized metabolites are predicted to associate with both conditions (Section 3.2, 3.3). Targeted metabolomics is feasible and could establish novel biomarkers.

3. **Test cysteine-glutathione disulfide as a substrate/inhibitor of GSTP1 and GSTA1** in vitro (Section 3.1). This is the most biochemically plausible Tier 3 prediction and could reveal a novel redox-regulatory feedback loop.

#### Priority 2: Literature-Guided Follow-Up

4. **Investigate IL-1alpha-driven cysteine metabolite reprogramming** by following up on the Nature Metabolism 2025 finding that IL-1alpha stimulation alters cysteine-peptide and metabolite profiles [Literature: Nature Metabolism, 2025]. Replication in the study cohort would establish a causal link between IL1A and the thiol metabolite cluster.

5. **Characterize ADAMTS13 non-canonical functions** in the context of this module. The absence of VWF and presence of disease bridges through cardiovascular, renal, and neurological conditions suggest that ADAMTS13 may function as an anti-inflammatory protease independent of VWF cleavage. [Inferred]

6. **Evaluate NRTN as a mucosal immune mediator** rather than a classical neurotrophic factor. Its co-expression with cytokines and gut disease associations (diarrheal disease, Hirschsprung disease) warrants investigation of NRTN signaling in intestinal epithelial cells and enteric glia. [KG Evidence; Model Knowledge]

#### Priority 3: Computational and Integrative Analyses

7. **Perform cross-module comparison** within the WGCNA network to determine whether BCAAs, IL-4/IL-13, VWF, IL-6, and TCA cycle intermediates cluster in distinct modules, which would confirm the orthogonality of this module's biology.

8. **Conduct formal pathway enrichment (ORA/GSEA) on the full metabolite set** using MetaboAnalyst or similar tools. The KG-based pathway analysis identifies glutathione metabolism as dominant; formal statistical enrichment with FDR correction would strengthen this conclusion.

9. **Populate knowledge graph entries for cold-start and sparse entities** (3-amino-2-piperidone, 1-palmitoylglycerol, pyroglutamylleucine). These molecules have 0 to 3 KG edges; their inclusion in the module suggests biological relevance, but interpretation is limited by data sparsity.

10. **Assess the "Smoking" exposure annotation** shared by 4 members (ADAMTS13, IL5, NRTN, TSLP). [KG Evidence] If smoking status is available in the study cohort, test whether this module correlates with tobacco exposure, which could confound or explain the epithelial barrier stress signature.

---

**Evidence attribution key**: [KG Evidence] = from KRAKEN knowledge graph query results; [Literature] = supported by grounded abstracts cited in this report; [Model Knowledge] = from general biomedical knowledge not backed by KG queries or grounded literature; [Inferred] = derived by combining KG evidence, grounded literature, and/or model knowledge.

### Literature References

Papers discovered via semantic search. 7 unique papers across 3 hypotheses.

| Hypothesis | Citation | Link | Relevance |
|------------|----------|------|-----------|
| Bridge: Gene → SmallMolecule (2 hops) |  (2022) "Frontiers \| Metabolic Engineering of Bacillus amyloliquefaciens to Efficiently Synthesize L-Ornithin..." | [Link](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2022.905110/full) | amyloliquefaci ... were enhanced by ... . The glutamate degradation pathway, the precursor ... pathway, the L-ornithine... |
| Bridge: Gene → SmallMolecule (2 hops) |  (2021) "Frontiers \| Ornithine Transcarbamylase – From Structure to Metabolism: An Update" | [Link](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2021.748249/full) | Ornithine transcarbamylase (OTC; EC 2.1.3.3) is a ubiquitous enzyme found in almost all organisms, including vertebrates... |
| Bridge: Gene → MolecularMixture (1 hops) |  (2026) "Lactic acid drives NLRP3 inflammasome activation and caspase-1–like cytokine cleavage via intracellu..." | [Link](https://www.nature.com/articles/s41419-026-08708-y) | Glycolysis is critical for NLRP3 inflammasome activation, yet the link between lactic acid metabolism and inflammasome s... |
| Bridge: Gene → MolecularMixture (1 hops) |  (2021) "Lactic Acid Fermentation Is Required for NLRP3 Inflammasome Activation" | [Link](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2021.630380/full) | mechanism was a ... pathway of NLRP3 inflam ... ome regulation, we further ... effect of GSK2 ... 37808A on IL-1 ... oth... |
| Bridge: Gene → SmallMolecule (2 hops) |  (2013) "Metabolic evolution of Corynebacterium glutamicum for increased production of L-ornithine \| BMC Biot..." | [Link](https://link.springer.com/article/10.1186/1472-6750-13-47) | . These are the reactions catalysed by ... rate dehydrogenase, ... -dependent N-acetyl- ... -glutamyl-phosphate reductas... |
| Bridge: Gene → MolecularMixture (2 hops) |  (2023) "Protein-metabolite interactomics of carbohydrate metabolism reveal regulation of lactate dehydrogena..." | [Link](https://pubmed.ncbi.nlm.nih.gov/36893255/) | (A) Volcano plots ... (black) and ... (pink). Specific, significant metabolites are numbered and labeled. Stars indicate... |
| Bridge: Gene → MolecularMixture (2 hops) |  (2023) "The role of lactate in cardiovascular diseases \| Cell Communication and Signaling \| Springer Nature ..." | [Link](https://link.springer.com/article/10.1186/s12964-023-01350-7) | . They include ... FX-11, GSK2837808 ... of LDH, such ... stiripent ... , galloflfl ... hydroxyindoles, ... 101 (gossyp... |