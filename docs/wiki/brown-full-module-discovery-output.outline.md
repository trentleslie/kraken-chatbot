# Brown Module Full Run: Discovery Output (203-analyte, dev, 2026-06-22)

> **Updated 2026-06-23 (corrected re-run, commit `edf351f`).** The first full-module run (commit `215aaa9`) was degraded by a triage-concurrency bug: an unbounded edge-count fan-out overwhelmed the knowledge-graph service, and failed measurements silently defaulted well-characterized hub genes (IL6, HMOX1, VEGFA, MMP9, and roughly sixty others) to cold-start. That bug is fixed: triage now bounds its edge-count concurrency, and a measurement failure routes to the direct-KG path with a visible marker rather than silently to cold-start. This document reports the corrected run, in which those hubs are analyzed on real knowledge-graph evidence. The cold-start bucket fell from 70 to 11 and well-characterized rose from 28 to 81; the earlier report's "identifier resolution or graph coverage gaps" self-diagnosis was a symptom of the bug, not a real gap.

This document presents the discovery output of the full-module run of the Brown WGCNA module through the Kraken 12-node discovery pipeline. The run submitted all 203 named Brown analytes (50 proteins, 153 metabolites and chemistry species) against the dev environment on 2026-06-23 (commit `edf351f`, which integrates the synthesis-context, intake-robustness, and triage-reliability fixes) with BioMapper entity resolution enabled. The companion [Pipeline Performance Report](https://phwiki.phenoma.ai/doc/brown-module-full-run-pipeline-performance-report-203-analyte-dev-2026-06-22-IosiN9wigV) reports the run's cost, latency, per-node accounting, and context-compression telemetry. It supersedes the 24-analyte [Brown Module C1 Pilot: Discovery Output](https://phwiki.phenoma.ai/doc/brown-module-c1-pilot-discovery-output-24-analyte-dev-2026-06-22-awfeh9Ork4) in scale.

Submitting the full module became possible only after an intake-parsing fix: the node now preserves internal commas in chemical names (for example `12,13-DiHOME`) and admits parenthetical synonyms, lifting the harness ceiling from roughly 130 analytes to the full 203 named rows. The 217-row module contains 14 chemistry entries with no chemical name, which remain inherently un-submittable; 203 is therefore the reachable maximum.

## Run Provenance

The pipeline accepted all 203 submitted rows and reduced them to 194 distinct entities (nine collapses: four duplicate gene rows and five numbered-isomer series), then resolved all 194 (135 via BioMapper, 49 via fuzzy match, 10 exact). Resolution mapped the 194 entities to 181 distinct CURIEs; the 13 collisions are isomers and synonyms that the knowledge graph represents with a single canonical node. Triage classified 81 entities as well-characterized, 41 moderate, 61 sparse, and 11 cold-start, with zero measurement failures: the hub genes that the buggy run had cold-started (IL6, HMOX1, and the rest) are now correctly well-characterized and analyzed on direct knowledge-graph evidence. Downstream analysis produced 2,729 direct-KG findings (965 disease associations, 782 pathway memberships, 11 hub flags), 15 cold-start findings, 12 biological themes across 175 shared neighbors, 30 cross-entity bridges (20 evidence-grounded), and 54 hypotheses supported by 32 literature references. Synthesis emitted a 26,432-character report. The run completed in approximately 1,602 s (~26.7 min) of wall-clock time with zero errors. All counts are taken from the reducer-merged pipeline state.

| Stage | Result |
|---|---|
| Submitted | 203 named analytes (50 proteins, 153 metabolites/chemistry) |
| Intake | 194 distinct entities (9 legitimate collapses: 4 duplicate genes, 5 numbered isomers) |
| Entity resolution | 194/194 resolved: 135 BioMapper, 49 fuzzy, 10 exact |
| Resolved CURIEs | 181 distinct (13 isomer/synonym collisions to shared canonical CURIEs) |
| Triage | 81 well-characterized, 41 moderate, 61 sparse, 11 cold-start (0 measurement failures) |
| Direct KG | 2,729 findings (965 disease associations, 782 pathways, 11 hub flags) |
| Cold-start | 15 findings (21 analogues, 9 inferred), 64 skipped |
| Pathway enrichment | 175 shared neighbors (154 non-hub), 12 biological themes |
| Integration | 30 bridges (20 evidence-grounded), 11 gap entities |
| Literature grounding | 32 papers across 15/54 hypotheses |
| Synthesis | 54 hypotheses, 26,432-character report |
| Run total | ~1,602 s wall-clock (~26.7 min), ~$2.27 estimated, 0 errors, status complete |

Correcting the triage misclassification roughly doubled the entities routed to direct-KG analysis (122 well-characterized plus moderate, versus 66 in the buggy run), so `pathway_enrichment` is now the dominant node (823 s, 49% of summed node time) because it genuinely computes shared neighbors for the recovered hubs. The synthesis context remained bounded at 34% of its character budget, confirming that module scale is limited by per-entity analysis work rather than by the synthesis context window.

## Related

- Companion run metrics: [Brown Module Full Run: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/brown-module-full-run-pipeline-performance-report-203-analyte-dev-2026-06-22-IosiN9wigV)
- Pilot-scale precursor: [Brown Module C1 Pilot: Discovery Output](https://phwiki.phenoma.ai/doc/brown-module-c1-pilot-discovery-output-24-analyte-dev-2026-06-22-awfeh9Ork4)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---



## Discovery Report: Brown WGCNA Module, Multi-Omics Co-Expression Analysis

### 1. Executive Summary

This Brown WGCNA module encodes a coordinated inflammatory, metabolic, and angiogenic programme linking 46 protein-coding genes and 148 metabolites through convergent cytokine signalling, lipid remodelling, and glucocorticoid stress axes. [KG Evidence] The module's dominant biological signature is a pro-inflammatory adipose/immune network centred on IL6, MMP9, VEGFA, and LEP, with recurrent disease associations to asthma (17 members), depressive disorder (16 members), coronary artery disease (12 members), and hypertensive disorder (12 members), all supported by curated knowledge graph evidence. [KG Evidence] The metabolite complement reveals active diacylglycerol turnover, gamma-glutamyl dipeptide accumulation, glucocorticoid metabolism (cortisol, cortisone, corticosterone), and piperine xenobiotic processing, collectively indicating a systemic state of chronic inflammation intersecting hepatic and adipose metabolic dysfunction. [Inferred]

### 2. Key Findings (Tier 1 to 2)

#### 2.1 Disease Associations with Strong Evidence

The module-level disease recurrence analysis identified multiple conditions with curated (GWAS or database-curated) associations shared across numerous module members:

| Disease | Members | Evidence Level | Key Contributing Proteins |
|---|---|---|---|
| Asthma | 17 | Curated | TNFRSF10A, MMP9, IL18, CD163, FABP4, SELE, HGF |
| Depressive disorder | 16 | Curated | CTSD, CTSZ, FABP4, IGFBP1, IGFBP2, MMP9, PON3 |
| Coronary artery disease | 12 | Curated | IL18, OLR1, SELE, CD163, SCGB3A2, IGFBP1, HGF |
| Hypertensive disorder | 12 | Curated | MMP9, OLR1, SELE, ACP5, IGFBP2, FABP4, CD163 |
| Obesity disorder | 7 | Curated | FABP4, IGFBP2, LEP, VEGFD, CD163, SELE |
| Cancer (broad) | 8 | Curated | IDUA, IL6, IL18, COL1A1, ACP5, PON3, TNFRSF10A |

[KG Evidence] All disease associations above derive from module-level disease recurrence data with curated provenance.

Notably, the convergence of asthma and depressive disorder as the two most broadly shared conditions suggests this module captures a chronic low-grade inflammatory state with both respiratory and neuropsychiatric sequelae. [Inferred] The co-occurrence of coronary artery disease, hypertensive disorder, and obesity is consistent with metabolic syndrome pathophysiology, a conclusion reinforced by the simultaneous presence of LEP, FGF21, IGFBP1, IGFBP2, and OLR1 in this module. [Model Knowledge]

#### 2.2 Validated Pathway Memberships

The pathway enrichment analysis demonstrates that this module is organized around several core biological processes:

**Inflammatory response** (GO:0006954): OLR1, IL18, AGER, CCL16, CCL20 participate directly; 6 module members annotated in total. [KG Evidence]

**Cell-cell signalling** (GO:0007267): The chemokine cluster (CCL3, CCL4, CCL7, CCL16, CCL20) shares this annotation, consistent with their role as paracrine immune mediators. [KG Evidence]

**Growth factor activity** (GO:0008083): VEGFA, VEGFD (FIGF), HGF, and GH1 share this molecular function, linking the module's angiogenic and somatotropic arms. [KG Evidence]

**Protein binding** (GO:0005515): 18 module members annotated, representing a broad protein interaction network encompassing FST, COL1A1, CTSD, IL1RN, IL6, CXCL10, LEP, MMP9, PRSS8, SELE, TNFRSF10A, and TNFSF11. [KG Evidence]

**Apoptosis and cell death**: TNFRSF10A, TNFSF10, TNFSF11, CTSD, and MMP9 converge on positive regulation of apoptotic process (GO:0043065) and TNF superfamily signalling. TNFRSF10A participates specifically in TRAIL-activated apoptotic signalling, extrinsic apoptotic signalling via death domain receptors, and NF-kappaB-inducing kinase activation. [KG Evidence]

**Skeletal system development** (GO:0001501): FST, COL1A1, MMP9, and TNFSF11 (RANKL) share this annotation, and bone resorption (MONDO:0000837) recurs across IGFBP1, IGFBP2, ACP5, and CD163. [KG Evidence]

#### 2.3 Protein Interaction Partners of Note

The pathway enrichment identified several non-hub shared neighbours that bridge module members to functionally coherent external genes:

**APOE** (NCBIGene:348; 300 edges): Connects LDLR to multiple glycerophospholipid species via direct physical interaction and related_to predicates, reinforcing the module's lipid-transport axis. [KG Evidence]

**CD36** (NCBIGene:948; 150 edges): Links glycerophospholipids to fatty acid uptake, consistent with the module's FABP4 membership and oxidized lipid processing through OLR1. [KG Evidence]

**Phospholipase enzymes** (PLA2G1B, PLA2G3, LIPC, LIPA, PNPLA3, LIPF): Multiple lipase genes converge as shared neighbours of the module's diverse glycerophospholipid and diacylglycerol metabolites, indicating that this module captures a state of active membrane lipid remodelling. [KG Evidence] PLA2G1B connects 7 input entities (protein-level evidence), while PNPLA3 (30 edges, non-hub) represents a genetically validated locus for non-alcoholic fatty liver disease. [KG Evidence]

#### 2.4 Cross-Type Bridges

Two-hop knowledge graph paths connect the module's protein and metabolite compartments through biologically meaningful intermediaries:

**TNFRSF10A → TNF → 1-methylnicotinamide**: TNFRSF10A interacts with TNF, which in turn affects 1-methylnicotinamide levels. [KG Evidence] This connection is supported by literature demonstrating that 1-methylnicotinamide modulates IL-10 secretion in hepatocytes and that the TNFRSF10A locus encodes lncRNAs that regulate cell death pathways. [Literature: "1-methylnicotinamide modulates IL-10 secretion and voriconazole metabolism," 2025; "Dysregulation of a lncRNA within the TNFRSF10A locus activates cell death pathways," 2023]

**GH1 → growth factor activity → VEGFD**: GH1 (somatotropin) and VEGFD share growth factor activity (GO:0008083) and extracellular space (GO:0005615) annotations, and both participate in cell-cell signalling. [KG Evidence] Literature confirms a theoretical link between the GH/IGF-1 axis and cytokine signalling, wherein inflammation induces peripheral GH resistance through SOCS1/SOCS3 upregulation. [Literature: "A Theoretical Link Between the GH/IGF-1 Axis and Cytokine Family in Children," 2025]

### 3. Novel Predictions (Tier 3)

#### 3.1 Imidazole Propionate as a Gut-Derived Inflammatory Mediator in This Module

**Logic chain**: Imidazole propionate is a microbially derived histidine metabolite that was resolved in this module (PUBCHEM.COMPOUND:129668652) but returned only 1 KG edge and no semantic analogues in cold-start analysis. Its co-expression with IL6, IL18, CXCL10, and the glucocorticoid triad (cortisol, cortisone, corticosterone) is consistent with its established role as a gut-derived activator of p38gamma/p62/mTORC1 signalling that impairs insulin signalling. [Inferred; Model Knowledge for imidazole propionate mechanism] Histidine (a module member with 1,270 edges) is the biosynthetic precursor of imidazole propionate via bacterial histidine decarboxylase; both entities co-occurring in this module suggests an active histidine-to-imidazole propionate conversion axis. [Inferred]

**Calibration note**: Approximately 18% of computational predictions of this nature progress to clinical investigation. No direct KG evidence was found linking imidazole propionate to the module's inflammatory cytokines.

**Validation step**: Measure urinary and plasma imidazole propionate concentrations in the cohort; correlate with IL6, glucose, and HOMA-IR. Perform 16S rRNA sequencing to assess Clostridium/Alistipes abundance as imidazole propionate producers.

#### 3.2 Piperine Metabolite Cluster as a Dietary Confounder or Active Metabolic Modifier

**Logic chain**: The module contains piperine (CHEBI:28821; 697 edges) and at least 6 piperine phase-II metabolites (glucuronide and sulfate conjugates), all resolved to the same PubChem parent (PUBCHEM.COMPOUND:90335471). Piperine inhibits hepatic CYP3A4 and CYP2D6, modulates NF-kappaB signalling, and enhances bioavailability of co-ingested compounds. [Model Knowledge] The co-expression of piperine metabolites with inflammatory cytokines (IL6, CCL3, CXCL10) and glucocorticoids (cortisol) raises the possibility that dietary piperine intake is either a confounder (correlated with dietary pattern) or an active modifier of this module's inflammatory tone. [Inferred]

**Calibration note**: Approximately 18% of such computational predictions advance to clinical investigation.

**Validation step**: Administer a dietary questionnaire focused on spice intake (black pepper); stratify module eigenvalue by piperine-metabolite quartiles. Test whether adjusting for piperine metabolites attenuates the IL6/cortisol correlation.

#### 3.3 1-Methylnicotinamide as a Hepatic NAD Salvage Marker Bridging Inflammation and Metabolism

**Logic chain**: 1-Methylnicotinamide (CHEBI:16797; 143 edges) is the product of nicotinamide N-methyltransferase (NNMT), a key enzyme in the NAD salvage pathway. TNFRSF10A connects to 1-methylnicotinamide via TNF-mediated interaction (2-hop path through TNF; KG Evidence). TNFRSF10A also connects via the cancer → treats_or_studied_to_treat path, and via shared blood expression. [KG Evidence] Literature evidence confirms that 1-methylnicotinamide modulates IL-10 secretion and that bacterial NAD biosynthesis contributes substantially to host NAD metabolism. [Literature: "1-methylnicotinamide modulates IL-10 secretion and voriconazole metabolism," 2025; "Bacteria Boost Mammalian Host NAD Metabolism by Engaging the Deamidated Biosynthesis Pathway," 2020] The presence of quinolinate (CHEBI:16675; 229 edges), the de novo NAD biosynthesis intermediate, alongside 1-methylnicotinamide suggests that both the de novo (tryptophan → quinolinate → NAD) and salvage (nicotinamide → 1-methylnicotinamide) pathways are represented in this module. [Inferred]

**Calibration note**: Approximately 18% of computational predictions advance to clinical investigation.

**Validation step**: Correlate 1-methylnicotinamide and quinolinate levels with hepatic inflammation markers (IL6, FGF21). Measure NNMT expression in available liver biopsy samples if the cohort includes them.

#### 3.4 Gamma-Glutamyl Dipeptide Accumulation as a Marker of Glutathione Turnover Stress

**Logic chain**: The module contains 8 gamma-glutamyl dipeptides (gamma-glutamylglutamate, gamma-glutamyltyrosine, gamma-glutamylleucine, gamma-glutamylvaline, gamma-glutamylglycine, gamma-glutamylthreonine, gamma-glutamylisoleucine, gamma-glutamylglycine). Gamma-glutamyl dipeptides are byproducts of the gamma-glutamyl cycle, in which gamma-glutamyl transpeptidase (GGT) cleaves glutathione to transfer the gamma-glutamyl moiety to amino acid acceptors. [Model Knowledge] Their co-expression with oxidative stress-responsive genes (HMOX1: 8,037 edges, heme oxygenase; AGER: 8,890 edges, receptor for advanced glycation end products) and with cystine (a glutathione precursor) suggests chronic oxidative stress driving glutathione consumption and compensatory GGT activity. [Inferred]

**Calibration note**: Approximately 18% of such predictions progress to validation.

**Validation step**: Measure serum GGT activity and reduced/oxidized glutathione ratios. Test whether gamma-glutamyl dipeptide concentrations predict HMOX1 protein levels or AGER-associated disease outcomes (e.g., diabetic complications).

### 4. Biological Themes

#### 4.1 Unifying Theme: Chronic Inflammatory Metabolic Stress

This module captures the intersection of three major biological axes operating as a coordinated programme:

**Axis 1: Pro-inflammatory cytokine and chemokine signalling.** IL6 (9,911 edges), CXCL10, CCL3, CCL4, CCL7, CCL19, CCL20, IL18, IL18R1, IL1RN, and OSM form a dense chemokine/cytokine subnetwork. [KG Evidence] These proteins share cell-cell signalling (GO:0007267), inflammatory response (GO:0006954), and response to stress (GO:0006950) annotations. [KG Evidence] The module includes both effectors (IL6, IL18, chemokines) and modulators (IL1RN as an anti-inflammatory brake; FST as an activin antagonist), suggesting that the module captures both the inflammatory stimulus and its counter-regulatory response. [Inferred]

**Axis 2: Adipose-hepatic metabolic dysregulation.** LEP, FABP4, IGFBP1, IGFBP2, FGF21, and GH1 encode an adipose-to-liver signalling axis. [KG Evidence] GH1 participates in the Food Intake and Energy Homeostasis Pathway, the Longevity Pathway, and positive regulation of D-glucose transmembrane transport. [KG Evidence] The metabolite complement supports this interpretation: cortisol, cortisone, corticosterone (HPA axis activation); glucose, fructose, mannose (carbohydrate dysregulation); and the extensive diacylglycerol/glycerophospholipid species (lipid remodelling). [KG Evidence for entity presence; Inferred for functional interpretation]

**Axis 3: Tissue remodelling and extracellular matrix turnover.** MMP9 (9,594 edges), COL1A1, CTSD, CTSZ, ACP5, and TNFSF11 (RANKL) participate in proteolysis (GO:0006508), skeletal system development (GO:0001501), and bone resorption. [KG Evidence] The presence of 5-hydroxylysine (a collagen cross-link byproduct) and pro-hydroxy-pro (a collagen degradation dipeptide) among the metabolites reinforces active collagen catabolism. [Model Knowledge]

#### 4.2 Hub-Filtered Insights

The following entities were flagged as hubs (more than 1,000 edges) and associations routed exclusively through them should be interpreted cautiously: IL6 (9,911), MMP9 (9,594), serine (9,226), AGER (8,890), HMOX1 (8,037), CTSD (7,949), glucose (7,471), VEGFA (5,873), cortisol (5,665), COL1A1 (5,546), and LDLR (5,305). [KG Evidence]

Despite their hub status, several of these entities are biologically central to the module's theme rather than merely promiscuous nodes. IL6, for instance, is the canonical upstream regulator of CRP and hepatic acute-phase response; its hub-bias warning does not diminish its biological relevance to this inflammatory module but does caution against over-interpreting any single IL6 disease association in isolation. [Inferred]

Non-hub associations of particular interest include:
- **PNPLA3** (30 edges): Shared neighbour of multiple glycerophospholipids; a well-validated NAFLD susceptibility gene. [KG Evidence]
- **CD36** (150 edges): Connects phospholipids to fatty acid uptake. [KG Evidence]
- **PLA2G3** (30 edges): Low-connectivity phospholipase connecting 5 glycerophospholipid module members. [KG Evidence]

#### 4.3 Metabolite Substructure Themes

The 148 metabolites organize into several coherent functional groups:

| Metabolite Class | Count (approx.) | Functional Interpretation |
|---|---|---|
| Diacylglycerols and glycerophospholipids | ~40 | Membrane remodelling; lipid second messengers |
| Amino acids and derivatives | ~25 | Protein catabolism; urea cycle intermediates |
| Gamma-glutamyl dipeptides | 8 | Glutathione turnover; oxidative stress |
| Glucocorticoids (cortisol, cortisone, corticosterone, cortolone glucuronide) | 4 | HPA axis activation |
| Piperine and metabolites | ~7 | Dietary xenobiotic processing |
| Acylcarnitines (C3, C4, C5) | 5 | Incomplete fatty acid and BCAA oxidation |
| Plant sterols (beta-sitosterol, campesterol) | 2 | Dietary sterol absorption |
| Carotene diols and beta-cryptoxanthin | 4 | Dietary carotenoid intake/metabolism |

[Inferred for classification; KG Evidence for entity presence]

### 5. Gap Analysis

#### 5.1 Informative Absences

**Adiponectin (ADIPOQ)**: This anti-inflammatory adipokine is expected alongside LEP in any metabolic dysfunction module. Its absence, coupled with the presence of LEP, FABP4, and IL6, indicates that the Brown module selectively captures the pro-inflammatory adipose axis. [Inferred] ADIPOQ likely resides in an inversely correlated module, and cross-module correlation analysis between Brown and the ADIPOQ-containing module could reveal the inflammatory switch point.

**Insulin (INS) and C-peptide**: The absence of insulin is biologically plausible because insulin regulation is primarily post-translational (secretory dynamics) rather than transcriptional. [Model Knowledge] The module contains its downstream effectors (IGFBP1, IGFBP2, glucose, FGF21), suggesting that insulin resistance, rather than insulin deficiency, is the operative state.

**BCAAs (leucine, isoleucine, valine)**: Canonical early predictors of type 2 diabetes conversion. Their absence despite the presence of N-acetylleucine and multiple acylcarnitines (butyrylcarnitine C4, isovalerylcarnitine C5, 2-methylbutyrylcarnitine C5) suggests that intact BCAAs segregate into a different WGCNA module. [Inferred] The acylcarnitine derivatives may represent BCAA catabolic intermediates, indicating the downstream consequences of BCAA metabolism are captured here while upstream accumulation is not.

**CRP (C-reactive protein)**: The absence of this canonical downstream target of IL6 signalling reinforces the interpretation that this module represents the cytokine source compartment (immune/adipose) rather than the hepatic acute-phase response compartment. [Inferred]

**TNF (TNF-alpha)**: TNFRSF10A (a TRAIL receptor, not a TNF-alpha receptor), TNFSF10 (TRAIL), and TNFSF11 (RANKL) are present, but TNF itself is absent. TNF protein levels are often regulated post-transcriptionally and by TACE/ADAM17-mediated shedding; these mechanisms may decouple TNF protein from the transcriptional co-expression pattern captured by WGCNA. [Model Knowledge]

#### 5.2 Standard Gaps

**Ceramides**: These lipotoxic sphingolipids are strongly predictive of cardiovascular events and type 2 diabetes. Their absence likely reflects metabolomic platform limitations (ceramides require targeted lipidomics) rather than biological irrelevance, given that OLR1 and the extensive diacylglycerol complement indicate active lipotoxic signalling. [Model Knowledge]

**HbA1c and HOMA-IR**: These are clinical indices, not molecular entities captured by proteomic or metabolomic platforms. Their absence is expected and uninformative about underlying biology. [Model Knowledge]

### 6. Temporal Context

#### 6.1 Upstream Causes versus Downstream Consequences

Although the analysis context does not specify a longitudinal design, the module's composition permits provisional causal reasoning:

**Likely upstream (causal) members**: The glucocorticoid triad (cortisol, cortisone, corticosterone) and GH1 represent endocrine axes that drive downstream metabolic and inflammatory changes. HPA axis activation (cortisol) induces insulin resistance, hepatic gluconeogenesis, and immunomodulation. [Model Knowledge] Dietary signals (piperine metabolites, plant sterols, carotenoids) are exogenous inputs. [Inferred]

**Likely intermediary signalling members**: IL6, IL18, chemokines (CCL3, CCL4, CCL7, CCL19, CCL20, CXCL10), and LEP propagate the inflammatory signal from adipose and immune tissue. [Model Knowledge]

**Likely downstream (consequential) members**: MMP9 (tissue remodelling), SELE (endothelial activation), COL1A1 breakdown products (5-hydroxylysine, pro-hydroxy-pro), gamma-glutamyl dipeptides (oxidative stress response), and diacylglycerols (membrane turnover) represent effector or damage-response outputs. [Inferred]

#### 6.2 Causal Inference Opportunities

If longitudinal time points are available, Granger causality or vector autoregression models testing whether cortisol and IL6 levels at time t predict MMP9 and gamma-glutamyl dipeptide levels at time t+1 would clarify the temporal ordering and strengthen causal claims. [Inferred]

### 7. Research Recommendations

#### 7.1 High-Priority Experimental Validations

1. **Imidazole propionate quantification**: Measure plasma and urinary imidazole propionate in the cohort. Correlate with IL6, glucose, insulin resistance indices, and gut microbiome composition (16S or shotgun metagenomics). This metabolite's presence in the module, combined with its sparse KG coverage (1 edge), represents the highest-value validation target for a gut-microbiome-inflammation axis. [Inferred]

2. **Gamma-glutamyl dipeptide panel and oxidative stress**: Assay serum GGT activity, glutathione (reduced and oxidized), and 8-isoprostane alongside the 8 gamma-glutamyl dipeptides. Test whether the gamma-glutamyl panel outperforms GGT alone as a predictor of oxidative stress-linked outcomes (AGER-associated diabetic complications, cardiovascular events). [Inferred]

3. **1-Methylnicotinamide and quinolinate as NAD metabolism reporters**: Quantify NNMT expression (if tissue is available) or use urinary 1-methylnicotinamide as a proxy. Test whether 1-methylnicotinamide/quinolinate ratios correlate with hepatic FGF21 production and inflammatory cytokine levels. [Inferred; Literature: "1-methylnicotinamide modulates IL-10 secretion and voriconazole metabolism," 2025]

#### 7.2 Targeted Literature Searches

4. **Piperine-cytokine interaction**: Conduct a systematic review of piperine's effects on NF-kappaB, IL6, and IL18 signalling in human studies. Determine whether the module's piperine-metabolite cluster is a dietary confounder (associated with spice-rich cuisines) or an active pharmacological modifier. [Inferred]

5. **DNER (Delta/Notch-like EGF repeat containing)**: This is the least characterized protein in the module (1,452 edges; no top disease association). DNER is a single-pass transmembrane protein that activates Notch signalling in glial cells; its co-expression with inflammatory and metabolic proteins is unexpected. A literature search for DNER in metabolic or inflammatory contexts is warranted. [KG Evidence for edge count; Model Knowledge for DNER function]

6. **SARS-CoV-2 signalling pathway**: The WikiPathways annotation (WP5115: Network map of SARS-CoV-2 signalling) includes CTSD, CTSZ, and CD163 from this module. If the cohort includes samples collected during or after the COVID-19 pandemic, this pathway enrichment may reflect post-viral inflammatory changes. [KG Evidence]

#### 7.3 Follow-Up Analyses

7. **Cross-module analysis**: Compare the Brown module eigenvalue with the eigenvalues of modules containing ADIPOQ, TNF, CRP, and BCAAs. Partial correlation analysis, conditioning on BMI, can distinguish whether the pro-inflammatory signal in the Brown module is adiposity-dependent or adiposity-independent. [Inferred]

8. **Mendelian randomization**: Use GWAS instruments for module members with strong genetic associations (LDLR for lipid metabolism, OLR1 for myocardial infarction susceptibility, TNFSF11 for osteoporosis) to test causal effects of genetically predicted protein levels on module eigenvalue. [Inferred]

9. **Lipidomics extension**: The 40+ diacylglycerol and glycerophospholipid species in this module, combined with the convergence of PNPLA3, LIPC, LIPA, and PLA2G3 as shared KG neighbours, strongly motivate targeted ceramide, sphingomyelin, and lysophosphatidylcholine measurements to complete the lipid remodelling picture. [KG Evidence for shared neighbours; Inferred for recommendation]

10. **Member prioritization for functional follow-up**: Based on the Member Prioritization Table, the following entities offer the best ratio of biological centrality to experimental tractability:

| Priority | Entity | Rationale |
|---|---|---|
| 1 | FGF21 | Hepatokine linking liver stress to metabolic inflammation; druggable target (analogues in clinical trials) |
| 2 | OLR1 | Oxidized LDL receptor with curated myocardial infarction association; links lipid metabolism to inflammation |
| 3 | CD163 | Macrophage activation marker; IgA nephropathy association suggests renal-inflammatory crosstalk |
| 4 | IGFBP1 | Primary biliary cholangitis association suggests hepatic specificity; insulin-regulated |
| 5 | SCGB3A2 | Present in 11 disease recurrence entries despite being a relatively specific lung epithelial marker; its role in this systemic module is unexplained |

[KG Evidence for disease associations; Inferred for prioritization logic]

---

**Evidence Attribution Summary**: This report draws on knowledge graph query results (disease recurrence, pathway enrichment, edge counts, cross-type bridges, gap analysis) tagged as [KG Evidence]; grounded PubMed abstracts tagged as [Literature]; general biomedical knowledge tagged as [Model Knowledge]; and synthetic conclusions combining multiple sources tagged as [Inferred]. No claims derived from model knowledge are presented as KG-derived findings.

### Literature References

Papers discovered via semantic search. 5 unique papers across 2 hypotheses.

| Hypothesis | Citation | Link | Relevance |
|------------|----------|------|-----------|
| Bridge: Gene → ChemicalEntity (2 hops) |  (2025) "A Theoretical Link Between the GH/IGF-1 Axis and Cytokine Family in Children: Current Knowledge and ..." | [Link](https://www.mdpi.com/2227-9067/12/4/495) | One of the consequences of inflammation is the induction of peripheral resistance to GH, which occurs through two major... |
| Bridge: Gene → SmallMolecule (2 hops) |  (2020) "Bacteria Boost Mammalian Host NAD Metabolism by Engaging the Deamidated Biosynthesis Pathway" | [Link](https://pubmed.ncbi.nlm.nih.gov/32130883/) | Bacteria Boost Mammalian Host NAD Metabolism by Engaging the Deamidated Biosynthesis Pathway Abstract Nicotinamide... |
| Bridge: Gene → SmallMolecule (2 hops) |  (2023) "Dysregulation of a lncRNA within the TNFRSF10A locus activates cell death pathways \| Cell Death Disc..." | [Link](https://www.nature.com/articles/s41420-023-01544-5) | The TNFRSF10A genomic locus contains three genes: the protein-coding tumor necrosis factor receptor superfamily member 1... |
| Bridge: Gene → SmallMolecule (2 hops) |  (2025) "Frontiers \| 1-methylnicotinamide modulates IL-10 secretion and voriconazole metabolism" | [Link](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2025.1529660/full) | 1-M ... by the enzymatic ... of nicotinamide N-methyltransferase and is primarily distributed in the liver ( ... 3). Pre... |
| Bridge: Gene → ChemicalEntity (2 hops) |  (2020) "Frontiers \| Tumor Necrosis Factor Receptor SF10A (TNFRSF10A) SNPs Correlate With Corticosteroid Resp..." | [Link](https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2020.00605/full) | 2 genes: VCAN (rs44 ... 0745 and rs12 ... 2199) for ... A (rs20 ... 5 and rs17 ... 20) for missense. We technically ...... |