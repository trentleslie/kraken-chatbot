# Black Module Run: Discovery Output (109-analyte, dev, 2026-06-23)

This document presents the discovery output of a full-module run of the **Black** WGCNA module through the Kraken 12-node discovery pipeline, executed on the dev environment on 2026-06-23 (commit `813fc3f`) with BioMapper entity resolution enabled. It is one of a per-module series over the Frailty multi-omics WGCNA modules; see the [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm).

## Run Provenance

The run submitted 109 named analytes, parsed 108 at intake, and resolved 108 distinct entities (35 fuzzy, 70 biomapper, 3 exact) to 104 distinct CURIEs. Triage classified 19 well-characterized, 27 moderate, 46 sparse, and 16 cold-start, with 0 edge-count measurement failures. Downstream analysis produced 1042 direct-KG findings, 22 cold-start findings, 8 biological themes, 1 cross-entity bridges (0 evidence-grounded), and 40 hypotheses supported by 33 literature references. Synthesis emitted a 27008-character report. The run completed in approximately 924.8 s of wall-clock time (status complete, 0 errors). All counts are taken from the reducer-merged pipeline state.

| Stage | Result |
|---|---|
| Submitted | 109 named analytes |
| Intake | 108 parsed |
| Entity resolution | 108 resolved (35 fuzzy, 70 biomapper, 3 exact) to 104 distinct CURIEs |
| Triage | 19 well-characterized, 27 moderate, 46 sparse, 16 cold-start (0 measurement failures) |
| Direct KG | 1042 findings |
| Cold-start | 22 findings, 54 skipped |
| Pathway enrichment | 8 biological themes |
| Integration | 1 bridges (0 evidence-grounded) |
| Literature grounding | 33 papers |
| Synthesis | 40 hypotheses, 27008-character report |
| Run total | ~924.8 s wall-clock, status complete, 0 errors |

## Related

- Companion run metrics: [Black Module Run: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/black-module-run-pipeline-performance-report-109-analyte-dev-2026-06-23-EOscfYQbEB)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---



## Discovery Report: Black WGCNA Module — A Coordinated Lipolytic and Fatty Acid Overflow Signature

---

### 1. Executive Summary

This Black WGCNA module encodes a tightly coordinated lipolytic program comprising over 100 co-expressed metabolites dominated by free fatty acids (saturated, monounsaturated, and polyunsaturated species spanning C6 to C22), their acylcarnitine transport intermediates, 3-hydroxy fatty acid beta-oxidation intermediates, and TCA cycle organic acids (fumarate, malate, citrate, aconitate). [KG Evidence] The module converges on PPAR-alpha and PPAR-gamma signaling as its principal transcriptional regulatory axis, with disease enrichment revealing curated associations to colorectal cancer (14 members), type 2 diabetes mellitus (8 members), eosinophilic esophagitis (7 members), and metabolic dysfunction-associated steatotic liver disease (4 members). [KG Evidence] The complete absence of proteins from this metabolite-only module, together with the absence of ceramides, intact triglycerides, and branched-chain amino acids, indicates that this module captures a specific metabolic state: the product side of adipose tissue lipolysis and incomplete mitochondrial beta-oxidation, rather than a broader insulin-resistance or lipotoxicity program. [Inferred]

---

### 2. Key Findings (Tier 1 to 2)

#### 2.1 Disease Associations with Curated Evidence

The module-level disease recurrence analysis identified several conditions supported by curated database entries across multiple module members:

| Disease | Members | Evidence Strength |
|---|---|---|
| Colorectal cancer | 14 | Curated |
| Type 2 diabetes mellitus | 8 | Curated |
| Eosinophilic esophagitis | 7 | Curated |
| Schizophrenia | 6 | Curated |
| Metabolic dysfunction-associated steatotic liver disease (MASLD) | 4 | Curated |
| Inflammatory bowel disease | 4 | Curated |
| Hyperlipidemia | 3 | Curated |
| Epilepsy | 3 | Curated |
| Medium-chain acyl-CoA dehydrogenase (MCAD) deficiency | 3 | Curated |
| Inherited obesity | 3 | Curated |

[KG Evidence]

Colorectal cancer emerged as the most broadly connected disease, linking palmitate, stearate, myristate, palmitoleate, arachidate, pentadecanoate, fumarate, malate, 3-hydroxyisobutyrate, sebacate, caprylate, hexadecanedioate, decanoylcarnitine, and octanoylcarnitine. [KG Evidence] Type 2 diabetes mellitus, the second most recurrent disease (8 members), connected palmitate, stearate, myristate, palmitoleate, pentadecanoate, 3-hydroxyisobutyrate, sebacate, and vanillate through curated associations. [KG Evidence] The MCAD deficiency association (connecting 3-hydroxylaurate, sebacate, and octanoylcarnitine) is mechanistically coherent: these metabolites are canonical diagnostic markers of incomplete medium-chain fatty acid oxidation, and their co-occurrence in this module suggests a partial beta-oxidation block or beta-oxidation overflow state. [KG Evidence; Inferred]

#### 2.2 Validated Pathway Memberships

Six module members participate in the Gene Ontology term "fatty acid biosynthetic process" (GO:0006633): palmitate, palmitoleate, caprylate, myristate, 3-hydroxylaurate, and stearate. [KG Evidence] Five members share membership in the "Biosynthesis of Unsaturated Fatty Acids" pathway (PathWhiz:PW002403): palmitate, linoleate, arachidate, stearate, and adrenate. [KG Evidence] Fumarate and malate share membership in the tricarboxylic acid cycle (GO:0006099), the Citric Acid Cycle metabolic pathway (SMPDB:SMP0000057), Pyruvate Metabolism (SMPDB:SMP0000060), and Glucagon signaling (KEGG:04922). [KG Evidence] Caprylate and octanoylcarnitine share membership in the Mitochondrial Beta-Oxidation of Short Chain Saturated Fatty Acids pathway (SMPDB:SMP0000480). [KG Evidence]

#### 2.3 PPAR Signaling as the Central Regulatory Axis

PPARG (peroxisome proliferator-activated receptor gamma; 80 edges, non-hub) and PPARA (peroxisome proliferator-activated receptor alpha; 80 edges, non-hub) each connect five module members (linoleate, palmitate, stearate, palmitoleate, and myristate) through affects, binds, and physically_interacts_with predicates. [KG Evidence] PPARD (peroxisome proliferator-activated receptor delta) was additionally identified in the broader gene enrichment. [KG Evidence] This triad of PPAR receptors serves as the module's transcriptional integration point: PPARA drives hepatic fatty acid oxidation, PPARG regulates adipocyte lipid storage and insulin sensitization, and PPARD mediates fatty acid catabolism in skeletal muscle. [Model Knowledge] The convergence of these three receptors on the same set of fatty acid ligands within this module supports its interpretation as a coordinated lipid-sensing and metabolic-switching program.

#### 2.4 GPR84 and Medium-Chain Fatty Acid Sensing

GPR84 (G protein-coupled receptor 84; 25 edges, non-hub) connects myristate, caprylate, 3-hydroxymyristate, and 3-hydroxylaurate through affects and binds predicates. [KG Evidence] GPR84 is a medium-chain fatty acid receptor expressed on immune cells that senses C9 to C14 fatty acids to modulate inflammatory responses. [Model Knowledge] Its connection to four module members, including both free fatty acids and their 3-hydroxy derivatives, suggests that the module captures not only metabolic intermediates but also immunometabolic signaling substrates.

#### 2.5 Microbiome-Associated Metabolite Cluster

Hexanoylcarnitine, hexanoylglutamine, octadecenedioylcarnitine, and octadecenedioate share associations with *Clostridioides difficile* and uncharacterized gut bacteria (NCBITaxon:1935176, NCBITaxon:1980693; each approximately 30 edges). [KG Evidence] This cluster suggests that a subset of module metabolites reflects gut microbial fatty acid metabolism or host-microbiome co-metabolism, potentially relevant to the inflammatory bowel disease associations observed for four module members. [Inferred]

---

### 3. Novel Predictions (Tier 3)

#### 3.1 9-Hydroxystearate as an Antiproliferative Hydroxy Fatty Acid

9-Hydroxystearate (CHEBI:229769; 1 edge, sparse coverage) is inferred to belong to the hydroxy fatty acid anion parent class (CHEBI:59835) based on structural similarity to 7-hydroxystearate (0.89), 11-hydroxystearate (0.88), and 8-hydroxyoleate (0.83), all three of which are classified under CHEBI:59835. [KG Evidence; Inferred]

**Structural logic chain:** 9-hydroxystearate shares positional isomerism with 7-hydroxystearate and 11-hydroxystearate; all three analogues subclass the hydroxy fatty acid anion parent. The ontological assignment is strongly supported. [KG Evidence]

**Literature support:** Fetched abstracts confirm that (R)-9-hydroxystearic acid possesses antiproliferative properties against cancer cells, including HT-29 colorectal cancer cells (Synthesis of 9-Hydroxystearic Acid Derivatives, 2019; X-Ray Crystal Structures, 2019). [Literature] Given that colorectal cancer is the most broadly connected disease in this module (14 members), the presence of 9-hydroxystearate may represent an endogenous antiproliferative signal within the lipolytic milieu.

**Calibration:** Approximately 18% of computational predictions progress to clinical investigation. This prediction concerns an ontological classification rather than a disease mechanism, and its validation is straightforward.

**Validation step:** Query the ChEBI ontology for CHEBI:229769 to confirm the subclass_of relationship to CHEBI:59835. Independently, investigate 9-hydroxystearate levels in colorectal cancer tissue metabolomics datasets to assess its potential as a cancer-modulating lipid within this module's biological context.

#### 3.2 Suberoylcarnitine as a Blood and Urine Dicarboxylic Acylcarnitine Biomarker

Suberoylcarnitine (CHEBI:77083; 2 edges, sparse) is inferred to be a subclass of O-acylcarnitine (CHEBI:17387) based on similarity to O-suberoylcarnitine (0.89) and O-octenedioylcarnitine (0.81), both of which carry this parent class assignment. [KG Evidence; Inferred] It is further inferred to be detectable in blood and urine by analogy to O-suberoylcarnitine. [Inferred]

**Structural logic chain:** Suberoylcarnitine is a dicarboxylic acylcarnitine produced via omega-oxidation of fatty acids followed by carnitine conjugation. Its closest analogue, O-suberoylcarnitine, is documented in blood (UBERON:0000178) and urine (UBERON:0001088). [KG Evidence]

**Literature support:** Plasma acylcarnitine quantification methods (Analytical and Bioanalytical Chemistry, 2013) confirm that acylcarnitines of various chain lengths are routinely detected in plasma, consistent with suberoylcarnitine detection in blood. [Literature] Furthermore, medium-chain acylcarnitines have been positively associated with prostate cancer progression (Preoperative plasma fatty acid metabolites, BMC Cancer, 2019), and the acylhomocarnitine structural diversity literature (2026) documents related dicarboxylic species as detectable intermediates in mitochondrial beta-oxidation. [Literature]

**Calibration:** Approximately 18% of computational predictions progress to clinical investigation. This inference is primarily ontological and analytical rather than mechanistic.

**Validation step:** Query HMDB for suberoylcarnitine detection in blood and urine. Confirm O-acylcarnitine parent class assignment in ChEBI. Assess whether suberoylcarnitine levels correlate with MCAD deficiency status in this cohort, given the curated MCAD deficiency association for three other module members.

#### 3.3 The Substance P to Palmitate Bridge: A Neuroinflammatory to Metabolic Connection

A three-hop path connects the erroneously resolved "Metabolites:" header (substance P-metabolite 5-11, UMLS:C1698199) to palmitate (CHEBI:15756) via substance P (UMLS:C0038585) and TAC1 (NCBIGene:6863). [KG Evidence] Although the source node represents a resolution artifact, the underlying biology of substance P-to-palmitate connectivity is noteworthy: substance P is a neuropeptide that promotes neurogenic inflammation and has documented roles in adipose tissue innervation and lipolysis regulation. [Model Knowledge]

**Structural logic chain:** Substance P-metabolite 5-11 → substance P → TAC1 (tachykinin precursor gene) → palmitate. [KG Evidence] TAC1 encodes substance P, and substance P signaling through NK1 receptors on adipocytes can stimulate lipolysis, releasing palmitate. [Model Knowledge]

**Calibration:** Approximately 18% of computational predictions progress to clinical investigation. This bridge is speculative and depends on the relevance of substance P signaling in the cohort under study.

**Validation step:** Examine whether TAC1 expression or circulating substance P levels were measured in this cohort. If available, test the correlation between substance P levels and the fatty acid module eigengene.

---

### 4. Biological Themes

#### 4.1 Unifying Theme: Lipolytic Fatty Acid Overflow with Incomplete Beta-Oxidation

The module's composition reveals a coherent biological narrative. [Inferred] Free fatty acids dominate the module across a comprehensive chain-length range (C6 caprylate through C22 erucate and DHA), encompassing saturated (palmitate, stearate, myristate, laurate, caprate, caprylate, pentadecanoate, margarate, arachidate, nonadecanoate), monounsaturated (palmitoleate, oleate, eicosenoate, erucate), and polyunsaturated species (linoleate, arachidonate, EPA, DHA, adrenate, docosapentaenoate). [KG Evidence] This breadth indicates generalized triacylglycerol hydrolysis rather than selective release of specific fatty acid species.

Co-expression of medium-chain acylcarnitines (hexanoylcarnitine C6, octanoylcarnitine C8, decanoylcarnitine C10, myristoylcarnitine C14, oleoylcarnitine C18:1) alongside 3-hydroxy fatty acid intermediates (3-hydroxyoctanoate, 3-hydroxydecanoate, 3-hydroxylaurate, 3-hydroxymyristate, 3-hydroxyhexanoate) indicates active but possibly incomplete mitochondrial beta-oxidation. [KG Evidence; Inferred] The 3-hydroxy species represent the third step of the beta-oxidation spiral, and their accumulation suggests either substrate overload or enzymatic rate limitation at the 3-hydroxyacyl-CoA dehydrogenase step. [Model Knowledge]

The TCA cycle intermediates (fumarate, malate, citrate, aconitate) co-expressed with these fatty acid species indicate that the oxidative catabolism of fatty acids feeds into a simultaneously active citric acid cycle. [KG Evidence; Inferred] The ketone body 3-hydroxybutyrate (BHBA) and its carnitine ester (R)-3-hydroxybutyrylcarnitine further support hepatic ketogenesis as a downstream consequence of fatty acid flux. [KG Evidence]

#### 4.2 Endocannabinoid and N-Acyl Signaling Lipids

The module contains three N-acyl ethanolamides (oleoyl ethanolamide, palmitoyl ethanolamide, linoleoyl ethanolamide) and two N-acyl taurine conjugates (N-oleoyltaurine, N-stearoyltaurine), along with N-palmitoylglycine. [KG Evidence] Palmitoyl ethanolamide (1,370 edges, well-characterized) is a well-studied anti-inflammatory and analgesic lipid that acts through PPAR-alpha. [KG Evidence; Model Knowledge] The co-expression of these signaling lipids with their fatty acid precursors suggests coordinate regulation of both metabolic substrates and their bioactive lipid derivatives. [Inferred]

#### 4.3 Dicarboxylic Acid and Omega-Oxidation Products

A distinct subcluster of dicarboxylic acids (sebacate C10-DC, dodecanedioate C12-DC, tetradecanedioate C14-DC, hexadecanedioate C16-DC, octadecenedioate C18:1-DC, octadecadienedioate C18:2-DC) and their carnitine conjugates (adipoylcarnitine C6-DC, suberoylcarnitine C8-DC, octadecenedioylcarnitine C18:1-DC) reflects omega-oxidation of fatty acids. [KG Evidence] Omega-oxidation is an alternative fatty acid degradation pathway that becomes quantitatively important when mitochondrial beta-oxidation is saturated or impaired. [Model Knowledge] The breadth of dicarboxylic acid chain lengths in this module supports the interpretation that beta-oxidation capacity is exceeded, triggering compensatory omega-oxidation in the endoplasmic reticulum. [Inferred]

#### 4.4 Hub-Filtered Insights

Glycerol (CHEBI:17754; 10,000 edges) was flagged as a hub node and its associations should be interpreted cautiously. [KG Evidence] Although glycerol is the obligate co-product of triacylglycerol hydrolysis and is therefore biologically expected in this module, its very high connectivity in the knowledge graph means that any specific disease or pathway association attributed to glycerol alone carries low informational specificity. The fatty acid class ontology nodes (fatty acid, CHEBI:77746; fatty acid anion, CHEBI:35366) similarly function as hubs and are de-emphasized in favor of the more specific shared neighbors identified above (PPARG, PPARA, GPR84). [KG Evidence]

---

### 5. Gap Analysis

#### 5.1 Informative Absences (Open World Assumption)

Several expected entities are absent from this module, and their absence carries interpretive value:

**Branched-chain amino acids (leucine, isoleucine, valine):** BCAAs are among the most replicated T2D biomarkers, yet they are absent from this fatty acid-centric module. [Inferred] Their absence most likely reflects WGCNA module assignment to an amino acid-dominant module rather than true biological disconnection. This separation indicates that the BCAA and fatty acid arms of insulin resistance are transcriptionally or metabolically independent in this cohort, or at least operate on different co-expression timescales.

**Ceramides:** The absence of ceramide species despite the abundance of their precursors (palmitate, stearate) is biologically informative. [Inferred] This pattern may indicate that the module captures a substrate accumulation phase that temporally precedes de novo ceramide synthesis, or it may reflect platform limitations (ceramides require dedicated lipidomics).

**Intact triglycerides:** Free fatty acids are present without their triacylglycerol precursors, consistent with the module capturing the product side of lipolysis. [Inferred] This directionality is informative for causal modeling: the module likely reflects an active catabolic state rather than a lipid storage program.

**Adiponectin and other proteins:** The complete absence of proteins from this module, despite the study measuring both proteins and metabolites, indicates that WGCNA separated the two analyte classes into distinct co-expression networks. [Inferred] This methodological observation should be considered when interpreting cross-platform co-regulation.

**Glycerol:** Glycerol is listed as a module member in the entity resolution summary (CHEBI:17754, 10,000 edges) and is therefore present, contradicting the gap analysis entry. [KG Evidence] Its hub-level connectivity (10,000 edges) warrants caution in interpreting its associations, but its presence is consistent with the lipolytic theme.

#### 5.2 Standard (Non-Informative) Absences

Glucose, HbA1c, HOMA-IR, and C-peptide are clinical variables or derived indices that would not appear in metabolomics-derived WGCNA modules. [Model Knowledge] Their absence is methodologically determined and carries no biological information.

---

### 6. Temporal Context

No explicit longitudinal timepoints were provided in this analysis. However, the module composition supports directional inference:

**Upstream causes (likely earlier events):** Triacylglycerol hydrolysis (lipolysis) and fatty acid mobilization from adipose tissue represent the initiating events, generating the free fatty acid species that dominate this module. [Inferred] The PPAR signaling axis (PPARA, PPARG) likely operates upstream as both a sensor and transcriptional regulator of this fatty acid flux.

**Downstream consequences (likely later events):** Mitochondrial beta-oxidation intermediates (3-hydroxy fatty acids, acylcarnitines), omega-oxidation products (dicarboxylic acids), TCA cycle intermediates (fumarate, malate, citrate), and ketone bodies (3-hydroxybutyrate) represent downstream metabolic processing of the mobilized fatty acids. [Inferred] The accumulation of incomplete oxidation intermediates suggests that catabolic capacity is exceeded by substrate supply, a hallmark of metabolic inflexibility in insulin-resistant states. [Model Knowledge]

**Causal inference opportunity:** If longitudinal data are available, testing whether free fatty acid levels precede acylcarnitine and 3-hydroxy fatty acid accumulation would validate the proposed "lipolytic overflow" model. Conversely, if TCA cycle intermediates rise before fatty acids, the module may instead reflect mitochondrial dysfunction as the primary event. [Inferred]

---

### 7. Research Recommendations

#### 7.1 High-Priority Experimental Validations

1. **PPAR target gene panel:** Measure expression of PPARA and PPARG target genes (e.g., CPT1A, ACOX1, FABP4, CD36, ADIPOQ) in the cohort to confirm that the fatty acid ligands identified in this module are actively engaging PPAR transcriptional programs. [Inferred]

2. **Beta-oxidation capacity assessment:** The co-expression of free fatty acids with 3-hydroxy intermediates and acylcarnitines suggests incomplete beta-oxidation. Measure plasma acylcarnitine ratios (e.g., C8/C2, C10/C2) as a functional readout of mitochondrial fatty acid oxidation efficiency. [Inferred]

3. **9-Hydroxystearate quantification in colorectal tissue:** Given the literature evidence for its antiproliferative activity against HT-29 cells [Literature] and the module's strong colorectal cancer association (14 members) [KG Evidence], quantify 9-hydroxystearate in matched tumor and normal tissue to assess its potential as a protective endogenous lipid.

4. **GPR84 activation assay:** Test whether the medium-chain fatty acid concentrations observed in this cohort (caprylate, myristate, 3-hydroxymyristate, 3-hydroxylaurate) are sufficient to activate GPR84 in immune cell models, linking the metabolic module to immunomodulatory signaling. [KG Evidence; Inferred]

#### 7.2 Recommended Literature Searches

1. Search for recent publications linking dicarboxylic acid accumulation (sebacate, dodecanedioate, hexadecanedioate) to T2D progression, as this omega-oxidation signature may be an underappreciated biomarker panel. [Inferred]

2. Search for studies examining the relationship between N-acyl taurine conjugates (N-oleoyltaurine, N-stearoyltaurine) and PPAR signaling in metabolic disease, as these emerging signaling lipids are poorly characterized. [KG Evidence; Model Knowledge]

3. Search for *Clostridioides difficile*-associated fatty acid metabolites in inflammatory bowel disease cohorts, given the microbiome cluster identified in this module (hexanoylcarnitine, hexanoylglutamine, octadecenedioylcarnitine, octadecenedioate). [KG Evidence]

#### 7.3 Follow-Up Analyses

1. **Cross-module comparison:** Compare this Black module with adjacent WGCNA modules to determine whether BCAAs, ceramides, and intact triglycerides cluster nearby or in distant modules. Proximity would suggest related but temporally or mechanistically separated programs; distance would suggest independent dysregulation.

2. **Module eigengene correlation with clinical variables:** Correlate the Black module eigengene with fasting insulin, HOMA-IR, BMI, and triglyceride levels to quantify the module's association with clinical insulin resistance.

3. **Enrichment for fatty acid oxidation disorder genes:** The curated associations with MCAD deficiency (3 members), VLCAD deficiency (decanoylcarnitine, octanoylcarnitine), and inherited obesity (3 members) suggest that rare inborn errors of fatty acid oxidation share metabolic signatures with this common-disease module. [KG Evidence] A formal enrichment test against all fatty acid oxidation disorder gene panels would clarify whether this module recapitulates a partial beta-oxidation defect phenocopy.

4. **Cold-start entity annotation:** Sixteen module members have zero knowledge graph edges (including laurate, caprate, butenoylglycine, trans-2-hexenoylglycine, and several novel dicarboxylic acid species). [KG Evidence] These represent annotation gaps. Submitting these structures to HMDB, ChEBI, and LIPID MAPS for formal curation would improve future knowledge graph analyses.

---

*Report generated from KRAKEN knowledge graph analysis. All claims are tagged with evidence sources: [KG Evidence] for direct knowledge graph findings, [Literature] for grounded literature citations, [Model Knowledge] for general biomedical knowledge, and [Inferred] for synthesized conclusions. Tier 3 predictions carry an estimated approximately 18% probability of progressing to clinical investigation and require independent experimental validation.*

### Literature References

Papers discovered via semantic search. 12 unique papers across 6 hypotheses.

| Hypothesis | Citation | Link | Relevance |
|------------|----------|------|-----------|
| Inferred role of UNII:14E51136PO |  (2005) "A kinetic and thermodynamic study on hydrolysis of sodium laurate in aqueous phase accompanied by tr..." | [Link](https://www.sciencedirect.com/science/article/abs/pii/S0927776505002390) | laurate (NaLA), and its ... 7]. However ... of NaLA. In ... species of lauric acid (LA) molecules are quickly ... phase;... |
| Inferred role of CHEBI:77083 |  (2025) "Assessment and Application of Acylcarnitines Summations as Auxiliary Quantization Indicator for Prim..." | [Link](https://www.mdpi.com/2409-515X/11/2/47) | s are referred primary carnitine deficiency (PCD) when a low free carnitine (C0) concentration (<10 μmol/L) is detected,... |
| Inferred role of CHEBI:88552 |  (2023) "Cis-2-Decenoic Acid and Bupivacaine Delivered from Electrospun Chitosan Membranes Increase Cytokine ..." | [Link](https://www.mdpi.com/1999-4923/15/10/2476) | Cis-2-Decenoic Acid and Bupivacaine Delivered from Electrospun Chitosan Membranes Increase Cytokine Production in Derm... |
| Inferred role of CHEBI:178069 |  (2020) "Dissecting Cellular Mechanisms of Long-Chain Acylcarnitines-Driven Cardiotoxicity: Disturbance of Ca..." | [Link](https://www.mdpi.com/1422-0067/21/20/7461) | Figure 1 Concentration-dependent effects of PC and MC: Ca 2+ ... sparks and Ca 2+ -enriched microdomains, Ca 2+ overload... |
| Inferred role of CHEBI:77083 |  (2019) "Preoperative plasma fatty acid metabolites inform risk of prostate cancer progression and may be use..." | [Link](https://link.springer.com/article/10.1186/s12885-019-6418-2) | 12) ... C16) ... 034) ... (p = ... 0033). In ... , medium-chain acylcarnitines were positively ... (p = ... 0.032, Log-R... |
| Inferred role of CHEBI:178069; Inferred role of CHEBI:77083; Inferred role of PUBCHEM.COMPOUND:129692017 |  (2013) "Quantification of plasma carnitine and acylcarnitines by high-performance liquid chromatography-tand..." | [Link](https://link.springer.com/article/10.1007/s00216-013-7309-z) | Carnitine is an amino acid derivative that plays a key role in energy metabolism. Endogenous carnitine is found in its f... |
| Inferred role of UNII:14E51136PO |  (2016) "Solvent Free Lipase Catalysed Synthesis of Ethyl Laurate: Optimization and Kinetic Studies \| Applied..." | [Link](https://link.springer.com/article/10.1007/s12010-016-2177-6) | Solvent Free Lipase Catalysed Synthesis of Ethyl Laurate: Optimization and Kinetic Studies \| Applied Biochemistry and Bi... |
| Inferred role of CHEBI:178069 |  (2025) "Structural annotation of acylcarnitines detected in SRM 1950 using collision-induced dissociation an..." | [Link](https://link.springer.com/article/10.1007/s00216-025-06234-y) | Acylcarnitines are esters formed through the conjugation of fatty acids with carnitine. Their primary biological role is... |
| Inferred role of CHEBI:229769 |  (2019) "Synthesis of 9-Hydroxystearic Acid Derivatives and Their Antiproliferative Activity on HT 29 Cancer ..." | [Link](https://www.mdpi.com/1420-3049/24/20/3714) | -Hydroxystearic acid ... -HSA) ... antiproliferative and ... effects against cancer cells ... A series of derivatives ..... |
| Inferred role of CHEBI:88552 |  (2016) "The Novel Effect of cis-2-Decenoic Acid on Biofilm Producing Pseudomonas aeruginosa" | [Link](https://www.mdpi.com/2036-7481/6/1/6158) | The Novel Effect of cis-2 ... The Novel Effect of cis-2-Decenoic Acid on Biofilm Producing Pseudomonas aeruginosa ...... |
| Inferred role of CHEBI:229769 |  (2022) "Towards an understanding of oleate hydratases and their application in industrial processes \| Microb..." | [Link](https://link.springer.com/article/10.1186/s12934-022-01777-6) | Fatty acid hydratases are able to hydroxylate unsaturated fatty acids. A plethora of fatty acid hydratases, which conver... |
| Inferred role of CHEBI:229769 |  (2019) "X-Ray Crystal Structures and Organogelator Properties of (R)-9-Hydroxystearic Acid" | [Link](https://www.mdpi.com/1420-3049/24/15/2854) | (R)-9-hydroxystearic acid ... (R)-9-HSA, is a chiral nonrac ... hydroxyacid of natural origin possessing interesting pro... |