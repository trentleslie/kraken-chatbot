# Green Module Run on Opus 4.8: Discovery Output (83-analyte, dev, 2026-06-24)

This document presents the discovery output of a full-module run of the **Green** WGCNA module through the Kraken 12-node discovery pipeline, executed on the dev environment on 2026-06-24 (commit `098093e`) with BioMapper entity resolution enabled. The SDK-backed pipeline nodes ran on the Opus 4.8 model rather than the default Sonnet; this run is published for direct model comparison. It is one of a per-module series over the Frailty multi-omics WGCNA modules; see the [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm).

## Run Provenance

The run submitted 83 named analytes, parsed 56 at intake, and resolved 56 distinct entities (13 biomapper, 42 fuzzy, 1 semantic) to 45 distinct CURIEs. Triage classified 6 well-characterized, 24 moderate, 19 sparse, and 7 cold-start, with 0 edge-count measurement failures. Downstream analysis produced 1169 direct-KG findings, 21 cold-start findings, 3 biological themes, 10 cross-entity bridges (10 evidence-grounded), and 49 hypotheses supported by 30 literature references. Synthesis emitted a 24889-character report. The run completed in approximately 642.1 s of wall-clock time (status complete, 44 errors). All counts are taken from the reducer-merged pipeline state.

| Stage | Result |
|---|---|
| Submitted | 83 named analytes |
| Intake | 56 parsed |
| Entity resolution | 56 resolved (13 biomapper, 42 fuzzy, 1 semantic) to 45 distinct CURIEs |
| Triage | 6 well-characterized, 24 moderate, 19 sparse, 7 cold-start (0 measurement failures) |
| Direct KG | 1169 findings |
| Cold-start | 21 findings, 18 skipped |
| Pathway enrichment | 3 biological themes |
| Integration | 10 bridges (10 evidence-grounded) |
| Literature grounding | 30 papers |
| Synthesis | 49 hypotheses, 24889-character report |
| Run total | ~642.1 s wall-clock, status complete, 44 errors |

## Related

- Companion run metrics: [Green Module Run on Opus 4.8: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/green-module-run-on-opus-48-pipeline-performance-report-83-analyte-dev-2026-06-24-lgVwoYdwWF)
- Model comparison baseline (Sonnet): [Green Module Run: Discovery Output](https://phwiki.phenoma.ai/doc/green-module-run-discovery-output-83-analyte-dev-2026-06-23-tAMwedxCJR)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---



## Discovery Report: Green WGCNA Module (Sphingolipid, Membrane Lipid, and Immune Signaling Network)

### 1. Executive Summary

This WGCNA Green module encodes a coordinated sphingolipid and membrane phospholipid biosynthetic program, unified by the co-expression of over 70 sphingomyelins, ceramides, glycosphingolipids, ether-linked phosphatidylcholines, and plasmalogens alongside cholesterol and the immune cytokine FLT3LG. [KG Evidence; Inferred] The module's composition indicates that it captures lipid packaging, membrane assembly, and lipoprotein remodeling rather than catabolic or beta-oxidative processes; the conspicuous absence of free ceramides, lysophosphatidylcholines, acylcarnitines, and branched-chain amino acids reinforces this interpretation and suggests that the analytes reflect a biosynthetic steady state downstream of ceramide synthesis. [KG Evidence; Inferred] The singular presence of FLT3LG (a hematopoietic cytokine that drives dendritic cell and natural killer cell differentiation) among these lipids reveals a previously underappreciated interface between innate immune progenitor signaling and circulating sphingolipid homeostasis, with shared disease associations in schizophrenia, diabetes mellitus, and arteriosclerosis that merit targeted investigation. [KG Evidence]

### 2. Key Findings (Tier 1 to 2)

#### 2.1 Module Unifying Theme: Sphingolipid Biosynthesis and Membrane Lipid Assembly

The module comprises approximately 40 sphingomyelin species spanning acyl chain lengths from C14:0 to C25:0 (including odd-chain, hydroxylated, and dihydro forms), 6 ceramide species, 4 glycosylceramides/lactosylceramides, over 15 ether-linked or plasmalogen phosphatidylcholines and phosphatidylethanolamines, cholesterol, and one protein (FLT3LG). [KG Evidence] This composition identifies the module as a sphingolipid and membrane lipid biosynthetic cluster. Pathway enrichment analysis confirmed that 12 input entities share connections to lipid-metabolizing genes (LIPA, APOA1, LIPC, PNLIP, PNPLA3, and others), and 9 entities connect to lipase/bile salt-stimulated lipase proteins (CEL, UniProtKB:P19835). [KG Evidence] The chemical entity bridges further confirm that ceramide (CHEBI:17761), N-acylsphingosines, and lipid second messengers serve as shared intermediary nodes. [KG Evidence]

#### 2.2 FLT3LG: Immune Signaling Hub

FLT3LG (2,425 KG edges) participates in B cell differentiation, dendritic cell differentiation, natural killer cell proliferation, hemopoiesis, embryonic hemopoiesis, leukocyte and lymphocyte activation, and PI3K-Akt signaling. [KG Evidence] It interacts with CD40LG, KITLG, PIK3CA, AKT1, KRAS, RAF1, KIT, MET, ABL1, SYK, FGFR3, and NAMPT, placing it at the nexus of receptor tyrosine kinase signaling cascades. [KG Evidence] Its top disease association is immunodeficiency 125, consistent with its canonical role in immune progenitor expansion. [KG Evidence]

#### 2.3 Cholesterol: Structural and Metabolic Anchor

Cholesterol (4,371 KG edges) exhibits the broadest disease association spectrum in the module, including hypercholesterolemia, Smith-Lemli-Opitz Syndrome, Zellweger Syndrome, mevalonic aciduria, cholesteryl ester storage disease, and Wolman disease, as well as pharmacological connections to the entire statin drug class and bisphosphonate pathways. [KG Evidence] Its co-expression with sphingomyelins and plasmalogens is consistent with its structural role in membrane lipid raft organization rather than bile acid metabolism (bile acids are notably absent; see Gap Analysis). [KG Evidence; Inferred]

#### 2.4 Module-Level Disease Recurrence

Three diseases recur across three or more module members: [KG Evidence]

| Disease | Members | Strongest Evidence |
|---|---|---|
| Schizophrenia (MONDO:0005090) | cholesterol, guanidinoacetate, FLT3LG | Curated |
| Colorectal cancer (MONDO:0005575) | cholesterol, phosphatidylethanolamine, sphingomyelin | Curated |
| Eosinophilic esophagitis type 1 (MONDO:0012451) | guanidinoacetate, phosphatidylethanolamine, sphingomyelin | Curated |
| Amyotrophic lateral sclerosis (MONDO:0004976) | cholesterol, guanidinoacetate, phosphatidylethanolamine | Text-mined |

Two-member recurrences include diabetes mellitus (cholesterol, FLT3LG), arthritic joint disease (guanidinoacetate, FLT3LG), arteriosclerosis (cholesterol, FLT3LG), obesity (cholesterol, guanidinoacetate), and inflammatory bowel disease type 1 (guanidinoacetate, phosphatidylethanolamine). [KG Evidence]

#### 2.5 Cross-Type Bridges: FLT3LG to Metabolites

Multiple two-hop paths connect FLT3LG to cholesterol and guanidinoacetate through shared cellular compartments and intermediary genes. [KG Evidence] The highest-confidence bridges traverse shared localization in the extracellular region (GO:0005576), membrane (GO:0016020), cytoplasm (GO:0005737), and blood (UBERON:0000178). [KG Evidence] A mechanistically notable bridge routes through IL1B: FLT3LG affects IL1B, which in turn affects cholesterol. [KG Evidence] This path suggests that FLT3LG-driven immune activation may modulate cholesterol metabolism via inflammatory cytokine cascades. The weakest evidence leg for compartment-based bridges is "curated-neutral," while tissue and IL1B bridges rest on "text-mined" evidence. [KG Evidence]

#### 2.6 High-Priority Individual Members

Phosphatidylethanolamine (4,442 edges) and ceramide (219 edges) serve as the most densely connected lipid members. [KG Evidence] Ceramide's top disease association is Fabry disease (a lysosomal storage disorder), consistent with the module's sphingolipid identity. [KG Evidence] Guanidinoacetate (210 edges) is biochemically distinct as a creatine biosynthesis intermediate; its top disease association is AGAT deficiency (cerebral creatine deficiency syndrome). [KG Evidence] Its co-expression with sphingolipids is unexpected and discussed further in Section 3.

### 3. Novel Predictions (Tier 3)

#### 3.1 FLT3LG as a Circulating Proxy for Sphingolipid-Modulated Immune Tone

**Prediction**: FLT3LG co-expression with the sphingolipid module reflects a biological coupling between dendritic cell/NK cell progenitor signaling and membrane sphingolipid composition, potentially mediated by sphingomyelin-rich lipid rafts that regulate FLT3 receptor clustering and downstream PI3K-Akt signaling.

**Structural logic chain**: FLT3LG drives dendritic cell differentiation and PI3K-Akt signaling [KG Evidence]. Sphingomyelins and cholesterol are canonical components of lipid rafts that regulate receptor tyrosine kinase signaling [Model Knowledge]. The module co-expresses FLT3LG with over 30 sphingomyelin species and cholesterol [KG Evidence]. Sphingomyelin synthase activity (converting ceramide to sphingomyelin) modulates membrane raft composition and receptor signaling, as demonstrated by Barcelo-Coblijn et al. (2011), who showed that sphingomyelin synthase activation alters membrane order and lipid raft packing, affecting Ras and Fas receptor localization [Literature: Barcelo-Coblijn et al., 2011]. Maceyka et al. (2014) further established that sphingolipid metabolites, particularly ceramide and sphingosine-1-phosphate, regulate immunity and inflammation through membrane-mediated signaling [Literature: Maceyka et al., 2014].

**Calibration**: Approximately 18% of computational predictions of this type progress to clinical investigation. This prediction is strengthened by the convergence of KG-derived FLT3LG pathway data and literature evidence on sphingolipid-mediated immune signaling.

**Validation step**: Measure FLT3LG protein levels in plasma alongside sphingomyelin panel quantitation in an independent cohort; test whether sphingomyelin synthase (SGMS1/SGMS2) expression correlates with FLT3LG levels or dendritic cell counts.

#### 3.2 Guanidinoacetate as a Marker of Creatine-Phospholipid Metabolic Crosstalk

**Prediction**: Guanidinoacetate co-expression with sphingolipids reflects shared hepatic and renal biosynthetic regulation, where methylation demand (creatine synthesis consumes S-adenosylmethionine) competes with phosphatidylethanolamine N-methyltransferase activity (generating phosphatidylcholine from phosphatidylethanolamine).

**Structural logic chain**: Guanidinoacetate is the immediate precursor to creatine, synthesized in the kidney and methylated in the liver by GAMT using S-adenosylmethionine (SAM) [Model Knowledge]. Phosphatidylethanolamine, a module member (4,442 KG edges), is converted to phosphatidylcholine by PEMT, also consuming SAM [Model Knowledge]. Guanidinoacetate and phosphatidylethanolamine share disease associations in eosinophilic esophagitis and inflammatory bowel disease [KG Evidence]. No direct KG edge links guanidinoacetate to sphingolipid metabolism; this connection is inferred from shared SAM dependency and co-expression.

**Calibration**: Approximately 18% of such computational predictions advance to validation. This prediction rests on [Inferred] metabolic logic rather than direct KG evidence and requires metabolic flux analysis for confirmation.

**Validation step**: Quantify SAM, SAH (S-adenosylhomocysteine), creatine, and phosphatidylcholine in the same samples to test whether guanidinoacetate levels inversely correlate with PEMT-derived phosphatidylcholine, indicative of methylation competition.

#### 3.3 Schizophrenia as a Sphingolipid-Immune Convergence Phenotype

**Prediction**: The three-member recurrence of schizophrenia (cholesterol, guanidinoacetate, FLT3LG) suggests that this module captures a lipid-immune axis relevant to neuropsychiatric risk, potentially through altered myelin lipid composition and microglial progenitor signaling.

**Structural logic chain**: Schizophrenia is associated with cholesterol (curated), guanidinoacetate (curated), and FLT3LG (curated) in the knowledge graph [KG Evidence]. Cholesterol and sphingomyelins are major myelin lipid components [Model Knowledge]. FLT3LG drives microglial precursor and dendritic cell differentiation [KG Evidence]. Guanidinoacetate reflects creatine metabolism, and cerebral creatine deficiency causes intellectual disability [KG Evidence: AGAT deficiency association]. The convergence of myelin lipids, immune progenitor signaling, and brain energy metabolism on schizophrenia risk is mechanistically coherent.

**Calibration**: Approximately 18% of such computational predictions progress to clinical investigation. The curated evidence level for all three members lends this prediction moderate credibility.

**Validation step**: Examine whether circulating sphingomyelin profiles, FLT3LG levels, and guanidinoacetate concentrations differ between schizophrenia cases and controls in existing biobank cohorts (e.g., UK Biobank metabolomics).

#### 3.4 Tryptophan Betaine as an IDO/Kynurenine Pathway Reporter

**Prediction**: Tryptophan betaine (hypaphorine), a cold-start entity with no KG edges, may reflect indoleamine 2,3-dioxygenase (IDO) pathway activity, linking immune activation (FLT3LG-driven dendritic cells express IDO) to tryptophan metabolism.

**Structural logic chain**: Tryptophan betaine is structurally a trimethylated tryptophan derivative, semantically similar to 1-methyltryptophan (similarity 0.96), a known IDO inhibitor [KG Evidence: semantic analogue]. Dendritic cells, whose differentiation is driven by FLT3LG [KG Evidence], are major expressors of IDO [Model Knowledge]. IDO catabolizes tryptophan via the kynurenine pathway, and its activity is immunomodulatory [Model Knowledge]. No direct KG evidence supports this connection; the claim relies entirely on structural analogy and model knowledge.

**Calibration**: Approximately 18% of such computational predictions progress to clinical investigation. This prediction is speculative (cold-start entity, no KG edges) and should be treated with particular caution.

**Validation step**: Quantify kynurenine, kynurenic acid, and tryptophan alongside tryptophan betaine in module samples; test correlation with FLT3LG levels and dendritic cell markers (e.g., CD11c+, HLA-DR+).

### 4. Biological Themes

#### 4.1 Sphingomyelin Biosynthetic Program

The dominant theme is sphingomyelin metabolism. The module contains sphingomyelins with acyl chains ranging from C14:0 to C25:0, encompassing saturated, monounsaturated, and diunsaturated species, as well as dihydrosphingomyelin and hydroxylated forms. [KG Evidence] This breadth indicates that the module captures the output of ceramide synthases (CerS) with diverse chain-length specificities, followed by sphingomyelin synthase activity. [Model Knowledge] The pathway enrichment analysis confirms connections to lipid-metabolizing genes including LIPA, LIPC, and CEL. [KG Evidence]

#### 4.2 Ether Lipid and Plasmalogen Membrane Composition

Over 15 ether-linked phosphatidylcholines (1-alkyl-2-acyl-GPC, designated "O-" prefix) and plasmalogens (1-alkenyl-2-acyl-GPC/GPE, designated "P-" prefix) populate the module. [KG Evidence] These lipids serve as endogenous antioxidants and structural membrane components enriched in neural tissue, immune cells, and lipid rafts. [Model Knowledge] Their co-expression with sphingomyelins and cholesterol reinforces the interpretation of this module as a membrane composition and lipid raft assembly program.

#### 4.3 Immune Progenitor Signaling

FLT3LG connects the lipid program to hematopoietic stem cell differentiation, B cell and dendritic cell lineage commitment, and NK cell proliferation. [KG Evidence] Hub-filtering note: FLT3LG (2,425 edges) and cholesterol (4,371 edges) are high-connectivity nodes. Their disease associations (especially generic categories such as "cancer pathways" or "smoking") should be interpreted cautiously, as high-degree nodes accumulate associations that may not reflect module-specific biology. [Inferred] The schizophrenia, diabetes mellitus, and arteriosclerosis recurrences, however, involve at least two well-characterized members and carry curated evidence, lending them higher specificity.

### 5. Gap Analysis

#### 5.1 Informative Absences

| Expected Entity | Interpretation |
|---|---|
| Free ceramides (Cer d18:1/16:0, d18:1/24:1) | The module contains sphingomyelins and glycosylceramides but no free ceramides, indicating capture of a post-ceramide biosynthetic state with active sphingomyelin synthase or glucosylceramide synthase flux. [KG Evidence; Inferred] |
| Lysophosphatidylcholines (e.g., lysoPC 18:2) | Intact diacyl and ether-linked PCs are present; lysoPCs are absent. The module captures lipid biosynthesis and packaging, not phospholipase A2-mediated degradation. [KG Evidence; Inferred] |
| Acylcarnitines (C2, C4-DC, C5) | Absence excludes mitochondrial beta-oxidation from this module, reinforcing the biosynthetic/structural lipid interpretation. [Inferred] |
| BCAAs and aromatic amino acids | These canonical insulin resistance markers likely segregate into amino acid-centric modules. Their absence confirms the module's specificity for sphingolipid over amino acid metabolism. [Inferred] |
| Bile acids (glycocholate, taurocholate) | Cholesterol is present in its structural/membrane role; its catabolic products are absent, indicating that the module does not capture enterohepatic cholesterol catabolism. [Inferred] |
| Adiponectin, leptin | FLT3LG is the sole protein; other adipokines/cytokines would be expected in adipose-specific or inflammatory modules. [Inferred] |

#### 5.2 Standard (Platform-Related) Gaps

Insulin, C-peptide, HbA1c, and HOMA-IR are clinical or immunoassay-based measurements not expected on untargeted metabolomics platforms. [Inferred] Their absence is non-informative regarding the biology of the module.

#### 5.3 Open World Assumption

Under the Open World Assumption, the absence of an entity from this module means "unstudied in this context" rather than "biologically irrelevant." The informative absences above constrain the biological interpretation of the module but do not exclude the possibility that the absent entities interact with module members in unstudied or unmeasured contexts.

### 6. Temporal Context

No explicit longitudinal metadata accompanies this module. The following causal inference opportunities are noted. [Inferred]

**Upstream causes (candidate drivers)**: FLT3LG signaling, which activates PI3K-Akt and downstream lipid biosynthetic transcription factors (e.g., SREBP family via Akt-mTOR axis [Model Knowledge]), could serve as an upstream driver of sphingomyelin and cholesterol production. Ceramide synthase (CerS) expression levels represent another candidate upstream determinant that would not appear as a module member if ceramides are rapidly converted to sphingomyelins.

**Downstream consequences**: Altered membrane lipid raft composition (sphingomyelin and cholesterol enrichment) could modulate receptor signaling sensitivity, immune cell activation thresholds, and lipoprotein assembly and secretion.

**Causal inference opportunity**: If longitudinal sampling is available, Granger causality or vector autoregressive models applied to FLT3LG and sphingomyelin species trajectories could test whether immune signaling precedes lipid remodeling or vice versa.

### 7. Research Recommendations

#### Priority 1: Experimental Validations

1. **FLT3LG and sphingomyelin correlation analysis**: Quantify FLT3LG protein and a targeted sphingomyelin panel (palmitoyl SM, stearoyl SM, behenoyl SM, lignoceroyl SM) in an independent cohort to confirm co-expression outside the discovery dataset. [KG Evidence supports the co-expression; validation requires replication]

2. **Sphingomyelin synthase expression profiling**: Measure SGMS1 and SGMS2 mRNA or protein in circulating immune cells (monocytes, dendritic cell precursors) and correlate with module metabolite levels. This tests whether the module reflects active sphingomyelin synthase flux in immune lineages. [Inferred]

3. **Schizophrenia biomarker panel**: Test the discriminative capacity of the cholesterol, guanidinoacetate, and FLT3LG triad (with or without sphingomyelin species) for schizophrenia case-control classification in psychiatric biobank samples. [KG Evidence: three-member curated disease recurrence]

#### Priority 2: Literature and Database Searches

4. **Tryptophan betaine (hypaphorine) in metabolomics**: Conduct a systematic literature search for tryptophan betaine in human metabolomics studies; this compound is reported in dietary sources (legumes) and may serve as a diet-microbiome interaction marker rather than an endogenous metabolite. [Model Knowledge]

5. **Guanidinoacetate and sphingolipid co-regulation**: Search for studies examining SAM flux partitioning between creatine synthesis and phospholipid methylation, particularly in the context of sphingolipid metabolism. [Inferred]

6. **Eosinophilic esophagitis and sphingolipids**: The three-member disease recurrence (guanidinoacetate, phosphatidylethanolamine, sphingomyelin) for eosinophilic esophagitis (MONDO:0012451) is unexpected. A targeted literature search for sphingolipid alterations in eosinophilic esophagitis could reveal novel mechanistic connections. [KG Evidence]

#### Priority 3: Follow-Up Computational Analyses

7. **Cross-module comparison**: Compare the Green module with other WGCNA modules (particularly those containing free ceramides, lysoPCs, BCAAs, and acylcarnitines) to map the broader metabolic network and identify inter-module regulatory edges. [Inferred]

8. **Hub-filtered disease enrichment**: Repeat disease enrichment analysis after removing cholesterol and phosphatidylethanolamine (the two highest-degree nodes at 4,371 and 4,442 edges, respectively) to identify disease associations specific to the sphingomyelin sub-network rather than driven by hub connectivity. [Inferred]

9. **FDR correction for disease recurrence**: Apply false discovery rate correction to the module-level disease recurrence analysis (currently nine diseases with two or more members) to identify statistically robust disease associations. [Inferred]

10. **Ceramide-to-sphingomyelin ratio modeling**: If ceramide species were measured but assigned to a different WGCNA module, compute inter-module Cer/SM ratios as a functional readout of sphingomyelin synthase activity and test associations with clinical outcomes. [Inferred]

---

**Methodological Notes**: Entity resolution achieved 56/56 matches, though many lipid species were resolved at 70% confidence via fuzzy matching to structurally related but non-identical compounds (e.g., triacylglycerols instead of diacyl phospholipids). Seven entities (tryptophan betaine, glycosyl-N-stearoyl-sphingosine, 2-aminoheptanoate, behenoyl dihydrosphingomyelin, myristoyl dihydrosphingomyelin, glycosyl-N-palmitoyl-sphingosine, N-behenoyl-sphingadienine) had zero KG edges (cold-start), limiting the knowledge graph analysis for these members. The moderate and sparse coverage entities (comprising the majority of sphingomyelin and phospholipid species) had 2 to 121 edges, providing limited but non-trivial connectivity. Findings for these entities should be interpreted with awareness of incomplete KG representation for specialized lipid species.

### Literature References

Papers discovered via semantic search. 12 unique papers across 5 hypotheses.

| Hypothesis | Citation | Link | Relevance |
|------------|----------|------|-----------|
| Inferred role of CHEBI:85814 | Gwendolyn Barceló‐Coblijn et al. (2011) "Sphingomyelin and sphingomyelin synthase (SMS) in the malignant transformation of glioma cells and i..." | [DOI](https://doi.org/10.1073/pnas.1115484108) | — |
| Bridge: Gene → SmallMolecule (2 hops) | Ana Viñuela et al. (2021) "Genetic analysis of blood molecular phenotypes reveals regulatory networks affecting complex traits:..." | [DOI](https://doi.org/10.1101/2021.03.26.21254347) | — |
| Inferred role of CHEBI:85814 | Richard Kolesnick (2002) "The therapeutic potential of modulating the ceramide/sphingomyelin pathway" | [DOI](https://doi.org/10.1172/jci0216127) | — |
| Bridge: Gene → SmallMolecule (2 hops) | Bartijn C. H. Pieters et al. (2019) "Macrophage-Derived Extracellular Vesicles as Carriers of Alarmins and Their Potential Involvement in..." | [DOI](https://doi.org/10.3389/fimmu.2019.01901) | — |
| Inferred role of CHEBI:85814 | Shin KO et al. (2021) "N-Palmitoyl Serinol Stimulates Ceramide Production through a CB1-Dependent Mechanism in In Vitro Mod..." | [DOI](https://doi.org/10.3390/ijms22158302) | — |
| Inferred role of PUBCHEM.COMPOUND:6443616 |  (2017) "(R)-N-((2S,3S,4R)-3,4-dihydroxy-15-methyl-1-(((2R,3R,4S,5S,6R)-3,4,5-trihydroxy-6-(hydroxymethyl)tet..." | [Link](http://www.nature.com/articles/nchembio.2347/compounds/11) | (((2R ... S,5S,6R ... pyran- ... TBAF (1 M in THF, 45 μL, 45 μmol) was added at 50 °C to a solution of Glucosylceramide... |
| Inferred role of PUBCHEM.COMPOUND:6443616 |  (2012) "4,8-Sphingadienine and 4-hydroxy-8-sphingenine activate ceramide production in the skin \| Lipids in ..." | [Link](https://link.springer.com/article/10.1186/1476-511X-11-108) | Ingestion of glucosylceramide improves transepidermal water loss (TEWL) from the skin, but the underlying mechanism by w... |
| Inferred role of PUBCHEM.COMPOUND:6443616 |  (2005) "Efficient stereocontrolled synthesis of sphingadienine derivatives - ScienceDirect" | [Link](https://www.sciencedirect.com/science/article/abs/pii/S0040402005012743) | Sphingolipids, for example, ceramides, sphingomyelin, cerebrosides, and gangliosides, are constituents of eukaryotic cel... |
| Bridge: Gene → SmallMolecule (2 hops) |  (2026) "Frontiers \| ANGPTL3 and residual atherosclerotic risk: from lipid metabolism to therapeutic targetin..." | [Link](https://www.frontiersin.org/journals/endocrinology/articles/10.3389/fendo.2025.1706091/full) | With respect to cholesterol metabolism, ANGPTL3 can be summarized as a dual hub acting upstream and downstream: on one e... |
| Inferred role of CHEBI:83363 |  (2021) "Hydroxylated Fatty Acids: The Role of the Sphingomyelin Synthase and the Origin of Selectivity" | [Link](https://www.mdpi.com/2077-0375/11/10/787) | Sphingolipids containing 2-hydroxylated fatty acids (2OHFA) are present in most organisms [32] and are important compone... |
| Inferred role of CHEBI:83363 |  (2021) "Stereoselective Synthesis of Novel Sphingoid Bases Utilized for Exploring the Secrets of Sphinx" | [Link](https://www.mdpi.com/1422-0067/22/15/8171) | Sphingolipids are ubiquitous in eukaryotic plasma membranes and play major roles in human and animal physiology and dise... |
| Inferred role of CHEBI:83899 |  (2025) "Table 1 Glycosphingolipid species detected by µL-flow 4D-RP-LC-TIMS-PASEF analysis in human serum" | [Link](https://www.nature.com/articles/s41467-025-59755-6/tables/1) | Table 1 Glycosphingolipid species detected by µL-flow 4D-RP-LC-TIMS-PASEF analysis in human serum ... Neutral glycosphin... |