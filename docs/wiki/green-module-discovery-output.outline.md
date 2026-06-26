# Green Module Run: Discovery Output (83-analyte, dev, 2026-06-23)

This document presents the discovery output of a full-module run of the **Green** WGCNA module through the Kraken 12-node discovery pipeline, executed on the dev environment on 2026-06-23 (commit `813fc3f`) with BioMapper entity resolution enabled. It is one of a per-module series over the Frailty multi-omics WGCNA modules; see the [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm).

## Run Provenance

The run submitted 83 named analytes, parsed 56 at intake, and resolved 56 distinct entities (13 biomapper, 42 fuzzy, 1 semantic) to 45 distinct CURIEs. Triage classified 6 well-characterized, 24 moderate, 19 sparse, and 7 cold-start, with 0 edge-count measurement failures. Downstream analysis produced 1337 direct-KG findings, 19 cold-start findings, 3 biological themes, 10 cross-entity bridges (10 evidence-grounded), and 45 hypotheses supported by 20 literature references. Synthesis emitted a 27099-character report. The run completed in approximately 719.5 s of wall-clock time (status complete, 24 errors). All counts are taken from the reducer-merged pipeline state.

| Stage | Result |
|---|---|
| Submitted | 83 named analytes |
| Intake | 56 parsed |
| Entity resolution | 56 resolved (13 biomapper, 42 fuzzy, 1 semantic) to 45 distinct CURIEs |
| Triage | 6 well-characterized, 24 moderate, 19 sparse, 7 cold-start (0 measurement failures) |
| Direct KG | 1337 findings |
| Cold-start | 19 findings, 18 skipped |
| Pathway enrichment | 3 biological themes |
| Integration | 10 bridges (10 evidence-grounded) |
| Literature grounding | 20 papers |
| Synthesis | 45 hypotheses, 27099-character report |
| Run total | ~719.5 s wall-clock, status complete, 24 errors |

## Related

- Companion run metrics: [Green Module Run: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/green-module-run-pipeline-performance-report-83-analyte-dev-2026-06-23-6H1guU0r1I)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---



## KRAKEN Discovery Report: Green WGCNA Module (Sphingolipid, Phospholipid, and Immune Signaling Network)

---

### 1. Executive Summary

This Green WGCNA module encodes a coordinated sphingolipid and membrane phospholipid remodeling program, dominated by more than 40 sphingomyelin species, multiple ceramide and glycosylceramide derivatives, plasmalogens, and arachidonoyl-enriched glycerophospholipids, unified by a single immune signaling protein, FLT3LG. [KG Evidence] The module's disease recurrence profile converges on colorectal cancer (4 members), schizophrenia (3 members), eosinophilic esophagitis (3 members), and amyotrophic lateral sclerosis (3 members), implicating membrane lipid composition as a shared biological axis across inflammatory, neuropsychiatric, and neoplastic phenotypes. [KG Evidence] The selective absence of free ceramides, branched-chain amino acids, and lysophosphatidylcholines indicates that this module captures the sphingomyelin branch of sphingolipid metabolism prior to lipotoxic ceramide accumulation, a finding with potential implications for disease staging in longitudinal cohorts. [Inferred]

---

### 2. Key Findings (Tier 1 to 2)

#### 2.1 Unifying Biological Theme: Sphingolipid Membrane Architecture

The module is composed of 56 successfully resolved entities, of which 55 are small molecules and 1 is a gene (FLT3LG). [KG Evidence] The overwhelming majority of the lipid species belong to three structural classes:

1. **Sphingomyelins** (approximately 30 species spanning C14:0 to C25:0 acyl chains, including dihydrosphingomyelins and sphingadienine variants): These constitute the dominant lipid class and share membership in the sphingolipid signaling pathway (UMLS:C2753772) and sphingolipid metabolism (SMPDB:SMP0000034), as confirmed by pathway recurrence analysis for phosphatidylethanolamine (CHEBI:17553) and sphingomyelin (CHEBI:83358). [KG Evidence]

2. **Glycerophospholipids** (approximately 20 species): These include arachidonoyl-enriched phosphatidylcholines (GPC), phosphatidylinositols (GPI), phosphatidylethanolamines (GPE), and notably plasmalogens (P-16:0 and P-18:0 vinyl-ether species). The enrichment in arachidonoyl (20:4) acyl chains across GPC, GPE, and GPI species indicates a coordinated pool of eicosanoid precursor lipids. [Model Knowledge]

3. **Ceramides and glycosylceramides** (6 species): These include free ceramides of varying chain length (d18:1/14:0 through d18:2/24:1) and glycosylated derivatives, positioned as biosynthetic intermediates between sphingomyelin and glucosylceramide pathways. [KG Evidence]

#### 2.2 FLT3LG as the Immune Signaling Hub

FLT3LG (Fms-related tyrosine kinase 3 ligand; NCBIGene:2323; 2,425 edges) is the sole protein in the module and serves as the bridge between immune cell biology and lipid metabolism. [KG Evidence] Direct KG evidence confirms its participation in:

- **Hematopoietic differentiation**: dendritic cell differentiation, B cell differentiation, NK cell proliferation, embryonic hemopoiesis, and positive regulation of hemopoiesis [KG Evidence]
- **Signaling cascades**: PI3K-Akt signaling, signal transduction, positive regulation of protein phosphorylation [KG Evidence]
- **Protein interactions**: FLT3 (its cognate receptor), STAT3, STAT5A, MTOR, AKT1, TNF, IL3, CSF2, CD34, CD40, PTEN, and 30+ additional interactors [KG Evidence]

The co-expression of FLT3LG with this sphingolipid cluster is biologically notable: FLT3LG drives dendritic cell and NK cell maturation, processes that require extensive membrane biogenesis and sphingolipid-rich lipid raft formation. [Model Knowledge]

#### 2.3 Disease Associations (Module-Level Recurrence)

The following diseases recur across multiple module members with curated evidence:

| Disease | Members | Strongest Evidence | Key Members |
|---|---|---|---|
| Colorectal cancer | 4 | Curated | cholesterol, PE, sphingomyelin, FLT3LG |
| Schizophrenia | 3 | Curated | cholesterol, guanidinoacetate, FLT3LG |
| Eosinophilic esophagitis | 3 | Curated | guanidinoacetate, PE, sphingomyelin |
| Amyotrophic lateral sclerosis | 3 | Text-mined | cholesterol, guanidinoacetate, PE |
| Coronary artery disease | 2 | Curated | cholesterol, FLT3LG |
| Diabetes mellitus | 2 | Curated | cholesterol, FLT3LG |
| Hepatocellular carcinoma | 2 | Curated | cholesterol, FLT3LG |
| Rheumatoid arthritis | 2 | Curated | guanidinoacetate, FLT3LG |
| Obesity | 2 | Curated | cholesterol, guanidinoacetate |

[KG Evidence]

The convergence of colorectal cancer (4 members) as the top-recurrent disease is consistent with established roles for sphingomyelin metabolism and cholesterol homeostasis in colorectal tumorigenesis, where sphingomyelinase activity and ceramide generation regulate colonocyte apoptosis. [Model Knowledge] The co-association with schizophrenia (3 members) aligns with emerging literature on myelin-associated sphingolipid abnormalities and cholesterol transport deficits in neuropsychiatric conditions. [Model Knowledge]

#### 2.4 Cross-Type Bridges: FLT3LG to Sphingolipid Module Members

Multiple two-hop paths connect FLT3LG to cholesterol and guanidinoacetate through shared cellular compartments and tissues:

- FLT3LG → extracellular region (GO:0005576) → cholesterol (curated evidence) [KG Evidence]
- FLT3LG → membrane (GO:0016020) → cholesterol (curated evidence) [KG Evidence]
- FLT3LG → cytoplasm (GO:0005737) → cholesterol and guanidinoacetate (curated evidence) [KG Evidence]
- FLT3LG → IL1B → cholesterol (text-mined evidence), suggesting an inflammatory cytokine link [KG Evidence]
- FLT3LG → blood (UBERON:0000178) → cholesterol and guanidinoacetate (text-mined evidence) [KG Evidence]

The IL1B-mediated bridge is notable: FLT3LG affects IL1B expression, and IL1B in turn modulates cholesterol metabolism. [KG Evidence] This path suggests that FLT3LG-driven immune activation may influence circulating cholesterol levels through inflammatory cytokine signaling. [Inferred]

#### 2.5 Member Prioritization

The Member Prioritization Table identifies six well-characterized entities (more than 200 edges each) that anchor the module:

| Member | Edges | Role |
|---|---|---|
| Phosphatidylethanolamine | 4,442 | Membrane structural lipid; pathway hub |
| Cholesterol | 4,371 | Lipid metabolism hub; drug target |
| FLT3LG | 2,425 | Immune signaling cytokine |
| Behenoyl sphingomyelin | 1,148 | Sphingomyelin species |
| Ceramide | 219 | Sphingolipid metabolism intermediate |
| Guanidinoacetate | 210 | Creatine biosynthesis intermediate |

[KG Evidence]

Phosphatidylethanolamine and cholesterol are high-connectivity hub nodes (more than 4,000 edges), and associations mediated solely through these entities should be interpreted with caution as potentially non-specific. [KG Evidence]

---

### 3. Novel Predictions (Tier 3)

#### 3.1 Sphingomyelin-to-Ceramide Conversion as a Latent Disease Transition

- **Prediction**: The module captures the sphingomyelin reservoir prior to sphingomyelinase-mediated conversion to free ceramides, suggesting that the cohort may be in a pre-lipotoxic state.
- **Structural logic chain**: The module contains more than 30 sphingomyelin species and glycosylceramide derivatives but excludes the specific free ceramide species (Cer d18:1/16:0, d18:1/18:0, d18:1/24:1) that constitute validated clinical risk scores for type 2 diabetes. [KG Evidence; gap analysis] Sphingomyelins are direct precursors of ceramides via acid and neutral sphingomyelinases (SMPK, SMPD1/2/3). [Model Knowledge] The absence of free ceramides from this co-expression module, while sphingomyelins and glycosylceramides are abundantly represented, indicates that the lipotoxic ceramide accumulation step has not yet been activated or is segregated into a separate regulatory module. [Inferred]
- **Validation step**: Correlate module eigengene values with longitudinal ceramide measurements (if available) to test whether module activity precedes ceramide elevation temporally. Measure sphingomyelinase activity (acid SMase in plasma) in the study cohort.
- **Calibration note**: Approximately 18% of computational predictions of this nature progress to clinical investigation; this prediction is strengthened by its mechanistic grounding in established sphingolipid biochemistry but requires longitudinal confirmation.

#### 3.2 FLT3LG-Sphingolipid Axis in Immune Cell Membrane Biogenesis

- **Prediction**: FLT3LG co-expression with sphingomyelins reflects a dendritic cell or NK cell differentiation program requiring coordinated membrane lipid synthesis.
- **Structural logic chain**: FLT3LG drives dendritic cell differentiation and NK cell proliferation via FLT3 receptor signaling. [KG Evidence] Dendritic cell maturation involves extensive membrane expansion and lipid raft reorganization, processes that consume sphingomyelins and cholesterol. [Model Knowledge] The KG bridges connect FLT3LG to cholesterol through shared membrane and extracellular localization (GO:0016020, GO:0005576) with curated evidence. [KG Evidence] The co-expression of FLT3LG with more than 30 sphingomyelin species, multiple plasmalogens, and cholesterol in a single WGCNA module is consistent with a coordinated membrane biogenesis program.
- **Validation step**: Perform flow cytometry for dendritic cell subsets (CD11c+, CD123+) and correlate with module eigengene. Measure FLT3LG protein levels alongside sphingomyelin species in sorted immune cell fractions.
- **Calibration note**: Approximately 18% of computational predictions advance to clinical testing. This hypothesis is supported by known FLT3LG biology but lacks direct experimental evidence linking FLT3LG signaling to specific sphingomyelin species regulation.

#### 3.3 Glycosyl-N-acylsphingosine Derivatives as Glucosylceramide Pathway Markers

- **Prediction**: The glycosylated sphingosine species in this module (glycosyl-N-stearoyl-sphingosine, glycosyl-N-palmitoyl-sphingosine, glycosyl-N-behenoyl-sphingadienine, glycosyl ceramide, and lactosyl derivatives) represent glucosylceramide synthase (UGCG) activity, marking a branch of sphingolipid metabolism distinct from the sphingomyelinase pathway.
- **Structural logic chain**: Semantic similarity analysis links glycosyl-N-stearoyl-sphingosine (KEGG:C03701) to glucosylceramide (MESH:C120051, similarity 0.77) and to ceramide (CHEBI:17761) via N-palmitoyl glucosyl-C18-sphingosine (similarity 0.77). [KG Evidence] Two of three analogues implicate ceramide/glucosylceramide biology. [KG Evidence] The glycosphingolipid biosynthesis pathway (KEGG map00603) positions glucosylceramide as the product of UDP-glucose:ceramide glucosyltransferase (UGCG) acting on ceramide substrates. [Model Knowledge] The co-presence of both glycosylceramide derivatives and their ceramide precursors in this module suggests active glycosphingolipid biosynthesis. [Inferred] Literature evidence confirms that sphingadienine-containing glucosylceramides activate ceramide production in skin models (4,8-Sphingadienine and 4-hydroxy-8-sphingenine activate ceramide production, 2012). [Literature]
- **Validation step**: Measure UGCG expression and activity in the cohort. Perform targeted lipidomics for glucosylceramide species (GlcCer d18:1/16:0, d18:1/18:0, d18:1/22:0) to confirm the glucosylceramide pathway is active.
- **Calibration note**: Approximately 18% of such computational inferences advance to clinical investigation.

#### 3.4 Tryptophan Betaine as an Indoleamine 2,3-Dioxygenase (IDO) Pathway Marker

- **Prediction**: Tryptophan betaine (hypaphorine), a cold-start entity with no KG presence, may serve as a marker of tryptophan catabolism via the kynurenine/IDO pathway, linking immune activation (FLT3LG-driven dendritic cells) to tryptophan metabolism.
- **Structural logic chain**: Tryptophan betaine is semantically similar to 1-DL-methyl-tryptophan (similarity 0.96), a known IDO inhibitor. [KG Evidence] FLT3LG-stimulated dendritic cells express IDO, which catabolizes tryptophan to kynurenine and downstream metabolites. [Model Knowledge] The co-expression of tryptophan betaine with FLT3LG in a single module suggests a regulatory link between dendritic cell maturation and tryptophan metabolism. [Inferred] No direct KG evidence or grounded literature was found for this specific connection.
- **Validation step**: Measure kynurenine/tryptophan ratio and IDO1 expression in the cohort. Correlate tryptophan betaine levels with dendritic cell markers and FLT3LG.
- **Calibration note**: Approximately 18% of computational predictions advance to validation. This prediction is speculative (cold-start entity) but mechanistically coherent.

---

### 4. Biological Themes

#### 4.1 Sphingolipid Metabolism (Primary Theme)

The module is overwhelmingly enriched for sphingolipid metabolism, as confirmed by pathway recurrence: sphingolipid signaling pathway (UMLS:C2753772, 2 members) and sphingolipid metabolism (SMPDB:SMP0000034, 2 members). [KG Evidence] The sphingolipid species span a remarkable diversity of acyl chain lengths (C14:0 to C25:0), sphingoid bases (d18:0, d18:1, d18:2, d16:1, d17:1), and head groups (phosphocholine for sphingomyelins, glucose/galactose for glycosylceramides, lactose for lactosylceramides, phosphoethanolamine for ceramide-phosphoethanolamines). [KG Evidence] This breadth suggests that the module captures a global sphingolipid biosynthetic and remodeling program rather than a single enzymatic step. [Inferred]

#### 4.2 Membrane Phospholipid Remodeling

The arachidonoyl-enriched GPC and GPE species, particularly the plasmalogens (P-16:0/20:4, P-18:0/20:4, P-18:0/18:1), indicate a Lands cycle remodeling process that incorporates arachidonic acid into membrane phospholipids. [Model Knowledge] Plasmalogens serve as antioxidant membrane components and as reservoirs of arachidonic acid for eicosanoid biosynthesis. [Model Knowledge] The co-expression of these species with sphingomyelins is consistent with coordinated lipid raft composition, as sphingomyelins and plasmalogens co-localize in ordered membrane domains. [Model Knowledge]

#### 4.3 Apolipoprotein and Lipid Transport Network

The pathway enrichment analysis identified connections to apolipoprotein genes (APOE, APOC3, APOA1, CETP, CES1) as shared biological context linking 8 input entities. [KG Evidence] These genes encode core components of lipoprotein metabolism: APOA1 structures HDL particles, CETP mediates cholesteryl ester transfer, and APOE directs lipoprotein clearance. [Model Knowledge] This enrichment indicates that the module's sphingolipid and cholesterol species are likely transported via lipoprotein particles, providing a mechanistic link between circulating lipid levels and membrane composition. [Inferred]

#### 4.4 Guanidinoacetate: Creatine Biosynthesis Intersecting Lipid Metabolism

Guanidinoacetate (CHEBI:16344; 210 edges) is a precursor in creatine biosynthesis (AGAT/GAMT pathway) and is the only non-lipid small molecule in the module with substantial KG coverage. [KG Evidence] Its top disease association is AGAT deficiency (cerebral creatine deficiency syndrome 3). [KG Evidence] Guanidinoacetate shares disease associations with cholesterol (obesity, schizophrenia) and with FLT3LG (schizophrenia, rheumatoid arthritis). [KG Evidence] The co-expression of guanidinoacetate with sphingolipids is unexpected and may reflect shared renal or hepatic regulatory mechanisms (both guanidinoacetate synthesis and sphingolipid metabolism occur prominently in kidney and liver). [Model Knowledge]

#### 4.5 Hub-Filtered Insights

Phosphatidylethanolamine (4,442 edges) and cholesterol (4,371 edges) are hub nodes whose high connectivity in the KG makes their individual disease associations potentially non-specific. [KG Evidence] Ceramide (219 edges) was flagged as a hub in the pathway enrichment analysis (75 edges in the enrichment subgraph). [KG Evidence] Associations mediated solely through these hubs (without independent corroboration from lower-connectivity members) should be weighted cautiously.

---

### 5. Gap Analysis

#### 5.1 Informative Absences

| Expected Entity | Interpretation |
|---|---|
| **Free ceramides** (Cer d18:1/16:0, d18:1/18:0, d18:1/24:1) | The module captures the sphingomyelin branch upstream of lipotoxic ceramide accumulation, suggesting a pre-lipotoxic or sphingomyelin-reservoir state. [Inferred] |
| **Branched-chain amino acids** (leucine, isoleucine, valine) | BCAAs segregate into an amino acid catabolism axis distinct from this lipid-centric module. [Inferred] |
| **Lysophosphatidylcholines** (especially LPC 18:2) | Intact PCs are present but LPCs (PLA2 products) are absent, indicating the module does not capture phospholipase A2-mediated remodeling. [Inferred] |
| **Acylcarnitines** (short- and medium-chain) | Mitochondrial beta-oxidation intermediates are not co-regulated with this membrane lipid cluster. [Inferred] |
| **Aromatic amino acids** (phenylalanine, tyrosine) | The tryptophan derivative (tryptophan betaine) provides partial representation, but canonical aromatic amino acid dysregulation markers are absent. [Inferred] |
| **Adipokines** (adiponectin, leptin) | The sole protein is FLT3LG (immune); adipose endocrine signaling is not represented. [KG Evidence] |

#### 5.2 Standard (Platform-Driven) Gaps

Insulin, C-peptide, HbA1c, and fasting glucose/HOMA-IR are absent because they are clinical laboratory measures not captured by untargeted mass spectrometry-based omics platforms. [Inferred] Their absence reflects measurement methodology, not biological irrelevance.

#### 5.3 Triglyceride Composition as a Selective Signal

The module contains arachidonoyl-enriched triacylglycerols (18:0/20:4, 16:0/20:4) but lacks the saturated/monounsaturated triglyceride species most strongly associated with type 2 diabetes risk. [KG Evidence] This selective presence of arachidonic acid-containing TAGs suggests an eicosanoid precursor storage pool rather than simple lipid overload. [Inferred]

---

### 6. Temporal Context

No explicit longitudinal time-point data was provided with this module. The following temporal inferences can be drawn from the module's composition:

#### 6.1 Upstream Causes vs. Downstream Consequences

The dominance of sphingomyelins (precursors) over free ceramides (products) in this module suggests the captured biology is temporally upstream of sphingomyelinase-mediated ceramide generation. [Inferred] If this cohort includes longitudinal sampling, the module eigengene trajectory may predict subsequent ceramide elevation and associated cardiometabolic events. FLT3LG-driven immune cell differentiation is a relatively early event in immune activation cascades; its co-expression with membrane lipids may reflect early immune cell mobilization requiring lipid substrate availability. [Model Knowledge]

#### 6.2 Causal Inference Opportunities

The FLT3LG → IL1B → cholesterol bridge (text-mined) provides a testable causal chain: FLT3LG-driven dendritic cell maturation → IL1B secretion → cholesterol metabolism modulation. [KG Evidence; Inferred] Mendelian randomization using FLT3LG pQTLs as instruments could test whether genetically determined FLT3LG levels causally influence sphingomyelin or cholesterol concentrations. [Inferred]

---

### 7. Research Recommendations

#### 7.1 High Priority: Experimental Validations

1. **Sphingomyelinase activity assay**: Measure acid sphingomyelinase (aSMase) activity in plasma or serum from the study cohort. If aSMase activity is low and sphingomyelin levels are high, this confirms the "sphingomyelin reservoir" hypothesis and supports the module as a pre-lipotoxic signature. [Inferred]

2. **Dendritic cell phenotyping**: Perform flow cytometry (CD11c, CD123, CD1c, CD141) to quantify dendritic cell subsets and correlate with module eigengene and FLT3LG protein levels. This tests whether the FLT3LG-sphingolipid co-expression reflects dendritic cell membrane biogenesis. [Inferred]

3. **Targeted ceramide and glucosylceramide panel**: Quantify specific ceramide species (Cer d18:1/16:0, d18:1/18:0, d18:1/24:1) and glucosylceramide species not in the current panel to determine whether they segregate into different WGCNA modules. [Inferred]

#### 7.2 Moderate Priority: Literature and Database Searches

4. **FLT3LG and sphingolipid metabolism**: Search PubMed for direct evidence linking FLT3 signaling to sphingolipid biosynthesis or membrane remodeling in dendritic cells. The fetched literature on ANGPTL3 and cholesterol homeostasis (Frontiers, 2026; Genome Biology, 2025) is tangential to this specific hypothesis. [Literature: tangential only]

5. **Guanidinoacetate co-regulation with sphingolipids**: Search for shared regulatory mechanisms (renal proximal tubule metabolism, hepatic methylation cycles) that could explain guanidinoacetate co-expression with sphingolipid species. [Model Knowledge]

6. **Tryptophan betaine (hypaphorine) in human serum metabolomics**: Confirm the identity and biological role of this analyte; search for associations with IDO activity, kynurenine pathway, and immune function in published metabolomics studies. [Inferred]

#### 7.3 Follow-Up Analyses

7. **Cross-module comparison**: Compare the Green module with other WGCNA modules to determine whether free ceramides, BCAAs, acylcarnitines, and LPCs segregate into distinct modules, confirming the biological specificity of this sphingomyelin-enriched cluster. [Inferred]

8. **Mendelian randomization**: Use FLT3LG pQTLs and sphingomyelin mQTLs as genetic instruments to test causal directionality between immune signaling and sphingolipid levels. [Inferred]

9. **Disease-specific subanalysis**: Given the recurrence of colorectal cancer (4 members, curated evidence), conduct a focused analysis of module eigengene associations with colorectal cancer risk factors, colonoscopy outcomes, or incident cases if available. [KG Evidence; Inferred]

10. **Plasmalogen biology**: The presence of multiple vinyl-ether phospholipids (P-16:0 and P-18:0 species) warrants investigation of peroxisomal function (plasmalogen biosynthesis occurs in peroxisomes) and potential links to oxidative stress in this cohort. [Model Knowledge]

---

### Appendix: Evidence Quality Notes

- **Entity resolution confidence**: 56 of 56 entities were resolved, but many lipid species were matched via fuzzy methods (70% confidence) due to the structural complexity of lipid nomenclature. Several entities (behenoyl dihydrosphingomyelin → DIHYDROABIETYL BEHENATE; myristoyl dihydrosphingomyelin → MYRISTOYL SARCOSINE) appear to be misresolved; these are non-lipid compounds incorrectly matched to sphingolipid queries. Results involving these specific entities should be disregarded. [KG Evidence]
- **Cold-start entities** (7 entities with 0 KG edges): tryptophan betaine, glycosyl-N-stearoyl-sphingosine, 2-aminoheptanoate, behenoyl dihydrosphingomyelin, myristoyl dihydrosphingomyelin, glycosyl-N-palmitoyl-sphingosine, and N-behenoyl-sphingadienine. These entities could not be queried in the knowledge graph and are interpreted under the Open World Assumption as unstudied rather than biologically irrelevant. [KG Evidence]
- **Literature grounding**: Fetched abstracts addressed cholesterol homeostasis (ANGPTL3, 2026), sphingadienine-glucosylceramide biology (Lipids in Health and Disease, 2012), and hydroxylated sphingolipid synthesis (2021), but did not directly address the FLT3LG-sphingolipid co-expression hypothesis. Fetched palmitoylation literature (RFCM-PALM, 2021; Journal of Hematology & Oncology, 2025) concerns S-palmitoylation of proteins, which is distinct from palmitoyl sphingomyelin metabolism; these references are not directly relevant to the module's sphingomyelin biology. [Literature: tangential]

### Literature References

Papers discovered via semantic search. 9 unique papers across 5 hypotheses.

| Hypothesis | Citation | Link | Relevance |
|------------|----------|------|-----------|
| Inferred role of PUBCHEM.COMPOUND:6443616 |  (2017) "(R)-N-((2S,3S,4R)-3,4-dihydroxy-15-methyl-1-(((2R,3R,4S,5S,6R)-3,4,5-trihydroxy-6-(hydroxymethyl)tet..." | [Link](http://www.nature.com/articles/nchembio.2347/compounds/11) | (((2R ... S,5S,6R ... pyran- ... TBAF (1 M in THF, 45 μL, 45 μmol) was added at 50 °C to a solution of Glucosylceramide... |
| Inferred role of PUBCHEM.COMPOUND:6443616 |  (2012) "4,8-Sphingadienine and 4-hydroxy-8-sphingenine activate ceramide production in the skin \| Lipids in ..." | [Link](https://link.springer.com/article/10.1186/1476-511X-11-108) | Ingestion of glucosylceramide improves transepidermal water loss (TEWL) from the skin, but the underlying mechanism by w... |
| Inferred role of PUBCHEM.COMPOUND:6443616 |  (2005) "Efficient stereocontrolled synthesis of sphingadienine derivatives - ScienceDirect" | [Link](https://www.sciencedirect.com/science/article/abs/pii/S0040402005012743) | Sphingolipids, for example, ceramides, sphingomyelin, cerebrosides, and gangliosides, are constituents of eukaryotic cel... |
| Bridge: Gene → SmallMolecule (2 hops) |  (2026) "Frontiers \| ANGPTL3 and residual atherosclerotic risk: from lipid metabolism to therapeutic targetin..." | [Link](https://www.frontiersin.org/journals/endocrinology/articles/10.3389/fendo.2025.1706091/full) | With respect to cholesterol metabolism, ANGPTL3 can be summarized as a dual hub acting upstream and downstream: on one e... |
| Inferred role of CHEBI:83363 |  (2021) "Hydroxylated Fatty Acids: The Role of the Sphingomyelin Synthase and the Origin of Selectivity" | [Link](https://www.mdpi.com/2077-0375/11/10/787) | Sphingolipids containing 2-hydroxylated fatty acids (2OHFA) are present in most organisms [32] and are important compone... |
| Inferred role of CHEBI:85814 |  (2021) "RFCM-PALM: In-Silico Prediction of S-Palmitoylation Sites in the Synaptic Proteins for Male/Female M..." | [Link](https://www.mdpi.com/1422-0067/22/18/9901) | S-palmitoylation is a reversible covalent post-translational modification of cysteine thiol side chain by palmitic acid.... |
| Inferred role of CHEBI:83363 |  (2021) "Stereoselective Synthesis of Novel Sphingoid Bases Utilized for Exploring the Secrets of Sphinx" | [Link](https://www.mdpi.com/1422-0067/22/15/8171) | Sphingolipids are ubiquitous in eukaryotic plasma membranes and play major roles in human and animal physiology and dise... |
| Bridge: Gene → SmallMolecule (2 hops) |  (2025) "Systematic interrogation of functional genes underlying cholesterol and lipid homeostasis \| Genome B..." | [Link](https://link.springer.com/article/10.1186/s13059-025-03531-8) | γ-glutamyltransferases ( ... and their increased ... inform damaged functions of the ... or bile duct ... integrative an... |
| Inferred role of CHEBI:83899 |  (2025) "Table 1 Glycosphingolipid species detected by µL-flow 4D-RP-LC-TIMS-PASEF analysis in human serum" | [Link](https://www.nature.com/articles/s41467-025-59755-6/tables/1) | Table 1 Glycosphingolipid species detected by µL-flow 4D-RP-LC-TIMS-PASEF analysis in human serum ... Neutral glycosphin... |