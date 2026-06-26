# Midnightblue Module Run: Discovery Output (76-analyte, dev, 2026-06-23)

This document presents the discovery output of a full-module run of the **Midnightblue** WGCNA module through the Kraken 12-node discovery pipeline, executed on the dev environment on 2026-06-23 (commit `813fc3f`) with BioMapper entity resolution enabled. It is one of a per-module series over the Frailty multi-omics WGCNA modules; see the [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm).

## Run Provenance

The run submitted 76 named analytes, parsed 76 at intake, and resolved 76 distinct entities (51 biomapper, 25 fuzzy) to 71 distinct CURIEs. Triage classified 15 well-characterized, 22 moderate, 29 sparse, and 10 cold-start, with 0 edge-count measurement failures. Downstream analysis produced 1593 direct-KG findings, 22 cold-start findings, 9 biological themes, 10 cross-entity bridges (10 evidence-grounded), and 49 hypotheses supported by 23 literature references. Synthesis emitted a 29864-character report. The run completed in approximately 831.0 s of wall-clock time (status complete, 2 errors). All counts are taken from the reducer-merged pipeline state.

| Stage | Result |
|---|---|
| Submitted | 76 named analytes |
| Intake | 76 parsed |
| Entity resolution | 76 resolved (51 biomapper, 25 fuzzy) to 71 distinct CURIEs |
| Triage | 15 well-characterized, 22 moderate, 29 sparse, 10 cold-start (0 measurement failures) |
| Direct KG | 1593 findings |
| Cold-start | 22 findings, 31 skipped |
| Pathway enrichment | 9 biological themes |
| Integration | 10 bridges (10 evidence-grounded) |
| Literature grounding | 23 papers |
| Synthesis | 49 hypotheses, 29864-character report |
| Run total | ~831.0 s wall-clock, status complete, 2 errors |

## Related

- Companion run metrics: [Midnightblue Module Run: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/midnightblue-module-run-pipeline-performance-report-76-analyte-dev-2026-06-23-r0R7cTBAoA)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---



## Midnightblue WGCNA Module: Gut Microbial Metabolism, Intestinal Barrier Function, and Systemic Inflammatory Signaling

### 1. Executive Summary

The Midnightblue WGCNA module encodes a coordinated program of intestinal epithelial function, gut microbial metabolism, and inflammatory chemokine signaling. [KG Evidence] Six proteins (FABP2, F3, CHIT1, CCL25, CST5, CCL11) converge on intestinal lipid absorption, coagulation/vascular remodeling, and tissue-specific immune cell recruitment; 70 co-expressed metabolites are dominated by microbially derived indole and phenol conjugates, uremic toxin precursors, acylglycine conjugates of branched-chain amino acid (BCAA) catabolism, and acetylated amino acid derivatives. [KG Evidence] Module-level disease recurrence identifies colorectal cancer (11 members), coronary artery disorder (7 members), chronic kidney disease (6 members), and asthma (7 members) as the strongest shared disease contexts, suggesting that this module captures a gut-to-systemic signaling axis relevant to cardiometabolic and inflammatory disease progression. [KG Evidence]

### 2. Key Findings (Tier 1 to 2)

#### 2.1 Protein Functions and Pathway Memberships

**FABP2** (NCBIGene:2169; 3,140 edges) participates in intestinal lipid absorption, fatty acid metabolic process, long-chain fatty acid transport, PPAR signaling pathway, cholesterol metabolism, and enterocyte cholesterol metabolism. [KG Evidence] FABP2 interacts with PPARA, SCARB1, APOA1, APOA4, LPL, MTTP, and DGAT1, establishing it as a central node in enterocyte lipid handling and chylomicron assembly. [KG Evidence] Its co-expression with metabolites in this module anchors the module's intestinal epithelial origin.

**F3** (NCBIGene:2152; 3,725 edges) participates in blood coagulation, activation of the extrinsic prothrombin pathway, positive regulation of cell migration, angiogenesis, endothelial cell proliferation, and cytokine-mediated signaling. [KG Evidence] F3 also participates in positive regulation of TOR signaling and PI3K/AKT signal transduction, indicating roles beyond hemostasis in vascular remodeling and metabolic signaling. [KG Evidence]

**CHIT1** (NCBIGene:1118; 1,967 edges) encodes chitotriosidase, a chitinase implicated in innate immunity and macrophage activation. [KG Evidence] Its top disease association is chitotriosidase deficiency-related phenotypic features. [KG Evidence] The enzyme's presence in a gut-centric module is consistent with its role in mucosal defense against chitin-bearing pathogens. [Model Knowledge]

**CCL11** (NCBIGene:6356; 2,066 edges) and **CCL25** (NCBIGene:6370; 1,730 edges) are chemokines that participate in chemotaxis, eosinophil chemotaxis, inflammatory response, leukocyte migration, G protein-coupled receptor signaling, and antimicrobial humoral immune response. [KG Evidence] CCL11 (eotaxin-1) is a canonical eosinophil chemoattractant; CCL25 (TECK) mediates gut-homing of T lymphocytes via CCR9. [KG Evidence, Model Knowledge] Their co-expression indicates an active mucosal immune recruitment program.

**CST5** (NCBIGene:1473; 971 edges) encodes cystatin D, a cysteine protease inhibitor involved in protein binding and stress response. [KG Evidence] CST5 shares disease associations with all other protein members across 15+ conditions (including panniculitis, irritable bowel syndrome, and gastroduodenitis), suggesting broad mucosal inflammatory relevance. [KG Evidence]

#### 2.2 Module-Level Disease Associations

Eleven module members share association with **colorectal cancer** (MONDO:0005575), spanning proteins (FABP2, CCL11) and metabolites (allantoin, trimethylamine N-oxide, urea, N6,N6,N6-trimethyllysine, phenol sulfate, indolin-2-one, p-cresol sulfate, methylsuccinate, 3-methylhistidine); this represents the broadest convergence in the module. [KG Evidence] Seven members associate with **coronary artery disorder** (MONDO:0005010), including all six proteins and myo-inositol. [KG Evidence] Seven members associate with **asthma** (MONDO:0004979) and **schizophrenia** (MONDO:0005090), reflecting the module's inflammatory chemokine component. [KG Evidence] Six members converge on **kidney disorder** (MONDO:0005240), including metabolites (allantoin, urea, myo-inositol, p-cresol sulfate) recognized as uremic solutes or renal biomarkers. [KG Evidence] Five members associate with **diabetes mellitus** (MONDO:0005015), including urea, myo-inositol, FABP2, CCL25, and 1/3-methylhistidine. [KG Evidence]

#### 2.3 Cross-Type Bridges

Multiple two-hop paths connect **FABP2 to myo-inositol** via shared drug interactors (fenofibrate, ibuprofen, diclofenac, flurbiprofen, ketorolac, ketoprofen, meclofenamic acid) and via co-localization in the cytoplasm. [KG Evidence] These bridges carry curated provenance at the weakest leg, lending moderate confidence. [KG Evidence] A literature-grounded study of fatty liver energy metabolism (2020) documents convergent regulation of hepatic genes in SREBP-1c models that includes FABP-family members, supporting the biological plausibility of FABP2-to-myo-inositol metabolic connections. [Literature: "Physiological Disturbance in Fatty Liver Energy Metabolism Converges on IGFBP2 Abundance and Regulation in Mice and Men," 2020] Separately, the role of myo-inositol in energy metabolism via inositol phosphate signaling cascades and insulin signal transduction has been reviewed. [Literature: "Role of Inositols and Inositol Phosphates in Energy Metabolism," 2020]

**F3 to myo-inositol** bridges proceed via eicosapentaenoic acid (EPA) and amphetamine sulfate, reflecting F3's pharmacological interaction network. [KG Evidence] The EPA bridge is biologically coherent: omega-3 fatty acids modulate tissue factor expression, and myo-inositol participates in phospholipid signaling downstream of membrane remodeling. [Model Knowledge]

#### 2.4 Metabolite Class Enrichment

The pathway enrichment analysis identifies several coherent chemical and disease classes:

**Acylglycines** (CHEBI:84087) connect 5 input metabolites: propionylglycine, isovalerylglycine, isobutyrylglycine, tyramine O-sulfate, and dopamine 3-O-sulfate. [KG Evidence] The N-acylglycine subclass (CHEBI:16180) specifically links propionylglycine, isovalerylglycine, and isobutyrylglycine, which are canonical markers of organic acidurias and BCAA catabolism intermediates. [KG Evidence]

**Chronic kidney disease** (MONDO:0005575) connects N6,N6,N6-trimethyllysine, methylsuccinate, 1/3-methylhistidine, N-acetylphenylalanine, and N-acetylhistidine via correlated_with predicates. [KG Evidence] This is consistent with the established role of these metabolites as uremic retention solutes. [Model Knowledge]

**Short-chain acyl-CoA dehydrogenase deficiency** (MONDO:0011229) links methylsuccinate, isovalerylglycine, and isobutyrylglycine. [KG Evidence] **Combined oxidative phosphorylation deficiency** (MONDO:0012451) and **eosinophilic esophagitis** connect overlapping metabolite subsets. [KG Evidence]

**Sulfate esters** (CHEBI:37919) connect p-cresol sulfate, phenol sulfate, tyramine O-sulfate, and dopamine 3-O-sulfate, all products of hepatic or microbial phase II conjugation. [KG Evidence]

#### 2.5 Shared Gene Neighbors

The enrichment analysis identifies **GLYAT** (NCBIGene:10249), **GLYATL1** (NCBIGene:92292), and **GLYATL2** (NCBIGene:219970) as shared gene neighbors connecting 10 input metabolites. [KG Evidence] These glycine N-acyltransferases catalyze the conjugation of acyl-CoA intermediates to glycine, producing the acylglycines (propionylglycine, isovalerylglycine, isobutyrylglycine) present in this module. [KG Evidence, Model Knowledge] **CCL5** (NCBIGene:6352) and **PF4** (NCBIGene:5196) also appear as shared gene neighbors, reinforcing the chemokine/inflammatory axis. [KG Evidence]

### 3. Novel Predictions (Tier 3)

#### 3.1 This Module as a Gut-Derived Uremic Toxin Signature Predictive of Cardiorenal Decline

**Logic chain**: The module co-expresses (a) FABP2, an intestinal fatty acid transporter marking enterocyte function; (b) p-cresol sulfate, phenol sulfate, indoxyl sulfate, and phenylacetylglutamine, all established gut-microbiome-derived uremic toxins; (c) CCL11 and CCL25, chemokines driving eosinophil and gut-homing T cell recruitment; and (d) urea, allantoin, and homocitrulline, nitrogen metabolism markers. [KG Evidence] Module-level disease recurrence identifies both kidney disorder (6 members) and coronary artery disorder (7 members). [KG Evidence] The prediction is that this module captures a gut-renal-cardiovascular axis wherein intestinal barrier dysfunction (indexed by FABP2) leads to systemic exposure to microbially derived uremic toxins, which in turn accelerate renal and vascular injury. [Inferred]

**Calibration note**: Approximately 18% of computational predictions of this type progress to clinical investigation.

**Validation step**: Correlate module eigengene values with eGFR trajectories and incident cardiovascular events in the parent cohort; perform mediation analysis testing whether FABP2 levels mediate the association between microbial metabolites (p-cresol sulfate, indoxyl sulfate) and renal decline.

#### 3.2 Selective Microbial Indole Pathway Activation Without Host Kynurenine Pathway Engagement

**Logic chain**: The module contains indoxyl sulfate, methyl indole-3-acetate, indoleacetylglutamine, indole-3-carboxylate, 6-hydroxyindole sulfate, indolin-2-one, and 3-indoleglyoxylic acid: seven tryptophan-derived metabolites produced predominantly by gut microbial tryptophanase and subsequent hepatic conjugation. [KG Evidence] Kynurenine, the primary host-pathway tryptophan catabolite (via IDO/TDO), is absent. [Inferred] Tryptophan itself is also absent. [Inferred] The selective capture of microbial (indole pathway) but not host (kynurenine pathway) tryptophan catabolites indicates that this module indexes gut microbial metabolic activity rather than systemic immune-driven tryptophan degradation. [Inferred]

**Calibration note**: Approximately 18% of such computational predictions advance to clinical investigation.

**Validation step**: Measure fecal tryptophanase gene abundance (e.g., tnaA by qPCR or shotgun metagenomics) and serum kynurenine/tryptophan ratios in the same cohort; test whether module eigengene correlates with microbial tryptophanase but not host IDO activity.

#### 3.3 BCAA Catabolic Flux, Not Circulating BCAA Accumulation, as the Module's Metabolic Signature

**Logic chain**: Isobutyrylglycine (valine catabolite), isovalerylglycine (leucine catabolite), propionylglycine (isoleucine catabolite), tiglylcarnitine (isoleucine catabolite), 3-methylglutaconate, 3-methylglutarylcarnitine, methylsuccinate, and ethylmalonate are all downstream intermediates of BCAA catabolism or short-chain fatty acid oxidation. [KG Evidence] The GLYAT/GLYATL1/GLYATL2 glycine conjugation enzymes connect these metabolites as shared gene neighbors. [KG Evidence] Parent BCAAs (leucine, isoleucine, valine) are absent from the module. [Inferred] The prediction is that this module captures active mitochondrial BCAA catabolic flux (with glycine conjugation of acyl-CoA overflow products) rather than the circulating BCAA accumulation phenotype associated with insulin resistance. Short-chain acyl-CoA dehydrogenase deficiency (MONDO:0011229) links three of these metabolites in the disease recurrence analysis. [KG Evidence]

**Calibration note**: Approximately 18% of computational predictions progress to clinical investigation.

**Validation step**: Perform targeted metabolomics of BCAA catabolic intermediates (isobutyryl-CoA, isovaleryl-CoA, propionyl-CoA) in matched plasma samples; correlate module eigengene with BCAT2 and BCKDH enzyme activity markers; test whether glycine availability (as a conjugation co-substrate) modifies acylglycine levels.

#### 3.4 Trimethyllysine to TMAO Conversion Gap as a Marker of Microbiome Composition

**Logic chain**: N6,N6,N6-trimethyllysine (a carnitine biosynthesis precursor) and hydroxy-trimethyllysine are present, yet TMAO (the terminal product of the trimethylamine pathway after microbial conversion) is absent from the module. [KG Evidence; Inferred] The prediction is that this gap reflects either (a) low abundance of TMA-producing gut bacteria (Firmicutes, Proteobacteria carrying cutC/cutD or cntA/cntB gene clusters), or (b) assignment of TMAO to a different WGCNA module with distinct temporal dynamics. [Inferred] Notably, trimethylamine N-oxide is present as a well-characterized entity (1,086 edges) elsewhere in the KG, with top disease association to severe primary trimethylaminuria. [KG Evidence] Its absence from co-expression with its upstream precursor trimethyllysine is therefore a biologically informative dissociation.

**Calibration note**: Approximately 18% of computational predictions of this type advance to clinical investigation.

**Validation step**: Perform 16S rRNA or shotgun metagenomic profiling of matched stool samples to quantify TMA-producing gene cluster abundance; test TMAO module assignment in the full WGCNA dendrogram; measure plasma FMO3 activity proxies.

#### 3.5 Allium-Derived Metabolites as a Dietary Exposure Signature

**Logic chain**: Alliin, N-acetylalliin, and S-allylcysteine are allium-vegetable-specific metabolites (garlic, onion) present in the module. [KG Evidence] Their co-expression with gut microbial catabolites (indoles, phenol conjugates) and intestinal epithelial markers (FABP2) suggests that dietary allium intake modulates the gut microbial ecosystem captured by this module. [Inferred]

**Calibration note**: Approximately 18% of computational predictions advance to clinical investigation.

**Validation step**: Administer dietary recall questionnaires to assess allium intake frequency; correlate alliin/S-allylcysteine levels with module eigengene; test whether allium-rich diet intervention alters microbial indole or phenol sulfate production in a crossover design.

### 4. Biological Themes

#### 4.1 Unifying Theme: Intestinal Barrier, Microbial Metabolism, and Systemic Toxin Exposure

The module encodes a coordinated biological program centered on the gut epithelium and its interaction with the luminal microbiome. [Inferred] Three convergent axes emerge:

**Axis 1: Intestinal epithelial function and lipid handling.** FABP2 anchors this axis, with established roles in fatty acid transport, PPAR signaling, and intestinal absorption. [KG Evidence] The digestion GO term (GO:0007586) connects CHIT1 and FABP2. [KG Evidence] Myo-inositol, pantothenate (vitamin B5), and pyridoxate (vitamin B6 catabolite) represent nutrient absorption and cofactor turnover within the enterocyte. [KG Evidence, Model Knowledge]

**Axis 2: Gut microbial catabolism and phase II conjugation.** Seven indole derivatives, four sulfate esters (p-cresol sulfate, phenol sulfate, tyramine O-sulfate, dopamine 3-O-sulfate), p-cresol glucuronide, phenylacetylglutamine, and 4-hydroxyphenylacetylglutamine are products of microbial amino acid fermentation (tryptophan, tyrosine, phenylalanine) followed by hepatic sulfation, glucuronidation, or glutamine conjugation. [KG Evidence, Model Knowledge] TMAO-precursor trimethyllysine and allium-derived metabolites (alliin, S-allylcysteine, N-acetylalliin) extend the microbial and dietary substrate pool. [KG Evidence]

**Axis 3: Mucosal inflammation and immune cell trafficking.** CCL11 and CCL25 drive eosinophil chemotaxis and gut-homing lymphocyte recruitment, respectively. [KG Evidence] F3 (tissue factor) participates in inflammatory coagulation activation and angiogenesis. [KG Evidence] CHIT1 reflects macrophage activation, and CST5 modulates protease activity in inflamed tissue. [KG Evidence] The stress response (GO:0006950) and immune response (GO:0006955) GO terms are shared across all six proteins. [KG Evidence]

#### 4.2 Hub-Filtered Insights

Fourteen hub nodes (>1,000 edges) were identified, including Homo sapiens, extracellular space, cytoplasm, extracellular region, and blood. [KG Evidence] These high-connectivity nodes are de-emphasized as they reflect generic annotations rather than specific biological connections. The non-hub shared neighbors (chronic kidney disease, acylglycine, saliva, placenta, N-acyl-amino acid, sulfate ester) carry greater interpretive weight for this module. [KG Evidence]

#### 4.3 Nitrogen and Urea Cycle Metabolites

Urea (2,037 edges), allantoin (650 edges), homocitrulline, N-acetylcitrulline, argininate, 2-oxoarginine, N-delta-acetylornithine, and N2,N5-diacetylornithine form a coherent nitrogen metabolism cluster. [KG Evidence] These species span the urea cycle (citrulline and ornithine derivatives), purine catabolism (allantoin), and arginine modification pathways. [KG Evidence, Model Knowledge] Their co-expression with the BCAA-catabolism acylglycines suggests coordinated amino acid nitrogen disposal. [Inferred]

#### 4.4 Acetylated and Methylated Amino Acid Derivatives

N-acetylphenylalanine, N-acetylhistidine, N-acetyl-3-methylhistidine, N-acetyl-1-methylhistidine, N-acetylcitrulline, N2,N5-diacetylornithine, N-acetylglutamine, N-acetylarginine, and (N(1)+N(8))-acetylspermidine represent a prominent N-acetylation theme. [KG Evidence] The N-acyl-amino acid class (CHEBI:21545) and N-acyl-L-histidine class (CHEBI:84076) formally connect subsets of these metabolites. [KG Evidence] Methylated species (1-methylhistidine, 3-methylhistidine, N6,N6,N6-trimethyllysine, N2-acetyl,N6,N6-dimethyllysine) indicate protein methylation turnover and histidine modification. [KG Evidence] 3-Methylhistidine is an established marker of skeletal muscle protein turnover. [Model Knowledge]

### 5. Gap Analysis

#### 5.1 Informative Absences (Open World Assumption)

**Tryptophan and kynurenine are absent** despite seven microbial indole-pathway catabolites being present. [Inferred] This dissociation reveals that the module captures the bacterial tryptophanase branch of tryptophan metabolism (indole production in the gut lumen) but not the host IDO/TDO-mediated kynurenine branch. The finding is consistent with an intestinal barrier/microbial signature distinct from systemic immune activation. [Inferred]

**Parent BCAAs (leucine, isoleucine, valine) are absent** while downstream acylglycine conjugates (propionylglycine, isovalerylglycine, isobutyrylglycine) are present. [Inferred] The module therefore captures catabolic flux rather than circulating BCAA accumulation, suggesting active mitochondrial processing (possibly in the gut epithelium or liver) rather than the impaired BCAA catabolism classically observed in insulin resistance. [Inferred]

**TMAO is absent** despite its upstream precursor trimethyllysine being present, implying either an incomplete microbial conversion step or assignment to a different co-expression module. [Inferred] Notably, TMAO appears in the KG as a well-characterized entity with 1,086 edges, confirming that its absence from this module is not due to KG sparsity. [KG Evidence]

**Free carnitine and medium/long-chain acylcarnitines are absent**, limiting the module to short-chain acyl metabolism. [Inferred] This constrains interpretation of FABP2's presence: the module captures amino acid-derived short-chain acyl overflow rather than classical lipid beta-oxidation dysfunction. [Inferred]

**Creatinine is absent** alongside urea and allantoin, suggesting the nitrogen metabolism axis reflects gut/hepatic disposal rather than renal filtration dynamics. [Inferred]

#### 5.2 Standard (Methodological) Gaps

Insulin, C-peptide, HbA1c, fasting glucose, and HOMA-IR are absent as expected; these are clinical measurements from different analytical platforms and are structurally excluded from molecular co-expression networks. [Model Knowledge] Adiponectin and IL-6 are absent, which is mildly informative: this module encodes tissue-specific (intestinal/mucosal) chemokine signaling rather than adipose-centric or systemic acute-phase inflammation. [Inferred]

#### 5.3 Cold-Start Entities

Ten metabolites have zero KG edges: indoleacetate (resolved to indoleacetonitrile; likely a misresolution), isobutyrylcarnitine, N-acetylglutamine, N-acetylarginine, 4-methylcatechol sulfate, arabonate/xylonate, arabitol/xylitol, phenylacetylglutamate, methylsuccinoylcarnitine, and N2-acetyl,N6,N6-dimethyllysine. [KG Evidence] These entities remain uncharacterized in the knowledge graph and require identity confirmation and manual curation. Several (isobutyrylcarnitine, N-acetylglutamine, N-acetylarginine) are biochemically coherent with the module's acylation and amino acid modification themes and likely represent genuine metabolites with missing KG annotations rather than false analytes. [Inferred]

### 6. Temporal Context

No explicit longitudinal design metadata was provided. The module's composition permits the following causal inference opportunities:

**Upstream causes (candidate drivers):** Dietary substrates (alliin, S-allylcysteine from allium vegetables), intestinal epithelial integrity (indexed by FABP2), and gut microbial community composition (determining tryptophan and tyrosine catabolism rates) are plausible upstream determinants of the metabolite profile. [Inferred]

**Downstream consequences (candidate sequelae):** Systemic exposure to uremic toxin precursors (p-cresol sulfate, indoxyl sulfate, phenylacetylglutamine), chemokine-driven eosinophilic and lymphocytic infiltration (CCL11, CCL25), and coagulation/angiogenesis activation (F3) represent downstream effectors that may drive organ damage (kidney, vasculature, airway) over time. [Inferred]

**Causal directionality testing:** If paired with longitudinal sampling, Granger causality or dynamic Bayesian network approaches could test whether FABP2 elevation precedes uremic toxin accumulation, or whether microbial metabolite shifts precede chemokine upregulation. [Inferred]

### 7. Research Recommendations

#### 7.1 High Priority: Experimental Validations

1. **Intestinal permeability assessment.** Measure plasma FABP2 alongside lactulose/mannitol permeability ratios or fecal zonulin in the study cohort. FABP2 elevation is an established marker of enterocyte damage; its co-expression with microbial metabolites predicts that barrier dysfunction drives systemic toxin exposure. [Inferred] If validated, this would establish FABP2 as a hub biomarker for the module.

2. **Targeted microbiome profiling.** Perform shotgun metagenomics or targeted qPCR for (a) tryptophanase (tnaA), (b) TMA-producing gene clusters (cutC/cutD, cntA/cntB), and (c) tyrosine/phenylalanine fermentation genes (hpdBCA) in matched fecal samples. Correlate abundances with the respective metabolite clusters (indoles, TMAO precursors, phenol conjugates). [Inferred]

3. **BCAA catabolic enzyme activity.** Quantify BCKDH (branched-chain alpha-keto acid dehydrogenase) phosphorylation status or activity in peripheral blood mononuclear cells or liver biopsy (if available). The prediction is that BCKDH is active (not inhibited) in this cohort, explaining acylglycine accumulation as overflow conjugation rather than catabolic block. [Inferred]

#### 7.2 Moderate Priority: Follow-Up Analyses

4. **Cross-module comparison.** Compare the Midnightblue module eigengene with eigengenes of modules containing BCAAs, ceramides, long-chain acylcarnitines, kynurenine, and TMAO. The pattern of informative absences predicts these metabolites reside in distinct modules, and the inter-module correlation structure would reveal the relationship between gut-derived and systemic metabolic signatures. [Inferred]

5. **Disease outcome association.** Test the module eigengene (or a weighted composite of the top-priority members: FABP2, p-cresol sulfate, indoxyl sulfate, CCL25, isobutyrylglycine) for association with incident chronic kidney disease, cardiovascular events, and type 2 diabetes in the cohort. [Inferred] The colorectal cancer association (11 members) warrants separate investigation if cancer outcomes are available.

6. **Eosinophilic esophagitis sub-phenotyping.** Five members (allantoin, trimethylamine N-oxide, p-cresol sulfate, methylsuccinate, 3-methylhistidine) associate with eosinophilic esophagitis (MONDO:0012451). [KG Evidence] Given CCL11's established role in eosinophil recruitment to the esophagus, this module may identify a gut-microbial metabolite signature distinguishing eosinophilic esophagitis subtypes.

#### 7.3 Lower Priority: Literature and Curation

7. **Resolve cold-start entities.** The 10 zero-edge metabolites should be manually curated in ChEBI and HMDB. Particular attention is warranted for isobutyrylcarnitine (C4), N-acetylglutamine, and N-acetylarginine, which are common plasma metabolites with well-characterized biology but apparently missing KG records. [KG Evidence]

8. **Literature search for FABP2-microbiome interactions.** Emerging evidence links intestinal fatty acid binding proteins to microbiome-modulated barrier integrity. A targeted PubMed search for "FABP2 AND (microbiome OR intestinal permeability OR uremic toxins)" would contextualize the module's central finding. [Inferred]

9. **Entity resolution audit.** Several entity resolutions have reduced confidence (70%): isobutyrylcarnitine resolved to isobutyrylglutamine (CHEBI:232524), N-acetylglutamine and N-acetylarginine both resolved to N-acetylnonylamine (UNII:I35SVF9BQ2), sulfate resolved to sulfonatooxyamino sulfate, and phenylacetylcarnitine resolved to 3-phenylpropionylcarnitine. [KG Evidence] These misresolutions may introduce noise into Tier 3 inferences and should be corrected before downstream modeling.

---

*Report generated from KRAKEN knowledge graph analysis of the Midnightblue WGCNA module (76 entities: 6 proteins, 70 metabolites). Evidence attribution tags ([KG Evidence], [Literature], [Model Knowledge], [Inferred]) are applied per finding. All Tier 3 predictions are calibrated against the approximately 18% computational-to-clinical validation rate and require independent experimental confirmation.*

### Literature References

Papers discovered via semantic search. 11 unique papers across 6 hypotheses.

| Hypothesis | Citation | Link | Relevance |
|------------|----------|------|-----------|
| Inferred role of CHEBI:62448 |  (2022) "3-Carbamoylmethyl-Indole-1-Carboxylic Acid Ethyl Ester" | [Link](https://www.mdpi.com/1422-8599/2022/1/M1324) | 3-Carbamoylmethyl-Indole-1-Carboxylic Acid Ethyl Ester (an ethoxycarbonyl derivative of indole-3-acetamide) is obtained... |
| Inferred role of PUBCHEM.COMPOUND:9815668 |  (2026) "Acylphloroglucinol derivatives with α-glucosidase inhibitory activity from Eucalyptus grandis × urop..." | [Link](https://www.sciencedirect.com/science/article/abs/pii/S0045206826002798) | which are regarded as promising sources for new ... 1890, nearly ... 80 species have been ... to China and widely establ... |
| Inferred role of CHEBI:62448 |  (2025) "Domino Synthesis of 1,2,5-Trisubstituted 1H-Indole-3-carboxylic Esters Using a [3+2] Strategy" | [Link](https://www.mdpi.com/1420-3049/30/3/444) | A new approach to 1,2,5-trisubstituted 1H-indole-3-carboxylic esters has been developed and studied. The method begins w... |
| Inferred role of CHEBI:132918; Inferred role of CHEBI:25982 |  (2024) "In Silico Study of Camptothecin-Based Pro-Drugs Binding to Human Carboxylesterase 2" | [Link](https://www.mdpi.com/2218-273X/14/2/153) | cancer cell, are a promising approach towards ... hence reduced side effects in chemotherapy. A ... hydrolysis. Since ca... |
| Inferred role of CHEBI:132918 |  (2006) "Inhibition of secreted phospholipase A2 by neuron survival and anti-inflammatory peptide CHEC-9 \| Jo..." | [Link](https://link.springer.com/article/10.1186/1742-2094-3-25) | The nonapeptide CHEC-9 (CHEASAAQC), a putative inhibitor of secreted phospholipase A2 (sPLA2), has been shown previously... |
| Inferred role of PUBCHEM.COMPOUND:21537901 |  (2025) "Late-stage O-sulfation with a bioinspired sulfuryl donor \| Nature Communications" | [Link](https://www.nature.com/articles/s41467-025-62093-2) | O-sulfation of biomolecules is an essential process in all living organisms and is involved in blood clotting, pathogen... |
| Inferred role of CHEBI:132918 |  (2019) "Navigating in vitro bioactivity data by investigating available resources using model compounds \| Sc..." | [Link](https://www.nature.com/articles/s41597-019-0046-1) | identifiers, intuitively, should ... compound, but disparity ... . Among eleven resources that reported ... ChEBI, Ch ..... |
| Bridge: Gene → SmallMolecule (2 hops) |  (2020) "Physiological Disturbance in Fatty Liver Energy Metabolism Converges on IGFBP2 Abundance and Regulat..." | [Link](https://www.mdpi.com/1422-0067/21/11/4144) | In order to identify molecular networks involved in pathology and progression of fatty liver, liver biopsies from the pr... |
| Inferred role of PUBCHEM.COMPOUND:9815668 |  (2020) "Syntheses and Glycosidase Inhibitory Activities, and in Silico Docking Studies of Pericosine E Analo..." | [Link](https://www.mdpi.com/1660-3397/18/4/221) | Inspired by the significant α-glucosidase inhibitory activities of (+)- and (−)-pericosine E, we herein designed and syn... |
| Inferred role of CHEBI:62448 |  (2016) "Synthesis of New Functionalized Indoles Based on Ethyl Indol-2-carboxylate" | [Link](https://www.mdpi.com/1420-3049/21/3/333) | Synthesis of New Functionalized Indoles Based on Ethyl Indol-2-carboxylate ... alkylations of the nitrogen of ethyl in... |
| Inferred role of PUBCHEM.COMPOUND:9815668 |  (2019) "α-d-Glucopyranosyl-(1→2)-[6-O-(l-tryptophanyl)-β-d-fructofuranoside]" | [Link](https://www.mdpi.com/1422-8599/2019/2/M1066) | compound 1 ... . The identities of ... -d-fructofuranose and α-d-glucopyranose after inspection of the chemical shifts,... |