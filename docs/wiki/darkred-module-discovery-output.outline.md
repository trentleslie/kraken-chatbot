# Darkred Module Run: Discovery Output (26-analyte, dev, 2026-06-23)

This document presents the discovery output of a full-module run of the **Darkred** WGCNA module through the Kraken 12-node discovery pipeline, executed on the dev environment on 2026-06-23 (commit `813fc3f`) with BioMapper entity resolution enabled. It is one of a per-module series over the Frailty multi-omics WGCNA modules; see the [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm).

## Run Provenance

The run submitted 26 named analytes, parsed 26 at intake, and resolved 26 distinct entities (22 biomapper, 1 exact, 3 fuzzy) to 25 distinct CURIEs. Triage classified 5 well-characterized, 11 moderate, 8 sparse, and 2 cold-start, with 0 edge-count measurement failures. Downstream analysis produced 644 direct-KG findings, 31 cold-start findings, 0 biological themes, 16 cross-entity bridges (13 evidence-grounded), and 76 hypotheses supported by 17 literature references. Synthesis emitted a 27023-character report. The run completed in approximately 843.6 s of wall-clock time (status complete, 44 errors). All counts are taken from the reducer-merged pipeline state.

| Stage | Result |
|---|---|
| Submitted | 26 named analytes |
| Intake | 26 parsed |
| Entity resolution | 26 resolved (22 biomapper, 1 exact, 3 fuzzy) to 25 distinct CURIEs |
| Triage | 5 well-characterized, 11 moderate, 8 sparse, 2 cold-start (0 measurement failures) |
| Direct KG | 644 findings |
| Cold-start | 31 findings, 3 skipped |
| Pathway enrichment | 0 biological themes |
| Integration | 16 bridges (13 evidence-grounded) |
| Literature grounding | 17 papers |
| Synthesis | 76 hypotheses, 27023-character report |
| Run total | ~843.6 s wall-clock, status complete, 44 errors |

## Related

- Companion run metrics: [Darkred Module Run: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/darkred-module-run-pipeline-performance-report-26-analyte-dev-2026-06-23-rTsggJIalU)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---



## Darkred WGCNA Module: Bile Acid Enterohepatic Signaling Network

### 1. Executive Summary

The Darkred WGCNA module encodes a tightly coordinated bile acid enterohepatic signaling program, uniting 24 conjugated and sulfated bile acid species with two protein markers of intestinal epithelial identity (EPCAM) and postprandial endocrine signaling (FGF19). [KG Evidence] [Inferred] This module captures the ileal output arm of the FXR→FGF19 axis: bile acids reabsorbed from the intestinal lumen activate FXR in EPCAM-positive enterocytes, triggering FGF19 secretion into portal blood, while the conjugated bile acid pool composition reflects intact hepatic conjugation machinery and active phase II sulfation detoxification. [Inferred] [Literature: "FXR-FGF19 signaling in the gut–liver axis," Hepatology International, 2024] The convergence of hepatocellular carcinoma (6 members), colorectal cancer (5 members), and bile acid biosynthesis disorder pathways (5 members each) across module members establishes hepatobiliary and gastrointestinal malignancy as the dominant disease axis, with cholestasis and drug-induced liver injury as mechanistically coherent secondary associations. [KG Evidence]

### 2. Key Findings (Tier 1 to 2)

#### 2.1 Unifying Biological Theme: Postprandial Bile Acid Enterohepatic Circulation

The module's composition reveals a single dominant biological process. Twenty-four of 26 members are bile acid species spanning the full conjugation and modification landscape: primary bile acids (cholate, chenodeoxycholate conjugates), secondary bile acids (deoxycholate and lithocholate conjugates), glycine and taurine conjugates, 3-sulfated derivatives, glucuronidated forms, and the unusual muricholate and hyocholate species. [KG Evidence] The two protein members, FGF19 and EPCAM, anchor this metabolite network to a defined tissue context.

**FGF19** participates in negative regulation of bile acid biosynthetic process and fibroblast growth factor receptor signaling pathway (via FGFR4/KLB). [KG Evidence] It interacts directly with FGFR4, FGFR1, KLB (β-Klotho), CYP7A1, and ABCB11. [KG Evidence] These interactions recapitulate the canonical postprandial signaling cascade: ileal bile acid reabsorption → FXR activation → FGF19 secretion → hepatic FGFR4/KLB engagement → CYP7A1 suppression. [Model Knowledge] The literature confirms that FXR-FGF19 signaling in the gut-liver axis modulates bile acid synthesis, enterohepatic circulation, hepatic inflammation, and intestinal defence. [Literature: "FXR-FGF19 signaling in the gut–liver axis," Hepatology International, 2024]

**EPCAM** participates in cell-cell adhesion, positive regulation of cell population proliferation, positive regulation of stem cell proliferation, and signal transduction involved in regulation of gene expression. [KG Evidence] EPCAM marks the intestinal epithelial compartment and interacts with CDH1, CLDN7, and CFTR, proteins defining the tight junction and adhesion apparatus of absorptive enterocytes. [KG Evidence] Notably, EPCAM interacts with FGFR4 in the knowledge graph, establishing a direct molecular link between the two protein members of this module. [KG Evidence]

#### 2.2 Module-Level Disease Recurrence

The recurrence analysis identifies diseases shared across multiple module members, prioritized by member count and evidence strength:

| Disease | Members | Evidence | Key Members |
|---|---|---|---|
| Hepatocellular carcinoma | 6 | Curated | EPCAM, FGF19, taurochenodeoxycholate, glycodeoxycholate, glycocholate, taurodeoxycholate |
| Colorectal cancer | 5 | Curated | taurochenodeoxycholate, glutarate, glycodeoxycholate, glycocholate, taurodeoxycholate |
| Liver cancer | 4 | Curated | EPCAM, taurochenodeoxycholate, glycodeoxycholate, glycocholate |
| Drug-induced liver injury | 4 | Curated | taurochenodeoxycholate, glycodeoxycholate, glycocholate, taurodeoxycholate |
| Cholestasis | 2 | Curated | glycodeoxycholate, taurodeoxycholate |

[KG Evidence]

Hepatocellular carcinoma emerges as the strongest disease signal, supported by 6 of 26 members including both proteins. This association is mechanistically coherent: FGF19 gene amplification is a recognized oncogenic driver in hepatocellular carcinoma (the top disease for FGF19 in the knowledge graph is "FGF19 Gene Amplification"), and EPCAM marks hepatic cancer stem cells. [KG Evidence] [Model Knowledge] The bile acid species associated with hepatocellular carcinoma (taurochenodeoxycholate, glycodeoxycholate, glycocholate, taurodeoxycholate) are established circulating biomarkers of hepatobiliary dysfunction. [Model Knowledge]

Colorectal cancer association (5 members, curated evidence) is likewise biologically expected: EPCAM is genetically associated with Lynch syndrome (hereditary nonpolyposis colorectal cancer), and bile acids (particularly secondary bile acids such as deoxycholate) are recognized promoters of colorectal carcinogenesis. [KG Evidence] [Model Knowledge]

Drug-induced liver injury (4 members, curated evidence) reflects the known sensitivity of bile acid homeostasis to hepatotoxic insults; elevated serum bile acids constitute a sensitive clinical marker of hepatocellular damage. [KG Evidence] [Model Knowledge]

#### 2.3 Module-Level Pathway Recurrence

Seven bile acid biosynthesis and inborn-error pathways each recruit 5 module members (taurochenodeoxycholate, glycodeoxycholate, glycocholate, glycolithocholate, taurodeoxycholate):

- Bile Acid Biosynthesis (SMPDB:SMP0000035)
- Zellweger Syndrome (SMPDB:SMP0000316)
- Familial Hypercholanemia (SMPDB:SMP0000317)
- Congenital Bile Acid Synthesis Defect Types II and III
- Cerebrotendinous Xanthomatosis
- 27-Hydroxylase Deficiency

[KG Evidence]

The two proteins share Gene Ontology annotations for positive regulation of cell population proliferation (GO:0008284), protein binding (GO:0005515), and response to stress (GO:0006950). [KG Evidence] This convergence indicates that EPCAM and FGF19, despite distinct primary functions (adhesion vs. endocrine signaling), operate within a shared proliferative and stress-responsive epithelial program.

#### 2.4 Cross-Type Bridges

Multiple two-hop knowledge graph paths connect FGF19 to cholate through biologically interpretable intermediaries:

- FGF19 → cholic acid → cholate (interacts_with → related_to) [KG Evidence]
- FGF19 → ABCB11 → cholate (interacts_with → affects): ABCB11 (bile salt export pump) transports conjugated bile acids [KG Evidence]
- FGF19 → SLC10A1 → cholate (physically_interacts_with → affects): SLC10A1 (NTCP) mediates hepatic bile acid uptake [KG Evidence]
- FGF19 → SLC10A2 → cholate (affects → physically_interacts_with): SLC10A2 (ASBT) mediates ileal bile acid reabsorption [KG Evidence]
- FGF19 → CYP8B1 → cholate (interacts_with → physically_interacts_with): CYP8B1 determines the cholic acid:chenodeoxycholic acid ratio [KG Evidence]

These bridges are independently supported by literature demonstrating that FXR signaling modulates bile acid synthesis and enterohepatic circulation, with FGF19 as the ileal effector. [Literature: "FXR-FGF19 signaling in the gut–liver axis," Hepatology International, 2024]

EPCAM connects to cholate via a disease-mediated bridge (EPCAM → Liver Cirrhosis, Experimental → cholate) and through colorectal cancer (EPCAM → colorectal cancer → glycocholic acid → cholate). [KG Evidence] The cholate-to-glycochenodeoxycholate glucuronide bridges traverse gut microbial taxa (Alistipes, Romboutsia, Streptococcus, Haemophilus, Butyricicoccus, Thomasclavelia), reflecting microbiome-mediated correlations between primary bile acids and their conjugated/glucuronidated derivatives. [KG Evidence] [Literature: "Microbial transformations of human bile acids," Microbiome, 2021]

#### 2.5 Member Prioritization: Highest-Leverage Entities

The Member Prioritization Table identifies three tiers of analytical leverage:

**Highest priority (well-characterized, >200 edges):** EPCAM (3,355 edges), FGF19 (1,445 edges), taurochenodeoxycholate (543 edges), glutarate (332 edges), and glycocholate (253 edges) anchor the module with extensive knowledge graph connectivity and disease annotations. [KG Evidence]

**Moderate priority (20 to 199 edges):** Taurodeoxycholate (187 edges; top disease: hepatocellular carcinoma) and glycodeoxycholate (95 edges; top disease: hepatocellular carcinoma) are the most informative secondary bile acid conjugates. [KG Evidence] Glycocholenate sulfate (76 edges; top disease: atrial fibrillation) is notable for its unexpected cardiac association, warranting further investigation.

**Low priority (sparse or cold-start):** Eleven entities possess fewer than 20 edges; 2 entities (taurodeoxycholic acid 3-sulfate, glyco-beta-muricholate) have zero knowledge graph presence. [KG Evidence] These represent measurement-platform-specific analytes whose biological context is currently underrepresented in curated databases.

### 3. Novel Predictions (Tier 3)

The following speculative associations were inferred via semantic similarity to well-characterized entities. Approximately 18% of computational predictions of this class progress to experimental or clinical investigation; all require independent validation.

#### 3.1 Glycolithocholate Sulfate → Bile Salt Sulfotransferase (EC:2.8.2.14) [HIGH confidence]

**Logic chain:** Glycolithocholate sulfate (CHEBI:132924) is semantically similar (0.73) to the KEGG reaction R07287 (3'-phosphoadenylyl-sulfate:glycolithocholate sulfotransferase). EC:2.8.2.14 catalyzes R07287, which consumes glycolithocholate (CHEBI:37998) and PAPS (CHEBI:17980) and produces sulfated bile acid and PAP (CHEBI:17985). Glycolithocholate sulfate is the expected product of this reaction. [KG Evidence] [Inferred]

**Validation step:** Confirm in KEGG/BRENDA that EC:2.8.2.14 (SULT2A1 in humans) acts on glycolithocholate as substrate. The reaction name itself ("glycolithocholate sulfotransferase") strongly implies this connection. This prediction effectively fills a knowledge graph gap: the precursor-product relationship between glycolithocholate (present in this module) and glycolithocholate sulfate (also present) via SULT2A1 is biochemically established but absent from the current KG linkage. (~18% calibration note: this prediction has unusually high biochemical plausibility, substantially exceeding the base rate.)

#### 3.2 Glycodeoxycholate 3-sulfate → Colorectal Cancer Association [LOW confidence]

**Logic chain:** Glycodeoxycholate 3-sulfate (RM:0156465) is semantically similar (0.88) to glycoursodeoxycholic acid (CHEBI:89929). CHEBI:89929 is correlated_with colorectal cancer (MONDO:0005575). As a sulfated secondary bile acid conjugate, glycodeoxycholate 3-sulfate may share this disease correlation. [KG Evidence] [Inferred]

**Validation step:** Search metabolomics biomarker databases (HMDB, MetaboLights) and colorectal cancer metabolomics studies for glycodeoxycholate 3-sulfate. The module-level recurrence of colorectal cancer (5 members) provides contextual support, but the analogy from glycoursodeoxycholic acid (a 7β-epimer) to glycodeoxycholate 3-sulfate (a sulfated 7α-dehydroxylation product) spans distinct structural classes. (~18% calibration applies.)

#### 3.3 Glyco-beta-muricholate → LIPID MAPS Sterol Lipid Class (LM:ST04010064) [LOW confidence]

**Logic chain:** Glyco-beta-muricholate (CHEBI:232566; cold-start, 0 edges) is semantically similar to beta-muricholic acid (0.87) and alpha-muricholic acid (0.78), both of which are subclass_of and close_match to LM:ST04010064. As the glycine conjugate of beta-muricholic acid, glyco-beta-muricholate likely belongs to the same LIPID MAPS sterol lipid classification. [KG Evidence] [Inferred]

**Validation step:** Query LIPID MAPS for glyco-beta-muricholate and confirm whether a specific identifier has been assigned. Note that glycine conjugation may place it in a distinct sub-hierarchy (bile acid glycine conjugates, ST05) rather than the parent muricholic acid class. (~18% calibration applies.)

#### 3.4 Glycocholenate Sulfate → Atrial Fibrillation [NOTABLE observation]

Glycocholenate sulfate (76 edges) carries atrial fibrillation as its top disease association in the knowledge graph. [KG Evidence] No direct KG evidence was found linking other module members to atrial fibrillation; this association stands as an outlier relative to the module's hepatobiliary theme. [Inferred] The mechanistic basis may involve bile acid activation of cardiac ion channels (e.g., KCNQ1 or hERG), a pathway described in model systems. [Model Knowledge] This observation requires cautious interpretation: glycocholenate sulfate may be a hub-adjacent entity whose atrial fibrillation link reflects metabolomic confounding rather than causality.

#### 3.5 2-Piperidinone: The Module Outlier

4-Phenyl-2-piperidinone (PUBCHEM.COMPOUND:19817787; 1 edge) is the sole non-bile-acid metabolite in the module. [KG Evidence] Semantic similarity analysis identified structurally related piperidine compounds (4-PHENYL-1-(4-PHENYLBUTYL)PIPERIDINE, 92%; 1-METHYL-4-PHENYL-4-PIPERIDINOL, 91%) but no biologically informative connections. [KG Evidence] No direct KG evidence was found linking 2-piperidinone to bile acid metabolism. The entity resolution matched to 4-phenyl-2-piperidinone (80% confidence, fuzzy), and the original analyte ("2-piperidinone") may represent delta-valerolactam, a product of lysine catabolism by gut bacteria. [Model Knowledge] Its co-expression with the bile acid module may reflect shared microbial metabolic origin rather than direct participation in bile acid pathways. (~18% calibration applies; this interpretation is speculative.)

### 4. Biological Themes

#### 4.1 Bile Acid Conjugation Landscape

The module comprehensively represents the three major phases of bile acid conjugation and detoxification:

**Phase I (amidation):** Both glycine-conjugated (glycocholate, glycochenodeoxycholate, glycodeoxycholate, glycolithocholate, glycohyocholate, glyco-beta-muricholate) and taurine-conjugated (taurocholate, taurochenodeoxycholate, taurodeoxycholate) species are present. The predominance of glycine conjugates (≥12 species) over taurine conjugates (≥5 species) is consistent with the known 3:1 glycine:taurine conjugation ratio in humans. [KG Evidence] [Model Knowledge]

**Phase II (sulfation):** Seven sulfated species (glycolithocholate sulfate, taurolithocholate 3-sulfate, glycocholenate sulfate, taurocholenate sulfate, glycochenodeoxycholate 3-sulfate, glycodeoxycholate 3-sulfate, taurodeoxycholic acid 3-sulfate, lithocholate sulfate, deoxycholic acid sulfate) indicate active hepatic sulfotransferase activity. [KG Evidence] The Tier 3 inference linking glycolithocholate sulfate to EC:2.8.2.14 (bile salt sulfotransferase / SULT2A1) provides mechanistic grounding for this observation.

**Phase II (glucuronidation):** Two glucuronidated species (glycochenodeoxycholate glucuronide, deoxycholic acid glucuronide) reflect UGT-mediated detoxification, typically upregulated when sulfation capacity is saturated. [KG Evidence] [Model Knowledge]

#### 4.2 Primary vs. Secondary Bile Acid Balance

The module contains both primary bile acids (cholate and chenodeoxycholate conjugates, synthesized in the liver) and secondary bile acids (deoxycholate and lithocholate conjugates, produced by gut bacterial 7α-dehydroxylation). [KG Evidence] [Literature: "Microbial transformations of human bile acids," Microbiome, 2021] The co-expression of primary and secondary species in a single module indicates that their circulating levels co-vary across subjects, consistent with shared regulation by enterohepatic circulation efficiency: subjects with higher bile acid pool recycling (via ASBT/SLC10A2) would have elevated levels of both classes. [Inferred]

#### 4.3 Gut Microbiome Interface

The cross-type bridges connecting cholate to glycochenodeoxycholate glucuronide traverse multiple gut bacterial genera (Alistipes, Romboutsia, Streptococcus, Haemophilus, Butyricicoccus, Thomasclavelia). [KG Evidence] These taxa are known bile acid transformers in the human gut. [Literature: "Microbial transformations of human bile acids," Microbiome, 2021] The presence of secondary bile acids (deoxycholate, lithocholate conjugates) directly implicates microbiome-mediated biotransformation as a contributing process; the unusual species glycohyocholate and glyco-beta-muricholate (a 6β-hydroxylated bile acid typically prominent in rodents) may indicate atypical microbiome-mediated hydroxylation. [Model Knowledge]

#### 4.4 Hub-Filtered Considerations

EPCAM (3,355 edges) qualifies as a high-connectivity hub node. Associations mediated solely through EPCAM (e.g., its connections to kidney disorder, breast cancer, depressive disorder) should be interpreted with caution, as these may reflect EPCAM's broad epithelial expression rather than module-specific biology. [KG Evidence] [Inferred] The most informative EPCAM associations are those corroborated by bile acid members (hepatocellular carcinoma: 6 members; colorectal cancer: 5 members). Similarly, generic Gene Ontology terms shared by both proteins (protein binding, GO:0005515) are de-emphasized as biologically uninformative for this module.

### 5. Gap Analysis

The Open World Assumption governs interpretation: absence of an entity means "unstudied or unmeasured," not "biologically irrelevant."

#### 5.1 Informative Absences

| Absent Entity | Expected Rationale | Interpretation |
|---|---|---|
| NR1H4 (FXR) | Master bile acid nuclear receptor; induces FGF19 | Low-abundance nuclear receptor below proteomics detection; constitutive expression excludes it from WGCNA modules [Inferred] |
| CYP7A1 | Rate-limiting enzyme in bile acid synthesis; FGF19 suppresses it | Hepatocyte-specific; would be inversely correlated (anti-correlated module expected) [Inferred] |
| C4 (7α-hydroxy-4-cholesten-3-one) | Gold-standard biomarker of bile acid synthesis rate | Requires specialized assay; would be inversely correlated with FGF19 [Inferred] |
| FGFR4 / KLB | Obligate hepatic receptor for FGF19 | Hepatocyte membrane proteins; module reflects ileal ligand, not hepatic response [Inferred] |
| FGF21 | Sister endocrine FGF | Inversely regulated (fasting/PPARα vs. postprandial/FXR); likely in a separate module [Inferred] |
| Unconjugated bile acids (CDCA, LCA) | Precursors of conjugated forms present in module | Indicates intact hepatic conjugation (BAAT activity) and efficient enterohepatic circulation; marker of healthy bile acid homeostasis [Inferred] |

The coherent absence of all hepatic enzymes (CYP7A1, CYP8B1, FXR, FGFR4/KLB) confirms that this module captures intestinal and circulating bile acid biology, not hepatic synthesis machinery. [Inferred] The absence of FGF21 alongside presence of FGF19 is independently meaningful: the module reflects postprandial (fed-state) physiology, not fasting/stress signaling. [Inferred]

#### 5.2 Standard Gaps

The absence of SLC10A2 (ASBT) is attributable to the systematic underrepresentation of integral membrane transporters in shotgun proteomics. [Inferred] The absence of ursodeoxycholate (UDCA) may indicate that UDCA-producing gut bacteria (e.g., Ruminococcus gnavus with 7β-HSDH activity) are not abundant in this cohort, or that no therapeutic UDCA supplementation was administered. [Inferred] The absence of cholesterol is expected: cholesterol levels are regulated by multiple pathways and would not tightly co-vary with bile acid species. [Inferred]

### 6. Temporal Context

This analysis lacks longitudinal data; however, the module's composition permits causal inference about directionality:

**Upstream causes (bile acid pool determinants):** Hepatic bile acid synthesis (CYP7A1, CYP8B1 activity), hepatic conjugation (BAAT), and gut microbial 7α-dehydroxylation collectively determine the composition of the conjugated bile acid pool. [Model Knowledge] These processes are upstream of the measured metabolites.

**Signaling intermediary:** Ileal bile acid reabsorption (SLC10A2) → FXR activation → FGF19 secretion constitutes the causal chain linking bile acid flux to FGF19 levels. [Model Knowledge] [Literature: "FXR-FGF19 signaling in the gut–liver axis," Hepatology International, 2024]

**Downstream consequences:** FGF19 acts on hepatic FGFR4/KLB to suppress CYP7A1, completing a negative feedback loop that would stabilize bile acid pool size. [Model Knowledge] Elevated circulating bile acids and FGF19 together may indicate either increased enterohepatic cycling (physiological) or impaired hepatic clearance (pathological, e.g., cholestasis or liver injury). [Inferred]

Longitudinal sampling at pre-prandial and post-prandial time points would resolve whether inter-individual variation in this module reflects constitutive bile acid pool size differences or dynamic postprandial responses.

### 7. Research Recommendations

#### 7.1 High Priority: Experimental Validations

1. **Validate the SULT2A1→glycolithocholate sulfate axis.** Confirm that human SULT2A1 (EC:2.8.2.14) sulfates glycolithocholate in vitro. This Tier 3 prediction has strong biochemical plausibility and would fill a knowledge graph gap linking two module members (glycolithocholate and glycolithocholate sulfate) through an explicit enzymatic reaction. [Inferred]

2. **Correlate FGF19 with individual bile acid species.** Compute pairwise Spearman correlations between FGF19 protein levels and each of the 24 bile acid metabolites within the Darkred module. Identify whether primary bile acids (cholate, chenodeoxycholate conjugates) or secondary bile acids (deoxycholate, lithocholate conjugates) drive the strongest co-expression with FGF19. [Inferred]

3. **Test the hepatocellular carcinoma biomarker panel.** The six-member hepatocellular carcinoma-associated subset (EPCAM, FGF19, taurochenodeoxycholate, glycodeoxycholate, glycocholate, taurodeoxycholate) constitutes a candidate multi-analyte diagnostic panel. Evaluate its discriminative performance in an independent cohort with hepatocellular carcinoma vs. cirrhosis controls. [KG Evidence] [Inferred]

#### 7.2 Medium Priority: Literature and Database Searches

4. **Investigate glycocholenate sulfate and atrial fibrillation.** Search PubMed for mechanistic studies linking bile acid sulfates to cardiac ion channel modulation. Determine whether this association reflects direct cardiotoxicity, metabolomic confounding through liver disease, or a hub-artifact. [KG Evidence]

5. **Resolve 2-piperidinone identity.** Confirm whether the original analyte is delta-valerolactam (a lysine catabolite) or 4-phenyl-2-piperidinone. If the former, investigate its gut microbial origin and whether its co-expression with bile acids reflects shared dependence on specific bacterial taxa. [Inferred]

6. **Characterize the cold-start entities.** Glyco-beta-muricholate and taurodeoxycholic acid 3-sulfate have zero knowledge graph edges. Search HMDB, LIPID MAPS, and recent bile acid metabolomics publications for biological context. Glyco-beta-muricholate is particularly interesting given that muricholic acids are typically considered rodent-predominant; its detection in human plasma may indicate atypical 6β-hydroxylation activity. [KG Evidence] [Model Knowledge]

#### 7.3 Lower Priority: Follow-Up Analyses

7. **Cross-module correlation with CYP7A1 / FGF21 modules.** Identify the WGCNA modules containing CYP7A1 and FGF21 (if measured) and compute inter-module correlation (module eigengene correlation). The Darkred module should anti-correlate with a CYP7A1-containing module and with an FGF21/fasting-associated module. [Inferred]

8. **Microbiome integration.** The microbial taxa identified in cross-type bridges (Alistipes, Romboutsia, Butyricicoccus) represent candidate mediators of bile acid transformation. If 16S rRNA or metagenomic data are available for this cohort, correlate the abundance of these taxa with the secondary:primary bile acid ratio within the module. [KG Evidence] [Literature: "Microbial transformations of human bile acids," Microbiome, 2021]

9. **Pathway enrichment with full bile acid ontology.** Perform enrichment analysis using the SMPDB bile acid disease pathways (Zellweger Syndrome, Familial Hypercholanemia, Congenital Bile Acid Synthesis Defects II and III, Cerebrotendinous Xanthomatosis) as background, testing whether the Darkred module is disproportionately enriched relative to all measured bile acids in the metabolomics panel. [KG Evidence]

---

*Report generated from KRAKEN knowledge graph analysis of 26 resolved entities (2 genes, 24 small molecules/chemical entities). Evidence tiers: Tier 1 (direct KG facts, >280 findings), Tier 2 (derived associations, >340 findings), Tier 3 (semantic inference, ~40 predictions). Two entities (taurodeoxycholic acid 3-sulfate, glyco-beta-muricholate) are cold-start with no KG representation. All Tier 3 predictions carry the ~18% computational-to-clinical progression calibration and require independent experimental validation.*

### Literature References

Papers discovered via semantic search. 5 unique papers across 3 hypotheses.

| Hypothesis | Citation | Link | Relevance |
|------------|----------|------|-----------|
| Bridge: Gene → SmallMolecule (3 hops) |  (2021) "Frontiers \| Path4Drug: Data Science Workflow for Identification of Tissue-Specific Biological Pathwa..." | [Link](https://www.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2021.708296/full) | ChEMBL ... Human Protein Atlas ... Path4Drug ... Step 4Connecting drugs and biological pathwaysWith the list of tissue-s... |
| Bridge: Gene → SmallMolecule (2 hops) |  (2024) "FXR-FGF19 signaling in the gut–liver axis is dysregulated in patients with cirrhosis and correlates ..." | [Link](https://link.springer.com/article/10.1007/s12072-023-10636-4) | ) and increased ... 01), indicating ... , p < ... Farnesoid X receptor (FXR) is the most important endogenous receptor f... |
| Bridge: Gene → SmallMolecule (3 hops) |  (2021) "NICEpath: Finding metabolic pathways in large networks through atom-conserving substrate-product pai..." | [Link](https://pubmed.ncbi.nlm.nih.gov/34003971/) | Results: Here, we propose the construction of searchable graph representations of metabolic networks. Each reaction is d... |
| Bridge: Gene → SmallMolecule (3 hops) |  (2019) "PathMe: merging and exploring mechanistic pathway knowledge \| BMC Bioinformatics \| Springer Nature L..." | [Link](https://link.springer.com/article/10.1186/s12859-019-2863-9) | Here, we introduce PathMe, an extensible package that harmonizes multiple databases using Biological Expression Language... |
| Bridge: SmallMolecule → ChemicalEntity (2 hops) |  (2021) "Review: microbial transformations of human bile acids \| Microbiome \| Springer Nature Link" | [Link](https://link.springer.com/article/10.1186/s40168-021-01101-1) | Primary BAs are those synthesized in the liver from cholesterol [4]. The primary BA pool in humans consists of cholic ac... |