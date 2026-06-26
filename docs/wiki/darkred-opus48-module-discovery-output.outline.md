# Darkred Module Run on Opus 4.8: Discovery Output (26-analyte, dev, 2026-06-24)

This document presents the discovery output of a full-module run of the **Darkred** WGCNA module through the Kraken 12-node discovery pipeline, executed on the dev environment on 2026-06-24 (commit `a7d7fd6`) with BioMapper entity resolution enabled. The SDK-backed pipeline nodes ran on the Opus 4.8 model rather than the default Sonnet; this run is published for direct model comparison. It is one of a per-module series over the Frailty multi-omics WGCNA modules; see the [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm).

## Run Provenance

The run submitted 26 named analytes, parsed 26 at intake, and resolved 26 distinct entities (22 biomapper, 1 exact, 3 fuzzy) to 25 distinct CURIEs. Triage classified 5 well-characterized, 11 moderate, 8 sparse, and 2 cold-start, with 0 edge-count measurement failures. Downstream analysis produced 652 direct-KG findings, 36 cold-start findings, 0 biological themes, 16 cross-entity bridges (13 evidence-grounded), and 86 hypotheses supported by 17 literature references. Synthesis emitted a 22201-character report. The run completed in approximately 1021.2 s of wall-clock time (status complete, 45 errors). All counts are taken from the reducer-merged pipeline state.

| Stage | Result |
|---|---|
| Submitted | 26 named analytes |
| Intake | 26 parsed |
| Entity resolution | 26 resolved (22 biomapper, 1 exact, 3 fuzzy) to 25 distinct CURIEs |
| Triage | 5 well-characterized, 11 moderate, 8 sparse, 2 cold-start (0 measurement failures) |
| Direct KG | 652 findings |
| Cold-start | 36 findings, 3 skipped |
| Pathway enrichment | 0 biological themes |
| Integration | 16 bridges (13 evidence-grounded) |
| Literature grounding | 17 papers |
| Synthesis | 86 hypotheses, 22201-character report |
| Run total | ~1021.2 s wall-clock, status complete, 45 errors |

## Related

- Companion run metrics: [Darkred Module Run on Opus 4.8: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/darkred-module-run-on-opus-48-pipeline-performance-report-26-analyte-dev-2026-06-24-3w7ArV8YHW)
- Model comparison baseline (Sonnet): [Darkred Module Run: Discovery Output](https://phwiki.phenoma.ai/doc/darkred-module-run-discovery-output-26-analyte-dev-2026-06-23-z1Qvn5R3I1)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---



## Darkred WGCNA Module: Bile Acid Enterohepatic Signaling and Intestinal Epithelial Identity

### 1. Executive Summary

The Darkred WGCNA module encodes a coherent biological program centered on the enterohepatic bile acid pool and its intestinal endocrine regulation through the FXR to FGF19 signaling axis. [KG Evidence] Twenty-four conjugated, sulfated, and glucuronidated bile acid species co-vary with two proteins (EPCAM, FGF19) whose intersection localizes this module to ileal enterocytes: the anatomical site where luminal bile acids activate FXR, which in turn induces FGF19 secretion into the portal circulation. [KG Evidence; Literature: "FXR-FGF19 signaling in the gut-liver axis," Hepatology International, 2024] The systematic absence of hepatic synthesis enzymes (CYP7A1, CYP8B1, BAAT), hepatic transporters (ABCB11, SLC10A1), and the hepatic FGF19 receptor complex (FGFR4, KLB) confirms that this module captures the circulating and intestinal compartments of bile acid metabolism rather than the hepatic biosynthetic machinery; this tissue-compartment dissociation is among the most informative structural features of the analysis. [KG Evidence; Inferred]

### 2. Key Findings (Tier 1 to 2)

#### 2.1 The Module Represents a Comprehensive Conjugated Bile Acid Pool

The module contains representatives of every major conjugation class in human bile acid metabolism. [KG Evidence] Primary bile acid conjugates include glycocholate, taurocholate, glycochenodeoxycholate, and taurochenodeoxycholate; secondary bile acid conjugates include glycodeoxycholate, taurodeoxycholate, glycolithocholate, and their sulfated derivatives. [KG Evidence] Phase II detoxification products (sulfated and glucuronidated species) account for at least 10 of the 24 metabolites, indicating active hepatic and possibly intestinal biotransformation. [KG Evidence] Five module members participate in the Bile Acid Biosynthesis pathway (SMPDB:SMP0000035), and the same five members recur across six inborn errors of bile acid metabolism (Zellweger Syndrome, Familial Hypercholanemia, Cerebrotendinous Xanthomatosis, Congenital Bile Acid Synthesis Defect Types II and III, and 27-Hydroxylase Deficiency). [KG Evidence]

#### 2.2 FGF19: The Endocrine Hub of the Module

FGF19 (1,445 KG edges) participates in negative regulation of bile acid biosynthetic process and fibroblast growth factor receptor signaling. [KG Evidence] Multiple two-hop KG bridge paths connect FGF19 to cholate through bile acid metabolism intermediaries: FGF19 interacts with cholic acid directly, and FGF19 connects to cholate via ABCB11, SLC10A1, SLC10A2, and CYP8B1. [KG Evidence] These bridges recapitulate the canonical FXR to FGF19 to CYP7A1 feedback loop: bile acids in the ileal lumen activate FXR, FGF19 is secreted and travels to the liver, where it suppresses de novo bile acid synthesis. [Literature: "FXR-FGF19 signaling in the gut-liver axis," Hepatology International, 2024] The literature evidence confirms that FXR activation in intestines leads to FGF19 induction, and that this axis modulates bile acid synthesis, hepatic inflammation, and fibrogenesis. [Literature: "FXR-FGF19 signaling in the gut-liver axis," Hepatology International, 2024]

#### 2.3 EPCAM: Marker of Intestinal Epithelial Origin

EPCAM (3,355 KG edges) is the highest-connectivity member of the module. [KG Evidence] Its participation in cell-cell adhesion, positive regulation of stem cell proliferation, and regulation of cadherin-mediated adhesion reflects its canonical role as an epithelial cell adhesion molecule expressed at high levels on intestinal epithelial cells. [KG Evidence] EPCAM connects to cholate via a three-hop bridge through colorectal cancer (MONDO:0005575) and glycocholic acid, and via Liver Cirrhosis, Experimental (MESH:D008106). [KG Evidence] Notably, EPCAM interacts with FGFR4 in the KG (Tier 2 evidence), suggesting a previously underappreciated signaling interface between epithelial identity and FGF receptor biology. [KG Evidence]

#### 2.4 Disease Recurrence: Hepatobiliary and Gastrointestinal Cancers Predominate

Hepatocellular carcinoma is the most recurrent disease association, shared by 6 module members (including both proteins and bile acids taurochenodeoxycholate, glycodeoxycholate, glycocholate, and taurodeoxycholate). [KG Evidence] Colorectal cancer is shared by 5 members, and liver cancer by 4 members, all with curated evidence strength. [KG Evidence] Drug-induced liver injury (4 members), cholestasis (2 members), and kidney disorder (3 members) further reinforce the hepatobiliary disease axis. [KG Evidence] The convergence of bile acid species and both proteins on hepatocellular carcinoma is consistent with the known role of aberrant FGF19 signaling and bile acid dysregulation in hepatocarcinogenesis. [KG Evidence; Model Knowledge]

#### 2.5 Microbiome-Mediated Bridges

Multiple two-hop bridges connect cholate to glycochenodeoxycholate glucuronide through gut microbial genera: Thomasclavelia, Streptococcus, Romboutsia, Butyricicoccus, Alistipes, and Haemophilus. [KG Evidence] These paths (cholate correlates with microbial taxon; microbial taxon associates with glucuronidated bile acid) indicate that gut microbiome composition mediates the correlation between primary and modified bile acid species within this module. [KG Evidence] This finding is consistent with the established role of intestinal bacteria in bile acid 7-alpha-dehydroxylation, deconjugation, and other biotransformations that shape the circulating bile acid pool. [Literature: "Microbial transformations of human bile acids," Microbiome, 2021]

### 3. Novel Predictions (Tier 3)

#### 3.1 Glycolithocholate Sulfate Is a Product of Bile Salt Sulfotransferase (EC:2.8.2.14)

**Confidence**: HIGH

**Structural logic chain**: Glycolithocholate sulfate (CHEBI:132924; 3 KG edges, sparse coverage) is semantically similar to KEGG.REACTION:R07287 (similarity 0.73), which is catalyzed by bile salt sulfotransferase EC:2.8.2.14. [KG Evidence] The reaction explicitly names glycolithocholate as its substrate and produces glycolithocholate sulfate using PAPS (CHEBI:17980) as the sulfate donor, yielding PAP (CHEBI:17985) as a co-product. [KG Evidence] The unsulfated precursor glycolithocholate (CHEBI:37998) is itself a module member, providing internal validation: precursor and product co-vary within the same WGCNA module. [KG Evidence]

**Calibration note**: Approximately 18% of computational predictions of this type progress to clinical investigation; however, this prediction concerns a well-characterized enzymatic reaction rather than a novel disease association, and the chemical logic is strong.

**Validation step**: Confirm in KEGG or BRENDA that EC:2.8.2.14 catalyzes this specific sulfation. The co-occurrence of glycolithocholate and glycolithocholate sulfate in the module provides indirect experimental support for active sulfotransferase activity in this cohort.

#### 3.2 Glycodeoxycholate 3-Sulfate Is a Substrate of SULT2A1 and ABC Transporters (MRP1/MRP4)

**Confidence**: LOW (inferred via semantic similarity)

**Structural logic chain**: Glycodeoxycholate 3-sulfate (RM:0156465; 4 KG edges) is semantically similar to glycochenodeoxycholic acid 7-sulfate (similarity 0.90) and sulfoglycolithocholic acid (similarity 0.90). [KG Evidence] Both analogues connect to SULT2A1 (NCBIGene:10249), the principal bile acid sulfotransferase. [KG Evidence] Furthermore, sulfoglycolithocholic acid binds ABCC4/MRP4 (NCBIGene:10257) and ABCC1/MRP1 (NCBIGene:4363), suggesting that glycodeoxycholate 3-sulfate may also be transported by these efflux pumps. [KG Evidence] This prediction is biologically coherent: sulfated bile acids are canonical MRP substrates, and their renal or biliary elimination depends on these transporters. [Model Knowledge]

**Calibration note**: Approximately 18% of such computational predictions advance to experimental validation. The structural similarity scores (0.90) are high, lending confidence to the analogy.

**Validation step**: In vitro sulfation assays with recombinant SULT2A1, and membrane vesicle transport assays with ABCC4/MRP4 and ABCC1/MRP1, would directly test these predictions.

#### 3.3 Glycodeoxycholate 3-Sulfate May Be Altered in Hepatobiliary Disease

**Confidence**: LOW

**Structural logic chain**: Both structural analogues of glycodeoxycholate 3-sulfate (sulfoglycolithocholic acid and glycoursodeoxycholic acid) are correlated with liver disease (MONDO:0005575) in the KG. [KG Evidence] Given the module's strong enrichment for hepatocellular carcinoma (6 members), liver cancer (4 members), and drug-induced liver injury (4 members), the prediction that this understudied sulfated bile acid may be a biomarker of hepatobiliary disease aligns with the module-level disease signature. [KG Evidence; Inferred]

**Calibration note**: Approximately 18% of computational predictions progress to clinical investigation.

**Validation step**: Targeted metabolomics measurement of glycodeoxycholate 3-sulfate in liver disease patient cohorts versus healthy controls.

#### 3.4 2-Piperidinone: An Unexplained Module Outlier

2-Piperidinone (PUBCHEM.COMPOUND:19817787; 1 KG edge) is the sole non-bile-acid metabolite in this module and has negligible KG coverage. [KG Evidence] Semantic similarity search returned only piperidine derivatives (4-phenyl-1-(4-phenylbutyl)piperidine, 1-methyl-4-phenyl-4-piperidinol, 4-(2-benzylphenyl)piperidine) that bear no obvious relationship to bile acid biology. [KG Evidence] No direct KG evidence was found connecting this entity to the bile acid or FGF19 signaling axis. [KG Evidence] The following is based on [Model Knowledge]: 2-piperidinone (also known as delta-valerolactam) is a product of lysine catabolism via the piperidine pathway; its co-expression with bile acids may reflect shared hepatic or intestinal metabolic activity, or it may represent a spurious correlation requiring cautious interpretation.

**Calibration note**: Approximately 18% of computational predictions advance to clinical investigation; this association has minimal supporting evidence and should be considered highly speculative.

**Validation step**: Independent replication of the co-expression relationship in an external cohort, coupled with pathway analysis of lysine catabolism intermediates in the context of bile acid metabolism.

### 4. Biological Themes

#### 4.1 Unifying Theme: Intestinal Bile Acid Sensing and Endocrine Feedback

The central biological narrative of this module is the integrated capture of circulating conjugated bile acid pools together with the intestinal endocrine signal (FGF19) they evoke. [KG Evidence; Inferred] The module includes representatives from every branch of bile acid conjugation: glycine conjugation (glycocholate, glycochenodeoxycholate, glycodeoxycholate, glycolithocholate, glycohyocholate), taurine conjugation (taurocholate, taurochenodeoxycholate, taurodeoxycholate), sulfation (at least 8 sulfated species), and glucuronidation (glycochenodeoxycholate glucuronide, deoxycholic acid glucuronide). [KG Evidence]

#### 4.2 Bile Acid Synthesis Defect Pathways as a Convergence Signal

The recurrence of five module members across seven bile acid synthesis disorder pathways (SMPDB entries for Zellweger Syndrome, Familial Hypercholanemia, Cerebrotendinous Xanthomatosis, Congenital Bile Acid Synthesis Defect Types II and III, 27-Hydroxylase Deficiency, and the canonical Bile Acid Biosynthesis pathway) indicates that this module captures a conserved metabolic core of bile acid biology. [KG Evidence] This convergence is not an artifact of hub inflation: the recurring members (taurochenodeoxycholate, glycodeoxycholate, glycocholate, glycolithocholate, taurodeoxycholate) are moderate-coverage entities (37 to 543 edges), not high-degree hub nodes. [KG Evidence]

#### 4.3 Cell Proliferation and Epithelial Identity as a Secondary Theme

Both EPCAM and FGF19 participate in positive regulation of cell population proliferation (GO:0008284), protein binding (GO:0005515), and response to stress (GO:0006950). [KG Evidence] EPCAM additionally participates in positive regulation of stem cell proliferation and regulation of cadherin-mediated cell-cell adhesion. [KG Evidence] This convergence suggests that the module does not merely reflect passive bile acid circulation but also captures an active epithelial renewal program in the ileal compartment. [KG Evidence; Inferred]

### 5. Gap Analysis

#### 5.1 The Hepatic Machinery Is Systematically Absent

The most striking pattern revealed by the gap analysis is the complete absence of hepatic bile acid metabolism genes. [KG Evidence] This absence spans four functional categories:

**Bile acid synthesis enzymes**: CYP7A1 (rate-limiting, classic pathway), CYP8B1 (12-alpha-hydroxylase controlling cholate:CDCA ratio), CYP27A1 (alternative pathway), and BAAT (amino acid conjugation). [KG Evidence]

**Hepatic transporters**: SLC10A1/NTCP (basolateral uptake), ABCB11/BSEP (canalicular export). [KG Evidence]

**Hepatic FGF19 signaling**: FGFR4 (receptor), KLB/beta-Klotho (co-receptor), NR0B2/SHP (intracellular mediator). [KG Evidence]

**Interpretation**: This pattern is biologically informative rather than artifactual. [Inferred] The module captures the endocrine arm of the FXR-mediated bile acid feedback loop (intestinal FGF19 secretion) but not the intracrine arm (hepatic SHP-mediated CYP7A1 repression). This tissue-compartment dissociation implies the source material is plasma, serum, or intestinal tissue rather than hepatocytes. [Inferred]

#### 5.2 Ileal Transport Machinery Is Also Absent

Despite the ileal epithelial signature (FGF19 + EPCAM), the complete ileal bile acid transport apparatus is absent: SLC10A2/ASBT (apical uptake), FABP6/I-BABP (intracellular shuttling), and SLC51A/SLC51B/OSTalpha/OSTbeta (basolateral export). [KG Evidence] These transporters are likely constitutively expressed in terminally differentiated ileal enterocytes and therefore do not co-vary with the fluctuating bile acid pool, consistent with the WGCNA correlation framework. [Inferred]

#### 5.3 TGR5 Signaling Axis Is Decoupled from FXR/FGF19

GPBAR1/TGR5 is absent despite the presence of multiple TGR5 agonists (taurodeoxycholate, taurolithocholate sulfate). [KG Evidence] This absence suggests that the two major bile acid sensing systems (nuclear receptor FXR and membrane receptor TGR5) are independently regulated. [Inferred] TGR5 is expressed predominantly in enteroendocrine L-cells, while FXR/FGF19 signaling operates in absorptive enterocytes; the module appears to capture the latter cell type exclusively. [Model Knowledge]

#### 5.4 Unconjugated Bile Acids: Informative Absences

Free chenodeoxycholate (CDCA) and free lithocholate are absent while their conjugated forms are abundant. [KG Evidence] Free CDCA is efficiently conjugated by BAAT in hepatocytes, and free lithocholate is rapidly sulfated due to its hepatotoxicity; their absence reflects efficient protective metabolism. [Model Knowledge] The absence of ursodeoxycholate (UDCA) may indicate limited bacterial 7-beta-epimerization activity in this cohort, a potentially informative microbiome signature. [Inferred]

### 6. Temporal Context

This analysis derives from a WGCNA co-expression module, which captures steady-state correlations rather than temporal dynamics. [Model Knowledge] Nonetheless, the causal architecture of the FXR to FGF19 axis permits directional inference:

**Upstream causes**: Luminal bile acid concentrations (determined by dietary fat intake, hepatic synthesis rate, and enterohepatic cycling efficiency) are the proximal drivers of FXR activation in ileal enterocytes. [Model Knowledge]

**Downstream consequences**: FGF19 secretion is a consequence of FXR activation and travels via the portal circulation to suppress hepatic CYP7A1 expression. [KG Evidence; Literature: "FXR-FGF19 signaling in the gut-liver axis," Hepatology International, 2024] Elevated circulating sulfated and glucuronidated bile acid species may reflect downstream hepatic detoxification responses to an expanded bile acid pool. [Inferred]

**Causal inference opportunity**: A longitudinal study design measuring FGF19 protein levels, total bile acid pool size, and individual sulfated/glucuronidated bile acid species at multiple time points could test whether FGF19 changes precede or follow shifts in the conjugated bile acid profile. [Inferred]

### 7. Research Recommendations

#### 7.1 Highest Priority: Experimental Validations

1. **FGF19 as a module driver**: Measure FGF19 protein in matched serum/plasma and ileal biopsies to confirm that circulating FGF19 co-varies with the bile acid species in this module. This is the single most informative experiment for establishing the biological coherence of the module. [Inferred]

2. **SULT2A1 substrate specificity for sulfated module members**: Test glycodeoxycholate 3-sulfate, glycochenodeoxycholate 3-sulfate, and deoxycholic acid (12 or 24)-sulfate as substrates of recombinant SULT2A1 in vitro to validate the Tier 3 predictions. [KG Evidence; Inferred]

3. **ABC transporter profiling**: Conduct vesicle transport assays for the sulfated bile acid species (glycolithocholate sulfate, glycodeoxycholate 3-sulfate, taurodeoxycholic acid 3-sulfate) with ABCC1/MRP1 and ABCC4/MRP4 to determine their elimination routes. [KG Evidence; Inferred]

#### 7.2 High Priority: Literature and Database Mining

4. **2-Piperidinone biological context**: Search for literature on 2-piperidinone (delta-valerolactam) in the context of gut-liver metabolism, lysine catabolism, and intestinal microbiome activity. Its presence as the sole non-bile-acid metabolite in this module is unexplained and warrants targeted investigation. [KG Evidence; Model Knowledge]

5. **FGF19 amplification in hepatocellular carcinoma**: The Member Prioritization Table lists "FGF19 Gene Amplification" as the top disease for FGF19. [KG Evidence] Cross-referencing the HCC-associated bile acid species in this module with FGF19-amplified HCC patient metabolomics data could reveal whether this module signature stratifies HCC subtypes.

6. **Ursodeoxycholate absence as a microbiome signature**: Compare the gut microbiome composition of subjects with high versus low module eigengene values to determine whether UDCA-producing bacterial taxa (particularly those performing 7-beta-epimerization) are depleted. [Inferred]

#### 7.3 Standard Priority: Follow-Up Analyses

7. **Cold-start entity characterization**: Taurodeoxycholic acid 3-sulfate and glyco-beta-muricholate have zero KG edges. [KG Evidence] Targeted metabolomics method development and database deposition (HMDB, ChEBI) would enable future knowledge graph analyses of these species.

8. **Module stability testing**: Assess whether the Darkred module is preserved across independent cohorts (e.g., different disease states, age groups, or dietary interventions) to determine its generalizability beyond this specific study.

9. **FXR agonist response profiling**: Treat ileal organoids with module bile acid species individually and in combination, measuring FGF19 secretion and EPCAM expression to determine which specific bile acids drive the module's co-expression structure. [Inferred]

10. **Hepatocellular carcinoma biomarker potential**: Given that hepatocellular carcinoma is the most recurrent disease association (6 members, curated evidence), evaluate the module eigengene or a weighted score of its top bile acid members as a diagnostic or prognostic biomarker for HCC. [KG Evidence; Inferred]

---

**Methodological Note**: This report was generated from KRAKEN knowledge graph analysis of 26 resolved entities. Evidence attribution tags ([KG Evidence], [Literature], [Model Knowledge], [Inferred]) are applied throughout. All Tier 3 predictions are calibrated against the approximately 18% historical validation rate for computational predictions progressing to clinical investigation. Two module members (taurodeoxycholic acid 3-sulfate, glyco-beta-muricholate) are cold-start entities with no KG representation, limiting the completeness of the analysis for these species.

### Literature References

Papers discovered via semantic search. 5 unique papers across 3 hypotheses.

| Hypothesis | Citation | Link | Relevance |
|------------|----------|------|-----------|
| Bridge: Gene → SmallMolecule (3 hops) |  (2021) "Frontiers \| Path4Drug: Data Science Workflow for Identification of Tissue-Specific Biological Pathwa..." | [Link](https://www.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2021.708296/full) | ChEMBL ... Human Protein Atlas ... Path4Drug ... Step 4Connecting drugs and biological pathwaysWith the list of tissue-s... |
| Bridge: Gene → SmallMolecule (2 hops) |  (2024) "FXR-FGF19 signaling in the gut–liver axis is dysregulated in patients with cirrhosis and correlates ..." | [Link](https://link.springer.com/article/10.1007/s12072-023-10636-4) | ) and increased ... 01), indicating ... , p < ... Farnesoid X receptor (FXR) is the most important endogenous receptor f... |
| Bridge: Gene → SmallMolecule (3 hops) |  (2021) "NICEpath: Finding metabolic pathways in large networks through atom-conserving substrate-product pai..." | [Link](https://pubmed.ncbi.nlm.nih.gov/34003971/) | Results: Here, we propose the construction of searchable graph representations of metabolic networks. Each reaction is d... |
| Bridge: Gene → SmallMolecule (3 hops) |  (2019) "PathMe: merging and exploring mechanistic pathway knowledge \| BMC Bioinformatics \| Springer Nature L..." | [Link](https://link.springer.com/article/10.1186/s12859-019-2863-9) | Here, we introduce PathMe, an extensible package that harmonizes multiple databases using Biological Expression Language... |
| Bridge: SmallMolecule → ChemicalEntity (2 hops) |  (2021) "Review: microbial transformations of human bile acids \| Microbiome \| Springer Nature Link" | [Link](https://link.springer.com/article/10.1186/s40168-021-01101-1) | Primary BAs are those synthesized in the liver from cholesterol [4]. The primary BA pool in humans consists of cholic ac... |