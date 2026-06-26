# Black Module Run on Opus 4.8: Discovery Output (109-analyte, dev, 2026-06-24)

This document presents the discovery output of a full-module run of the **Black** WGCNA module through the Kraken 12-node discovery pipeline, executed on the dev environment on 2026-06-24 (commit `098093e`) with BioMapper entity resolution enabled. The SDK-backed pipeline nodes ran on the Opus 4.8 model rather than the default Sonnet; this run is published for direct model comparison. It is one of a per-module series over the Frailty multi-omics WGCNA modules; see the [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm).

## Run Provenance

The run submitted 109 named analytes, parsed 108 at intake, and resolved 108 distinct entities (35 fuzzy, 70 biomapper, 3 exact) to 104 distinct CURIEs. Triage classified 19 well-characterized, 27 moderate, 46 sparse, and 16 cold-start, with 0 edge-count measurement failures. Downstream analysis produced 1733 direct-KG findings, 22 cold-start findings, 8 biological themes, 1 cross-entity bridges (0 evidence-grounded), and 40 hypotheses supported by 39 literature references. Synthesis emitted a 27040-character report. The run completed in approximately 848.5 s of wall-clock time (status complete, 0 errors). All counts are taken from the reducer-merged pipeline state.

| Stage | Result |
|---|---|
| Submitted | 109 named analytes |
| Intake | 108 parsed |
| Entity resolution | 108 resolved (35 fuzzy, 70 biomapper, 3 exact) to 104 distinct CURIEs |
| Triage | 19 well-characterized, 27 moderate, 46 sparse, 16 cold-start (0 measurement failures) |
| Direct KG | 1733 findings |
| Cold-start | 22 findings, 54 skipped |
| Pathway enrichment | 8 biological themes |
| Integration | 1 bridges (0 evidence-grounded) |
| Literature grounding | 39 papers |
| Synthesis | 40 hypotheses, 27040-character report |
| Run total | ~848.5 s wall-clock, status complete, 0 errors |

## Related

- Companion run metrics: [Black Module Run on Opus 4.8: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/black-module-run-on-opus-48-pipeline-performance-report-109-analyte-dev-2026-06-24-dpbOsheqm9)
- Model comparison baseline (Sonnet): [Black Module Run: Discovery Output](https://phwiki.phenoma.ai/doc/black-module-run-discovery-output-109-analyte-dev-2026-06-23-UTui957Bee)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---



## KRAKEN Discovery Report: Black WGCNA Module (Lipid and Fatty Acid Metabolism)

### 1. Executive Summary

The Black WGCNA module encodes a coordinated program of lipid mobilization, fatty acid trafficking, and incomplete mitochondrial beta-oxidation, unified by peroxisome proliferator-activated receptor (PPAR) signaling and anchored to a tricarboxylic acid (TCA) cycle node comprising fumarate, malate, and citrate. [KG Evidence] This module comprises 108 metabolites spanning saturated and unsaturated free fatty acids (C8 to C22), acylcarnitines (C2 to C18:1), medium- and long-chain 3-hydroxy fatty acids, dicarboxylic acids, N-acyl amino acid conjugates, and endocannabinoid-like ethanolamides; the composite signature implicates active lipolysis coupled with constrained oxidative capacity, a metabolic configuration consistently associated with insulin resistance and the transition to type 2 diabetes mellitus (T2DM). [KG Evidence; Inferred] Module-level disease recurrence analysis reveals statistically recurrent associations with colorectal cancer (19 members), T2DM (9 members), diabetes mellitus (9 members), obesity (7 members), fatty liver disease (6 members), and coronary artery disorder (6 members), positioning this module at the intersection of cardiometabolic risk and oncometabolic reprogramming. [KG Evidence]

### 2. Key Findings (Tier 1 to 2)

#### 2.1 Disease Associations with Strong Evidence

The module's disease landscape is dominated by metabolic and proliferative disorders. [KG Evidence]

| Disease | Members | Evidence Type |
|---|---|---|
| Colorectal cancer | 19 | Curated |
| Diabetes mellitus | 9 | Curated |
| Type 2 diabetes mellitus | 9 | Curated |
| Cancer (general) | 8 | Text-mined |
| Breast cancer | 8 | Text-mined |
| Eosinophilic esophagitis | 7 | Curated |
| Schizophrenia | 7 | Curated |
| Obesity | 7 | Text-mined |
| Fatty liver disease | 6 | Curated |
| Coronary artery disorder | 6 | Curated |
| Atherosclerosis | 6 | Text-mined |
| MASLD | 5 | Curated |
| Alzheimer disease | 5 | Text-mined |
| IBD (type 1) | 5 | Curated |
| MCAD deficiency | 4 | Curated |

Colorectal cancer represents the most broadly supported association, with 19 module members independently curated to this disease. [KG Evidence] The convergence of palmitate, stearate, myristate, palmitoleate, and several medium-chain fatty acids on a single cancer phenotype suggests that the module captures a lipid milieu permissive for colorectal tumorigenesis, potentially through fumarate-mediated oncometabolic signaling (fumarate participates in the Warburg Effect and six oncogenic action pathways per direct KG evidence). [KG Evidence]

T2DM and general diabetes mellitus each recruit 9 members with curated evidence, including palmitate, stearate, pentadecanoate, palmitoleate, margarate, sebacate, DHA, 3-hydroxyisobutyrate, and acetylcarnitine. [KG Evidence] Notably, 3-hydroxyisobutyrate (CHEBI:37373), a valine catabolism product, carries a direct curated association with T2DM and has been independently validated as an early biomarker of insulin resistance. [KG Evidence]

The medium-chain acyl-CoA dehydrogenase (MCAD) deficiency association (4 members: caprylate, 3-hydroxylaurate, sebacate, octanoylcarnitine) points to a specific block in mitochondrial beta-oxidation at the C6 to C12 chain-length window, consistent with the module's enrichment in medium-chain 3-hydroxy fatty acids and dicarboxylic acids that accumulate when MCAD activity is insufficient. [KG Evidence; Inferred]

#### 2.2 Validated Pathway Memberships

Pathway recurrence analysis identifies three principal biological axes. [KG Evidence]

**Fatty acid biosynthesis and elongation.** Seven members participate in the GO fatty acid biosynthetic process (GO:0006633): palmitate, palmitoleate, caprylate, myristate, 3-hydroxylaurate, 3-hydroxymyristate, and stearate. [KG Evidence] An overlapping set of 6 members maps to the Biosynthesis of Unsaturated Fatty Acids pathway (PathWhiz:PW002403), connecting linoleate, DHA, arachidate, palmitate, stearate, and adrenate. [KG Evidence]

**TCA cycle and oncometabolite signaling.** Fumarate, citrate, and malate form a coherent TCA cycle triad (GO:0006099), simultaneously mapping to 12 disease pathways including the Warburg Effect, oncogenic actions of succinate and fumarate, glutaminolysis, pyruvate dehydrogenase deficiency, and mitochondrial complex II deficiency. [KG Evidence] This triple membership indicates that the module captures not merely fatty acid supply but also central carbon flux perturbation.

**Triacylglycerol degradation.** Linoleate and arachidate (and linoleate with stearate) co-participate in multiple specific triacylglycerol degradation pathways, confirming that lipolysis of stored triglycerides is a likely upstream driver of the free fatty acid abundance in this module. [KG Evidence]

#### 2.3 PPAR Signaling as a Unifying Regulatory Axis

Pathway enrichment identifies PPARA, PPARG, and PPARD as the top three gene-level shared neighbors, collectively connecting 12 input entities. [KG Evidence]

- PPARA (800 edges) binds or interacts with linoleate, stearate, palmitoleate, myristate, and caprylate. [KG Evidence]
- PPARG (900 edges) binds linoleate, stearate, palmitoleate, myristate, and arachidate. [KG Evidence]
- PPARD (400 edges) binds linoleate, stearate, and oleoyl ethanolamide. [KG Evidence]

FFAR4 (GPR120) and FFAR1 (GPR40), free fatty acid receptors that mediate insulin sensitization and incretin secretion, are established interaction partners of linoleate. [KG Evidence] GPR84, a medium-chain fatty acid receptor implicated in inflammatory signaling, connects myristate, caprylate, and 3-hydroxylaurate. [KG Evidence] These receptor-level connections provide a mechanistic scaffold linking the module's chemical composition to metabolic and inflammatory phenotypes.

### 3. Novel Predictions (Tier 3)

All Tier 3 predictions are speculative and require experimental validation. Approximately 18% of computational predictions of this nature advance to clinical investigation; confidence should be calibrated accordingly.

#### 3.1 Impaired Beta-Oxidation as the Module's Central Metabolic Defect

**Logic chain:** The module co-expresses free fatty acids (C8 to C22), their corresponding acylcarnitines (hexanoylcarnitine C6, octanoylcarnitine C8, decanoylcarnitine C10, laurylcarnitine C12, myristoylcarnitine C14, oleoylcarnitine C18:1), 3-hydroxy fatty acid intermediates (3-hydroxyhexanoate, 3-hydroxyoctanoate, 3-hydroxydecanoate, 3-hydroxylaurate, 3-hydroxymyristate), and dicarboxylic acids (sebacate C10-DC, dodecanedioate C12-DC, tetradecanedioate C14-DC, hexadecanedioate C16-DC) within a single co-expression module. [KG Evidence; Inferred] The simultaneous elevation of substrates (free fatty acids), partial-oxidation intermediates (3-hydroxy fatty acids and acylcarnitines), and omega-oxidation overflow products (dicarboxylic acids) constitutes the biochemical fingerprint of incomplete mitochondrial beta-oxidation. The KG association of four module members with MCAD deficiency confirms the mechanistic plausibility of a beta-oxidation bottleneck at the medium-chain stage. [KG Evidence] Acylcarnitine elevations have been reported in coronary artery disease, where serum levels correlate with disease severity (Gander et al., 2021). [Literature]

**Validation step:** Measure the ratios of acylcarnitines to their corresponding free fatty acids (e.g., octanoylcarnitine/caprylate, decanoylcarnitine/caprate) across the study cohort. Decreased ratios would confirm impaired carnitine palmitoyltransferase (CPT) or acyl-CoA dehydrogenase flux. Targeted enzymatic assays for MCAD and VLCAD activity in accessible tissue (e.g., peripheral blood mononuclear cells) would provide direct validation.

#### 3.2 The Ceramide Gap: a Potentially Protective Phenotype

**Logic chain:** Palmitate and stearate, the obligate precursors for de novo ceramide biosynthesis via serine palmitoyltransferase, are present in the module, yet no ceramide species co-express with them. [KG Evidence; Inferred] Ceramides (particularly Cer(d18:1/16:0) and Cer(d18:1/18:0)) are established mediators of lipotoxic insulin resistance and have been validated as cardiovascular risk biomarkers. The dissociation between elevated precursors and absent products suggests one of three scenarios: (1) the metabolomics platform lacked lipidomics-grade ceramide coverage; (2) ceramides segregate to a distinct WGCNA module with different temporal dynamics; or (3) ceramide synthesis is actively suppressed (e.g., via myriocin-like endogenous inhibition) despite substrate excess, representing a potentially protective metabolic configuration. [Inferred]

**Validation step:** If ceramides were measured, examine their module assignment and their correlation with palmitate/stearate. If not measured, a targeted sphingolipidomics assay on the same samples would directly test whether the ceramide gap reflects measurement limitation or genuine biology. A negative ceramide-to-palmitate correlation would support the protective-phenotype hypothesis.

#### 3.3 PPAR-Endocannabinoid Crosstalk via N-Acyl Ethanolamides

**Logic chain:** The module contains three N-acyl ethanolamides (palmitoyl ethanolamide (PEA), oleoyl ethanolamide (OEA), and linoleoyl ethanolamide), which are endogenous ligands for PPARalpha (PEA, OEA) and TRPV1/CB1 receptors. [KG Evidence; Model Knowledge] Linoleate is a KG-established binder of CNR1 (CB1 receptor). [KG Evidence] The co-expression of PPAR ligands (free fatty acids), endocannabinoid-like signaling lipids (ethanolamides), and N-acyltaurine species (N-oleoyltaurine, N-stearoyltaurine) within a single module suggests a coordinated lipid-signaling program that bridges PPAR-mediated metabolic regulation with endocannabinoid tone. [Inferred] This convergence is biologically plausible: OEA activates PPARalpha to promote satiety and fatty acid oxidation, while PEA exerts anti-inflammatory effects via PPARalpha and GPR55. [Model Knowledge]

**Validation step:** Correlate ethanolamide concentrations with fatty acid amide hydrolase (FAAH) activity or expression levels. Test whether module eigenvalue associates with clinical measures of appetite, satiety, or inflammatory status. Examine whether PPAR agonist treatment (e.g., fibrates) shifts module membership.

#### 3.4 Suberoylcarnitine as a Circulating Biomarker of Omega-Oxidation

**Logic chain:** Suberoylcarnitine (CHEBI:77083) is inferred via semantic similarity (0.89 to O-suberoylcarnitine) to be an O-acylcarnitine detectable in blood and urine. [KG Evidence; Inferred] The module's enrichment in dicarboxylic acids (sebacate, dodecanedioate, hexadecanedioate, and multiple dicarboxylic acid diacylcarnitines) indicates active omega-oxidation, the alternative fatty acid degradation pathway engaged when beta-oxidation capacity is exceeded. [KG Evidence; Inferred] Suberoylcarnitine has been identified as a metabolomic marker in DASH dietary pattern studies (Rebholz et al., 2018) and has been associated with metabolic perturbation in Parkinson's disease (Shao et al., 2019) and septic shock (Shrestha et al., 2024). [Literature] Its elevation, together with other dicarboxylic acylcarnitines, in coronary artery disease further supports its role as a marker of mitochondrial metabolic dysfunction (Gander et al., 2021). [Literature]

**Validation step:** Quantify suberoylcarnitine (C8-DC carnitine) by LC-MS/MS in plasma and correlate with clinical insulin resistance indices (HOMA-IR) and with other dicarboxylic acid species in the module.

#### 3.5 9-Hydroxystearate as an Antiproliferative Lipid Signal

**Logic chain:** 9-Hydroxystearate (CHEBI:229769, sparse KG coverage with 1 edge) is inferred to belong to the hydroxy fatty acid anion class (CHEBI:59835) based on structural analogy with 7-hydroxystearate and 11-hydroxystearate. [KG Evidence; Inferred] (R)-9-Hydroxystearic acid has been reported to possess antiproliferative activity against HT 29 colorectal cancer cells (2019 synthesis study). [Literature] Its co-expression within a module strongly associated with colorectal cancer (19 members) creates a hypothesis that 9-hydroxystearate may serve as an endogenous antiproliferative counterbalance to the pro-tumorigenic lipid milieu defined by the module's saturated fatty acids.

**Validation step:** Measure 9-hydroxystearate concentrations across colorectal cancer cases and controls within the cohort. Test correlation with colorectal cancer outcomes in a survival analysis framework.

### 4. Biological Themes

#### 4.1 Lipolysis and Fatty Acid Release

The module's dominant chemical signature comprises saturated fatty acids spanning C8 (caprylate) through C22 (arachidate, erucate), unsaturated fatty acids from multiple omega series (n-3: EPA, DHA, DPA, stearidonate; n-6: linoleate, arachidonate, adrenate; n-7: palmitoleate; n-9: oleate, eicosenoate, erucate), glycerol (a triacylglycerol hydrolysis co-product), and multiple triacylglycerol degradation pathways. [KG Evidence] This comprehensive representation of lipolysis products, rather than a single fatty acid chain length, indicates that the module captures a systemic lipolytic event (e.g., adipose tissue triglyceride lipase/hormone-sensitive lipase activation) rather than selective fatty acid release.

#### 4.2 Incomplete Beta-Oxidation and Omega-Oxidation Overflow

The presence of acylcarnitines across a broad chain-length range (C2 to C18:1), 3-hydroxy fatty acid intermediates (the product of the second step of each beta-oxidation cycle), and dicarboxylic acids (the products of microsomal omega-oxidation) creates a coherent picture of mitochondrial overload. [KG Evidence; Inferred] PPARA, the master transcriptional regulator of beta-oxidation genes, connects to 5 module members, suggesting that PPAR-driven transcriptional upregulation of oxidation machinery may be insufficient to handle the lipolytic fatty acid load. [KG Evidence]

#### 4.3 TCA Cycle Perturbation

Fumarate, malate, and citrate form a TCA cycle triad that maps to 12 disease-associated pathways, including oncometabolic pathways (Warburg effect, oncogenic actions of succinate and fumarate) and mitochondrial enzyme deficiencies. [KG Evidence] Aconitate (cis or trans), a TCA cycle intermediate between citrate and isocitrate, is also present. [KG Evidence] Alpha-ketobutyrate, a product of threonine and methionine catabolism that feeds into propionyl-CoA and thence succinyl-CoA, links amino acid catabolism to TCA cycle anaplerosis. [KG Evidence; Model Knowledge] The co-expression of these intermediates with fatty acid oxidation products suggests that mitochondrial carbon flux is perturbed at multiple entry points.

#### 4.4 Endocannabinoid and N-Acyl Signaling

Three ethanolamides (PEA, OEA, linoleoyl ethanolamide), two N-acyltaurines (N-oleoyltaurine, N-stearoyltaurine), and N-palmitoylglycine constitute a lipid-signaling submodule. [KG Evidence] PEA alone has 1,370 KG edges, reflecting its extensively characterized pharmacology. [KG Evidence] These species activate PPARs, TRPV1, and cannabinoid receptors, positioning the module at the interface of metabolic and inflammatory regulation. [KG Evidence; Model Knowledge]

#### 4.5 Gut Microbial Connections

Pathway enrichment identified several gut-associated organisms: Bacteroides dorei (hub-flagged), Odoribacter laneus, and additional taxa connecting to 3 input entities. [KG Evidence] The presence of odd-chain fatty acids (pentadecanoate C15:0, margarate C17:0, nonadecanoate C19:0) is notable because odd-chain fatty acids are partly of microbial (gut) origin. [KG Evidence; Model Knowledge] Note: glycerol (CHEBI:17754, 10,000 edges) is flagged as a hub entity, and associations mediated exclusively through glycerol should be interpreted cautiously.

### 5. Gap Analysis

The Open World Assumption governs interpretation: absence of an entity from the module signifies "not co-expressed here" rather than "not present."

#### 5.1 Informative Absences

**Branched-chain amino acids (leucine, isoleucine, valine).** BCAAs are among the most replicated early T2DM biomarkers, yet they are absent from this lipid-dominated module. [Inferred] Their absence likely reflects WGCNA topology (amino acid modules segregating from lipid modules) rather than non-measurement; this separation is itself informative, indicating that BCAA elevation and FFA mobilization may operate through partially independent mechanisms or temporal windows.

**Ceramides.** The module contains ceramide precursors (palmitate, stearate) but not ceramides. [Inferred] This is the most informative absence in the analysis (see Tier 3 prediction 3.2).

**Acylcarnitine completeness.** Although 14 acylcarnitine species are present, the set is incomplete relative to what a fully characterized beta-oxidation program would generate. [KG Evidence; Inferred] The presence of acylcarnitines at some chain lengths but not others may indicate chain-length-specific bottlenecks in beta-oxidation.

**Insulin, C-peptide, HbA1c, HOMA-IR, fasting glucose.** These clinical variables are expected in T2DM research but are not metabolomics platform analytes. [Inferred] Their absence reflects measurement-domain boundaries and is non-informative regarding the module's biology.

**Adiponectin.** No proteins were included in the module input. Adiponectin is the canonical adipokine counter-regulator of the processes this module captures (lipolysis, PPAR signaling, FFA-induced insulin resistance). [Model Knowledge] If measured, adiponectin would be expected to correlate inversely with the module eigenvalue.

#### 5.2 Cold-Start Entities

Sixteen metabolites (e.g., laurate, caprate, butenoylglycine, trans-2-hexenoylglycine, dodecadienoate, decadienedioic acid) have zero KG edges. [KG Evidence] These entities are real metabolites measured by the platform but lack representation in the knowledge graph, limiting the ability to derive disease or pathway associations for them. Their co-expression with well-characterized module members provides indirect functional annotation through guilt-by-association.

### 6. Temporal Context

No explicit longitudinal timepoint metadata was provided with this analysis. However, the module's composition permits provisional causal reasoning.

**Upstream causes (likely preceding module activation):**
- Adipose tissue lipolysis (evidenced by co-expressed glycerol and free fatty acids) [Inferred]
- PPAR-mediated transcriptional activation of fatty acid oxidation genes (evidenced by PPARA/PPARG/PPARD as shared neighbors) [KG Evidence; Inferred]
- Insulin resistance or relative insulin deficiency (evidenced by disease recurrence for T2DM and obesity; absence of insulin from the module) [KG Evidence; Inferred]

**Downstream consequences (likely following module activation):**
- Incomplete beta-oxidation and accumulation of acylcarnitines and 3-hydroxy intermediates [Inferred]
- Omega-oxidation overflow generating dicarboxylic acids [Inferred]
- TCA cycle metabolite accumulation (fumarate, malate, citrate, aconitate), potentially indicating mitochondrial congestion [KG Evidence; Inferred]
- Endocannabinoid-like signaling perturbation via N-acyl ethanolamides [Inferred]

If longitudinal clinical data are available, testing whether the module eigenvalue at baseline predicts incident T2DM, cardiovascular events, or colorectal cancer would establish the causal direction.

### 7. Research Recommendations

#### 7.1 Highest Priority (Experimental Validation)

1. **Acylcarnitine-to-FFA ratio profiling.** Calculate ratios of each acylcarnitine to its corresponding free fatty acid across the cohort to quantify beta-oxidation efficiency. Correlate with HOMA-IR and HbA1c. [Inferred]

2. **Ceramide targeted measurement.** If ceramides were not measured on the original platform, perform targeted sphingolipidomics (Cer(d18:1/16:0), Cer(d18:1/18:0), Cer(d18:1/24:0), Cer(d18:1/24:1)) on the same samples. Test whether ceramide levels correlate with palmitate and stearate concentrations and whether their absence from the Black module reflects suppressed ceramide synthesis. [Inferred]

3. **Module eigenvalue as a disease predictor.** Test the module eigenvalue (first principal component) as a predictor of incident T2DM, coronary artery disease, and colorectal cancer in the study cohort. The breadth of disease associations (19 members for colorectal cancer, 9 for T2DM, 6 for coronary artery disorder) warrants systematic prognostic evaluation. [KG Evidence; Inferred]

#### 7.2 Medium Priority (Literature and Database Mining)

4. **9-Hydroxystearate antiproliferative activity.** The literature evidence for antiproliferative effects against HT 29 cancer cells (2019 synthesis studies) [Literature] warrants a focused literature review and, if confirmed, measurement of 9-hydroxystearate in colorectal cancer cohorts.

5. **Odd-chain fatty acid and gut microbiome connection.** Pentadecanoate (C15:0), margarate (C17:0), and nonadecanoate (C19:0) are partly microbially derived. Cross-reference module membership with available 16S rRNA or metagenomic data to test whether gut microbial community composition (e.g., Bacteroides dorei abundance) predicts odd-chain fatty acid levels. [KG Evidence; Model Knowledge]

6. **Endocannabinoid tone assessment.** Correlate PEA, OEA, and linoleoyl ethanolamide concentrations with FAAH expression or activity, body mass index, and inflammatory markers (CRP, IL-6) to test the PPAR-endocannabinoid crosstalk hypothesis. [KG Evidence; Inferred]

#### 7.3 Follow-Up Analyses

7. **Cross-module integration.** Examine the BCAA-containing WGCNA module and test for correlation, anti-correlation, or temporal offset with the Black module eigenvalue. A positive correlation would suggest coordinated insulin resistance; anti-correlation could reveal compensatory mechanisms. [Inferred]

8. **Hub-adjusted network analysis.** Glycerol (10,000 edges) should be de-weighted in any network-based analysis. Re-run module-disease and module-pathway analyses excluding glycerol to confirm that associations are not driven by this hub entity. [KG Evidence]

9. **Cold-start entity annotation.** The 16 entities with zero KG edges would benefit from manual literature curation, particularly butenoylglycine, trans-2-hexenoylglycine, and the glutamine conjugates, which may represent novel biomarkers of peroxisomal or microsomal fatty acid metabolism not yet captured in biomedical knowledge graphs. [KG Evidence; Inferred]

---

*Report generated from KRAKEN knowledge graph analysis. All Tier 1 and 2 findings derive from direct KG queries. Tier 3 predictions carry an estimated 18% progression rate to clinical investigation and require independent experimental validation. Hub-bias warnings apply to glycerol (CHEBI:17754; 10,000 edges); associations mediated solely through this node should be interpreted with caution.*

### Literature References

Papers discovered via semantic search. 13 unique papers across 6 hypotheses.

| Hypothesis | Citation | Link | Relevance |
|------------|----------|------|-----------|
| Inferred role of CHEBI:77083 | Casey M. Rebholz et al. (2018) "Serum untargeted metabolomic profile of the Dietary Approaches to Stop Hypertension (DASH) dietary p..." | [DOI](https://doi.org/10.1093/ajcn/nqy099) | — |
| Inferred role of CHEBI:77083 | Yaping Shao et al. (2019) "Recent advances and perspectives of metabolomics-based investigations in Parkinson's disease" | [DOI](https://doi.org/10.1186/s13024-018-0304-2) | — |
| Inferred role of UNII:14E51136PO | Faten Dhawi et al. (2025) "<i>In Silico</i> Evaluation of Anti-<i>SARS-CoV-2</i> Bioactive Compounds from <i>Jatropha curcas</i..." | [DOI](https://doi.org/10.21926/obm.genet.2502295) | — |
| Inferred role of UNII:14E51136PO |  (2005) "A kinetic and thermodynamic study on hydrolysis of sodium laurate in aqueous phase accompanied by tr..." | [Link](https://www.sciencedirect.com/science/article/abs/pii/S0927776505002390) | laurate (NaLA), and its ... 7]. However ... of NaLA. In ... species of lauric acid (LA) molecules are quickly ... phase;... |
| Inferred role of CHEBI:88552 |  (2023) "Cis-2-Decenoic Acid and Bupivacaine Delivered from Electrospun Chitosan Membranes Increase Cytokine ..." | [Link](https://www.mdpi.com/1999-4923/15/10/2476) | Cis-2-Decenoic Acid and Bupivacaine Delivered from Electrospun Chitosan Membranes Increase Cytokine Production in Derm... |
| Inferred role of CHEBI:178069 |  (2020) "Dissecting Cellular Mechanisms of Long-Chain Acylcarnitines-Driven Cardiotoxicity: Disturbance of Ca..." | [Link](https://www.mdpi.com/1422-0067/21/20/7461) | Figure 1 Concentration-dependent effects of PC and MC: Ca 2+ ... sparks and Ca 2+ -enriched microdomains, Ca 2+ overload... |
| Inferred role of CHEBI:229769 | Edward Mubiru (2018) "Epoxy fatty acids in foods : analytics, formation and risk assessment" | — | — |
| Inferred role of CHEBI:178069; Inferred role of CHEBI:77083; Inferred role of PUBCHEM.COMPOUND:129692017 |  (2013) "Quantification of plasma carnitine and acylcarnitines by high-performance liquid chromatography-tand..." | [Link](https://link.springer.com/article/10.1007/s00216-013-7309-z) | Carnitine is an amino acid derivative that plays a key role in energy metabolism. Endogenous carnitine is found in its f... |
| Inferred role of UNII:14E51136PO |  (2016) "Solvent Free Lipase Catalysed Synthesis of Ethyl Laurate: Optimization and Kinetic Studies \| Applied..." | [Link](https://link.springer.com/article/10.1007/s12010-016-2177-6) | Solvent Free Lipase Catalysed Synthesis of Ethyl Laurate: Optimization and Kinetic Studies \| Applied Biochemistry and Bi... |
| Inferred role of CHEBI:178069 |  (2025) "Structural annotation of acylcarnitines detected in SRM 1950 using collision-induced dissociation an..." | [Link](https://link.springer.com/article/10.1007/s00216-025-06234-y) | Acylcarnitines are esters formed through the conjugation of fatty acids with carnitine. Their primary biological role is... |
| Inferred role of CHEBI:229769 |  (2019) "Synthesis of 9-Hydroxystearic Acid Derivatives and Their Antiproliferative Activity on HT 29 Cancer ..." | [Link](https://www.mdpi.com/1420-3049/24/20/3714) | -Hydroxystearic acid ... -HSA) ... antiproliferative and ... effects against cancer cells ... A series of derivatives ..... |
| Inferred role of CHEBI:88552 |  (2016) "The Novel Effect of cis-2-Decenoic Acid on Biofilm Producing Pseudomonas aeruginosa" | [Link](https://www.mdpi.com/2036-7481/6/1/6158) | The Novel Effect of cis-2 ... The Novel Effect of cis-2-Decenoic Acid on Biofilm Producing Pseudomonas aeruginosa ...... |
| Inferred role of CHEBI:229769 |  (2019) "X-Ray Crystal Structures and Organogelator Properties of (R)-9-Hydroxystearic Acid" | [Link](https://www.mdpi.com/1420-3049/24/15/2854) | (R)-9-hydroxystearic acid ... (R)-9-HSA, is a chiral nonrac ... hydroxyacid of natural origin possessing interesting pro... |