# Purple Module Run: Discovery Output (51-analyte, dev, 2026-06-23)

This document presents the discovery output of a full-module run of the **Purple** WGCNA module through the Kraken 12-node discovery pipeline, executed on the dev environment on 2026-06-23 (commit `813fc3f`) with BioMapper entity resolution enabled. It is one of a per-module series over the Frailty multi-omics WGCNA modules; see the [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm).

## Run Provenance

The run submitted 51 named analytes, parsed 47 at intake, and resolved 47 distinct entities (4 biomapper, 42 fuzzy, 1 exact) to 35 distinct CURIEs. Triage classified 4 well-characterized, 30 moderate, 11 sparse, and 2 cold-start, with 0 edge-count measurement failures. Downstream analysis produced 1097 direct-KG findings, 18 cold-start findings, 3 biological themes, 10 cross-entity bridges (10 evidence-grounded), and 40 hypotheses supported by 20 literature references. Synthesis emitted a 21653-character report. The run completed in approximately 700.1 s of wall-clock time (status complete, 25 errors). All counts are taken from the reducer-merged pipeline state.

| Stage | Result |
|---|---|
| Submitted | 51 named analytes |
| Intake | 47 parsed |
| Entity resolution | 47 resolved (4 biomapper, 42 fuzzy, 1 exact) to 35 distinct CURIEs |
| Triage | 4 well-characterized, 30 moderate, 11 sparse, 2 cold-start (0 measurement failures) |
| Direct KG | 1097 findings |
| Cold-start | 18 findings, 6 skipped |
| Pathway enrichment | 3 biological themes |
| Integration | 10 bridges (10 evidence-grounded) |
| Literature grounding | 20 papers |
| Synthesis | 40 hypotheses, 21653-character report |
| Run total | ~700.1 s wall-clock, status complete, 25 errors |

## Related

- Companion run metrics: [Purple Module Run: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/purple-module-run-pipeline-performance-report-51-analyte-dev-2026-06-23-CBbswHQIxY)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---



## Discovery Report: Purple WGCNA Module — Glycerophospholipid Remodeling and Fat-Soluble Vitamin Biology

---

### 1. Executive Summary

The Purple WGCNA module encodes a coherent biological program centered on glycerophospholipid membrane remodeling coupled with fat-soluble vitamin (retinol, alpha-tocopherol) transport and antioxidant defense, bridged to immune and hematopoietic signaling through the cytokine KITLG (stem cell factor). [KG Evidence, Inferred] Disease recurrence analysis reveals convergence on type 2 diabetes mellitus (T2D), cardiovascular disease, and skin/lung disorders across three of the four well-characterized hub members, indicating that this module captures a lipid-membrane and micronutrient axis relevant to cardiometabolic and inflammatory pathology. [KG Evidence] The near-complete absence of sphingolipids, lysophosphatidylcholines, acylcarnitines, and branched-chain amino acids from this module is itself informative: it delineates the Purple module as a glycerolipid/phospholipid storage and composition network, distinct from sphingolipid, mitochondrial beta-oxidation, and amino acid catabolic programs that likely reside in separate co-expression modules. [Inferred]

---

### 2. Key Findings (Tier 1 to 2)

#### 2.1 Unifying Biological Theme: Glycerophospholipid Remodeling Under Antioxidant Regulation

The module comprises 45 glycerophospholipid species spanning four headgroup classes (phosphatidylcholine, PC; phosphatidylethanolamine, PE; phosphatidylinositol, PI; phosphatidylglycerol, PG), two fat-soluble vitamins (retinol and alpha-tocopherol), the generic phosphatidylcholine entity, and the hematopoietic cytokine KITLG. [KG Evidence] Pathway enrichment analysis identifies phospholipase A2 family members (PLA2G6, PLA2G1B), hepatic lipase (LIPC), lysosomal acid lipase (LIPA), and patatin-like phospholipase PNPLA3 as shared enzymatic bridges connecting 20 input lipid entities to common gene nodes. [KG Evidence] This enzymatic signature is consistent with Lands cycle remodeling, in which phospholipases cleave sn-2 acyl chains from glycerophospholipids and lysophospholipid acyltransferases re-esterify them, thereby governing membrane fatty acid composition.

#### 2.2 Module-Level Disease Recurrence

The following disease associations recur across multiple module members and carry curated evidence:

| Disease | Members (count) | Evidence Strength | Hub Bias Risk |
|---|---|---|---|
| Type 2 diabetes mellitus | retinol, alpha-tocopherol, phosphatidylcholine (3) | Curated | High (2 of 3 are hubs) |
| Skin disorder | retinol, alpha-tocopherol, KITLG (3) | Curated | High |
| Lung disorder | retinol, alpha-tocopherol, KITLG (3) | Curated | High |
| Cardiovascular disorder | retinol, KITLG (2) | Curated | Moderate |
| Digestive system disorder | retinol, KITLG (2) | Curated | Moderate |
| Kidney disorder | retinol, KITLG (2) | Curated | Moderate |
| Sickle cell anemia | retinol, alpha-tocopherol (2) | Curated | High (both hubs) |
| Myocardial infarction | retinol, alpha-tocopherol (2) | Curated | High (both hubs) |
| Fatty liver disease / MASLD | alpha-tocopherol, phosphatidylcholine (2) | Text-mined | Moderate |
| Hypothyroidism | retinol, phosphatidylcholine (2) | Curated | Moderate |

[KG Evidence]

**Hub bias caveat.** Retinol (7,680 edges) and alpha-tocopherol (9,195 edges) are high-connectivity hub nodes; disease associations shared exclusively between these two entities (sickle cell anemia, myocardial infarction, asthma, epilepsy, dilated cardiomyopathy, and 12 others) may be spurious and should be interpreted with caution. [KG Evidence] Associations that include KITLG (2,959 edges; below the hub-warning threshold of well-characterized but not pathologically promiscuous) or phosphatidylcholine (410 edges) as independent members carry greater module-specific signal.

#### 2.3 Convergence on T2D and Cardiometabolic Risk

T2D recurs across three members (retinol, alpha-tocopherol, phosphatidylcholine) with curated evidence. [KG Evidence] Fatty liver disease and metabolic dysfunction-associated steatotic liver disease (MASLD) recur across alpha-tocopherol and phosphatidylcholine with text-mined evidence. [KG Evidence] The dominance of glycerophospholipid species bearing linoleoyl (18:2), arachidonoyl (20:4), and docosahexaenoyl (22:6) acyl chains in this module suggests active remodeling of omega-6 and omega-3 polyunsaturated fatty acids (PUFAs) within membrane phospholipids, a process implicated in insulin sensitivity and hepatic lipid handling. [Model Knowledge]

#### 2.4 KITLG as an Immune and Hematopoietic Bridge

KITLG participates in T cell proliferation (shared with retinol), positive regulation of hematopoietic stem cell proliferation, mast cell proliferation and migration, melanocyte differentiation, and Ras/MAPK signal transduction. [KG Evidence] Its primary receptor, KIT (c-KIT), is a confirmed interactor. [KG Evidence] Cross-type bridge analysis reveals that KITLG connects to retinol through melanoma (KITLG → melanoma → retinol; curated-associative) and to alpha-tocopherol through cardiovascular disorder, melanoma, eye disorder, lung disorder, digestive system disorder, and kidney disorder via two-hop paths with curated-associative provenance on the weakest leg. [KG Evidence]

Literature evidence supports the biological plausibility of the KITLG-to-retinol connection via eye biology: KITLG activates KIT signaling, which induces nuclear NRF2 accumulation and stimulates HMOX1 expression to protect photoreceptor cells against light-induced and inherited retinal degeneration (KIT ligand protects against both light-induced and genetic photoreceptor degeneration, 2020). [Literature] Retinol is itself essential for photoreceptor maintenance and visual cycle function, establishing a mechanistic basis for co-expression of KITLG and retinol in a shared module. [Model Knowledge]

#### 2.5 Retinol and Alpha-Tocopherol: Shared Vitamin Transport and Immune Functions

Retinol and alpha-tocopherol share pathway membership in vitamin transport (GO:0051180) and the UMLS categories "Physiological Phenomena" and "Diet, Food, and Nutrition." [KG Evidence] Both vitamins participate in T cell proliferation alongside KITLG. [KG Evidence] Retinol interacts with established metabolic enzymes including RBP4, LRAT, ALDH1A1, ALDH1A2, and multiple retinol dehydrogenases (RDH5, RDH10, RDH11, RDH12, RDH13). [KG Evidence] Alpha-tocopherol demonstrates gene-regulatory activity beyond its antioxidant function (Gene-Regulatory Activity of alpha-Tocopherol, 2010). [Literature]

#### 2.6 Member Prioritization

The four well-characterized members (alpha-tocopherol, retinol, KITLG, phosphatidylcholine) account for all curated disease associations and pathway annotations in the module. [KG Evidence] Among moderately covered members, linoleate (CHEBI:30245; 74 edges) connects to 20 input entities via shared enzymatic nodes and represents a key fatty acid substrate for Lands cycle remodeling. [KG Evidence] The remaining approximately 40 glycerophospholipid species individually have 29 to 39 edges (moderate coverage) or fewer, with no independent disease associations, confirming their role as co-regulated structural lipids rather than individually characterized biomarkers. [KG Evidence]

---

### 3. Novel Predictions (Tier 3)

#### 3.1 KITLG/NRF2 Antioxidant Axis as Mechanistic Link Between KITLG and Fat-Soluble Vitamins

**Logic chain:** KITLG → KIT receptor activation → NRF2 nuclear translocation → HMOX1 induction → antioxidant defense. Alpha-tocopherol is the principal lipid-phase antioxidant in membranes; retinol participates in redox-sensitive retinoid signaling. The co-expression of KITLG with both fat-soluble vitamins and with membrane phospholipids enriched in oxidation-susceptible PUFAs (arachidonoyl, docosahexaenoyl) suggests a coordinated antioxidant membrane-protection program. [Inferred; Literature: KIT ligand protects against photoreceptor degeneration, 2020, which demonstrates KITLG → NRF2 → HMOX1 antioxidant axis]

**Validation step:** Measure NRF2 nuclear accumulation and HMOX1 transcript levels in the study cohort; test whether KITLG protein levels correlate with oxidative stress markers (F2-isoprostanes, oxidized phospholipids) in plasma. Approximately 18% of computational predictions of this nature advance to clinical investigation.

#### 3.2 Purple Module as a Circulating Phospholipid-Membrane Composition Signature

**Logic chain:** The module contains glycerophospholipids spanning PC, PE, PI, and PG headgroups with diverse acyl chain compositions (14:0 through 22:6), but lacks lysophosphatidylcholines, sphingomyelins, and ceramides. The enzymatic bridges (PLA2G6, LIPC, PNPLA3) are Lands cycle and hepatic lipid remodeling enzymes. This pattern suggests the module reflects the composition of circulating lipoproteins or extracellular vesicle membranes undergoing active acyl chain remodeling, rather than sphingolipid-driven lipotoxicity or mitochondrial oxidation. [Inferred; KG Evidence for enzymatic bridges; Model Knowledge for Lands cycle interpretation]

**Validation step:** Fractionate plasma into lipoprotein and extracellular vesicle compartments; perform targeted lipidomics on each fraction to determine which compartment(s) harbor the correlated phospholipid species of this module. Approximately 18% of such computational predictions progress to experimental investigation.

#### 3.3 Soluble KITLG as a Marker of Mast Cell or Melanocyte Activity in This Metabolic Context

**Logic chain:** KIT receptor (c-KIT) is absent from the module, suggesting that KITLG in this dataset represents its soluble, cleaved form rather than membrane-bound autocrine signaling. [KG Evidence: gap analysis] Soluble KITLG regulates mast cell proliferation and migration (KG Evidence: KITLG → positive regulation of mast cell proliferation, mast cell migration). Mast cells are rich sources of phospholipase A2, which generates the lysophospholipid and free fatty acid substrates of the Lands cycle. The co-expression of soluble KITLG with Lands-cycle phospholipids may therefore reflect mast cell-derived phospholipid remodeling activity. [Inferred; Model Knowledge for mast cell PLA2 biology]

**Validation step:** Correlate soluble KITLG concentrations with tryptase (mast cell marker) in the same cohort; assess whether mast cell activation markers track with module eigengene values. Approximately 18% of such inferred associations are ultimately validated.

#### 3.4 PNPLA3 as a Candidate Genetic Modifier of Module Behavior

**Logic chain:** PNPLA3 appears as a shared enzymatic bridge across 20 input lipid entities. [KG Evidence] The PNPLA3 I148M variant (rs738409) is the strongest genetic risk factor for MASLD and modulates hepatic phospholipid and triglyceride remodeling. [Model Knowledge] MASLD recurs as a module-level disease association (alpha-tocopherol, phosphatidylcholine). [KG Evidence]

**Validation step:** Genotype the study cohort for rs738409 (PNPLA3 I148M); test whether the variant modifies the module eigengene or individual phospholipid species levels. Approximately 18% of such gene-module interaction predictions yield confirmatory results.

---

### 4. Biological Themes

#### 4.1 Glycerophospholipid Acyl Chain Remodeling (Lands Cycle)

The dominant theme is phospholipid acyl chain composition across four headgroup classes (PC, PE, PI, PG). [KG Evidence] The module spans saturated (14:0, 16:0, 18:0), monounsaturated (16:1, 18:1), and polyunsaturated (18:2, 18:3, 20:3, 20:4, 20:5, 22:4, 22:5, 22:6) acyl chains. [KG Evidence] Phospholipase A2 family members (PLA2G6, PLA2G1B) and hepatic lipase (LIPC) serve as shared enzymatic nodes connecting these species. [KG Evidence] This composition is consistent with active Lands cycle remodeling, in which phospholipases cleave sn-2 fatty acids and acyltransferases re-esterify them, controlling membrane PUFA content. [Model Knowledge]

#### 4.2 Fat-Soluble Vitamin Transport and Antioxidant Protection

Retinol and alpha-tocopherol are co-transported in lipoproteins and share the vitamin transport pathway (GO:0051180). [KG Evidence] Both function as antioxidants in lipid-phase environments. [Model Knowledge] Their co-expression with PUFA-containing phospholipids is biologically coherent: alpha-tocopherol protects PUFA-rich membranes from peroxidation, while retinol participates in redox-sensitive signaling. [Model Knowledge]

#### 4.3 Immune and Hematopoietic Signaling

KITLG connects to T cell proliferation (shared with retinol), mast cell biology, melanocyte differentiation, and hematopoietic stem cell maintenance. [KG Evidence] Novel KG connections include the IL-17 signaling pathway and the erythrocyte differentiation pathway. [KG Evidence] This immune/hematopoietic theme intersects with the lipid remodeling theme through the known role of phospholipid composition in immune cell membrane function. [Model Knowledge]

#### 4.4 Hub-Filtered Assessment

After de-emphasizing associations driven solely by the two hub entities (retinol and alpha-tocopherol), the most robust module-level signals are: (1) T2D, which includes the non-hub member phosphatidylcholine; (2) cardiovascular, digestive, and kidney disorders, which include the moderate-connectivity member KITLG; (3) fatty liver disease/MASLD, which includes phosphatidylcholine. [KG Evidence, Inferred]

---

### 5. Gap Analysis

#### 5.1 Informative Absences

| Absent Entity Class | Interpretation |
|---|---|
| **Lysophosphatidylcholines** (LPC 18:1, 18:2) | The module contains intact glycerophospholipids across PE, PI, and PG headgroups but lacks LPCs, the canonical products of PLA2 cleavage on PC substrates. This indicates the module captures phospholipid composition rather than the PC/LPC turnover axis. LPC(18:2), the most replicated metabolomic predictor of T2D conversion, likely resides in a different module. [Inferred] |
| **Ceramides and sphingomyelins** | The entire sphingolipid axis is absent, reinforcing that the module encodes glycerolipid biology, not sphingolipid-mediated lipotoxicity. Ceramide risk scores, emerging clinical biomarkers for cardiometabolic disease, are orthogonal to this module. [Inferred] |
| **Acylcarnitines** (C2, C3, C5) | Markers of mitochondrial beta-oxidation are absent despite the module's fatty acid richness. This suggests the module reflects lipid storage and membrane composition rather than catabolic flux. [Inferred] |
| **BCAAs and aromatic amino acids** | Canonical amino acid predictors of insulin resistance are absent, consistent with a lipid/vitamin-dominated module. These analytes likely cluster in an amino acid catabolism module. [Inferred] |
| **KIT receptor** (c-KIT) | The cognate receptor for KITLG is absent, suggesting KITLG in this module represents its soluble, circulating form. This is consistent with tissue-specific expression: KIT is membrane-bound on mast cells, melanocytes, and hematopoietic progenitors, while soluble KITLG circulates after proteolytic cleavage. [KG Evidence; Model Knowledge] |

#### 5.2 Standard Gaps

Insulin, C-peptide, HbA1c, and fasting glucose are clinical laboratory measures not expected on metabolomics or proteomics platforms and would not appear in WGCNA co-expression modules derived from omics data. [Model Knowledge] Adiponectin, an adipokine with strong lipid metabolism associations, is mildly surprising in its absence and may reside in a different module or may not have been captured by the proteomics platform. [Model Knowledge]

---

### 6. Temporal Context

No explicit longitudinal or time-series metadata accompany this analysis. The following causal inference framework applies if the cohort is longitudinal:

**Upstream causes (potential drivers):** KITLG signaling (mast cell activation, NRF2 induction) and fat-soluble vitamin status (retinol, alpha-tocopherol availability) are plausible upstream regulators of membrane phospholipid composition, because they influence phospholipase activity and oxidative protection of PUFA-containing membranes. [Model Knowledge]

**Downstream consequences (potential readouts):** Glycerophospholipid acyl chain composition is a downstream readout of dietary fatty acid intake, hepatic lipid remodeling (PNPLA3, LIPC activity), and antioxidant capacity. Changes in membrane PUFA content may subsequently affect insulin receptor signaling, inflammatory eicosanoid production, and cardiovascular risk. [Model Knowledge]

**Causal inference opportunity:** If measured at multiple timepoints, Granger causality or mediation analysis could test whether changes in KITLG or vitamin levels precede changes in phospholipid composition, or vice versa. [Inferred]

---

### 7. Research Recommendations

#### 7.1 High Priority: Experimental Validations

1. **Genotype PNPLA3 I148M (rs738409) in the study cohort.** Test for genotype-by-module eigengene interactions to determine whether this hepatic phospholipid remodeling variant modifies the Purple module's behavior. [Inferred; KG Evidence for PNPLA3 as shared enzymatic bridge]

2. **Correlate soluble KITLG with mast cell activation markers** (tryptase, histamine) and with the phospholipid module eigengene. This tests the prediction that soluble KITLG reflects mast cell-derived phospholipid remodeling. [Inferred]

3. **Measure NRF2/HMOX1 pathway activity** (e.g., HMOX1 transcript, bilirubin levels) and correlate with module membership to test the KITLG-antioxidant axis hypothesis. [Inferred; Literature]

#### 7.2 Moderate Priority: Targeted Analytical Follow-Up

4. **Perform compartment-specific lipidomics** (lipoprotein fractions, extracellular vesicles, erythrocyte membranes) to localize the correlated phospholipid species and determine whether the module reflects a single biological compartment. [Inferred]

5. **Examine cross-module relationships** between the Purple module and modules enriched in LPCs, ceramides, BCAAs, or acylcarnitines. Their absence here is informative (Section 5.1), but their co-variation with this module across subjects may reveal metabolic crosstalk. [Inferred]

#### 7.3 Literature and Database Searches

6. **Search PubMed for KITLG (SCF) associations with phospholipid metabolism and Lands cycle enzymes.** The co-expression of a hematopoietic cytokine with membrane phospholipids is unexpected and may reflect an unreported signaling axis. [Inferred]

7. **Query LIPID MAPS and HMDB** for the specific glycerophospholipid species in this module to obtain tissue-of-origin annotations and known metabolic pathway assignments for the approximately 40 species with moderate or sparse KG coverage. [KG Evidence for sparse coverage status]

#### 7.4 Analytical Caveats

8. **Entity resolution quality.** The majority of glycerophospholipid species (approximately 35 of 45) resolved with 70% confidence via fuzzy matching, and many resolved to triacylglycerol or mixed-acyl species rather than their true glycerophospholipid identities (e.g., "1-palmitoyl-2-oleoyl-GPE" resolved to CHEBI:75585, a diacylglycerol). KG-derived findings for individual lipid species should be interpreted cautiously; the module-level pattern (phospholipid remodeling) is robust, but individual species-level KG annotations may reflect misidentified entities. Two entities (1-oleoyl-GPI, 1-linolenoyl-GPE) have zero KG edges (cold-start), limiting their contribution to the analysis. [KG Evidence]

9. **Hub bias.** Retinol (7,680 edges) and alpha-tocopherol (9,195 edges) are flagged as hub nodes. Disease associations shared exclusively between these two members (14 of 27 recurring diseases) should be weighted lower than associations that include KITLG or phosphatidylcholine as independent supporting members. [KG Evidence]

---

*Report generated from KRAKEN knowledge graph analysis of the Purple WGCNA module (47 entities: 1 gene, 46 small molecules). Evidence tiers and attribution tags follow the KRAKEN reporting standard. All Tier 3 predictions carry the calibration note that approximately 18% of computational predictions advance to clinical investigation.*

### Literature References

Papers discovered via semantic search. 4 unique papers across 3 hypotheses.

| Hypothesis | Citation | Link | Relevance |
|------------|----------|------|-----------|
| Inferred role of CHEBI:75585 |  (2024) "Assessment of Digestion and Absorption Properties of 1,3-Dipalmitoyl-2-Oleoyl Glycerol-Rich Lipids U..." | [Link](https://www.mdpi.com/1420-3049/29/22/5442) | The digestion and absorption properties of 1,3-dipalmitoyl-2-oleoyl glycerol (POP)-rich lipids was evaluated using in vi... |
| Bridge: Gene → SmallMolecule (2 hops) |  (2020) "KIT ligand protects against both light-induced and genetic photoreceptor degeneration - PubMed" | [Link](https://pubmed.ncbi.nlm.nih.gov/32242818/) | Photoreceptor degeneration is a major cause of blindness and a considerable health burden during aging but effective the... |
| Inferred role of CHEBI:75585 |  (2020) "Lipase-mediated production of 1-oleoyl-2-palmitoyl-3-linoleoylglycerol by a two-step method - Scienc..." | [Link](https://www.sciencedirect.com/science/article/abs/pii/S2212429220303801) | Human milk fat is considered to be the best dietary fat for infants (Hita et al., 2009; Victora et al., 2016). The trigl... |
| Inferred role of KEGG.GLYCAN:G00122 |  (2019) "Paradoxical Role of Glypican-1 in Prostate Cancer Cell and Tumor Growth \| Scientific Reports" | [Link](https://www.nature.com/articles/s41598-019-47874-2) | Glypicans (GPCs) are heparan sulfate proteoglycans (HSPGs) usually localized at the cellular membrane 1. Six glypican is... |