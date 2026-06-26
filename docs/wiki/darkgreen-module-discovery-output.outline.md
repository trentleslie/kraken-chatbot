# Darkgreen Module Run: Discovery Output (20-analyte, dev, 2026-06-23)

This document presents the discovery output of a full-module run of the **Darkgreen** WGCNA module through the Kraken 12-node discovery pipeline, executed on the dev environment on 2026-06-23 (commit `813fc3f`) with BioMapper entity resolution enabled. It is one of a per-module series over the Frailty multi-omics WGCNA modules; see the [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm).

## Run Provenance

The run submitted 20 named analytes, parsed 21 at intake, and resolved 21 distinct entities (3 fuzzy, 17 biomapper, 1 exact) to 21 distinct CURIEs. Triage classified 3 well-characterized, 8 moderate, 9 sparse, and 1 cold-start, with 0 edge-count measurement failures. Downstream analysis produced 454 direct-KG findings, 36 cold-start findings, 7 biological themes, 0 cross-entity bridges (0 evidence-grounded), and 71 hypotheses supported by 29 literature references. Synthesis emitted a 23720-character report. The run completed in approximately 676.0 s of wall-clock time (status complete, 41 errors). All counts are taken from the reducer-merged pipeline state.

| Stage | Result |
|---|---|
| Submitted | 20 named analytes |
| Intake | 21 parsed |
| Entity resolution | 21 resolved (3 fuzzy, 17 biomapper, 1 exact) to 21 distinct CURIEs |
| Triage | 3 well-characterized, 8 moderate, 9 sparse, 1 cold-start (0 measurement failures) |
| Direct KG | 454 findings |
| Cold-start | 36 findings, 4 skipped |
| Pathway enrichment | 7 biological themes |
| Integration | 0 bridges (0 evidence-grounded) |
| Literature grounding | 29 papers |
| Synthesis | 71 hypotheses, 23720-character report |
| Run total | ~676.0 s wall-clock, status complete, 41 errors |

## Related

- Companion run metrics: [Darkgreen Module Run: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/darkgreen-module-run-pipeline-performance-report-20-analyte-dev-2026-06-23-3rOl51fryq)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---



## Darkgreen WGCNA Module: Caffeine Metabolism and Xenobiotic Disposition

---

### 1. Executive Summary

The Darkgreen WGCNA module encodes a coherent caffeine metabolism signature comprising the parent compound, all three primary demethylation products (paraxanthine, theobromine, theophylline), six downstream methylxanthine and methyluric acid intermediates, and two ring-opened uracil derivatives, collectively mapping to a single canonical pathway (SMPDB:SMP0000028) with eight of twenty members assigned [KG Evidence]. Three non-xanthine metabolites (trigonelline, quinate, N-(2-furoyl)glycine) and four phase II sulfate conjugates (o-cresol sulfate, 3-methyl catechol sulfate, 3-hydroxypyridine sulfate, cyclo(pro-val)) co-segregate with this core, implicating coffee consumption as the dominant upstream exposure and revealing supplementary gut-microbial and hepatic conjugation processes. Disease recurrence analysis identifies colorectal cancer (seven members), inherited asthma susceptibility (four members), and cardiometabolic disorders (hypertension, coronary artery disease, obesity; three members each) as the most broadly shared disease associations across the module [KG Evidence].

---

### 2. Key Findings (Tier 1 to 2)

#### 2.1 The Module is a Near-Complete Map of the Caffeine Metabolic Pathway

Eight of the twenty module members participate directly in the canonical Caffeine Metabolism pathway (SMPDB:SMP0000028): caffeine, theophylline, theobromine, paraxanthine, 7-methylxanthine, 1,7-dimethylurate, 3,7-dimethylurate, and 5-acetylamino-6-formylamino-3-methyluracil (AFMU) [KG Evidence]. The remaining xanthine-scaffold metabolites (1,3-dimethylurate, 1,3,7-trimethylurate, 3-methylxanthine, 1-methylxanthine, 5-acetylamino-6-amino-3-methyluracil (AAMU)) are known constituents of the same biotransformation cascade, although they carry sparse KG annotation (3 to 15 edges each) [KG Evidence]. This near-complete pathway coverage confirms that the module captures inter-individual variation in CYP1A2-mediated N-demethylation, xanthine oxidase (XDH/XO)-catalysed C-8 oxidation, and NAT2-mediated acetylation of caffeine [Model Knowledge].

The presence of both AFMU and AAMU is particularly informative: AFMU undergoes non-enzymatic deformylation to AAMU, and the AFMU:1-methylxanthine ratio is a validated phenotypic probe for NAT2 acetylator status [Model Knowledge]. The co-expression of these two analytes within the module indicates that NAT2-dependent variation is a detectable source of metabolite covariance in this cohort.

#### 2.2 Coffee-Associated Non-Xanthine Metabolites Confirm Dietary Origin

Trigonelline (N'-methylnicotinate) is a pyridinium alkaloid abundant in coffee beans; its participation in N-methylnicotinate transport [KG Evidence] and its co-expression with the xanthine metabolites corroborates coffee intake as the shared upstream exposure. Quinate, a cyclitol produced by hydrolysis of chlorogenic acid (the dominant coffee polyphenol), provides additional confirmatory evidence [Model Knowledge]. N-(2-furoyl)glycine, annotated to the fucose catabolic process [KG Evidence], is a recognised urinary marker of coffee and furfural-containing food consumption [Model Knowledge].

#### 2.3 Disease Associations: Colorectal Cancer and Respiratory Disease Dominate

Colorectal cancer is the most broadly associated disease across the module (seven members: trigonelline, paraxanthine, theobromine, 7-methylxanthine, 1,7-dimethylurate, 3,7-dimethylurate, N-(2-furoyl)glycine; strongest evidence: curated) [KG Evidence]. Inherited susceptibility to asthma involves four members (paraxanthine, theobromine, 7-methylxanthine, 1,7-dimethylurate; curated) [KG Evidence], and asthma itself is associated with three (caffeine, theophylline, theobromine; curated) [KG Evidence]. COPD is associated with three members (paraxanthine, caffeine, theophylline; text-mined) [KG Evidence]. These respiratory associations are pharmacologically coherent: caffeine and theophylline are established bronchodilators that antagonise adenosine receptors and inhibit phosphodiesterases [KG Evidence].

Cardiometabolic associations recur across the module. Hypertensive disorder (caffeine, theophylline, trigonelline; curated), coronary artery disorder (caffeine, theophylline, theobromine; curated), and obesity disorder (caffeine, theophylline, theobromine; curated) each involve three members [KG Evidence]. Type 2 diabetes mellitus (caffeine, theobromine; text-mined) and kidney disorder (caffeine, theophylline, trigonelline; curated) are also represented [KG Evidence].

#### 2.4 Gene Interaction Landscape

Caffeine interacts with adenosine receptors ADORA1, ADORA2A, ADORA2B, and ADORA3 [KG Evidence], which collectively form the strongest gene-level biological theme connecting eight module members [KG Evidence]. Phosphodiesterase family members (PDE4B, PDE10A, PDE4D, PDE2A, PDE3A, PDE5A, PDE1C, PDE1A, PDE9A, PDE11A) constitute a second major target class for caffeine [KG Evidence]. CYP1A2 (the primary caffeine-metabolising enzyme), CYP3A4, and CYP2D6 are documented caffeine interactors in the KG [KG Evidence]; the absence of CYP1A2 expression data from the module itself reflects a multi-omics integration gap rather than a lack of biological relevance (see Gap Analysis below).

Notable novel ("hidden gem") gene interactions for caffeine include HSD11B1 (11-beta-hydroxysteroid dehydrogenase 1; cortisol metabolism), HMGCR (HMG-CoA reductase; cholesterol biosynthesis), PKD1 (polycystin 1; polycystic kidney disease), and CTNNB1 (beta-catenin; Wnt signalling) [KG Evidence]. These interactions suggest mechanistic links between caffeine exposure and the observed cardiometabolic and renal disease associations.

#### 2.5 Pharmacological Classification

The module's well-characterised members share multiple chemical roles: bronchodilator, phosphodiesterase inhibitor, vasodilator agent, adenosine receptor antagonist, and diuretic (connecting seven input entities) [KG Evidence]. This convergent pharmacological profile positions the module as a readout of both exogenous methylxanthine exposure and its downstream physiological consequences.

---

### 3. Novel Predictions (Tier 3)

All Tier 3 predictions are inferred via semantic similarity to annotated analogues and require experimental validation. Approximately 18% of computational predictions in biomedical knowledge graphs progress to clinical investigation; the estimates below should be calibrated accordingly.

#### 3.1 o-Cresol Sulfate as a Candidate Uremic Toxin Marker

**Prediction**: o-Cresol sulfate (CHEBI:133089) is correlated with chronic kidney disease and contributes to cardiovascular disease, analogous to its positional isomer p-cresol sulfate [Inferred].

**Structural logic chain**: o-Cresol sulfate exhibits 87% vector similarity to p-cresol sulfate (CHEBI:82914). p-Cresol sulfate is correlated with chronic kidney disease (MONDO:0012451) and contributes to cardiovascular disease (MONDO:0003634) [KG Evidence via analogue]. Both compounds share the same biosynthetic origin: gut microbial decarboxylation of tyrosine to cresol, followed by hepatic sulfation by SULT1A1 [Model Knowledge]. The ortho-isomer may exhibit comparable protein binding and endothelial toxicity.

**Validation step**: Measure o-cresol sulfate concentrations in CKD patient cohorts (e.g., via EUTox database or targeted LC-MS/MS); test endothelial cell viability and reactive oxygen species generation in vitro. The inferred interaction with NCBIGene:50507 (likely NOX4) suggests an oxidative stress mechanism amenable to gene knockdown experiments [Inferred].

**Calibration**: ~18% estimated progression probability. The strong structural analogy to p-cresol sulfate and shared metabolic origin elevate this prediction above baseline.

#### 3.2 1,3,7-Trimethylurate as a Protective Factor in Erectile Dysfunction

**Prediction**: 1,3,7-Trimethylurate (CHEBI:132940) has a protective association with erectile dysfunction [Literature].

**Structural logic chain**: 1,3,7-Trimethylurate is a terminal oxidative metabolite of caffeine produced by XDH/XO [KG Evidence; Literature (Ferrero JL et al., 1983)]. A bidirectional Mendelian randomisation study identified 1,3,7-trimethylurate as protective against erectile dysfunction (OR 0.85, 95% CI 0.73 to 0.99, P = 0.037) [Literature (Xu R et al., 2025)]. The KG currently lacks a direct edge between this metabolite and caffeine, but the metabolic relationship is confirmed by urinary caffeine metabolite profiling studies [Literature (Determination of Urinary Caffeine Metabolites, 2019)].

**Validation step**: Replicate the Mendelian randomisation finding in an independent GWAS cohort; assess whether the association is mediated by improved endothelial nitric oxide signalling (consistent with the module's vasodilation and cAMP/calcium signalling themes).

**Calibration**: ~18% estimated progression probability. Grounded Mendelian randomisation evidence strengthens this prediction relative to purely computational inferences.

#### 3.3 Cyclo(pro-val) as a Potential CCK Receptor-Binding Peptide

**Prediction**: The cyclic dipeptide cyclo(pro-val) (resolved to PUBCHEM.COMPOUND:53321822) may bind cholecystokinin-related targets (NCBIGene:5021) [Inferred].

**Structural logic chain**: The resolved entity exhibits 89% similarity to cyclo(CYIQNCPLG) (PUBCHEM.COMPOUND:53321821), which binds NCBIGene:5021 [KG Evidence via analogue]. Cyclic dipeptides (diketopiperazines) are known coffee roasting products and have documented neuroactive properties [Literature (Focus on cyclo(His-Pro), 2007)]. The entity resolution for this analyte is uncertain (90% confidence, mapped to a much larger cyclic peptide), and this prediction is contingent on the true identity of the measured species.

**Validation step**: Confirm the molecular identity of the measured analyte by high-resolution MS/MS fragmentation; if confirmed as cyclo(pro-val), test binding to CCK receptors in competitive displacement assays.

**Calibration**: ~18% estimated progression probability. Entity resolution uncertainty and single-analogue support reduce confidence substantially. This prediction warrants identity confirmation before biological follow-up.

#### 3.4 3-Methyl Catechol Sulfate: Polyphenol Metabolism Marker

**Prediction**: 3-Methyl catechol sulfate (PUBCHEM.COMPOUND:102232874; cold-start, zero KG edges) is chemically related to the catechin sulfate class (CHEBI:72010) and to catechins broadly (CHEBI:23053) [Inferred].

**Structural logic chain**: This compound exhibits 94% similarity to catechin 3'-sulfate (CHEBI:149590) and 93% similarity to catechin sulfate (MESH:C466639), both of which are linked to catechins (CHEBI:23053) [KG Evidence via analogue]. Despite the name similarity, catechol (1,2-dihydroxybenzene) and catechin (flavan-3-ol) are structurally distinct; 3-methyl catechol sulfate is more likely a gut-microbial metabolite of caffeic acid or catechol-containing compounds than a true catechin derivative [Model Knowledge].

**Validation step**: Perform structural comparison using Tanimoto fingerprints; trace biosynthetic origin via isotope labelling studies with coffee polyphenols in gnotobiotic models.

**Calibration**: ~18% estimated progression probability. Chemical classification inferences carry higher intrinsic validity than disease predictions but require confirmation of the compound's true biosynthetic origin.

---

### 4. Biological Themes

#### 4.1 Adenosine Receptor Antagonism as the Unifying Pharmacological Mechanism

The adenosine receptor gene family (ADORA1, ADORA2A, ADORA2B, ADORA3) constitutes the strongest gene-level theme, connecting eight module members [KG Evidence]. Caffeine and theophylline are established competitive antagonists at these receptors [KG Evidence], and their downstream metabolites (paraxanthine, theobromine) retain partial antagonist activity [Model Knowledge]. This theme mechanistically explains the module's associations with bronchodilation, cardiovascular activity alteration, and neuroactive ligand-receptor interaction [KG Evidence].

#### 4.2 Phosphodiesterase Inhibition and Cyclic Nucleotide Signalling

Caffeine and theophylline participate in cAMP-mediated and calcium-mediated signalling [KG Evidence]. The extensive PDE interaction network (PDE1A, PDE1C, PDE2A, PDE3A, PDE4B, PDE4D, PDE5A, PDE9A, PDE10A, PDE11A) [KG Evidence] indicates that cyclic nucleotide hydrolysis is a second major pharmacological axis of this module. This theme connects to the observed disease associations with asthma (PDE4 inhibition is anti-inflammatory) and cardiovascular disorders (PDE3/5 inhibition affects vascular tone) [Inferred].

#### 4.3 Hub-Filtered Anatomical and Cellular Localization

All anatomical entities (blood, urine, feces, liver, kidney) and the cellular component cytoplasm (GO:0005737) connecting module members are flagged as hubs [KG Evidence]. These high-connectivity nodes reflect the expected tissue distribution and elimination routes of xenobiotic metabolites and should be de-emphasised as informative signals. Their presence is consistent with, but not specific to, caffeine metabolism.

#### 4.4 Fucose Catabolism: An Unexpected Shared Theme

N-(2-furoyl)glycine and 3-hydroxypyridine sulfate both annotate to the fucose catabolic process (GO:0019317) [KG Evidence]. This connection is unexpected for a caffeine-centric module. N-(2-furoyl)glycine is a downstream product of furfural metabolism (furfural derives from pentose dehydration during coffee roasting) [Model Knowledge], and 3-hydroxypyridine sulfate is a pyridine derivative potentially originating from trigonelline degradation during roasting [Model Knowledge]. The shared GO annotation likely reflects co-membership in gut microbial carbohydrate processing rather than a direct biological connection between these analytes and fucose catabolism per se [Inferred].

---

### 5. Gap Analysis

Using the Open World Assumption, absence of an entity from this module indicates that it was either unstudied, undetected, or uncorrelated with the caffeine-specific cluster; absence does not imply non-existence of the biological relationship.

#### 5.1 Informative Absences

| Absent Entity | Interpretation | Significance |
|---|---|---|
| **CYP1A2** | Gene/protein data not included in metabolomics-only study; this is a multi-omics integration gap [KG Evidence; Model Knowledge] | CYP1A2 genotype (e.g., rs762551) would enable fast/slow metabolizer stratification and substantially enhance interpretation of metabolite ratio variation |
| **NAT2** | Same multi-omics gap; NAT2 acetylator status is directly assessable from the AFMU:1-methylxanthine ratio already present in the module [Model Knowledge] | ~50% of European-ancestry individuals are slow acetylators; genotype data would partition the module's variance |
| **Uric acid** | Endogenous purine catabolism sources overwhelm exogenous caffeine-derived contribution; variance determined by renal function and genetics, not coffee intake [Model Knowledge] | This absence confirms the module's specificity for exogenous (dietary) compounds, a biologically meaningful characteristic |
| **Hippuric acid** | Multiple non-coffee dietary sources (fruits, vegetables, benzoate preservatives) dilute correlation with caffeine-specific cluster [Model Knowledge] | Reflects signal specificity rather than platform limitation |
| **1-Methyluric acid** (and all monomethyluric acids) | Likely below detection in plasma/serum due to rapid renal clearance; dimethyluric acids ARE detected, indicating platform chemistry is adequate [KG Evidence; Model Knowledge] | Suggests a compartment/concentration effect: the study matrix (blood) favors dimethyl over monomethyl uric acids |
| **Chlorogenic acid** | Rapidly hydrolysed in vivo; presence captured indirectly through quinate [Model Knowledge] | Absence reflects biotransformation kinetics, not non-measurement |

#### 5.2 Multi-Omics Integration Gaps

The absence of CYP1A2, NAT2, and XDH/XO from the module represents the most consequential analytical limitation. All three enzymes directly catalyse the reactions whose substrates and products define this module [Model Knowledge]. Integration of genomic (CYP1A2*1F, NAT2 acetylator alleles) or proteomic data would enable: (i) metabolizer phenotype stratification, (ii) refinement of metabolite ratio-based pharmacogenomic indices, and (iii) identification of gene-by-environment interactions with the disease associations observed.

---

### 6. Temporal Context

This module does not derive from a longitudinal study design, and the WGCNA co-expression matrix captures cross-sectional covariance. The following causal architecture can nonetheless be inferred from biochemical first principles:

**Upstream cause**: Coffee consumption is the singular exogenous exposure that generates the entire xanthine metabolite cascade. Caffeine is the parent compound; all demethylated, oxidised, and acetylated derivatives are downstream products [Model Knowledge].

**Metabolic cascade directionality**: Caffeine → paraxanthine / theobromine / theophylline (CYP1A2 N-demethylation) → monomethylxanthines (further N-demethylation) → methyluric acids (XDH/XO C-8 oxidation) → AFMU (NAT2 acetylation of 1-methylxanthine) → AAMU (non-enzymatic deformylation) [Model Knowledge; Literature (Determination of Urinary Caffeine Metabolites, 2019)].

**Causal inference opportunity**: The metabolite ratios within this module (e.g., paraxanthine:caffeine for CYP1A2 activity; AFMU:1-methylxanthine for NAT2 activity) are established pharmacogenomic indices [Model Knowledge]. If genotype data are available, Mendelian randomisation using CYP1A2 or NAT2 genetic instruments could distinguish causal effects of specific metabolites from confounding by total coffee intake in the observed disease associations.

---

### 7. Research Recommendations

#### 7.1 High Priority: Multi-Omics Integration

1. **Genotype CYP1A2 and NAT2** in the study cohort. CYP1A2*1F (rs762551) determines caffeine metabolizer status; NAT2 slow-acetylator alleles modulate AFMU/AAMU levels. These genotypes would decompose the module's variance into genetic and environmental (coffee intake) components and enable Mendelian randomisation analyses for the disease associations identified [Inferred].

2. **Compute pharmacogenomic metabolite ratios** from existing data: paraxanthine:caffeine (CYP1A2 activity index), AFMU:1-methylxanthine (NAT2 acetylator phenotype), (1-methyluric acid + 1-methylxanthine + AFMU):total caffeine metabolites (CYP1A2 flux) [Model Knowledge]. Even without genotype data, these ratios may stratify participants into metabolizer groups.

#### 7.2 High Priority: Disease Association Validation

3. **Investigate the colorectal cancer association** (seven members). Determine whether the association reflects a protective effect of coffee consumption (consistent with IARC and WCRF conclusions) or metabolite-specific mechanisms. Prioritise N-(2-furoyl)glycine and trigonelline, which are the non-xanthine members in this disease cluster and may mark distinct coffee-derived exposures [KG Evidence; Model Knowledge].

4. **Follow up the 1,3,7-trimethylurate and erectile dysfunction Mendelian randomisation finding** (OR 0.85, P = 0.037) [Literature (Xu R et al., 2025)] in the current cohort by testing the association of this metabolite with relevant cardiometabolic or vascular endpoints.

#### 7.3 Moderate Priority: Novel Metabolite Characterisation

5. **Confirm the identity of cyclo(pro-val)** by targeted MS/MS fragmentation. The entity resolution (mapped to a large cyclic peptide, PUBCHEM.COMPOUND:53321822, at 90% confidence) is likely incorrect for a WGCNA metabolomics module; the true analyte is probably the diketopiperazine cyclo(Pro-Val), a known coffee roasting product [Model Knowledge]. Identity clarification is prerequisite to any biological inference.

6. **Characterise 3-methyl catechol sulfate** (zero KG edges). This cold-start metabolite requires structural confirmation and biosynthetic pathway determination. Its co-expression with the caffeine module suggests it derives from coffee polyphenol metabolism via gut microbial catechol production and hepatic sulfation [Inferred].

7. **Profile o-cresol sulfate** in relation to renal function markers. Its structural similarity to p-cresol sulfate (87%) and shared gut-microbial origin support its candidacy as a uremic toxin marker [Inferred]. Measurement in CKD cohorts would test this prediction directly.

#### 7.4 Lower Priority: Literature and Database Curation

8. **Populate KG edges** for sparse-coverage entities (1,3,7-trimethylurate: 3 edges; 1-methylxanthine: 4 edges; 3-methylxanthine: 7 edges). These are well-characterised biochemical intermediates in caffeine metabolism whose sparse KG representation reflects curation gaps rather than genuine novelty [Inferred]. The literature evidence (Ferrero JL et al., 1983; Determination of Urinary Caffeine Metabolites, 2019) confirms their pathway membership [Literature].

9. **Resolve entity annotation discrepancies**: theophylline is categorised as "OrganismTaxon" in the KG (CHEBI:28177), which is a misclassification; it is a small-molecule drug [KG Evidence; Model Knowledge]. The "Metabolites:" header token was resolved to substance P-metabolite 5-11 (UMLS:C1698199) at 70% confidence, which is spurious and should be excluded from downstream analyses [KG Evidence].

---

*Report generated from KRAKEN knowledge graph analysis. All evidence attributions ([KG Evidence], [Literature], [Model Knowledge], [Inferred]) are provided to enable independent assessment of claim provenance. Tier 3 predictions carry an estimated ~18% probability of progressing to clinical validation and require experimental confirmation before incorporation into biological models.*

### Literature References

Papers discovered via semantic search. 5 unique papers across 3 hypotheses.

| Hypothesis | Citation | Link | Relevance |
|------------|----------|------|-----------|
| Inferred role of CHEBI:132940 |  (2023) "7-Methylxanthine Inhibits the Formation of Monosodium Urate Crystals by Increasing Its Solubility" | [Link](https://www.mdpi.com/2218-273X/13/12/1769) | The effects of nine methylxanthines and two methylated UA derivatives on the crystallization of NaU were studied (Figure... |
| Inferred role of UMLS:C1698199 |  (2024) "Cellular metabolism of substance P produces neurokinin-1 receptor peptide agonists with diminished c..." | [Link](https://pubmed.ncbi.nlm.nih.gov/38798270/) | Substance P (SP) is released from sensory nerves in the arteries and heart. It activates neurokinin-1 receptors (NK1Rs)... |
| Inferred role of CHEBI:132940 |  (2019) "Determination of Urinary Caffeine Metabolites as Biomarkers for Drug Metabolic Enzyme Activities" | [Link](https://www.mdpi.com/2072-6643/11/8/1947) | Figure 1 The metabolic pathway of caffeine. " Figure 2 Urinary concentration levels of caffeine and its metabolites in... |
| Inferred role of CHEBI:68443 |  (2025) "Progress in Methylxanthine Biosynthesis: Insights into Pathways and Engineering Strategies" | [Link](https://www.mdpi.com/1422-0067/26/4/1510) | Figure 1 Functions of methylxanthine derivatives. Methylxanthine ... 7-methylxanthine ... -methylxanthine, 1, ... -dimet... |
| Inferred role of CHEBI:132940 |  (2018) "Systematic Structure-Activity Relationship (SAR) Exploration of Diarylmethane Backbone and Discovery..." | [Link](https://www.mdpi.com/1420-3049/23/2/252) | benzbromarone, respectively (IC ... AT1 for ... , respectively). Compound ... . The present study demonstrates that ...... |