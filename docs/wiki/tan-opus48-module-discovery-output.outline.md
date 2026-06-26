# Tan Module Run on Opus 4.8: Discovery Output (43-analyte, dev, 2026-06-24)

This document presents the discovery output of a full-module run of the **Tan** WGCNA module through the Kraken 12-node discovery pipeline, executed on the dev environment on 2026-06-24 (commit `098093e`) with BioMapper entity resolution enabled. The SDK-backed pipeline nodes ran on the Opus 4.8 model rather than the default Sonnet; this run is published for direct model comparison. It is one of a per-module series over the Frailty multi-omics WGCNA modules; see the [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm).

## Run Provenance

The run submitted 43 named analytes, parsed 39 at intake, and resolved 39 distinct entities (38 biomapper, 1 fuzzy) to 37 distinct CURIEs. Triage classified 13 well-characterized, 5 moderate, 21 sparse, and 0 cold-start, with 0 edge-count measurement failures. Downstream analysis produced 577 direct-KG findings, 5 cold-start findings, 0 biological themes, 0 cross-entity bridges (0 evidence-grounded), and 5 hypotheses supported by 4 literature references. Synthesis emitted a 21320-character report. The run completed in approximately 326.0 s of wall-clock time (status complete, 25 errors). All counts are taken from the reducer-merged pipeline state.

| Stage | Result |
|---|---|
| Submitted | 43 named analytes |
| Intake | 39 parsed |
| Entity resolution | 39 resolved (38 biomapper, 1 fuzzy) to 37 distinct CURIEs |
| Triage | 13 well-characterized, 5 moderate, 21 sparse, 0 cold-start (0 measurement failures) |
| Direct KG | 577 findings |
| Cold-start | 5 findings, 16 skipped |
| Pathway enrichment | 0 biological themes |
| Integration | 0 bridges (0 evidence-grounded) |
| Literature grounding | 4 papers |
| Synthesis | 5 hypotheses, 21320-character report |
| Run total | ~326.0 s wall-clock, status complete, 25 errors |

## Related

- Companion run metrics: [Tan Module Run on Opus 4.8: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/tan-module-run-on-opus-48-pipeline-performance-report-43-analyte-dev-2026-06-24-d83ZPQwnDK)
- Model comparison baseline (Sonnet): [Tan Module Run: Discovery Output](https://phwiki.phenoma.ai/doc/tan-module-run-discovery-output-43-analyte-dev-2026-06-23-xSd82cW2It)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---



## Tan WGCNA Module Discovery Report: Adrenal Steroid Sulfation and Extracellular Matrix Remodeling

### 1. Executive Summary

The Tan WGCNA module encodes a dual biological signature comprising (i) adrenal androgen sulfate metabolism (28 steroid sulfate and glucuronide conjugates spanning the DHEA to pregnenolone biosynthetic axis) and (ii) extracellular matrix remodeling coupled with non-canonical immune signaling (MMP12, TIMP4, DCN, PRELP, IL17D, IL1RL2, HAVCR1). [KG Evidence] [Inferred] The co-expression of these two arms converges on tissue compartments where sulfated steroid reservoirs regulate local matrix turnover, most plausibly bone, adipose, and mucosal epithelia; this interpretation is reinforced by the module-level enrichment for bone remodeling genes (TNFRSF11B/OPG, MEPE) and the informative absence of cortisol, testosterone, and canonical adipokines (leptin, adiponectin). [KG Evidence] [Inferred] The module's disease recurrence profile implicates chronic inflammatory and metabolic conditions (depressive disorder, asthma, essential hypertension, irritable bowel syndrome), each shared by six module members, suggesting a systemic steroid-immune interface with clinical relevance across organ systems. [KG Evidence]

---

### 2. Key Findings (Tier 1 to 2)

#### 2.1 Steroid Sulfate Metabolite Cluster

The metabolite compartment of this module is dominated by sulfated and glucuronidated steroid conjugates originating from the adrenal androgen pathway. [KG Evidence] DHEA-S (CHEBI:16814; 588 edges) and androsterone sulfate serve as the best-characterized anchors, with curated disease associations to polycystic ovary syndrome and rheumatoid arthritis. [KG Evidence] The remaining 26 metabolites span mono- and disulfated forms of androstandiol, pregnanediol, androstenediol, and pregnenolone, together with glucuronidated conjugates (androsterone glucuronide, etiocholanolone glucuronide, pregnanediol-3-glucuronide, 11-beta-hydroxyandrosterone glucuronide). [KG Evidence] Most of these conjugates fall into the sparse coverage bucket (1 to 18 edges), indicating that they are under-represented in current knowledge graphs, a pattern consistent with the recent emergence of sulfated steroid metabolomics. [KG Evidence]

#### 2.2 Extracellular Matrix Remodeling Core

Four protein members constitute a coherent ECM remodeling axis. MMP12 (3,017 edges) participates in extracellular matrix organization and disassembly and shares a WikiPathways annotation ("Matrix metalloproteinases," WP129) with TIMP4 (1,551 edges), its endogenous inhibitor. [KG Evidence] DCN (decorin; 2,633 edges) is a small leucine-rich proteoglycan that regulates collagen fibrillogenesis, TGF-beta signaling, and autophagy; PRELP (1,063 edges) is a structurally related proteoglycan contributing to cartilage and connective-tissue integrity. [KG Evidence] DCN and MMP12 share participation in positive regulation of transcription by RNA polymerase II (GO:0045944) and extracellular matrix organization (GO:0030198). [KG Evidence]

#### 2.3 Bone Remodeling Signature

TNFRSF11B (OPG; 3,164 edges) is the decoy receptor for RANKL and a central regulator of osteoclastogenesis. [KG Evidence] MEPE (1,486 edges) encodes matrix extracellular phosphoglycoprotein, a bone mineralization regulator. [KG Evidence] The co-presence of OPG and MEPE without RANKL (TNFSF11) suggests the module captures an anti-resorptive, mineralization-promoting state. [KG Evidence] [Inferred]

#### 2.4 Non-Canonical Immune Signaling

IL17D (1,723 edges), IL1RL2 (1,732 edges), and HAVCR1 (TIM-1; 2,266 edges) represent an unconventional immune/mucosal inflammatory program. [KG Evidence] IL17D is a poorly characterized IL-17 family member linked to innate immune surveillance; IL1RL2 (IL-36 receptor) mediates epithelial inflammatory responses; HAVCR1 participates in T-cell co-stimulation and kidney injury signaling. [KG Evidence] [Model Knowledge] The enrichment for response to stress (GO:0006950; 6 members) and smoking (UMLS:C0037369; 5 members) across these proteins, alongside AGRP, DCN, and TIMP4, indicates a shared exposure-response biology. [KG Evidence]

#### 2.5 AGRP and Energy Homeostasis

AGRP (1,204 edges) functions as a melanocortin receptor antagonist central to feeding behavior regulation. [KG Evidence] Its Tier 1 annotations include neuropeptide signaling pathway, adult feeding behavior, regulation of feeding behavior, response to insulin, adipocytokine signaling pathway, and circadian rhythm. [KG Evidence] AGRP interacts directly with melanocortin receptors MC3R and MC4R and with metabolic regulators FOXO1, NPY, GHRL (ghrelin), and LEPR (leptin receptor). [KG Evidence] Its presence in this module alongside sulfated steroid metabolites rather than canonical appetite-regulatory peptides (leptin, ghrelin) suggests that peripheral circulating AGRP levels track adrenal steroid metabolism rather than central melanocortin tone. [Inferred]

#### 2.6 Module-Level Disease Recurrence

Nine disease associations recur across six module members each (the maximum breadth observed), all supported by curated evidence: depressive disorder, gastroesophageal reflux disease, psoriasis, gastroduodenitis, hiatus hernia, asthma, chronic rhinitis, irritable bowel syndrome, and essential hypertension. [KG Evidence] Five-member recurrences include hypophysitis, schizophrenia, coronary artery disorder, urothelial carcinoma, and necrotizing ulcerative gingivitis. [KG Evidence] The recurrent gene members driving these associations are DCN, AGRP, PRELP, IL17D, MMP12, and LGALS4 (galectin-4; 3,304 edges), indicating that the ECM remodeling and mucosal immune proteins, rather than the metabolites, underpin the disease connectivity. [KG Evidence]

---

### 3. Novel Predictions (Tier 3)

#### 3.1 Sulfated Progesterone Metabolites as Modulators of Insulin Secretion via TRPM3

**Structural logic chain:** The module contains multiple pregnanediol sulfates (5-alpha-pregnan-3-beta,20-alpha-diol disulfate and monosulfate, pregnenediol disulfate, pregnenetriol disulfate) that are structurally analogous to epiallopregnanolone sulfate (PM5S). [KG Evidence] PM5S has been demonstrated to enhance glucose-stimulated insulin secretion (GSIS) via TRPM3 activation in mouse and human islets; PM5S concentrations are reduced in women with gestational diabetes mellitus (GDM) and correlate inversely with fasting plasma glucose. [Literature: "Sulfated Progesterone Metabolites That Enhance Insulin Secretion via TRPM3," 2022] The semantic similarity search identified 5-alpha-pregnane-3-beta,20-alpha-diol monosulfate(1-) at 99% similarity to the sparse-coverage module member 5-alpha-pregnan-3-beta,20-beta-diol monosulfate (CHEBI:133712). [KG Evidence]

**Prediction:** The pregnanediol sulfate cluster in this module may index TRPM3-mediated insulin secretory capacity, and the module's co-expression pattern could serve as a circulating biomarker of beta-cell sulfated steroid exposure.

**Validation step:** Measure GSIS responses to the specific pregnanediol sulfate isomers present in this module (particularly 5-alpha-pregnan-3-beta,20-alpha-diol disulfate) in isolated human islets with and without TRPM3 inhibition (isosakuranetin).

**Calibration:** Approximately 18% of computational predictions of this type progress to clinical investigation; this prediction is strengthened by the availability of direct experimental evidence for a structural analogue.

#### 3.2 Androstenediol Sulfates as Biomarkers of Peripheral Androgen Activation (PCOS and Hirsutism)

**Structural logic chain:** The module contains multiple androstenediol disulfate and monosulfate isomers (3-beta,17-beta and 3-alpha,17-alpha forms) alongside androstandiol sulfates and glucuronidated testosterone metabolites (androsterone glucuronide, etiocholanolone glucuronide). [KG Evidence] 5-alpha-androstane-3-alpha,17-beta-diol 17-glucuronide (3-alpha-diol G) is an established biomarker of peripheral 5-alpha-reductase activity in PCOS and idiopathic hirsutism. [Literature: "Development and validation of an LC-MS/MS assay for serum 5-alpha-androstane-3-alpha,17-beta-diol 17-glucuronide," 2026] The co-expression of sulfated precursors with glucuronidated end-products suggests coordinated Phase II steroid conjugation.

**Prediction:** The ratio of sulfated to glucuronidated androgen metabolites within this module may distinguish adrenal-origin hyperandrogenism (sulfated forms predominant) from peripheral 5-alpha-reductase-driven hyperandrogenism (glucuronidated forms predominant).

**Validation step:** Quantify sulfated-to-glucuronidated androgen metabolite ratios in a PCOS cohort stratified by adrenal versus ovarian androgen excess (using DHEA-S and testosterone levels as classifiers).

**Calibration:** Approximately 18% of computational predictions progress to clinical investigation; this prediction benefits from established clinical utility of 3-alpha-diol G but extends to novel sulfated isomers.

#### 3.3 OPG-MEPE Co-Expression as an Osteoblast-Protective Signature Linked to Adrenal Androgens

**Structural logic chain:** TNFRSF11B (OPG) and MEPE co-express in this module alongside DHEA-S and pregnenolone sulfate. [KG Evidence] DHEA and its sulfated conjugate are known to promote osteoblast differentiation and inhibit osteoclastogenesis, effects that phenocopy OPG action. [Model Knowledge] The absence of RANKL from the module while OPG is present suggests an anti-resorptive state. [KG Evidence]

**Prediction:** The Tan module captures a circulating signature of DHEA-S-driven bone anabolism, wherein adrenal androgen sulfates co-regulate OPG/MEPE expression to favor bone formation over resorption.

**Validation step:** Correlate serum DHEA-S and OPG concentrations with bone mineral density (DXA) and bone turnover markers (P1NP, CTX) in an age-stratified cohort; test whether the module eigengene predicts fracture risk independently of individual analytes.

**Calibration:** Approximately 18% of computational predictions progress to clinical investigation. No direct literature grounding was fetched for this specific combination; the prediction relies on established but separate lines of evidence for DHEA-S effects on bone and OPG biology. [Model Knowledge] [Inferred]

#### 3.4 LGALS4 as a Mucosal Immune Integrator Connecting GI Disease Associations

**Structural logic chain:** LGALS4 (galectin-4; 3,304 edges, highest connectivity in the module) recurs in six disease associations including gastroesophageal reflux disease, gastroduodenitis, irritable bowel syndrome, and psoriasis. [KG Evidence] Galectin-4 is expressed in intestinal epithelium and regulates epithelial cell adhesion, lipid raft dynamics, and mucosal immune homeostasis. [Model Knowledge] Its co-expression with IL17D, IL1RL2 (IL-36R), and HAVCR1 positions it as a potential hub connecting mucosal inflammation to steroid sulfate metabolism.

**Prediction:** LGALS4 circulating levels may serve as a biomarker of mucosal barrier integrity in the context of adrenal androgen dysregulation, linking the GI disease recurrence pattern to the steroid metabolite cluster.

**Validation step:** Measure plasma galectin-4 in IBS and IBD cohorts alongside sulfated steroid panels; assess correlation with intestinal permeability markers (zonulin, LPS-binding protein).

**Calibration:** Approximately 18% of computational predictions progress to clinical investigation. LGALS4 is a high-connectivity node (3,304 edges), warranting caution that some disease associations may reflect hub-driven annotation bias rather than specific biology. [KG Evidence]

---

### 4. Biological Themes

#### 4.1 Adrenal Androgen Sulfation Axis

The dominant theme is the coordinated sulfation and glucuronidation of adrenal-origin C19 and C21 steroids. [KG Evidence] [Inferred] The module captures intermediates from pregnenolone through DHEA to androstandiol and androstenediol, conjugated with sulfate (mono- and di-) and glucuronate. The informative absence of cortisol/cortisone indicates selective representation of the CYP17A1/SULT2A1 (androgen) arm over the CYP11B1 (glucocorticoid) arm of adrenal steroidogenesis. [Inferred] The absence of unconjugated testosterone and estradiol further suggests the module indexes the sulfated steroid reservoir rather than active hormonal signaling, consistent with intracrine biology wherein steroid sulfatase (STS) mediates local activation in target tissues. [Inferred]

#### 4.2 ECM Turnover and Tissue Remodeling

MMP12, TIMP4, DCN, and PRELP collectively represent a balanced matrix degradation and maintenance program. [KG Evidence] MMP12 is an elastase implicated in emphysema, atherosclerosis, and adipose tissue remodeling; TIMP4 is its endogenous inhibitor. [KG Evidence] [Model Knowledge] DCN and PRELP are small leucine-rich proteoglycans that organize collagen fibrils and sequester growth factors (TGF-beta, EGF). [KG Evidence] The co-expression of these ECM regulators with steroid sulfates may reflect steroid-dependent regulation of connective-tissue turnover in bone, skin, or adipose compartments. [Inferred]

#### 4.3 Non-Classical Inflammation

The immune component (IL17D, IL1RL2, HAVCR1) is notable for the absence of canonical pro-inflammatory cytokines (IL-6, TNF-alpha). [KG Evidence] This pattern suggests a tissue-resident, epithelial-driven immune program rather than systemic macrophage-mediated inflammation. [Inferred] The pathway enrichment for "response to stress" (6 members) and "Smoking" (5 members) supports an environmental exposure-response axis. [KG Evidence]

#### 4.4 Bone Mineralization and Remodeling

OPG (TNFRSF11B) and MEPE anchor a skeletal biology sub-theme. [KG Evidence] OPG inhibits RANKL-mediated osteoclast activation; MEPE regulates phosphate metabolism and mineralization. [Model Knowledge] The co-expression with DHEA-S and related androgens is biologically coherent given the established role of adrenal androgens in maintaining bone density, particularly in postmenopausal women and aging men. [Model Knowledge] [Inferred]

#### 4.5 Hub-Filtering Note

LGALS4 (3,304 edges), TNFRSF11B (3,164 edges), and MMP12 (3,017 edges) are high-connectivity nodes whose disease associations should be interpreted with caution. [KG Evidence] Disease recurrences driven exclusively by these members may reflect annotation density rather than specific module biology. The most informative disease associations are those shared with lower-connectivity members (PRELP at 1,063 edges, AGRP at 1,204 edges).

---

### 5. Gap Analysis

#### 5.1 Informative Absences

The following absences carry biological interpretive value under the Open World Assumption:

| Expected Entity | Interpretation | Significance |
|---|---|---|
| **Cortisol/cortisone** | Module captures the adrenal androgen (CYP17A1) arm, not the glucocorticoid (CYP11B1) arm | High: reveals pathway-specific steroidogenic regulation [Inferred] |
| **Testosterone/estradiol** | Module indexes sulfated steroid reservoir, not active hormones; consistent with intracrine biology | High: implicates STS-mediated tissue-level activation [Inferred] |
| **RANKL (TNFSF11)** | OPG present without its ligand; consistent with anti-resorptive bone state or platform limitation for membrane-bound proteins | Moderate: reinforces osteoblast-protective interpretation [KG Evidence] [Inferred] |
| **IL-6/TNF-alpha** | Non-canonical immune members (IL17D, IL1RL2, HAVCR1) present without classical cytokines | Moderate: suggests tissue-specific rather than systemic inflammation [KG Evidence] [Inferred] |
| **Leptin/adiponectin** | AGRP present without canonical adipokines | Moderate: peripheral AGRP may be decoupled from central leptin signaling [KG Evidence] [Inferred] |
| **BCAAs** | Steroid sulfate-dominated metabolite panel excludes amino acid metabolism | Low: platform or module segregation effect [Inferred] |
| **Ceramides** | Sulfation axis distinct from sphingolipid metabolism | Low: module captures Phase II conjugation, not lipotoxicity [Inferred] |

#### 5.2 Standard (Platform-Dependent) Gaps

Insulin, C-peptide, HbA1c, and fasting glucose are absent due to platform limitations rather than biological dissociation; these clinical measures are not captured by mass spectrometry-based omics or proteomic platforms used in WGCNA analyses. [Inferred]

---

### 6. Temporal Context

No explicit longitudinal or time-series data were provided for this module. The following causal architecture is inferred from the biological relationships:

**Upstream (cause-proximal):** Adrenal steroidogenesis (pregnenolone sulfate → DHEA-S → androstandiol/androstenediol sulfates) represents the biosynthetic origin of the metabolite cluster. [Inferred] AGRP-mediated energy homeostasis and HPA-axis-adjacent adrenal activity may constitute upstream regulatory inputs. [Inferred]

**Midstream (effector):** Sulfotransferase and sulfatase activity (SULT2A1, STS) determine the balance between sulfated steroid storage and local activation. [Model Knowledge] OPG/MEPE co-regulation modulates bone remodeling downstream of steroid signaling. [Inferred]

**Downstream (consequence-proximal):** ECM remodeling (MMP12/TIMP4/DCN/PRELP) and mucosal immune activation (IL17D/IL1RL2/HAVCR1/LGALS4) likely represent tissue-level consequences of steroid-modulated inflammation and matrix turnover. [Inferred]

If longitudinal samples are available, the module eigengene trajectory could be tested for temporal precedence of steroid sulfate changes relative to ECM/immune protein changes, enabling causal inference via Granger causality or mediation analysis. [Inferred]

---

### 7. Research Recommendations

#### Priority 1: Experimental Validation

1. **Test pregnanediol sulfate isomers for TRPM3-mediated insulin secretion.** The literature-grounded evidence for PM5S (epiallopregnanolone sulfate) acting via TRPM3 to enhance GSIS provides a direct experimental template. [Literature: "Sulfated Progesterone Metabolites," 2022] Measure GSIS in human islets exposed to the specific 5-alpha-pregnanediol sulfate isomers present in this module.

2. **Correlate the module eigengene with bone density and turnover markers.** The OPG-MEPE-DHEA-S co-expression signature predicts association with bone mineral density. Test in a cohort with concurrent DXA and serum osteocalcin/CTX/P1NP measurements.

3. **Quantify sulfated-to-glucuronidated androgen ratios in hyperandrogenic cohorts.** The co-presence of sulfated and glucuronidated androgen metabolites enables construction of a conjugation ratio that may discriminate adrenal from peripheral androgen excess in PCOS.

#### Priority 2: Literature and Database Mining

4. **Systematic review of DHEA-S effects on MMP12/TIMP4 balance in adipose and vascular tissue.** The co-expression of adrenal androgens with matrix metalloproteinases is underexplored; targeted literature review may reveal mechanistic links.

5. **Query STS (steroid sulfatase) expression in tissues relevant to the disease recurrence profile** (GI mucosa, bone, skin, lung). The module's intracrine hypothesis depends on tissue-specific STS activity.

#### Priority 3: Follow-Up Computational Analyses

6. **Perform module preservation analysis across independent cohorts** to determine whether the Tan module replicates, particularly the cross-platform co-expression of steroid sulfates with ECM proteins.

7. **Conduct conditional WGCNA or partial correlation analysis** to test whether DHEA-S mediates the correlation between AGRP and the ECM remodeling genes, or whether these represent independent sub-modules.

8. **Expand knowledge graph coverage for sparse metabolites.** Twenty-one metabolite members have fewer than 20 edges; targeted annotation of these steroid sulfate conjugates in MetaCyc, Reactome, or HMDB would improve future analyses. [KG Evidence]

9. **Test for sex-stratified module behavior.** Given the adrenal androgen content, the module eigengene likely shows sex-dimorphic behavior; sex-stratified analysis may reveal whether the module is driven by adrenal (sex-independent) or gonadal (sex-dependent) steroid production.

---

*Report generated from KRAKEN knowledge graph analysis. All factual claims are attributed to their evidence source. Tier 3 predictions require experimental validation; approximately 18% of computational predictions of this nature progress to clinical investigation.*

### Literature References

Papers discovered via semantic search. 2 unique papers across 4 hypotheses.

| Hypothesis | Citation | Link | Relevance |
|------------|----------|------|-----------|
| Inferred role of CHEBI:133714; Inferred role of CHEBI:133715 |  (2026) "Development and validation of an LC-MS/MS assay for serum 5α-androstane-3α, 17β-diol 17-glucuronide ..." | [Link](https://pubmed.ncbi.nlm.nih.gov/41967453/) | Development and validation of an LC-MS/MS assay for serum 5α-androstane-3α, 17β-diol 17-glucuronide with enhanced interf... |
| Inferred role of CHEBI:133699; Inferred role of CHEBI:133712 |  (2022) "Sulfated Progesterone Metabolites That Enhance Insulin Secretion via TRPM3 Are Reduced in Serum From..." | [Link](https://pubmed.ncbi.nlm.nih.gov/35073578/) | Serum progesterone sulfates were evaluated in the etiology of gestational diabetes mellitus (GDM). Serum progesterone su... |