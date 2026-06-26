# Pink Module Run: Discovery Output (54-analyte, dev, 2026-06-23)

This document presents the discovery output of a full-module run of the **Pink** WGCNA module through the Kraken 12-node discovery pipeline, executed on the dev environment on 2026-06-23 (commit `813fc3f`) with BioMapper entity resolution enabled. It is one of a per-module series over the Frailty multi-omics WGCNA modules; see the [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm).

## Run Provenance

The run submitted 54 named analytes, parsed 53 at intake, and resolved 53 distinct entities (14 biomapper, 35 fuzzy, 4 exact) to 41 distinct CURIEs. Triage classified 3 well-characterized, 6 moderate, 30 sparse, and 14 cold-start, with 0 edge-count measurement failures. Downstream analysis produced 469 direct-KG findings, 27 cold-start findings, 3 biological themes, 0 cross-entity bridges (0 evidence-grounded), and 51 hypotheses supported by 20 literature references. Synthesis emitted a 24173-character report. The run completed in approximately 692.9 s of wall-clock time (status complete, 20 errors). All counts are taken from the reducer-merged pipeline state.

| Stage | Result |
|---|---|
| Submitted | 54 named analytes |
| Intake | 53 parsed |
| Entity resolution | 53 resolved (14 biomapper, 35 fuzzy, 4 exact) to 41 distinct CURIEs |
| Triage | 3 well-characterized, 6 moderate, 30 sparse, 14 cold-start (0 measurement failures) |
| Direct KG | 469 findings |
| Cold-start | 27 findings, 36 skipped |
| Pathway enrichment | 3 biological themes |
| Integration | 0 bridges (0 evidence-grounded) |
| Literature grounding | 20 papers |
| Synthesis | 51 hypotheses, 24173-character report |
| Run total | ~692.9 s wall-clock, status complete, 20 errors |

## Related

- Companion run metrics: [Pink Module Run: Pipeline Performance Report](https://phwiki.phenoma.ai/doc/pink-module-run-pipeline-performance-report-54-analyte-dev-2026-06-23-sBZjXUAb57)
- [BioMapper: Collection Index](https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm)

---



## Pink WGCNA Module Discovery Report: Glycerophospholipid Remodeling as a Coordinated Metabolic Program

### 1. Executive Summary

The Pink WGCNA module encodes a coordinated glycerophospholipid remodeling program dominated by lysophosphatidylcholines (LPCs), lysophosphatidylethanolamines (LPEs), lysophosphatidylinositols (LPIs), acylcholines, and hydroxylated fatty acids, unified by the enzymatic activities of phospholipases and anchored by two proteins: MMP7 (a matrix metalloproteinase linked to extracellular matrix degradation and Wnt/AGE-RAGE signaling) and GDF2/BMP9 (a hepatic sinusoidal endothelium-derived growth factor governing vascular homeostasis and iron metabolism). [KG Evidence; Inferred] The module's composition reveals qualitative lipid remodeling rather than quantitative lipid accumulation, with shared disease associations pointing toward gastrointestinal malignancy, liver cancer, and digestive system disorders, while the absence of glycemic indices, ceramides, and inflammatory cytokines positions this module as a para-glycemic, tissue-remodeling signature potentially upstream of or parallel to classical metabolic syndrome cascades. [KG Evidence; Inferred]

### 2. Key Findings (Tier 1 to 2)

#### 2.1 Module-Level Disease Convergence

Seven diseases recur across two or more module members, establishing the module's disease relevance. [KG Evidence]

| Disease | Members Sharing | Evidence Strength |
|---|---|---|
| Malignant colon neoplasm (MONDO:0021063) | phosphatidylcholine, MMP7 | Curated |
| Liver cancer (MONDO:0002691) | phosphatidylcholine, MMP7 | Curated |
| Psoriasis (MONDO:0005083) | phosphatidylcholine, MMP7 | Curated |
| Prostate cancer (MONDO:0008315) | phosphatidylcholine, MMP7 | Curated |
| Digestive system disorder (MONDO:0004335) | GDF2, MMP7 | Curated |
| Cancer (MONDO:0004992) | GDF2, MMP7 | Curated |
| Visual epilepsy (MONDO:0001386) | phosphatidylcholine, GDF2 | Curated |

The convergence of MMP7 and phosphatidylcholine on gastrointestinal and hepatic cancers is notable: MMP7 participates in extracellular matrix disassembly, collagen catabolic processes, and the AGE/RAGE and Wnt signaling pathways (WikiPathways WP399, WP2324), all of which are established drivers of tumor invasion in these tissues. [KG Evidence] GDF2 (BMP9), the second protein member, is expressed predominantly in hepatic sinusoidal endothelium and participates in angiogenesis, BMP signaling, and vascular morphogenesis; its top individual disease association is hereditary hemorrhagic telangiectasia (HHT), reflecting its canonical role in ALK1/ENG-mediated vascular integrity. [KG Evidence] 2-Hydroxypalmitate, the only metabolite member with a direct disease association, links to colorectal cancer. [KG Evidence]

#### 2.2 Module-Level Pathway Convergence

Three pathways/processes are shared by both protein members (MMP7 and GDF2). [KG Evidence]

- **Protein binding** (GO:0005515): both MMP7 and GDF2 engage in protein-protein interactions as part of their signaling and catalytic mechanisms.
- **Smoking** (UMLS:C0037369): both genes are associated with smoking-related biological responses, consistent with the known effects of tobacco exposure on both MMP activity and BMP/TGF-beta signaling.
- **Response to stress** (GO:0006950): both genes participate in stress response pathways, linking the module to tissue damage and repair programs.

#### 2.3 MMP7: The Matrix Remodeling Hub

MMP7 is the most highly connected module member (3,637 edges). [KG Evidence] Its pathway memberships define the module's tissue-remodeling identity:

- Extracellular matrix disassembly and organization (GO:0022617, GO:0030198) [KG Evidence]
- Collagen catabolic and metabolic processes (GO:0030574, GO:0032963) [KG Evidence]
- AGE/RAGE pathway and chronic hyperglycemia-induced neuronal impairment (WikiPathways WP2324) [KG Evidence]
- Wnt signaling pathway and pluripotency (WikiPathways WP399) [KG Evidence]
- Matrix metalloproteinases pathway (WikiPathways WP129) [KG Evidence]
- Defense response to bacteria (Gram-positive and Gram-negative) and antibacterial peptide biosynthesis (GO:0042742, GO:0002774) [KG Evidence]
- Gastrin signaling pathway [KG Evidence]

MMP7 interacts with FASLG, LGALS3, DCN, SPP1, A2M, CTSB, CTSK, and COL15A1 (among others), reflecting its role in proteolytic processing of both ECM components and immunomodulatory substrates. [KG Evidence] A novel interaction with CXCR2 and CXCL5 (hidden gems) suggests a previously underappreciated role in neutrophil recruitment. [KG Evidence]

#### 2.4 GDF2 (BMP9): The Vascular Endothelial Regulator

GDF2 (1,394 edges) governs vascular biology through interactions with ACVRL1 (ALK1), ENG (endoglin), BMPR2, SMAD4, NOTCH1, and EDN1. [KG Evidence] Its biological processes span:

- Angiogenesis and negative regulation thereof (GO:0001525, GO:0016525) [KG Evidence]
- Blood vessel morphogenesis and vasculogenesis (GO:0048514, GO:0001570) [KG Evidence]
- Intracellular iron ion homeostasis (GO:0006879), reflecting BMP9's established role in hepcidin regulation [KG Evidence]
- Osteoblast differentiation and ossification (GO:0001649, GO:0001503) [KG Evidence]
- Cartilage development (GO:0051216) [KG Evidence]

#### 2.5 Phosphatidylcholine: The Metabolic Anchor

Phosphatidylcholine (410 edges) is the only well-characterized metabolite member. [KG Evidence] Its top disease association is hypothyroidism, and it shares malignant colon neoplasm, liver cancer, psoriasis, and prostate cancer associations with MMP7. [KG Evidence]

#### 2.6 Pathway Enrichment: Phospholipase Gene Network

The pathway enrichment analysis identified a network of lipid-metabolizing genes connecting five or more input entities: PLA2G6, LIPC, LIPA, PNPLA3, and LIPF, among 18 total gene connections. [KG Evidence] This gene set encodes phospholipases and lipases responsible for generating the lysophospholipid species that dominate the module. CEL (bile salt-activated lipase; UniProtKB:P19835) was the sole protein-level enrichment connecting five input entities. [KG Evidence] *Homo sapiens* appeared as a hub organism connecting two entities but is flagged as a high-connectivity node and accordingly de-emphasized. [KG Evidence]

### 3. Novel Predictions (Tier 3)

#### 3.1 Glycerophosphorylcholine Participates in the WikiPathways Glycerolipids and Glycerophospholipids Pathway (WP4722)

**Logic chain**: Glycerophosphorylcholine (KEGG:C06041) shares 0.86 embedding similarity with the WikiPathways "Glycerolipids and glycerophospholipids" pathway (WP4722). This pathway explicitly covers glycerophospholipid metabolism. As a direct product of phosphatidylcholine hydrolysis by phospholipase A2, GPC is biochemically expected to participate in this pathway. [KG Evidence; Inferred] Literature evidence supports GPC's role as a central intermediate in glycerophospholipid metabolism and choline recycling; a 2026 review characterizes GPC as "a water-soluble, choline-containing glycerophosphodiester" naturally generated in phospholipid catabolism. [Literature: "A Friend or Foe: Understanding the Physiological Significance... of Glycerophosphocholine," 2026] Additionally, DHRS2's tumor-suppressive role in ovarian cancer operates partly through modulation of the phosphorylcholine/GPC ratio, confirming GPC's position as a metabolically regulated node. [Literature: Li Z et al., 2022]

**Validation step**: Query WikiPathways WP4722 directly for GPC (C06041) as a listed metabolite participant.

**Calibration**: Approximately 18% of computational predictions of this type progress to experimental validation. The biochemical plausibility here is high, suggesting this prediction falls in the more favorable fraction.

#### 3.2 GPC Is Enzymatically Linked to PLD1 (Phospholipase D1) and PCYT2

**Logic chain**: Through the same WP4722 pathway similarity (0.86), GPC is inferred to be metabolically related to PLD1 (NCBIGene:5337) and PCYT2 (NCBIGene:5833), both participants in WP4722. [KG Evidence; Inferred] PLD1 catalyzes phosphatidylcholine hydrolysis to phosphatidic acid and choline; while GPC arises from the PLA2 (not PLD) branch, both enzymes act on the same substrate pool. Literature confirms that phospholipase D activation participates in phosphatidylcholine hydrolysis signaling. [Literature: Avila MA et al., 1993] PCYT2 has been shown to synthesize CDP-glycerol from glycerol-3-phosphate in mammals, a pathway intersecting with GPC metabolism. [Literature: "PCYT2 synthesizes CDP-glycerol in mammals," 2021]

**Validation step**: Cross-reference KEGG reaction databases for direct enzymatic relationships between GPC and PLD1/PCYT2 gene products.

**Calibration**: ~18% validation rate applies. The PLD1 connection is biochemically indirect (parallel substrate, different products); the PCYT2 connection has stronger literature grounding.

#### 3.3 2-Hydroxystearate Subclasses Under Hydroxy Fatty Acid Anion (CHEBI:59835)

**Logic chain**: 2-Hydroxystearate (CHEBI:229769) shares structural similarity with 7-hydroxystearate(1-) (0.89), 11-hydroxystearate(1-) (0.88), and 8-hydroxyoleate(1-) (0.83), all of which are subclasses of CHEBI:59835 (hydroxy fatty acid anion). As a positional isomer differing only in the hydroxyl group location, 2-hydroxystearate almost certainly belongs to this same chemical class. [KG Evidence] Literature on fatty acid hydratases confirms the biosynthetic spectrum of hydroxy fatty acids across chain lengths C11:1 to C22:6, with multiple positional isomers generated by different hydratase subtypes. [Literature: "Expanding the biosynthesis spectrum of hydroxy fatty acids," 2024]

**Validation step**: ChEBI ontology lookup to confirm CHEBI:229769 classification under CHEBI:59835.

**Calibration**: ~18% validation rate applies generally, though this ontological classification prediction has higher intrinsic confidence than functional predictions.

#### 3.4 Module-Level Inference: Hepatic Phospholipase Activity as the Upstream Driver

**Logic chain**: GDF2/BMP9 is produced almost exclusively by hepatic sinusoidal endothelial cells. [Model Knowledge] MMP7 participates in liver cancer and digestive system disorder pathways. [KG Evidence] The enrichment gene network includes LIPC (hepatic lipase), LIPA (lysosomal acid lipase), and PNPLA3 (patatin-like phospholipase domain-containing 3, the strongest GWAS gene for non-alcoholic fatty liver disease). [KG Evidence] The module's metabolite composition (dozens of LPC, LPE, and LPI species) represents the enzymatic products of phospholipase A-type activity on hepatic membrane phospholipids. [Inferred] This convergence suggests the module captures a hepatic phospholipid remodeling program, potentially driven by PNPLA3-mediated lipid droplet remodeling in hepatocytes and LIPC-mediated lipoprotein processing.

**Validation step**: Correlate module eigengene with hepatic imaging (MRI-PDFF or FibroScan) and serum ALT/AST in the parent cohort. Test whether PNPLA3 I148M genotype modifies module expression.

**Calibration**: ~18% of such integrative predictions advance to clinical investigation.

### 4. Biological Themes

#### 4.1 Unifying Theme: Enzymatic Glycerophospholipid Remodeling

The module's dominant biological identity is the coordinated enzymatic remodeling of glycerophospholipid membranes. [Inferred] This conclusion rests on three convergent lines of evidence:

1. **Metabolite composition**: 51 of 53 module members are glycerophospholipid species or their catabolic products (LPCs, LPEs, LPIs, GPC, acylcholines, hydroxylated fatty acids, arachidonoyl glycerolipids). The module contains sn-1 and sn-2 positional isomers across multiple acyl chain lengths (C15:0 through C22:6), plasmalogen species (P-16:0, P-18:0, P-18:1), and ether-linked species (O-16:0, O-18:0). [KG Evidence; Inferred]

2. **Enzyme gene connections**: The pathway enrichment identifies phospholipase genes (PLA2G6, PLD1), hepatic/lysosomal lipases (LIPC, LIPA, LIPF), and PNPLA3 as shared connectors. [KG Evidence] CEL (bile salt-activated lipase) connects five input entities at the protein level. [KG Evidence]

3. **Product-substrate relationships**: LPC species are the direct products of PLA2-type cleavage of phosphatidylcholines; GPC is a further degradation product; acylcholines (palmitoylcholine, oleoylcholine, arachidonoylcholine, stearoylcholine, linoleoylcholine, docosahexaenoylcholine) represent the choline-ester branch of phospholipid catabolism. [Model Knowledge]

#### 4.2 Arachidonoyl Species as Signaling Precursors

The module is notably enriched in arachidonoyl-containing species: 1-arachidonylglycerol (an endocannabinoid precursor), arachidonoylcholine, 1-arachidonoyl-GPC, 2-arachidonoyl-GPC, 1-arachidonoyl-GPE, 2-arachidonoyl-GPE, 1-arachidonoyl-GPI, and stearoyl-arachidonoyl-glycerol. [KG Evidence] Arachidonic acid (20:4n6), released by phospholipase A2 from these species, is the obligate precursor for prostaglandin, leukotriene, and thromboxane biosynthesis. [Model Knowledge] The co-expression of these species with MMP7 (which processes pro-inflammatory substrates including FASLG and SPP1) suggests coordinate regulation of eicosanoid precursor availability and tissue remodeling. [Inferred]

#### 4.3 Hydroxylated Fatty Acids: Specialized Modifications

Two hydroxylated fatty acids (2-hydroxystearate and 2-hydroxypalmitate) co-express with the LPC species. [KG Evidence] 2-Hydroxypalmitate links to colorectal cancer in the KG. [KG Evidence] These 2-hydroxy fatty acids are generated by fatty acid 2-hydroxylase (FA2H), and their presence alongside LPC species may reflect coordinated membrane lipid turnover involving both phospholipase-mediated headgroup cleavage and fatty acid hydroxylation. [Model Knowledge; Inferred]

#### 4.4 Acylcholines: An Emerging Lipid Class

The module contains six acylcholine species (palmitoylcholine, oleoylcholine, arachidonoylcholine, docosahexaenoylcholine, dihomo-linolenoyl-choline, stearoylcholine, linoleoylcholine). [KG Evidence] Acylcholines are a recently described class of endogenous lipids generated by carnitine/choline acetyltransferases, with proposed roles in cholinergic signaling modulation. [Model Knowledge] Their co-expression with LPC species suggests a shared upstream phospholipid remodeling event that simultaneously generates both LPCs (via PLA2) and acylcholines (via acyl-CoA:choline acyltransferase activity on released fatty acids). [Inferred]

### 5. Gap Analysis

#### 5.1 Informative Absences

The absences from this module are as diagnostically revealing as its contents. The analysis identified 12 expected-but-absent entity classes; the most informative are summarized below. [KG Evidence; Inferred]

**Bulk lipids absent (triglycerides, free fatty acids)**: The module contains enzymatically *remodeled* lipids (LPCs, hydroxylated fatty acids, specific arachidonoyl species) but no bulk lipids. This distinction indicates the module captures *qualitative* lipid remodeling (which phospholipids are enzymatically cleaved and which acyl chains released) rather than *quantitative* lipid accumulation. The phospholipase gene connections (PLA2G6, LIPC, LIPA) confirm this is an enzymatic remodeling signature, not mass-action dyslipidemia. [KG Evidence; Inferred]

**Sphingolipid branch entirely absent (ceramides, sphingomyelins)**: The complete exclusion of ceramides, sphingomyelins, sphingosine-1-phosphate, and hexosylceramides reveals a clean biochemical partition. The enzymes involved are distinct: phospholipases (PLA2G6, PLD1, LIPC) for this module versus serine palmitoyltransferase/ceramide synthases for sphingolipid metabolism. [KG Evidence; Inferred] Ceramides almost certainly reside in a separate WGCNA module, indicating glycerophospholipid membrane remodeling and sphingolipid-mediated lipotoxicity are independently regulated programs in this cohort.

**Glycemic indices absent (insulin, C-peptide, HbA1c, HOMA-IR, fasting glucose)**: The module's independence from all glycemic markers, despite MMP7's connection to AGE/RAGE and chronic hyperglycemia pathways, suggests the module captures a *downstream tissue remodeling consequence* or a *parallel metabolic program* that operates on a different regulatory axis than integrated glycemic control. [KG Evidence; Inferred] For a diabetes conversion study, this independence may indicate the module captures an early or para-glycemic disease process.

**Classical inflammatory cytokines absent (TNF-alpha, IL-6, CRP)**: Despite MMP7's inflammatory pathway connections, the absence of acute-phase cytokines suggests the module captures *tissue remodeling consequences* (chronic/resolution phase) rather than active inflammatory initiation. [KG Evidence; Inferred]

**BCAAs and acylcarnitines absent**: These amino acid catabolism and mitochondrial beta-oxidation markers, respectively, occupy biologically distinct axes. Acylcarnitines reflect *intracellular* mitochondrial fatty acid handling, whereas the module's lipids are predominantly *extracellular* or *membrane-associated*. [KG Evidence; Inferred]

**TCF7L2 absent despite Wnt pathway membership of MMP7**: TCF7L2 is a transcription factor whose transcript/protein levels need not co-vary with downstream targets at the circulating protein level. The connection is mechanistic (TCF7L2 as upstream regulator) rather than co-expressed; this distinction is critical in WGCNA interpretation. [KG Evidence; Inferred]

#### 5.2 Entity Resolution Limitations

A substantial fraction of the metabolite members (14 of 53, 26%) resolved to cold-start identifiers with zero KG edges, and many others (30 entities) have sparse coverage (1 to 19 edges). [KG Evidence] Notably, many entities resolved to incorrect or mismatched identifiers: multiple distinct LPC species mapped to the same KEGG glycan identifier (G00122/GP1c), and several arachidonoyl species mapped to ARACHIDETH-3 (a cosmetic ingredient) or PALMITOYLYSYLAMIDE. These misresolutions reflect the inherent difficulty of mapping specialized lipidomic nomenclature to general-purpose biomedical ontologies. The true biological connectivity of this module is almost certainly higher than the KG analysis reveals.

### 6. Temporal Context

No explicit longitudinal metadata accompanies this analysis; however, the module's composition permits inference about causal ordering. [Inferred]

**Upstream causes (candidate)**: Hepatic phospholipase activity (PLA2G6, PNPLA3, LIPC) generates the LPC, LPE, and GPC species that dominate the module. GDF2/BMP9, produced by hepatic sinusoidal endothelial cells, may serve as both a marker and regulator of hepatic vascular integrity that co-varies with hepatic lipid processing. [KG Evidence; Model Knowledge]

**Downstream consequences (candidate)**: MMP7-mediated extracellular matrix degradation, collagen catabolism, and Wnt pathway activation represent tissue remodeling responses that may follow from sustained phospholipid remodeling and the release of arachidonic acid-derived eicosanoid signals. [KG Evidence; Inferred]

**Causal inference opportunity**: If the parent cohort has longitudinal sampling, testing whether the metabolite (LPC/GPC/acylcholine) components of the module eigengene precede, co-occur with, or follow the protein (MMP7, GDF2) components would distinguish between the module capturing (a) a coordinated hepatic remodeling event or (b) an artifactual co-expression of temporally staggered processes.

### 7. Research Recommendations

#### 7.1 High Priority

1. **Correlate module eigengene with hepatic phenotypes**: Test associations with MRI-PDFF (liver fat), FibroScan (stiffness), and serum ALT/AST to determine whether the module reflects hepatic lipid remodeling specifically. [Inferred]

2. **Test PNPLA3 I148M genotype as a modifier**: The I148M variant (rs738409) is the strongest common genetic determinant of hepatic lipid metabolism. Its effect on module eigengene would establish a causal anchor. [Model Knowledge; Inferred]

3. **Validate the GDF2-MMP7 co-expression biologically**: Determine whether GDF2 and MMP7 co-localize in hepatic tissue (sinusoidal endothelium and Kupffer cells/biliary epithelium, respectively) or whether their co-expression reflects shared systemic regulators. [Inferred]

4. **Quantify phospholipase A2 activity**: Direct measurement of PLA2 activity (sPLA2-IIA or Lp-PLA2) in the cohort would test whether the LPC species reflect enzymatic activity rather than passive membrane breakdown. [Inferred]

#### 7.2 Moderate Priority

5. **Examine acylcholine biology**: The six acylcholine species represent an emerging and understudied lipid class. Targeted quantification and correlation with cholinergic markers (acetylcholinesterase activity, butyrylcholinesterase) would clarify their biological significance. [Model Knowledge; Inferred]

6. **Cross-module comparisons**: Compare the Pink module eigengene with eigengenes of other WGCNA modules to test the predicted independence from (a) sphingolipid/ceramide modules, (b) BCAA/amino acid modules, and (c) glycemic index modules. [Inferred]

7. **Improve entity resolution for lipidomics**: Re-map the 14 cold-start and 30 sparse-coverage metabolites using specialized lipidomics ontologies (LIPID MAPS, SwissLipids, HMDB) rather than general-purpose identifiers. This would substantially improve KG query coverage. [KG Evidence]

#### 7.3 Lower Priority

8. **Literature search for GDF2 and phospholipid metabolism**: No direct literature links between GDF2/BMP9 and glycerophospholipid remodeling were identified. A targeted search for BMP9-regulated lipid metabolism in hepatic endothelial cells could reveal a mechanistic link. [Model Knowledge]

9. **Validate WikiPathways WP4722 membership for GPC**: Confirm whether glycerophosphorylcholine is formally listed in the Glycerolipids and Glycerophospholipids pathway, which would upgrade this Tier 3 prediction to Tier 1. [KG Evidence; Inferred]

10. **Investigate MMP7-CXCR2/CXCL5 axis**: The novel (hidden gem) interaction between MMP7 and the neutrophil chemokine axis (CXCR2, CXCL5) warrants literature validation and could link the module to neutrophil-mediated tissue remodeling. [KG Evidence]

---

*Report generated from KRAKEN knowledge graph analysis of 53 resolved entities from the Pink WGCNA module. Evidence coverage: 3 well-characterized entities (MMP7, GDF2, phosphatidylcholine), 5 moderate-coverage entities, 30 sparse-coverage entities, and 14 cold-start entities with no KG representation. All Tier 3 predictions carry an approximate 18% historical validation rate and require experimental confirmation.*

### Literature References

Papers discovered via semantic search. 4 unique papers across 2 hypotheses.

| Hypothesis | Citation | Link | Relevance |
|------------|----------|------|-----------|
| Inferred role of CHEBI:229769 |  (2024) "Expanding the biosynthesis spectrum of hydroxy fatty acids: unleashing the potential of novel bacter..." | [Link](https://link.springer.com/article/10.1186/s13068-024-02578-2) | ratases ( ... EC 4.2.1.53) have drawn considerable attention due to ... substrate range as well as their widespread dist... |
| Inferred role of CHEBI:229769 |  (2020) "Fatty Acyl Esters of Hydroxy Fatty Acid (FAHFA) Lipid Families" | [Link](https://www.mdpi.com/2218-1989/10/12/512) | O-1 ... 0) and ... 12 ... D) are formed by the ... ylation of a ... lipid, which contrasts with the charged FAHFA lipids... |
| Inferred role of KEGG.GLYCAN:G00122 |  (2023) "Functional identification of  PGM1  in the regulating development and depositing of inosine monophos..." | [Link](https://www.frontiersin.org/journals/veterinary-science/articles/10.3389/fvets.2023.1276582/full) | However, IMP deposition is regulated by numerous genes and complex molecular networks. In order to ... candidate genes t... |
| Inferred role of CHEBI:229769 |  (2022) "Towards an understanding of oleate hydratases and their application in industrial processes \| Microb..." | [Link](https://link.springer.com/article/10.1186/s12934-022-01777-6) | Fatty acid hydratases are able to hydroxylate unsaturated fatty acids. A plethora of fatty acid hydratases, which conver... |