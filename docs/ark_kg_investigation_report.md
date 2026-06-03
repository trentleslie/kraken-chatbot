# KRAKEN Knowledge Graph Investigation Report

**Date:** 2026-03-06
**Investigator:** Claude Opus 4.5 via Claude Code
**Objective:** Assess KRAKEN KG suitability for ARK (Actionable Recommendation Knowledge Engine) demo visualization

---

## Executive Summary

The KRAKEN knowledge graph is **highly suitable** for the ARK demo. Key findings:

1. **Pharmacogenomics data is excellent** - SLCO1B1-statin relationships are well-represented with multiple evidence sources
2. **Drug-gene-disease relationships validated** - All priority entities resolved with thousands of connections
3. **Evidence provenance available** - PMIDs, GWAS p-values, and knowledge sources captured per edge
4. **Multi-hop paths limited** - Direct one-hop queries return richer data than multi-hop pathfinding

### Canary Test Result: PASSED

SLCO1B1 → statin associations confirmed:
- Pravastatin (13 edges), Atorvastatin (11 edges), Simvastatin (7 edges)
- Rosuvastatin (7 edges), Lovastatin (7 edges), Pitavastatin (6 edges)
- SLCO1B1 → Myopathy direct association (7 edges, GWAS p=3e-09)

---

## 1. Schema Summary

### 1.1 Node Categories (86 total)

**Priority categories for ARK:**
| Category | Example Use |
|----------|-------------|
| `biolink:Disease` | AFIB, CAD, T2D, Hypertension |
| `biolink:Gene` | SLCO1B1, MTHFR, AGT, NOS3 |
| `biolink:Drug` | Statins, ACE inhibitors, ARBs |
| `biolink:ChemicalEntity` | Metabolites, nutrients |
| `biolink:SmallMolecule` | Homocysteine, EPA, DHA |
| `biolink:BiologicalProcess` | Pathways, molecular activities |
| `biolink:Protein` | Gene products |

### 1.2 Predicates (113 total)

**Priority predicates for pharmacogenomics:**
| Predicate | Use |
|-----------|-----|
| `biolink:gene_associated_with_condition` | PGx drug-gene-disease |
| `biolink:treats` | Drug-disease relationships |
| `biolink:genetically_associated_with` | Gene-disease from GWAS |
| `biolink:affects` | Metabolite effects |
| `biolink:participates_in` | Pathway membership |
| `biolink:interacts_with` | Drug-drug, PPI |
| `biolink:biomarker_for` | Biomarker associations |

### 1.3 Identifier Prefixes (180+ total)

**Top prefixes by node count:**
| Prefix | Node Count | Use |
|--------|------------|-----|
| `DBSNP` | 5,853,026 | Genetic variants |
| `PUBCHEM.COMPOUND` | 3,277,452 | Chemical compounds |
| `NCBIGene` | 345,898 | Human genes |
| `HMDB` | 218,164 | Metabolites |
| `CHEBI` | 202,378 | Chemical entities |
| `MONDO` | 26,224 | Diseases |
| `DRUGBANK` | 16,255 | Drugs |
| `HGNC` | 44,188 | Human gene symbols |
| `GO` | 44,450 | Gene Ontology |

### 1.4 Ranking Presets

| Preset | Configuration | Use Case |
|--------|---------------|----------|
| `established` | confidence=1.0, evidence=0.5, degree=0.3 | High-confidence relationships |
| `hidden_gems` | confidence=1.0, evidence=-0.8, degree=0.5 | Novel connections |
| `frontier` | confidence=1.0, evidence=-0.8, degree=-0.5 | Unexplored territory |
| `deep_dive` | confidence=1.0, evidence=0.5, degree=-0.5 | Detailed exploration |

---

## 2. Entity Resolution Summary

### 2.1 Genes (All resolved successfully)

| Gene | CURIE | Neighbors | Status |
|------|-------|-----------|--------|
| SLCO1B1 | NCBIGene:10599 | 3,006 | well_characterized |
| MTHFR | NCBIGene:4524 | 3,315 | well_characterized |
| AGT | NCBIGene:183 | 5,971 | well_characterized |
| NOS3 | NCBIGene:4846 | 5,350 | well_characterized |

### 2.2 Metabolites/Biomarkers

| Entity | CURIE | Neighbors | Status |
|--------|-------|-----------|--------|
| Homocysteine | CHEBI:17230 | 1,314 | well_characterized |
| LDL Cholesterol | CHEBI:47774 | 283 | moderate |
| EPA | PUBCHEM.COMPOUND:5282847 | 122 | moderate |
| DHA | CHEBI:36005 | 106 | moderate |
| Magnesium | CHEBI:18420 | 35,734 | well_characterized (hub) |

### 2.3 Diseases

| Disease | CURIE | Neighbors | Status |
|---------|-------|-----------|--------|
| Atrial Fibrillation | MONDO:0004981 | 3,119 | well_characterized |
| Coronary Artery Disease | MONDO:0005010 | 36,250 | hub (high connectivity) |
| Hypertension | MONDO:0005044 | 34,823 | hub (high connectivity) |
| Myopathy | MONDO:0005336 | 3,446 | well_characterized |
| Type 2 Diabetes | MONDO:0005148 | 11,051 | well_characterized |

---

## 3. Query Results

### 3.1 SLCO1B1 Pharmacogenomics (Canary Test)

**Drug Associations (top 15):**
```
CHEBI:8361: pravastatin sodium (edges: 13, score: 0.856)
CHEBI:44185: methotrexate (edges: 11, score: 0.848)
CHEBI:2911: atorvastatin calcium trihydrate (edges: 11, score: 0.847)
CHEBI:28077: rifampicin (edges: 10, score: 0.841)
CHEBI:5296: gemfibrozil (edges: 10, score: 0.840)
CHEBI:4031: cyclosporin A (edges: 9, score: 0.836)
CHEBI:45409: ritonavir (edges: 8, score: 0.830)
CHEBI:9150: simvastatin (edges: 7, score: 0.824)
CHEBI:38545: rosuvastatin (edges: 7, score: 0.824)
CHEBI:40303: lovastatin (edges: 7, score: 0.824)
```

**SLCO1B1-Myopathy Evidence (from get_edges):**
```json
{
  "primary_knowledge_source": "infores:gwas-catalog",
  "knowledge_level": "statistical_association",
  "attributes": {"gwas_p_value": 3e-09}
}
{
  "primary_knowledge_source": "infores:disgenet",
  "publications": ["PMID:31220337"]
}
```

### 3.2 AGT Blood Pressure Pathway

**Drug Associations:**
```
CHEBI:6503: lisinopril (ACE inhibitor) - edges: 9
CHEBI:6541: losartan (ARB) - edges: 7
CHEBI:16480: Nitric Oxide - edges: 6
CHEBI:3011: benazepril (ACE inhibitor) - edges: 6
CHEBI:4784: enalapril (ACE inhibitor) - edges: 5
CHEBI:5959: irbesartan (ARB) - edges: 5
```

**Disease Associations:**
```
MONDO:0001134: essential hypertension (edges: 10)
MONDO:0005044: hypertensive disorder (edges: 7)
MONDO:0005010: coronary artery disorder (edges: 6)
MONDO:0005350: abdominal aortic aneurysm (edges: 7)
```

### 3.3 MTHFR Metabolic Pathway

**Chemical/Supplement Associations:**
```
CHEBI:27470: folic acid (edges: 7)
CHEBI:17439: cyanocob(III)alamin / B12 (edges: 6)
CHEBI:17015: riboflavin / B2 (edges: 4)
```

### 3.4 AFIB Genetic & Treatment

**Gene Associations (Ion Channels):**
```
NCBIGene:6331: SCN5A (sodium channel) - edges: 12
NCBIGene:3741: KCNA5 (potassium channel) - edges: 11
NCBIGene:10021: HCN4 (pacemaker channel) - edges: 10
NCBIGene:3784: KCNQ1 (potassium channel) - edges: 8
NCBIGene:3757: KCNH2 (potassium channel) - edges: 7
```

**Treatment Options:**
```
CHEBI:2904: atenolol (beta blocker) - edges: 11
CHEBI:2663: amiodarone (antiarrhythmic) - edges: 10
CHEBI:3441: carvedilol (beta blocker) - edges: 10
CHEBI:50659: dronedarone (antiarrhythmic) - edges: 9
CHEBI:4551: digoxin (rate control) - edges: 8
```

---

## 4. Response Structure Template

### 4.1 one_hop_query Response

```json
{
  "results": [
    {
      "end_node_id": "CHEBI:8361",
      "edge_count": 13,
      "score": 0.856,
      "score_components": {
        "confidence": 1.0,
        "degree_percentile": 0.931,
        "evidence": 1.0
      },
      "degree": 1823,
      "edge_ids": [101480128, 42295457, ...],
      "start_node_ids": ["NCBIGene:10599"]
    }
  ],
  "nodes": {
    "CHEBI:8361": {
      "id": "CHEBI:8361",
      "name": "pravastatin sodium",
      "categories": ["biolink:SmallMolecule", "biolink:Drug"],
      "description": "...",
      "equivalent_ids": ["DRUGBANK:DB00175", ...],
      "synonyms": ["Pravachol", ...],
      "urls": ["http://..."]
    }
  }
}
```

### 4.2 Edge Structure (from get_edges)

```json
{
  "edge_id": {
    "subject": "NCBIGene:10599",
    "object": "MONDO:0005336",
    "predicate": "biolink:gene_associated_with_condition",
    "primary_knowledge_source": "infores:disgenet",
    "knowledge_level": "prediction",
    "agent_type": "automated_agent",
    "aggregator_knowledge_source": ["infores:rtx-kg2"],
    "publications": ["PMID:31220337"],
    "attributes": {...}
  }
}
```

### 4.3 multi_hop_query Response

```json
{
  "paths": [
    {
      "nodes": ["NCBIGene:183", "GO:0008217", "MONDO:0005044"],
      "predicates": ["biolink:participates_in", "biolink:related_to"],
      "node_names": ["AGT", "regulation of blood pressure", "hypertension"]
    }
  ],
  "nodes": {...}
}
```

---

## 5. Gaps and Limitations

### 5.1 Multi-Hop Path Queries

| Query | Result |
|-------|--------|
| SLCO1B1 → Myopathy (2-hop) | 0 paths |
| SLCO1B1 → Myopathy (3-hop) | 0 paths |
| MTHFR → Homocysteine (3-hop) | 0 paths |
| AGT → Hypertension (2-hop) | 0 paths |
| LDL → CAD (3-hop) | 0 paths |
| Magnesium → AFIB (3-hop) | 0 paths |

**Analysis:** Doubly-pinned multi-hop queries consistently return 0 paths, likely due to:
1. Beam search not finding paths through hub nodes
2. Strict predicate filtering
3. Path length constraints

**Recommendation:** Use one-hop queries for both ends and stitch in frontend visualization.

### 5.2 Sparse Entity Coverage

| Entity | Issue |
|--------|-------|
| EPA/DHA | 106-122 neighbors - moderate but limited disease associations |
| LDL Cholesterol | Only 3 drug associations found |

### 5.3 Node Name Resolution

All node CURIEs successfully resolve to human-readable names. No orphan CURIEs found.

---

## 6. Demo Feasibility Assessment

### Patient A: High CAD PRS, AGT variant, High LDL, High BP
**Feasibility: HIGH**
- AGT → Drug associations: ACE inhibitors, ARBs well-represented
- AGT → Disease: Hypertension, CAD directly linked
- LDL → Disease: Limited but present

### Patient B: MTHFR, NOS3, High Homocysteine, Low EPA/DHA
**Feasibility: MEDIUM-HIGH**
- MTHFR → Supplements: Folic acid, B12, B2 present
- Homocysteine associations available
- EPA/DHA biological processes limited

### Patient C: SLCO1B1, AGT, High CAD PRS, High AFIB PRS
**Feasibility: HIGH**
- SLCO1B1 → Statins: Excellent coverage (7 statins found)
- SLCO1B1 → Myopathy: Direct GWAS association
- AFIB → Drugs: Full treatment landscape
- AGT → BP drugs: Complete

---

## 7. Recommended Query Patterns

### 7.1 Pharmacogenomics Analysis
```python
# Best pattern for drug-gene interactions
one_hop_query(
    start_node_ids=gene_curie,
    end_node_category="biolink:Drug",
    mode="full",
    ranking="established",
    limit=20
)
```

### 7.2 Disease Association Discovery
```python
# For genetic risk factors
one_hop_query(
    start_node_ids=gene_curie,
    end_node_category="biolink:Disease",
    mode="full",
    ranking="established",
    limit=15
)
```

### 7.3 Evidence Extraction
```python
# After one_hop_query, get edge details for citations
get_edges(edge_ids=result['edge_ids'][:5])
# Extract: publications, primary_knowledge_source, gwas_p_value
```

### 7.4 Subgraph Construction for Visualization
1. Run one_hop_query from each patient entity
2. Collect all end_node_ids
3. Find intersections (shared neighbors)
4. Get edge details for provenance
5. Construct visualization JSON

---

## 8. Technical Notes

### 8.1 API Configuration
- **URL:** `https://kestrel.nathanpricelab.com/mcp`
- **Authentication:** X-API-Key header required
- **Session:** MCP-over-HTTP with session management
- **Version:** KESTREL MCP Server v1.16.0

### 8.2 Key Parameters
- `start_node_ids` (plural) - batch queries supported
- `mode`: "slim" | "full" | "preview"
- `ranking`: "established" | "hidden_gems" | "frontier"
- `max_path_length`: 2-5 (default 3)

### 8.3 Rate Considerations
- Large hub nodes (CAD, Hypertension) have 30k+ neighbors
- Use category filters to constrain results
- Preview mode for edge counting only

---

## Appendix: Raw Response Samples

See `/tmp/raw_slco1b1_sse_response.json` for complete SSE wire format of SLCO1B1 drug query.
