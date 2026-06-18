"""Tests for intake entity-type detection — disease hint (Tier 1 wrong-namespace fix, Unit 1).

detect_entity_types historically emitted only metabolite/protein/gene. Unit 1 adds a "disease"
hint via (1) disease section headers and (2) the existing extract_study_context disease_patterns
regex list applied per entity token. Gene/metabolite heuristics take precedence.
"""

from kestrel_backend.graph.nodes.intake import detect_entity_types


class TestDetectEntityTypesDisease:
    # ---- Happy path: section headers ----

    def test_diseases_header_tags_disease(self):
        query = (
            "Significant Metabolites: glucose\n"
            "Diseases: chronic myeloid leukemia, Parkinson disease"
        )
        hints = detect_entity_types(
            query, ["chronic myeloid leukemia", "Parkinson disease", "glucose"]
        )
        assert hints.get("chronic myeloid leukemia") == "disease"
        assert hints.get("Parkinson disease") == "disease"
        # glucose sits under the Metabolites header -> stays metabolite (header precedence unchanged)
        assert hints.get("glucose") == "metabolite"

    def test_conditions_header_tags_disease(self):
        query = "Conditions: type 2 diabetes"
        hints = detect_entity_types(query, ["type 2 diabetes"])
        assert hints.get("type 2 diabetes") == "disease"

    # ---- Happy path: prose lexicon (no header) ----

    def test_prose_disease_via_reused_lexicon(self):
        query = "We studied patients with Parkinson's disease and type 2 diabetes."
        hints = detect_entity_types(query, ["Parkinson's disease", "type 2 diabetes"])
        assert hints.get("Parkinson's disease") == "disease"
        assert hints.get("type 2 diabetes") == "disease"

    def test_prose_cancer_token(self):
        query = "a cohort enriched for pancreatic cancer"
        hints = detect_entity_types(query, ["pancreatic cancer"])
        assert hints.get("pancreatic cancer") == "disease"

    # ---- Precedence: gene/metabolite heuristics win over the disease lexicon ----

    def test_gene_casing_precedence_over_disease_lexicon(self):
        # "AD" matches both the gene regex ^[A-Z]{2,6}\d?$ and the disease pattern r"ad\b".
        # Gene-symbol casing must take precedence (preserves pre-existing behavior).
        query = "AD biomarkers in plasma"
        hints = detect_entity_types(query, ["AD"])
        assert hints.get("AD") == "gene"

    def test_gene_symbol_stays_gene(self):
        query = "VKORC1 variant carriers"
        hints = detect_entity_types(query, ["VKORC1"])
        assert hints.get("VKORC1") == "gene"

    def test_metabolite_suffix_not_disease(self):
        query = "fasting glucose measured"
        hints = detect_entity_types(query, ["glucose"])
        assert hints.get("glucose") == "metabolite"

    # ---- Edge cases ----

    def test_entity_absent_from_query_gets_no_hint(self):
        hints = detect_entity_types("unrelated text", ["nonexistent entity xyz"])
        assert "nonexistent entity xyz" not in hints

    def test_non_disease_non_gene_non_metabolite_gets_no_hint(self):
        # "albumin" is not a gene (lowercase), not a chemical suffix, not in the disease lexicon.
        query = "serum albumin level"
        hints = detect_entity_types(query, ["albumin"])
        assert hints.get("albumin") is None
