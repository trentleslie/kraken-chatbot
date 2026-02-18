"""Tests for literature grounding node and Semantic Scholar client."""

import math
import sys
import pytest

# Import state directly to avoid triggering langgraph import from graph/__init__.py
import importlib.util
spec = importlib.util.spec_from_file_location(
    "state",
    "src/kestrel_backend/graph/state.py"
)
state_module = importlib.util.module_from_spec(spec)
sys.modules["state"] = state_module
spec.loader.exec_module(state_module)
LiteratureSupport = state_module.LiteratureSupport
Hypothesis = state_module.Hypothesis

from src.kestrel_backend.semantic_scholar import (
    score_relevance,
    classify_relationship,
    extract_key_passage,
    format_authors,
    extract_doi,
)


class TestSemanticScholarClient:
    """Tests for semantic_scholar.py helper functions."""

    def test_score_relevance_with_overlap(self):
        """Test relevance scoring with keyword overlap."""
        paper = {
            "abstract": "Fructose consumption increases diabetes risk in mice",
            "citationCount": 100,
        }
        claim = "fructose may be associated with diabetes risk"
        score = score_relevance(paper, claim)

        # Should have positive score due to overlap
        assert 0 < score <= 1
        # Should be weighted toward keyword match (70%)
        assert score > 0.2

    def test_score_relevance_no_overlap(self):
        """Test relevance scoring with no keyword overlap."""
        paper = {
            "abstract": "Quantum computing advances in 2024",
            "citationCount": 50,
        }
        claim = "fructose and diabetes relationship"
        score = score_relevance(paper, claim)

        # Should have low score (only citation component)
        assert 0 <= score < 0.3

    def test_score_relevance_high_citations(self):
        """Test that high citation count contributes to score."""
        paper_low = {"abstract": "Test abstract", "citationCount": 1}
        paper_high = {"abstract": "Test abstract", "citationCount": 10000}
        claim = "test claim"

        score_low = score_relevance(paper_low, claim)
        score_high = score_relevance(paper_high, claim)

        # High citations should boost score
        assert score_high > score_low

    def test_score_relevance_citation_log_scale(self):
        """Test that citation scoring uses log10 scale."""
        # 10k citations should give citation_score = 1.0
        # citation_score = min(1.0, log10(10001) / 4) ≈ min(1.0, 4.0/4) = 1.0
        paper = {"abstract": "", "citationCount": 10000}
        score = score_relevance(paper, "claim")
        # 30% weight on citations = 0.3 max from citations alone
        assert score >= 0.29  # Allow small floating point variance

    def test_classify_relationship_always_supporting(self):
        """Test that v1 always returns 'supporting'."""
        # Even with contradicting keywords, should return supporting
        paper = {"abstract": "However, we found no evidence for this claim"}
        result = classify_relationship(paper, "test claim")
        assert result == "supporting"

    def test_extract_key_passage_finds_relevant_sentence(self):
        """Test key passage extraction finds best match."""
        paper = {
            "abstract": (
                "Background: This study examines diabetes risk factors. "
                "Fructose consumption was strongly associated with increased diabetes. "
                "Conclusion: Diet modification is recommended."
            )
        }
        claim = "fructose associated with diabetes"
        passage = extract_key_passage(paper, claim)

        # Should select sentence with most overlap
        assert "fructose" in passage.lower() or "diabetes" in passage.lower()

    def test_extract_key_passage_truncates_long_sentences(self):
        """Test that long passages are truncated."""
        paper = {
            "abstract": "A" * 400  # Very long single sentence
        }
        passage = extract_key_passage(paper, "test")

        assert len(passage) <= 300
        assert passage.endswith("...")

    def test_extract_key_passage_falls_back_to_title(self):
        """Test fallback to title when no abstract."""
        paper = {
            "abstract": "",
            "title": "Fructose and Diabetes: A Review",
        }
        passage = extract_key_passage(paper, "test")
        assert passage == "Fructose and Diabetes: A Review"

    def test_format_authors_single(self):
        """Test author formatting with single author."""
        authors = [{"name": "John Smith"}]
        result = format_authors(authors)
        assert result == "John Smith"

    def test_format_authors_multiple(self):
        """Test author formatting with multiple authors."""
        authors = [{"name": "John Smith"}, {"name": "Jane Doe"}]
        result = format_authors(authors)
        assert result == "John Smith et al."

    def test_format_authors_empty(self):
        """Test author formatting with no authors."""
        assert format_authors([]) == "Unknown"

    def test_extract_doi_present(self):
        """Test DOI extraction when present."""
        paper = {"externalIds": {"DOI": "10.1234/example"}}
        assert extract_doi(paper) == "10.1234/example"

    def test_extract_doi_missing(self):
        """Test DOI extraction when missing."""
        paper = {"externalIds": {}}
        assert extract_doi(paper) is None

    def test_extract_doi_no_external_ids(self):
        """Test DOI extraction when externalIds is None."""
        paper = {"externalIds": None}
        assert extract_doi(paper) is None


class TestLiteratureSupportModel:
    """Tests for the LiteratureSupport Pydantic model."""

    def test_create_literature_support(self):
        """Test creating a LiteratureSupport instance."""
        lit = LiteratureSupport(
            paper_id="abc123",
            title="Test Paper Title",
            authors="Smith et al.",
            year=2024,
            relevance_score=0.85,
            relationship="supporting",
            key_passage="Key finding here.",
            citation_count=50,
        )
        assert lit.paper_id == "abc123"
        assert lit.title == "Test Paper Title"
        assert lit.relevance_score == 0.85
        assert lit.relationship == "supporting"
        assert lit.doi is None  # Optional field

    def test_literature_support_with_doi(self):
        """Test LiteratureSupport with DOI."""
        lit = LiteratureSupport(
            paper_id="xyz789",
            title="Test",
            authors="Doe",
            year=2023,
            doi="10.1234/test",
            relevance_score=0.5,
            relationship="supporting",
            key_passage="Test passage",
            citation_count=0,
        )
        assert lit.doi == "10.1234/test"

    def test_literature_support_frozen(self):
        """Test that LiteratureSupport is immutable."""
        lit = LiteratureSupport(
            paper_id="test",
            title="Test",
            authors="Author",
            year=2024,
            relevance_score=0.5,
            relationship="supporting",
            key_passage="Passage",
            citation_count=10,
        )
        with pytest.raises(Exception):  # Should raise validation error
            lit.title = "Changed"


class TestHypothesisLiteratureField:
    """Tests for the literature_support field on Hypothesis model."""

    def test_hypothesis_with_empty_literature(self):
        """Test creating hypothesis without literature."""
        hyp = Hypothesis(
            title="Test Hypothesis",
            tier=2,
            claim="Test claim about biology",
            supporting_entities=["CHEBI:12345"],
            structural_logic="A relates to B",
            validation_steps=["Run experiment X"],
        )
        assert hyp.literature_support == []

    def test_hypothesis_with_literature(self):
        """Test creating hypothesis with literature support."""
        lit = LiteratureSupport(
            paper_id="paper1",
            title="Supporting Paper",
            authors="Smith et al.",
            year=2024,
            relevance_score=0.9,
            relationship="supporting",
            key_passage="Evidence here",
            citation_count=100,
        )
        hyp = Hypothesis(
            title="Test Hypothesis",
            tier=1,
            claim="Well-supported claim",
            supporting_entities=["GENE:1234"],
            structural_logic="Direct evidence",
            validation_steps=["Verify in lab"],
            literature_support=[lit],
        )
        assert len(hyp.literature_support) == 1
        assert hyp.literature_support[0].paper_id == "paper1"

    def test_hypothesis_model_copy_with_literature(self):
        """Test updating hypothesis with literature using model_copy."""
        hyp = Hypothesis(
            title="Original",
            tier=2,
            claim="Original claim",
            supporting_entities=[],
            structural_logic="Logic",
            validation_steps=[],
        )
        lit = LiteratureSupport(
            paper_id="new_paper",
            title="New Paper",
            authors="Author",
            year=2025,
            relevance_score=0.8,
            relationship="supporting",
            key_passage="New evidence",
            citation_count=50,
        )

        # Use model_copy to update frozen model
        updated = hyp.model_copy(update={"literature_support": [lit]})

        assert len(updated.literature_support) == 1
        assert updated.literature_support[0].paper_id == "new_paper"
        # Original unchanged
        assert len(hyp.literature_support) == 0


class TestLiteratureUtils:
    """Tests for literature_utils.py helper functions."""

    def test_pmid_to_url_with_prefix(self):
        """Test PMID to URL conversion with PMID: prefix."""
        from src.kestrel_backend.literature_utils import pmid_to_url
        url = pmid_to_url("PMID:12345678")
        assert url == "https://pubmed.ncbi.nlm.nih.gov/12345678"

    def test_pmid_to_url_numeric_only(self):
        """Test PMID to URL conversion with numeric only."""
        from src.kestrel_backend.literature_utils import pmid_to_url
        url = pmid_to_url("12345678")
        assert url == "https://pubmed.ncbi.nlm.nih.gov/12345678"

    def test_doi_to_url(self):
        """Test DOI to URL conversion."""
        from src.kestrel_backend.literature_utils import doi_to_url
        url = doi_to_url("10.1234/example")
        assert url == "https://doi.org/10.1234/example"

    def test_doi_to_url_already_url(self):
        """Test DOI that's already a URL passes through."""
        from src.kestrel_backend.literature_utils import doi_to_url
        url = doi_to_url("https://doi.org/10.1234/example")
        assert url == "https://doi.org/10.1234/example"

    def test_extract_pmid_number(self):
        """Test PMID numeric extraction."""
        from src.kestrel_backend.literature_utils import extract_pmid_number
        assert extract_pmid_number("PMID:12345678") == "12345678"
        assert extract_pmid_number("12345678") == "12345678"

    def test_format_pmid_link(self):
        """Test PMID markdown link formatting."""
        from src.kestrel_backend.literature_utils import format_pmid_link
        link = format_pmid_link("PMID:12345678")
        assert link == "[PMID:12345678](https://pubmed.ncbi.nlm.nih.gov/12345678)"

    def test_extract_pmid_from_string(self):
        """Test PMID extraction from arbitrary text."""
        from src.kestrel_backend.literature_utils import extract_pmid_from_string
        assert extract_pmid_from_string("See PMID:12345678 for details") == "12345678"
        assert extract_pmid_from_string("No PMID here") is None

    def test_extract_pmid_from_string_short(self):
        """Test PMID extraction for older short PMIDs."""
        from src.kestrel_backend.literature_utils import extract_pmid_from_string
        assert extract_pmid_from_string("PMID:1") == "1"
        assert extract_pmid_from_string("See PMID:123 for details") == "123"
        assert extract_pmid_from_string("PMID:999999") == "999999"


class TestOpenAlexClient:
    """Tests for openalex.py helper functions."""

    def test_extract_pmid_from_work_with_pmid(self):
        """Test PMID extraction from OpenAlex work with PMID."""
        from src.kestrel_backend.openalex import extract_pmid_from_work
        work = {"ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/12345678"}}
        assert extract_pmid_from_work(work) == "12345678"

    def test_extract_pmid_from_work_no_pmid(self):
        """Test PMID extraction from work without PMID."""
        from src.kestrel_backend.openalex import extract_pmid_from_work
        work = {"ids": {"doi": "https://doi.org/10.1234/test"}}
        assert extract_pmid_from_work(work) is None

    def test_extract_pmid_from_work_no_ids(self):
        """Test PMID extraction from work without ids."""
        from src.kestrel_backend.openalex import extract_pmid_from_work
        work = {}
        assert extract_pmid_from_work(work) is None

    def test_extract_doi_from_work(self):
        """Test DOI extraction from OpenAlex work."""
        from src.kestrel_backend.openalex import extract_doi_from_work
        work = {"doi": "https://doi.org/10.1234/test"}
        assert extract_doi_from_work(work) == "10.1234/test"

    def test_extract_doi_from_work_no_doi(self):
        """Test DOI extraction when missing."""
        from src.kestrel_backend.openalex import extract_doi_from_work
        work = {}
        assert extract_doi_from_work(work) is None

    def test_format_authors_from_work_single(self):
        """Test author formatting with single author."""
        from src.kestrel_backend.openalex import format_authors_from_work
        work = {
            "authorships": [
                {"author": {"display_name": "John Smith"}}
            ]
        }
        assert format_authors_from_work(work) == "John Smith"

    def test_format_authors_from_work_multiple(self):
        """Test author formatting with multiple authors."""
        from src.kestrel_backend.openalex import format_authors_from_work
        work = {
            "authorships": [
                {"author": {"display_name": "John Smith"}},
                {"author": {"display_name": "Jane Doe"}},
            ]
        }
        assert format_authors_from_work(work) == "John Smith et al."

    def test_format_authors_from_work_empty(self):
        """Test author formatting with no authors."""
        from src.kestrel_backend.openalex import format_authors_from_work
        work = {"authorships": []}
        assert format_authors_from_work(work) == "Unknown"


class TestLiteratureSupportSources:
    """Tests for multi-source LiteratureSupport creation."""

    def test_literature_support_kg_source(self):
        """Test LiteratureSupport with KG source."""
        lit = LiteratureSupport(
            paper_id="PMID:12345678",
            title="PubMed:12345678",
            authors="",
            year=0,
            url="https://pubmed.ncbi.nlm.nih.gov/12345678",
            relevance_score=1.0,
            source="kg",
        )
        assert lit.source == "kg"
        assert lit.url == "https://pubmed.ncbi.nlm.nih.gov/12345678"

    def test_literature_support_openalex_source(self):
        """Test LiteratureSupport with OpenAlex source."""
        lit = LiteratureSupport(
            paper_id="W123456789",
            title="OpenAlex Paper",
            authors="Smith et al.",
            year=2024,
            url="https://doi.org/10.1234/test",
            relevance_score=0.85,
            source="openalex",
        )
        assert lit.source == "openalex"
        assert "doi.org" in lit.url

    def test_literature_support_s2_source(self):
        """Test LiteratureSupport with S2 source."""
        lit = LiteratureSupport(
            paper_id="abc123def456",
            title="S2 Paper",
            authors="Doe et al.",
            year=2023,
            doi="10.1234/s2test",
            url="https://pubmed.ncbi.nlm.nih.gov/87654321",
            relevance_score=0.75,
            source="s2",
        )
        assert lit.source == "s2"
        assert lit.doi == "10.1234/s2test"

    def test_literature_support_default_source(self):
        """Test LiteratureSupport defaults to s2 source."""
        lit = LiteratureSupport(
            paper_id="test",
            title="Test",
            authors="Author",
            year=2024,
        )
        assert lit.source == "s2"  # Default value
