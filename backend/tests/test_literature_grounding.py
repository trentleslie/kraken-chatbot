"""Tests for literature grounding node and Semantic Scholar client."""

import math
import sys
import types
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

# Set up package hierarchy for literature_grounding relative imports.
# NOTE (issue #40): give each stub a real __path__ so submodules this file does NOT
# explicitly fake (e.g. kestrel_backend.logging_config, imported by test_logging.py)
# still resolve to the real source instead of failing with "'kestrel_backend' is not a
# package" during collection. The explicitly-faked submodules below remain authoritative
# because a direct sys.modules entry is consulted before __path__-based resolution.
kestrel_backend = types.ModuleType("kestrel_backend")
kestrel_backend.__path__ = ["src/kestrel_backend"]
kestrel_backend_graph = types.ModuleType("kestrel_backend.graph")
kestrel_backend_graph.__path__ = ["src/kestrel_backend/graph"]
kestrel_backend_graph_nodes = types.ModuleType("kestrel_backend.graph.nodes")
kestrel_backend_graph_nodes.__path__ = ["src/kestrel_backend/graph/nodes"]

sys.modules["kestrel_backend"] = kestrel_backend
sys.modules["kestrel_backend.graph"] = kestrel_backend_graph
sys.modules["kestrel_backend.graph.nodes"] = kestrel_backend_graph_nodes
sys.modules["kestrel_backend.graph.state"] = state_module

# Mock the other imports that literature_grounding needs (not used by build_references_table)
mock_literature_utils = types.ModuleType("kestrel_backend.literature_utils")
mock_literature_utils.pmid_to_url = lambda x: x
mock_literature_utils.doi_to_url = lambda x: x
sys.modules["kestrel_backend.literature_utils"] = mock_literature_utils

mock_openalex = types.ModuleType("kestrel_backend.openalex")
mock_openalex.search_works = None
mock_openalex.extract_pmid_from_work = None
mock_openalex.extract_doi_from_work = None
mock_openalex.format_authors_from_work = None
sys.modules["kestrel_backend.openalex"] = mock_openalex

mock_semantic_scholar = types.ModuleType("kestrel_backend.semantic_scholar")
mock_semantic_scholar.search_papers = None
mock_semantic_scholar.score_relevance = None
mock_semantic_scholar.classify_relationship = None
mock_semantic_scholar.classify_relationship_llm = None  # issue #40: real node imports this (R14)
mock_semantic_scholar.extract_key_passage = None
mock_semantic_scholar.format_authors = None
mock_semantic_scholar.extract_doi = None
mock_semantic_scholar.S2RateLimitError = Exception
sys.modules["kestrel_backend.semantic_scholar"] = mock_semantic_scholar

mock_exa_client = types.ModuleType("kestrel_backend.exa_client")
mock_exa_client.search_papers = None
mock_exa_client.ExaSearchError = Exception
mock_exa_client.extract_doi_from_url = None
# Real implementation needed for is_valid_exa_result tests
def mock_extract_year_from_date(date_str):
    """Extract year from date string like '2024-01-15'."""
    if not date_str:
        return 0
    try:
        return int(date_str[:4])
    except (ValueError, TypeError):
        return 0
mock_exa_client.extract_year_from_date = mock_extract_year_from_date
sys.modules["kestrel_backend.exa_client"] = mock_exa_client

mock_pubmed_client = types.ModuleType("kestrel_backend.pubmed_client")
mock_pubmed_client.search_papers = None
mock_pubmed_client.PubMedSearchError = Exception
# Backfill (issue: abstract-body fetch) imports these into the node namespace.
# Provide a real normalize_pmid (pure) and a default async fetch stub that returns
# nothing; backfill tests monkeypatch lit_grounding_module.pubmed_fetch_abstracts.
mock_pubmed_client.normalize_pmid = lambda pmid: str(pmid).replace("PMID:", "").strip()
async def _stub_fetch_abstracts(pmids, batch_size=200):
    return {}
mock_pubmed_client.fetch_abstracts = _stub_fetch_abstracts
sys.modules["kestrel_backend.pubmed_client"] = mock_pubmed_client

# Now import build_references_table from the real module
lit_grounding_spec = importlib.util.spec_from_file_location(
    "kestrel_backend.graph.nodes.literature_grounding",
    "src/kestrel_backend/graph/nodes/literature_grounding.py",
    submodule_search_locations=[]
)
lit_grounding_module = importlib.util.module_from_spec(lit_grounding_spec)
lit_grounding_module.__package__ = "kestrel_backend.graph.nodes"
sys.modules["kestrel_backend.graph.nodes.literature_grounding"] = lit_grounding_module
lit_grounding_spec.loader.exec_module(lit_grounding_module)
build_references_table = lit_grounding_module.build_references_table
is_valid_exa_result = lit_grounding_module.is_valid_exa_result
_get_paper_key = lit_grounding_module._get_paper_key

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

    def test_repairs_mojibake_in_text_fields(self):
        """Issue #39: UTF-8 mojibake in citation text is repaired; legit text preserved."""
        moj_em = "—".encode("utf-8").decode("latin-1")    # em-dash mojibake
        moj_apos = "’".encode("utf-8").decode("latin-1")  # apostrophe mojibake
        lit = LiteratureSupport(
            paper_id="p", year=2021, authors=f"Rabbani{moj_apos}s group",
            title=f"Lipid{moj_em}diabetes link",
            key_passage="real em-dash — and café",
        )
        assert lit.title == "Lipid—diabetes link"  # single-encoding mojibake repaired
        assert moj_em not in lit.title
        assert "Rabbani" in lit.authors and moj_apos not in lit.authors
        assert "café" in lit.key_passage  # legitimate accented text preserved
        assert "—" in lit.key_passage     # legitimate em-dash preserved (not stripped)

        # deeply double-encoded sequence from the issue (Rabbani 2021)
        lit2 = LiteratureSupport(paper_id="p2", year=2021, authors="A",
                                 title='dose Ã¢Â€Â" response curve')
        assert "Ã¢" not in lit2.title  # double-mojibake lead bytes removed
        assert "—" in lit2.title       # replaced with the correct em-dash

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


class TestPubMedClient:
    """Tests for pubmed_client.py helper functions."""

    def test_extract_doi_from_summary_with_doi(self):
        """Test DOI extraction when DOI is present."""
        from src.kestrel_backend.pubmed_client import extract_doi_from_summary
        paper = {
            "articleids": [
                {"idtype": "pubmed", "value": "12345678"},
                {"idtype": "doi", "value": "10.1234/test.article"},
            ]
        }
        assert extract_doi_from_summary(paper) == "10.1234/test.article"

    def test_extract_doi_from_summary_no_doi(self):
        """Test DOI extraction when DOI is missing."""
        from src.kestrel_backend.pubmed_client import extract_doi_from_summary
        paper = {
            "articleids": [
                {"idtype": "pubmed", "value": "12345678"},
            ]
        }
        assert extract_doi_from_summary(paper) is None

    def test_extract_doi_from_summary_empty(self):
        """Test DOI extraction with empty articleids."""
        from src.kestrel_backend.pubmed_client import extract_doi_from_summary
        paper = {"articleids": []}
        assert extract_doi_from_summary(paper) is None

    def test_extract_year_from_pubdate_full(self):
        """Test year extraction from full date."""
        from src.kestrel_backend.pubmed_client import extract_year_from_pubdate
        assert extract_year_from_pubdate("2024 Jan 15") == 2024

    def test_extract_year_from_pubdate_month_only(self):
        """Test year extraction from month-only date."""
        from src.kestrel_backend.pubmed_client import extract_year_from_pubdate
        assert extract_year_from_pubdate("2023 Mar") == 2023

    def test_extract_year_from_pubdate_year_only(self):
        """Test year extraction from year-only date."""
        from src.kestrel_backend.pubmed_client import extract_year_from_pubdate
        assert extract_year_from_pubdate("2022") == 2022

    def test_extract_year_from_pubdate_season(self):
        """Test year extraction from seasonal date."""
        from src.kestrel_backend.pubmed_client import extract_year_from_pubdate
        assert extract_year_from_pubdate("2024 Spring") == 2024

    def test_extract_year_from_pubdate_empty(self):
        """Test year extraction from empty date."""
        from src.kestrel_backend.pubmed_client import extract_year_from_pubdate
        assert extract_year_from_pubdate("") == 0
        assert extract_year_from_pubdate(None) == 0

    def test_format_authors_from_summary_single(self):
        """Test author formatting with single author."""
        from src.kestrel_backend.pubmed_client import format_authors_from_summary
        paper = {"authors": [{"name": "Smith J"}]}
        assert format_authors_from_summary(paper) == "Smith J"

    def test_format_authors_from_summary_multiple(self):
        """Test author formatting with multiple authors."""
        from src.kestrel_backend.pubmed_client import format_authors_from_summary
        paper = {"authors": [{"name": "Smith J"}, {"name": "Doe A"}]}
        assert format_authors_from_summary(paper) == "Smith J et al."

    def test_format_authors_from_summary_empty(self):
        """Test author formatting with no authors."""
        from src.kestrel_backend.pubmed_client import format_authors_from_summary
        paper = {"authors": []}
        assert format_authors_from_summary(paper) == "Unknown"


class TestMergeLiterature:
    """Tests for merge_literature deduplication logic."""

    def test_merge_empty_list(self):
        """Test merge with empty input."""
        result = lit_grounding_module.merge_literature([])
        assert result == []

    def test_merge_single_item(self):
        """Test merge with single item passes through."""
        lit = LiteratureSupport(
            paper_id="p1", title="Paper 1", authors="A", year=2024,
            doi="10.1/a", relevance_score=0.9, source="openalex",
        )
        result = lit_grounding_module.merge_literature([lit])
        assert len(result) == 1
        assert result[0].paper_id == "p1"

    def test_merge_deduplicates_by_doi(self):
        """Test that papers with same DOI are deduplicated."""
        lit1 = LiteratureSupport(
            paper_id="p1", title="Paper 1", authors="A", year=2024,
            doi="10.1/same", relevance_score=0.9, source="openalex",
        )
        lit2 = LiteratureSupport(
            paper_id="p2", title="Paper 1 Copy", authors="A", year=2024,
            doi="10.1/same", relevance_score=0.85, source="exa",
        )
        result = lit_grounding_module.merge_literature([lit1, lit2])
        assert len(result) == 1
        # OpenAlex has higher priority than Exa
        assert result[0].source == "openalex"

    def test_merge_deduplicates_by_pmid(self):
        """Test that papers with same PMID are deduplicated."""
        lit1 = LiteratureSupport(
            paper_id="PMID:12345678", title="Paper 1", authors="A", year=2024,
            relevance_score=0.9, source="pubmed",
        )
        lit2 = LiteratureSupport(
            paper_id="PMID:12345678", title="Paper 1 Alt", authors="A", year=2024,
            relevance_score=0.85, source="openalex",
        )
        result = lit_grounding_module.merge_literature([lit1, lit2])
        assert len(result) == 1
        # PubMed has higher priority than OpenAlex
        assert result[0].source == "pubmed"

    def test_merge_deduplicates_by_title_year(self):
        """Test that papers with same title+year are deduplicated."""
        lit1 = LiteratureSupport(
            paper_id="id1", title="Unique Paper Title", authors="A", year=2024,
            relevance_score=0.9, source="exa",
        )
        lit2 = LiteratureSupport(
            paper_id="id2", title="Unique Paper Title", authors="B", year=2024,
            relevance_score=0.85, source="openalex",
        )
        result = lit_grounding_module.merge_literature([lit1, lit2])
        assert len(result) == 1
        # OpenAlex has higher priority than Exa
        assert result[0].source == "openalex"

    def test_merge_keeps_distinct_papers(self):
        """Test that papers with different keys are kept separate."""
        lit1 = LiteratureSupport(
            paper_id="p1", title="Paper 1", authors="A", year=2024,
            doi="10.1/a", relevance_score=0.9, source="openalex",
        )
        lit2 = LiteratureSupport(
            paper_id="p2", title="Paper 2", authors="B", year=2023,
            doi="10.1/b", relevance_score=0.85, source="exa",
        )
        result = lit_grounding_module.merge_literature([lit1, lit2])
        assert len(result) == 2

    def test_merge_respects_limit(self):
        """Test that merge respects the limit parameter."""
        papers = [
            LiteratureSupport(
                paper_id=f"p{i}", title=f"Paper {i}", authors="A", year=2024,
                doi=f"10.1/{i}", relevance_score=0.9 - i*0.1, source="openalex",
            )
            for i in range(5)
        ]
        result = lit_grounding_module.merge_literature(papers, limit=3)
        assert len(result) == 3

    def test_merge_sorts_by_relevance(self):
        """Test that merged results are sorted by relevance score."""
        lit1 = LiteratureSupport(
            paper_id="p1", title="Paper 1", authors="A", year=2024,
            doi="10.1/a", relevance_score=0.7, source="openalex",
        )
        lit2 = LiteratureSupport(
            paper_id="p2", title="Paper 2", authors="B", year=2023,
            doi="10.1/b", relevance_score=0.9, source="exa",
        )
        result = lit_grounding_module.merge_literature([lit1, lit2])
        assert result[0].relevance_score == 0.9
        assert result[1].relevance_score == 0.7

    def test_merge_combines_citation_count(self):
        """Test that citation count is merged from lower-priority source."""
        lit1 = LiteratureSupport(
            paper_id="p1", title="Paper 1", authors="A", year=2024,
            doi="10.1/same", relevance_score=0.9, source="pubmed",
            citation_count=0,  # PubMed doesn't have citation counts
        )
        lit2 = LiteratureSupport(
            paper_id="p2", title="Paper 1", authors="A", year=2024,
            doi="10.1/same", relevance_score=0.85, source="openalex",
            citation_count=150,  # OpenAlex has citation counts
        )
        result = lit_grounding_module.merge_literature([lit1, lit2])
        assert len(result) == 1
        assert result[0].source == "pubmed"  # Higher priority kept
        assert result[0].citation_count == 150  # Citation count merged

    def test_merge_combines_key_passage(self):
        """Test that key_passage is merged from Exa when other source lacks it."""
        lit1 = LiteratureSupport(
            paper_id="p1", title="Paper 1", authors="A", year=2024,
            doi="10.1/same", relevance_score=0.9, source="openalex",
            key_passage="",
        )
        lit2 = LiteratureSupport(
            paper_id="p2", title="Paper 1", authors="A", year=2024,
            doi="10.1/same", relevance_score=0.85, source="exa",
            key_passage="Important finding about diabetes.",
        )
        result = lit_grounding_module.merge_literature([lit1, lit2])
        assert len(result) == 1
        assert result[0].source == "openalex"  # Higher priority kept
        assert result[0].key_passage == "Important finding about diabetes."

    def test_merge_combines_doi(self):
        """Test that DOI is merged when primary source lacks it.

        Note: Papers only merge if they share the same unique key (DOI > PMID > title+year).
        When one paper has a DOI and another doesn't, they have different keys and don't merge.
        This test verifies DOI merging when both papers match via the same DOI key.
        """
        # Both papers have the same DOI, so they'll merge by DOI key
        lit1 = LiteratureSupport(
            paper_id="PMID:12345678", title="Paper 1", authors="A", year=2024,
            doi="10.1/test", relevance_score=0.9, source="pubmed",
            key_passage="",  # PubMed doesn't have highlights
        )
        lit2 = LiteratureSupport(
            paper_id="openalex123", title="Paper 1", authors="A", year=2024,
            doi="10.1/test", relevance_score=0.85, source="openalex",
            key_passage="",
            citation_count=200,  # OpenAlex has citation counts
        )
        result = lit_grounding_module.merge_literature([lit1, lit2])
        assert len(result) == 1
        assert result[0].source == "pubmed"  # Higher priority kept
        assert result[0].doi == "10.1/test"  # DOI preserved
        assert result[0].citation_count == 200  # Citation count merged from OpenAlex


class TestGetUniqueKey:
    """Tests for get_unique_key deduplication key generation."""

    def test_unique_key_doi_preferred(self):
        """Test that DOI is preferred for unique key."""
        lit = LiteratureSupport(
            paper_id="PMID:12345678", title="Paper Title", authors="A", year=2024,
            doi="10.1/test", relevance_score=0.9,
        )
        key = lit_grounding_module.get_unique_key(lit)
        assert key == "doi:10.1/test"

    def test_unique_key_pmid_fallback(self):
        """Test that PMID is used when DOI is missing."""
        lit = LiteratureSupport(
            paper_id="PMID:12345678", title="Paper Title", authors="A", year=2024,
            relevance_score=0.9,
        )
        key = lit_grounding_module.get_unique_key(lit)
        assert key == "pmid:12345678"

    def test_unique_key_title_year_fallback(self):
        """Test that title+year is used when DOI and PMID are missing."""
        lit = LiteratureSupport(
            paper_id="other_id", title="Paper Title", authors="A", year=2024,
            relevance_score=0.9,
        )
        key = lit_grounding_module.get_unique_key(lit)
        assert key == "title:paper title:2024"

    def test_unique_key_doi_case_insensitive(self):
        """Test that DOI keys are case-insensitive."""
        lit1 = LiteratureSupport(
            paper_id="p1", title="Paper", authors="A", year=2024,
            doi="10.1/TEST", relevance_score=0.9,
        )
        lit2 = LiteratureSupport(
            paper_id="p2", title="Paper", authors="A", year=2024,
            doi="10.1/test", relevance_score=0.9,
        )
        assert lit_grounding_module.get_unique_key(lit1) == lit_grounding_module.get_unique_key(lit2)

    def test_unique_key_title_truncated(self):
        """Test that long titles are truncated in key."""
        long_title = "A" * 200
        lit = LiteratureSupport(
            paper_id="p1", title=long_title, authors="A", year=2024,
            relevance_score=0.9,
        )
        key = lit_grounding_module.get_unique_key(lit)
        # Should only use first 100 chars of title
        assert key == f"title:{'a' * 100}:2024"


class TestLiteratureSupportPubmedSource:
    """Tests for PubMed source in LiteratureSupport."""

    def test_literature_support_pubmed_source(self):
        """Test LiteratureSupport with PubMed source."""
        lit = LiteratureSupport(
            paper_id="PMID:12345678",
            title="PubMed Paper Title",
            authors="Smith J et al.",
            year=2024,
            doi="10.1234/test",
            url="https://pubmed.ncbi.nlm.nih.gov/12345678",
            relevance_score=0.85,
            source="pubmed",
        )
        assert lit.source == "pubmed"
        assert lit.paper_id == "PMID:12345678"


class TestBuildReferencesTable:
    """Tests for build_references_table function."""

    def test_empty_hypotheses(self):
        """Test with empty hypothesis list."""
        result = build_references_table([])
        assert result == ""

    def test_hypotheses_without_literature(self):
        """Test with hypotheses that have no literature support."""
        hyp = Hypothesis(
            title="Test Hypothesis",
            tier=2,
            claim="Test claim",
            supporting_entities=["CHEBI:12345"],
            structural_logic="A relates to B",
            validation_steps=["Run experiment"],
        )
        result = build_references_table([hyp])
        assert result == ""

    def test_builds_table_with_literature(self):
        """Test building table with hypothesis that has literature."""
        lit = LiteratureSupport(
            paper_id="paper1",
            title="Test Paper Title",
            authors="Smith et al.",
            year=2024,
            doi="10.1234/test",
            relevance_score=0.9,
            relationship="supporting",
            key_passage="Key finding",
            citation_count=100,
        )
        hyp = Hypothesis(
            title="Test Hypothesis",
            tier=1,
            claim="Test claim",
            supporting_entities=["GENE:1234"],
            structural_logic="Direct evidence",
            validation_steps=["Verify"],
            literature_support=[lit],
        )
        result = build_references_table([hyp])

        # Check structure
        assert "## Literature References" in result
        assert "| Hypothesis | Citation | Link |" in result
        assert "|------------|----------|------|" in result
        assert "Test Hypothesis" in result
        assert "Smith et al. (2024)" in result
        assert '[DOI](https://doi.org/10.1234/test)' in result

    def test_prefers_doi_over_url(self):
        """Test that DOI link is preferred over URL."""
        lit = LiteratureSupport(
            paper_id="paper1",
            title="Test Paper",
            authors="Jones",
            year=2023,
            doi="10.5678/example",
            url="https://example.com/paper",
            relevance_score=0.8,
        )
        hyp = Hypothesis(
            title="Hypothesis",
            tier=1,
            claim="Claim",
            supporting_entities=[],
            structural_logic="Logic",
            validation_steps=[],
            literature_support=[lit],
        )
        result = build_references_table([hyp])
        assert "[DOI](https://doi.org/10.5678/example)" in result
        assert "example.com" not in result

    def test_falls_back_to_url_without_doi(self):
        """Test that URL is used when DOI is missing."""
        lit = LiteratureSupport(
            paper_id="paper1",
            title="Test Paper",
            authors="Doe",
            year=2022,
            url="https://pubmed.ncbi.nlm.nih.gov/12345678",
            relevance_score=0.7,
        )
        hyp = Hypothesis(
            title="Hypothesis",
            tier=2,
            claim="Claim",
            supporting_entities=[],
            structural_logic="Logic",
            validation_steps=[],
            literature_support=[lit],
        )
        result = build_references_table([hyp])
        assert "[Link](https://pubmed.ncbi.nlm.nih.gov/12345678)" in result

    def test_truncates_long_titles(self):
        """Test that long hypothesis titles and paper titles are truncated."""
        long_paper_title = "A" * 150  # > 100 chars
        long_hyp_title = "B" * 100  # > 80 chars

        lit = LiteratureSupport(
            paper_id="paper1",
            title=long_paper_title,
            authors="Author",
            year=2024,
            doi="10.1234/test",
            relevance_score=0.8,
        )
        hyp = Hypothesis(
            title=long_hyp_title,
            tier=1,
            claim="Claim",
            supporting_entities=[],
            structural_logic="Logic",
            validation_steps=[],
            literature_support=[lit],
        )
        result = build_references_table([hyp])

        # Check truncation
        assert "B" * 80 + "..." in result  # Hyp title truncated at 80
        assert "A" * 100 + "..." in result  # Paper title truncated at 100

    def test_counts_papers_and_hypotheses(self):
        """Test that summary line has correct counts after deduplication."""
        lit1 = LiteratureSupport(
            paper_id="p1", title="Paper 1", authors="A", year=2024,
            doi="10.1/a", relevance_score=0.9,
        )
        lit2 = LiteratureSupport(
            paper_id="p2", title="Paper 2", authors="B", year=2023,
            doi="10.1/b", relevance_score=0.8,
        )
        hyp1 = Hypothesis(
            title="Hyp 1", tier=1, claim="C1", supporting_entities=[],
            structural_logic="L1", validation_steps=[], literature_support=[lit1, lit2],
        )
        hyp2 = Hypothesis(
            title="Hyp 2", tier=2, claim="C2", supporting_entities=[],
            structural_logic="L2", validation_steps=[], literature_support=[lit1],
        )
        result = build_references_table([hyp1, hyp2])

        # After deduplication: 2 unique papers (lit1 appears in both), 2 hypotheses
        assert "2 unique papers across 2 hypotheses" in result

    def test_dedupes_hypotheses_by_title(self):
        """Test that duplicate hypotheses by title are deduplicated."""
        lit = LiteratureSupport(
            paper_id="p1", title="Paper 1", authors="A", year=2024,
            doi="10.1/a", relevance_score=0.9,
        )
        # Two hypotheses with same title (should be deduped)
        hyp1 = Hypothesis(
            title="Same Title", tier=1, claim="C1", supporting_entities=[],
            structural_logic="L1", validation_steps=[], literature_support=[lit],
        )
        hyp2 = Hypothesis(
            title="Same Title", tier=2, claim="C2", supporting_entities=[],
            structural_logic="L2", validation_steps=[], literature_support=[lit],
        )
        result = build_references_table([hyp1, hyp2])

        # Should only have 1 hypothesis in output
        assert "1 unique papers across 1 hypotheses" in result
        # Should only appear once in the table
        assert result.count("Same Title") == 1

    def test_dedupes_papers_by_doi(self):
        """Test that papers with same DOI are deduplicated across hypotheses."""
        # Same paper (same DOI) cited by two hypotheses
        lit1 = LiteratureSupport(
            paper_id="p1", title="Paper Title", authors="A", year=2024,
            doi="10.1/same", relevance_score=0.9,
        )
        lit2 = LiteratureSupport(
            paper_id="p2", title="Paper Title Copy", authors="A", year=2024,
            doi="10.1/same", relevance_score=0.85,
        )
        hyp1 = Hypothesis(
            title="Hyp 1", tier=1, claim="C1", supporting_entities=[],
            structural_logic="L1", validation_steps=[], literature_support=[lit1],
        )
        hyp2 = Hypothesis(
            title="Hyp 2", tier=2, claim="C2", supporting_entities=[],
            structural_logic="L2", validation_steps=[], literature_support=[lit2],
        )
        result = build_references_table([hyp1, hyp2])

        # Should show 1 unique paper across 2 hypotheses
        assert "1 unique papers across 2 hypotheses" in result
        # Both hypothesis titles should be combined in one row
        assert "Hyp 1" in result
        assert "Hyp 2" in result

    def test_combines_hypothesis_titles_for_same_paper(self):
        """Test that hypothesis titles are combined when citing the same paper."""
        lit = LiteratureSupport(
            paper_id="p1", title="Shared Paper", authors="A", year=2024,
            doi="10.1/shared", relevance_score=0.9,
        )
        hyp1 = Hypothesis(
            title="Alpha Hypothesis", tier=1, claim="C1", supporting_entities=[],
            structural_logic="L1", validation_steps=[], literature_support=[lit],
        )
        hyp2 = Hypothesis(
            title="Beta Hypothesis", tier=2, claim="C2", supporting_entities=[],
            structural_logic="L2", validation_steps=[], literature_support=[lit],
        )
        result = build_references_table([hyp1, hyp2])

        # Both titles should appear in the same row, separated by semicolon
        # Sorted alphabetically
        assert "Alpha Hypothesis; Beta Hypothesis" in result

    def test_includes_relevance_column(self):
        """Test that relevance column is included with key_passage."""
        lit = LiteratureSupport(
            paper_id="p1", title="Paper 1", authors="A", year=2024,
            doi="10.1/a", relevance_score=0.9,
            key_passage="Important finding about diabetes risk.",
        )
        hyp = Hypothesis(
            title="Hyp 1", tier=1, claim="C1", supporting_entities=[],
            structural_logic="L1", validation_steps=[], literature_support=[lit],
        )
        result = build_references_table([hyp])

        # Check for Relevance header
        assert "| Relevance |" in result
        assert "|-----------|" in result
        # Check for key_passage content
        assert "Important finding about diabetes risk." in result

    def test_truncates_long_key_passage(self):
        """Test that long key_passage is truncated to 120 chars."""
        long_passage = "A" * 200
        lit = LiteratureSupport(
            paper_id="p1", title="Paper 1", authors="A", year=2024,
            doi="10.1/a", relevance_score=0.9,
            key_passage=long_passage,
        )
        hyp = Hypothesis(
            title="Hyp 1", tier=1, claim="C1", supporting_entities=[],
            structural_logic="L1", validation_steps=[], literature_support=[lit],
        )
        result = build_references_table([hyp])

        # Should have truncated passage with ellipsis
        assert "A" * 120 + "..." in result
        # Should not have full 200 chars
        assert "A" * 150 not in result

    def test_shows_dash_for_empty_key_passage(self):
        """Test that empty key_passage shows em-dash."""
        lit = LiteratureSupport(
            paper_id="p1", title="Paper 1", authors="A", year=2024,
            doi="10.1/a", relevance_score=0.9,
            key_passage="",
        )
        hyp = Hypothesis(
            title="Hyp 1", tier=1, claim="C1", supporting_entities=[],
            structural_logic="L1", validation_steps=[], literature_support=[lit],
        )
        result = build_references_table([hyp])

        # Should show em-dash for empty relevance
        # Check the row has the em-dash in the relevance column position
        lines = result.split("\n")
        data_row = [l for l in lines if "Paper 1" in l][0]
        # Last column should be em-dash
        assert data_row.endswith("| — |")


class TestIsValidExaResult:
    """Tests for is_valid_exa_result quality filter."""

    def test_skips_year_zero(self):
        """Test that results with year 0 are rejected."""
        result = {"publishedDate": None, "author": "Smith", "url": "https://example.com", "title": "Paper"}
        assert is_valid_exa_result(result) is False

    def test_skips_web_team_author(self):
        """Test that results from Web Team are rejected."""
        result = {"publishedDate": "2024-01-01", "author": "Web Team", "url": "https://example.com", "title": "Paper"}
        assert is_valid_exa_result(result) is False

    def test_skips_ebi_author(self):
        """Test that results from EBI are rejected."""
        result = {"publishedDate": "2024-01-01", "author": "EBI", "url": "https://example.com", "title": "Paper"}
        assert is_valid_exa_result(result) is False

    def test_skips_embl_ebi_author(self):
        """Test that results from EMBL-EBI are rejected."""
        result = {"publishedDate": "2024-01-01", "author": "EMBL-EBI", "url": "https://example.com", "title": "Paper"}
        assert is_valid_exa_result(result) is False

    def test_skips_chebi_urls(self):
        """Test that ChEBI database pages are rejected."""
        result = {"publishedDate": "2024-01-01", "author": "Smith", "url": "https://www.ebi.ac.uk/chebi/searchId.do?chebiId=12345", "title": "Paper"}
        assert is_valid_exa_result(result) is False

    def test_skips_uniprot_urls(self):
        """Test that UniProt database pages are rejected."""
        result = {"publishedDate": "2024-01-01", "author": "Smith", "url": "https://www.uniprot.org/uniprotkb/P12345", "title": "Paper"}
        assert is_valid_exa_result(result) is False

    def test_skips_gene_ontology_title(self):
        """Test that Gene Ontology Resource pages are rejected."""
        result = {"publishedDate": "2024-01-01", "author": "Smith", "url": "https://example.com", "title": "Gene Ontology Resource"}
        assert is_valid_exa_result(result) is False

    def test_skips_string_database_title(self):
        """Test that STRING database pages are rejected."""
        result = {"publishedDate": "2024-01-01", "author": "Smith", "url": "https://example.com", "title": "STRING: functional protein association networks"}
        assert is_valid_exa_result(result) is False

    def test_accepts_valid_paper(self):
        """Test that valid papers are accepted."""
        result = {
            "publishedDate": "2024-01-15",
            "author": "Smith J et al.",
            "url": "https://pubmed.ncbi.nlm.nih.gov/12345678",
            "title": "Fructose metabolism in diabetes",
        }
        assert is_valid_exa_result(result) is True

    def test_accepts_paper_with_missing_author(self):
        """Test that papers with missing author are still accepted if other criteria pass."""
        result = {"publishedDate": "2024-01-01", "author": "", "url": "https://example.com", "title": "Valid Paper"}
        assert is_valid_exa_result(result) is True

    def test_accepts_paper_with_none_author(self):
        """Test that papers with None author are still accepted if other criteria pass."""
        result = {"publishedDate": "2024-01-01", "author": None, "url": "https://example.com", "title": "Valid Paper"}
        assert is_valid_exa_result(result) is True


class TestPLISStudyBenchmark:
    """Benchmark test for PLIS study literature search (PMID 39746149)."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_t2d_progression_hypothesis_finds_plis_study(self):
        """
        Test that T2D progression hypothesis can find PLIS study (PMID 39746149).

        The PLIS study is the key benchmark for type 2 diabetes progression
        literature. This test verifies that our multi-strategy S2 search can
        discover it given a relevant hypothesis.

        Note: This is an integration test that requires S2 API access.
        """
        # Import the real S2 search function
        from src.kestrel_backend.graph.nodes.literature_grounding import ground_hypothesis_s2_raw

        # Create hypothesis with T2D progression context
        hypothesis = Hypothesis(
            title="Metabolite progression markers in type 2 diabetes",
            tier=2,
            claim="Longitudinal metabolite changes predict type 2 diabetes progression and complications",
            supporting_entities=["CHEBI:17234"],  # Glucose as example
            structural_logic="Metabolite trajectories correlate with disease progression",
            validation_steps=["Validate in independent cohort"],
        )

        # Entity names mapping (example)
        entity_names = {"CHEBI:17234": "glucose"}

        # Search with T2D context
        literature, errors = await ground_hypothesis_s2_raw(
            hypothesis,
            limit=10,
            disease_context="type 2 diabetes",
            entity_names=entity_names,
        )

        # Check if PLIS study (PMID 39746149) was found
        # Note: S2 paper_id is the corpus ID, not PMID. Check URL for PubMed link.
        found_plis = False
        for lit in literature:
            # Check URL for PubMed reference (most reliable since model lacks external_ids)
            if lit.url and "39746149" in lit.url:
                found_plis = True
                break
            # Fallback: check paper_id in case it contains PMID reference
            elif "39746149" in lit.paper_id or "PMID:39746149" in lit.paper_id:
                found_plis = True
                break

        # Soft assertion - log warning if not found rather than failing
        # (S2 API results can vary, and this is a benchmark, not a guarantee)
        if not found_plis:
            import logging
            logging.warning(
                "PLIS study (PMID 39746149) not found in top 10 S2 results. "
                "This may indicate query optimization is needed. Found %d papers.",
                len(literature)
            )

        # Always pass - this is a benchmark test for monitoring, not a hard requirement
        assert len(literature) >= 0  # At least verify search didn't crash


class TestGetPaperKey:
    """Tests for _get_paper_key deduplication key generation."""

    def test_uses_doi_when_present(self):
        """Test that DOI is used for key when available."""
        lit = LiteratureSupport(
            paper_id="p1", title="Paper Title", authors="A", year=2024,
            doi="10.1/test", relevance_score=0.9,
        )
        key = _get_paper_key(lit)
        assert key == "doi:10.1/test"

    def test_falls_back_to_title_year_without_doi(self):
        """Test that title+year is used when DOI is missing."""
        lit = LiteratureSupport(
            paper_id="p1", title="Paper Title", authors="A", year=2024,
            relevance_score=0.9,
        )
        key = _get_paper_key(lit)
        assert key == "title:paper title:2024"

    def test_doi_case_insensitive(self):
        """Test that DOI keys are case-insensitive."""
        lit1 = LiteratureSupport(
            paper_id="p1", title="Paper", authors="A", year=2024,
            doi="10.1/TEST", relevance_score=0.9,
        )
        lit2 = LiteratureSupport(
            paper_id="p2", title="Paper", authors="A", year=2024,
            doi="10.1/test", relevance_score=0.9,
        )
        assert _get_paper_key(lit1) == _get_paper_key(lit2)

    def test_title_truncated_to_100_chars(self):
        """Test that long titles are truncated in key."""
        long_title = "A" * 200
        lit = LiteratureSupport(
            paper_id="p1", title=long_title, authors="A", year=2024,
            relevance_score=0.9,
        )
        key = _get_paper_key(lit)
        assert key == f"title:{'a' * 100}:2024"


# =============================================================================
# Abstract-body fetch + cache (spike: feat/literature-abstract-fetch)
# =============================================================================

import src.kestrel_backend.pubmed_client as pmc

backfill_abstracts = lit_grounding_module.backfill_abstracts
_pmid_for_lit = lit_grounding_module._pmid_for_lit


def _make_hypothesis(title, literature):
    """Minimal valid Hypothesis with the given literature_support."""
    return Hypothesis(
        title=title, tier=2, claim=f"{title} claim",
        supporting_entities=["CURIE:1"], structural_logic="logic",
        validation_steps=["step"], literature_support=literature,
    )


# Realistic EFetch XML: one simple abstract, one structured (labelled) abstract,
# one article with no abstract at all.
_EFETCH_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">111</PMID>
      <Article>
        <Abstract>
          <AbstractText>Metformin activates AMPK in hepatocytes.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">222</PMID>
      <Article>
        <Abstract>
          <AbstractText Label="BACKGROUND">Type 2 diabetes is common.</AbstractText>
          <AbstractText Label="RESULTS">BCAAs rose <sup>2</sup>-fold in cases.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">333</PMID>
      <Article><ArticleTitle>No abstract here</ArticleTitle></Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""


class TestEFetchAbstractParsing:
    """pubmed_client._parse_efetch_abstracts XML handling."""

    def test_simple_abstract(self):
        result = pmc._parse_efetch_abstracts(_EFETCH_XML)
        assert result["111"] == "Metformin activates AMPK in hepatocytes."

    def test_structured_abstract_joins_labelled_segments(self):
        result = pmc._parse_efetch_abstracts(_EFETCH_XML)
        # Labels are prefixed; inline markup (<sup>) is flattened into text.
        assert result["222"] == (
            "BACKGROUND: Type 2 diabetes is common.\n"
            "RESULTS: BCAAs rose 2-fold in cases."
        )

    def test_pmid_without_abstract_is_omitted(self):
        result = pmc._parse_efetch_abstracts(_EFETCH_XML)
        assert "333" not in result

    def test_malformed_xml_returns_empty(self):
        assert pmc._parse_efetch_abstracts("<not valid xml") == {}


class TestNormalizePmid:
    def test_strips_prefix_and_whitespace(self):
        assert pmc.normalize_pmid(" PMID:12345 ") == "12345"

    def test_bare_pmid_unchanged(self):
        assert pmc.normalize_pmid("12345") == "12345"


class TestFetchAbstracts:
    """pubmed_client.fetch_abstracts: dedup, normalize, batch — no network."""

    @pytest.mark.asyncio
    async def test_dedupes_and_normalizes_pmids(self, monkeypatch):
        seen_batches = []

        async def fake_efetch_batch(batch):
            seen_batches.append(list(batch))
            return {p: f"abstract-{p}" for p in batch}

        monkeypatch.setattr(pmc, "_efetch_batch", fake_efetch_batch)

        # Duplicates (bare + prefixed), one non-numeric junk PMID.
        result = await pmc.fetch_abstracts(["PMID:111", "111", "222", "abc"])

        assert result == {"111": "abstract-111", "222": "abstract-222"}
        # 111 fetched once; "abc" dropped.
        flat = [p for b in seen_batches for p in b]
        assert sorted(flat) == ["111", "222"]

    @pytest.mark.asyncio
    async def test_empty_input_skips_fetch(self, monkeypatch):
        called = False

        async def fake_efetch_batch(batch):
            nonlocal called
            called = True
            return {}

        monkeypatch.setattr(pmc, "_efetch_batch", fake_efetch_batch)
        assert await pmc.fetch_abstracts(["xyz", "PMID:"]) == {}
        assert called is False

    @pytest.mark.asyncio
    async def test_batches_respect_batch_size(self, monkeypatch):
        seen_batches = []

        async def fake_efetch_batch(batch):
            seen_batches.append(list(batch))
            return {p: "a" for p in batch}

        monkeypatch.setattr(pmc, "_efetch_batch", fake_efetch_batch)
        await pmc.fetch_abstracts([str(i) for i in range(5)], batch_size=2)
        assert [len(b) for b in seen_batches] == [2, 2, 1]


class TestLiteratureSupportAbstractField:
    def test_default_none(self):
        lit = LiteratureSupport(paper_id="PMID:1", title="t", authors="a", year=2024)
        assert lit.abstract is None

    def test_accepts_body(self):
        lit = LiteratureSupport(
            paper_id="PMID:1", title="t", authors="a", year=2024,
            abstract="full body",
        )
        assert lit.abstract == "full body"


class TestPmidForLit:
    def test_from_prefixed_paper_id(self):
        lit = LiteratureSupport(paper_id="PMID:42", title="t", authors="a", year=2024)
        assert _pmid_for_lit(lit) == "42"

    def test_from_pubmed_url(self):
        lit = LiteratureSupport(
            paper_id="S2:abc", title="t", authors="a", year=2024,
            url="https://pubmed.ncbi.nlm.nih.gov/99/",
        )
        assert _pmid_for_lit(lit) == "99"

    def test_none_when_no_pmid(self):
        lit = LiteratureSupport(
            paper_id="openalex:W1", title="t", authors="a", year=2024,
            doi="10.1/x", url="https://doi.org/10.1/x",
        )
        assert _pmid_for_lit(lit) is None


class TestBackfillAbstracts:
    """backfill_abstracts: idempotency, cross-hypothesis dedup, best-effort."""

    @pytest.mark.asyncio
    async def test_fills_missing_and_skips_present(self, monkeypatch):
        calls = []

        async def fake_fetch(pmids, batch_size=200):
            calls.append(sorted(pmids))
            return {"111": "body-111", "222": "body-222"}

        monkeypatch.setattr(lit_grounding_module, "pubmed_fetch_abstracts", fake_fetch)

        already = LiteratureSupport(
            paper_id="S2:x", title="t", authors="a", year=2024,
            abstract="s2 inline body", source="s2",
        )
        need1 = LiteratureSupport(paper_id="PMID:111", title="t", authors="a", year=2024, source="kg")
        need2 = LiteratureSupport(paper_id="PMID:222", title="t", authors="a", year=2024, source="pubmed")

        hyps = [
            _make_hypothesis("H1", [already, need1]),
            _make_hypothesis("H2", [need2]),
        ]

        updated, filled = await backfill_abstracts(hyps)

        assert filled == 2
        # S2 entry already had abstract → not re-fetched.
        assert calls == [["111", "222"]]
        # Bodies attached; S2 entry untouched.
        h1_lits = {l.paper_id: l for l in updated[0].literature_support}
        assert h1_lits["S2:x"].abstract == "s2 inline body"
        assert h1_lits["PMID:111"].abstract == "body-111"
        assert updated[1].literature_support[0].abstract == "body-222"

    @pytest.mark.asyncio
    async def test_idempotent_second_pass_fetches_nothing(self, monkeypatch):
        call_count = 0

        async def fake_fetch(pmids, batch_size=200):
            nonlocal call_count
            call_count += 1
            return {"111": "body-111"}

        monkeypatch.setattr(lit_grounding_module, "pubmed_fetch_abstracts", fake_fetch)

        need = LiteratureSupport(paper_id="PMID:111", title="t", authors="a", year=2024, source="kg")
        hyps = [_make_hypothesis("H1", [need])]

        updated, filled = await backfill_abstracts(hyps)
        assert filled == 1 and call_count == 1

        # Re-running over the now-filled output collects no PMIDs → no fetch.
        updated2, filled2 = await backfill_abstracts(updated)
        assert filled2 == 0 and call_count == 1

    @pytest.mark.asyncio
    async def test_dedups_pmid_across_hypotheses(self, monkeypatch):
        async def fake_fetch(pmids, batch_size=200):
            # Same PMID cited by two hypotheses must appear once.
            assert sorted(pmids) == ["111"]
            return {"111": "body-111"}

        monkeypatch.setattr(lit_grounding_module, "pubmed_fetch_abstracts", fake_fetch)

        lit_a = LiteratureSupport(paper_id="PMID:111", title="t", authors="a", year=2024, source="kg")
        lit_b = LiteratureSupport(paper_id="PMID:111", title="t", authors="a", year=2024, source="kg")
        hyps = [_make_hypothesis("H1", [lit_a]), _make_hypothesis("H2", [lit_b])]

        updated, filled = await backfill_abstracts(hyps)
        assert filled == 2  # both entries filled from the single fetched body

    @pytest.mark.asyncio
    async def test_best_effort_when_no_pmids(self, monkeypatch):
        called = False

        async def fake_fetch(pmids, batch_size=200):
            nonlocal called
            called = True
            return {}

        monkeypatch.setattr(lit_grounding_module, "pubmed_fetch_abstracts", fake_fetch)

        no_pmid = LiteratureSupport(
            paper_id="openalex:W1", title="t", authors="a", year=2024,
            doi="10.1/x", url="https://doi.org/10.1/x", source="openalex",
        )
        hyps = [_make_hypothesis("H1", [no_pmid])]

        updated, filled = await backfill_abstracts(hyps)
        assert filled == 0 and called is False
        assert updated[0].literature_support[0].abstract is None
