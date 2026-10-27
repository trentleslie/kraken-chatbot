"""Unit 1 — rank inference from analyte labels (pure).

v1 is taxa-only: a label's requested taxonomic rank is derived from its *shape*
(binomial / single capitalized token / -aceae / -ales suffix). Non-taxa labels
(metabolites, genes, diseases) and ambiguous/garbage input return ``None`` so the
rank guard stays inert and the resolution path is byte-identical.
"""

import pytest

from kestrel_backend.graph.rank_utils import Rank, infer_requested_rank, is_rank_collapse


class TestInferRequestedRank:
    def test_binomial_is_species(self):
        assert infer_requested_rank("Ruminococcus gnavus") == Rank.SPECIES

    def test_single_capitalized_token_is_genus(self):
        assert infer_requested_rank("Ruminococcus") == Rank.GENUS

    def test_aceae_suffix_is_family(self):
        assert infer_requested_rank("Ruminococcaceae") == Rank.FAMILY

    def test_ales_suffix_is_order(self):
        assert infer_requested_rank("Clostridiales") == Rank.ORDER

    @pytest.mark.parametrize(
        "label",
        [
            "glucose",                     # lowercase metabolite
            "N-lactoylphenylalanine",      # chemical name
            "gamma-glutamylvaline",        # dipeptide
            "KIF6",                        # gene symbol (all caps + digit)
            "APOE",                        # gene symbol (all caps)
            "IL6",                         # gene symbol
            "vitamin D",                   # two tokens, first lowercase
        ],
    )
    def test_non_taxa_labels_return_none(self, label):
        assert infer_requested_rank(label) is None

    @pytest.mark.parametrize("label", ["", "   ", None, "123", "!!", "a b c"])
    def test_empty_or_garbage_returns_none(self, label):
        assert infer_requested_rank(label) is None

    def test_rank_ordering_species_is_finer_than_genus(self):
        assert Rank.SPECIES > Rank.GENUS > Rank.FAMILY > Rank.ORDER

    def test_second_binomial_example(self):
        assert infer_requested_rank("Faecalibacterium prausnitzii") == Rank.SPECIES


class TestIsRankCollapse:
    def test_species_resolving_to_genus_is_collapse(self):
        assert is_rank_collapse("Ruminococcus gnavus", "Ruminococcus") is True

    def test_genus_resolving_to_genus_is_not_collapse(self):
        assert is_rank_collapse("Ruminococcus", "Ruminococcus") is False

    def test_species_resolving_to_same_species_is_not_collapse(self):
        assert is_rank_collapse("Ruminococcus gnavus", "Ruminococcus gnavus") is False

    def test_species_resolving_to_different_genus_is_not_collapse(self):
        # A wrong-genus resolution is a mis-resolution, not a rank collapse (different concern).
        assert is_rank_collapse("Ruminococcus gnavus", "Blautia") is False

    def test_non_taxa_never_collapses(self):
        assert is_rank_collapse("glucose", "glucose-6-phosphate") is False

    def test_none_resolved_name_never_collapses(self):
        assert is_rank_collapse("Ruminococcus gnavus", None) is False

    def test_lowercase_resolved_name_has_no_rank_so_not_collapse(self):
        # KG canonical names are Titlecase; a non-canonical lowercase name yields no rank and the
        # guard stays inert (never a false-positive abstain).
        assert is_rank_collapse("Ruminococcus gnavus", "ruminococcus") is False
