"""Unit 3: verify the frailty multiomic benchmark prompts parse through the EXISTING
intake regex into the intended analytes.

This is the correctness gate for the generated prompts (Units 1-2): the prompts are only
useful as benchmark inputs if intake.extract_entities() extracts the analytes we put in,
without fragmenting comma-bearing metabolite names and without leaking Chemistry analytes.
"""

import importlib.util
import json
import re
from pathlib import Path

import pytest

from kestrel_backend.graph.nodes.intake import extract_entities

_HERE = Path(__file__).resolve()
_ASSESS = _HERE.parents[1] / "assessment_data"
_QUERIES = _ASSESS / "queries.json"
PATH_TYPE = "multiomic-module"

# Load the generator module from its file path (assessment_data is not a package).
_spec = importlib.util.spec_from_file_location(
    "generate_frailty_prompts", _ASSESS / "generate_frailty_prompts.py"
)
genmod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(genmod)

_ROWS = list(__import__("csv").DictReader(genmod.MODULES_TSV.open(), delimiter="\t"))

# A "fragment" = a token that is a piece of lipid shorthand or a split parenthetical,
# e.g. "d18:1", "18:2", "23:0)", or a token with a close-paren but no open-paren.
_FRAGMENT = re.compile(r"^[dtoe]?\d+:\d+|^\d+:\d+|\)\s*$")
_CURIE = re.compile(r"^(NCBIGene|CHEBI|UniProtKB|HMDB|KEGG):[A-Za-z0-9:.\-]+$")


def _multiomic_entries():
    entries = [e for e in json.loads(_QUERIES.read_text()) if e.get("path_type") == PATH_TYPE]
    return entries


def _module_of(entry) -> str:
    return re.search(r'"(\w+)" module', entry["query"]).group(1)


def test_six_modules_present():
    mods = {_module_of(e) for e in _multiomic_entries()}
    assert mods == set(genmod.CURATED_MODULES)


@pytest.mark.parametrize("entry", _multiomic_entries(), ids=_module_of)
class TestPromptParsing:
    def test_extracts_nonempty(self, entry):
        assert len(extract_entities(entry["query"])) > 0

    def test_all_expected_proteins_extracted(self, entry):
        """Protein GeneSymbols are clean tokens and must each be extracted exactly."""
        module = _module_of(entry)
        proteins, _ = genmod.select_analytes(_ROWS, module)
        extracted = set(extract_entities(entry["query"]))
        missing = [p for p in proteins if p not in extracted]
        assert not missing, f"{module}: proteins missing from extraction: {missing}"

    def test_no_fragmented_tokens(self, entry):
        """Comma-bearing metabolite names must not be split into lipid-shorthand fragments."""
        frags = [e for e in extract_entities(entry["query"]) if _FRAGMENT.search(e)]
        assert not frags, f"{_module_of(entry)}: fragmented tokens: {frags}"

    def test_no_chemistry_leaked(self, entry):
        """Chemistry analytes (excluded by the generator) must not appear in extraction."""
        module = _module_of(entry)
        chem_names = {
            r["ChemName"].strip()
            for r in _ROWS
            if r["ModuleID"] == module and r["Dataset"] == "Chemistry"
            and r["ChemName"].strip() not in genmod._NA
        }
        extracted = set(extract_entities(entry["query"]))
        leaked = chem_names & extracted
        assert not leaked, f"{module}: chemistry analytes leaked: {leaked}"

    def test_expected_curies_well_formed(self, entry):
        for c in entry.get("expected_curies", []):
            assert _CURIE.match(c), f"malformed expected_curie: {c!r}"
