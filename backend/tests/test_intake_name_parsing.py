"""Phase 1 (intake analyte-name robustness): labeled-section comma/newline parsing.

Plan/requirements: docs/brainstorms/2026-06-22-intake-analyte-name-robustness-requirements.md

R1 (no fragmentation): an internal comma in a chemical name must not split it.
R2 (unambiguous delimiter): a newline-delimited section is one analyte per line (commas safe);
a single-line (legacy comma-joined) section still splits on commas.
"""

from kestrel_backend.graph.nodes.intake import extract_entities


def test_internal_comma_name_kept_when_newline_delimited():
    # Newline-delimited section: each line is one analyte, so the internal comma survives.
    query = "Metabolites:\nglucose\n12,13-DiHOME\nlactate\n"
    ents = extract_entities(query)
    assert "12,13-DiHOME" in ents          # R1: not fragmented into "12" + "13-DiHOME"
    assert "12" not in ents and "13-DiHOME" not in ents
    assert "glucose" in ents and "lactate" in ents


def test_newline_delimited_multiple_sections():
    query = "Proteins:\nTNFRSF10A\nFIGF\n\nMetabolites:\nglucose\n9,10-DiHOME\n"
    ents = extract_entities(query)
    assert ents == ["TNFRSF10A", "FIGF", "glucose", "9,10-DiHOME"]


def test_legacy_single_line_comma_section_still_splits():
    # Single-line comma-joined content (the prior harness format) still splits on commas.
    query = "Metabolites:\nglucose, fructose, lactate\n"
    ents = extract_entities(query)
    assert ents == ["glucose", "fructose", "lactate"]


def test_harness_newline_style_query_with_comma_name():
    # The Phase-1 harness build_query format: labeled sections, one analyte per line.
    query = (
        "Analyze the biological relationships among these co-expressed analytes.\n\n"
        "Proteins:\nGH1\nIL6\n\n"
        "Metabolites:\n1-oleoylglycerol (18:1)\n12,13-DiHOME\nquinolinate\n"
    )
    ents = extract_entities(query)
    assert "12,13-DiHOME" in ents                       # internal comma preserved
    assert "1-oleoylglycerol" in ents                   # alias-paren still stripped to primary (Phase 1 unchanged)
    assert "GH1" in ents and "IL6" in ents and "quinolinate" in ents


def test_single_analyte_section_with_internal_comma_kept():
    # Greptile #88 P2: a single-line section whose one entry has an internal comma must NOT be
    # fragmented. The fallback splits on comma+whitespace, so "12,13-DiHOME" (comma-digit) survives.
    ents = extract_entities("Metabolites:\n12,13-DiHOME\n")
    assert ents == ["12,13-DiHOME"]


def test_single_line_comma_space_list_still_splits():
    # The legacy single-line list uses ", " delimiters, which still split.
    ents = extract_entities("Metabolites:\nglucose, fructose, lactate\n")
    assert ents == ["glucose", "fructose", "lactate"]


def test_internal_comma_with_alias_paren_both():
    # A "both" row: internal comma + trailing suffix. Newline-delimited keeps it whole;
    # paren-strip does not fire (no balanced trailing paren), so it passes through intact.
    query = "Metabolites:\ndiacylglycerol (14:0/18:1, 16:0/16:1) [1]\nglucose\n"
    ents = extract_entities(query)
    assert "glucose" in ents
    # the complex name is not fragmented on its internal comma
    assert any("diacylglycerol" in e and "14:0/18:1" in e and "16:0/16:1" in e for e in ents)
    assert "16:0/16:1) [1]" not in ents                 # not the trailing fragment of a comma split
