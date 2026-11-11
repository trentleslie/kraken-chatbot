"""Tests for the literature-source comparison spike harness (evals/lit_sources).

Gates proven here (plan 2026-11-11-001): P4 matcher mutation, P5 failure states, P6 replay
fidelity (offline, on captured real payloads), rule freeze, resume/retry, rehydration KG gate,
selection, decision rule. Fixtures under fixtures/lit_sources/ are captured REAL payloads, except
where a fixture's "note" says constructed (OpenAlex 500 and Consensus raw output cannot be
captured on demand).
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import httpx
import pytest
import respx

from kestrel_backend.assessment.capture import _serialize_state
from kestrel_backend.evals.lit_sources import analyze, judge, run_study, sources
from kestrel_backend.graph.nodes import literature_grounding as lg
from kestrel_backend.graph.state import (
    DiseaseAssociation,
    EntityResolution,
    Finding,
    Hypothesis,
    LiteratureSupport,
)

FX = Path(__file__).parent / "fixtures" / "lit_sources"
RULES = run_study.RULES_DIR


def fx(name: str) -> dict:
    return json.loads((FX / f"{name}.json").read_text())


def fx_interaction(name: str, source: str) -> dict:
    """A recorded interaction (sources.CallRecorder shape) built from a captured fixture."""
    d = fx(name)
    return {
        "source": source,
        "method": d["request"]["method"],
        "url": d["request"]["url"],
        "content": d["request"]["content"],
        "status_code": d["response"]["status_code"],
        "text": d["response"]["text"],
        "exception_type": None,
    }


def make_hyp(title="Bridge: leucine -> insulin resistance", claim=None, entities=None, tier=2):
    return Hypothesis(
        title=title,
        tier=tier,
        claim=claim or "Leucine is linked to insulin resistance through mTORC1 signalling.",
        supporting_entities=entities if entities is not None else ["LABEL:leucine", "LABEL:insulin resistance"],
        structural_logic="leucine -> mTORC1 -> IRS1",
        validation_steps=["check literature"],
    )


def make_pinned(hyp_id="h1", kind="real", hyp=None, **kw):
    hyp = hyp or make_hyp()
    base = {
        "hyp_id": hyp_id,
        "kind": kind,
        "hypothesis": hyp.model_dump(),
        "disease_context": "type 2 diabetes",
        "entity_names": {"LABEL:leucine": "leucine", "LABEL:insulin resistance": "insulin resistance"},
        "synonyms": {},
        "kg_pmids": {},
        "panel": "t2d_basic",
        "hyp_type": "bridge",
    }
    base.update(kw)
    return base


# ---------------------------------------------------------------------------------------------
# Unit 1: rules freeze, manifest, resume, retry, panel parsing, rehydration, selection, controls
# ---------------------------------------------------------------------------------------------


def test_rules_lock_matches_committed_rules():
    hashes = run_study.verify_rules_lock(RULES)
    assert {"query_rules.json", "decision_rule.json", "capture.json",
            "kraken_test_panel_modules.csv", "kraken_test_panel_basic.csv"} <= set(hashes)


def test_rule_hash_mismatch_refuses_stage(tmp_path):
    rules = tmp_path / "rules"
    shutil.copytree(RULES, rules)
    p = rules / "decision_rule.json"
    p.write_text(p.read_text().replace('"min_ppv": 0.75', '"min_ppv": 0.5'))
    with pytest.raises(run_study.RulesHashMismatch):
        run_study.verify_rules_lock(rules)

    async def work(item):  # pragma: no cover - must never run
        raise AssertionError("stage ran despite a rule-hash mismatch")

    ck = run_study.JsonlCheckpoint(tmp_path / "rows.jsonl", key_fields=("id",))
    with pytest.raises(run_study.RulesHashMismatch):
        asyncio.run(run_study.run_stage_items([{"id": 1}], work, ck, rules_dir=rules))


def test_manifest_written_before_any_stage(tmp_path):
    m = run_study.Manifest.create(tmp_path / "run", repo_sha="abc123", rules_dir=RULES,
                                  kestrel_mcp_url="http://kestrel.example/mcp")
    data = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert data["repo_sha"] == "abc123"
    assert data["cache"] == "cold"
    assert data["g0"] == "lifted 2026-11-11"
    assert data["rule_hashes"]["decision_rule.json"]
    assert data["judge"]["model"] == judge.JUDGE_MODEL
    assert data["judge"]["prompt_hash"] == judge.prompt_hash()
    assert data["stages"] == {}
    m.update_stage("capture", "complete", n=3)
    m.add_deviation("set module directions")
    data = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert data["stages"]["capture"]["status"] == "complete"
    assert data["deviations"][0]["text"] == "set module directions"


def test_killed_run_resumes_to_same_row_count(tmp_path):
    items = [{"id": i} for i in range(5)]
    ck = run_study.JsonlCheckpoint(tmp_path / "rows.jsonl", key_fields=("id",))
    calls: list[int] = []

    class Killed(Exception):
        pass

    async def dies_at_3(item):
        if item["id"] == 3:
            raise Killed()
        calls.append(item["id"])
        return {"id": item["id"], "v": item["id"] * 2}

    with pytest.raises(Killed):
        asyncio.run(run_study.run_stage_items(items, dies_at_3, ck, rules_dir=RULES))
    assert len(ck.rows()) == 3

    async def ok(item):
        calls.append(item["id"])
        return {"id": item["id"], "v": item["id"] * 2}

    asyncio.run(run_study.run_stage_items(items, ok, ck, rules_dir=RULES))
    rows = ck.rows()
    assert sorted(r["id"] for r in rows) == [0, 1, 2, 3, 4]
    assert calls == [0, 1, 2, 3, 4]  # resumed items were not re-run


def test_injected_timeout_is_retried_then_raises_and_is_never_written(tmp_path):
    ck = run_study.JsonlCheckpoint(tmp_path / "rows.jsonl", key_fields=("id",))
    attempts = {"n": 0}

    async def times_out(item):
        attempts["n"] += 1
        raise httpx.ReadTimeout("injected")

    async def no_sleep(_):
        return None

    with pytest.raises(httpx.ReadTimeout):
        asyncio.run(run_study.run_stage_items([{"id": 1}], times_out, ck, rules_dir=RULES,
                                              attempts=3, sleep=no_sleep))
    assert attempts["n"] == 3
    assert ck.rows() == []


def test_modules_csv_parses_and_passes_validate_and_normalize():
    parsed = run_study.parse_panel_csv(RULES / "kraken_test_panel_modules.csv")
    names = [a["name"] for a in parsed["structured_analytes"]]
    assert len(names) == 12
    assert "1,5-anhydroglucitol" in names  # quoted comma survives
    glycine = next(a for a in parsed["structured_analytes"] if a["name"] == "glycine")
    assert glycine == {"name": "glycine", "group": "M1_bcaa", "type": "metabolite", "kme": -0.71, "kim": 7.3}

    directions = [
        {"group": "M1_bcaa", "eigengene_trait_correlation": 0.5, "trait_label": "type 2 diabetes"},
        {"group": "M3_adipokine", "eigengene_trait_correlation": -0.4, "trait_label": "type 2 diabetes"},
    ]
    panel = run_study.validate_panel(parsed, module_directions=directions)
    assert panel.errors == []
    assert set(panel.module_members) == {"m1_bcaa", "m2_glycemic", "m3_adipokine"}
    assert panel.module_directions["m1_bcaa"]["eigengene_trait_correlation"] == 0.5
    assert panel.module_directions["m3_adipokine"]["eigengene_trait_correlation"] == -0.4


def test_basic_csv_parses_with_group_and_type():
    parsed = run_study.parse_panel_csv(RULES / "kraken_test_panel_basic.csv")
    assert len(parsed["structured_analytes"]) == 12
    assert parsed["structured_analytes"][3] == {"name": "TCF7L2", "group": "glycemic", "type": "gene"}
    assert run_study.validate_panel(parsed).errors == []


def test_capture_spec_refuses_unset_module_directions():
    specs = run_study.load_capture_specs(RULES)
    modules = next(s for s in specs if s["id"] == "t2d_modules")
    assert len(run_study.capture_inputs(modules, RULES)["module_directions"]) == 3
    with pytest.raises(run_study.CaptureSpecIncomplete):
        run_study.capture_inputs({**modules, "module_directions": "UNSET"}, RULES)
    basic = next(s for s in specs if s["id"] == "t2d_basic")
    inputs = run_study.capture_inputs(basic, RULES)
    assert len(inputs["structured_analytes"]) == 12
    ad = next(s for s in specs if s["id"] == "ad_freetext")
    assert run_study.capture_inputs(ad, RULES)["structured_analytes"] == []


def _typed_state():
    hyp = make_hyp(entities=["CHEBI:1", "HGNC:2"])
    return {
        "hypotheses": [hyp],
        "direct_findings": [Finding(entity="CHEBI:1", claim="c", tier=1, pmids=["111", "222"])],
        "disease_associations": [DiseaseAssociation(
            entity_curie="HGNC:2", disease_curie="MONDO:1", disease_name="d", predicate="p",
            source="s", pmids=["333"])],
        "resolved_entities": [EntityResolution(raw_name="x", curie="CHEBI:1", resolved_name="X")],
        "study_context": {"disease_focus": "type 2 diabetes"},
    }


def test_rehydrated_state_yields_same_kg_pmids_and_plain_dicts_fail_gate():
    live = _typed_state()
    live_count = analyze.kg_pmid_total(live)
    assert live_count == 3
    serialized = json.loads(json.dumps(_serialize_state(live)))
    rehydrated = analyze.rehydrate_state(serialized)
    assert analyze.kg_pmid_total(rehydrated) == live_count
    analyze.check_kg_gate(live_count, rehydrated)  # passes
    assert isinstance(rehydrated["hypotheses"][0], Hypothesis)
    # Plain-dict state (no rehydration) silently yields 0 PMIDs -> the gate must fail.
    assert analyze.kg_pmid_total(serialized) == 0
    with pytest.raises(analyze.GateFailure):
        analyze.check_kg_gate(live_count, serialized)


def _run(capture_id, hyps):
    return {"capture_id": capture_id, "hypotheses": hyps}


def _bridge(i, tier=2):
    return make_hyp(title=f"Bridge: a{i} -> b{i}", claim=f"bridge claim {i}", tier=tier)


def _inferred(i):
    return make_hyp(title=f"Inferred role of X{i}", claim=f"inferred claim {i}", tier=3)


def test_selection_round_robin_tier_order_and_dedup():
    r1 = _run("t2d_modules", [_bridge(1, tier=3), _bridge(2, tier=1), _inferred(3)])
    r2 = _run("t2d_basic", [_bridge(2, tier=1), _inferred(4), _bridge(5)])
    r3 = _run("ad_freetext", [_bridge(6), _inferred(7)])
    sel = run_study.select_hypotheses([r1, r2, r3], target=10, min_n=6)
    titles = [p["hypothesis"]["title"] for p in sel["selected"]]
    # r1 tier order puts bridge 2 (tier 1) first; r2's duplicate of bridge 2 is skipped, so r2
    # contributes its next tier-ordered hypothesis (bridge 5, tier 2) in the same round.
    assert titles[:3] == ["Bridge: a2 -> b2", "Bridge: a5 -> b5", "Bridge: a6 -> b6"]
    assert len(titles) == len(set(titles)) == 7
    assert sel["fallback"] is True  # fewer than 10 available, >= 6 kept
    assert sel["coverage_met"] is True


def test_selection_continues_round_robin_until_coverage_met():
    # First 10 in round-robin order are all bridges; the inferred role sits at position 11.
    r1 = _run("t2d_modules", [_bridge(i) for i in range(0, 6)] + [_inferred(99)])
    r2 = _run("t2d_basic", [_bridge(i) for i in range(10, 15)])
    sel = run_study.select_hypotheses([r1, r2], target=10, min_n=6)
    types = [p["hyp_type"] for p in sel["selected"]]
    assert "inferred_role" in types
    assert len(sel["selected"]) > 10
    assert sel["coverage_met"] is True


def test_selection_below_min_requires_another_capture():
    with pytest.raises(run_study.NeedMoreCaptures):
        run_study.select_hypotheses([_run("t2d_basic", [_bridge(1), _inferred(2)])], target=10, min_n=6)


def test_controls_load_verified_and_the_live_gate_refuses_unverified_copies():
    controls = run_study.load_controls(run_study.CONTROLS_DIR)
    kinds = {c["kind"] for c in controls}
    assert kinds == {"control_true", "control_false", "control_corrupt"}
    for c in controls:
        Hypothesis.model_validate(c["hypothesis"])
        assert c["hypothesis"]["title"].startswith("Bridge: ")
        assert c["disease_context"]
    trues = [c for c in controls if c["kind"] == "control_true"]
    assert all(c["landmarks"] for c in trues)
    corrupt = [c for c in controls if c["kind"] == "control_corrupt"]
    assert len(corrupt) == 1 and corrupt[0]["inject_landmarks"]
    # The committed controls are verified, and the gate still fires when a verification is missing.
    run_study.require_verified_controls(controls)
    import copy
    unverified = copy.deepcopy(controls)
    trues_u = [c for c in unverified if c["kind"] == "control_true"]
    trues_u[0]["landmarks"][0]["verified_at"] = None
    with pytest.raises(run_study.ControlsNotVerified):
        run_study.require_verified_controls(unverified)
    unsearched = copy.deepcopy(controls)
    next(c for c in unsearched if c["kind"] == "control_false")["pre_run_search"] = None
    with pytest.raises(run_study.ControlsNotVerified):
        run_study.require_verified_controls(unsearched)


# ---------------------------------------------------------------------------------------------
# Unit 2: call states (P5), matcher (P4), Arm 1 / Arm 2 adapters, Consensus, normalization
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("fixture,source,expected", [
    ("s2_403", "s2", "lookup_failed"),
    ("exa_401", "exa", "lookup_failed"),
    ("ncbi_esearch_error", "pubmed", "lookup_failed"),
    ("openalex_500_constructed", "openalex", "lookup_failed"),
    ("openalex_empty", "openalex", "empty"),
    ("ncbi_esearch_sentence_no_items", "pubmed", "empty"),
])
def test_p5_failure_fixtures_classify(fixture, source, expected):
    inter = fx_interaction(fixture, source)
    st = sources.classify_call(source, [inter], key_present=True, n_items=0)
    assert st["state"] == expected, st


def test_ncbi_stopword_warning_with_hits_is_ok_and_keeps_the_warning():
    # Real payload: a sentence query with 106 hits AND phrasesignored. Must not be lookup_failed.
    inter = fx_interaction("ncbi_esearch_hits_with_warning", "pubmed")
    st = sources.classify_call("pubmed", [inter], key_present=True, n_items=6)
    assert st["state"] == "ok"
    assert any("WarningList" in w for w in st["warnings"])


def test_p5_raised_timeout_is_lookup_failed():
    inter = {"source": "openalex", "method": "GET", "url": "https://api.openalex.org/works?search=x",
             "content": None, "status_code": None, "text": None, "exception_type": "ReadTimeout"}
    st = sources.classify_call("openalex", [inter], key_present=True, n_items=0)
    assert st["state"] == "lookup_failed"
    assert st["transient"] is True


def test_p5_missing_key_is_unavailable_and_no_interaction_with_key_is_failure():
    assert sources.classify_call("exa", [], key_present=False, n_items=0)["state"] == "unavailable"
    assert sources.classify_call("exa", [], key_present=True, n_items=0)["state"] == "lookup_failed"


def test_p5_ok_requires_items_and_never_ok_on_exception():
    ok = fx_interaction("openalex_ok", "openalex")
    assert sources.classify_call("openalex", [ok], key_present=True, n_items=6)["state"] == "ok"
    boom = dict(ok, exception_type="ConnectError", status_code=None, text=None)
    st = sources.classify_call("openalex", [ok, boom], key_present=True, n_items=6)
    assert st["state"] == "lookup_failed"


def test_p5_interactions_from_other_hosts_are_ignored():
    other = fx_interaction("ncbi_esearch_ok", "pubmed")
    st = sources.classify_call("openalex", [other], key_present=True, n_items=0)
    assert st["state"] == "lookup_failed"
    assert "no expected-host" in st["reason"]


LANDMARK = {
    "pmid": "19356713",
    "doi": "10.1016/j.cmet.2009.02.002",
    "title": "A branched-chain amino acid-related metabolic signature that differentiates obese "
             "and lean humans and contributes to insulin resistance",
}


def test_p4_matcher_mutation():
    doi_only = sources.canonical_ids({"doi": "https://doi.org/10.1016/J.CMET.2009.02.002", "title": "x"})
    pmid_only = sources.canonical_ids({"paper_id": "PMID:19356713", "title": "y"})
    mojibake = LANDMARK["title"].replace("-", "–").encode("utf-8").decode("cp1252", errors="replace")
    assert mojibake != LANDMARK["title"]
    title_only = sources.canonical_ids({"title": mojibake})
    decoy = sources.canonical_ids({"doi": "10.1016/j.cmet.2009.02.003"})
    assert sources.matches_landmark(doi_only, LANDMARK)
    assert sources.matches_landmark(pmid_only, LANDMARK)
    assert sources.matches_landmark(title_only, LANDMARK)
    assert not sources.matches_landmark(decoy, LANDMARK)
    # Proves the gate can fail: a conflicting identifier blocks a title-only coincidence.
    conflicting = sources.canonical_ids({"doi": "10.9999/other", "title": LANDMARK["title"]})
    assert not sources.matches_landmark(conflicting, LANDMARK)


def test_canonical_ids_from_pubmed_url_and_openalex_style():
    ids = sources.canonical_ids({"paper_id": "W123", "url": "https://pubmed.ncbi.nlm.nih.gov/34044180/",
                                 "doi": "doi:10.1/ABC"})
    assert ids["pmid"] == "34044180"
    assert ids["doi"] == "10.1/abc"
    assert sources.paper_key(ids) == "doi:10.1/abc"


def _respx_real_routes(router):
    router.get(url__regex=r"https://api\.openalex\.org/works\?.*").mock(
        return_value=httpx.Response(200, text=fx("openalex_ok")["response"]["text"]))
    router.get(url__regex=r".*/esearch\.fcgi.*").mock(
        return_value=httpx.Response(200, text=fx("ncbi_esearch_ok")["response"]["text"]))
    router.get(url__regex=r".*/esummary\.fcgi.*").mock(
        return_value=httpx.Response(200, text=fx("ncbi_esummary_ok")["response"]["text"]))
    router.get(url__regex=r".*/efetch\.fcgi.*").mock(
        return_value=httpx.Response(200, text=fx("ncbi_efetch_ok")["response"]["text"]))
    router.get(url__regex=r"https://api\.semanticscholar\.org/.*").mock(
        return_value=httpx.Response(403, text=fx("s2_403")["response"]["text"]))


@pytest.fixture
def fast_clients(monkeypatch):
    # Patch the client modules the harness actually holds: test_literature_grounding.py swaps
    # stub modules into sys.modules at collection, so a fresh `from kestrel_backend import ...`
    # here could return a stub instead of the module production code calls.
    exa_client, pubmed_client, semantic_scholar = sources.exa_client, sources.pubmed_client, sources.semantic_scholar
    monkeypatch.setattr(semantic_scholar, "S2_DELAY", 0)
    monkeypatch.setattr(semantic_scholar, "S2_API_KEY", None)
    monkeypatch.setattr(exa_client, "EXA_API_KEY", "")
    monkeypatch.setattr(pubmed_client, "_pubmed_delay", lambda: 0)
    monkeypatch.delenv("NCBI_API_KEY", raising=False)


async def test_arm1_openalex_gets_production_query_and_snapshot_holds_six(fast_clients):
    pinned = make_pinned()
    hyp = Hypothesis.model_validate(pinned["hypothesis"])
    expected_q = lg.build_search_query(hyp, pinned["disease_context"])
    with respx.mock(assert_all_called=False) as router:
        _respx_real_routes(router)
        rec = await sources.snapshot_arm1(pinned, "openalex")
        sent = router.calls[0].request
    assert httpx.URL(str(sent.url)).params["search"] == expected_q[:200]
    assert rec["query"] == expected_q
    assert rec["state"] == "ok"
    assert len(rec["papers"]) == 6
    assert rec["papers"][0]["relevance_score"] == 0.9
    assert rec["interactions"][0]["status_code"] == 200


async def test_arm1_exa_without_key_is_unavailable(fast_clients):
    with respx.mock(assert_all_called=False):
        rec = await sources.snapshot_arm1(make_pinned(), "exa")
    assert rec["state"] == "unavailable"
    assert rec["interactions"] == []


async def test_arm1_kg_uses_pinned_pmids_without_network():
    pinned = make_pinned(kg_pmids={"LABEL:leucine": ["111", "222"]})
    rec = await sources.snapshot_arm1(pinned, "kg")
    assert rec["state"] == "ok"
    assert [p["paper_id"] for p in rec["papers"]] == ["PMID:111", "PMID:222"]
    rec0 = await sources.snapshot_arm1(make_pinned(), "kg")
    assert rec0["state"] == "empty"


async def test_matched_pair_runs_direct_client_on_identical_arm1_sentence(fast_clients):
    pinned = make_pinned()
    with respx.mock(assert_all_called=False) as router:
        _respx_real_routes(router)
        arm1 = await sources.snapshot_arm1(pinned, "pubmed")
        n_before = len(router.calls)
        direct = await sources.snapshot_pubmed_direct(pinned, arm="arm1_direct")
        direct_terms = [httpx.URL(str(c.request.url)).params.get("term")
                        for c in router.calls[n_before:] if "esearch" in str(c.request.url)]
    assert direct["query"] == arm1["query"]
    assert direct_terms == [arm1["query"]]
    assert direct["state"] == "ok"
    assert [p["paper_id"] for p in direct["papers"]] == [p["paper_id"] for p in arm1["papers"]]


async def test_direct_client_surfaces_ncbi_error_body(fast_clients):
    with respx.mock(assert_all_called=False) as router:
        router.get(url__regex=r".*/esearch\.fcgi.*").mock(
            return_value=httpx.Response(200, text=fx("ncbi_esearch_error")["response"]["text"]))
        rec = await sources.snapshot_pubmed_direct(make_pinned(), arm="arm1_direct")
    assert rec["state"] == "lookup_failed"
    assert rec["papers"] == []


def test_pubmed_boolean_rule():
    pinned = make_pinned(synonyms={"LABEL:leucine": ["L-leucine"]})
    q = sources.pubmed_boolean_query(pinned, sources.load_query_rules(RULES))
    assert q == ('("leucine"[tiab] OR "L-leucine"[tiab]) AND ("insulin resistance"[tiab]) '
                 'AND ("type 2 diabetes"[tiab])')
    assert sources.pubmed_boolean_query(make_pinned(entity_names={}), sources.load_query_rules(RULES)) is None


def test_consensus_question_rule():
    rules = sources.load_query_rules(RULES)
    q = sources.consensus_question(make_pinned(), rules)
    assert q == ("Is there evidence that leucine is linked to insulin resistance through mTORC1 "
                 "signalling in type 2 diabetes?")
    hyp = make_hyp(claim="BCAA levels predict type 2 diabetes.")
    q2 = sources.consensus_question(make_pinned(hyp=hyp), rules)
    assert q2 == "Is there evidence that BCAA levels predict type 2 diabetes?"


def test_consensus_worklist_and_ingest_states(tmp_path):
    rules = sources.load_query_rules(RULES)
    pinned = [make_pinned("h1"), make_pinned("h2"), make_pinned("h3")]
    wl = sources.emit_consensus_worklist(pinned, tmp_path / "consensus_worklist.jsonl", rules)
    assert [r["hyp_id"] for r in wl] == ["h1", "h2", "h3"]
    raw = {
        wl[0]["call_id"]: {"call_id": wl[0]["call_id"], "ts": "2026-11-12T00:00:00Z",
                           "output": fx("consensus_raw_constructed")["output"]},
        wl[1]["call_id"]: {"call_id": wl[1]["call_id"], "ts": "2026-11-12T00:00:01Z",
                           "error": "Rate limit exceeded"},
    }
    recs = sources.ingest_consensus(wl, list(raw.values()), preflight_ok=True)
    by = {r["hyp_id"]: r for r in recs}
    assert by["h1"]["state"] == "ok" and len(by["h1"]["papers"]) == 3
    assert by["h2"]["state"] == "lookup_failed"
    assert by["h3"]["state"] == "not_run"
    down = sources.ingest_consensus(wl, list(raw.values()), preflight_ok=False)
    assert {r["state"] for r in down} == {"unavailable"}


def test_consensus_title_resolution_rejects_wrong_year():
    paper = {"title": LANDMARK["title"], "year": 2009, "first_author": "Newgard"}
    good = {"title": LANDMARK["title"] + ".", "year": 2009, "first_author": "Newgard",
            "doi": LANDMARK["doi"]}
    wrong_year = dict(good, year=2011)
    assert sources.resolve_consensus_title(paper, [wrong_year]) is None
    assert sources.resolve_consensus_title(paper, [wrong_year, good])["doi"] == LANDMARK["doi"]
    far = dict(good, title="Branched chain amino acids in obesity")
    assert sources.resolve_consensus_title(paper, [far]) is None


def test_elicit_unavailable_unless_tos_yes_and_headroom():
    assert sources.elicit_status(tos="unknown", headroom=True)["state"] == "unavailable"
    assert sources.elicit_status(tos="yes", headroom=False)["state"] == "unavailable"
    assert sources.elicit_status(tos="yes", headroom=True)["state"] == "not_run"


def test_reconstruct_abstract_from_real_inverted_index():
    work = json.loads(fx("openalex_doi_abstract")["response"]["text"])
    text = sources.reconstruct_abstract(work["abstract_inverted_index"])
    assert len(text.split()) > 50
    assert "insulin" in text.lower()
    # Real case: OpenAlex carries no abstract for this Elsevier DOI -> resolver yields None.
    no_ab = json.loads(fx("openalex_doi_no_abstract")["response"]["text"])
    assert sources.reconstruct_abstract(no_ab["abstract_inverted_index"]) is None


async def test_abstract_resolver_chain_order():
    calls = []

    async def efetch(pmids):
        calls.append(("efetch", tuple(pmids)))
        return {}

    async def by_doi(doi):
        calls.append(("doi", doi))
        return "Abstract from DOI â€” repaired"

    async def by_title(title):  # pragma: no cover - must not be reached
        calls.append(("title", title))
        return "nope"

    ab, how = await sources.resolve_abstract({"pmid": "1", "doi": "10.1/x", "title_key": "t"},
                                             title="T", efetch=efetch, by_doi=by_doi, by_title=by_title)
    assert how == "doi"
    assert calls == [("efetch", ("1",)), ("doi", "10.1/x")]
    assert "—" in ab  # mojibake repaired, not stripped


def test_normalization_report_shares():
    rows = [
        {"source": "exa", "resolved": True, "abstract": "a"},
        {"source": "exa", "resolved": False, "abstract": None},
        {"source": "openalex", "resolved": True, "abstract": None},
    ]
    rep = sources.normalization_report(rows)
    assert rep["exa"] == {"n": 2, "unresolved_share": 0.5, "no_abstract_share": 0.5}
    assert rep["openalex"]["no_abstract_share"] == 1.0


def test_every_snapshot_call_has_exactly_one_state():
    for st in ("ok", "empty", "lookup_failed", "unavailable", "not_run"):
        assert st in sources.CALL_STATES
    assert len(sources.CALL_STATES) == 5


# ---------------------------------------------------------------------------------------------
# Unit 3: judge, audit, replay, analysis, decision rule
# ---------------------------------------------------------------------------------------------


def test_judge_prompt_is_blind_to_source():
    p = judge.build_judge_prompt("A causes B", "Title", "Abstract body")
    for s in ("openalex", "exa", "pubmed", "semantic scholar", "consensus", "kg"):
        assert s not in p.lower()
    assert "A causes B" in p and "Abstract body" in p


@pytest.mark.parametrize("text,expected", [
    ('{"hit": true, "direction": "supports", "reason": "r"}', {"hit": True, "direction": "supports"}),
    ('```json\n{"hit": false, "direction": "na", "reason": "r"}\n```', {"hit": False, "direction": "na"}),
    ('{"hit": "yes"}', None),
    ("not json", None),
    ('{"hit": false, "direction": "supports", "reason": "r"}', None),
])
def test_parse_verdict(text, expected):
    v = judge.parse_verdict(text)
    if expected is None:
        assert v is None
    else:
        assert {k: v[k] for k in expected} == expected


async def test_judge_semaphore_missing_verdicts_and_no_abstract():
    active = {"now": 0, "max": 0}

    async def model(prompt):
        active["now"] += 1
        active["max"] = max(active["max"], active["now"])
        await asyncio.sleep(0.01)
        active["now"] -= 1
        if "BROKEN" in prompt:
            return "garbage"
        if "RAISES" in prompt:
            raise RuntimeError("sdk down")
        return '{"hit": true, "direction": "supports", "reason": "ok"}'

    items = [{"item_id": f"i{i}", "hyp_id": "h", "claim": "c", "title": f"t{i}", "abstract": "ab"}
             for i in range(12)]
    items += [{"item_id": "bad", "hyp_id": "h", "claim": "c", "title": "BROKEN", "abstract": "ab"},
              {"item_id": "err", "hyp_id": "h", "claim": "c", "title": "RAISES", "abstract": "ab"},
              {"item_id": "noab", "hyp_id": "h", "claim": "c", "title": "t", "abstract": None}]
    out = {r["item_id"]: r for r in await judge.judge_items(items, model, concurrency=4)}
    assert active["max"] <= 4
    assert out["i0"]["status"] == "judged" and out["i0"]["hit"] is True
    assert out["bad"]["status"] == "not_run"
    assert out["err"]["status"] == "not_run"
    assert out["noab"]["status"] == "no_abstract"


def _judgments(n_per=10):
    rows = []
    for src in ("openalex", "exa", "pubmed"):
        for i in range(n_per):
            rows.append({"item_id": f"{src}{i}", "hyp_id": f"h{i}", "paper_key": f"doi:{src}/{i}",
                         "sources": [src], "status": "judged", "hit": i % 3 == 0,
                         "direction": "supports" if i % 3 == 0 else "na", "kind": "real",
                         "claim": "c", "title": "t", "abstract": "a"})
    return rows


def test_audit_sample_is_seeded_stratified_capped_and_verdict_blind(tmp_path):
    judg = _judgments(20)
    fs = [{"item_id": f"fs{i}", "hyp_id": f"f{i % 10}", "paper_key": f"doi:fs/{i % 18}",
           "sources": ["exa"], "status": "judged", "hit": True, "direction": "supports",
           "kind": "control_false", "claim": "c", "title": "t", "abstract": "a"} for i in range(40)]
    cons = [{"item_id": f"cc{i}", "consensus_title": "x", "resolved_title": "y"} for i in range(9)]
    rule = json.loads((RULES / "decision_rule.json").read_text())["audit"]
    s1 = judge.draw_audit_sample(judg, false_support_events=fs, consensus_checks=cons, rule=rule)
    s2 = judge.draw_audit_sample(judg, false_support_events=fs, consensus_checks=cons, rule=rule)
    assert [i["item_id"] for i in s1] == [i["item_id"] for i in s2]
    assert len(s1) <= 60
    assert sum(1 for i in s1 if i["audit_role"] == "sample") == 40
    assert sum(1 for i in s1 if i["audit_role"] == "false_support") == 15
    assert sum(1 for i in s1 if i["audit_role"] == "consensus_identity") == 5
    strata = {(i["sources"][0], i["judge_hit"]) for i in s1 if i["audit_role"] == "sample"}
    assert len(strata) == 6  # every source x verdict stratum represented
    path = judge.export_audit_packet(s1, tmp_path / "audit_packet.jsonl")
    for line in path.read_text().splitlines():
        row = json.loads(line)
        assert "hit" not in row and "judge_hit" not in row and "direction" not in row
        assert "sources" not in row


def test_agreement_kappa_ppv_and_certification():
    rule = json.loads((RULES / "decision_rule.json").read_text())["certification"]
    pairs = [(True, True, "exa")] * 18 + [(False, False, "exa")] * 20 + [(True, False, "pubmed")] * 2
    st = judge.agreement_stats(pairs)
    assert st["n"] == 40
    assert st["agreement"] == pytest.approx(0.95)
    assert st["ppv"] == pytest.approx(0.9)
    assert st["npv"] == pytest.approx(1.0)
    assert st["kappa"] == pytest.approx(0.9, abs=0.01)
    assert st["per_source"]["pubmed"] == {"n": 2, "agreement": 0.0}
    ok, _ = judge.certify(st, rule)
    assert ok
    bad = judge.agreement_stats([(True, False, "exa")] * 12 + [(True, True, "exa")] * 28)
    ok2, reasons = judge.certify(bad, rule)
    assert not ok2 and any("ppv" in r for r in reasons)


def test_corrected_precision():
    assert judge.corrected_precision(0.5, ppv=0.8, npv=0.9) == pytest.approx(0.5 * 0.8 + 0.5 * 0.1)


def _lit(source, i, score, doi=None, pmid=None, title=None):
    return LiteratureSupport(
        paper_id=f"PMID:{pmid}" if pmid else f"{source}-{i}",
        title=title or f"{source} paper {i}", authors="A", year=2020, doi=doi,
        relevance_score=score, source=source)


def test_harness_merge_matches_production_merge_with_duplicates_and_ties():
    lits = [
        _lit("openalex", 0, 0.9, doi="10.1/dup"), _lit("openalex", 1, 0.87), _lit("openalex", 2, 0.84),
        _lit("exa", 0, 0.9), _lit("exa", 1, 0.87, doi="10.1/DUP"), _lit("exa", 2, 0.5),
        _lit("pubmed", 0, 0.88, doi="10.1/dup", pmid="5"), _lit("pubmed", 1, 0.85), _lit("pubmed", 2, 0.82),
        _lit("s2", 0, 0.9), _lit("s2", 1, 0.2),
    ]
    prod = lg.merge_literature(lits, limit=3)
    harness = analyze.harness_merge([l.model_dump() for l in lits], analyze.PRIORITY, limit=3)
    assert [lg.get_unique_key(p) for p in prod] == [analyze.unique_key(r) for r in harness]
    # duplicate DOI keeps the PubMed copy (higher SOURCE_PRIORITY)
    dup = next(r for r in harness if analyze.unique_key(r) == "doi:10.1/dup")
    assert dup["source"] == "pubmed"
    # tie at 0.9 between openalex-1? (exa0 and s2_0) keeps concatenation order
    full_p = lg.merge_literature(lits, limit=20)
    full_h = analyze.harness_merge([l.model_dump() for l in lits], analyze.PRIORITY, limit=20)
    assert [lg.get_unique_key(p) for p in full_p] == [analyze.unique_key(r) for r in full_h]


def _snaps(hyp_id, kg=None, **by_source):
    out = {}
    for src, papers in by_source.items():
        out[(hyp_id, src, "arm1")] = {"hyp_id": hyp_id, "source": src, "arm": "arm1", "state": "ok",
                                      "papers": [p.model_dump() for p in papers]}
    out[(hyp_id, "kg", "arm1")] = {"hyp_id": hyp_id, "source": "kg", "arm": "arm1",
                                   "state": "ok" if kg else "empty",
                                   "papers": [p.model_dump() for p in (kg or [])]}
    return out


def test_kg_short_circuit_in_replay():
    snaps = _snaps("h1", kg=[lg.create_literature_from_pmid("9")],
                   openalex=[_lit("openalex", 0, 0.9)])
    kept = analyze.replay_kept("h1", snaps, "current")
    assert [r["source"] for r in kept] == ["kg"]
    snaps2 = _snaps("h2", openalex=[_lit("openalex", 0, 0.9)])
    assert [r["source"] for r in analyze.replay_kept("h2", snaps2, "current")] == ["openalex"]


def test_mix_c_reweight_and_a_swaps_pubmed_arm():
    snaps = _snaps("h1", openalex=[_lit("openalex", i, 0.9 - 0.03 * i) for i in range(3)],
                   exa=[_lit("exa", i, 0.2) for i in range(3)])
    snaps[("h1", "pubmed", "arm2")] = {"hyp_id": "h1", "source": "pubmed", "arm": "arm2", "state": "ok",
                                      "papers": [_lit("pubmed", 9, 0.88).model_dump()]}
    cur = analyze.replay_kept("h1", snaps, "current")
    assert [r["source"] for r in cur] == ["openalex"] * 3
    c = analyze.replay_kept("h1", snaps, "c_reweight")
    # rank-normalised: openalex0 and exa0 tie at 1.0 -> SOURCE_PRIORITY puts openalex first
    assert [r["paper_id"] for r in c][:2] == ["openalex-0", "exa-0"]
    a = analyze.replay_kept("h1", snaps, "a_fix_query")
    assert "pubmed-9" in [r["paper_id"] for r in a]


def test_mix_b_consensus_mappings():
    snaps = _snaps("h1", openalex=[_lit("openalex", i, 0.6) for i in range(3)])
    snaps[("h1", "consensus", "native")] = {
        "hyp_id": "h1", "source": "consensus", "arm": "native", "state": "ok",
        "papers": [{"paper_id": "c0", "title": "consensus paper", "year": 2020, "doi": "10.9/c0",
                    "source": "consensus", "relevance_score": None}]}
    best = analyze.replay_kept("h1", snaps, "b_add_consensus", consensus_mapping="best")
    worst = analyze.replay_kept("h1", snaps, "b_add_consensus", consensus_mapping="worst")
    assert best[0]["source"] == "consensus"
    assert "consensus" not in [r["source"] for r in worst]


def test_planted_known_precision_is_recovered_exactly():
    kept = {f"h{i}": [{"paper_key": f"h{i}-p{j}"} for j in range(3)] for i in range(4)}
    verdicts = {}
    for i in range(4):
        for j in range(3):
            verdicts[(f"h{i}", f"h{i}-p{j}")] = {"status": "judged", "hit": j == 0}
    verdicts[("h0", "h0-p2")] = {"status": "no_abstract"}
    res = analyze.kept_set_precision(kept, verdicts)
    assert res["pooled"] == pytest.approx(4 / 11)
    assert res["per_hypothesis"]["h0"] == pytest.approx(0.5)
    assert res["n_no_abstract"] == 1


def test_controls_never_enter_headline_frame():
    pinned = [make_pinned("r1"), make_pinned("t1", kind="control_true"),
              make_pinned("f1", kind="control_false"), make_pinned("x1", kind="control_corrupt")]
    rows = [{"hyp_id": p["hyp_id"], "v": 1} for p in pinned]
    frame = analyze.headline_frame(rows, pinned)
    assert [r["hyp_id"] for r in frame] == ["r1"]


def test_wilson_and_bootstrap():
    assert analyze.wilson_upper(0, 30) < 0.12
    assert analyze.wilson_upper(0, 10) > 0.2
    lo, hi = analyze.paired_bootstrap_ci([0.0] * 8, [0.5] * 8, n=500, seed=1)
    assert lo == hi == pytest.approx(0.5)
    lo, hi = analyze.paired_bootstrap_ci([0.0, 1.0] * 4, [1.0, 0.0] * 4, n=500, seed=1)
    assert lo < 0 < hi


RULE = json.loads((RULES / "decision_rule.json").read_text())


def _inputs(**over):
    base = {
        "certified": True, "not_run": 0, "search_path_n": 8,
        "sources": {
            s: {"available": True, "reason": None, "zero_hits_all_mixes": False,
                "judged_with_abstract": 40, "hits": 10}
            for s in ("pubmed", "openalex", "exa", "s2")
        },
        "pubmed_diagnosis": "fine",
        "fix_query": {"diff_ci": (-0.1, 0.1), "arm2_precision": 0.3},
        "reweight": {"diff_ci": (-0.05, 0.05)},
        "judge_filter": {"precision_ci": (0.1, 0.3), "false_support_ci": (-3, -1)},
        "consensus": {"tos": "yes", "available": True, "reason": None, "mappings": {
            m: {"unique_hit_hypotheses": 4, "false_support_delta": 0} for m in ("frozen", "best", "worst")}},
    }
    base.update(over)
    return base


def test_decision_rule_keep_add_filter():
    t = analyze.apply_decision_rule(_inputs(), RULE)
    assert t["sources"]["openalex"]["verdict"] == "keep"
    assert t["sources"]["consensus"]["verdict"] == "add"
    assert t["judge_filter"] == "yes"
    assert t["underpowered"] is False


def test_decision_rule_insufficient_data_not_drop():
    inp = _inputs()
    inp["sources"]["exa"].update(zero_hits_all_mixes=True, hits=0, judged_with_abstract=12)
    assert analyze.apply_decision_rule(inp, RULE)["sources"]["exa"]["verdict"] == "insufficient data"
    inp["sources"]["exa"].update(judged_with_abstract=40)
    assert analyze.apply_decision_rule(inp, RULE)["sources"]["exa"]["verdict"] == "drop"


def test_decision_rule_client_failure_precedence_blocks_drop():
    inp = _inputs(pubmed_diagnosis="client_failure",
                  fix_query={"diff_ci": (0.1, 0.4), "arm2_precision": 0.5})
    inp["sources"]["pubmed"].update(zero_hits_all_mixes=True, hits=0, judged_with_abstract=60)
    v = analyze.apply_decision_rule(inp, RULE)["sources"]["pubmed"]
    assert v["verdict"] == "fix client (#126)"


def test_decision_rule_fix_query_and_reweight():
    inp = _inputs(fix_query={"diff_ci": (0.05, 0.3), "arm2_precision": 0.25},
                  reweight={"diff_ci": (0.02, 0.2)})
    t = analyze.apply_decision_rule(inp, RULE)
    assert t["sources"]["pubmed"]["verdict"] == "fix query"
    assert t["sources"]["openalex"]["verdict"] == "reweight"
    assert analyze.apply_decision_rule(
        _inputs(fix_query={"diff_ci": (0.05, 0.3), "arm2_precision": 0.15}), RULE
    )["sources"]["pubmed"]["verdict"] == "keep"


def test_consensus_add_fails_under_worst_case_mapping():
    inp = _inputs()
    inp["consensus"]["mappings"]["worst"]["unique_hit_hypotheses"] = 1
    assert analyze.apply_decision_rule(inp, RULE)["sources"]["consensus"]["verdict"] == "keep out"
    inp2 = _inputs()
    inp2["consensus"]["tos"] = "unknown"
    assert analyze.apply_decision_rule(inp2, RULE)["sources"]["consensus"]["verdict"] == "keep out"


def test_unavailable_source_and_uncertified_block_verdicts():
    inp = _inputs()
    inp["sources"]["s2"].update(available=False, reason="S2 403 with installed key")
    t = analyze.apply_decision_rule(inp, RULE)
    assert t["sources"]["s2"] == {"verdict": "unavailable", "reason": "S2 403 with installed key"}
    for blocked in (_inputs(certified=False), _inputs(not_run=1)):
        t = analyze.apply_decision_rule(blocked, RULE)
        assert t["verdicts_blocked"] is True
        assert {v["verdict"] for v in t["sources"].values()} <= {"no verdict (uncertified)", "unavailable"}
        assert t["judge_filter"] == "no verdict (uncertified)"


def test_underpowered_flag():
    assert analyze.apply_decision_rule(_inputs(search_path_n=5), RULE)["underpowered"] is True


def test_pubmed_three_way_diagnosis():
    rule = RULE["rules"]
    pairs = [(0, 6), (0, 5), (6, 6), (0, 0)]
    assert analyze.diagnose_pubmed(pairs, arm1_precision=0.1, arm2_precision=0.1,
                                   fix_query_met=False, rule=rule) == "client_failure"
    pairs2 = [(6, 6), (6, 6)]
    assert analyze.diagnose_pubmed(pairs2, 0.1, 0.4, True, rule) == "query_failure"
    assert analyze.diagnose_pubmed(pairs2, 0.1, 0.1, False, rule) == "corpus_ranking_failure"
    assert analyze.diagnose_pubmed(pairs2, 0.4, 0.4, False, rule) == "fine"


async def test_p6_replay_fidelity_against_real_run(fast_clients, tmp_path):
    """Snapshot via the harness on real payloads, then the PRODUCTION run() under cassette replay
    must keep exactly the harness's current-mix kept set; a mutated replay must halt."""
    pinned = make_pinned("h1")
    with respx.mock(assert_all_called=False) as router:
        _respx_real_routes(router)
        snaps = {}
        for src in ("kg", "openalex", "exa", "pubmed", "s2"):
            rec = await sources.snapshot_arm1(pinned, src)
            snaps[("h1", src, "arm1")] = rec
        kept = analyze.replay_kept("h1", snaps, "current")
        backfill = await sources.record_backfill(pinned, kept)
    assert len(kept) == 3 and {r["source"] for r in kept} == {"openalex", "pubmed"}
    assert any("efetch" in i["url"] for i in backfill)
    # No S2 key -> unavailable for verdicts, but production still calls S2, so the call is
    # recorded (and replayed) for fidelity.
    assert snaps[("h1", "s2", "arm1")]["state"] == "unavailable"
    assert snaps[("h1", "s2", "arm1")]["interactions"]
    assert snaps[("h1", "exa", "arm1")]["state"] == "unavailable"
    cassette = analyze.build_snapshot_cassette("h1", snaps, backfill)
    state = {"hypotheses": [Hypothesis.model_validate(pinned["hypothesis"])],
             "study_context": {"disease_focus": pinned["disease_context"]},
             "resolved_entities": [EntityResolution(raw_name=n, curie=c, resolved_name=n)
                                   for c, n in pinned["entity_names"].items()],
             "direct_findings": [], "disease_associations": []}
    result = await analyze.p6_check(state, cassette, kept)
    assert result["identical"] is True
    assert result["unmatched"] == []
    assert result["production_keys"] == [analyze.unique_key(r) for r in kept]

    mutated = list(reversed(kept))
    with pytest.raises(analyze.ReplayMismatch):
        await analyze.p6_check(state, cassette, mutated)

    # A cassette missing the EFetch backfill is an unmatched request -> halt.
    thin = analyze.build_snapshot_cassette("h1", snaps, [])
    with pytest.raises(analyze.ReplayMismatch):
        await analyze.p6_check(state, thin, kept)


async def test_p6_kg_path_replay(fast_clients):
    pinned = make_pinned("k1", kg_pmids={"LABEL:leucine": ["34044180"]})
    with respx.mock(assert_all_called=False) as router:
        _respx_real_routes(router)
        snaps = {("k1", "kg", "arm1"): await sources.snapshot_arm1(pinned, "kg")}
        kept = analyze.replay_kept("k1", snaps, "current")
        backfill = await sources.record_backfill(pinned, kept)
    cassette = analyze.build_snapshot_cassette("k1", snaps, backfill)
    state = {"hypotheses": [Hypothesis.model_validate(pinned["hypothesis"])],
             "study_context": {"disease_focus": pinned["disease_context"]},
             "resolved_entities": [],
             "direct_findings": [Finding(entity="LABEL:leucine", claim="c", tier=1, pmids=["34044180"])],
             "disease_associations": []}
    result = await analyze.p6_check(state, cassette, kept)
    assert result["identical"] and result["production_keys"] == ["pmid:34044180"]


def test_recommendation_table_renders_every_source():
    t = analyze.apply_decision_rule(_inputs(), RULE)
    md = analyze.render_recommendation_table(t)
    for s in ("pubmed", "openalex", "exa", "s2", "consensus", "Judge filter"):
        assert s in md


# --- resume must redo exhausted transient failures and judge not_run rows (review P1/P2) ---

def test_exhausted_transient_and_not_run_rows_are_redone_on_resume(tmp_path):
    ck = run_study.JsonlCheckpoint(tmp_path / "rows.jsonl", key_fields=("id",))
    ck.append({"id": 1, "state": "ok"})
    ck.append({"id": 2, "state": "lookup_failed", "transient": True, "retries_exhausted": True})
    ck.append({"id": 3, "state": "lookup_failed", "transient": False})  # e.g. a 403: final
    ck.append({"id": 4, "status": "not_run", "error": "unparseable"})
    ck.append({"id": 5, "state": "not_run", "reason": "no named entity"})  # deterministic: final
    assert ck.done_keys() == {(1,), (3,), (5,)}

    calls = []

    async def work(item):
        calls.append(item["id"])
        return {"id": item["id"], "state": "ok"}

    rows = asyncio.run(run_study.run_stage_items([{"id": i} for i in range(1, 6)], work, ck, rules_dir=RULES))
    assert calls == [2, 4]
    assert {r["id"]: r["state"] if "state" in r else r["status"] for r in rows} == {
        1: "ok", 2: "ok", 3: "lookup_failed", 4: "ok", 5: "not_run"}


def test_torn_final_line_is_skipped_and_redone(tmp_path):
    p = tmp_path / "rows.jsonl"
    p.write_text('{"id": 1, "state": "ok"}\n{"id": 2, "sta')
    ck = run_study.JsonlCheckpoint(p, key_fields=("id",))
    assert ck.done_keys() == {(1,)}


async def test_snapshot_with_retry_counts_interactions_over_all_attempts():
    attempts = iter([
        {"state": "lookup_failed", "transient": True, "interactions": [{"s": 503}]},
        {"state": "ok", "interactions": [{"s": 200}]},
    ])

    async def snap(pinned, source):
        return dict(next(attempts))

    async def no_sleep(_):
        return None

    rec = await sources.snapshot_with_retry({}, "exa", snap_fn=snap, sleep=no_sleep)
    assert rec["state"] == "ok" and rec["interactions_all_attempts"] == 2


def test_audit_packet_ids_are_opaque_and_map_back_on_ingest(tmp_path):
    sample = [{"item_id": "H1|exa|arm1|10.1/x", "claim": "c", "title": "t", "abstract": "a"},
              {"item_id": "H2|pubmed|arm2|PMID:1", "claim": "c2", "title": "t2", "abstract": "a2"}]
    packet = judge.export_audit_packet(sample, tmp_path / "packet.jsonl", seed=7)
    text = packet.read_text()
    assert "exa" not in text and "pubmed" not in text and "arm" not in text
    rows = [json.loads(line) for line in text.splitlines()]
    labels_path = tmp_path / "labels.jsonl"
    labels_path.write_text("".join(json.dumps({"item_id": r["item_id"], "hit": r["title"] == "t"}) + "\n"
                                   for r in rows))
    labels = judge.ingest_audit_labels(labels_path, key_path=Path(f"{packet}.key.json"))
    assert labels == {"H1|exa|arm1|10.1/x": True, "H2|pubmed|arm2|PMID:1": False}


# --- Unit 4 tail driver: pure helpers ---

from kestrel_backend.evals.lit_sources import run_tail  # noqa: E402


def _pin(hyp_id, kind, claim="A affects B", **kw):
    return {"hyp_id": hyp_id, "kind": kind, "hypothesis": {"claim": claim}, **kw}


def test_build_items_dedups_across_sources_and_injects_p2_twins():
    pinned = [_pin("R1", "real"), _pin("T4", "control_true", "TCF7L2 -> T2D"),
              _pin("P2", "control_corrupt", "TCF7L2 -> T1D", twin_of="T4",
                   inject_landmarks=[{"pmid": "16415884", "doi": "10.1038/ng1732", "title": "Variant of TCF7L2"}])]
    recs = [
        {"hyp_id": "R1", "source": "openalex", "arm": "arm1", "papers": [{"paper_id": "W1", "doi": "10.1/X", "title": "t"}]},
        {"hyp_id": "R1", "source": "exa", "arm": "arm1", "papers": [{"paper_id": "e", "doi": "10.1/x", "title": "t"}]},
    ]
    items = run_tail.build_items(recs, pinned)
    r1 = [i for i in items.values() if i["hyp_id"] == "R1"]
    assert len(r1) == 1 and sorted(r1[0]["sources"]) == ["exa", "openalex"]
    roles = {i["hyp_id"]: i["roles"] for i in items.values() if i["hyp_id"] in ("P2", "T4")}
    assert roles == {"P2": ["p2_corrupt"], "T4": ["p2_clean"]}


def test_false_support_events_only_supporting_hits_on_false_controls():
    items = {"F1|doi:a": {"item_id": "F1|doi:a", "hyp_id": "F1", "paper_key": "doi:a", "kind": "control_false"},
             "F1|doi:b": {"item_id": "F1|doi:b", "hyp_id": "F1", "paper_key": "doi:b", "kind": "control_false"},
             "R1|doi:c": {"item_id": "R1|doi:c", "hyp_id": "R1", "paper_key": "doi:c", "kind": "real"}}
    judgments = [{"item_id": "F1|doi:a", "status": "judged", "hit": True, "direction": "supports"},
                 {"item_id": "F1|doi:b", "status": "judged", "hit": True, "direction": "refutes"},
                 {"item_id": "R1|doi:c", "status": "judged", "hit": True, "direction": "supports"}]
    assert [e["item_id"] for e in run_tail.false_support_events(judgments, items)] == ["F1|doi:a"]


def test_pubmed_pairs_use_matched_sentence_calls_on_real_hypotheses_only():
    pinned = [_pin("R1", "real"), _pin("T1", "control_true")]
    recs = [{"hyp_id": "R1", "source": "pubmed", "arm": "arm1", "state": "empty", "papers": []},
            {"hyp_id": "R1", "source": "pubmed", "arm": "arm1_direct", "state": "ok", "papers": [{}, {}]},
            {"hyp_id": "T1", "source": "pubmed", "arm": "arm1", "state": "empty", "papers": []},
            {"hyp_id": "T1", "source": "pubmed", "arm": "arm1_direct", "state": "ok", "papers": [{}]}]
    assert run_tail.pubmed_pairs(recs, pinned) == [(0, 2)]


def test_json_safe_stringifies_tuple_keys_so_capture_state_serializes():
    state = {"specificity_by_bridge": {("CHEBI:1", "HGNC:2"): {"score": 1}}, "x": [{("a",): 1}]}
    out = run_study.json_safe(state)
    assert json.loads(json.dumps(out)) == {"specificity_by_bridge": {"CHEBI:1|HGNC:2": {"score": 1}},
                                           "x": [{"a": 1}]}
