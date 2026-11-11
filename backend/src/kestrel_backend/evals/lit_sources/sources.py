"""Source calls, call states and normalization (Unit 2).

* ``CallRecorder`` wraps ``httpx.AsyncClient.send`` (the ``assessment/capture.py`` pattern) and
  records every request with its status and body, or ``{exception_type, url}`` when it raises.
  The production clients turn most failures into ``[]``; the recorder is how a failure stays a
  failure here (R7a). URLs are stored with API keys redacted.
* ``classify_call`` maps one (hypothesis, source, arm) call to exactly one state:
  ``unavailable`` / ``lookup_failed`` / ``empty`` / ``ok`` (plus ``not_run`` for work never done).
* Arm 1 calls the production ``*_raw`` functions unchanged; Arm 2 PubMed and the matched
  same-query pair use a direct E-utilities client that raises instead of swallowing.
* Consensus is collected by Trent's interactive session from a work-list; this module only emits
  the work-list and derives states from the raw connector output.
"""

from __future__ import annotations

import difflib
import json
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable
from unittest.mock import patch

import ftfy
import httpx

from ... import exa_client, pubmed_client, semantic_scholar
from ...graph.nodes import literature_grounding as lg
from ...graph.pipeline_config import get_pipeline_config
from ...graph.state import _MOJIBAKE_REPLACEMENTS, Hypothesis, LiteratureSupport

CALL_STATES = ("ok", "empty", "lookup_failed", "unavailable", "not_run")

HOST_SOURCE = {
    "eutils.ncbi.nlm.nih.gov": "pubmed",
    "api.openalex.org": "openalex",
    "api.exa.ai": "exa",
    "api.semanticscholar.org": "s2",
}

_SECRET_PARAMS = {"api_key", "apikey", "key", "x-api-key", "mailto", "email"}
_TRANSIENT_EXC = ("Timeout", "ConnectError", "NetworkError", "RemoteProtocolError", "ReadError",
                  "WriteError", "ProtocolError")

_RAW_FNS = {
    "openalex": lg.ground_hypothesis_openalex_raw,
    "exa": lg.ground_hypothesis_exa_raw,
    "pubmed": lg.ground_hypothesis_pubmed_raw,
    "s2": lg.ground_hypothesis_s2_raw,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_query_rules(rules_dir: Path) -> dict[str, Any]:
    return json.loads((Path(rules_dir) / "query_rules.json").read_text())


# ---------------------------------------------------------------------------------------------
# Recording and call states
# ---------------------------------------------------------------------------------------------


def redact_url(url: str) -> str:
    u = httpx.URL(url)
    if not u.query:
        return str(u)
    kept = [(k, v) for k, v in u.params.multi_items() if k.lower() not in _SECRET_PARAMS]
    return str(u.copy_with(params=kept)) if kept else str(u.copy_with(query=None))


def host_source(url: str) -> str | None:
    return HOST_SOURCE.get(httpx.URL(url).host)


class CallRecorder:
    """Context manager: record every httpx request made while active (redacted, with bodies)."""

    def __init__(self) -> None:
        self.interactions: list[dict[str, Any]] = []
        self._patch = None

    def __enter__(self) -> "CallRecorder":
        original_send = httpx.AsyncClient.send
        recorder = self

        async def recording_send(client, request, *args, **kwargs):
            started, t0 = _now(), time.monotonic()
            base = {
                "source": host_source(str(request.url)),
                "method": request.method,
                "url": redact_url(str(request.url)),
                "content": request.content.decode("utf-8") if request.content else None,
                "started_at": started,
            }
            try:
                response = await original_send(client, request, *args, **kwargs)
            except Exception as e:
                recorder.interactions.append({**base, "status_code": None, "text": None,
                                              "exception_type": type(e).__name__,
                                              "latency_s": round(time.monotonic() - t0, 3)})
                raise
            recorder.interactions.append({
                **base, "status_code": response.status_code, "text": response.text,
                "content_type": response.headers.get("content-type", ""),
                "exception_type": None, "latency_s": round(time.monotonic() - t0, 3),
            })
            return response

        self._patch = patch.object(httpx.AsyncClient, "send", recording_send)
        self._patch.start()
        return self

    def __exit__(self, *exc) -> None:
        self._patch.stop()


def key_present(source: str) -> bool:
    """Read at call time (the clients read keys at import; tests monkeypatch the module attrs)."""
    if source == "exa":
        return bool(exa_client.EXA_API_KEY)
    if source == "s2":
        return bool(semantic_scholar.S2_API_KEY)
    return True


def _json(text: str | None) -> Any:
    if not text:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def body_error(source: str, text: str | None) -> tuple[str | None, list[str]]:
    """Return (error, warnings) for a 2xx body that still carries a source error."""
    warnings: list[str] = []
    if source == "pubmed":
        if text and text.lstrip().startswith("<") and "<ERROR>" in text:
            return "NCBI XML <ERROR>", warnings
        data = _json(text)
        if isinstance(data, dict):
            if data.get("error"):
                return f"NCBI error: {data['error']}", warnings
            es = data.get("esearchresult") or {}
            if es.get("ERROR"):
                return f"NCBI ERROR: {es['ERROR']}", warnings
            wl = es.get("warninglist") or {}
            if wl.get("phrasesignored") or wl.get("quotedphrasesnotfound") or wl.get("outputmessages"):
                # Informational: NCBI drops stop-words on most sentence queries, with or without hits,
                # and reports a genuine zero-hit search as "No items found." Never a lookup failure.
                warnings.append(f"NCBI WarningList: {wl}")
            el = es.get("errorlist") or {}
            if el.get("phrasesnotfound") or el.get("fieldsnotfound"):
                warnings.append(f"NCBI errorlist (query terms dropped): {el}")
        return None, warnings
    data = _json(text)
    if isinstance(data, dict) and data.get("error"):
        return f"{source} error body: {str(data['error'])[:200]}", warnings
    return None, warnings


def classify_call(source: str, interactions: list[dict[str, Any]], *, key_present: bool,
                  n_items: int) -> dict[str, Any]:
    """Exactly one state per call. Never ``ok`` with an exception, never ``empty`` on failure."""
    if not key_present:
        return {"state": "unavailable", "reason": f"{source} key absent", "transient": False, "warnings": []}
    mine = [i for i in interactions if host_source(i["url"]) == source]
    if not mine:
        return {"state": "lookup_failed", "reason": "no expected-host interaction while key present",
                "transient": False, "warnings": []}
    warnings: list[str] = []
    for i in mine:
        if i.get("exception_type"):
            transient = any(t in i["exception_type"] for t in _TRANSIENT_EXC)
            return {"state": "lookup_failed", "reason": f"raised {i['exception_type']} for {i['url']}",
                    "transient": transient, "warnings": warnings}
        status = i.get("status_code") or 0
        if not 200 <= status < 300:
            return {"state": "lookup_failed", "reason": f"HTTP {status} for {i['url']}",
                    "transient": status >= 500 or status == 429, "warnings": warnings}
        err, warns = body_error(source, i.get("text"))
        warnings.extend(warns)
        if err:
            return {"state": "lookup_failed", "reason": err, "transient": False, "warnings": warnings}
    state = "ok" if n_items > 0 else "empty"
    return {"state": state, "reason": None, "transient": False, "warnings": warnings}


# ---------------------------------------------------------------------------------------------
# Arm 1 (production) and direct PubMed (matched pair + Arm 2)
# ---------------------------------------------------------------------------------------------


def _hyp(pinned: dict[str, Any]) -> Hypothesis:
    return Hypothesis.model_validate(pinned["hypothesis"])


def _record(pinned, source, arm, query, papers, state, interactions, started, t0, errors=None):
    return {
        "hyp_id": pinned["hyp_id"], "kind": pinned.get("kind"), "source": source, "arm": arm,
        "query": query, "papers": [p.model_dump() if hasattr(p, "model_dump") else p for p in papers],
        "state": state["state"], "reason": state.get("reason"), "transient": state.get("transient", False),
        "warnings": state.get("warnings", []), "errors": errors or [], "interactions": interactions,
        "started_at": started, "latency_s": round(time.monotonic() - t0, 3),
    }


async def snapshot_arm1(pinned: dict[str, Any], source: str, *, limit: int | None = None) -> dict[str, Any]:
    """One cold Arm 1 call through the unchanged production ``*_raw`` function (or KG lookup)."""
    hyp = _hyp(pinned)
    limit = limit or get_pipeline_config().literature_grounding.parallel_fetch_limit
    dc = pinned.get("disease_context", "")
    started, t0 = _now(), time.monotonic()
    if source == "kg":
        updated, has_kg = lg.ground_hypothesis_kg(hyp, pinned.get("kg_pmids") or {})
        papers = list(updated.literature_support) if has_kg else []
        st = {"state": "ok" if papers else "empty", "reason": None}
        return _record(pinned, "kg", "arm1", None, papers, st, [], started, t0)

    query = lg.build_search_query(hyp, dc)
    kwargs: dict[str, Any] = {"limit": limit, "disease_context": dc}
    if source == "s2":
        kwargs["entity_names"] = pinned.get("entity_names") or {}
    raised = None
    with CallRecorder() as rec:
        try:
            papers, errors = await _RAW_FNS[source](hyp, **kwargs)
        except Exception as e:  # noqa: BLE001 - recorded as lookup_failed, never as empty
            papers, errors, raised = [], [f"{type(e).__name__}: {e}"], e
    st = classify_call(source, rec.interactions, key_present=key_present(source), n_items=len(papers))
    if raised is not None and st["state"] not in ("lookup_failed", "unavailable"):
        st = {"state": "lookup_failed", "reason": f"raised {type(raised).__name__}",
              "transient": False, "warnings": st.get("warnings", [])}
    return _record(pinned, source, "arm1", query, papers, st, rec.interactions, started, t0, errors)


async def snapshot_with_retry(pinned, source, *, attempts: int = 3, sleep=None, snap_fn=None):
    """Retry a transient (5xx/429/timeout) failure; an exhausted retry stays ``lookup_failed``."""
    import asyncio

    sleep = sleep or asyncio.sleep
    snap_fn = snap_fn or snapshot_arm1
    rec = None
    total_interactions = 0
    for attempt in range(attempts):
        rec = await snap_fn(pinned, source)
        total_interactions += len(rec.get("interactions") or [])
        rec["attempts"] = attempt + 1
        rec["interactions_all_attempts"] = total_interactions
        if not (rec["state"] == "lookup_failed" and rec.get("transient")):
            return rec
        if attempt + 1 < attempts:
            await sleep(2 ** attempt)
    rec["retries_exhausted"] = True
    return rec


class DirectPubMedError(Exception):
    pass


async def direct_pubmed_search(query: str, limit: int) -> list[dict[str, Any]]:
    """ESearch (relevance) -> ESummary over raw httpx. Raises on non-2xx or an NCBI error body."""
    base = pubmed_client.EUTILS_BASE
    key = pubmed_client._ncbi_api_key()
    async with httpx.AsyncClient(timeout=30.0) as client:
        params: dict[str, Any] = {"db": "pubmed", "term": query[:500], "retmode": "json",
                                  "retmax": limit, "sort": "relevance"}
        if key:
            params["api_key"] = key
        await _sleep(pubmed_client._pubmed_delay())
        r = await client.get(f"{base}/esearch.fcgi", params=params)
        if r.status_code != 200:
            raise DirectPubMedError(f"ESearch HTTP {r.status_code}")
        err, _ = body_error("pubmed", r.text)
        if err:
            raise DirectPubMedError(err)
        pmids = (r.json().get("esearchresult") or {}).get("idlist") or []
        if not pmids:
            return []
        params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "json"}
        if key:
            params["api_key"] = key
        await _sleep(pubmed_client._pubmed_delay())
        r = await client.get(f"{base}/esummary.fcgi", params=params)
        if r.status_code != 200:
            raise DirectPubMedError(f"ESummary HTTP {r.status_code}")
        err, _ = body_error("pubmed", r.text)
        if err:
            raise DirectPubMedError(err)
        result = r.json().get("result") or {}
        return [pubmed_client._normalize_esummary(result[p], p) for p in pmids
                if p in result and "error" not in result[p]]


async def _sleep(seconds: float) -> None:
    import asyncio

    if seconds:
        await asyncio.sleep(seconds)


def pubmed_boolean_query(pinned: dict[str, Any], rules: dict[str, Any]) -> str | None:
    """Frozen Arm 2 rule: (entity OR synonyms)[tiab] AND ... AND (disease)[tiab]."""
    r = rules["arm2"]["pubmed"]
    tag = r["field_tag"]
    hyp = _hyp(pinned)
    names = pinned.get("entity_names") or {}
    syns = pinned.get("synonyms") or {}

    def term(t: str) -> str:
        return f'"{t.replace(chr(34), "").strip()}"{tag}'

    groups = []
    for curie in hyp.supporting_entities[: r["max_entities"]]:
        name = names.get(curie)
        if not name:
            continue
        terms = [name] + [s for s in syns.get(curie, []) if s != name]
        groups.append("(" + " OR ".join(term(t) for t in terms) + ")")
    if not groups:
        return None
    dc = pinned.get("disease_context") or ""
    if dc:
        groups.append(f"({term(dc)})")
    return f" {r['join']} ".join(groups)


async def snapshot_pubmed_direct(pinned: dict[str, Any], *, arm: str, rules: dict[str, Any] | None = None,
                                 limit: int | None = None) -> dict[str, Any]:
    """``arm1_direct``: identical Arm 1 sentence (matched pair). ``arm2``: frozen Boolean rule."""
    limit = limit or get_pipeline_config().literature_grounding.parallel_fetch_limit
    started, t0 = _now(), time.monotonic()
    if arm == "arm1_direct":
        query = lg.build_search_query(_hyp(pinned), pinned.get("disease_context", ""))
    elif arm == "arm2":
        query = pubmed_boolean_query(pinned, rules)
        if query is None:
            st = {"state": "not_run", "reason": "no named entity to build the Boolean query"}
            return _record(pinned, "pubmed", arm, None, [], st, [], started, t0)
    else:
        raise ValueError(arm)
    papers: list[LiteratureSupport] = []
    errors: list[str] = []
    with CallRecorder() as rec:
        try:
            raw = await direct_pubmed_search(query, limit)
            papers = [lg.create_literature_from_pubmed(p, 0.88 - i * 0.03) for i, p in enumerate(raw)]
        except Exception as e:  # noqa: BLE001
            errors.append(f"{type(e).__name__}: {e}")
    st = classify_call("pubmed", rec.interactions, key_present=True, n_items=len(papers))
    if errors and st["state"] in ("ok", "empty"):
        transient = any(t in errors[0] for t in _TRANSIENT_EXC)
        st = {"state": "lookup_failed", "reason": errors[0], "transient": transient, "warnings": st["warnings"]}
    return _record(pinned, "pubmed", arm, query, papers, st, rec.interactions, started, t0, errors)


async def certify_direct_pubmed(rules: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover - live
    cfg = rules["pubmed_direct_matched"]
    with CallRecorder() as rec:
        try:
            raw = await direct_pubmed_search(cfg["certification_query"], 20)
            err = None
        except Exception as e:  # noqa: BLE001
            raw, err = [], f"{type(e).__name__}: {e}"
    return {"ok": err is None and len(raw) >= cfg["certification_min_pmids"], "n": len(raw),
            "error": err, "interactions": rec.interactions, "at": _now()}


async def record_backfill(pinned: dict[str, Any], kept: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Record the EFetch calls production ``run()`` makes for this kept set (P6 cassette input)."""
    lits = [LiteratureSupport.model_validate(r) for r in kept if r.get("source") in
            ("kg", "openalex", "s2", "exa", "pubmed")]
    hyp = _hyp(pinned).model_copy(update={"literature_support": lits})
    with CallRecorder() as rec:
        await lg.backfill_abstracts([hyp])
    return rec.interactions


# ---------------------------------------------------------------------------------------------
# Consensus (interactive connector) and Elicit (gated)
# ---------------------------------------------------------------------------------------------


def consensus_question(pinned: dict[str, Any], rules: dict[str, Any]) -> str:
    claim = _hyp(pinned).claim.strip().rstrip(".!?;: ")
    if claim and not re.match(r"[A-Z]{2,}", claim):
        claim = claim[0].lower() + claim[1:]
    dc = pinned.get("disease_context") or ""
    suffix = f" in {dc}" if dc and dc.lower() not in claim.lower() else ""
    return rules["arm2"]["consensus"]["template"].format(claim=claim, disease_suffix=suffix)


def emit_consensus_worklist(pinned_list: list[dict[str, Any]], path: Path,
                            rules: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [{"call_id": f"consensus:{p['hyp_id']}", "hyp_id": p["hyp_id"], "kind": p.get("kind"),
             "question": consensus_question(p, rules)} for p in pinned_list]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


_CONSENSUS_LINE = re.compile(
    r"^\s*\[(?P<rank>\d+)\]\s*\[(?P<title>.+?)\]\((?P<url>[^)\s]+)\)\s*\((?P<meta>[^)]*)\)")


def parse_consensus_raw(text: str) -> list[dict[str, Any]]:
    """Parse raw connector output: JSON list/object if it is JSON, else numbered markdown lines."""
    data = _json(text)
    items: list[dict[str, Any]] = []
    if isinstance(data, dict):
        data = data.get("papers") or data.get("results") or data.get("data")
    if isinstance(data, list):
        for i, d in enumerate(data):
            if not isinstance(d, dict) or not d.get("title"):
                continue
            authors = d.get("authors") or d.get("author") or ""
            if isinstance(authors, list):
                authors = authors[0] if authors else ""
                authors = authors.get("name", "") if isinstance(authors, dict) else authors
            year = d.get("year") or d.get("publication_year")
            items.append({"rank": i, "title": str(d["title"]), "url": d.get("url"),
                          "year": int(year) if str(year or "").isdigit() else None,
                          "first_author": _surname(str(authors)), "doi": d.get("doi")})
        return items
    for line in (text or "").splitlines():
        m = _CONSENSUS_LINE.match(line)
        if not m:
            continue
        meta = [s.strip() for s in m.group("meta").split(",")]
        year = next((int(s) for s in meta if re.fullmatch(r"(19|20)\d{2}", s)), None)
        items.append({"rank": len(items), "title": m.group("title"), "url": m.group("url"),
                      "year": year, "first_author": _surname(meta[0] if meta else ""), "doi": None})
    return items


def _surname(author: str) -> str | None:
    author = re.sub(r"\bet al\.?", "", author or "").strip()
    if not author:
        return None
    if "," in author:
        return author.split(",")[0].strip().casefold()
    parts = author.split()
    # "Newgard C" (surname first) vs "Christopher B. Newgard" (surname last)
    if len(parts) >= 2 and len(parts[-1].rstrip(".")) <= 2:
        return parts[0].casefold()
    return parts[-1].casefold()


def ingest_consensus(worklist: list[dict[str, Any]], raw_rows: list[dict[str, Any]], *,
                     preflight_ok: bool) -> list[dict[str, Any]]:
    """Derive one state per work-list call from the raw connector output Trent's session wrote."""
    by_id = {r["call_id"]: r for r in raw_rows}
    out = []
    for w in worklist:
        rec = {"hyp_id": w["hyp_id"], "kind": w.get("kind"), "source": "consensus", "arm": "native",
               "query": w["question"], "papers": [], "interactions": [], "transient": False,
               "warnings": [], "errors": []}
        r = by_id.get(w["call_id"])
        if not preflight_ok:
            out.append({**rec, "state": "unavailable", "reason": "connector preflight failed"})
            continue
        if r is None:
            out.append({**rec, "state": "not_run", "reason": "no raw output recorded for this call"})
            continue
        rec["started_at"] = r.get("ts")
        if r.get("error"):
            out.append({**rec, "state": "lookup_failed", "reason": f"connector error: {r['error']}"[:300]})
            continue
        papers = [{**p, "source": "consensus", "paper_id": p.get("url") or f"consensus:{p['rank']}",
                   "relevance_score": None} for p in parse_consensus_raw(r.get("output") or "")]
        out.append({**rec, "papers": papers, "raw_output": r.get("output"),
                    "state": "ok" if papers else "empty", "reason": None})
    return out


def resolve_consensus_title(paper: dict[str, Any], candidates: list[dict[str, Any]],
                            min_similarity: float = 0.9) -> dict[str, Any] | None:
    """Title similarity >= threshold AND same year AND first-author surname (when both known)."""
    want = normalize_title(paper.get("title"))
    if not want:
        return None
    best, best_sim = None, 0.0
    for c in candidates:
        got = normalize_title(c.get("title"))
        if not got:
            continue
        sim = difflib.SequenceMatcher(None, want, got).ratio()
        if sim < min_similarity:
            continue
        if paper.get("year") is None or c.get("year") is None or int(paper["year"]) != int(c["year"]):
            continue
        fa, fb = paper.get("first_author"), c.get("first_author")
        if fa and fb and _surname(fa) != _surname(fb):
            continue
        if sim > best_sim:
            best, best_sim = c, sim
    return best


def elicit_status(*, tos: str, headroom: bool) -> dict[str, Any]:
    """Tier 2 gate: built only if ToS is 'yes' AND the timing run shows budget headroom."""
    if tos != "yes":
        return {"source": "elicit", "state": "unavailable", "reason": f"ToS gate is '{tos}'"}
    if not headroom:
        return {"source": "elicit", "state": "unavailable", "reason": "no budget headroom (cut step 1)"}
    return {"source": "elicit", "state": "not_run",
            "reason": "gate passed; adapter to be built in Unit 4 (source-level diagnostics only)"}


# ---------------------------------------------------------------------------------------------
# Normalization: canonical IDs, landmark matching, abstracts
# ---------------------------------------------------------------------------------------------


def repair_text(text: str | None) -> str | None:
    if not text:
        return text
    for bad, good in _MOJIBAKE_REPLACEMENTS.items():
        text = text.replace(bad, good)
    return ftfy.fix_text(text)


def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    d = doi.strip()
    d = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", d, flags=re.IGNORECASE)
    return d.lower() or None


def normalize_title(title: str | None) -> str | None:
    if not title or title == "Unknown" or title.startswith("PubMed:"):
        return None
    t = unicodedata.normalize("NFKC", repair_text(title)).casefold()
    t = re.sub(r"[^0-9a-z]+", "", t)
    return t or None


def canonical_ids(paper: dict[str, Any]) -> dict[str, str | None]:
    pmid = None
    pid = str(paper.get("paper_id") or "")
    if pid.startswith("PMID:"):
        pmid = pid[5:].strip()
    elif pid.isdigit() and paper.get("source") in (None, "openalex"):
        pmid = pid
    url = paper.get("url") or ""
    m = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", url)
    if not pmid and m:
        pmid = m.group(1)
    if not pmid and paper.get("pmid"):
        pmid = str(paper["pmid"])
    doi = normalize_doi(paper.get("doi"))
    if not doi and "doi.org/" in url:
        doi = normalize_doi(url)
    return {"pmid": pmid if pmid and pmid.isdigit() else None, "doi": doi,
            "title_key": normalize_title(paper.get("title"))}


def paper_key(ids: dict[str, str | None]) -> str | None:
    if ids.get("doi"):
        return f"doi:{ids['doi']}"
    if ids.get("pmid"):
        return f"pmid:{ids['pmid']}"
    if ids.get("title_key"):
        return f"title:{ids['title_key']}"
    return None


def matches_landmark(ids: dict[str, str | None], landmark: dict[str, Any]) -> bool:
    """Match on EVERY identifier; title only as a fallback when no identifier conflicts."""
    lid = canonical_ids({"paper_id": f"PMID:{landmark.get('pmid')}" if landmark.get("pmid") else "",
                         "doi": landmark.get("doi"), "title": landmark.get("title")})
    for k in ("pmid", "doi"):
        if ids.get(k) and lid.get(k) and ids[k] == lid[k]:
            return True
    for k in ("pmid", "doi"):
        if ids.get(k) and lid.get(k) and ids[k] != lid[k]:
            return False
    return bool(ids.get("title_key") and ids["title_key"] == lid.get("title_key"))


def reconstruct_abstract(inverted: dict[str, list[int]] | None) -> str | None:
    if not inverted:
        return None
    pos = sorted((p, w) for w, ps in inverted.items() for p in ps)
    return " ".join(w for _, w in pos) or None


AbstractFetch = Callable[..., Awaitable[Any]]


async def resolve_abstract(ids: dict[str, str | None], *, title: str | None, efetch: AbstractFetch,
                           by_doi: AbstractFetch, by_title: AbstractFetch) -> tuple[str | None, str | None]:
    """One resolver chain for every source: PMID -> EFetch, DOI -> OpenAlex, title -> OpenAlex."""
    if ids.get("pmid"):
        got = await efetch([ids["pmid"]])
        if got and got.get(ids["pmid"]):
            return repair_text(got[ids["pmid"]]), "pmid"
    if ids.get("doi"):
        ab = await by_doi(ids["doi"])
        if ab:
            return repair_text(ab), "doi"
    if title and normalize_title(title):
        ab = await by_title(title)
        if ab:
            return repair_text(ab), "title"
    return None, None


async def openalex_abstract_by_doi(doi: str) -> str | None:  # pragma: no cover - live, Unit 4
    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.get(f"https://api.openalex.org/works/doi:{doi}",
                        params={"select": "abstract_inverted_index"})
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return reconstruct_abstract(r.json().get("abstract_inverted_index"))


async def openalex_title_candidates(title: str, n: int = 5) -> list[dict[str, Any]]:  # pragma: no cover
    clean = re.sub(r"[,:|]", " ", title)[:300]
    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.get("https://api.openalex.org/works",
                        params={"filter": f"title.search:{clean}", "per_page": n,
                                "select": "id,title,doi,ids,publication_year,authorships,abstract_inverted_index"})
        r.raise_for_status()
    out = []
    for w in r.json().get("results", []):
        auth = (w.get("authorships") or [{}])[0].get("author", {}).get("display_name", "")
        out.append({"title": w.get("title"), "year": w.get("publication_year"), "first_author": auth,
                    "doi": normalize_doi(w.get("doi")), "pmid": (w.get("ids") or {}).get("pmid", "").split("/")[-1] or None,
                    "abstract": reconstruct_abstract(w.get("abstract_inverted_index"))})
    return out


async def openalex_abstract_by_title(title: str) -> str | None:  # pragma: no cover - live
    for c in await openalex_title_candidates(title, n=3):
        if normalize_title(c["title"]) == normalize_title(title) and c.get("abstract"):
            return c["abstract"]
    return None


def normalization_report(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-source unresolved and no-abstract shares (R2a/R11)."""
    out: dict[str, dict[str, Any]] = {}
    for src in sorted({r["source"] for r in rows}):
        rs = [r for r in rows if r["source"] == src]
        n = len(rs)
        out[src] = {"n": n,
                    "unresolved_share": sum(1 for r in rs if not r.get("resolved")) / n,
                    "no_abstract_share": sum(1 for r in rs if not r.get("abstract")) / n}
    return out
