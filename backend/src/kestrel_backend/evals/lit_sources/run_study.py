"""Run harness for the literature-source comparison spike (Unit 1 + stage CLI).

Stages (run in order; each refuses to run if a frozen rule file's hash changed):

    init               manifest written FIRST (repo SHA, rule/CSV hashes, judge model+prompt hash,
                       KESTREL_MCP_URL, cache: cold, G0 lifted, budget counters)
    capture            3 fresh run_discovery captures (rules/capture.json); live KG-count recorded
    select             rehydrate -> KG-count gate -> round-robin selection -> pinned hypotheses
    snapshot           Arm 1 x source, PubMed direct matched pair + Arm 2, per-item JSONL resume
    consensus-worklist emit consensus_worklist.jsonl and STOP (Trent's session runs the connector)
    consensus-ingest   derive Consensus call states from the raw connector output

Normalization, judging, audit and analysis live in sources.py / judge.py / analyze.py and are
driven interactively in Unit 4. Output goes to ``~/kraken-studies/lit-sources/<UTC-ts>/`` by
default (outside the repo and any worktree); ``--run-dir`` only overrides it.

Run:  cd backend && uv run python -m kestrel_backend.evals.lit_sources.run_study init
"""

from __future__ import annotations

if __name__ == "__main__":  # API keys are read at client import time: load .env first (CLI only)
    from dotenv import load_dotenv

    load_dotenv()

import argparse
import asyncio
import csv
import hashlib
import io
import json
import math
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx

from ...analyte_ingest import validate_and_normalize
from ...graph.nodes import literature_grounding as lg
from ...graph.state import Hypothesis
from . import analyze, judge, sources

PKG_DIR = Path(__file__).parent
RULES_DIR = PKG_DIR / "rules"
CONTROLS_DIR = PKG_DIR / "controls"
LOCK_NAME = "rules_lock.json"
DEFAULT_ROOT = Path.home() / "kraken-studies" / "lit-sources"


class RulesHashMismatch(Exception):
    pass


class CaptureSpecIncomplete(Exception):
    pass


class NeedMoreCaptures(Exception):
    pass


class ControlsNotVerified(Exception):
    pass


class TransientError(Exception):
    """Raised by stage work to request a retry (5xx/timeout) without caching a result."""


TRANSIENT = (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError, TransientError)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------------------------
# Rule freeze
# ---------------------------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_rules_hashes(rules_dir: Path = RULES_DIR) -> dict[str, str]:
    return {p.name: _sha256(p) for p in sorted(Path(rules_dir).iterdir())
            if p.is_file() and p.name != LOCK_NAME}


def write_rules_lock(rules_dir: Path = RULES_DIR) -> dict[str, str]:
    """Freeze: called once at plan approval. Re-running it later is a logged deviation."""
    hashes = compute_rules_hashes(rules_dir)
    (Path(rules_dir) / LOCK_NAME).write_text(json.dumps(
        {"frozen_at": _now(), "algorithm": "sha256", "files": hashes}, indent=2) + "\n")
    return hashes


def verify_rules_lock(rules_dir: Path = RULES_DIR) -> dict[str, str]:
    lock_path = Path(rules_dir) / LOCK_NAME
    if not lock_path.exists():
        raise RulesHashMismatch(f"no {LOCK_NAME} in {rules_dir}: rules were never frozen")
    locked = json.loads(lock_path.read_text())["files"]
    current = compute_rules_hashes(rules_dir)
    if current != locked:
        changed = sorted(k for k in set(current) | set(locked) if current.get(k) != locked.get(k))
        raise RulesHashMismatch(f"frozen rule files changed since the freeze: {changed}")
    return current


# ---------------------------------------------------------------------------------------------
# Manifest, checkpoints, retry
# ---------------------------------------------------------------------------------------------


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


class Manifest:
    """Written at start, updated per stage. Every reported number must trace back to it."""

    def __init__(self, run_dir: Path, data: dict[str, Any]):
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / "manifest.json"
        self.data = data

    @classmethod
    def create(cls, run_dir: Path, *, repo_sha: str, rules_dir: Path = RULES_DIR,
               kestrel_mcp_url: str | None = None, kg_fingerprint: Any = None) -> "Manifest":
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "study": "literature-source comparison spike (plan 2026-11-11-001)",
            "created_at": _now(),
            "repo_sha": repo_sha,
            "rule_hashes": verify_rules_lock(rules_dir),
            "judge": {"model": judge.JUDGE_MODEL, "prompt_hash": judge.prompt_hash(),
                      "concurrency": judge.JUDGE_CONCURRENCY},
            "kestrel_mcp_url": kestrel_mcp_url,
            "kg_fingerprint": kg_fingerprint,
            "cache": "cold",
            "g0": "lifted 2026-11-11",
            "source_status": {},
            "stages": {},
            "budget": {"metered_usd": 0.0, "exa_calls": 0, "elicit_credits": 0, "judge_calls": 0,
                       "cap_metered_usd": 25.0, "cap_audit_readings": 60, "cap_wallclock_h": 2.0},
            "deviations": [],
        }
        m = cls(run_dir, data)
        m.save()
        return m

    @classmethod
    def load(cls, run_dir: Path) -> "Manifest":
        return cls(run_dir, json.loads((Path(run_dir) / "manifest.json").read_text()))

    def save(self) -> None:
        _atomic_write(self.path, json.dumps(self.data, indent=2, default=str))

    def update_stage(self, stage: str, status: str, **info: Any) -> None:
        self.data["stages"][stage] = {"status": status, "at": _now(), **info}
        self.save()

    def add_deviation(self, text: str) -> None:
        self.data["deviations"].append({"at": _now(), "text": text})
        self.save()

    def bump(self, counter: str, by: float = 1) -> None:
        self.data["budget"][counter] = self.data["budget"].get(counter, 0) + by
        self.save()


def is_retryable(row: dict[str, Any]) -> bool:
    """Rows a resume must redo: exhausted transient source failures and judge not_run rows.

    Deterministic outcomes (ok, empty, a 4xx lookup_failed, a snapshot not_run such as "no named
    entity for the Boolean query") are final.
    """
    if row.get("retries_exhausted"):
        return True
    if row.get("state") == "lookup_failed" and row.get("transient"):
        return True
    return row.get("status") == "not_run"


class JsonlCheckpoint:
    """Append-only per-item rows. The latest row per key wins; resume redoes retryable rows."""

    def __init__(self, path: Path, key_fields: tuple[str, ...]):
        self.path = Path(path)
        self.key_fields = key_fields
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn final line from a kill mid-write; that item is simply redone
        return rows

    def latest_rows(self) -> list[dict[str, Any]]:
        latest: dict[tuple, dict[str, Any]] = {}
        for r in self.rows():
            latest[self.key(r)] = r
        return list(latest.values())

    def key(self, row: dict[str, Any]) -> tuple:
        return tuple(row[k] for k in self.key_fields)

    def done_keys(self) -> set[tuple]:
        return {self.key(r) for r in self.latest_rows() if not is_retryable(r)}

    def append(self, row: dict[str, Any]) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())


async def call_with_retry(fn: Callable[[], Awaitable[Any]], *, attempts: int = 3,
                          sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> Any:
    """One transient-exception tuple; retry, then RAISE (never return a cached empty)."""
    for attempt in range(attempts):
        try:
            return await fn()
        except TRANSIENT:
            if attempt + 1 >= attempts:
                raise
            await sleep(2 ** attempt)
    raise AssertionError("unreachable")


async def run_stage_items(items: list[dict[str, Any]], work_fn: Callable[[dict], Awaitable[dict]],
                          checkpoint: JsonlCheckpoint, *, rules_dir: Path = RULES_DIR, attempts: int = 3,
                          sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> list[dict[str, Any]]:
    """Run work per item with resume; refuses to start if the frozen rules changed."""
    verify_rules_lock(rules_dir)
    done = checkpoint.done_keys()
    for item in items:
        if checkpoint.key(item) in done:
            continue
        row = await call_with_retry(lambda: work_fn(item), attempts=attempts, sleep=sleep)
        checkpoint.append(row)
    return checkpoint.latest_rows()


# ---------------------------------------------------------------------------------------------
# Panel CSV parsing (mirrors client/src/lib/analyteParse.ts) and capture inputs
# ---------------------------------------------------------------------------------------------

_ANALYTE = [r"^analyte$", r"^metabolite$", r"^protein$", r"^gene$", r"^feature$"]
_GROUP = [r"^module$", r"^group$", r"^category$", r"^cluster$"]
_TYPE = [r"^analyte[_ ]?type$", r"^omic$", r"^omics$"]
_KME = [r"^k[._ ]?me$", r"^signed[_ ]?kme$", r"^module[_ ]?membership$"]
_KIM = [r"^k[._ ]?im$", r"^k[._ ]?within$", r"^intramodular[_ ]?connectivity$"]
_RECOGNIZED_TYPES = ("metabolite", "protein", "gene")
_KME_EPSILON = 1e-6


def _match(patterns: list[str], header: str) -> bool:
    return any(re.match(p, header, flags=re.IGNORECASE) for p in patterns)


def suggest_mapping(headers: list[str]) -> dict[str, str]:
    """Mirror of ``suggestMapping``: unambiguous headers only, first match wins per target."""
    mapping: dict[str, str] = {}
    for h in headers:
        if "analyte" not in mapping and _match(_ANALYTE, h):
            mapping["analyte"] = h
        elif "group" not in mapping and _match(_GROUP, h):
            mapping["group"] = h
        elif "type" not in mapping and _match(_TYPE, h):
            mapping["type"] = h
        elif "kme" not in mapping and _match(_KME, h):
            mapping["kme"] = h
        elif "kim" not in mapping and _match(_KIM, h):
            mapping["kim"] = h
    return mapping


def _finite(raw: str | None) -> float | None | str:
    s = (raw or "").strip()
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return "invalid"
    return v if math.isfinite(v) else "invalid"


def _kme(raw: str | None) -> float | None:
    v = _finite(raw)
    if v is None or v == "invalid" or v < -1 - _KME_EPSILON or v > 1 + _KME_EPSILON:
        return None
    return max(-1.0, min(1.0, v))


def _kim(raw: str | None) -> float | None:
    v = _finite(raw)
    if v is None or v == "invalid" or v < 0:
        return None
    return v


def parse_panel_csv(path: Path) -> dict[str, Any]:
    """Mirror of ``parseDelimitedText`` + ``buildAnalytes`` (trim, skip empty names, dedup by
    (lower(name), group), unknown type -> unset, only valid weight cells attached)."""
    path = Path(path)
    text = path.read_text()
    delim = "\t" if path.suffix.lower() in (".tsv", ".tab") else ","
    matrix = [r for r in csv.reader(io.StringIO(text), delimiter=delim) if any(c.strip() for c in r)]
    if not matrix:
        raise ValueError("No rows found in the file.")
    headers = [h.strip() for h in matrix[0]]
    if any(h == "" for h in headers):
        raise ValueError("The header row has a blank column name.")
    if len({h.lower() for h in headers}) != len(headers):
        raise ValueError("Duplicate column header.")
    rows = []
    for i, cells in enumerate(matrix[1:]):
        if any(c.strip() for c in cells[len(headers):]):
            raise ValueError(f"Row {i + 2} has more values than there are column headers.")
        rows.append({h: (cells[j].strip() if j < len(cells) else "") for j, h in enumerate(headers)})
    mapping = suggest_mapping(headers)
    if "analyte" not in mapping:
        raise ValueError(f"no analyte column among {headers}")

    analytes: list[dict[str, Any]] = []
    retained: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = row[mapping["analyte"]].strip()
        if not name:
            continue
        group = row[mapping["group"]].strip() if "group" in mapping else ""
        type_ = row[mapping["type"]].strip().lower() if "type" in mapping else ""
        kme = _kme(row.get(mapping["kme"])) if "kme" in mapping else None
        kim = _kim(row.get(mapping["kim"])) if "kim" in mapping else None
        key = f"{name.lower()}\0{group}"
        kept = retained.get(key)
        if kept is not None:
            if "kme" not in kept and kme is not None:
                kept["kme"] = kme
                if kim is not None:
                    kept["kim"] = kim
                else:
                    kept.pop("kim", None)
            continue
        a: dict[str, Any] = {"name": name}
        if group:
            a["group"] = group
        if type_ in _RECOGNIZED_TYPES:
            a["type"] = type_
        if kme is not None:
            a["kme"] = kme
        if kim is not None:
            a["kim"] = kim
        analytes.append(a)
        retained[key] = a
    groups: list[str] = []
    for a in analytes:
        if a.get("group") and a["group"].lower() not in [g.lower() for g in groups]:
            groups.append(a["group"])
    return {"structured_analytes": analytes, "groups": groups, "mapping": mapping}


def validate_panel(parsed: dict[str, Any], module_directions: list[dict] | None = None):
    """The backend R19 gate (``analyte_ingest.validate_and_normalize``) on the parsed panel."""
    return validate_and_normalize(parsed["structured_analytes"], [], None,
                                  module_directions=module_directions or [])


def load_capture_specs(rules_dir: Path = RULES_DIR) -> list[dict[str, Any]]:
    return json.loads((Path(rules_dir) / "capture.json").read_text())["captures"]


def capture_inputs(spec: dict[str, Any], rules_dir: Path = RULES_DIR) -> dict[str, Any]:
    """run_discovery inputs for one capture spec; refuses an incomplete spec."""
    directions = spec.get("module_directions")
    if directions == "UNSET" or not isinstance(directions, list):
        raise CaptureSpecIncomplete(f"capture {spec['id']}: module_directions is UNSET")
    analytes: list[dict[str, Any]] = []
    if spec.get("panel_csv"):
        parsed = parse_panel_csv(Path(rules_dir) / spec["panel_csv"])
        panel = validate_panel(parsed, directions)
        if panel.errors:
            raise CaptureSpecIncomplete(f"capture {spec['id']}: panel rejected: {panel.errors}")
        analytes = parsed["structured_analytes"]
    return {"query": spec["query"], "structured_analytes": analytes,
            "selected_groups": spec.get("selected_groups") or [], "module_directions": directions}


async def capture_run(spec: dict[str, Any], run_dir: Path, rules_dir: Path = RULES_DIR) -> dict[str, Any]:  # pragma: no cover - live
    """One fresh pipeline capture; the live KG PMID count is recorded for the rehydration gate."""
    from ...assessment.capture import _serialize_state
    from ...graph.runner import run_discovery

    inputs = capture_inputs(spec, rules_dir)
    t0 = time.monotonic()
    state = await run_discovery(inputs["query"], structured_analytes=inputs["structured_analytes"],
                                selected_groups=inputs["selected_groups"],
                                module_directions=inputs["module_directions"])
    out = {"capture_id": spec["id"], "inputs": inputs, "captured_at": _now(),
           "duration_s": round(time.monotonic() - t0, 1),
           "live_kg_pmid_count": analyze.kg_pmid_total(state),
           "state": _serialize_state(state)}
    path = Path(run_dir) / "captures" / f"{spec['id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, json.dumps(out, indent=2, default=str))
    return out


# ---------------------------------------------------------------------------------------------
# Selection and pinning
# ---------------------------------------------------------------------------------------------


def classify_hyp_type(title: str) -> str:
    if title.startswith("Bridge:"):
        return "bridge"
    if title.startswith(("Inferred role of", "Inferred:")):
        return "inferred_role"
    return "other"


def run_from_capture(capture: dict[str, Any], path: str | None = None) -> dict[str, Any]:
    """Rehydrate a capture, enforce the KG-count gate, and expose what pinning needs."""
    state = analyze.rehydrate_state(capture["state"])
    analyze.check_kg_gate(capture["live_kg_pmid_count"], state)
    sc = state.get("study_context") or {}
    return {
        "capture_id": capture["capture_id"],
        "hypotheses": state.get("hypotheses") or [],
        "disease_context": sc.get("disease_focus", "") if isinstance(sc, dict) else "",
        "entity_names": {r.curie: r.resolved_name for r in state.get("resolved_entities", [])
                         if r.curie and r.resolved_name},
        "kg_pmids": lg.collect_pmids_from_state(state),
        "source_path": path,
    }


def _dedup_key(h: Hypothesis) -> str:
    return re.sub(r"\s+", " ", f"{h.title}|{h.claim}".lower()).strip()


def _coverage(selected: list[dict[str, Any]], cov: dict[str, Any]) -> bool:
    panels = {p["panel"] for p in selected}
    types = {p["hyp_type"] for p in selected}
    return len(panels) >= cov["min_panels"] and set(cov["required_types"]) <= types


def select_hypotheses(runs: list[dict[str, Any]], *, target: int = 10, min_n: int = 6,
                      coverage: dict[str, Any] | None = None) -> dict[str, Any]:
    """Round-robin across runs, each in production tier order, deduplicated, until ``target``;
    continue past ``target`` until coverage is met. No stage filters on grounding output."""
    coverage = coverage or {"min_panels": 2, "required_types": ["bridge", "inferred_role"]}
    queues = []
    for run in runs:
        hyps = [h if isinstance(h, Hypothesis) else Hypothesis.model_validate(h) for h in run["hypotheses"]]
        indexed = sorted(enumerate(hyps), key=lambda t: t[1].tier)  # stable: production order
        queues.append((run, indexed))
    cursors = [0] * len(queues)
    seen: set[str] = set()
    selected: list[dict[str, Any]] = []

    def next_from(qi: int) -> dict[str, Any] | None:
        run, indexed = queues[qi]
        while cursors[qi] < len(indexed):
            idx, h = indexed[cursors[qi]]
            cursors[qi] += 1
            k = _dedup_key(h)
            if k in seen:
                continue
            seen.add(k)
            return {
                "hyp_id": f"{run['capture_id']}#{idx}", "kind": analyze.REAL,
                "hypothesis": h.model_dump(), "disease_context": run.get("disease_context", ""),
                "entity_names": run.get("entity_names") or {}, "synonyms": {},
                "kg_pmids": {c: run["kg_pmids"][c] for c in h.supporting_entities
                             if c in (run.get("kg_pmids") or {})},
                "panel": run["capture_id"], "hyp_type": classify_hyp_type(h.title),
                "source_run": run.get("source_path"), "source_index": idx,
            }
        return None

    while True:
        progressed = False
        for qi in range(len(queues)):
            if len(selected) >= target and _coverage(selected, coverage):
                break
            item = next_from(qi)
            if item is not None:
                selected.append(item)
                progressed = True
        if (len(selected) >= target and _coverage(selected, coverage)) or not progressed:
            break
    if len(selected) < min_n:
        raise NeedMoreCaptures(f"only {len(selected)} distinct hypotheses (< {min_n}); capture one more run")
    return {"selected": selected, "fallback": len(selected) < target,
            "coverage_met": _coverage(selected, coverage)}


# ---------------------------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------------------------


def _control(link: dict[str, Any], kind: str, b: str | None = None) -> dict[str, Any]:
    a, b = link["a"], b or link["b"]
    ca, cb = f"LABEL:{a}", f"LABEL:{b}"
    hyp = Hypothesis(title=f"Bridge: {a} -> {b}", tier=1, claim=link["claim"],
                     supporting_entities=[ca, cb], structural_logic=f"{a} -> {b}",
                     validation_steps=["Verify path in literature"])
    syn = link.get("synonyms") or {}
    return {"hyp_id": link["id"], "kind": kind, "hypothesis": hyp.model_dump(),
            "disease_context": link["disease_context"], "entity_names": {ca: a, cb: b},
            "synonyms": {c: syn[k] for k, c in (("a", ca), ("b", cb)) if syn.get(k)},
            "kg_pmids": {}, "panel": "controls", "hyp_type": "bridge"}


def load_controls(controls_dir: Path = CONTROLS_DIR) -> list[dict[str, Any]]:
    true_doc = json.loads((Path(controls_dir) / "true_links.json").read_text())
    false_doc = json.loads((Path(controls_dir) / "false_links.json").read_text())
    out = []
    for link in true_doc["links"]:
        out.append({**_control(link, "control_true"), "landmarks": link["landmarks"],
                    "obscurity": link.get("obscurity")})
    corr = true_doc["corruption"]
    base = next(l for l in true_doc["links"] if l["id"] == corr["base_id"])
    twin = {**base, "id": corr["id"], "claim": corr["claim"], "disease_context": corr["disease_context"]}
    out.append({**_control(twin, "control_corrupt", b=corr["replace_b_with"]),
                "twin_of": base["id"], "inject_landmarks": base["landmarks"], "landmarks": []})
    for link in false_doc["links"]:
        out.append({**_control(link, "control_false"), "pre_run_search": link.get("pre_run_search"),
                    "landmarks": []})
    return out


def require_verified_controls(controls: list[dict[str, Any]]) -> None:
    """Live gates refuse controls whose landmarks are unverified or false links unsearched."""
    problems = []
    for c in controls:
        for lm in c.get("landmarks") or []:
            if not lm.get("verified_at"):
                problems.append(f"{c['hyp_id']}: landmark PMID {lm.get('pmid')} not PubMed-verified")
        if c["kind"] == "control_false" and not c.get("pre_run_search"):
            problems.append(f"{c['hyp_id']}: no recorded pre-run search")
    if problems:
        raise ControlsNotVerified("; ".join(problems))


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def default_run_dir() -> Path:
    return DEFAULT_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def repo_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PKG_DIR, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


async def _snapshot_stage(run_dir: Path, manifest: Manifest) -> None:  # pragma: no cover - live
    pinned = json.loads((run_dir / "pinned_hypotheses.json").read_text())
    rules = sources.load_query_rules(RULES_DIR)
    ck = JsonlCheckpoint(run_dir / "snapshot.jsonl", key_fields=("hyp_id", "source", "arm"))
    items = []
    for p in pinned:
        for src in ("kg", "openalex", "exa", "pubmed", "s2"):
            if src == "kg" and p["kind"] != analyze.REAL:
                continue
            items.append({"hyp_id": p["hyp_id"], "source": src, "arm": "arm1", "_p": p})
        items.append({"hyp_id": p["hyp_id"], "source": "pubmed", "arm": "arm1_direct", "_p": p})
        items.append({"hyp_id": p["hyp_id"], "source": "pubmed", "arm": "arm2", "_p": p})

    async def work(item):
        p = item["_p"]
        if item["arm"] == "arm1":
            rec = await sources.snapshot_with_retry(p, item["source"])
        else:
            rec = await sources.snapshot_with_retry(
                p, "pubmed",
                snap_fn=lambda pin, _src, arm=item["arm"]: sources.snapshot_pubmed_direct(pin, arm=arm, rules=rules))
        if item["source"] == "exa" and rec.get("interactions_all_attempts"):
            manifest.bump("exa_calls", rec["interactions_all_attempts"])
        return rec

    await run_stage_items(items, work, ck)
    latest = ck.latest_rows()
    manifest.update_stage("snapshot", "complete", rows=len(latest),
                          retryable_remaining=sum(1 for r in latest if is_retryable(r)))


def main() -> None:  # pragma: no cover - CLI glue
    parser = argparse.ArgumentParser(description="Literature-source comparison spike harness")
    parser.add_argument("stage", choices=["init", "capture", "select", "snapshot", "consensus-worklist",
                                          "consensus-ingest"])
    parser.add_argument("--run-dir", default=None, help="override (default ~/kraken-studies/lit-sources/<ts>)")
    parser.add_argument("--consensus-raw", default=None, help="JSONL of raw connector outputs")
    parser.add_argument("--consensus-preflight-ok", action="store_true")
    args = parser.parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir()

    if args.stage == "init":
        m = Manifest.create(run_dir, repo_sha=repo_sha(), kestrel_mcp_url=os.getenv("KESTREL_MCP_URL"))
        print(f"Manifest written to {m.path}")
        return
    manifest = Manifest.load(run_dir)
    verify_rules_lock()
    if args.stage == "capture":
        for spec in load_capture_specs():
            out = asyncio.run(capture_run(spec, run_dir))
            manifest.update_stage(f"capture:{spec['id']}", "complete", duration_s=out["duration_s"],
                                  live_kg_pmid_count=out["live_kg_pmid_count"])
    elif args.stage == "select":
        runs = []
        for spec in load_capture_specs():
            path = run_dir / "captures" / f"{spec['id']}.json"
            runs.append(run_from_capture(json.loads(path.read_text()), str(path)))
        cfg = json.loads((RULES_DIR / "capture.json").read_text())["selection"]
        sel = select_hypotheses(runs, target=cfg["target"], min_n=cfg["min_fallback"],
                                coverage=cfg["coverage"])
        controls = load_controls()
        require_verified_controls(controls)
        _atomic_write(run_dir / "pinned_hypotheses.json",
                      json.dumps(sel["selected"] + controls, indent=2, default=str))
        manifest.update_stage("select", "complete", n_real=len(sel["selected"]),
                              fallback=sel["fallback"], coverage_met=sel["coverage_met"])
    elif args.stage == "snapshot":
        asyncio.run(_snapshot_stage(run_dir, manifest))
    elif args.stage == "consensus-worklist":
        pinned = json.loads((run_dir / "pinned_hypotheses.json").read_text())
        sources.emit_consensus_worklist(pinned, run_dir / "consensus_worklist.jsonl",
                                        sources.load_query_rules(RULES_DIR))
        manifest.update_stage("consensus-worklist", "stopped_for_connector")
        print(f"STOP: run the Consensus connector over {run_dir / 'consensus_worklist.jsonl'} "
              f"in the interactive session, write raw outputs, then run consensus-ingest.")
    elif args.stage == "consensus-ingest":
        wl = [json.loads(l) for l in (run_dir / "consensus_worklist.jsonl").read_text().splitlines() if l]
        raw = [json.loads(l) for l in Path(args.consensus_raw).read_text().splitlines() if l]
        recs = sources.ingest_consensus(wl, raw, preflight_ok=args.consensus_preflight_ok)
        _atomic_write(run_dir / "consensus_snapshot.json", json.dumps(recs, indent=2))
        manifest.update_stage("consensus-ingest", "complete",
                              states={s: sum(1 for r in recs if r["state"] == s) for s in sources.CALL_STATES})
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
