"""Parse DrugMechDB indication_paths.yaml into drug→[interior bridges]→disease records.

DrugMechDB (SuLab, CC0). Each record is a NetworkX node-link graph: ordered `links`
walk drug→…→disease; `nodes` carry {id (CURIE), label (type), name}. drug = links[0].source,
disease = links[-1].target, interior bridges = the nodes strictly between.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import yaml

# Pinned to a frozen commit (NOT `main`) for reproducibility — the gold-set sample
# is only stable if the source records are identical across rebuilds. Keep this SHA
# in sync with CONFIG.drugmechdb_commit_sha.
DMDB_COMMIT = "aef224217071216748740c10faeb6db8e3f15901"
DMDB_RAW = f"https://raw.githubusercontent.com/SuLab/DrugMechDB/{DMDB_COMMIT}/indication_paths.yaml"
# Cache key includes the SHA so re-pinning invalidates a stale download.
CACHE = Path(os.getenv("DMDB_CACHE", f"/tmp/drugmechdb_indication_paths_{DMDB_COMMIT[:12]}.yaml"))


def fetch_yaml(force: bool = False) -> Path:
    if force or not CACHE.exists():
        CACHE.write_text(httpx.get(DMDB_RAW, timeout=120, follow_redirects=True).text)
    return CACHE


def load_records() -> list[dict]:
    return yaml.safe_load(fetch_yaml().read_text())


@dataclass
class DmdbInterior:
    id: str
    name: str
    label: str  # Protein / GeneFamily / Pathway / ...


@dataclass
class DmdbRecord:
    id: str
    drug_name: str
    drug_id: str
    disease_name: str
    disease_id: str
    interior: list[DmdbInterior] = field(default_factory=list)


def extract(record: dict) -> DmdbRecord | None:
    """Walk the ordered links into a drug→interior→disease record. Returns None if
    the record is malformed (not a simple ordered chain)."""
    links = record.get("links") or []
    nodes = {n["id"]: n for n in (record.get("nodes") or []) if "id" in n}
    if not links or not nodes:
        return None
    drug_id = links[0].get("source")
    disease_id = links[-1].get("target")
    if not drug_id or not disease_id or drug_id not in nodes or disease_id not in nodes:
        return None
    # interior = the node sequence between drug and disease (targets of all but the last link)
    interior_ids = [lk.get("target") for lk in links[:-1]]
    interior: list[DmdbInterior] = []
    for nid in interior_ids:
        n = nodes.get(nid)
        if not n:
            return None  # broken chain
        interior.append(DmdbInterior(id=nid, name=n.get("name", ""), label=n.get("label", "")))
    if not interior:
        return None  # need at least one bridge
    g = record.get("graph", {})
    return DmdbRecord(
        id=g.get("_id", f"{drug_id}->{disease_id}"),
        drug_name=g.get("drug") or nodes[drug_id].get("name", ""),
        drug_id=drug_id,
        disease_name=g.get("disease") or nodes[disease_id].get("name", ""),
        disease_id=disease_id,
        interior=interior,
    )


def all_records() -> list[DmdbRecord]:
    out = []
    for r in load_records():
        rec = extract(r)
        if rec:
            out.append(rec)
    return out
