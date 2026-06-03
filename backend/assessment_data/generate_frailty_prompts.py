"""Generate frailty multiomic WGCNA benchmark prompts for the assessment harness.

Reads the frailty WGCNA modules TSV (docs/data/), selects the top-K analytes by
intramodular connectivity per curated module (excluding clinical-chemistry analytes,
which have no names and no parseable section header), and emits benchmark cases in the
existing `queries.json` schema using the intake node's labeled-section format
(``Proteins:`` / ``Metabolites:``, one analyte per line).

Re-runnable: replaces any existing ``path_type == "multiomic-module"`` entries in
queries.json and preserves curated ``expected_curies`` by reading the optional sidecar
``frailty_expected_curies.json`` (analyte display-name -> CURIE). All other entries are
left untouched.

Usage (stdlib only):
    python backend/assessment_data/generate_frailty_prompts.py
"""

import csv
import json
import re
from pathlib import Path

PATH_TYPE = "multiomic-module"
TOP_K = 25
CURATED_MODULES = ["Brown", "Blue", "Turquoise", "Black", "Green", "Midnightblue"]

# Repo root = .../<repo>/backend/assessment_data/<this file>
REPO_ROOT = Path(__file__).resolve().parents[2]
MODULES_TSV = REPO_ROOT / "docs/data/Frailty_Multiomic_WGCNA-modules.tsv"
QUERIES_JSON = Path(__file__).resolve().parent / "queries.json"
CURIE_SIDECAR = Path(__file__).resolve().parent / "frailty_expected_curies.json"

_NA = {"NA", "", "None", "nan", "NaN"}


def _connectivity(row: dict) -> float:
    try:
        return float(row["IntramodularConnectivity"])
    except (ValueError, KeyError, TypeError):
        return 0.0


def normalize_metabolite_name(name: str) -> str:
    """Make a Metabolon name safe for the intake comma/newline split + cleaner to resolve.

    - Strip trailing Metabolon confidence flags (``*``, ``**``).
    - Replace commas *inside* the name (e.g. lipid shorthand ``(d18:2/23:0, d18:1/23:1)``)
      with ``;`` so the parser's ``re.split(r",\\s*|\\n+")`` does not fragment one analyte
      into several. (intake's extract_entities strips the parenthetical to a primary name
      for resolution; we only need it to survive the split as one token here.)
    """
    n = name.strip().rstrip("*").strip()
    n = n.replace(",", ";")
    return n


def select_analytes(rows: list[dict], module: str) -> tuple[list[str], list[str]]:
    """Return (protein GeneSymbols, normalized metabolite names) for a module's top-K
    by connectivity, excluding Chemistry and unnamed analytes."""
    members = [r for r in rows if r["ModuleID"] == module and r["Dataset"] in ("Protein", "Metabolite")]
    members.sort(key=_connectivity, reverse=True)
    top = members[:TOP_K]
    proteins, metabolites = [], []
    for r in top:
        if r["Dataset"] == "Protein":
            gs = r["GeneSymbol"].strip()
            if gs not in _NA:
                proteins.append(gs)
        else:
            cn = r["ChemName"].strip()
            if cn not in _NA:
                metabolites.append(normalize_metabolite_name(cn))
    return proteins, metabolites


def build_prompt(module: str, proteins: list[str], metabolites: list[str]) -> str:
    """Frailty-framed preamble + labeled sections in the intake Priority-1 format."""
    lines = [
        f'The following analytes are co-expressed in the "{module}" module from a '
        f"multiomic WGCNA analysis of frailty. Analyze their collective biological "
        f"significance — the mechanisms they implicate and any actionable hypotheses.",
        "",
    ]
    if proteins:
        lines.append("Proteins:")
        lines.extend(proteins)
        lines.append("")
    if metabolites:
        lines.append("Metabolites:")
        lines.extend(metabolites)
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    rows = list(csv.DictReader(MODULES_TSV.open(), delimiter="\t"))
    curated = json.loads(CURIE_SIDECAR.read_text()) if CURIE_SIDECAR.exists() else {}

    new_entries = []
    for module in CURATED_MODULES:
        proteins, metabolites = select_analytes(rows, module)
        prompt = build_prompt(module, proteins, metabolites)
        # expected_curies: curated {analyte: CURIE} subset keyed by module in the sidecar.
        module_curated = curated.get(module, {})
        expected = sorted(set(module_curated.values())) if isinstance(module_curated, dict) else list(module_curated)
        new_entries.append({
            "query": prompt,
            "path_type": PATH_TYPE,
            "expected_curies": expected,
            "notes": (
                f"Frailty WGCNA module '{module}', top-{TOP_K} by intramodular connectivity "
                f"({len(proteins)} proteins, {len(metabolites)} metabolites; Chemistry excluded)."
            ),
        })

    existing = json.loads(QUERIES_JSON.read_text()) if QUERIES_JSON.exists() else []
    kept = [e for e in existing if e.get("path_type") != PATH_TYPE]
    QUERIES_JSON.write_text(json.dumps(kept + new_entries, indent=2) + "\n")

    print(f"Wrote {len(new_entries)} '{PATH_TYPE}' entries (kept {len(kept)} existing) -> {QUERIES_JSON}")
    for e in new_entries:
        m = re.search(r'"(\w+)" module', e["query"])
        print(f"  {m.group(1) if m else '?':<14} {e['notes']}")


if __name__ == "__main__":
    main()
