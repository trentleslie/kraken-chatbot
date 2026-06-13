"""Map a DrugMechDB record to a Kestrel-CURIE gold item via REST resolution.

Pragmatic v1 (deviation from the plan's UniChem/MONDO-SSSOM file crosswalks): rather
than download gigabyte mapping files, resolve DrugMechDB entity *names* to Kestrel
CURIEs with the same category-filtered, degree-preferring resolver used for the
anchors (handles the ortholog-mismatch class, finding #3). Reachability filtering
downstream removes anything that doesn't land in Kestrel. The file-crosswalk remains
the heavier "defensible" alternative if a published mapping is later required.
"""
from __future__ import annotations

from .drugmechdb import DmdbRecord
from .kestrel_rest import KestrelREST

# DrugMechDB node label -> Kestrel biolink category hint for resolution
LABEL_TO_CATEGORY = {
    "Protein": "biolink:Gene",
    "GeneFamily": "biolink:Gene",
    "Drug": None,
    "ChemicalSubstance": None,
    "Disease": "biolink:Disease",
    "Pathway": "biolink:Pathway",
    "BiologicalProcess": "biolink:BiologicalProcess",
    "MolecularActivity": "biolink:MolecularActivity",
    "PhenotypicFeature": "biolink:PhenotypicFeature",
}


async def to_gold_item(rest: KestrelREST, rec: DmdbRecord, stratum: str = "random") -> dict | None:
    """Resolve drug→CHEBI, disease→MONDO, and every interior bridge→Kestrel CURIE.
    Returns a gold-item dict, or None if any endpoint or interior node fails to resolve
    (we require a fully-grounded gold path for a clean recall reference)."""
    drug_curie, _ = await rest.resolve(rec.drug_name, category="biolink:ChemicalEntity")
    if not drug_curie:
        drug_curie, _ = await rest.resolve(rec.drug_name)  # fallback: no category
    disease_curie, _ = await rest.resolve(rec.disease_name, category="biolink:Disease")
    if not (drug_curie and disease_curie):
        return None

    bridge_curies: list[str] = []
    for node in rec.interior:
        cat = LABEL_TO_CATEGORY.get(node.label)
        curie, deg = await rest.resolve(node.name, category=cat)
        if not curie:
            curie, deg = await rest.resolve(node.name)
        if not curie:
            return None  # broken gold path -> skip the whole item
        bridge_curies.append(curie)

    return {
        "trial_id": f"dmdb-{rec.id}",
        "drug": rec.drug_name,
        "start_curie": drug_curie,
        "bridge": " / ".join(n.name for n in rec.interior),
        "gold_bridge_curies": bridge_curies,
        "gold_target_curie": disease_curie,
        "stratum": stratum,
        "hop_length": len(rec.interior) + 1,
        "difficulty": None,  # measured by the baseline/recall arms, not pre-labeled
        "source": "drugmechdb",
        "dmdb_id": rec.id,
    }
