"""U3 — per-leg labeling prompt + strict output schema for the bridge_grounding scorer.

Provenance: adapts the mediated-chain composition framing and the schema-property-order-as-
forced-CoT pattern (emit the `evidence` quote BEFORE the `label`) from stewart-lab/skimgpt,
which declares MIT (see plan open item O1). The prompt text here is a clean-room
reimplementation, not copied verbatim.

A leg is one directed hop of a 3-node chain (A→B or B→C). For each abstract in the leg's
co-occurrence pool, the model emits an evidence quote then a label:

  - supports_leg   : evidence the directed relation holds
  - refutes_leg    : evidence against it, OR evidence for the OPPOSITE direction
  - neither_or_inconclusive : on-topic but not decisive (inflates the scoring denominator)
  - off_topic      : not about this relation (reviews of other topics, unrelated co-mention)
"""

from typing import Any

LEG_LABELS: tuple[str, ...] = (
    "supports_leg",
    "refutes_leg",
    "neither_or_inconclusive",
    "off_topic",
)


def leg_label_schema() -> dict[str, Any]:
    """Strict OpenAI-style JSON schema for labeling a batch of abstracts for one leg.

    `evidence` precedes `label` in property order so the model quotes evidence before committing
    to a label (forced chain-of-thought). additionalProperties:false everywhere; all required.
    """
    item = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            # Order matters: pmid, then evidence (the forced CoT), THEN label.
            "pmid": {"type": "string", "description": "PMID of the abstract being labeled"},
            "evidence": {
                "type": "string",
                "description": "Short verbatim quote from the abstract (or 'none') justifying "
                "the label — emitted BEFORE the label (forced chain-of-thought)",
            },
            "label": {"type": "string", "enum": list(LEG_LABELS)},
        },
        "required": ["pmid", "evidence", "label"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {"labels": {"type": "array", "items": item}},
        "required": ["labels"],
    }


def clean_predicate(predicate: str | None) -> str:
    """'biolink:gene_associated_with_condition' -> 'gene associated with condition'."""
    if not predicate:
        return ""
    return predicate.split(":", 1)[-1].replace("_", " ").strip()


def relation_phrase(
    name_x: str, name_y: str, predicate: str | None, forward: bool | None
) -> str:
    """The KG-asserted directed relation for this leg, oriented per the edge direction.

    The path runs name_x → name_y. `forward=True` means the KG edge runs that way; `forward=False`
    means it is stored reversed, so the asserted relation subject/object are swapped. When the
    predicate is unknown (O0 not yet populated), fall back to a direction-agnostic association.
    """
    pred = clean_predicate(predicate)
    if not pred:
        return f'an association between "{name_x}" and "{name_y}"'
    if forward is False:
        subject, obj = name_y, name_x
    else:  # True or None (unknown orientation → assume along the path)
        subject, obj = name_x, name_y
    return f'"{subject}" {pred} "{obj}"'


def _format_abstracts(abstracts: list[dict[str, Any]]) -> str:
    blocks = []
    for a in abstracts:
        pmid = a.get("pmid", "")
        title = a.get("title", "")
        body = a.get("abstract") or a.get("body") or ""
        header = f"[PMID:{pmid}]" + (f" {title}" if title else "")
        blocks.append(f"{header}\n{body}".strip())
    return "\n\n".join(blocks)


def build_leg_prompt(
    name_x: str,
    name_y: str,
    predicate: str | None,
    forward: bool | None,
    abstracts: list[dict[str, Any]],
    direction_known: bool = True,
) -> str:
    """Build the per-leg labeling prompt for one hop of the chain.

    `direction_known=False` (e.g. when per-hop predicate orientation could not be established,
    plan O3 fallback) drops the directional refute rule and tells the model to label
    direction-agnostically, which the scorer treats as lower-confidence.
    """
    relation = relation_phrase(name_x, name_y, predicate, forward)
    directed = direction_known and bool(clean_predicate(predicate))

    rules = [
        "Emit a short verbatim `evidence` quote from the abstract BEFORE the `label` "
        "(if nothing is relevant, set evidence to \"none\").",
        f"supports_leg: the abstract gives evidence that {relation}.",
        "refutes_leg: the abstract gives evidence AGAINST the relation"
        + (", OR evidence for the OPPOSITE direction (a reversed-direction claim is refutes_leg, "
           "not supports_leg)." if directed else "."),
        "neither_or_inconclusive: on-topic but not decisive for or against the relation.",
        "off_topic: not about this specific relation (e.g. a review of an unrelated topic, or "
        "the two terms co-mentioned only incidentally).",
    ]

    guards = [
        "Topical co-mention is NOT support: two terms appearing in the same abstract "
        "(\"both are implicated in X\") is off_topic or neither, not supports_leg.",
        "Direction matters: \"X reduces Y\" is NOT the same claim as \"X causes Y\"; judge against "
        "the exact directed relation above.",
        "Belief / survey / media reports of opinion are NOT evidence (neither_or_inconclusive).",
        "Epidemiology / association-only stated as correlation is NOT mechanism "
        "(neither_or_inconclusive), unless it directly evidences the relation.",
        "Negation / hedging flips meaning: \"no significant association\", \"did not reduce\", "
        "\"unlikely\" → refutes_leg or neither_or_inconclusive, NEVER supports_leg.",
    ]

    return (
        "You are labeling PubMed abstracts as evidence for a single directed biomedical "
        f"relation.\n\nRELATION UNDER TEST: {relation}\n\n"
        "For EACH abstract below, return an object with its pmid, an evidence quote, and one "
        "label from: " + ", ".join(LEG_LABELS) + ".\n\n"
        "LABELING RULES:\n- " + "\n- ".join(rules) + "\n\n"
        "COMMON ERRORS TO AVOID:\n- " + "\n- ".join(guards) + "\n\n"
        "Cite ONLY the PMIDs provided below; do not invent PMIDs or evidence. Return one label "
        "object per abstract.\n\nABSTRACTS:\n" + _format_abstracts(abstracts)
    )
