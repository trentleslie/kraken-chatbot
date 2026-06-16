"""U3↔U4 glue: LLM labeling adapter for one leg's abstract pool.

The SDK call is a thin wrapper; the JSON parsing and tally are pure and unit-tested.

Reproducibility caveats (verified empirically in this env): the Claude Agent SDK exposes no
`temperature`, AND setting `ClaudeAgentOptions(model=...)` — prefixed or bare — breaks the bundled
CLI subprocess (exit 1), so the model cannot be pinned via the SDK either. The CLI uses its
configured default model. Reproducibility therefore rests solely on snapshotting the per-abstract
labels to disk (scoring is recomputable from frozen labels). See plan U8.
"""

import json
import logging
import re
from typing import Any

from .prompts import LEG_LABELS, leg_label_schema

logger = logging.getLogger(__name__)

LABELING_SYSTEM_PROMPT = (
    "You are a careful biomedical evidence labeler. You read PubMed abstracts and decide, for a "
    "single directed relation, whether each abstract supports it, refutes it, is inconclusive, or "
    "is off-topic. You quote verbatim evidence before labeling and never invent PMIDs or evidence."
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def json_output_instruction() -> str:
    """Appended to the leg prompt: force a single JSON object matching the schema."""
    return (
        "\n\nReturn ONLY a single JSON object (no prose, no markdown fences) of the form:\n"
        '{"labels": [{"pmid": "<pmid>", "evidence": "<quote or none>", "label": "<one of: '
        + ", ".join(LEG_LABELS) + '>"}, ...]}\n'
        "Include exactly one entry per abstract provided."
    )


def parse_label_response(text: str) -> list[dict[str, Any]]:
    """Extract and validate the label list from a model response.

    Tolerant of markdown fences and surrounding prose. Drops entries that lack a string pmid or
    whose label is not in LEG_LABELS. Returns [] on unparseable input (never raises).
    """
    if not text:
        return []
    candidate = text.strip()
    fence = _FENCE_RE.search(candidate)
    if fence:
        candidate = fence.group(1).strip()
    else:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = candidate[start:end + 1]
    try:
        data = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        logger.warning("bridge_grounding labeling: unparseable label response")
        return []
    raw = data.get("labels") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        pmid, label = item.get("pmid"), item.get("label")
        if not isinstance(pmid, str) or label not in LEG_LABELS:
            continue
        out.append({"pmid": pmid, "evidence": item.get("evidence", ""), "label": label})
    return out


def tally_labels(labels: list[dict[str, Any]]) -> dict[str, int]:
    """Count labels into the scoring buckets. off_topic is tallied but excluded from scoring."""
    counts = {"support": 0, "refute": 0, "neither": 0, "off_topic": 0}
    bucket = {
        "supports_leg": "support",
        "refutes_leg": "refute",
        "neither_or_inconclusive": "neither",
        "off_topic": "off_topic",
    }
    for item in labels:
        key = bucket.get(item.get("label", ""))
        if key:
            counts[key] += 1
    return counts


async def label_leg_via_sdk(
    leg_prompt: str, *, model: str, node_name: str = "bridge_grounding"
) -> tuple[list[dict[str, Any]], Any]:
    """Label one leg's abstracts via the Claude Agent SDK. Returns (parsed_labels, usage_record).

    Thin: builds pure-reasoning options (no tools, single turn) and parses the JSON response.
    The schema is advisory (embedded via the prompt); validation happens in parse_label_response.
    """
    from ..graph.sdk_utils import ClaudeAgentOptions, query_with_usage

    _ = leg_label_schema()  # schema documents the contract; embedded via json_output_instruction
    # NOTE: do NOT set options.model — empirically, setting it (prefixed OR bare) breaks the
    # bundled Claude Code CLI subprocess in this environment (exit 1); synthesis.py omits it too.
    # The CLI uses its configured default model; `model` here only labels the usage record.
    options = ClaudeAgentOptions(
        system_prompt=LABELING_SYSTEM_PROMPT,
        allowed_tools=[],
        max_turns=1,
    )
    text, usage = await query_with_usage(
        leg_prompt + json_output_instruction(), options, node_name=node_name, model_name=model)
    return parse_label_response(text), usage
