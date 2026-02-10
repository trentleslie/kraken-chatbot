"""
Temporal Node: Temporal reasoning for longitudinal studies.

This node is CONDITIONAL — it only executes when is_longitudinal is True.
It classifies accumulated findings by their temporal relationship to
disease progression:

- UPSTREAM_CAUSE: Metabolic shifts that precede disease manifestation
  (e.g., beta-oxidation dysfunction, early metabolic changes)

- DOWNSTREAM_CONSEQUENCE: Results of disease process
  (e.g., glycation products, tissue damage markers)

- PARALLEL_EFFECT: Concurrent changes not directly causal
  (e.g., inflammatory markers, compensatory responses)

This analysis is primarily LLM reasoning over accumulated findings
with minimal tool use for occasional pathway validation.
"""

import asyncio
import json
import re
from typing import Any

from ..state import (
    DiscoveryState, TemporalClassification, Finding,
    DiseaseAssociation, PathwayMembership, InferredAssociation, Bridge
)

# Try to import Claude Agent SDK - graceful fallback if not available
try:
    from claude_agent_sdk import query, ClaudeAgentOptions
    from claude_agent_sdk.types import McpSSEServerConfig
    HAS_SDK = True
except ImportError:
    HAS_SDK = False


# Kestrel MCP server configuration
KESTREL_URL = "https://kestrel.nathanpricelab.com/mcp"


TEMPORAL_PROMPT = """You are a temporal biomedical analyst specializing in longitudinal studies.

Given findings from a longitudinal study, classify each by temporal relationship
to disease progression. This is critical for understanding causality.

## Classification Categories

1. **UPSTREAM_CAUSE** — Changes that PRECEDE disease manifestation
   - Early metabolic dysfunction (lipid metabolism, amino acid changes)
   - Insulin resistance precursors
   - Beta-oxidation dysfunction
   - Mitochondrial changes

2. **DOWNSTREAM_CONSEQUENCE** — Changes RESULTING FROM disease process
   - Glycation products (AGEs)
   - Tissue damage markers
   - Inflammatory cascades triggered by disease
   - Organ dysfunction markers

3. **PARALLEL_EFFECT** — Concurrent but not directly causal
   - Compensatory responses
   - Co-occurring lifestyle factors
   - Shared risk factor associations
   - Non-specific inflammatory markers

## Reasoning Guidelines

For each finding, consider:
- Biological mechanism: Does this typically precede or follow the disease?
- Timeline: In T2D, BCAAs rise years before diagnosis (upstream)
- Pathway position: Metabolic precursors vs end products
- Known biology: Insulin resistance → glucose dysregulation → complications

Return ONLY a valid JSON object:
{
  "classifications": [
    {
      "entity": "CHEBI:17234",
      "finding_claim": "Associated with insulin resistance",
      "classification": "upstream_cause",
      "reasoning": "Elevated glucose levels precede T2D diagnosis and contribute to beta-cell dysfunction",
      "confidence": "high"
    }
  ]
}

If no findings to classify, return:
{"classifications": []}
"""


def collect_findings_for_classification(state: DiscoveryState) -> list[dict]:
    """
    Collect all findings from previous nodes for temporal classification.

    Prioritizes:
    1. Disease associations (most relevant for temporal analysis)
    2. Inferred associations from cold-start
    3. Bridge findings from integration
    4. Direct and cold-start findings
    """
    findings_to_classify = []

    # Disease associations (high priority)
    for d in state.get("disease_associations", [])[:10]:
        findings_to_classify.append({
            "entity": d.entity_curie,
            "claim": f"Associated with {d.disease_name} via {d.predicate}",
            "source": "disease_association",
        })

    # Inferred associations (medium priority)
    for i in state.get("inferred_associations", [])[:8]:
        findings_to_classify.append({
            "entity": i.source_entity,
            "claim": f"Inferred connection to {i.target_name}: {i.logic_chain}",
            "source": "cold_start",
        })

    # Bridge findings (medium priority)
    for b in state.get("bridges", [])[:5]:
        if isinstance(b, Bridge):
            findings_to_classify.append({
                "entity": b.entities[0] if b.entities else "unknown",
                "claim": f"Bridge: {b.path_description} — {b.significance}",
                "source": "integration",
            })

    # Direct findings (lower priority, often already covered)
    for f in state.get("direct_findings", [])[:5]:
        if f.tier <= 2:  # Only higher confidence findings
            findings_to_classify.append({
                "entity": f.entity,
                "claim": f.claim,
                "source": f.source or "direct_kg",
            })

    return findings_to_classify


def build_study_context_for_temporal(state: DiscoveryState) -> str:
    """Build study context specifically for temporal analysis."""
    parts = []

    # Longitudinal duration
    duration = state.get("duration_years")
    if duration:
        parts.append(f"Study duration: {duration} years")

    # Query context
    raw_query = state.get("raw_query", "")
    if "t2d" in raw_query.lower() or "diabetes" in raw_query.lower():
        parts.append("Disease focus: Type 2 Diabetes conversion")
    elif "cardiovascular" in raw_query.lower():
        parts.append("Disease focus: Cardiovascular disease")

    # FDR context
    fdr = state.get("fdr_entities", [])
    if fdr:
        parts.append(f"First-degree relative entities: {', '.join(fdr[:5])}")

    return "\n".join(parts) if parts else "Longitudinal study (details not specified)"


def parse_temporal_result(
    result_text: str
) -> tuple[list[TemporalClassification], list[str]]:
    """
    Parse LLM response into temporal classifications.

    Returns:
        tuple of (classifications, errors)
    """
    classifications: list[TemporalClassification] = []
    errors: list[str] = []
    data = None

    # Tier 1: Look for JSON code block
    code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', result_text, re.DOTALL)
    if code_block_match:
        try:
            data = json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    # Tier 2: Look for bare JSON object
    if data is None:
        json_match = re.search(r'\{(?:[^{}]|\{[^{}]*\}|\{(?:[^{}]|\{[^{}]*\})*\})*\}', result_text)
        if json_match:
            try:
                data = json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

    # Tier 3: Try entire response
    if data is None:
        try:
            data = json.loads(result_text.strip())
        except json.JSONDecodeError:
            errors.append("Failed to parse temporal classification JSON response")
            return classifications, errors

    if data is None:
        return classifications, errors

    # Parse classifications
    for c in data.get("classifications", []):
        try:
            classification = c.get("classification", "parallel_effect")
            # Validate classification value
            if classification not in ["upstream_cause", "downstream_consequence", "parallel_effect"]:
                classification = "parallel_effect"

            confidence = c.get("confidence", "moderate")
            if confidence not in ["high", "moderate", "low"]:
                confidence = "moderate"

            classifications.append(TemporalClassification(
                entity=c.get("entity", "unknown"),
                finding_claim=c.get("finding_claim", ""),
                classification=classification,
                reasoning=c.get("reasoning", ""),
                confidence=confidence,
            ))
        except Exception as e:
            errors.append(f"Error parsing temporal classification: {e}")

    return classifications, errors


async def run(state: DiscoveryState) -> dict[str, Any]:
    """
    Classify findings by temporal relationship to disease progression.

    This node ONLY runs for longitudinal studies (is_longitudinal=True).
    It should be checked via conditional routing before invocation.

    Uses primarily LLM reasoning with minimal KG tool use for validation.

    Returns:
        temporal_classifications: Classified findings
        errors: Any errors encountered
    """
    # Double-check longitudinal flag (should be handled by routing, but be safe)
    if not state.get("is_longitudinal", False):
        return {
            "temporal_classifications": [],
            "errors": ["Temporal analysis skipped: not a longitudinal study"],
        }

    # Collect findings for classification
    findings_to_classify = collect_findings_for_classification(state)

    if not findings_to_classify:
        return {
            "temporal_classifications": [],
            "errors": ["No findings available for temporal classification"],
        }

    # Check SDK availability
    if not HAS_SDK:
        return {
            "temporal_classifications": [],
            "errors": ["Claude Agent SDK not available for temporal analysis"],
        }

    # Build study context
    study_context = build_study_context_for_temporal(state)

    # Format findings for the prompt
    findings_text = "\n".join([
        f"- Entity: {f['entity']}\n  Claim: {f['claim']}\n  Source: {f['source']}"
        for f in findings_to_classify
    ])

    # Construct the full prompt
    full_prompt = f"""{TEMPORAL_PROMPT}

## Study Context
{study_context}

## Findings to Classify
{findings_text}

Classify each finding by its temporal relationship to disease progression.
"""

    try:
        # Configure Kestrel MCP server (minimal tool use)
        kestrel_config: McpSSEServerConfig = {
            "type": "sse",
            "url": KESTREL_URL,
        }

        options = ClaudeAgentOptions(
            allowed_tools=["mcp__kestrel__one_hop_query"],  # Minimal - just for validation
            mcp_servers={"kestrel": kestrel_config},
            max_turns=3,  # Lightweight reasoning-focused node
            permission_mode="auto",
        )

        # Execute the query
        result = await query(full_prompt, options=options)
        result_text = result.response if hasattr(result, "response") else str(result)

        # Parse the result
        classifications, parse_errors = parse_temporal_result(result_text)

        return {
            "temporal_classifications": classifications,
            "errors": parse_errors,
        }

    except Exception as e:
        return {
            "temporal_classifications": [],
            "errors": [f"Temporal analysis failed: {str(e)}"],
        }
