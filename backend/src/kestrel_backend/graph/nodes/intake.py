"""
Intake Node: Input parsing and mode detection.

This node processes the raw user query to:
1. Detect query type (retrieval vs discovery)
2. Extract entity mentions
3. Detect study context (longitudinal, FDR, etc.)

No LLM call for v1 - uses heuristic parsing for speed.
"""

import logging
import re
import time
from typing import Any
from ..state import DiscoveryState

logger = logging.getLogger(__name__)


# Trigger phrases that indicate discovery mode
DISCOVERY_TRIGGERS = [
    "what connects",
    "analyze",
    "novel",
    "predict",
    "hypothesis",
    "what's missing",
    "panel",
    "pathway",
    "mechanism",
    "relationship between",
    "how does",
    "why does",
    "explore",
    "investigate",
    "discover",
    "find connections",
    "link between",
    "in common",  # Added for "what do X, Y, Z have in common"
    "related",
    "compare",
]

# Keywords indicating longitudinal study context
LONGITUDINAL_KEYWORDS = [
    "longitudinal",
    "ogtt",
    "oral glucose tolerance",
    "fdr",
    "first-degree relative",
    "converter",
    "baseline",
    "follow-up",
    "progression",
    "over time",
    "time course",
    "temporal",
]


def extract_entities(query: str) -> list[str]:
    """
    Extract entity mentions from query text.

    Handles:
    - Comma-separated lists: "glucose, fructose, mannose"
    - Newline-separated lists
    - Bullet points
    - Numbered lists
    - Natural language mentions
    - Parenthetical synonyms: "hexadecanedioate (C16-DC)" → both extracted
    - Asterisk annotations: "lignoceroylcarnitine (C24)*" → stripped
    """
    entities: list[str] = []

    # First, try to find explicit list patterns
    # Pattern 1: Comma-separated (most common)
    if "," in query:
        # Look for phrases like "analyze: X, Y, Z" or "panel of X, Y, Z"
        list_patterns = [
            r"(?:analyze|panel|metabolites?|genes?|proteins?|entities?)[:\s]+([^.?!]+)",
            r"(?:between|among|connecting)\s+([^.?!]+(?:,\s*and\s+|\s+and\s+|,\s*)[^.?!]+)",
            r"(?:what do|do)\s+([^.?!]+(?:,\s*and\s+|\s+and\s+|,\s*)[^.?!]+)\s+(?:have|share)",  # "What do X, Y, Z have"
        ]
        for pattern in list_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                items = re.split(r",\s*(?:and\s+)?|\s+and\s+", match.group(1))
                entities.extend(item.strip() for item in items if item.strip())
                break

        # If no pattern matched but commas exist, try extracting from comma-separated list
        if not entities:
            # Check if query is primarily a raw comma-separated list (no question words at start)
            query_stripped = query.strip()
            if not re.match(r"^(what|how|why|where|which|who|when|can|does|is|are)\b", query_stripped, re.IGNORECASE):
                # Direct split on commas, handling "and" as well
                items = re.split(r",\s*(?:and\s+)?|\s+and\s+", query_stripped)
                for item in items:
                    cleaned = item.strip().rstrip("*")
                    if cleaned and len(cleaned) > 1:
                        entities.append(cleaned)
            else:
                # Fall back to old pattern for question-like queries
                comma_list_match = re.search(r"([A-Za-z]+(?:,\s*[A-Za-z]+)+(?:,?\s*and\s+[A-Za-z]+)?)", query)
                if comma_list_match:
                    items = re.split(r",\s*(?:and\s+)?|\s+and\s+", comma_list_match.group(1))
                    entities.extend(item.strip() for item in items if item.strip() and len(item.strip()) > 2)

        # If still empty, try splitting the whole query
        if not entities:
            # Remove common words and split
            cleaned = re.sub(r"\b(what|how|does|the|a|an|is|are|these|this|that|do|have|in|common)\b", "", query, flags=re.IGNORECASE)
            parts = re.split(r",\s*(?:and\s+)?|\s+and\s+", cleaned)
            for part in parts:
                # Extract potential entity names (capitalized words, chemical names, gene symbols)
                words = re.findall(r"\b[A-Z][A-Za-z0-9]{2,}\b|\b[A-Za-z]{2,}\d+[A-Za-z]*\b", part)
                entities.extend(words)

    # Pattern 2: Bullet points or numbered lists
    if not entities:
        bullet_match = re.findall(r"(?:[*-]|\d+\.)\s*([^\n*-]+)", query)
        entities.extend(item.strip() for item in bullet_match if item.strip())

    # Pattern 3: Look for quoted entities
    if not entities:
        quoted = re.findall(r"[\"']([^\"']+)[\"']", query)
        entities.extend(quoted)

    # Pattern 4: Extract capitalized terms as potential entities (genes, proteins)
    if not entities:
        # Gene symbols (all caps, 2-6 chars)
        gene_pattern = re.findall(r"\b[A-Z]{2,6}\d?\b", query)
        # Chemical names (capitalized, may have numbers)
        chem_pattern = re.findall(r"\b[A-Z][a-z]+(?:ose|ine|ate|ol|ide)\b", query)
        entities.extend(gene_pattern + chem_pattern)

    # Clean up: strip asterisks and trailing punctuation
    # Also extract parenthetical abbreviations as separate entities
    cleaned_entities = []
    for e in entities:
        cleaned = e.rstrip("*").strip()
        if cleaned and len(cleaned) > 1:
            # Check for parenthetical abbreviation: "hexadecanedioate (C16-DC)"
            paren_match = re.match(r"(.+?)\s*\(([^)]+)\)\s*\*?$", cleaned)
            if paren_match:
                main_name = paren_match.group(1).strip()
                abbreviation = paren_match.group(2).strip()
                if main_name and len(main_name) > 1:
                    cleaned_entities.append(main_name)
                if abbreviation and len(abbreviation) > 1:
                    cleaned_entities.append(abbreviation)
            else:
                cleaned_entities.append(cleaned)

    # Deduplicate while preserving order
    seen = set()
    unique_entities = []
    for e in cleaned_entities:
        e_lower = e.lower()
        if e_lower not in seen:
            seen.add(e_lower)
            unique_entities.append(e)

    return unique_entities


def detect_query_type(query: str, entities: list[str]) -> str:
    """
    Determine if query is retrieval, discovery, or hybrid.

    Discovery queries typically:
    - Ask about relationships between multiple entities
    - Use exploratory language (predict, novel, mechanism)
    - Have 3+ entities to analyze

    Retrieval queries typically:
    - Ask about a single entity
    - Want known facts (what is, tell me about)
    """
    query_lower = query.lower()

    # Check for discovery triggers
    has_discovery_trigger = any(trigger in query_lower for trigger in DISCOVERY_TRIGGERS)

    # Multiple entities suggests discovery
    many_entities = len(entities) >= 3

    # Simple "what is X" questions are retrieval
    is_simple_question = re.match(r"^(what|who|where|when) (is|are|was|were)\b", query_lower)

    if has_discovery_trigger or many_entities:
        if is_simple_question and not many_entities:
            return "hybrid"
        return "discovery"

    return "retrieval"


def detect_longitudinal_context(query: str) -> tuple[bool, int | None]:
    """
    Detect if query involves longitudinal study context.

    Returns:
        Tuple of (is_longitudinal, duration_years)
    """
    query_lower = query.lower()

    is_longitudinal = any(kw in query_lower for kw in LONGITUDINAL_KEYWORDS)

    # Try to extract duration - handle both "5 years" and "5-year"
    duration = None
    duration_match = re.search(r"(\d+)[-\s]*(?:year|yr)s?", query_lower)
    if duration_match:
        duration = int(duration_match.group(1))

    return is_longitudinal, duration


async def run(state: DiscoveryState) -> dict[str, Any]:
    """
    Process input query and extract structured information.

    This is the entry point node for the KRAKEN discovery workflow.
    Returns only the fields this node sets (LangGraph merges with existing state).
    """
    logger.info("Starting intake")
    start = time.time()

    query = state.get("raw_query", "")

    # Extract entities from query
    entities = extract_entities(query)

    # Determine query type
    query_type = detect_query_type(query, entities)

    # Detect longitudinal context
    is_longitudinal, duration = detect_longitudinal_context(query)

    duration_sec = time.time() - start
    logger.info(
        "Completed intake in %.1fs — entities=%d, type=%s, longitudinal=%s",
        duration_sec, len(entities), query_type, is_longitudinal
    )

    return {
        "query_type": query_type,
        "raw_entities": entities,
        "is_longitudinal": is_longitudinal,
        "duration_years": duration,
    }
