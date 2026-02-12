"""
Intake Node: Input parsing and mode detection.

This node processes the raw user query to:
1. Detect query type (retrieval vs discovery)
2. Extract entity mentions (keeping aliases separate)
3. Detect study context (longitudinal, FDR significance, entity types)
4. Parse analytical directives

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

# FDR significance keywords
FDR_SIGNIFICANT_KEYWORDS = [
    "survived fdr",
    "fdr-significant",
    "fdr significant",
    "q < 0.05",
    "q<0.05",
    "fdr corrected",
    "significant after correction",
    "passed fdr",
]

MARGINAL_KEYWORDS = [
    "just above 0.05",
    "nominally significant",
    "trending",
    "marginal",
    "did not survive fdr",
    "failed fdr",
    "suggestive",
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
    - Parenthetical aliases: "hexadecanedioate (C16-DC)" → extracts ONLY primary name
    - Asterisk annotations: "lignoceroylcarnitine (C24)*" → stripped

    NOTE: Parenthetical aliases are NOT extracted here. Use extract_aliases() 
    to get the alias mappings for entity resolution fallback.
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

    # Clean up: strip asterisks and extract ONLY primary names (not aliases)
    cleaned_entities = []
    for e in entities:
        cleaned = e.rstrip("*").strip()
        if cleaned and len(cleaned) > 1:
            # Check for parenthetical alias: "hexadecanedioate (C16-DC)"
            # Extract ONLY the primary name, NOT the alias
            paren_match = re.match(r"^(.+?)\s*\([^)]+\)\s*\*?$", cleaned)
            if paren_match:
                main_name = paren_match.group(1).strip()
                if main_name and len(main_name) > 1:
                    cleaned_entities.append(main_name)
                # Do NOT add the alias - it goes in extract_aliases()
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


def extract_aliases(query: str) -> dict[str, list[str]]:
    """
    Extract entity aliases from parenthetical patterns.

    Handles patterns like:
    - "hexadecanedioate (C16-DC)" → {"hexadecanedioate": ["C16-DC"]}
    - "3-hydroxybutyrate (BHBA)" → {"3-hydroxybutyrate": ["BHBA"]}
    - "entity (alias1, alias2)" → {"entity": ["alias1", "alias2"]}

    Returns:
        Dict mapping primary entity names to their list of aliases.
    """
    aliases: dict[str, list[str]] = {}
    
    # Pattern: "primary_name (ALIAS)" or "primary_name (ALIAS)*"
    # Matches: word/phrase followed by parenthetical content
    pattern = r"([A-Za-z0-9][\w\-\s]+?)\s*\(([^)]+)\)\s*\*?"
    
    for match in re.finditer(pattern, query):
        primary = match.group(1).strip()
        alias_content = match.group(2).strip()
        
        # Skip if primary is empty or too short
        if not primary or len(primary) < 2:
            continue
            
        # Skip common non-alias parentheticals
        skip_patterns = [
            r"^\d+$",  # Just numbers
            r"^[nN]=[0-9]+$",  # Sample sizes
            r"^p[=<>]",  # P-values
            r"^q[=<>]",  # Q-values
            r"^e\.?g\.?",  # Example markers
            r"^i\.?e\.?",  # That is markers
        ]
        if any(re.match(p, alias_content) for p in skip_patterns):
            continue
        
        # Handle multiple aliases separated by comma
        alias_list = [a.strip() for a in alias_content.split(",") if a.strip()]
        
        if alias_list:
            if primary in aliases:
                aliases[primary].extend(alias_list)
            else:
                aliases[primary] = alias_list
    
    # Deduplicate aliases for each primary
    for primary in aliases:
        aliases[primary] = list(dict.fromkeys(aliases[primary]))
    
    return aliases


def extract_fdr_groups(query: str, entities: list[str]) -> tuple[list[str], list[str]]:
    """
    Separate FDR-significant from marginal entities based on context.

    Looks for explicit markers in the query like:
    - "KIF6 and NLGN1 survived FDR correction"
    - "The following are FDR-significant: ..."
    - Section headers indicating significance

    Returns:
        Tuple of (fdr_entities, marginal_entities)
    """
    fdr_entities: list[str] = []
    marginal_entities: list[str] = []
    query_lower = query.lower()
    
    # Check for FDR significance context
    has_fdr_context = any(kw in query_lower for kw in FDR_SIGNIFICANT_KEYWORDS)
    has_marginal_context = any(kw in query_lower for kw in MARGINAL_KEYWORDS)
    
    if not has_fdr_context and not has_marginal_context:
        # No FDR context mentioned - return empty lists
        return fdr_entities, marginal_entities
    
    # Look for explicit FDR-significant entity mentions
    # Pattern: "X and Y survived FDR" or "X, Y are FDR-significant"
    fdr_pattern = r"([A-Za-z0-9,\s]+?)\s+(?:survived|passed|are|were)\s+(?:fdr|FDR)"
    fdr_match = re.search(fdr_pattern, query, re.IGNORECASE)
    if fdr_match:
        mentioned = fdr_match.group(1)
        mentioned_entities = re.split(r",\s*(?:and\s+)?|\s+and\s+", mentioned)
        for e in mentioned_entities:
            e_clean = e.strip()
            if e_clean and any(e_clean.lower() == ent.lower() for ent in entities):
                # Find the original casing from entities list
                for ent in entities:
                    if ent.lower() == e_clean.lower():
                        fdr_entities.append(ent)
                        break
    
    # Look for marginal entity mentions
    marginal_pattern = r"([A-Za-z0-9,\s]+?)\s+(?:did not survive|failed|are|were)\s+(?:fdr|FDR|marginal)"
    marginal_match = re.search(marginal_pattern, query, re.IGNORECASE)
    if marginal_match:
        mentioned = marginal_match.group(1)
        mentioned_entities = re.split(r",\s*(?:and\s+)?|\s+and\s+", mentioned)
        for e in mentioned_entities:
            e_clean = e.strip()
            if e_clean and any(e_clean.lower() == ent.lower() for ent in entities):
                for ent in entities:
                    if ent.lower() == e_clean.lower():
                        marginal_entities.append(ent)
                        break
    
    # Deduplicate
    fdr_entities = list(dict.fromkeys(fdr_entities))
    marginal_entities = list(dict.fromkeys(marginal_entities))
    
    return fdr_entities, marginal_entities


def detect_entity_types(query: str, entities: list[str]) -> dict[str, str]:
    """
    Tag entities with type hints based on context.

    Uses:
    - Section headers: "Significant Metabolites:" → metabolite
    - Gene symbol heuristics: 2-6 uppercase chars → protein/gene
    - Chemical suffixes: -ose, -ate, -ine, -ol → metabolite

    Returns:
        Dict mapping entity names to type hints.
    """
    type_hints: dict[str, str] = {}
    query_lower = query.lower()
    
    # Look for section headers
    metabolite_headers = ["metabolites:", "metabolite:", "significant metabolites:"]
    protein_headers = ["proteins:", "protein:", "significant proteins:"]
    gene_headers = ["genes:", "gene:", "significant genes:"]
    
    # Check which section each entity might be under
    # This is a simplified approach - looks for proximity to headers
    for entity in entities:
        entity_pos = query_lower.find(entity.lower())
        if entity_pos == -1:
            # Try finding with different casing or partial match
            for i, c in enumerate(query_lower):
                if query_lower[i:i+len(entity)].lower() == entity.lower():
                    entity_pos = i
                    break
        
        if entity_pos == -1:
            continue
        
        # Check for headers before this entity
        best_header_type = None
        best_header_pos = -1
        
        for header in metabolite_headers:
            pos = query_lower.rfind(header, 0, entity_pos)
            if pos > best_header_pos:
                best_header_pos = pos
                best_header_type = "metabolite"
        
        for header in protein_headers:
            pos = query_lower.rfind(header, 0, entity_pos)
            if pos > best_header_pos:
                best_header_pos = pos
                best_header_type = "protein"
        
        for header in gene_headers:
            pos = query_lower.rfind(header, 0, entity_pos)
            if pos > best_header_pos:
                best_header_pos = pos
                best_header_type = "gene"
        
        if best_header_type:
            type_hints[entity] = best_header_type
            continue
        
        # Heuristic: Gene symbols are typically 2-6 uppercase characters
        if re.match(r"^[A-Z]{2,6}\d?$", entity):
            type_hints[entity] = "gene"
            continue
        
        # Heuristic: Chemical suffixes
        entity_lower = entity.lower()
        if any(entity_lower.endswith(suffix) for suffix in ["ose", "ate", "ine", "ol", "ide", "yl"]):
            type_hints[entity] = "metabolite"
            continue
    
    return type_hints


def extract_study_context(query: str) -> dict[str, str]:
    """
    Extract structured study metadata from query.

    Returns dict with keys:
    - study_type: "longitudinal", "cross-sectional", "case-control", etc.
    - disease_focus: Primary disease being studied
    - design: Study design details
    - timepoints: Time points mentioned
    - measurement: Measurement types (metabolomics, proteomics)
    - key_insight: Main finding or hypothesis
    """
    context: dict[str, str] = {}
    query_lower = query.lower()
    
    # Detect study type
    if any(kw in query_lower for kw in ["longitudinal", "over time", "follow-up", "baseline"]):
        context["study_type"] = "longitudinal"
    elif "case-control" in query_lower or "case control" in query_lower:
        context["study_type"] = "case-control"
    elif "cross-sectional" in query_lower or "cross sectional" in query_lower:
        context["study_type"] = "cross-sectional"
    elif "cohort" in query_lower:
        context["study_type"] = "cohort"
    
    # Detect disease focus
    disease_patterns = [
        (r"diabetes|t2d|type 2 diabetes|t2dm", "type 2 diabetes"),
        (r"t1d|type 1 diabetes|t1dm", "type 1 diabetes"),
        (r"alzheimer|ad\b", "Alzheimer's disease"),
        (r"parkinson", "Parkinson's disease"),
        (r"cancer|tumor|carcinoma", "cancer"),
        (r"cardiovascular|cvd|heart disease", "cardiovascular disease"),
        (r"obesity|bmi|overweight", "obesity"),
    ]
    for pattern, disease in disease_patterns:
        if re.search(pattern, query_lower):
            context["disease_focus"] = disease
            break
    
    # Detect measurement type
    if "metabolom" in query_lower:
        context["measurement"] = "metabolomics"
    elif "proteom" in query_lower:
        context["measurement"] = "proteomics"
    elif "genom" in query_lower or "gwas" in query_lower:
        context["measurement"] = "genomics"
    elif "transcriptom" in query_lower or "rna" in query_lower:
        context["measurement"] = "transcriptomics"
    
    # Extract time points
    time_match = re.search(r"(\d+)[-\s]*(?:year|yr)s?", query_lower)
    if time_match:
        context["timepoints"] = f"{time_match.group(1)} years"
    
    # Extract study design keywords
    design_keywords = []
    if "ogtt" in query_lower or "oral glucose tolerance" in query_lower:
        design_keywords.append("OGTT challenge")
    if "fdr" in query_lower or "first-degree relative" in query_lower:
        design_keywords.append("first-degree relatives")
    if "converter" in query_lower:
        design_keywords.append("disease converters")
    if "baseline" in query_lower:
        design_keywords.append("baseline measurements")
    if design_keywords:
        context["design"] = ", ".join(design_keywords)
    
    return context


def extract_analytical_directives(query: str) -> list[str]:
    """
    Extract user's specific analysis requests/priorities.

    Looks for patterns like:
    - "Pay special attention to:"
    - "Please focus on..."
    - "I'm particularly interested in..."
    - Numbered instructions after "Please"

    Returns:
        List of directive strings.
    """
    directives: list[str] = []
    
    # Pattern: "Pay special attention to X"
    attention_pattern = r"pay (?:special )?attention to[:\s]+([^.!?]+)"
    for match in re.finditer(attention_pattern, query, re.IGNORECASE):
        directives.append(f"Focus on: {match.group(1).strip()}")
    
    # Pattern: "Please focus on X" or "Please analyze X"
    please_pattern = r"please (?:focus on|analyze|investigate|examine|look at)[:\s]+([^.!?]+)"
    for match in re.finditer(please_pattern, query, re.IGNORECASE):
        directives.append(match.group(1).strip())
    
    # Pattern: "I'm particularly interested in X"
    interest_pattern = r"(?:i'm|i am|we're|we are) (?:particularly |especially )?interested in[:\s]+([^.!?]+)"
    for match in re.finditer(interest_pattern, query, re.IGNORECASE):
        directives.append(f"Interest: {match.group(1).strip()}")
    
    # Pattern: Numbered lists after directive keywords
    numbered_pattern = r"(?:please|should|want to)[^.]*?:\s*(?:\n|^)\s*\d+\.\s*([^\n]+)"
    for match in re.finditer(numbered_pattern, query, re.IGNORECASE | re.MULTILINE):
        directives.append(match.group(1).strip())
    
    # Deduplicate while preserving order
    return list(dict.fromkeys(directives))


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

    # Extract entities from query (aliases NOT included)
    entities = extract_entities(query)

    # Determine query type
    query_type = detect_query_type(query, entities)

    # Detect longitudinal context
    is_longitudinal, duration = detect_longitudinal_context(query)

    # NEW: Extract structured context
    entity_aliases = extract_aliases(query)
    entity_type_hints = detect_entity_types(query, entities)
    study_context = extract_study_context(query)
    fdr_entities, marginal_entities = extract_fdr_groups(query, entities)
    analytical_directives = extract_analytical_directives(query)

    duration_sec = time.time() - start
    logger.info(
        "Completed intake in %.1fs — entities=%d, aliases=%d, type=%s, longitudinal=%s",
        duration_sec, len(entities), len(entity_aliases), query_type, is_longitudinal
    )

    return {
        "query_type": query_type,
        "raw_entities": entities,
        "is_longitudinal": is_longitudinal,
        "duration_years": duration,
        # NEW fields
        "entity_aliases": entity_aliases,
        "entity_type_hints": entity_type_hints,
        "study_context": study_context,
        "fdr_entities": fdr_entities,
        "marginal_entities": marginal_entities,
        "analytical_directives": analytical_directives,
    }
