"""Claude Agent SDK integration with security hardening for public-facing deployment."""
AGENT_VERSION = "1.5.1"  # Add discovery mode checklist for consistent framework application

import json
import os
import sys
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langfuse import get_client

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    UserMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
    HookMatcher,
)
from claude_agent_sdk.types import McpStdioServerConfig
from .config import get_settings
from .bash_sandbox import bash_security_hook


# Langfuse client (lazy initialized)
_langfuse = None  # Type annotation removed since get_client() return type differs


def _get_langfuse():
    """Get or create Langfuse client for observability tracing.

    SDK v3 reads credentials from environment variables:
    - LANGFUSE_PUBLIC_KEY
    - LANGFUSE_SECRET_KEY
    - LANGFUSE_HOST
    """
    global _langfuse
    if _langfuse is not None:
        return _langfuse
    settings = get_settings()
    if settings.langfuse_enabled and settings.langfuse_public_key and settings.langfuse_secret_key:
        # get_client() reads from LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST env vars
        _langfuse = get_client()
    return _langfuse


# Tools whitelist - these tools are allowed
# Tool names follow MCP naming convention: mcp__<server>__<tool>
ALLOWED_TOOLS = frozenset([
    # Kestrel knowledge graph tools (via stdio proxy)
    "mcp__kestrel__one_hop_query",
    "mcp__kestrel__text_search",
    "mcp__kestrel__vector_search",
    "mcp__kestrel__similar_nodes",
    "mcp__kestrel__hybrid_search",
    "mcp__kestrel__get_nodes",
    "mcp__kestrel__get_edges",
    "mcp__kestrel__get_valid_categories",
    "mcp__kestrel__get_valid_predicates",
    "mcp__kestrel__get_valid_prefixes",
    "mcp__kestrel__health_check",
    # Multi-hop graph reasoning tools
    "mcp__kestrel__guilt_by_association",
    "mcp__kestrel__missing_edge_prediction",
    "mcp__kestrel__pathway_enrichment",
    "mcp__kestrel__novelty_score",
    "mcp__kestrel__gap_analysis",
    # Development/testing tools (sandboxed)
    "Bash",
    "Read",  # Non-destructive file inspection
    # Task tracking
    "TodoWrite",
    "Task",  # Spawn subagents for complex tasks
])


def _get_kestrel_mcp_config() -> McpStdioServerConfig:
    """Get the Kestrel MCP server configuration using stdio proxy."""
    # Get the path to the mcp_proxy module
    backend_dir = Path(__file__).parent.parent.parent
    proxy_module = "src.kestrel_backend.mcp_proxy"

    # Pass environment variables to the proxy
    env = {
        "KESTREL_API_KEY": os.getenv("KESTREL_API_KEY", ""),
        "KESTREL_MCP_URL": os.getenv("KESTREL_MCP_URL", "https://kestrel.nathanpricelab.com/mcp"),
    }

    return McpStdioServerConfig(
        type="stdio",
        command=sys.executable,  # Use the current Python interpreter
        args=["-m", proxy_module],
        env=env,
    )

# Dangerous tools that should be blocked in production
# NOTE: Bash temporarily removed for testing
BLOCKED_TOOLS = frozenset([
    # "Read" removed - now allowed for non-destructive file inspection
    "Write",
    "Edit",
    "Glob",
    "Grep",
    "WebFetch",
    "WebSearch",
    "NotebookEdit",
])

# System prompt defining the agent's explorer role with dual-mode operation
SYSTEM_PROMPT = """You are the KRAKEN Knowledge Graph Explorer, a biomedical research assistant with access to a comprehensive knowledge graph integrating data from CTD, DrugCentral, HMDB, ChEMBL, FAERS, clinical trials, text-mining, and other biomedical sources.

## Operating Modes

You operate in two modes depending on the user's query:

### Mode 1: Knowledge Retrieval (default)
When a user asks a specific question — "What is the CHEBI ID for glucose?", "What genes are associated with Alzheimer's?", "What does metformin treat?" — provide accurate, well-sourced answers directly from the knowledge graph. Cite provenance (sources, PMIDs) where available. This is straightforward retrieval and the answer should be authoritative.

### Mode 2: Discovery & Hypothesis Generation
When a user provides a **list of entities** (analytes, metabolites, proteins, genes) for analysis, or explicitly asks about novel/predicted relationships, switch into discovery mode. In this mode:
- Known associations are the starting point, not the destination
- Always push beyond direct edges to find what the graph *implies* but no one has published yet
- The most valuable insight is a structurally-supported hypothesis that surprises the researcher
- Sparse entities (few KG edges) are the most interesting, not the least — they represent the frontier of what's known
- Lead with novel findings and structural inferences; put established associations in supporting context

**Trigger phrases for discovery mode**: lists of entities, "what connects these", "analyze this panel", "what's novel here", "predict associations", "hypothesis", "what's missing", "what would you expect to see".

**Discovery Mode Checklist — apply ALL of these before presenting findings:**
1. **Novelty score first**: Triage every entity by edge count before any analysis. State the classification (well-characterized / moderate / sparse) and your strategy for each.
2. **Hub bias check**: For every shared neighbor identified during pathway enrichment or guilt-by-association, check its degree. If it has 1000+ neighbors, flag it as a hub and discount it. Prioritize specific shared neighbors (low degree) over generic hubs.
3. **Tier label everything**: Every finding gets a Tier label (1/2/3). No unlabeled claims.
4. **Predicate awareness**: When citing KG edges, name the predicate type and distinguish evidence strength (e.g., GWAS association vs. curated database vs. text-mining).
5. **Validation gap**: For all Tier 3 predictions, include: (a) the structural logic chain that supports the inference, (b) confidence calibration, and (c) at least one concrete validation step (literature search terms, experimental assay, or database cross-reference).
6. **Cross-type bridges**: Actively look for connections that bridge entity types (metabolite → gene → disease → pathway) — these are the highest-value discoveries.

## Biomedical Link Prediction Framework

When exploring the knowledge graph, you are implicitly performing **link prediction** —
inferring relationships that may exist but haven't been directly observed. Understanding
the methodological landscape helps you reason about evidence quality.

### Core Concepts

**The Cold-Start Problem**: Entities with few or no edges in the KG (degree-zero or
near-zero) cannot be reasoned about using topological heuristics like shared neighbors.
When you encounter sparse entities during novelty scoring, recognize this as a cold-start
scenario. These entities require feature-based reasoning (what category are they? what are
their chemical properties? what similar entities DO have connections?) rather than purely
structural inference. Cold-start entities are where novel discoveries live.

**The Sparsity Reality**: Biomedical KGs are ~99% sparse — the absence of an edge between
two entities is almost never evidence of no relationship. It usually means the relationship
hasn't been studied or curated yet. When performing gap analysis or missing edge prediction,
frame absent connections as "unstudied" rather than "nonexistent." This is the Open World
Assumption.

**Guilt-by-Association vs. Semantic Inference**: Direct shared-neighbor analysis (if Gene A
and Gene B share 5 pathway neighbors, they likely interact) is interpretable but limited to
well-connected subgraphs. When you find structural patterns, explain the reasoning chain
explicitly — this is the primary value over black-box embedding methods.

### Evidence Calibration

When generating hypotheses, calibrate confidence against the **biological validation hierarchy**:

- **Tier 1 (Direct KG edges)**: Curated from databases (CTD, DrugCentral, HMDB). High
  confidence but reflects what's already known. Check provenance — edges from text-mining
  alone are weaker than edges from curated databases with PMID support.
- **Tier 2 (Multi-source convergence)**: Multiple independent paths or sources support the
  same relationship. Stronger than any single edge. When pathway enrichment reveals a gene
  connected to 3+ input entities through different predicates, that's convergent evidence.
- **Tier 3 (Structural inference)**: Predicted from graph topology — shared neighbors,
  missing edges, pathway gaps. These are hypotheses, not facts. Always state: what pattern
  supports this inference, how many independent structural signals converge, and what
  experiment or literature search would confirm or refute it.

**The Validation Gap**: Computational predictions with high confidence scores do NOT
guarantee biological reality. Only ~18% of computationally predicted drug-disease links
historically progress to clinical investigation. When presenting Tier 3 findings, always
suggest concrete next steps: literature search terms, experimental assays, or database
cross-references that would elevate confidence.

### Discovery Heuristics

When in discovery mode, apply these principles:

1. **Predicate-aware reasoning**: Not all edges are equal. `biolink:treats` implies clinical
   evidence; `biolink:related_to` is weak. `biolink:gene_associated_with_condition` from
   GWAS is population-level; `biolink:causes` implies mechanistic understanding. Name the
   predicate types when explaining findings.

2. **Hub awareness**: High-degree nodes (e.g., TP53 connects to thousands of entities)
   create spurious shared-neighbor signals. When a "shared connection" is a known hub gene,
   discount it and look for more specific shared neighbors. The Adamic-Adar principle:
   a shared neighbor with few connections is more informative than one with thousands.

3. **Cross-entity-type bridges**: The most interesting findings often bridge entity types —
   a metabolite panel sharing a pathway neighbor that also connects to a disease the
   researcher didn't ask about. Actively look for these cross-type bridges during pathway
   enrichment.

4. **Temporal reasoning**: If the user provides longitudinal data (baseline vs. follow-up,
   converters vs. controls), consider whether graph-predicted associations represent
   upstream causes, downstream consequences, or parallel effects. The graph is static but
   the biology is dynamic.

## Source Attribution (IMPORTANT)

You have two knowledge sources:
1. **Knowledge Graph (KG)**: Retrieved via tools - cite with CURIEs and source databases
2. **Training Knowledge**: Your pre-trained biomedical understanding

When providing context NOT retrieved from the KG, mark it with a compact inline tag:

**[Model Knowledge]** - Use this before statements from your training data

Examples requiring the marker:
- Clinical practice guidelines and standard-of-care statements
- Drug mechanism explanations beyond what's in the KG
- Epidemiological context or prevalence statistics
- Historical developments in the field

**First occurrence**: When you first use [Model Knowledge] in a response, briefly explain: "Note: [Model Knowledge] indicates information from my training data rather than the knowledge graph."

**Subsequent uses**: Just use the inline marker without explanation.

Example:
"Metformin is a first-line treatment for T2D [KG: MONDO:0005148 → treats → CHEBI:6801]. **[Model Knowledge]** Current ADA guidelines recommend it alongside lifestyle modifications, with GLP-1 agonists increasingly preferred for patients with cardiovascular risk."

## Evidence Quality Tiers

When reporting findings, always label by evidence tier:
- **Tier 1 (Direct)**: Entity has a direct KG edge with publication provenance (PMIDs, DOIs)
- **Tier 2 (Multi-source)**: Entity appears across multiple KG sources but may lack specific publications
- **Tier 3 (Structural inference)**: No direct edge exists; association inferred from graph topology via guilt-by-association or missing-edge-prediction techniques

In retrieval mode, you'll mostly report Tier 1-2 evidence. In discovery mode, Tier 3 inferences are the primary value — but never present them as established facts. Use language like "graph structure suggests" or "structurally predicted association."

## Available Tools

### Search Tools
- **hybrid_search**: Combined text + vector search. Best first step for finding entities by name.
- **text_search**: Search by exact text in names, synonyms, descriptions.
- **vector_search**: Semantic search using embeddings — finds conceptually similar entities.

### Graph Navigation
- **one_hop_query**: Core graph exploration — finds connected nodes from a starting entity. Supports filtering by direction, predicate, and category. Returns edges with provenance (sources, publications).
- **similar_nodes**: Find entities semantically similar to a given node by CURIE.

### Detail Retrieval
- **get_nodes**: Get full details for entities by CURIE (e.g., MONDO:0005148).
- **get_edges**: Get full edge details including provenance, publications, and sources.

### Metadata
- **get_valid_categories**: List all valid biolink categories (e.g., biolink:Disease, biolink:Gene).
- **get_valid_predicates**: List all valid relationship types (e.g., biolink:treats, biolink:causes).
- **get_valid_prefixes**: List all valid CURIE prefixes with examples.
- **health_check**: Check if the Kestrel API is running.

### Multi-Hop Graph Reasoning Tools (Discovery Mode)
These tools implement advanced graph reasoning for discovery and hypothesis generation:
- **guilt_by_association**: Find structurally similar entities via shared neighbors
- **missing_edge_prediction**: Predict novel associations (2-hop reachable, no direct edge)
- **pathway_enrichment**: Find shared biological themes across entity sets
- **novelty_score**: Triage entities by characterization level (well-characterized/moderate/sparse)
- **gap_analysis**: Find "expected but missing" entities sharing neighbors with input set

### Local Processing
- **Bash**: Run shell commands for data processing. Use for parsing large result sets with python, jq, grep, awk, etc. ALLOWED commands: jq, python, python3, grep, awk, sed, sort, uniq, wc, head, tail, cat, ls, find, echo, printf, date. Network and destructive commands are blocked.

## Analytical Workflow for Entity Lists (Discovery Mode)

When a user provides a list of analytes, metabolites, proteins, genes, or other biological entities for analysis, follow this systematic workflow:

### Step 1: Resolve and Triage
- Use `hybrid_search` to resolve entity names to CURIEs
- Apply the **novelty_score** tool to classify each entity
- State your analytical strategy based on the triage results

### Step 2: Find Shared Themes
- Apply the **pathway_enrichment** tool across the full entity list
- Identify biological themes connecting multiple entities
- Group findings by category and note the type of evidence each represents

### Step 3: Direct Associations for Well-Characterized Entities
- Use `one_hop_query` to retrieve known disease associations, pathway memberships, and literature-supported relationships
- Note publication provenance where available

### Step 4: Structural Inference for Sparse Entities
For entities classified as sparse or moderate:
- Apply **guilt_by_association** to find structurally similar well-characterized entities
- Apply **missing_edge_prediction** with relevant target categories (e.g., biolink:Disease, biolink:BiologicalProcess)
- **Clearly label all inferences as Tier 3 evidence**

### Step 5: Gap Analysis
- Apply the **gap_analysis** tool to identify expected-but-absent entities
- Note whether absences are informative given the study context

### Step 6: Synthesis
Combine all findings into a narrative that:
- **Leads with novel findings** — structural inferences and unexpected connections first
- Groups entities by biological theme
- Presents Tier 1-2 evidence separately from Tier 3 structural predictions
- Proposes 2-3 biological hypotheses explaining the overall pattern
- For each hypothesis: which entities support it, which don't fit, and the mechanistic logic
- Lists expected-but-absent entities that would confirm or refute each hypothesis
- Recommends specific follow-up experiments or measurements
- Highlights the most surprising or potentially impactful novel predictions

## General Guidelines

- You are a read-only knowledge graph explorer. You cannot modify the graph.
- When results are large, use Bash with Python or jq to process and aggregate data rather than trying to reason over hundreds of results in context.
- Always cite knowledge sources (HMDB, CTD, text-mining, etc.) and include PMIDs when available.
- If a query returns no results, try alternative identifiers or synonyms before reporting failure.
- For entity resolution, prefer CHEBI for metabolites, MONDO for diseases, NCBIGene/HGNC for genes, UniProtKB for proteins, and HP for phenotypes.
- When the user provides common names (e.g., "glucose"), always resolve to CURIEs first using hybrid_search.
"""


@dataclass
class TurnMetrics:
    """Metrics collected during a single agent turn."""
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    tool_calls_count: int = 0
    start_time: float = field(default_factory=time.time)
    model: str = ""

    @property
    def duration_ms(self) -> int:
        """Calculate duration in milliseconds."""
        return int((time.time() - self.start_time) * 1000)

    @property
    def cost_usd(self) -> float:
        """Estimate cost based on Claude Sonnet 4.5 pricing."""
        # Sonnet 4.5 pricing: $3/1M input, $15/1M output
        # Cache read: $0.30/1M, Cache creation: $3.75/1M
        input_cost = self.input_tokens * 3.0 / 1_000_000
        output_cost = self.output_tokens * 15.0 / 1_000_000
        cache_read_cost = self.cache_read_tokens * 0.30 / 1_000_000
        cache_create_cost = self.cache_creation_tokens * 3.75 / 1_000_000
        return input_cost + output_cost + cache_read_cost + cache_create_cost


@dataclass
class AgentEvent:
    """Event emitted during agent processing."""
    type: str
    data: dict[str, Any]


async def run_agent_turn(user_message: str) -> AsyncIterator[AgentEvent]:
    """
    Run a single agent turn and yield events for streaming to the client.

    Note: This is a single-turn implementation. Each call to query() is
    independent - the agent does not retain memory of previous messages
    within the session. This is acceptable for exploratory queries where
    each question is self-contained.

    Yields:
        AgentEvent objects for each streaming event (text, tool_use, tool_result, etc.)
    """
    settings = get_settings()
    metrics = TurnMetrics(model=settings.model or "default")

    # Start Langfuse trace for observability
    langfuse = _get_langfuse()
    trace = None
    if langfuse:
        trace = langfuse.start_span(
            name="kraken_agent_turn",
            input={"user_message": user_message},
            metadata={"turn_id": metrics.turn_id, "model": metrics.model},
        )

    # Configure Kestrel MCP server via stdio proxy
    # The proxy handles Kestrel's non-standard MCP-over-HTTP protocol
    kestrel_config = _get_kestrel_mcp_config()

    # Build options with Kestrel MCP server and Bash security hook
    options_kwargs = {
        "allowed_tools": list(ALLOWED_TOOLS),
        "system_prompt": SYSTEM_PROMPT,
        "mcp_servers": {
            "kestrel": kestrel_config,
        },
        # Security: Validate all Bash commands before execution
        "hooks": {
            "PreToolUse": [
                HookMatcher(matcher="Bash", hooks=[bash_security_hook])
            ]
        },
        "max_buffer_size": 10 * 1024 * 1024,  # 10MB buffer for large KG responses
    }
    if settings.model:
        options_kwargs["model"] = settings.model

    options = ClaudeAgentOptions(**options_kwargs)

    # Track tool_use_id -> tool_name mapping for matching results
    tool_id_to_name: dict[str, str] = {}
    # Track Langfuse tool spans for correlation
    tool_spans: dict[str, Any] = {}

    try:
        async for event in query(
            prompt=user_message,
            options=options,
        ):
            # Handle different message types from the SDK
            if isinstance(event, AssistantMessage):
                # Process content blocks from assistant
                for block in event.content:
                    if isinstance(block, TextBlock):
                        yield AgentEvent(type="text", data={"content": block.text})
                    elif isinstance(block, ToolUseBlock):
                        tool_name = block.name
                        tool_input = block.input
                        tool_id = getattr(block, "id", None)

                        # Security check: verify tool is allowed
                        if tool_name not in ALLOWED_TOOLS:
                            # Build context based on tool type for better error messages
                            context = ""
                            if tool_name == "Write" and "file_path" in tool_input:
                                context = f" (tried to write: {tool_input['file_path']})"
                            elif tool_name == "Edit" and "file_path" in tool_input:
                                context = f" (tried to edit: {tool_input['file_path']})"
                            elif tool_name == "WebFetch" and "url" in tool_input:
                                context = f" (tried to fetch: {tool_input['url']})"
                            elif tool_name == "WebSearch" and "query" in tool_input:
                                context = f" (tried to search web: {tool_input['query']})"
                            elif tool_name == "Grep" and "pattern" in tool_input:
                                context = f" (tried to grep: {tool_input['pattern']})"
                            elif tool_name == "Glob" and "pattern" in tool_input:
                                context = f" (tried to find files: {tool_input['pattern']})"

                            yield AgentEvent(
                                type="error",
                                data={"message": f"Tool {tool_name} is not allowed{context}"}
                            )
                            continue

                        # Track the mapping for later result matching
                        if tool_id:
                            tool_id_to_name[tool_id] = tool_name
                            # Create Langfuse child span for tool call
                            if trace:
                                tool_spans[tool_id] = trace.start_span(
                                    name=f"tool:{tool_name}",
                                    input=tool_input,
                                )

                        metrics.tool_calls_count += 1
                        yield AgentEvent(
                            type="tool_use",
                            data={"tool": tool_name, "args": tool_input}
                        )

            elif isinstance(event, UserMessage):
                # Tool results come in UserMessage with ToolResultBlock
                if hasattr(event, "content"):
                    for block in event.content:
                        if isinstance(block, ToolResultBlock):
                            tool_use_id = getattr(block, "tool_use_id", None)
                            content = getattr(block, "content", "")

                            # Look up the tool name from our mapping
                            tool_name = tool_id_to_name.get(tool_use_id, "unknown")

                            # Parse JSON content if it's a string
                            if isinstance(content, str):
                                try:
                                    content = json.loads(content)
                                except json.JSONDecodeError:
                                    # Keep as string wrapped in dict
                                    content = {"raw": content}

                            # Ensure content is a dict
                            if not isinstance(content, dict):
                                content = {"result": content}

                            # End Langfuse span for this tool
                            if tool_use_id and tool_use_id in tool_spans:
                                tool_span = tool_spans[tool_use_id]
                                tool_span.update(output=content)
                                tool_span.end()

                            yield AgentEvent(
                                type="tool_result",
                                data={
                                    "tool": tool_name,
                                    "data": content
                                }
                            )

            # Collect usage metrics from ResultMessage (final event with totals)
            if isinstance(event, ResultMessage):
                if event.usage:
                    usage = event.usage
                    # ResultMessage.usage is a dict, not an object with attributes
                    if isinstance(usage, dict):
                        metrics.input_tokens = usage.get("input_tokens", 0)
                        metrics.output_tokens = usage.get("output_tokens", 0)
                        metrics.cache_creation_tokens = usage.get("cache_creation_input_tokens", 0)
                        metrics.cache_read_tokens = usage.get("cache_read_input_tokens", 0)

            # Also check for usage on other event types (e.g., AssistantMessage)
            elif hasattr(event, "usage") and event.usage:
                usage = event.usage
                # Handle both dict and object-style usage
                if isinstance(usage, dict):
                    metrics.input_tokens += usage.get("input_tokens", 0)
                    metrics.output_tokens += usage.get("output_tokens", 0)
                    metrics.cache_creation_tokens += usage.get("cache_creation_input_tokens", 0)
                    metrics.cache_read_tokens += usage.get("cache_read_input_tokens", 0)
                else:
                    metrics.input_tokens += getattr(usage, "input_tokens", 0)
                    metrics.output_tokens += getattr(usage, "output_tokens", 0)
                    metrics.cache_creation_tokens += getattr(usage, "cache_creation_input_tokens", 0)
                    metrics.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0)

    except Exception as e:
        error_msg = str(e).lower()
        # Detect authentication/authorization errors
        if any(keyword in error_msg for keyword in [
            "unauthorized", "authentication", "auth", "login",
            "credential", "token expired", "not logged in",
            "permission denied", "access denied", "401", "403"
        ]):
            yield AgentEvent(
                type="error",
                data={
                    "message": "Authentication expired. Please contact the administrator to re-authenticate the server.",
                    "code": "AUTH_ERROR"
                }
            )
        else:
            yield AgentEvent(type="error", data={"message": str(e)})

    # Emit trace with metrics
    yield AgentEvent(
        type="trace",
        data={
            "turn_id": metrics.turn_id,
            "input_tokens": metrics.input_tokens,
            "output_tokens": metrics.output_tokens,
            "cache_creation_tokens": metrics.cache_creation_tokens,
            "cache_read_tokens": metrics.cache_read_tokens,
            "cost_usd": metrics.cost_usd,
            "duration_ms": metrics.duration_ms,
            "tool_calls_count": metrics.tool_calls_count,
            "model": metrics.model,
        }
    )

    # Finalize Langfuse trace with collected metrics
    if trace and langfuse:
        trace.update(
            output={"turn_id": metrics.turn_id},
            usage={
                "input": metrics.input_tokens,
                "output": metrics.output_tokens,
                "total": metrics.input_tokens + metrics.output_tokens,
            },
            metadata={
                "cost_usd": metrics.cost_usd,
                "duration_ms": metrics.duration_ms,
                "tool_calls_count": metrics.tool_calls_count,
                "cache_creation_tokens": metrics.cache_creation_tokens,
                "cache_read_tokens": metrics.cache_read_tokens,
            },
        )
        trace.end()  # SDK v3 requires explicit end() for manually created spans
        langfuse.flush()  # Ensure trace is sent before response completes

    # Signal completion
    yield AgentEvent(type="done", data={})
