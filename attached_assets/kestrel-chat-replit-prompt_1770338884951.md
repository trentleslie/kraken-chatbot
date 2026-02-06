# Replit Agent Prompt: Kestrel Knowledge Graph Chat Interface

## Project Overview

Build a web application that provides a conversational chat interface to a biomedical knowledge graph called KRAKEN, accessed through the Kestrel API. The app uses the Claude Agent SDK (Python) to run an AI agent that can query the knowledge graph on behalf of users using natural language.

Users type questions like "What drugs treat type 2 diabetes?" or "What are the side effects of metformin?" and the agent autonomously calls the appropriate Kestrel API tools (text search, graph traversal, semantic search, etc.) to find answers, then synthesizes the results into a coherent response.

## Tech Stack

- **Backend**: Python 3.12+, FastAPI, `claude-agent-sdk` (Python package)
- **Frontend**: React (TypeScript), Tailwind CSS, shadcn/ui components
- **Transport**: WebSocket for streaming agent responses to the frontend
- **No database needed initially** — sessions are ephemeral (in-memory)

## Architecture

```
Browser (React)  <--WebSocket-->  FastAPI Backend  <--Agent SDK-->  Claude API
                                                    <--MCP/HTTP-->  Kestrel API (remote)
```

The FastAPI backend:
1. Accepts user messages via WebSocket
2. Passes them to a Claude Agent SDK `ClaudeSDKClient` instance
3. Streams back agent messages (text, tool calls, tool results) to the frontend in real-time
4. Maintains conversation state per WebSocket connection

## Environment Variables Required

```
ANTHROPIC_API_KEY=<claude api key>
KESTREL_API_KEY=<kestrel api key, optional>
```

## Backend Implementation Details

### Dependencies (requirements.txt or pyproject.toml)

```
fastapi
uvicorn[standard]
claude-agent-sdk
websockets
```

### Agent Configuration

The agent should be configured with:

```python
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

options = ClaudeAgentOptions(
    model="sonnet",
    system_prompt=SYSTEM_PROMPT,  # See below
    permission_mode="bypassPermissions",
    allowed_tools=[
        "mcp__kestrel__one_hop_query",
        "mcp__kestrel__text_search",
        "mcp__kestrel__vector_search",
        "mcp__kestrel__similar_nodes",
        "mcp__kestrel__hybrid_search",
        "mcp__kestrel__get_nodes",
        "mcp__kestrel__get_edges",
        "mcp__kestrel__canonicalize",
        "mcp__kestrel__get_valid_categories",
        "mcp__kestrel__get_valid_predicates",
        "mcp__kestrel__get_valid_prefixes",
        "mcp__kestrel__health_check",
    ],
    mcp_servers={
        "kestrel": {
            "type": "streamable-http",
            "url": "https://kestrel.nathanpricelab.com/mcp",
        }
    },
)
```

**Important**: The Kestrel MCP server uses `streamable-http` transport (FastMCP default), NOT SSE. The URL is `https://kestrel.nathanpricelab.com/mcp`.

If the Agent SDK requires headers for the MCP connection and Kestrel needs an API key, pass it via:
```python
"headers": {"X-API-Key": os.environ.get("KESTREL_API_KEY", "")}
```

### System Prompt

```python
SYSTEM_PROMPT = """You are a biomedical knowledge graph assistant. You help researchers explore KRAKEN, a comprehensive biomedical knowledge graph built on the Biolink model (v4.2.5) that aggregates vocabularies including CHEBI, HMDB, KEGG, PubChem, UniProt, MONDO, HP, GO, and many more. You access KRAKEN through the Kestrel API tools.

## Critical Insight: Search vs Graph Traversal

Search and graph traversal excel at DIFFERENT query types — they are not interchangeable:

- **Entity Resolution** ("What's the ID for glucose?") → Use `hybrid_search` — **95.2% exact match**
- **Semantic Relations** ("What pathways include glucose?") → Use `one_hop_query` — **99.4% recall**
- **Multi-hop Reasoning** ("Find path from X to Y") → Possible but expect **~15% success** due to sparse connectivity

Using search for relationship queries, or graph traversal for name resolution, will produce poor results.

## Query Strategy Decision Tree

1. User asks "What is X?" or "Find the ID for X" or "Map X to vocabulary Y"
   → `hybrid_search` with the entity name. This is the best method for entity resolution.
   
2. User asks "What relates to X?" or "What [pathways/diseases/genes] involve X?"
   → First resolve the name to a CURIE via `hybrid_search`, then `one_hop_query` with appropriate filters.
   → Always start with `mode="preview"` to gauge result size before pulling `mode="slim"` or `mode="full"`.

3. User asks "What is similar to X?"
   → `similar_nodes` with the CURIE (or `hybrid_search` first to get the CURIE).

4. User asks about an entity they already identified by CURIE
   → `get_nodes` for details, `one_hop_query` for relationships.

5. User asks a vague/conceptual question
   → `vector_search` or `hybrid_search` for fuzzy matching.

## Entity Type → Vocabulary Mapping

When users mention entity types, expect these prefixes in results:

| Entity Type | Common Prefixes | Example CURIEs |
|---|---|---|
| Metabolites/Small Molecules | CHEBI, HMDB, KEGG.COMPOUND, UNII | CHEBI:4167 (glucose), HMDB:HMDB0000122 |
| Diseases | MONDO, HP | MONDO:0004946 (hypoglycemia), MONDO:0005148 (T2DM) |
| Genes | NCBIGene | NCBIGene:2645 (GCK), NCBIGene:6095 (RORA) |
| Proteins | UniProtKB | UniProtKB:D4N3P2 |
| Pathways/Processes | GO, REACT | GO:0006094 (gluconeogenesis), GO:0006006 (glucose metabolic process) |
| Anatomical Locations | UBERON | UBERON:0001088 (urine) |
| Phenotypes | HP | HP:0012531 (Pain) |
| Chemical Roles | CHEBI | CHEBI:78675 (fundamental metabolite) |

## Predicate Reference by Use Case

Use these predicate_filter values for common question types:

**Pathways**: participates_in, has_participant, actively_involved_in
  Example: glucose (CHEBI:4167) → participates_in → gluconeogenesis (GO:0006094)

**Disease associations**: treats, applied_to_treat, in_clinical_trials_for, mentioned_in_clinical_trials_for, associated_with, causes
  Example: glucose → treats → hypoglycemia (MONDO:0004946)

**Gene/Protein interactions**: interacts_with, physically_interacts_with, affects
  Example: cholesterol (CHEBI:16113) → physically_interacts_with → RORA (NCBIGene:6095)

**Chemical relationships**: chemically_similar_to, has_chemical_role, subclass_of, close_match, has_part
  Example: NAD+ → chemically_similar_to → deamido-NAD (CHEBI:14105)

**Process inputs/outputs**: has_input, has_output, produces
  Example: gluconeogenesis (GO:0006094) → has_output → glucose (CHEBI:4167)

**Classification**: subclass_of, has_chemical_role
  Example: glucose → subclass_of → aldohexose

High-frequency predicates by edge count: related_to (862), subclass_of (405), has_participant (336), interacts_with (158), has_part (140), close_match (115), physically_interacts_with (102), has_input (92), in_clinical_trials_for (87), has_chemical_role (70), treats (57).

## Recommended Workflow

For most questions, follow this pattern:
1. **Resolve**: hybrid_search to convert entity name → CURIE(s)
2. **Explore**: one_hop_query (preview mode first) to find relationships
3. **Filter**: Apply predicate_filter and/or end_category_filter to narrow results
4. **Detail**: get_nodes on interesting results for full information
5. **Present**: Summarize findings with entity names, CURIEs, and relationship types

## Known Gotchas

1. **Entity Disambiguation**: "Glycine" matches both the plant genus AND the amino acid. Check neighbor categories/predicates to disambiguate (has_chemical_role = amino acid; in_taxon = plant).

2. **Multi-hop is sparse**: Only ~15% of entity pairs have a connecting path. Don't promise multi-hop results. If one_hop returns nothing, say so rather than chaining hops that will likely fail.

3. **Equivalency edges are rare**: Only ~8 biolink:same_as edges exist. Do NOT rely on graph edges for vocabulary mapping — use hybrid_search instead.

4. **Vector search underperforms on identifiers**: For exact name/identifier lookups, hybrid_search >> vector_search. Vector search is only useful for fuzzy conceptual queries.

5. **Direction matters**: "glucose participates_in gluconeogenesis" needs direction="forward". "gluconeogenesis has_participant glucose" needs direction="reverse". When unsure, use direction="both".

6. **Predicate variations**: The same relationship type may use multiple predicates (e.g., treats, applied_to_treat, in_clinical_trials_for). Query multiple related predicates or omit the filter and post-filter results.

7. **Category filter values use Biolink names**: Use values like "biolink:Disease", "biolink:Drug", "biolink:SmallMolecule", "biolink:Gene" — not colloquial terms.

## Response Style

- Present findings with entity names and CURIEs: "metformin (CHEBI:6801)"
- Format relationships as readable triples: Subject → predicate → Object
- If results are large, summarize key findings and offer to drill deeper
- Be transparent about which tools you called and why
- If search returns no results, suggest alternative strategies (different search method, broader terms, different predicate filters)
- When showing multiple vocabulary mappings, organize by prefix for clarity
"""
```

### WebSocket Handler

The backend should implement a WebSocket endpoint that:

1. Creates a `ClaudeSDKClient` per connection
2. On each user message, calls `client.send()` (or equivalent)
3. Iterates over the response stream and forwards each message to the frontend
4. Messages should be forwarded as JSON with a `type` field:
   - `type: "text"` — agent's text response (stream incrementally if possible)
   - `type: "tool_use"` — agent is calling a tool (include tool name and arguments)
   - `type: "tool_result"` — tool returned results (include the data)
   - `type: "error"` — something went wrong
   - `type: "done"` — agent finished responding

Example message flow to the frontend:
```json
{"type": "tool_use", "tool": "text_search", "args": {"search_text": "metformin", "limit": 5}}
{"type": "tool_result", "tool": "text_search", "data": {"metformin": [{"id": "CHEBI:6801", "name": "metformin", ...}]}}
{"type": "text", "content": "I found metformin (CHEBI:6801). Let me look up what conditions it treats..."}
{"type": "tool_use", "tool": "one_hop_query", "args": {"start_node_ids": "CHEBI:6801", "predicate_filter": "biolink:treats", "mode": "preview"}}
{"type": "tool_result", "tool": "one_hop_query", "data": {"nodes": [...], "edges": [...]}}
{"type": "text", "content": "Metformin is used to treat several conditions including..."}
{"type": "done"}
```

### FastAPI App Structure

```
backend/
├── main.py              # FastAPI app, WebSocket endpoint, CORS
├── agent.py             # Agent configuration, system prompt, session management
├── models.py            # Pydantic models for WebSocket messages
└── config.py            # Environment variable loading
```

The FastAPI app should:
- Serve the React frontend from a static directory (or use separate dev servers in development)
- Have a `/ws/chat` WebSocket endpoint
- Have a `/health` HTTP endpoint
- Configure CORS for development (allow localhost origins)

## Frontend Implementation Details

### UI Layout

The interface should be a **single-page chat application** with:

1. **Header**: App title "Kestrel KG Explorer" with a subtle description
2. **Chat area**: Scrollable message list showing the conversation
3. **Input area**: Text input with send button at the bottom

### Message Display

Each message in the chat should render differently based on type:

- **User messages**: Right-aligned, simple text bubble
- **Agent text**: Left-aligned, rendered as Markdown (support bold, lists, code blocks, tables)
- **Tool calls**: Collapsible card showing:
  - Tool name as a header badge (e.g., "🔍 text_search")
  - Arguments as formatted JSON (collapsed by default)
  - Results as formatted JSON (collapsed by default, with a summary line visible)
  - Visual indicator (spinner while pending, checkmark when done)
- **Errors**: Red-tinted card with error message

### Tool Call Visualization

This is a key differentiator. When the agent calls a Kestrel tool, show it as an inline expandable card:

```
┌─────────────────────────────────────────┐
│ 🔍 text_search                    ✓     │
│ Query: "type 2 diabetes"                 │
│ ▶ 3 results found (click to expand)     │
└─────────────────────────────────────────┘
```

When expanded:
```
┌─────────────────────────────────────────┐
│ 🔍 text_search                    ✓     │
│ Query: "type 2 diabetes"                 │
│ ▼ 3 results:                             │
│   • type 2 diabetes mellitus             │
│     MONDO:0005148 | biolink:Disease      │
│     Score: 45.2                          │
│   • maturity-onset diabetes of the young │
│     MONDO:0018911 | biolink:Disease      │
│     Score: 32.1                          │
│   ...                                    │
└─────────────────────────────────────────┘
```

For `one_hop_query` results, consider a mini relationship visualization:
```
CHEBI:6801 (metformin) ──biolink:treats──► MONDO:0005148 (type 2 diabetes)
```

### Color Scheme & Design

- Clean, professional look suitable for a research tool
- Light background with a subtle blue/teal accent color
- Monospace font for CURIEs and technical identifiers
- Good contrast for readability
- Responsive — should work on desktop and tablet

### React App Structure

```
frontend/
├── src/
│   ├── App.tsx                  # Main app with WebSocket connection
│   ├── components/
│   │   ├── ChatArea.tsx         # Scrollable message list
│   │   ├── ChatInput.tsx        # Text input + send button
│   │   ├── MessageBubble.tsx    # User/agent text messages
│   │   ├── ToolCallCard.tsx     # Expandable tool call display
│   │   └── Header.tsx           # App header
│   ├── hooks/
│   │   └── useWebSocket.ts     # WebSocket connection management
│   ├── types/
│   │   └── messages.ts          # TypeScript types for WS messages
│   └── utils/
│       └── formatters.ts        # Format CURIE, predicate display helpers
```

## Suggested Starter Queries for Testing

Include these as placeholder/suggestion chips in the UI. They are ordered from simple to complex and cover the main query patterns:

**Entity Resolution (hybrid_search)**:
- "What is the CHEBI ID for glucose?"
- "Find all identifiers for vitamin B12"
- "Map metformin to different vocabularies"

**Relationship Exploration (one_hop_query)**:
- "What pathways does glucose participate in?"
- "What diseases is cholesterol associated with?"
- "What genes interact with NAD+?"

**Combined Workflow (search → graph)**:
- "What are the side effects of aspirin?"
- "Find drugs that treat type 2 diabetes"
- "What biological processes involve serotonin?"

**Semantic/Fuzzy (vector_search, similar_nodes)**:
- "What metabolites are similar to cholesterol?"
- "Find entities related to insulin resistance"

## Important Technical Notes

1. **The Claude Agent SDK bundles its own CLI** — no separate Claude Code installation needed. Just `pip install claude-agent-sdk`.

2. **The Agent SDK's `ClaudeSDKClient`** supports multi-turn conversations. Create one client per WebSocket connection. Use `client.send()` for each user message and iterate the response.

3. **If `ClaudeSDKClient` proves difficult to integrate**, fall back to using `query()` with conversation history manually assembled. The trade-off is you lose built-in session management but gain simplicity:
   ```python
   async for message in query(prompt=user_message, options=options):
       # forward to WebSocket
   ```

4. **MCP transport**: The Kestrel MCP server runs FastMCP with `streamable-http` transport. The Agent SDK should support this natively via the `mcp_servers` config. If there are issues, try `"type": "sse"` as a fallback (FastMCP supports both).

5. **Rate limiting**: Each user message may trigger multiple Claude API calls (the agent loop). Consider adding a simple rate limiter (e.g., max 10 messages per minute per connection) to prevent runaway costs.

6. **Error handling**: If the Kestrel API is down, the agent's tool calls will return error dicts. The agent should handle these gracefully in its responses. The frontend should also handle WebSocket disconnections with auto-reconnect.

## Stretch Goals (Don't implement initially, but design for)

- **Session persistence**: Save conversations to a database (PostgreSQL) for later reference
- **Authentication**: Add user login (even just a shared password for internal use)
- **Export**: Allow users to export a conversation as Markdown or JSON
- **Bookmarks**: Let users bookmark interesting entities/relationships found during exploration
- **Graph visualization**: Render one_hop_query results as an interactive node-link diagram (e.g., using D3 or vis.js)
