# Kestrel KG Explorer

## Overview
A React TypeScript frontend chat application for querying the KRAKEN biomedical knowledge graph via the Kestrel API, through an AI-powered conversational interface. The frontend connects to an external FastAPI + Claude Agent SDK backend via WebSocket. Kestrel is the API for the KRAKEN knowledge graph.

## Architecture
- **Frontend**: React 18, TypeScript, Vite, Tailwind CSS, shadcn/ui
- **Backend**: External (FastAPI on AWS Lightsail, not part of this Replit project)
- **Transport**: WebSocket connection to external backend via `VITE_WS_URL`
- **No database** needed — sessions are ephemeral

```
Browser (React on Replit) <--WebSocket--> External Backend (Lightsail)
```

## Project Structure
```
client/src/
├── App.tsx                      # Main app with routing
├── pages/
│   └── chat.tsx                 # Main chat page composing all components
├── components/
│   ├── Header.tsx               # App title, connection status, session stats, clear chat
│   ├── ConnectionStatus.tsx     # Green/red/yellow dot indicator + demo mode
│   ├── ChatArea.tsx             # Scrollable message list, empty state, trace bars
│   ├── ChatInput.tsx            # Text input with auto-resize, send button
│   ├── MessageBubble.tsx        # User message bubble (right-aligned)
│   ├── AgentMessage.tsx         # Agent text with Markdown rendering
│   ├── ToolCallCard.tsx         # Expandable tool call visualization with latency badge
│   ├── ToolResultDisplay.tsx    # Smart formatting per Kestrel tool type
│   ├── TraceBar.tsx             # Per-turn trace metrics (duration, tokens, cost)
│   ├── SessionStatsBar.tsx      # Cumulative session stats pill in header
│   ├── ErrorCard.tsx            # Error display card
│   └── StarterChips.tsx         # Suggestion query chips
├── hooks/
│   └── useWebSocket.ts          # WebSocket connection, reconnect, message handling
├── types/
│   └── messages.ts              # TypeScript types for WebSocket protocol
└── utils/
    ├── formatters.ts            # CURIE, predicate, category display helpers
    └── toolResultParsers.ts     # Parse and summarize Kestrel API tool results
```

## Environment Variables
- `VITE_WS_URL` — WebSocket URL for the backend (default: `ws://localhost:8000/ws/chat`)

## Key Features
- Real-time WebSocket streaming with auto-reconnect (exponential backoff)
- Markdown rendering (tables, code blocks, lists) via react-markdown + remark-gfm
- Interactive tool call cards with expand/collapse, args display, smart result formatting
- Tool-specific result rendering (search results, graph relationships, node details)
- Starter query suggestion chips for common biomedical queries
- Connection status indicator (connected/disconnected/reconnecting)
- Sequential text message grouping
- Demo mode: Simulates full conversation scenario with tool calls and trace messages when backend unavailable
- Agent Trace & Debug Display: Per-turn trace bar (duration, tokens, cost, model) and cumulative session stats in header
- Tool call latency badges showing client-side timing between tool_use and tool_result

## Recent Changes
- 2026-02-06: Added Agent Trace & Debug Display (TraceBar, SessionStatsBar, tool latency badges, demo mode)
- 2026-02-06: Initial implementation of the complete chat frontend

## Design Tokens
- Primary color: Teal/blue (195 hue) — professional research tool aesthetic
- Font: Inter (body), JetBrains Mono (CURIEs, code, technical identifiers)
- Clean, minimal layout with consistent spacing
