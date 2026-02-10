import { useState, useCallback, useRef, useEffect } from "react";
import type {
  ChatMessage,
  ConnectionStatus,
  IncomingMessage,
  ToolUseMessage,
  TraceMessage,
  SessionStats,
  AgentMode,
  PipelineProgress,
} from "@/types/messages";

const WS_URL = import.meta.env.VITE_WS_URL || "";

const RECONNECT_DELAYS = [1000, 2000, 4000, 8000, 16000, 30000];
const MAX_CONNECT_ATTEMPTS_BEFORE_DEMO = 3;

let messageIdCounter = 0;
function generateId(): string {
  return `msg-${Date.now()}-${++messageIdCounter}`;
}

function emptySessionStats(): SessionStats {
  return {
    total_input_tokens: 0,
    total_output_tokens: 0,
    total_cost_usd: 0,
    total_tool_calls: 0,
    turn_count: 0,
    traces: [],
  };
}

const DEMO_SCENARIO: IncomingMessage[] = [
  {
    type: "tool_use",
    tool: "mcp__kestrel__hybrid_search",
    args: { search_text: "type 2 diabetes treatments", limit: 5 },
  },
  {
    type: "tool_result",
    tool: "mcp__kestrel__hybrid_search",
    data: {
      results: [
        { id: "MONDO:0005148", name: "type 2 diabetes mellitus", category: "biolink:Disease", score: 0.95 },
        { id: "CHEBI:6801", name: "metformin", category: "biolink:SmallMolecule", score: 0.88 },
        { id: "CHEBI:5441", name: "glipizide", category: "biolink:SmallMolecule", score: 0.82 },
        { id: "CHEBI:75209", name: "empagliflozin", category: "biolink:SmallMolecule", score: 0.79 },
      ],
    },
  },
  {
    type: "tool_use",
    tool: "mcp__kestrel__one_hop_query",
    args: { start_node_ids: ["MONDO:0005148"], predicates: ["biolink:treated_by"], limit: 10 },
  },
  {
    type: "tool_result",
    tool: "mcp__kestrel__one_hop_query",
    data: {
      preview: { node_count: 8, edge_count: 7 },
      nodes: [
        { id: "MONDO:0005148", name: "type 2 diabetes mellitus", category: "biolink:Disease" },
        { id: "CHEBI:6801", name: "metformin", category: "biolink:SmallMolecule" },
        { id: "CHEBI:5441", name: "glipizide", category: "biolink:SmallMolecule" },
      ],
      edges: [
        { subject: "MONDO:0005148", predicate: "biolink:treated_by", object: "CHEBI:6801" },
        { subject: "MONDO:0005148", predicate: "biolink:treated_by", object: "CHEBI:5441" },
      ],
    },
  },
  {
    type: "text",
    content: "Based on my search of the KRAKEN knowledge graph, here are the key findings about **type 2 diabetes treatments**:\n\n",
  },
  {
    type: "text",
    content: "### First-Line Treatments\n\n| Drug | ID | Mechanism |\n|------|------|------|\n| Metformin | `CHEBI:6801` | Biguanide — reduces hepatic glucose production |\n| Glipizide | `CHEBI:5441` | Sulfonylurea — stimulates insulin secretion |\n| Empagliflozin | `CHEBI:75209` | SGLT2 inhibitor — increases urinary glucose excretion |\n\n",
  },
  {
    type: "text",
    content: "The graph shows **7 treatment relationships** connecting type 2 diabetes mellitus (`MONDO:0005148`) to various therapeutic agents. Metformin remains the most strongly associated first-line treatment.",
  },
  {
    type: "trace",
    turn_id: "demo-turn-1",
    input_tokens: 2847,
    output_tokens: 531,
    cache_creation_tokens: 0,
    cache_read_tokens: 1200,
    cost_usd: 0.012,
    duration_ms: 3400,
    tool_calls_count: 2,
    model: "claude-sonnet-4-5",
  },
  { type: "done" },
];

const DEMO_DELAYS = [400, 1200, 300, 900, 200, 300, 200, 100, 100];

export function useWebSocket() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [connectionStatus, setConnectionStatus] =
    useState<ConnectionStatus>("disconnected");
  const [isAgentResponding, setIsAgentResponding] = useState(false);
  const [sessionStats, setSessionStats] = useState<SessionStats>(emptySessionStats());
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [agentMode, setAgentMode] = useState<AgentMode>("classic");
  const [pipelineProgress, setPipelineProgress] = useState<PipelineProgress | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptRef = useRef(0);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(true);
  const demoModeRef = useRef(false);
  const demoTimeoutsRef = useRef<ReturnType<typeof setTimeout>[]>([]);

  const addTraceToSession = useCallback((trace: TraceMessage) => {
    setSessionStats((prev) => ({
      total_input_tokens: prev.total_input_tokens + (trace.input_tokens || 0),
      total_output_tokens: prev.total_output_tokens + (trace.output_tokens || 0),
      total_cost_usd: prev.total_cost_usd + (trace.cost_usd || 0),
      total_tool_calls: prev.total_tool_calls + (trace.tool_calls_count || 0),
      turn_count: prev.turn_count + 1,
      traces: [...prev.traces, trace],
    }));
  }, []);

  const handleIncomingMessage = useCallback((data: IncomingMessage) => {
    switch (data.type) {
      case "text":
        setMessages((prev) => {
          const last = prev[prev.length - 1];
          if (last && last.type === "text") {
            return [
              ...prev.slice(0, -1),
              { ...last, content: last.content + data.content },
            ];
          }
          return [
            ...prev,
            {
              id: generateId(),
              type: "text" as const,
              content: data.content,
              timestamp: Date.now(),
            },
          ];
        });
        break;

      case "tool_use":
        setMessages((prev) => [
          ...prev,
          {
            id: generateId(),
            type: "tool_use" as const,
            tool: data.tool,
            args: data.args,
            timestamp: Date.now(),
            status: "pending" as const,
          },
        ]);
        break;

      case "tool_result":
        setMessages((prev) => {
          const idx = [...prev]
            .reverse()
            .findIndex(
              (m) =>
                m.type === "tool_use" &&
                (m as ToolUseMessage).tool === data.tool &&
                (m as ToolUseMessage).status === "pending",
            );
          if (idx === -1) return prev;
          const actualIdx = prev.length - 1 - idx;
          const updated = [...prev];
          const existing = updated[actualIdx] as ToolUseMessage;
          updated[actualIdx] = {
            ...existing,
            status: "complete" as const,
            result: data.data,
            resultTimestamp: Date.now(),
          };
          return updated;
        });
        break;

      case "trace": {
        const traceMsg: TraceMessage = {
          id: generateId(),
          type: "trace" as const,
          turn_id: data.turn_id,
          input_tokens: data.input_tokens,
          output_tokens: data.output_tokens,
          cache_creation_tokens: data.cache_creation_tokens,
          cache_read_tokens: data.cache_read_tokens,
          cost_usd: data.cost_usd,
          duration_ms: data.duration_ms,
          tool_calls_count: data.tool_calls_count,
          model: data.model,
          timestamp: Date.now(),
        };
        setMessages((prev) => [...prev, traceMsg]);
        addTraceToSession(traceMsg);
        break;
      }

      case "error":
        setMessages((prev) => [
          ...prev,
          {
            id: generateId(),
            type: "error" as const,
            message: data.message,
            code: data.code,
            timestamp: Date.now(),
          },
        ]);
        setIsAgentResponding(false);
        setPipelineProgress(null);
        break;

      case "done":
        setIsAgentResponding(false);
        setPipelineProgress(null);
        break;

      case "status":
        break;

      case "conversation_started":
        setConversationId(data.conversation_id);
        break;

      case "pipeline_progress":
        setPipelineProgress({
          node: data.node,
          message: data.message,
          nodesCompleted: data.nodes_completed,
          totalNodes: data.total_nodes,
        });
        break;

      case "pipeline_complete":
        setPipelineProgress(null);
        // Add the synthesis report as a text message
        setMessages((prev) => [
          ...prev,
          {
            id: generateId(),
            type: "pipeline_complete" as const,
            synthesis_report: data.synthesis_report,
            hypotheses_count: data.hypotheses_count,
            entities_resolved: data.entities_resolved,
            duration_ms: data.duration_ms,
            timestamp: Date.now(),
          },
        ]);
        break;
    }
  }, [addTraceToSession]);

  const enterDemoMode = useCallback(() => {
    demoModeRef.current = true;
    setConnectionStatus("demo");
  }, []);

  const scheduleReconnect = useCallback(() => {
    const attempt = reconnectAttemptRef.current;

    if (!WS_URL || attempt >= MAX_CONNECT_ATTEMPTS_BEFORE_DEMO) {
      enterDemoMode();
      return;
    }

    const delay =
      RECONNECT_DELAYS[Math.min(attempt, RECONNECT_DELAYS.length - 1)];

    setConnectionStatus("reconnecting");
    reconnectAttemptRef.current++;

    reconnectTimeoutRef.current = setTimeout(() => {
      if (mountedRef.current) {
        connectWs();
      }
    }, delay);
  }, [enterDemoMode]);

  const connectWs = useCallback(() => {
    if (!WS_URL) {
      enterDemoMode();
      return;
    }

    if (
      wsRef.current?.readyState === WebSocket.OPEN ||
      wsRef.current?.readyState === WebSocket.CONNECTING
    ) {
      return;
    }

    setConnectionStatus("connecting");

    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!mountedRef.current) return;
        setConnectionStatus("connected");
        reconnectAttemptRef.current = 0;
        demoModeRef.current = false;
      };

      ws.onmessage = (event) => {
        if (!mountedRef.current) return;
        try {
          const data: IncomingMessage = JSON.parse(event.data);
          handleIncomingMessage(data);
        } catch (e) {
          console.error("Failed to parse WebSocket message:", e);
        }
      };

      ws.onclose = () => {
        if (!mountedRef.current) return;
        wsRef.current = null;
        setIsAgentResponding(false);
        setPipelineProgress(null);
        scheduleReconnect();
      };

      ws.onerror = () => {
        if (!mountedRef.current) return;
      };
    } catch {
      setConnectionStatus("disconnected");
      scheduleReconnect();
    }
  }, [handleIncomingMessage, scheduleReconnect, enterDemoMode]);

  const runDemoScenario = useCallback(
    (userContent: string) => {
      setIsAgentResponding(true);

      const userMessage: ChatMessage = {
        id: generateId(),
        type: "user",
        content: userContent,
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, userMessage]);

      let cumulative = 300;
      DEMO_SCENARIO.forEach((msg, i) => {
        cumulative += DEMO_DELAYS[i] || 200;
        const timeout = setTimeout(() => {
          if (mountedRef.current) {
            handleIncomingMessage(msg);
          }
        }, cumulative);
        demoTimeoutsRef.current.push(timeout);
      });
    },
    [handleIncomingMessage],
  );

  const sendMessage = useCallback(
    (content: string) => {
      if (demoModeRef.current) {
        runDemoScenario(content);
        return;
      }

      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

      const userMessage: ChatMessage = {
        id: generateId(),
        type: "user",
        content,
        timestamp: Date.now(),
      };

      setMessages((prev) => [...prev, userMessage]);
      setIsAgentResponding(true);

      wsRef.current.send(
        JSON.stringify({
          type: "user_message",
          content,
          agent_mode: agentMode,
        }),
      );
    },
    [runDemoScenario, agentMode],
  );

  const clearMessages = useCallback(() => {
    setMessages([]);
    setIsAgentResponding(false);
    setSessionStats(emptySessionStats());
    setConversationId(null);
    setPipelineProgress(null);
    for (const t of demoTimeoutsRef.current) clearTimeout(t);
    demoTimeoutsRef.current = [];
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    connectWs();

    return () => {
      mountedRef.current = false;
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      for (const t of demoTimeoutsRef.current) clearTimeout(t);
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connectWs]);

  return {
    messages,
    connectionStatus,
    isAgentResponding,
    sessionStats,
    conversationId,
    agentMode,
    setAgentMode,
    pipelineProgress,
    sendMessage,
    clearMessages,
  };
}
