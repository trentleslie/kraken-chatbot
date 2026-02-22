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
const AUTH_TOKEN = import.meta.env.VITE_AUTH_TOKEN || "";

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

const DEMO_CLASSIC_SCENARIO: IncomingMessage[] = [
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

const DEMO_CLASSIC_DELAYS = [400, 1200, 300, 900, 200, 300, 200, 100, 100];

const DEMO_PIPELINE_SCENARIO: IncomingMessage[] = [
  { type: "pipeline_progress", node: "intake", message: "Parsing your query...", nodes_completed: 1, total_nodes: 9 },
  {
    type: "pipeline_node_detail", node: "intake", summary: "3 entities extracted, query_type=discovery", duration_ms: 120,
    details: {
      entities: ["metformin", "PPARG", "type 2 diabetes"],
      query_type: "discovery",
      is_longitudinal: false,
      duration_years: null,
      aliases_count: 1,
      study_context: {},
      directives: ["Find novel drug-target relationships"],
    },
  },
  { type: "pipeline_progress", node: "entity_resolution", message: "Resolving entities in knowledge graph...", nodes_completed: 2, total_nodes: 9 },
  {
    type: "pipeline_node_detail", node: "entity_resolution", summary: "3/3 entities resolved (2 exact, 1 fuzzy)", duration_ms: 1840,
    details: {
      resolved: 3, failed: 0, total: 3,
      methods: { exact: 2, fuzzy: 1 },
      entities: [
        { raw_name: "metformin", curie: "CHEBI:6801", resolved_name: "metformin", category: "biolink:SmallMolecule", confidence: 0.95, method: "exact" },
        { raw_name: "PPARG", curie: "NCBIGene:5468", resolved_name: "PPARG", category: "biolink:Gene", confidence: 0.95, method: "exact" },
        { raw_name: "type 2 diabetes", curie: "MONDO:0005148", resolved_name: "type 2 diabetes mellitus", category: "biolink:Disease", confidence: 0.80, method: "fuzzy" },
      ],
    },
  },
  { type: "pipeline_progress", node: "triage", message: "Scoring entity novelty...", nodes_completed: 3, total_nodes: 9 },
  {
    type: "pipeline_node_detail", node: "triage", summary: "2 well-characterized, 1 moderate, 0 sparse, 0 cold-start", duration_ms: 650,
    details: {
      well_characterized: 2, moderate: 1, sparse: 0, cold_start: 0,
      entities: [
        { curie: "CHEBI:6801", raw_name: "metformin", edge_count: 847, classification: "well_characterized" },
        { curie: "MONDO:0005148", raw_name: "type 2 diabetes", edge_count: 3210, classification: "well_characterized" },
        { curie: "NCBIGene:5468", raw_name: "PPARG", edge_count: 156, classification: "moderate" },
      ],
      well_characterized_curies: ["CHEBI:6801", "MONDO:0005148"],
      cold_start_curies: [],
    },
  },
  { type: "pipeline_progress", node: "direct_kg", message: "Analyzing well-characterized entities...", nodes_completed: 4, total_nodes: 9 },
  {
    type: "pipeline_node_detail", node: "direct_kg", summary: "12 disease associations, 8 pathways, 24 findings", duration_ms: 3200,
    details: {
      diseases_count: 12, pathways_count: 8, findings_count: 24, hub_flags: [],
      top_diseases: [
        { entity: "CHEBI:6801", disease: "type 2 diabetes mellitus", disease_curie: "MONDO:0005148", predicate: "biolink:treats", evidence: "curated", preset: "established", pmids: 5 },
        { entity: "CHEBI:6801", disease: "polycystic ovary syndrome", disease_curie: "MONDO:0008315", predicate: "biolink:treats", evidence: "curated", preset: "hidden_gems", pmids: 2 },
        { entity: "NCBIGene:5468", disease: "obesity", disease_curie: "MONDO:0011122", predicate: "biolink:gene_associated_with_condition", evidence: "gwas", preset: "established", pmids: 8 },
      ],
      top_pathways: [
        { entity: "CHEBI:6801", pathway: "AMPK signaling pathway", pathway_curie: "GO:0031588", preset: "established" },
        { entity: "NCBIGene:5468", pathway: "adipogenesis", pathway_curie: "GO:0045444", preset: "established" },
      ],
    },
  },
  { type: "pipeline_progress", node: "pathway_enrichment", message: "Finding shared biological pathways...", nodes_completed: 5, total_nodes: 9 },
  {
    type: "pipeline_node_detail", node: "pathway_enrichment", summary: "6 shared neighbors (4 non-hub), 2 biological themes", duration_ms: 2100,
    details: {
      shared_neighbors_count: 6, non_hub_count: 4, themes_count: 2,
      themes: [
        { category: "biolink:BiologicalProcess", members_count: 3, member_names: ["insulin signaling", "glucose homeostasis", "lipid metabolism"], input_coverage: 3 },
        { category: "biolink:Disease", members_count: 2, member_names: ["metabolic syndrome", "insulin resistance"], input_coverage: 2 },
      ],
      top_neighbors: [
        { curie: "GO:0008286", name: "insulin receptor signaling pathway", category: "biolink:BiologicalProcess", degree: 145, connected_inputs: ["CHEBI:6801", "NCBIGene:5468"] },
        { curie: "GO:0006006", name: "glucose metabolic process", category: "biolink:BiologicalProcess", degree: 210, connected_inputs: ["CHEBI:6801", "MONDO:0005148"] },
      ],
    },
  },
  { type: "pipeline_progress", node: "integration", message: "Detecting cross-type bridges...", nodes_completed: 6, total_nodes: 9 },
  {
    type: "pipeline_node_detail", node: "integration", summary: "3 bridges, 1 gap entities", duration_ms: 4500,
    details: {
      bridges_count: 3, gaps_count: 1,
      top_bridges: [
        { path: "metformin \u2192 AMPK \u2192 PPARG \u2192 adipogenesis", entity_names: ["metformin", "AMPK", "PPARG", "adipogenesis"], tier: 2, novelty: "inferred", significance: "Suggests metformin may modulate PPARG activity through AMPK, linking drug mechanism to gene function" },
        { path: "PPARG \u2192 insulin sensitivity \u2192 T2D", entity_names: ["PPARG", "insulin sensitivity", "type 2 diabetes"], tier: 2, novelty: "known", significance: "Well-established pathway connecting PPARG gene variants to diabetes risk" },
      ],
      top_gaps: [
        { name: "GLP-1", reason: "Expected in T2D treatment landscape", interpretation: "Absence may reflect incomplete coverage rather than lack of association", informative: false },
      ],
    },
  },
  { type: "pipeline_progress", node: "synthesis", message: "Generating discovery report...", nodes_completed: 7, total_nodes: 9 },
  {
    type: "pipeline_node_detail", node: "synthesis", summary: "3 hypotheses generated, 4200 char report", duration_ms: 5200,
    details: {
      hypotheses_count: 3, report_length: 4200,
      hypotheses: [
        { title: "Metformin-PPARG Axis in Metabolic Regulation", tier: 1, confidence: "high", claim: "Metformin modulates PPARG transcriptional activity via AMPK phosphorylation, creating a drug-gene synergy for metabolic disease" },
        { title: "Novel PPARG-Mediated Anti-Inflammatory Pathway", tier: 2, confidence: "moderate", claim: "PPARG activation by metformin may reduce inflammation through adiponectin upregulation" },
        { title: "Cross-Indication Potential for PCOS", tier: 3, confidence: "low", claim: "The metformin-PPARG connection suggests repurposing potential for polycystic ovary syndrome" },
      ],
    },
  },
  {
    type: "pipeline_complete",
    synthesis_report: "## Discovery Report: Metformin-PPARG-T2D Network\n\nThis analysis explored the relationships between metformin, PPARG, and type 2 diabetes in the KRAKEN knowledge graph.\n\n### Key Findings\n\n1. **Metformin-PPARG axis**: Evidence suggests metformin modulates PPARG activity through AMPK signaling\n2. **Shared pathways**: Insulin signaling and glucose homeostasis connect all three entities\n3. **Novel connections**: Polycystic ovary syndrome emerged as a hidden gem association\n\n### Hypotheses Generated\n- **H1 (Tier 1)**: Metformin-PPARG synergy in metabolic regulation\n- **H2 (Tier 2)**: Anti-inflammatory pathway via adiponectin\n- **H3 (Tier 3)**: Cross-indication potential for PCOS",
    hypotheses_count: 3,
    entities_resolved: 3,
    duration_ms: 17610,
  },
  { type: "done" },
];

const DEMO_PIPELINE_DELAYS = [300, 400, 300, 600, 300, 400, 300, 800, 300, 600, 300, 700, 300, 500, 300, 800, 200, 100];

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

      case "pipeline_node_detail":
        setMessages((prev) => [
          ...prev,
          {
            id: generateId(),
            type: "pipeline_node_detail" as const,
            node: data.node,
            summary: data.summary,
            duration_ms: data.duration_ms,
            details: data.details,
            timestamp: Date.now(),
          },
        ]);
        break;

      case "pipeline_complete":
        setPipelineProgress(null);
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
      // Append token to URL if available
      const wsUrlWithAuth = AUTH_TOKEN ? `${WS_URL}?token=${AUTH_TOKEN}` : WS_URL;
      const ws = new WebSocket(wsUrlWithAuth);
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

      ws.onclose = (event) => {
        if (!mountedRef.current) return;
        wsRef.current = null;
        setIsAgentResponding(false);
        setPipelineProgress(null);

        // Handle authentication failure (close code 4001)
        if (event.code === 4001) {
          setConnectionStatus("auth_failed");
          setMessages((prev) => [
            ...prev,
            {
              id: generateId(),
              type: "error" as const,
              message: "Authentication failed. Please check your credentials.",
              code: "AUTH_FAILED",
              timestamp: Date.now(),
            },
          ]);
          return;
        }

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

      const isPipeline = agentMode === "pipeline";
      const scenario = isPipeline ? DEMO_PIPELINE_SCENARIO : DEMO_CLASSIC_SCENARIO;
      const delays = isPipeline ? DEMO_PIPELINE_DELAYS : DEMO_CLASSIC_DELAYS;

      let cumulative = 300;
      scenario.forEach((msg: IncomingMessage, i: number) => {
        cumulative += delays[i] || 200;
        const timeout = setTimeout(() => {
          if (mountedRef.current) {
            handleIncomingMessage(msg);
          }
        }, cumulative);
        demoTimeoutsRef.current.push(timeout);
      });
    },
    [handleIncomingMessage, agentMode],
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
