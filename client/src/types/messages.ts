export type ConnectionStatus = "connecting" | "connected" | "disconnected" | "reconnecting" | "demo" | "auth_failed";

export type AgentMode = "classic" | "pipeline";

// Which biomapper2 API the discovery pipeline's entity resolution targets (prod/dev toggle).
export type BiomapperEnv = "production" | "dev";

// One analyte row from a client-parsed file upload (mirrors backend protocol.StructuredAnalyte).
// The full parsed panel travels with a pipeline user_message alongside the free-text query.
export type StructuredAnalyte = {
  name: string;
  group?: string;
  type?: "metabolite" | "protein" | "gene";
  // Signed within-module weights (Axis A). kME ∈ [-1, 1]; kIM (kWithin) >= 0. Backend authoritative.
  // A number when the mapped cell parsed; a raw string is preserved for a present-but-non-numeric
  // cell so the backend R19 gate REJECTS it rather than the client silently dropping the weight.
  kme?: number | string;
  kim?: number | string;
};

// One per-module eigengene→outcome direction row (Axis A). Optional; the backend validates it.
// eigengene_trait_correlation is a number when the mapped cell parsed, else the raw string so the
// backend rejects a malformed direction instead of it silently vanishing from the module spine.
export type ModuleDirectionInput = {
  group: string;
  eigengene_trait_correlation: number | string;
  trait_label: string;
};

export type UserMessage = {
  id: string;
  type: "user";
  content: string;
  timestamp: number;
};

export type AgentTextMessage = {
  id: string;
  type: "text";
  content: string;
  timestamp: number;
};

export type ToolUseMessage = {
  id: string;
  type: "tool_use";
  tool: string;
  args: Record<string, unknown>;
  timestamp: number;
  status: "pending" | "complete" | "error";
  result?: Record<string, unknown>;
  resultTimestamp?: number;
};

export type ErrorMessage = {
  id: string;
  type: "error";
  message: string;
  code?: string;  // Optional error code (e.g., "AUTH_ERROR", "PIPELINE_ERROR")
  timestamp: number;
};

export type DoneMessage = {
  id: string;
  type: "done";
  timestamp: number;
};

export type StatusMessage = {
  id: string;
  type: "status";
  status: string;
  timestamp: number;
};

export type TraceMessage = {
  id: string;
  type: "trace";
  turn_id?: string;
  trace_id?: string;  // Langfuse trace ID for feedback
  input_tokens?: number;
  output_tokens?: number;
  cache_creation_tokens?: number;
  cache_read_tokens?: number;
  cost_usd?: number;
  duration_ms?: number;
  tool_calls_count?: number;
  model?: string;
  timestamp: number;
};

// Phase 6: Pipeline-specific message types
export type PipelineProgressMessage = {
  id: string;
  type: "pipeline_progress";
  node: string;
  message: string;
  nodes_completed: number;
  total_nodes: number;
  timestamp: number;
};

export type PipelineNodeDetailMessage = {
  id: string;
  type: "pipeline_node_detail";
  node: string;
  summary: string;
  duration_ms: number;
  details: Record<string, unknown>;
  timestamp: number;
};

export type PipelineCompleteMessage = {
  id: string;
  type: "pipeline_complete";
  synthesis_report: string;
  hypotheses_count: number;
  entities_resolved: number;
  duration_ms: number;
  turn_id?: string;   // Database turn ID for feedback
  trace_id?: string;  // Langfuse trace ID for feedback
  timestamp: number;
};

export type SessionStats = {
  total_input_tokens: number;
  total_output_tokens: number;
  total_cost_usd: number;
  total_tool_calls: number;
  turn_count: number;
  traces: TraceMessage[];
};

export type PipelineProgress = {
  node: string;
  message: string;
  nodesCompleted: number;
  totalNodes: number;
};

export type ChatMessage =
  | UserMessage
  | AgentTextMessage
  | ToolUseMessage
  | ErrorMessage
  | DoneMessage
  | TraceMessage
  | PipelineNodeDetailMessage
  | PipelineCompleteMessage;

// BYOK: which key is powering the current turn ("byok" = user's key, "server" = operator key).
export type KeySource = "byok" | "server";

// BYOK: outgoing frame to register or clear the user's API key for this session.
export type SetKeyRequest = {
  type: "set_key";
  key: string | null;
};

export type IncomingMessage =
  | { type: "text"; content: string }
  | { type: "tool_use"; tool: string; args: Record<string, unknown> }
  | { type: "tool_result"; tool: string; data: Record<string, unknown> }
  | { type: "error"; message: string; code?: string }
  | { type: "done" }
  | { type: "status"; status: string }
  | { type: "conversation_started"; conversation_id: string }
  | {
      type: "trace";
      turn_id?: string;
      trace_id?: string;
      input_tokens?: number;
      output_tokens?: number;
      cache_creation_tokens?: number;
      cache_read_tokens?: number;
      cost_usd?: number;
      duration_ms?: number;
      tool_calls_count?: number;
      model?: string;
    }
  | {
      // BYOK: server sends this at the start of each turn to indicate which key is in use.
      type: "key_source";
      source: KeySource;
    }
  | {
      type: "pipeline_progress";
      node: string;
      message: string;
      nodes_completed: number;
      total_nodes: number;
    }
  | {
      type: "pipeline_node_detail";
      node: string;
      summary: string;
      duration_ms: number;
      details: Record<string, unknown>;
    }
  | {
      type: "pipeline_complete";
      synthesis_report: string;
      hypotheses_count: number;
      entities_resolved: number;
      duration_ms: number;
      turn_id?: string;
      trace_id?: string;
    };
