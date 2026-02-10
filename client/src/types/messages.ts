export type ConnectionStatus = "connecting" | "connected" | "disconnected" | "reconnecting" | "demo";

export type AgentMode = "classic" | "pipeline";

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

export type PipelineCompleteMessage = {
  id: string;
  type: "pipeline_complete";
  synthesis_report: string;
  hypotheses_count: number;
  entities_resolved: number;
  duration_ms: number;
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
  | PipelineCompleteMessage;

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
      type: "pipeline_progress";
      node: string;
      message: string;
      nodes_completed: number;
      total_nodes: number;
    }
  | {
      type: "pipeline_complete";
      synthesis_report: string;
      hypotheses_count: number;
      entities_resolved: number;
      duration_ms: number;
    };
