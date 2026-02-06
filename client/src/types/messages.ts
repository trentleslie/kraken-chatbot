export type ConnectionStatus = "connecting" | "connected" | "disconnected" | "reconnecting" | "demo";

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

export type SessionStats = {
  total_input_tokens: number;
  total_output_tokens: number;
  total_cost_usd: number;
  total_tool_calls: number;
  turn_count: number;
  traces: TraceMessage[];
};

export type ChatMessage =
  | UserMessage
  | AgentTextMessage
  | ToolUseMessage
  | ErrorMessage
  | DoneMessage
  | TraceMessage;

export type IncomingMessage =
  | { type: "text"; content: string }
  | { type: "tool_use"; tool: string; args: Record<string, unknown> }
  | { type: "tool_result"; tool: string; data: Record<string, unknown> }
  | { type: "error"; message: string }
  | { type: "done" }
  | { type: "status"; status: string }
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
    };
