export type ConnectionStatus = "connecting" | "connected" | "disconnected" | "reconnecting";

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

export type ChatMessage =
  | UserMessage
  | AgentTextMessage
  | ToolUseMessage
  | ErrorMessage
  | DoneMessage;

export type IncomingMessage =
  | { type: "text"; content: string }
  | { type: "tool_use"; tool: string; args: Record<string, unknown> }
  | { type: "tool_result"; tool: string; data: Record<string, unknown> }
  | { type: "error"; message: string }
  | { type: "done" }
  | { type: "status"; status: string };
