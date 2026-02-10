import { useEffect, useState } from "react";
import { useParams } from "wouter";
import { Header } from "@/components/Header";
import { ChatArea } from "@/components/ChatArea";
import type { ChatMessage, SessionStats, TraceMessage } from "@/types/messages";

interface ConversationData {
  id: string;
  started_at: string;
  total_turns: number;
  total_tokens: number;
  total_cost_usd: number;
  model: string;
  status: string;
  turns: Turn[];
}

interface Turn {
  turn_number: number;
  user_query: string;
  assistant_response: string;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
  duration_ms: number;
  tool_calls_count: number;
  created_at: string;
  model: string;
  tool_calls: ToolCall[];
}

interface ToolCall {
  tool_name: string;
  tool_args: string | Record<string, unknown>;
  tool_result: string | Record<string, unknown>;
  sequence_order: number;
}

function parseJsonField(value: string | Record<string, unknown>): Record<string, unknown> {
  if (typeof value === 'string') {
    try {
      return JSON.parse(value);
    } catch {
      return {};
    }
  }
  return value || {};
}

function transformToMessages(data: ConversationData): ChatMessage[] {
  const messages: ChatMessage[] = [];

  for (const turn of data.turns) {
    const timestamp = new Date(turn.created_at).getTime();

    // User message
    messages.push({
      id: `user-${turn.turn_number}`,
      type: "user",
      content: turn.user_query,
      timestamp,
    });

    // Tool calls (interleaved before response)
    for (const tc of turn.tool_calls) {
      messages.push({
        id: `tool-${turn.turn_number}-${tc.sequence_order}`,
        type: "tool_use",
        tool: tc.tool_name,
        args: parseJsonField(tc.tool_args),
        timestamp,
        status: "complete",
        result: parseJsonField(tc.tool_result),
        resultTimestamp: timestamp,
      });
    }

    // Assistant response
    if (turn.assistant_response) {
      messages.push({
        id: `text-${turn.turn_number}`,
        type: "text",
        content: turn.assistant_response,
        timestamp,
      });
    }

    // Trace message
    messages.push({
      id: `trace-${turn.turn_number}`,
      type: "trace",
      input_tokens: turn.input_tokens,
      output_tokens: turn.output_tokens,
      cost_usd: turn.cost_usd,
      duration_ms: turn.duration_ms,
      tool_calls_count: turn.tool_calls_count,
      model: turn.model,
      timestamp,
    });
  }

  return messages;
}

function computeSessionStats(data: ConversationData): SessionStats {
  const traces: TraceMessage[] = data.turns.map((t, i) => ({
    id: `trace-${i}`,
    type: "trace" as const,
    input_tokens: t.input_tokens,
    output_tokens: t.output_tokens,
    cost_usd: t.cost_usd,
    duration_ms: t.duration_ms,
    tool_calls_count: t.tool_calls_count,
    model: t.model,
    timestamp: new Date(t.created_at).getTime(),
  }));

  return {
    total_input_tokens: data.turns.reduce((sum, t) => sum + (t.input_tokens || 0), 0),
    total_output_tokens: data.turns.reduce((sum, t) => sum + (t.output_tokens || 0), 0),
    total_cost_usd: data.total_cost_usd,
    total_tool_calls: data.turns.reduce((sum, t) => sum + (t.tool_calls_count || 0), 0),
    turn_count: data.total_turns,
    traces,
  };
}

export default function SharedConversation() {
  const params = useParams<{ conversationId: string }>();
  const [data, setData] = useState<ConversationData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchConversation() {
      try {
        const res = await fetch(`/api/conversations/${params.conversationId}`);
        if (!res.ok) {
          if (res.status === 404) {
            setError("Conversation not found");
          } else {
            setError("Failed to load conversation");
          }
          return;
        }
        const json = await res.json();
        setData(json);
      } catch (e) {
        setError("Failed to load conversation");
      } finally {
        setLoading(false);
      }
    }
    fetchConversation();
  }, [params.conversationId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen bg-background">
        <div className="text-muted-foreground">Loading conversation...</div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="flex flex-col items-center justify-center h-screen bg-background gap-4">
        <div className="text-destructive text-lg">{error || "Unknown error"}</div>
        <a href="/" className="text-primary hover:underline">
          Start a new conversation
        </a>
      </div>
    );
  }

  const messages = transformToMessages(data);
  const sessionStats = computeSessionStats(data);

  return (
    <div className="flex flex-col h-screen bg-background">
      <Header
        connectionStatus="disconnected"
        sessionStats={sessionStats}
        onClearChat={() => {}}
        hasMessages={true}
        isReadOnly={true}
        conversationDate={data.started_at}
      />
      <ChatArea
        messages={messages}
        isAgentResponding={false}
        isConnected={false}
        onSelectStarter={() => {}}
      />
      {/* No ChatInput - read-only view */}
      <div className="border-t bg-muted/50 px-4 py-3 text-center text-sm text-muted-foreground">
        This is a shared conversation from {new Date(data.started_at).toLocaleDateString()}
        <span className="mx-2">·</span>
        <a href="/" className="text-primary hover:underline">
          Start a new conversation
        </a>
      </div>
    </div>
  );
}
