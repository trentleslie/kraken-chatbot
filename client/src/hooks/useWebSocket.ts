import { useState, useCallback, useRef, useEffect } from "react";
import type {
  ChatMessage,
  ConnectionStatus,
  IncomingMessage,
  ToolUseMessage,
} from "@/types/messages";

const WS_URL =
  import.meta.env.VITE_WS_URL || "ws://localhost:8000/ws/chat";

const RECONNECT_DELAYS = [1000, 2000, 4000, 8000, 16000, 30000];

let messageIdCounter = 0;
function generateId(): string {
  return `msg-${Date.now()}-${++messageIdCounter}`;
}

export function useWebSocket() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [connectionStatus, setConnectionStatus] =
    useState<ConnectionStatus>("disconnected");
  const [isAgentResponding, setIsAgentResponding] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptRef = useRef(0);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(
    null,
  );
  const mountedRef = useRef(true);

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
          };
          return updated;
        });
        break;

      case "error":
        setMessages((prev) => [
          ...prev,
          {
            id: generateId(),
            type: "error" as const,
            message: data.message,
            timestamp: Date.now(),
          },
        ]);
        setIsAgentResponding(false);
        break;

      case "done":
        setIsAgentResponding(false);
        break;

      case "status":
        break;
    }
  }, []);

  const scheduleReconnect = useCallback(() => {
    const attempt = reconnectAttemptRef.current;
    const delay =
      RECONNECT_DELAYS[Math.min(attempt, RECONNECT_DELAYS.length - 1)];

    setConnectionStatus("reconnecting");
    reconnectAttemptRef.current++;

    reconnectTimeoutRef.current = setTimeout(() => {
      if (mountedRef.current) {
        connectWs();
      }
    }, delay);
  }, []);

  const connectWs = useCallback(() => {
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
        scheduleReconnect();
      };

      ws.onerror = () => {
        if (!mountedRef.current) return;
      };
    } catch {
      setConnectionStatus("disconnected");
      scheduleReconnect();
    }
  }, [handleIncomingMessage, scheduleReconnect]);

  const sendMessage = useCallback((content: string) => {
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
      }),
    );
  }, []);

  const clearMessages = useCallback(() => {
    setMessages([]);
    setIsAgentResponding(false);
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    connectWs();

    return () => {
      mountedRef.current = false;
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connectWs]);

  return {
    messages,
    connectionStatus,
    isAgentResponding,
    sendMessage,
    clearMessages,
  };
}
