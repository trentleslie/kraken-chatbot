import { AlertCircle } from "lucide-react";
import { Header } from "@/components/Header";
import { ChatArea } from "@/components/ChatArea";
import { ChatInput } from "@/components/ChatInput";
import { useWebSocket } from "@/hooks/useWebSocket";
import type { ErrorMessage } from "@/types/messages";

export default function ChatPage() {
  const {
    messages,
    connectionStatus,
    isAgentResponding,
    sessionStats,
    sendMessage,
    clearMessages,
  } = useWebSocket();

  const isConnected = connectionStatus === "connected" || connectionStatus === "demo";

  // Check for AUTH_ERROR in messages
  const hasAuthError = messages.some(
    (m) => m.type === "error" && (m as ErrorMessage).code === "AUTH_ERROR"
  );

  const handleSelectStarter = (query: string) => {
    if (isConnected && !isAgentResponding) {
      sendMessage(query);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-background">
      {hasAuthError && (
        <div className="bg-destructive text-destructive-foreground px-4 py-3 flex items-center gap-3">
          <AlertCircle className="h-5 w-5 flex-shrink-0" />
          <span className="text-sm font-medium">
            Server authentication has expired. Please contact the administrator to re-authenticate.
          </span>
        </div>
      )}
      <Header
        connectionStatus={connectionStatus}
        sessionStats={sessionStats}
        onClearChat={clearMessages}
        hasMessages={messages.length > 0}
      />
      <ChatArea
        messages={messages}
        isAgentResponding={isAgentResponding}
        isConnected={isConnected}
        onSelectStarter={handleSelectStarter}
      />
      <ChatInput
        onSend={sendMessage}
        disabled={isAgentResponding || hasAuthError}
        isConnected={isConnected}
      />
    </div>
  );
}
