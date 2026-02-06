import { Header } from "@/components/Header";
import { ChatArea } from "@/components/ChatArea";
import { ChatInput } from "@/components/ChatInput";
import { useWebSocket } from "@/hooks/useWebSocket";

export default function ChatPage() {
  const {
    messages,
    connectionStatus,
    isAgentResponding,
    sendMessage,
    clearMessages,
  } = useWebSocket();

  const isConnected = connectionStatus === "connected";

  const handleSelectStarter = (query: string) => {
    if (isConnected && !isAgentResponding) {
      sendMessage(query);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-background">
      <Header
        connectionStatus={connectionStatus}
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
        disabled={isAgentResponding}
        isConnected={isConnected}
      />
    </div>
  );
}
