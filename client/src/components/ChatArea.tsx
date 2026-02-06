import { useRef, useEffect } from "react";
import { MessageBubble } from "@/components/MessageBubble";
import { AgentMessage } from "@/components/AgentMessage";
import { ToolCallCard } from "@/components/ToolCallCard";
import { ErrorCard } from "@/components/ErrorCard";
import { StarterChips } from "@/components/StarterChips";
import type { ChatMessage, ToolUseMessage } from "@/types/messages";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Network, Loader2 } from "lucide-react";

interface ChatAreaProps {
  messages: ChatMessage[];
  isAgentResponding: boolean;
  isConnected: boolean;
  onSelectStarter: (query: string) => void;
}

export function ChatArea({
  messages,
  isAgentResponding,
  isConnected,
  onSelectStarter,
}: ChatAreaProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    requestAnimationFrame(() => {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    });
  }, [messages, isAgentResponding]);

  if (messages.length === 0) {
    return (
      <div
        className="flex-1 flex flex-col items-center justify-center p-6 gap-8"
        data-testid="empty-state"
      >
        <div className="text-center space-y-3">
          <div className="flex items-center justify-center w-14 h-14 rounded-md bg-primary/10 mx-auto">
            <Network className="w-7 h-7 text-primary" />
          </div>
          <div className="space-y-1">
            <h2 className="text-lg font-semibold text-foreground">
              Welcome to Kestrel KG Explorer
            </h2>
            <p className="text-sm text-muted-foreground max-w-md">
              Ask questions about the KRAKEN biomedical knowledge graph.
              Search for drugs, diseases, genes, metabolites, and their
              relationships.
            </p>
          </div>
        </div>
        <StarterChips onSelect={onSelectStarter} disabled={!isConnected} />
      </div>
    );
  }

  return (
    <ScrollArea className="flex-1" ref={scrollAreaRef}>
      <div className="max-w-3xl mx-auto px-4 py-6 space-y-4">
        {messages.map((msg) => {
          switch (msg.type) {
            case "user":
              return (
                <MessageBubble key={msg.id} content={msg.content} />
              );
            case "text":
              return (
                <AgentMessage key={msg.id} content={msg.content} />
              );
            case "tool_use":
              return (
                <ToolCallCard
                  key={msg.id}
                  message={msg as ToolUseMessage}
                />
              );
            case "error":
              return <ErrorCard key={msg.id} message={msg.message} />;
            case "done":
              return null;
            default:
              return null;
          }
        })}

        {isAgentResponding &&
          messages.length > 0 &&
          messages[messages.length - 1]?.type !== "text" && (
            <div className="flex items-center gap-2 px-1" data-testid="thinking-indicator">
              <div className="flex gap-1">
                <div className="w-1.5 h-1.5 rounded-full bg-primary animate-bounce" style={{ animationDelay: "0ms" }} />
                <div className="w-1.5 h-1.5 rounded-full bg-primary animate-bounce" style={{ animationDelay: "150ms" }} />
                <div className="w-1.5 h-1.5 rounded-full bg-primary animate-bounce" style={{ animationDelay: "300ms" }} />
              </div>
              <span className="text-xs text-muted-foreground">Thinking...</span>
            </div>
          )}

        <div ref={bottomRef} />
      </div>
    </ScrollArea>
  );
}
