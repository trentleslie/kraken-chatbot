import { useState, useRef, type ChangeEvent, type KeyboardEvent } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Send } from "lucide-react";

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  isConnected: boolean;
  // True when a structured analyte panel is staged for upload. Enables send with an empty
  // textarea (R17 file-only submit).
  hasPanel?: boolean;
  // Reports whether the textarea currently holds a non-whitespace query. Lets the composer show
  // the file-only degradation warning (R17) when a panel is staged with no typed context.
  onQueryEmptyChange?: (empty: boolean) => void;
}

export function ChatInput({
  onSend,
  disabled,
  isConnected,
  hasPanel = false,
  onQueryEmptyChange,
}: ChatInputProps) {
  const [input, setInput] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = () => {
    const trimmed = input.trim();
    // R17: allow a file-only submit (panel staged, no typed query).
    if ((!trimmed && !hasPanel) || disabled || !isConnected) return;
    onSend(trimmed);
    setInput("");
    onQueryEmptyChange?.(true);
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    onQueryEmptyChange?.(e.target.value.trim().length === 0);
    const el = e.target;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const canSend = (input.trim().length > 0 || hasPanel) && !disabled && isConnected;

  return (
    <div
      className="border-t bg-card px-4 py-3 sticky bottom-0 z-50"
      data-testid="chat-input-area"
    >
      <div className="flex items-end gap-2 max-w-3xl mx-auto">
        <Textarea
          ref={textareaRef}
          value={input}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          placeholder={
            isConnected
              ? "Ask about the KRAKEN knowledge graph..."
              : "Waiting for connection..."
          }
          disabled={disabled || !isConnected}
          className="resize-none min-h-[44px] max-h-[200px] text-sm flex-1"
          rows={1}
          data-testid="input-message"
        />
        <Button
          onClick={handleSend}
          disabled={!canSend}
          size="icon"
          data-testid="button-send"
        >
          <Send className="h-4 w-4" />
        </Button>
      </div>

      {!isConnected && (
        <p className="text-xs text-muted-foreground text-center mt-2 max-w-md mx-auto">
          Cannot connect to backend. The KRAKEN backend may not be running.
        </p>
      )}

      {isConnected && (
        <p className="text-[10px] text-muted-foreground text-center mt-1.5">
          Press Enter to send, Shift+Enter for new line
        </p>
      )}
    </div>
  );
}
