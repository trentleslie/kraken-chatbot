import { ConnectionStatus } from "@/components/ConnectionStatus";
import type { ConnectionStatus as ConnectionStatusType } from "@/types/messages";
import { Button } from "@/components/ui/button";
import { Network, Trash2 } from "lucide-react";

interface HeaderProps {
  connectionStatus: ConnectionStatusType;
  onClearChat: () => void;
  hasMessages: boolean;
}

export function Header({
  connectionStatus,
  onClearChat,
  hasMessages,
}: HeaderProps) {
  return (
    <header
      className="flex items-center justify-between gap-4 flex-wrap px-4 sm:px-6 py-3 border-b bg-card sticky top-0 z-50"
      data-testid="header"
    >
      <div className="flex items-center gap-3">
        <div className="flex items-center justify-center w-8 h-8 rounded-md bg-primary">
          <Network className="w-4 h-4 text-primary-foreground" />
        </div>
        <div className="min-w-0">
          <h1 className="text-sm font-semibold leading-tight truncate">
            Kestrel KG Explorer
          </h1>
          <p className="text-xs text-muted-foreground truncate hidden sm:block">
            Explore the KRAKEN biomedical knowledge graph
          </p>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <ConnectionStatus status={connectionStatus} />
        {hasMessages && (
          <Button
            variant="ghost"
            size="icon"
            onClick={onClearChat}
            aria-label="Clear chat"
            data-testid="button-clear-chat"
          >
            <Trash2 className="w-4 h-4" />
          </Button>
        )}
      </div>
    </header>
  );
}
