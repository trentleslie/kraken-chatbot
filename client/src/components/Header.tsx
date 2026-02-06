import { ConnectionStatus } from "@/components/ConnectionStatus";
import { SessionStatsBar } from "@/components/SessionStatsBar";
import type { ConnectionStatus as ConnectionStatusType, SessionStats } from "@/types/messages";
import { Button } from "@/components/ui/button";
import { Trash2 } from "lucide-react";

interface HeaderProps {
  connectionStatus: ConnectionStatusType;
  sessionStats: SessionStats;
  onClearChat: () => void;
  hasMessages: boolean;
}

export function Header({
  connectionStatus,
  sessionStats,
  onClearChat,
  hasMessages,
}: HeaderProps) {
  return (
    <header
      className="flex items-center justify-between gap-4 flex-wrap px-4 sm:px-6 py-2 border-b bg-card sticky top-0 z-50"
      data-testid="header"
    >
      <ConnectionStatus status={connectionStatus} />

      <div className="flex items-center gap-2">
        <SessionStatsBar stats={sessionStats} />
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
