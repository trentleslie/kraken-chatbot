import type { ConnectionStatus as ConnectionStatusType } from "@/types/messages";

interface ConnectionStatusProps {
  status: ConnectionStatusType;
}

const statusConfig: Record<
  ConnectionStatusType,
  { color: string; label: string; pulse: boolean }
> = {
  connected: { color: "bg-status-online", label: "Connected", pulse: false },
  connecting: { color: "bg-status-away", label: "Connecting...", pulse: true },
  reconnecting: {
    color: "bg-status-away",
    label: "Reconnecting...",
    pulse: true,
  },
  disconnected: {
    color: "bg-status-busy",
    label: "Disconnected",
    pulse: false,
  },
  auth_failed: {
    color: "bg-status-busy",
    label: "Auth Failed",
    pulse: false,
  },
  demo: {
    color: "bg-status-online",
    label: "Demo Mode",
    pulse: false,
  },
};

export function ConnectionStatus({ status }: ConnectionStatusProps) {
  const { color, label, pulse } = statusConfig[status];

  return (
    <div
      className="flex items-center gap-1.5"
      data-testid="connection-status"
    >
      <div
        className={`h-2 w-2 rounded-full ${color} ${pulse ? "animate-pulse" : ""}`}
      />
      <span className="text-xs text-muted-foreground select-none">
        {label}
      </span>
    </div>
  );
}
