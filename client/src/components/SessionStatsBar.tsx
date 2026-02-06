import type { SessionStats } from "@/types/messages";
import { formatTokens, formatCost } from "@/utils/formatters";
import { Activity } from "lucide-react";

interface SessionStatsBarProps {
  stats: SessionStats;
}

export function SessionStatsBar({ stats }: SessionStatsBarProps) {
  if (stats.turn_count === 0) return null;

  const totalTokens = stats.total_input_tokens + stats.total_output_tokens;

  const parts: string[] = [];

  if (totalTokens > 0) {
    parts.push(`${formatTokens(totalTokens)} tokens`);
  }

  if (stats.total_cost_usd > 0) {
    parts.push(formatCost(stats.total_cost_usd));
  }

  parts.push(
    `${stats.turn_count} turn${stats.turn_count !== 1 ? "s" : ""}`,
  );

  if (stats.total_tool_calls > 0) {
    parts.push(
      `${stats.total_tool_calls} tool call${stats.total_tool_calls !== 1 ? "s" : ""}`,
    );
  }

  return (
    <div
      className="flex items-center gap-1.5 text-[11px] text-muted-foreground/70 select-none px-2 py-1 rounded-md bg-muted/50"
      data-testid="session-stats"
    >
      <Activity className="w-3 h-3 flex-shrink-0" />
      <span>{parts.join(" \u00b7 ")}</span>
    </div>
  );
}
