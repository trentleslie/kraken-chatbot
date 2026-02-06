import type { TraceMessage } from "@/types/messages";
import { formatTokens, formatCost, formatDuration } from "@/utils/formatters";
import { Zap } from "lucide-react";

interface TraceBarProps {
  trace: TraceMessage;
}

export function TraceBar({ trace }: TraceBarProps) {
  const parts: string[] = [];

  if (trace.duration_ms != null) {
    parts.push(formatDuration(trace.duration_ms));
  }

  if (trace.input_tokens != null || trace.output_tokens != null) {
    const segments: string[] = [];
    if (trace.input_tokens != null) segments.push(`${formatTokens(trace.input_tokens)} in`);
    if (trace.output_tokens != null) segments.push(`${formatTokens(trace.output_tokens)} out`);
    parts.push(segments.join(" / "));
  }

  if (trace.cache_read_tokens != null && trace.cache_read_tokens > 0) {
    parts.push(`${formatTokens(trace.cache_read_tokens)} cached`);
  }

  if (trace.tool_calls_count != null && trace.tool_calls_count > 0) {
    parts.push(
      `${trace.tool_calls_count} tool call${trace.tool_calls_count !== 1 ? "s" : ""}`,
    );
  }

  if (trace.cost_usd != null) {
    parts.push(formatCost(trace.cost_usd));
  }

  if (parts.length === 0) return null;

  return (
    <div
      className="flex items-center justify-center py-2"
      data-testid={`trace-bar-${trace.id}`}
    >
      <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground/60 select-none">
        <Zap className="w-3 h-3 flex-shrink-0" />
        <span>{parts.join(" \u00b7 ")}</span>
        {trace.model && (
          <span className="text-muted-foreground/40 font-mono text-[10px] ml-1">
            {trace.model}
          </span>
        )}
      </div>
    </div>
  );
}
