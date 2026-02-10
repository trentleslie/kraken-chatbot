import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Beaker, Clock, Network } from "lucide-react";
import type { PipelineCompleteMessage } from "@/types/messages";

interface PipelineReportCardProps {
  message: PipelineCompleteMessage;
}

export function PipelineReportCard({ message }: PipelineReportCardProps) {
  const durationSec = (message.duration_ms / 1000).toFixed(1);

  return (
    <div className="flex justify-start" data-testid="pipeline-report">
      <div className="max-w-[85%] min-w-0 space-y-3">
        {/* Metadata bar */}
        <div className="flex items-center gap-4 text-xs text-muted-foreground">
          <div className="flex items-center gap-1">
            <Beaker className="h-3.5 w-3.5" />
            <span>{message.hypotheses_count} hypotheses</span>
          </div>
          <div className="flex items-center gap-1">
            <Network className="h-3.5 w-3.5" />
            <span>{message.entities_resolved} entities</span>
          </div>
          <div className="flex items-center gap-1">
            <Clock className="h-3.5 w-3.5" />
            <span>{durationSec}s</span>
          </div>
        </div>

        {/* Synthesis report */}
        <div className="prose prose-sm dark:prose-invert max-w-none prose-headings:font-semibold prose-a:text-primary prose-a:no-underline hover:prose-a:underline prose-code:font-mono prose-code:text-xs prose-code:before:content-none prose-code:after:content-none prose-pre:bg-muted prose-pre:border prose-pre:rounded-md prose-table:text-sm prose-td:px-3 prose-td:py-1.5 prose-th:px-3 prose-th:py-1.5 prose-th:text-left prose-th:font-semibold prose-thead:border-b prose-tr:border-b prose-tr:border-border">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              code({ className, children, ...props }) {
                const isInline = !className;
                if (isInline) {
                  return (
                    <code
                      className="rounded-sm bg-muted px-1.5 py-0.5 text-xs font-mono text-foreground"
                      {...props}
                    >
                      {children}
                    </code>
                  );
                }
                return (
                  <code className={className} {...props}>
                    {children}
                  </code>
                );
              },
              table({ children, ...props }) {
                return (
                  <div className="overflow-x-auto rounded-md border my-3">
                    <table className="w-full" {...props}>
                      {children}
                    </table>
                  </div>
                );
              },
            }}
          >
            {message.synthesis_report}
          </ReactMarkdown>
        </div>
      </div>
    </div>
  );
}
