import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  ChevronRight,
  Check,
  Loader2,
  Search,
  GitBranch,
  Info,
  Link2,
  List,
  Heart,
  CircleDot,
  Shuffle,
  type LucideIcon,
} from "lucide-react";
import { getToolDisplayName } from "@/utils/formatters";
import { getResultSummary } from "@/utils/toolResultParsers";
import { ToolResultDisplay } from "@/components/ToolResultDisplay";
import type { ToolUseMessage } from "@/types/messages";

interface ToolCallCardProps {
  message: ToolUseMessage;
}

function getToolIcon(tool: string): LucideIcon {
  const cleanName = tool.replace(/^mcp__kestrel__/, "");
  const icons: Record<string, LucideIcon> = {
    hybrid_search: Search,
    text_search: Search,
    vector_search: Search,
    similar_nodes: CircleDot,
    one_hop_query: GitBranch,
    get_nodes: Info,
    get_edges: Link2,
    canonicalize: Shuffle,
    get_valid_categories: List,
    get_valid_predicates: List,
    get_valid_prefixes: List,
    health_check: Heart,
  };
  return icons[cleanName] || Search;
}

function getArgsPreview(args: Record<string, unknown>): string {
  if (args.search_text) return `"${args.search_text}"`;
  if (args.query) return `"${args.query}"`;
  if (args.start_node_ids) return String(args.start_node_ids);
  if (args.node_ids) return String(args.node_ids);
  if (args.curie) return String(args.curie);

  for (const val of Object.values(args)) {
    if (typeof val === "string" && val.length > 0) return `"${val}"`;
  }
  return "";
}

export function ToolCallCard({ message }: ToolCallCardProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [showRawJson, setShowRawJson] = useState(false);

  const Icon = getToolIcon(message.tool);
  const isPending = message.status === "pending";
  const argsPreview = getArgsPreview(message.args);
  const summary = message.result
    ? getResultSummary(message.tool, message.result)
    : null;

  return (
    <div
      className="flex justify-start"
      data-testid={`tool-call-${message.id}`}
    >
      <div className="w-full max-w-[85%]">
        <Collapsible open={isOpen} onOpenChange={setIsOpen}>
          <Card className="overflow-visible bg-muted/40">
            <CollapsibleTrigger asChild>
              <button
                className="w-full text-left p-3 rounded-md hover-elevate active-elevate-2 cursor-pointer"
                data-testid={`button-toggle-tool-${message.id}`}
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2 min-w-0">
                    <Icon className="h-3.5 w-3.5 text-primary flex-shrink-0" />
                    <Badge variant="secondary" className="text-xs font-mono flex-shrink-0">
                      {getToolDisplayName(message.tool)}
                    </Badge>
                  </div>
                  <div className="flex items-center gap-1.5 flex-shrink-0">
                    {isPending ? (
                      <Loader2 className="h-3.5 w-3.5 text-muted-foreground animate-spin" />
                    ) : (
                      <Check className="h-3.5 w-3.5 text-primary" />
                    )}
                    <ChevronRight
                      className={`h-3.5 w-3.5 text-muted-foreground transition-transform duration-150 ${isOpen ? "rotate-90" : ""}`}
                    />
                  </div>
                </div>

                {argsPreview && (
                  <p className="text-xs text-muted-foreground mt-1.5 font-mono truncate">
                    {argsPreview}
                  </p>
                )}

                {summary && !isOpen && (
                  <p className="text-xs text-muted-foreground mt-1">
                    {summary}
                  </p>
                )}
              </button>
            </CollapsibleTrigger>

            <CollapsibleContent>
              <div className="px-3 pb-3 space-y-3">
                <div className="border-t pt-3 space-y-3">
                  <div className="space-y-1.5">
                    <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
                      Arguments
                    </p>
                    <div className="text-xs font-mono bg-background rounded-md p-2.5 space-y-0.5 overflow-x-auto">
                      {Object.entries(message.args).map(([key, value]) => (
                        <div key={key} className="flex gap-2">
                          <span className="text-muted-foreground flex-shrink-0">
                            {key}:
                          </span>
                          <span className="text-foreground break-all">
                            {JSON.stringify(value)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {message.result && (
                    <div className="space-y-1.5">
                      <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
                        Results
                        {summary ? ` \u2014 ${summary}` : ""}
                      </p>
                      <ToolResultDisplay
                        tool={message.tool}
                        data={message.result}
                      />

                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setShowRawJson(!showRawJson);
                        }}
                        className="text-xs text-muted-foreground hover:text-foreground transition-colors flex items-center gap-1 pt-1"
                        data-testid={`button-raw-json-${message.id}`}
                      >
                        <ChevronRight
                          className={`h-3 w-3 transition-transform duration-150 ${showRawJson ? "rotate-90" : ""}`}
                        />
                        Raw JSON
                      </button>

                      {showRawJson && (
                        <pre className="text-xs font-mono bg-background rounded-md p-2.5 overflow-x-auto max-h-60 overflow-y-auto text-foreground">
                          {JSON.stringify(message.result, null, 2)}
                        </pre>
                      )}
                    </div>
                  )}

                  {isPending && (
                    <div className="flex items-center gap-2">
                      <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />
                      <span className="text-xs text-muted-foreground">
                        Waiting for results...
                      </span>
                    </div>
                  )}
                </div>
              </div>
            </CollapsibleContent>
          </Card>
        </Collapsible>
      </div>
    </div>
  );
}
