import { Badge } from "@/components/ui/badge";
import {
  formatPredicate,
  formatCategory,
  formatScore,
} from "@/utils/formatters";
import {
  parseSearchResults,
  parseOneHopResults,
  parseNodeDetails,
} from "@/utils/toolResultParsers";

interface ToolResultDisplayProps {
  tool: string;
  data: Record<string, unknown>;
}

export function ToolResultDisplay({ tool, data }: ToolResultDisplayProps) {
  const cleanTool = tool.replace(/^mcp__kestrel__/, "");

  if (
    ["hybrid_search", "text_search", "vector_search", "similar_nodes"].includes(
      cleanTool,
    )
  ) {
    const results = parseSearchResults(data);
    if (results.length === 0)
      return (
        <p className="text-xs text-muted-foreground py-1">No results found</p>
      );

    return (
      <div className="space-y-1.5">
        {results.map((result, idx) => (
          <div
            key={idx}
            className="flex flex-col gap-0.5 p-2 bg-background rounded-md"
          >
            <span className="text-xs font-mono text-primary font-medium">
              {result.id}
            </span>
            <span className="text-sm font-medium text-foreground">
              {result.name}
            </span>
            <div className="flex items-center gap-2 flex-wrap">
              {result.category && (
                <Badge variant="outline" className="text-xs">
                  {formatCategory(result.category)}
                </Badge>
              )}
              {result.score != null && (
                <span className="text-xs text-muted-foreground">
                  Score: {formatScore(result.score)}
                </span>
              )}
            </div>
          </div>
        ))}
      </div>
    );
  }

  if (cleanTool === "one_hop_query") {
    const { preview, relationships } = parseOneHopResults(data);

    if (preview) {
      return (
        <div className="p-2 bg-background rounded-md">
          <p className="text-sm text-foreground">
            <span className="font-medium">{preview.nodeCount}</span> node
            {preview.nodeCount !== 1 ? "s" : ""},{" "}
            <span className="font-medium">{preview.edgeCount}</span> edge
            {preview.edgeCount !== 1 ? "s" : ""}
          </p>
        </div>
      );
    }

    if (relationships.length === 0) {
      return (
        <p className="text-xs text-muted-foreground py-1">
          No relationships found
        </p>
      );
    }

    return (
      <div className="space-y-1">
        {relationships.slice(0, 10).map((rel, idx) => (
          <div
            key={idx}
            className="text-xs p-2 bg-background rounded-md flex items-center gap-1.5 flex-wrap"
          >
            <span className="font-mono text-primary font-medium">
              {rel.subject.id}
            </span>
            {rel.subject.name !== rel.subject.id && (
              <span className="text-muted-foreground">
                ({rel.subject.name})
              </span>
            )}
            <span className="text-muted-foreground mx-0.5">
              {"\u2192"}
            </span>
            <span className="font-mono text-accent-foreground">
              {formatPredicate(rel.predicate)}
            </span>
            <span className="text-muted-foreground mx-0.5">
              {"\u2192"}
            </span>
            <span className="font-mono text-primary font-medium">
              {rel.object.id}
            </span>
            {rel.object.name !== rel.object.id && (
              <span className="text-muted-foreground">
                ({rel.object.name})
              </span>
            )}
          </div>
        ))}
        {relationships.length > 10 && (
          <p className="text-xs text-muted-foreground px-2 pt-1">
            ...and {relationships.length - 10} more
          </p>
        )}
      </div>
    );
  }

  if (cleanTool === "get_nodes") {
    const nodes = parseNodeDetails(data);
    if (nodes.length === 0)
      return (
        <p className="text-xs text-muted-foreground py-1">No node details</p>
      );

    return (
      <div className="space-y-1.5">
        {nodes.map((node, idx) => (
          <div key={idx} className="p-2 bg-background rounded-md space-y-1">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-xs font-mono text-primary font-medium">
                {node.id}
              </span>
              {node.category && (
                <Badge variant="outline" className="text-xs">
                  {formatCategory(node.category)}
                </Badge>
              )}
            </div>
            <p className="text-sm font-medium text-foreground">{node.name}</p>
            {node.description && (
              <p className="text-xs text-muted-foreground line-clamp-2">
                {node.description}
              </p>
            )}
            {node.synonyms && node.synonyms.length > 0 && (
              <p className="text-xs text-muted-foreground">
                Synonyms: {node.synonyms.slice(0, 5).join(", ")}
                {node.synonyms.length > 5
                  ? ` +${node.synonyms.length - 5} more`
                  : ""}
              </p>
            )}
            {node.equivalentIds && node.equivalentIds.length > 0 && (
              <p className="text-xs text-muted-foreground font-mono">
                IDs: {node.equivalentIds.slice(0, 4).join(", ")}
                {node.equivalentIds.length > 4
                  ? ` +${node.equivalentIds.length - 4} more`
                  : ""}
              </p>
            )}
          </div>
        ))}
      </div>
    );
  }

  if (
    [
      "get_valid_categories",
      "get_valid_predicates",
      "get_valid_prefixes",
    ].includes(cleanTool)
  ) {
    let items: unknown[] = [];
    for (const val of Object.values(data)) {
      if (Array.isArray(val)) {
        items = val;
        break;
      }
    }

    return (
      <div className="flex flex-wrap gap-1 p-2 bg-background rounded-md">
        {items.slice(0, 30).map((item, idx) => (
          <Badge key={idx} variant="outline" className="text-xs font-mono">
            {typeof item === "string" ? item : JSON.stringify(item)}
          </Badge>
        ))}
        {items.length > 30 && (
          <span className="text-xs text-muted-foreground self-center ml-1">
            +{items.length - 30} more
          </span>
        )}
      </div>
    );
  }

  return (
    <pre className="text-xs font-mono bg-background rounded-md p-2 overflow-x-auto max-h-40 overflow-y-auto">
      {JSON.stringify(data, null, 2)}
    </pre>
  );
}
