export function formatPredicate(predicate: string): string {
  return predicate.replace(/^biolink:/, "");
}

export function formatScore(score: number): string {
  return score.toFixed(1);
}

export function formatCategory(category: string): string {
  return category.replace(/^biolink:/, "");
}

export function getToolDisplayName(tool: string): string {
  const names: Record<string, string> = {
    hybrid_search: "Hybrid Search",
    text_search: "Text Search",
    vector_search: "Vector Search",
    similar_nodes: "Similar Nodes",
    one_hop_query: "One-Hop Query",
    get_nodes: "Get Nodes",
    get_edges: "Get Edges",
    canonicalize: "Canonicalize",
    get_valid_categories: "Valid Categories",
    get_valid_predicates: "Valid Predicates",
    get_valid_prefixes: "Valid Prefixes",
    health_check: "Health Check",
  };
  const cleanName = tool.replace(/^mcp__kestrel__/, "");
  return names[cleanName] || cleanName.replace(/_/g, " ");
}

export function getToolCleanName(tool: string): string {
  return tool.replace(/^mcp__kestrel__/, "");
}
