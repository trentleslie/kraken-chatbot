export function getResultSummary(
  tool: string,
  data: Record<string, unknown>,
): string {
  const cleanTool = tool.replace(/^mcp__kestrel__/, "");

  if (
    ["hybrid_search", "text_search", "vector_search", "similar_nodes"].includes(
      cleanTool,
    )
  ) {
    let total = 0;
    for (const val of Object.values(data)) {
      if (Array.isArray(val)) total += val.length;
    }
    if (total > 0) return `${total} result${total !== 1 ? "s" : ""} found`;
    return "No results found";
  }

  if (cleanTool === "one_hop_query") {
    if (data.preview && typeof data.preview === "object") {
      const preview = data.preview as Record<string, unknown>;
      const nodeCount = preview.node_count || 0;
      const edgeCount = preview.edge_count || 0;
      return `${nodeCount} node${nodeCount !== 1 ? "s" : ""}, ${edgeCount} edge${edgeCount !== 1 ? "s" : ""}`;
    }
    if (data.nodes && Array.isArray(data.nodes)) {
      return `${data.nodes.length} node${data.nodes.length !== 1 ? "s" : ""} found`;
    }
    return "Query complete";
  }

  if (cleanTool === "get_nodes") {
    const nodes = Array.isArray(data.nodes) ? data.nodes : Array.isArray(data) ? data : [];
    return `${nodes.length} node${nodes.length !== 1 ? "s" : ""} retrieved`;
  }

  if (cleanTool === "get_edges") {
    const edges = Array.isArray(data.edges) ? data.edges : [];
    return `${edges.length} edge${edges.length !== 1 ? "s" : ""} retrieved`;
  }

  if (
    ["get_valid_categories", "get_valid_predicates", "get_valid_prefixes"].includes(cleanTool)
  ) {
    for (const val of Object.values(data)) {
      if (Array.isArray(val)) {
        return `${val.length} item${val.length !== 1 ? "s" : ""}`;
      }
    }
    return "List retrieved";
  }

  if (cleanTool === "health_check") {
    return "Health check complete";
  }

  return "Complete";
}

export interface SearchResult {
  id: string;
  name: string;
  category?: string;
  score?: number;
  description?: string;
}

export function parseSearchResults(
  data: Record<string, unknown>,
): SearchResult[] {
  const results: SearchResult[] = [];

  for (const value of Object.values(data)) {
    if (Array.isArray(value)) {
      for (const item of value) {
        if (item && typeof item === "object") {
          results.push({
            id: (item as Record<string, unknown>).id as string || (item as Record<string, unknown>).curie as string || "",
            name: (item as Record<string, unknown>).name as string || (item as Record<string, unknown>).label as string || "",
            category: (item as Record<string, unknown>).category as string || "",
            score: (item as Record<string, unknown>).score as number | undefined,
            description: (item as Record<string, unknown>).description as string || "",
          });
        }
      }
    }
  }

  return results;
}

export interface GraphRelationship {
  subject: { id: string; name: string };
  predicate: string;
  object: { id: string; name: string };
}

export function parseOneHopResults(data: Record<string, unknown>): {
  preview?: { nodeCount: number; edgeCount: number };
  relationships: GraphRelationship[];
} {
  if (data.preview && typeof data.preview === "object") {
    const preview = data.preview as Record<string, unknown>;
    return {
      preview: {
        nodeCount: (preview.node_count as number) || 0,
        edgeCount: (preview.edge_count as number) || 0,
      },
      relationships: [],
    };
  }

  const relationships: GraphRelationship[] = [];

  if (Array.isArray(data.edges) && Array.isArray(data.nodes)) {
    const nodeMap = new Map<string, { id: string; name: string }>();
    for (const node of data.nodes as Array<Record<string, unknown>>) {
      nodeMap.set(node.id as string, {
        id: node.id as string,
        name: (node.name || node.label || node.id) as string,
      });
    }

    for (const edge of data.edges as Array<Record<string, unknown>>) {
      const subjectId = (edge.subject || edge.source) as string;
      const objectId = (edge.object || edge.target) as string;
      const predicate = (edge.predicate || edge.relation) as string;

      relationships.push({
        subject: nodeMap.get(subjectId) || { id: subjectId, name: subjectId },
        predicate: predicate || "related_to",
        object: nodeMap.get(objectId) || { id: objectId, name: objectId },
      });
    }
  }

  return { relationships };
}

export interface NodeDetail {
  id: string;
  name: string;
  category: string;
  description?: string;
  synonyms?: string[];
  equivalentIds?: string[];
}

export function parseNodeDetails(
  data: Record<string, unknown>,
): NodeDetail[] {
  const nodeArray = Array.isArray(data.nodes)
    ? data.nodes
    : Array.isArray(data)
      ? data
      : [];

  return (nodeArray as Array<Record<string, unknown>>).map((node) => ({
    id: (node.id || node.curie || "") as string,
    name: (node.name || node.label || "") as string,
    category: (node.category || "") as string,
    description: node.description as string | undefined,
    synonyms: Array.isArray(node.synonyms) ? (node.synonyms as string[]) : undefined,
    equivalentIds: Array.isArray(node.equivalent_identifiers)
      ? (node.equivalent_identifiers as string[])
      : undefined,
  }));
}
