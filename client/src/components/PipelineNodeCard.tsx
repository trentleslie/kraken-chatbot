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
  FileText,
  Search,
  GitBranch,
  BarChart3,
  Zap,
  Share2,
  Layers,
  Clock,
  FlaskConical,
  type LucideIcon,
} from "lucide-react";
import { formatDuration } from "@/utils/formatters";
import type { PipelineNodeDetailMessage } from "@/types/messages";

interface PipelineNodeCardProps {
  message: PipelineNodeDetailMessage;
}

const NODE_META: Record<string, { icon: LucideIcon; label: string }> = {
  intake: { icon: FileText, label: "Intake" },
  entity_resolution: { icon: Search, label: "Entity Resolution" },
  triage: { icon: BarChart3, label: "Triage" },
  direct_kg: { icon: GitBranch, label: "Direct KG Analysis" },
  cold_start: { icon: Zap, label: "Cold-Start Analysis" },
  pathway_enrichment: { icon: Share2, label: "Pathway Enrichment" },
  integration: { icon: Layers, label: "Integration" },
  temporal: { icon: Clock, label: "Temporal Analysis" },
  synthesis: { icon: FlaskConical, label: "Synthesis" },
};

function IntakeDetails({ details }: { details: Record<string, unknown> }) {
  const entities = (details.entities as string[]) || [];
  const queryType = details.query_type as string;
  const isLongitudinal = details.is_longitudinal as boolean;
  const directives = (details.directives as string[]) || [];
  const studyCtx = (details.study_context as Record<string, string>) || {};

  return (
    <div className="space-y-2">
      {entities.length > 0 && (
        <div className="space-y-1">
          <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Extracted Entities</p>
          <div className="flex flex-wrap gap-1">
            {entities.map((e, i) => (
              <Badge key={i} variant="secondary" className="text-xs font-mono">{e}</Badge>
            ))}
          </div>
        </div>
      )}
      <div className="flex flex-wrap gap-2">
        <Badge variant="outline" className="text-xs">{queryType}</Badge>
        {isLongitudinal && <Badge variant="outline" className="text-xs">longitudinal</Badge>}
      </div>
      {directives.length > 0 && (
        <div className="space-y-1">
          <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Directives</p>
          <ul className="text-xs text-muted-foreground list-disc list-inside">
            {directives.map((d, i) => <li key={i}>{d}</li>)}
          </ul>
        </div>
      )}
      {Object.keys(studyCtx).length > 0 && (
        <div className="space-y-1">
          <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Study Context</p>
          <div className="text-xs font-mono bg-background rounded-md p-2 space-y-0.5">
            {Object.entries(studyCtx).map(([k, v]) => (
              <div key={k} className="flex gap-2">
                <span className="text-muted-foreground">{k}:</span>
                <span className="text-foreground">{v}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function EntityResolutionDetails({ details }: { details: Record<string, unknown> }) {
  const entities = (details.entities as Array<{
    raw_name: string;
    curie: string | null;
    resolved_name: string | null;
    confidence: number;
    method: string;
  }>) || [];
  const methods = (details.methods as Record<string, number>) || {};

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-1.5">
        {Object.entries(methods).map(([m, count]) => (
          <Badge key={m} variant="secondary" className="text-xs">
            {count} {m}
          </Badge>
        ))}
      </div>
      {entities.length > 0 && (
        <div className="text-xs font-mono bg-background rounded-md p-2 space-y-1 max-h-48 overflow-y-auto">
          {entities.map((e, i) => (
            <div key={i} className="flex items-center gap-2">
              <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${e.curie ? "bg-green-500" : "bg-red-400"}`} />
              <span className="text-muted-foreground truncate flex-shrink-0 min-w-[120px]">{e.raw_name}</span>
              {e.curie ? (
                <>
                  <span className="text-foreground">{e.curie}</span>
                  <span className="text-muted-foreground/60">{Math.round(e.confidence * 100)}%</span>
                </>
              ) : (
                <span className="text-muted-foreground/60 italic">unresolved</span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function TriageDetails({ details }: { details: Record<string, unknown> }) {
  const entities = (details.entities as Array<{
    curie: string;
    raw_name: string;
    edge_count: number;
    classification: string;
  }>) || [];

  const classColors: Record<string, string> = {
    well_characterized: "bg-green-500",
    moderate: "bg-blue-500",
    sparse: "bg-yellow-500",
    cold_start: "bg-red-400",
  };

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-1.5">
        {[
          { key: "well_characterized", label: "Well-char.", count: details.well_characterized as number },
          { key: "moderate", label: "Moderate", count: details.moderate as number },
          { key: "sparse", label: "Sparse", count: details.sparse as number },
          { key: "cold_start", label: "Cold-start", count: details.cold_start as number },
        ].filter(b => b.count > 0).map(b => (
          <Badge key={b.key} variant="secondary" className="text-xs gap-1">
            <span className={`w-1.5 h-1.5 rounded-full ${classColors[b.key]}`} />
            {b.count} {b.label}
          </Badge>
        ))}
      </div>
      {entities.length > 0 && (
        <div className="text-xs font-mono bg-background rounded-md p-2 space-y-1 max-h-40 overflow-y-auto">
          {entities.map((e, i) => (
            <div key={i} className="flex items-center gap-2">
              <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${classColors[e.classification] || "bg-gray-400"}`} />
              <span className="text-muted-foreground truncate min-w-[120px]">{e.raw_name}</span>
              <span className="text-foreground">{e.edge_count.toLocaleString()} edges</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function DirectKgDetails({ details }: { details: Record<string, unknown> }) {
  const topDiseases = (details.top_diseases as Array<{
    entity: string;
    disease: string;
    predicate: string;
    evidence: string;
    preset: string;
    pmids: number;
  }>) || [];
  const topPathways = (details.top_pathways as Array<{
    entity: string;
    pathway: string;
    preset: string;
  }>) || [];

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-1.5">
        <Badge variant="secondary" className="text-xs">{details.diseases_count as number} diseases</Badge>
        <Badge variant="secondary" className="text-xs">{details.pathways_count as number} pathways</Badge>
        <Badge variant="secondary" className="text-xs">{details.findings_count as number} findings</Badge>
      </div>
      {topDiseases.length > 0 && (
        <div className="space-y-1">
          <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Top Disease Associations</p>
          <div className="text-xs font-mono bg-background rounded-md p-2 space-y-1 max-h-40 overflow-y-auto">
            {topDiseases.slice(0, 8).map((d, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="text-foreground truncate">{d.disease}</span>
                <span className="text-muted-foreground/60">{d.evidence}</span>
                {d.preset === "hidden_gems" && <Badge variant="outline" className="text-[10px] px-1 py-0">novel</Badge>}
                {d.pmids > 0 && <span className="text-muted-foreground/60">{d.pmids} PMIDs</span>}
              </div>
            ))}
          </div>
        </div>
      )}
      {topPathways.length > 0 && (
        <div className="space-y-1">
          <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Top Pathways</p>
          <div className="text-xs font-mono bg-background rounded-md p-2 space-y-0.5 max-h-32 overflow-y-auto">
            {topPathways.slice(0, 6).map((p, i) => (
              <div key={i} className="text-foreground truncate">{p.pathway}</div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function ColdStartDetails({ details }: { details: Record<string, unknown> }) {
  const topAnalogues = (details.top_analogues as Array<{
    curie: string;
    name: string;
    similarity: number;
  }>) || [];
  const topInferred = (details.top_inferred as Array<{
    source: string;
    target: string;
    logic: string;
    confidence: string;
  }>) || [];

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-1.5">
        <Badge variant="secondary" className="text-xs">{details.analogues_count as number} analogues</Badge>
        <Badge variant="secondary" className="text-xs">{details.inferred_count as number} inferred</Badge>
      </div>
      {topAnalogues.length > 0 && (
        <div className="space-y-1">
          <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Top Analogues</p>
          <div className="text-xs font-mono bg-background rounded-md p-2 space-y-0.5 max-h-32 overflow-y-auto">
            {topAnalogues.map((a, i) => (
              <div key={i} className="flex gap-2">
                <span className="text-foreground">{a.name}</span>
                <span className="text-muted-foreground/60">{(a.similarity * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}
      {topInferred.length > 0 && (
        <div className="space-y-1">
          <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Inferred Associations</p>
          <div className="text-xs bg-background rounded-md p-2 space-y-1.5 max-h-40 overflow-y-auto">
            {topInferred.map((inf, i) => (
              <div key={i} className="space-y-0.5">
                <span className="font-mono text-foreground">{inf.target}</span>
                <p className="text-muted-foreground/80 text-[11px]">{inf.logic}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function PathwayEnrichmentDetails({ details }: { details: Record<string, unknown> }) {
  const themes = (details.themes as Array<{
    category: string;
    members_count: number;
    member_names: string[];
    input_coverage: number;
  }>) || [];
  const topNeighbors = (details.top_neighbors as Array<{
    curie: string;
    name: string;
    category: string;
    degree: number;
    connected_inputs: string[];
  }>) || [];

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-1.5">
        <Badge variant="secondary" className="text-xs">{details.shared_neighbors_count as number} shared neighbors</Badge>
        <Badge variant="secondary" className="text-xs">{details.themes_count as number} themes</Badge>
      </div>
      {themes.length > 0 && (
        <div className="space-y-1">
          <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Biological Themes</p>
          <div className="text-xs bg-background rounded-md p-2 space-y-1.5">
            {themes.map((t, i) => (
              <div key={i}>
                <span className="font-mono text-foreground">{t.category.replace("biolink:", "")}</span>
                <span className="text-muted-foreground/60 ml-2">{t.members_count} members, {t.input_coverage} inputs</span>
                {t.member_names.length > 0 && (
                  <p className="text-muted-foreground/80 text-[11px] truncate">{t.member_names.join(", ")}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
      {topNeighbors.length > 0 && (
        <div className="space-y-1">
          <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Top Shared Neighbors</p>
          <div className="text-xs font-mono bg-background rounded-md p-2 space-y-0.5 max-h-32 overflow-y-auto">
            {topNeighbors.slice(0, 6).map((n, i) => (
              <div key={i} className="flex gap-2">
                <span className="text-foreground">{n.name}</span>
                <span className="text-muted-foreground/60">{n.connected_inputs.length} inputs</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function IntegrationDetails({ details }: { details: Record<string, unknown> }) {
  const topBridges = (details.top_bridges as Array<{
    path: string;
    entity_names: string[];
    tier: number;
    novelty: string;
    significance: string;
  }>) || [];
  const topGaps = (details.top_gaps as Array<{
    name: string;
    reason: string;
    interpretation: string;
    informative: boolean;
  }>) || [];

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-1.5">
        <Badge variant="secondary" className="text-xs">{details.bridges_count as number} bridges</Badge>
        <Badge variant="secondary" className="text-xs">{details.gaps_count as number} gaps</Badge>
      </div>
      {topBridges.length > 0 && (
        <div className="space-y-1">
          <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Cross-Type Bridges</p>
          <div className="text-xs bg-background rounded-md p-2 space-y-1.5 max-h-40 overflow-y-auto">
            {topBridges.map((b, i) => (
              <div key={i} className="space-y-0.5">
                <div className="font-mono text-foreground">{b.path}</div>
                <p className="text-muted-foreground/80 text-[11px]">{b.significance}</p>
              </div>
            ))}
          </div>
        </div>
      )}
      {topGaps.length > 0 && (
        <div className="space-y-1">
          <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Gap Entities</p>
          <div className="text-xs bg-background rounded-md p-2 space-y-1 max-h-32 overflow-y-auto">
            {topGaps.map((g, i) => (
              <div key={i}>
                <span className="font-mono text-foreground">{g.name}</span>
                {g.informative && <Badge variant="outline" className="text-[10px] px-1 py-0 ml-1">informative</Badge>}
                <p className="text-muted-foreground/80 text-[11px]">{g.interpretation}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function TemporalDetails({ details }: { details: Record<string, unknown> }) {
  const classifications = (details.classifications as Array<{
    entity: string;
    finding: string;
    classification: string;
    reasoning: string;
    confidence: string;
  }>) || [];

  const classColors: Record<string, string> = {
    upstream_cause: "bg-orange-500",
    downstream_consequence: "bg-blue-500",
    parallel_effect: "bg-purple-500",
  };

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-1.5">
        {[
          { key: "upstream_cause", label: "Upstream", count: details.upstream_count as number },
          { key: "downstream_consequence", label: "Downstream", count: details.downstream_count as number },
          { key: "parallel_effect", label: "Parallel", count: details.parallel_count as number },
        ].filter(b => b.count > 0).map(b => (
          <Badge key={b.key} variant="secondary" className="text-xs gap-1">
            <span className={`w-1.5 h-1.5 rounded-full ${classColors[b.key]}`} />
            {b.count} {b.label}
          </Badge>
        ))}
      </div>
      {classifications.length > 0 && (
        <div className="text-xs bg-background rounded-md p-2 space-y-1.5 max-h-48 overflow-y-auto">
          {classifications.map((c, i) => (
            <div key={i} className="flex items-start gap-2">
              <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 mt-1.5 ${classColors[c.classification] || "bg-gray-400"}`} />
              <div>
                <span className="font-mono text-foreground">{c.entity}</span>
                <p className="text-muted-foreground/80 text-[11px]">{c.reasoning}</p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function SynthesisDetails({ details }: { details: Record<string, unknown> }) {
  const hypotheses = (details.hypotheses as Array<{
    title: string;
    tier: number;
    confidence: string;
    claim: string;
  }>) || [];

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-1.5">
        <Badge variant="secondary" className="text-xs">{details.hypotheses_count as number} hypotheses</Badge>
        <Badge variant="secondary" className="text-xs">{((details.report_length as number) / 1000).toFixed(1)}k chars</Badge>
      </div>
      {hypotheses.length > 0 && (
        <div className="space-y-1">
          <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Hypotheses</p>
          <div className="text-xs bg-background rounded-md p-2 space-y-1.5 max-h-48 overflow-y-auto">
            {hypotheses.map((h, i) => (
              <div key={i} className="space-y-0.5">
                <div className="flex items-center gap-1.5">
                  <Badge variant="outline" className="text-[10px] px-1 py-0">T{h.tier}</Badge>
                  <span className="font-medium text-foreground">{h.title}</span>
                </div>
                <p className="text-muted-foreground/80 text-[11px]">{h.claim}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function NodeDetails({ node, details }: { node: string; details: Record<string, unknown> }) {
  switch (node) {
    case "intake": return <IntakeDetails details={details} />;
    case "entity_resolution": return <EntityResolutionDetails details={details} />;
    case "triage": return <TriageDetails details={details} />;
    case "direct_kg": return <DirectKgDetails details={details} />;
    case "cold_start": return <ColdStartDetails details={details} />;
    case "pathway_enrichment": return <PathwayEnrichmentDetails details={details} />;
    case "integration": return <IntegrationDetails details={details} />;
    case "temporal": return <TemporalDetails details={details} />;
    case "synthesis": return <SynthesisDetails details={details} />;
    default: return <pre className="text-xs font-mono">{JSON.stringify(details, null, 2)}</pre>;
  }
}

export function PipelineNodeCard({ message }: PipelineNodeCardProps) {
  const [isOpen, setIsOpen] = useState(false);

  const meta = NODE_META[message.node] || { icon: FileText, label: message.node };
  const Icon = meta.icon;
  const hasDetails = Object.keys(message.details).length > 0;

  return (
    <div className="flex justify-start" data-testid={`pipeline-node-${message.node}`}>
      <div className="w-full max-w-[85%]">
        <Collapsible open={isOpen} onOpenChange={setIsOpen}>
          <Card className="overflow-visible bg-muted/40">
            <CollapsibleTrigger asChild>
              <button
                className="w-full text-left p-3 rounded-md hover-elevate active-elevate-2 cursor-pointer"
                data-testid={`button-toggle-node-${message.node}`}
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2 min-w-0">
                    <Icon className="h-3.5 w-3.5 text-primary flex-shrink-0" />
                    <Badge variant="secondary" className="text-xs font-mono flex-shrink-0">
                      {meta.label}
                    </Badge>
                  </div>
                  <div className="flex items-center gap-1.5 flex-shrink-0">
                    {message.duration_ms > 0 && (
                      <span className="text-[10px] text-muted-foreground/60 font-mono" data-testid={`node-duration-${message.node}`}>
                        {formatDuration(message.duration_ms)}
                      </span>
                    )}
                    {hasDetails && (
                      <ChevronRight
                        className={`h-3.5 w-3.5 text-muted-foreground transition-transform duration-150 ${isOpen ? "rotate-90" : ""}`}
                      />
                    )}
                  </div>
                </div>
                <p className="text-xs text-muted-foreground mt-1 truncate">
                  {message.summary}
                </p>
              </button>
            </CollapsibleTrigger>

            {hasDetails && (
              <CollapsibleContent>
                <div className="px-3 pb-3">
                  <div className="border-t pt-3">
                    <NodeDetails node={message.node} details={message.details} />
                  </div>
                </div>
              </CollapsibleContent>
            )}
          </Card>
        </Collapsible>
      </div>
    </div>
  );
}
