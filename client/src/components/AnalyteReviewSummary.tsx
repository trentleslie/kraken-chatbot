import { useMemo } from "react";
import { AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  applyGroupFilter,
  distinctGroups,
  distinctNameCount,
  type StructuredAnalyte,
} from "@/lib/analyteParse";

// Soft UX warning threshold on the SELECTED distinct-name count (R18). The authoritative hard
// ceiling is enforced server-side (R19); this is a non-blocking heads-up.
const SOFT_WARN_THRESHOLD = 150;

interface AnalyteReviewSummaryProps {
  analytes: StructuredAnalyte[];
  selectedGroups: string[];
  onSelectedGroupsChange: (groups: string[]) => void;
  onRemove: () => void;
  /** True when the composer textarea is empty (drives the file-only degradation warning). */
  queryEmpty: boolean;
}

/**
 * Read-only structured review + group filter (Unit 6, R15/R16/R18). Analytes are never
 * round-tripped through the textarea; the full panel + the selection are what get sent. Shows the
 * total, per-group counts, dedup line, a >150 soft warning on the selected count, and a file-only
 * degradation notice. Remove-all re-exposes the drop zone.
 */
export function AnalyteReviewSummary({
  analytes,
  selectedGroups,
  onSelectedGroupsChange,
  onRemove,
  queryEmpty,
}: AnalyteReviewSummaryProps) {
  const groups = useMemo(() => distinctGroups(analytes), [analytes]);
  const hasGroups = groups.length > 0;

  const perGroupCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const a of analytes) {
      if (a.group) counts[a.group] = (counts[a.group] ?? 0) + 1;
    }
    return counts;
  }, [analytes]);

  const selectedAnalytes = useMemo(
    () => applyGroupFilter(analytes, hasGroups ? selectedGroups : []),
    [analytes, selectedGroups, hasGroups],
  );
  const selectedCount = distinctNameCount(selectedAnalytes);
  const typesMapped = analytes.some((a) => a.type !== undefined);

  const toggleGroup = (group: string, checked: boolean) => {
    if (checked) onSelectedGroupsChange([...selectedGroups, group]);
    else onSelectedGroupsChange(selectedGroups.filter((g) => g !== group));
  };

  return (
    <div
      className="max-w-3xl mx-auto rounded-md border p-3 space-y-2"
      data-testid="analyte-review-summary"
    >
      <div className="flex items-center justify-between">
        <p className="text-sm font-medium">
          {analytes.length} analytes ready{" "}
          <span className="text-muted-foreground font-normal">
            ({selectedCount} selected{typesMapped ? ", types mapped" : ""})
          </span>
        </p>
        <Button variant="ghost" size="sm" onClick={onRemove} data-testid="analyte-remove-all">
          Remove
        </Button>
      </div>

      {hasGroups && (
        <div className="space-y-1">
          <p className="text-xs font-medium text-muted-foreground">Groups to run</p>
          <div className="flex flex-wrap gap-3">
            {groups.map((g) => (
              <label key={g} className="flex items-center gap-1.5 text-xs cursor-pointer">
                <Checkbox
                  checked={selectedGroups.includes(g)}
                  onCheckedChange={(c) => toggleGroup(g, c === true)}
                  data-testid={`group-filter-${g}`}
                />
                <span>
                  {g} ({perGroupCounts[g] ?? 0})
                </span>
              </label>
            ))}
          </div>
        </div>
      )}

      {selectedCount > SOFT_WARN_THRESHOLD && (
        <p className="flex items-center gap-1.5 text-xs text-amber-600" role="status">
          <AlertTriangle className="h-3.5 w-3.5" />
          Large panel ({selectedCount} analytes) — Triage classification may degrade and the run may
          take longer. You can still proceed.
        </p>
      )}

      {queryEmpty && (
        <p className="text-xs text-muted-foreground" role="status">
          No study-context question typed — the analysis will run with less grounding. Add a question
          for a more focused report.
        </p>
      )}
    </div>
  );
}
