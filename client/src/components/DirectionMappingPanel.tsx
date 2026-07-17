import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  buildModuleDirections,
  isFormulaInjection,
  suggestDirectionMapping,
  type DirectionMapping,
  type ModuleDirectionInput,
  type ParsedFile,
} from "@/lib/analyteParse";

type DirectionTarget = keyof DirectionMapping; // "group" | "correlation" | "trait"

interface DirectionMappingPanelProps {
  parsed: ParsedFile;
  fileName: string;
  /** Called with the built per-module directions when the user confirms. */
  onConfirm: (directions: ModuleDirectionInput[]) => void;
  onCancel: () => void;
}

const UNMAPPED = "__none__";

/**
 * Column-mapping panel for the (separately-exported) ME-trait direction table (Axis A). Mirrors
 * {@link ColumnMappingPanel}: Radix Select dropdowns assign file columns to the three direction
 * targets (module/group, eigengene→trait correlation, trait label). All three are required before
 * Confirm; the rows are built VERBATIM via `buildModuleDirections` (malformed cells preserved) so
 * the backend R19 gate — authoritative — re-validates and rejects a bad direction rather than the
 * client silently dropping it. These directions make the module-spine sign-inversion metric
 * computable; without them every module spine is emitted without a direction.
 */
export function DirectionMappingPanel({
  parsed,
  fileName,
  onConfirm,
  onCancel,
}: DirectionMappingPanelProps) {
  const [mapping, setMapping] = useState<DirectionMapping>(() =>
    suggestDirectionMapping(parsed.headers),
  );

  const setTarget = (target: DirectionTarget, value: string) => {
    setMapping((prev) => ({ ...prev, [target]: value === UNMAPPED ? undefined : value }));
  };

  const hasDataRows = parsed.rows.length > 0;
  const canConfirm = Boolean(mapping.group && mapping.correlation && mapping.trait) && hasDataRows;

  const directions = useMemo(
    () =>
      mapping.group && mapping.correlation && mapping.trait
        ? buildModuleDirections(parsed.rows, mapping)
        : [],
    [parsed.rows, mapping],
  );

  const renderSelect = (target: DirectionTarget, label: string) => (
    <div className="flex flex-col gap-1">
      <label className="text-xs font-medium text-muted-foreground">{label} *</label>
      <Select value={mapping[target] ?? UNMAPPED} onValueChange={(v) => setTarget(target, v)}>
        <SelectTrigger className="h-8 text-sm" data-testid={`dir-map-${target}`}>
          <SelectValue placeholder="Unmapped" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={UNMAPPED}>Unmapped</SelectItem>
          {parsed.headers.map((h) => (
            <SelectItem key={h} value={h}>
              {h}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );

  return (
    <div
      className="max-w-3xl mx-auto rounded-md border p-3 space-y-3"
      data-testid="direction-mapping-panel"
    >
      <div className="flex items-center justify-between">
        <p className="text-sm font-medium">Map module-direction columns — {fileName}</p>
        <span className="text-xs text-muted-foreground">{parsed.rows.length} rows</span>
      </div>

      <p className="text-xs text-muted-foreground">
        The per-module eigengene→outcome direction table (one row per module). Its signed correlation
        is what makes the module-spine sign-inversion metric computable.
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {renderSelect("group", "Module / Group")}
        {renderSelect("correlation", "ME→trait correlation")}
        {renderSelect("trait", "Trait label")}
      </div>

      {/* Preview (first ~5 rows). Formula-injection cells are badged and rendered as text. */}
      <div className="overflow-x-auto rounded border">
        <table className="w-full text-xs">
          <thead>
            <tr className="bg-muted/50">
              {parsed.headers.map((h) => (
                <th key={h} className="px-2 py-1 text-left font-medium">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {parsed.previewRows.map((row, i) => (
              <tr key={i} className="border-t">
                {parsed.headers.map((h) => (
                  <td key={h} className="px-2 py-1">
                    {isFormulaInjection(row[h]) ? (
                      <span title="Looks like a spreadsheet formula" className="text-amber-600">
                        ⚠ {row[h]}
                      </span>
                    ) : (
                      row[h]
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {!hasDataRows && <p className="text-xs text-destructive">No data rows found.</p>}
      {canConfirm && (
        <p className="text-xs text-muted-foreground">
          {directions.length} module {directions.length === 1 ? "direction" : "directions"} mapped
          (the server validates each).
        </p>
      )}

      <div className="flex justify-end gap-2">
        <Button variant="ghost" size="sm" onClick={onCancel} data-testid="direction-mapping-cancel">
          Cancel
        </Button>
        <Button
          size="sm"
          disabled={!canConfirm}
          onClick={() => onConfirm(directions)}
          data-testid="direction-mapping-confirm"
        >
          Attach directions
        </Button>
      </div>
    </div>
  );
}
