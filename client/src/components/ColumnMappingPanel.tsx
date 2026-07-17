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
  buildAnalytes,
  buildModuleDirections,
  countInvalidWeightCells,
  distinctGroups,
  isFormulaInjection,
  suggestMapping,
  type ColumnMapping,
  type MappingTarget,
  type ModuleDirectionInput,
  type ParsedFile,
  type StructuredAnalyte,
} from "@/lib/analyteParse";

interface ColumnMappingPanelProps {
  parsed: ParsedFile;
  fileName: string;
  onConfirm: (analytes: StructuredAnalyte[], moduleDirections: ModuleDirectionInput[]) => void;
  onCancel: () => void;
}

const UNMAPPED = "__none__";

/**
 * Column-mapping panel (Unit 6, R5–R8; extended for Axis A signed weights). Radix Select dropdowns
 * assign file columns to targets (analyte required; group/type/kME/kIM optional). Auto-suggests
 * unambiguous mappings; Continue is disabled until the analyte column is mapped AND every mapped
 * kME/kIM cell is valid (reject-don't-clip: a mapped-but-invalid weight column must not silently
 * degrade to a no-kME run). When groups are known, an optional per-module direction table
 * (one signed eigengene→outcome correlation per group + one shared trait label) is offered.
 * Preview cells that look like formula-injection are badged — EXCEPT numeric (kME/kIM) columns,
 * whose legitimately-negative values would otherwise be false-flagged.
 */
export function ColumnMappingPanel({
  parsed,
  fileName,
  onConfirm,
  onCancel,
}: ColumnMappingPanelProps) {
  const [mapping, setMapping] = useState<ColumnMapping>(() => suggestMapping(parsed.headers));
  // Per-group eigengene→outcome correlation (as raw strings) + one shared trait label.
  const [dirByGroup, setDirByGroup] = useState<Record<string, string>>({});
  const [traitLabel, setTraitLabel] = useState("");

  const setTarget = (target: MappingTarget, value: string) => {
    setMapping((prev) => ({ ...prev, [target]: value === UNMAPPED ? undefined : value }));
  };

  const hasDataRows = parsed.rows.length > 0;

  const built = useMemo(
    () => (mapping.analyte ? buildAnalytes(parsed.rows, mapping) : null),
    [parsed.rows, mapping],
  );

  // Mapped-but-invalid weight cells across the FULL panel (not just the preview) — the Continue
  // gate reads these so a bad kME/kIM column can't slip through as a clean no-kME run.
  const invalidCounts = useMemo(
    () => countInvalidWeightCells(parsed.rows, mapping),
    [parsed.rows, mapping],
  );
  const hasInvalidWeights = invalidCounts.kme > 0 || invalidCounts.kim > 0;

  // Numeric-mapped columns (kME/kIM) are excluded from the formula-injection badge so a signed
  // (leading-"-") kME value isn't false-flagged.
  const numericHeaders = useMemo(() => {
    const s = new Set<string>();
    if (mapping.kme) s.add(mapping.kme);
    if (mapping.kim) s.add(mapping.kim);
    return s;
  }, [mapping.kme, mapping.kim]);

  const groups = built ? distinctGroups(built.analytes) : [];

  const canConfirm = Boolean(mapping.analyte) && hasDataRows && !hasInvalidWeights;

  const renderSelect = (target: MappingTarget, label: string, required?: boolean) => (
    <div className="flex flex-col gap-1">
      <label className="text-xs font-medium text-muted-foreground">
        {label}
        {required ? " *" : " (optional)"}
      </label>
      <Select value={mapping[target] ?? UNMAPPED} onValueChange={(v) => setTarget(target, v)}>
        <SelectTrigger className="h-8 text-sm" data-testid={`map-${target}`}>
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

  const handleConfirm = () => {
    if (!built) return;
    const directions = buildModuleDirections(dirByGroup, traitLabel);
    onConfirm(built.analytes, directions);
  };

  return (
    <div className="max-w-3xl mx-auto rounded-md border p-3 space-y-3" data-testid="mapping-panel">
      <div className="flex items-center justify-between">
        <p className="text-sm font-medium">Map columns — {fileName}</p>
        <span className="text-xs text-muted-foreground">{parsed.rows.length} rows</span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {renderSelect("analyte", "Analyte name", true)}
        {renderSelect("group", "Group / Category")}
        {renderSelect("type", "Analyte type")}
      </div>

      {/* Signed weights (Axis A) — visually distinct optional mapping row. */}
      <div className="rounded border border-dashed p-2 space-y-2">
        <p className="text-xs font-medium text-muted-foreground">Signed weights (optional)</p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {renderSelect("kme", "kME (module membership, −1…1)")}
          {renderSelect("kim", "kIM (kWithin, ≥ 0)")}
        </div>
        {hasInvalidWeights && (
          <p className="text-xs text-amber-600" data-testid="weight-invalid-warning">
            ⚠{" "}
            {invalidCounts.kme > 0 &&
              `${invalidCounts.kme} kME value(s) out of range or non-numeric`}
            {invalidCounts.kme > 0 && invalidCounts.kim > 0 && "; "}
            {invalidCounts.kim > 0 && `${invalidCounts.kim} kIM value(s) negative or non-numeric`}
            {" "}— the server will reject this upload. Fix the column mapping to continue.
          </p>
        )}
      </div>

      {/* Preview (first ~5 rows). Formula-injection cells are badged (numeric columns excluded). */}
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
                    {!numericHeaders.has(h) && isFormulaInjection(row[h]) ? (
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
      {built && (
        <p className="text-xs text-muted-foreground">
          {built.rowsRead} rows read, {built.rowsKept} analytes kept.
        </p>
      )}

      {/* Per-module direction table (optional). Only shown once groups + a kME column are present,
          since the direction is only meaningful for signed modules. Group is bound to the panel's
          distinct groups (a typo is structurally impossible); one shared trait label for the run. */}
      {mapping.kme && groups.length > 0 && (
        <div className="rounded border p-2 space-y-2" data-testid="direction-table">
          <p className="text-xs font-medium text-muted-foreground">
            Module → outcome direction (optional)
          </p>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-muted-foreground">Outcome / trait label</label>
            <input
              type="text"
              className="h-8 rounded border px-2 text-sm"
              placeholder="e.g. frailty"
              value={traitLabel}
              data-testid="direction-trait-label"
              onChange={(e) => setTraitLabel(e.target.value)}
            />
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {groups.map((g) => (
              <div key={g} className="flex items-center gap-2">
                <span className="text-xs w-28 truncate" title={g}>
                  {g}
                </span>
                <input
                  type="number"
                  step="any"
                  min={-1}
                  max={1}
                  className="h-8 w-24 rounded border px-2 text-sm"
                  placeholder="corr −1…1"
                  value={dirByGroup[g] ?? ""}
                  data-testid={`direction-corr-${g}`}
                  onChange={(e) =>
                    setDirByGroup((prev) => ({ ...prev, [g]: e.target.value }))
                  }
                />
              </div>
            ))}
          </div>
          <p className="text-[11px] text-muted-foreground">
            Leave all blank to run without a direction. Required to compute the sign-inversion metric.
          </p>
        </div>
      )}

      <div className="flex justify-end gap-2">
        <Button variant="ghost" size="sm" onClick={onCancel} data-testid="mapping-cancel">
          Cancel
        </Button>
        <Button
          size="sm"
          disabled={!canConfirm}
          onClick={handleConfirm}
          data-testid="mapping-confirm"
        >
          Continue
        </Button>
      </div>
    </div>
  );
}
