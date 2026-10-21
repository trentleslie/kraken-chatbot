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
  isFormulaInjection,
  suggestMapping,
  type ColumnMapping,
  type MappingTarget,
  type ParsedFile,
  type StructuredAnalyte,
} from "@/lib/analyteParse";

interface ColumnMappingPanelProps {
  parsed: ParsedFile;
  fileName: string;
  onConfirm: (analytes: StructuredAnalyte[]) => void;
  onCancel: () => void;
}

const UNMAPPED = "__none__";

/**
 * Column-mapping panel (Unit 6, R5–R8). Radix Select dropdowns assign file columns to three
 * targets (analyte required, group/type optional). Auto-suggests unambiguous mappings; Confirm is
 * disabled until the analyte column is mapped. Preview cells that look like formula-injection are
 * badged and rendered as plain text nodes (never innerHTML).
 */
export function ColumnMappingPanel({
  parsed,
  fileName,
  onConfirm,
  onCancel,
}: ColumnMappingPanelProps) {
  const [mapping, setMapping] = useState<ColumnMapping>(() => suggestMapping(parsed.headers));

  const setTarget = (target: MappingTarget, value: string) => {
    setMapping((prev) => ({ ...prev, [target]: value === UNMAPPED ? undefined : value }));
  };

  const hasDataRows = parsed.rows.length > 0;
  const canConfirm = Boolean(mapping.analyte) && hasDataRows;

  const built = useMemo(
    () => (mapping.analyte ? buildAnalytes(parsed.rows, mapping) : null),
    [parsed.rows, mapping],
  );

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

      {!hasDataRows && (
        <p className="text-xs text-destructive">No data rows found.</p>
      )}
      {built && (
        <p className="text-xs text-muted-foreground">
          {built.rowsRead} rows read, {built.rowsKept} analytes kept.
        </p>
      )}

      <div className="flex justify-end gap-2">
        <Button variant="ghost" size="sm" onClick={onCancel} data-testid="mapping-cancel">
          Cancel
        </Button>
        <Button
          size="sm"
          disabled={!canConfirm}
          onClick={() => built && onConfirm(built.analytes)}
          data-testid="mapping-confirm"
        >
          Continue
        </Button>
      </div>
    </div>
  );
}
