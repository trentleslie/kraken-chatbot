/**
 * Pure, dependency-light analyte-file parsing + mapping logic (Unit 4).
 *
 * No React imports — all parse/dedup/filter logic lives here so the UI components (Unit 6) stay
 * thin glue and this module is unit-testable under vitest. Handles CSV/TSV via papaparse. Excel
 * (.xlsx) support is deferred for V1 (SheetJS's decompression-bomb/CVE posture failed the plan's
 * blocking security gate; the maintained build is off the public npm registry) — .xlsx files
 * return a typed "unsupported" error rather than silently mis-parsing.
 */

import Papa from "papaparse";

// Recognized analyte types (lowercase, mirrors the backend biolink_class_for + RECOGNIZED_TYPES).
// `disease` is deliberately excluded (a disease type inverts Direct-KG analysis).
export const RECOGNIZED_TYPES = ["metabolite", "protein", "gene"] as const;
export type AnalyteType = (typeof RECOGNIZED_TYPES)[number];

// Client-side resource caps (defense-in-depth; the backend R19 caps are authoritative).
export const MAX_FILE_BYTES = 10 * 1024 * 1024; // ~10 MB
export const MAX_ROWS = 10_000;

export type MappingTarget = "analyte" | "group" | "type";

/** User's column→target assignment. Values are header names; undefined = unmapped. */
export interface ColumnMapping {
  analyte?: string;
  group?: string;
  type?: string;
}

export interface StructuredAnalyte {
  name: string;
  group?: string;
  type?: AnalyteType;
}

export interface ParsedFile {
  headers: string[];
  /** Up to the first ~5 data rows, each as a header→cell map, for the mapping preview. */
  previewRows: Record<string, string>[];
  /** All data rows as header→cell maps. */
  rows: Record<string, string>[];
}

export type ParseResult =
  | { ok: true; data: ParsedFile }
  | { ok: false; error: string };

const PREVIEW_ROW_COUNT = 5;

/** Detect delimiter from filename/content: TSV for .tsv/.tab, else let papaparse auto-detect. */
function delimiterForName(fileName: string): string | undefined {
  const lower = fileName.toLowerCase();
  if (lower.endsWith(".tsv") || lower.endsWith(".tab")) return "\t";
  return undefined; // undefined → papaparse auto-detect (handles CSV + guesses others)
}

/**
 * Parse raw CSV/TSV text into headers + rows. Pure (no File/DOM), so it is directly testable.
 * Returns a typed error result (never throws) for empty / header-less / header-only inputs.
 */
export function parseDelimitedText(text: string, fileName: string): ParseResult {
  if (!text || !text.trim()) {
    return { ok: false, error: "The file is empty." };
  }

  const parsed = Papa.parse<string[]>(text, {
    delimiter: delimiterForName(fileName) ?? "",
    skipEmptyLines: "greedy",
  });

  const matrix = (parsed.data as unknown as string[][]).filter(
    (row) => Array.isArray(row) && row.some((c) => String(c ?? "").trim() !== ""),
  );

  if (matrix.length === 0) {
    return { ok: false, error: "No rows found in the file." };
  }

  const headers = matrix[0].map((h) => String(h ?? "").trim());
  if (headers.length === 0 || headers.every((h) => h === "")) {
    return { ok: false, error: "The file has no usable column headers." };
  }
  if (headers.some((h) => h === "")) {
    return { ok: false, error: "The header row has a blank column name." };
  }
  const seen = new Set<string>();
  for (const h of headers) {
    const key = h.toLowerCase();
    if (seen.has(key)) {
      return { ok: false, error: `Duplicate column header: "${h}".` };
    }
    seen.add(key);
  }

  // Surface real parse errors (e.g. unbalanced quotes). Papaparse emits UndetectableDelimiter for
  // harmless single-column files, so that one is not an error.
  const parseError = parsed.errors.find((e) => e.code !== "UndetectableDelimiter");
  if (parseError) {
    const where = parseError.row !== undefined ? ` near row ${parseError.row + 1}` : "";
    return { ok: false, error: `Could not parse the file${where}: ${parseError.message}.` };
  }

  const dataRows = matrix.slice(1);
  // A row with more non-empty cells than headers usually means a stray delimiter; dropping the
  // extra cells would silently shift which values become analyte names or groups.
  const wideRow = dataRows.findIndex((cells) =>
    cells.slice(headers.length).some((c) => String(c ?? "").trim() !== ""),
  );
  if (wideRow !== -1) {
    return {
      ok: false,
      error: `Row ${wideRow + 2} has more values than there are column headers (${headers.length}). Check for an unquoted comma or tab.`,
    };
  }
  if (dataRows.length === 0) {
    return { ok: false, error: "No data rows found (the file has only a header)." };
  }
  if (dataRows.length > MAX_ROWS) {
    return {
      ok: false,
      error: `The file has ${dataRows.length} rows, exceeding the ${MAX_ROWS}-row limit.`,
    };
  }

  const rows: Record<string, string>[] = dataRows.map((cells) => {
    const record: Record<string, string> = {};
    headers.forEach((h, i) => {
      record[h] = String(cells[i] ?? "").trim();
    });
    return record;
  });

  return {
    ok: true,
    data: { headers, previewRows: rows.slice(0, PREVIEW_ROW_COUNT), rows },
  };
}

/**
 * Parse a browser File (CSV/TSV/Excel). Enforces the pre-parse size cap, rejects .xlsx as
 * unsupported-in-this-build (deferred), and delegates CSV/TSV to parseDelimitedText.
 */
export async function parseFile(file: File): Promise<ParseResult> {
  if (file.size > MAX_FILE_BYTES) {
    return {
      ok: false,
      error: `File is too large (limit ${Math.round(MAX_FILE_BYTES / 1024 / 1024)} MB).`,
    };
  }
  const lower = file.name.toLowerCase();
  if (lower.endsWith(".xlsx") || lower.endsWith(".xls")) {
    return {
      ok: false,
      error:
        "Excel files aren't supported yet — please export the sheet as CSV or TSV and upload that.",
    };
  }
  if (!lower.endsWith(".csv") && !lower.endsWith(".tsv") && !lower.endsWith(".tab")) {
    return { ok: false, error: "Unsupported file type. Upload a .csv or .tsv file." };
  }
  const text = await file.text();
  return parseDelimitedText(text, file.name);
}

const ANALYTE_HEADER_PATTERNS = [
  /^analyte$/i,
  /^metabolite$/i,
  /^protein$/i,
  /^gene$/i,
  /^feature$/i,
];
const GROUP_HEADER_PATTERNS = [/^module$/i, /^group$/i, /^category$/i, /^cluster$/i];
const TYPE_HEADER_PATTERNS = [/^analyte[_ ]?type$/i, /^omic$/i, /^omics$/i];

/**
 * Auto-suggest a column mapping from header names, restricted to UNAMBIGUOUS matches (R6).
 * Ambiguous headers (`name`, `type`, `class`) are intentionally left unset rather than risk a
 * silent mis-map. First matching header wins per target.
 */
export function suggestMapping(headers: string[]): ColumnMapping {
  const mapping: ColumnMapping = {};
  for (const header of headers) {
    if (!mapping.analyte && ANALYTE_HEADER_PATTERNS.some((p) => p.test(header))) {
      mapping.analyte = header;
    } else if (!mapping.group && GROUP_HEADER_PATTERNS.some((p) => p.test(header))) {
      mapping.group = header;
    } else if (!mapping.type && TYPE_HEADER_PATTERNS.some((p) => p.test(header))) {
      mapping.type = header;
    }
  }
  return mapping;
}

function normalizeType(raw: string | undefined): AnalyteType | undefined {
  if (!raw) return undefined;
  const lower = raw.trim().toLowerCase();
  return (RECOGNIZED_TYPES as readonly string[]).includes(lower)
    ? (lower as AnalyteType)
    : undefined;
}

export interface BuildResult {
  /** Deduped analytes by (lower(name), group) for DISPLAY (cross-group entries kept as two). */
  analytes: StructuredAnalyte[];
  rowsRead: number;
  rowsKept: number;
}

/**
 * Build the structured analyte list from mapped columns (R8–R11).
 * - trims all mapped values, skips rows with an empty analyte name (R9);
 * - dedups by (lower(name), group) — a name in two groups stays as two entries (R10);
 * - unknown type → undefined (R11); stores the analyte name VERBATIM (post-trim).
 */
export function buildAnalytes(
  rows: Record<string, string>[],
  mapping: ColumnMapping,
): BuildResult {
  if (!mapping.analyte) {
    return { analytes: [], rowsRead: rows.length, rowsKept: 0 };
  }
  const analytes: StructuredAnalyte[] = [];
  const seen = new Set<string>();
  let rowsRead = 0;

  for (const row of rows) {
    const name = (row[mapping.analyte] ?? "").trim();
    if (!name) continue; // R9: skip empty-name rows
    rowsRead += 1;

    const group = mapping.group ? (row[mapping.group] ?? "").trim() || undefined : undefined;
    const type = mapping.type ? normalizeType(row[mapping.type]) : undefined;

    const dedupKey = `${name.toLowerCase()} ${group ?? ""}`;
    if (seen.has(dedupKey)) continue; // (name, group) collapse
    seen.add(dedupKey);

    const analyte: StructuredAnalyte = { name };
    if (group) analyte.group = group;
    if (type) analyte.type = type;
    analytes.push(analyte);
  }

  return { analytes, rowsRead, rowsKept: analytes.length };
}

/** Distinct group values present in a built analyte list, in first-seen order. */
/** Group identity used by the server's selection filter (analyte_ingest.py): trim + lowercase. */
export function groupKey(group: string): string {
  return group.trim().toLowerCase();
}

/** Distinct groups by server identity, labelled by their first-seen spelling. */
export function distinctGroups(analytes: StructuredAnalyte[]): string[] {
  const groups: string[] = [];
  const seen = new Set<string>();
  for (const a of analytes) {
    if (!a.group || !a.group.trim()) continue;
    const key = groupKey(a.group);
    if (!seen.has(key)) {
      seen.add(key);
      groups.push(a.group);
    }
  }
  return groups;
}

/** Row count per group, keyed by groupKey. A Map, so labels like "constructor" are safe keys. */
export function groupCounts(analytes: StructuredAnalyte[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const a of analytes) {
    if (!a.group || !a.group.trim()) continue;
    const key = groupKey(a.group);
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return counts;
}

/**
 * The selection to send to the server. Selecting every group means "run all rows", so it is sent
 * as an empty selection; otherwise the server would also drop rows with a blank group cell.
 */
export function effectiveGroupSelection(
  analytes: StructuredAnalyte[],
  selectedGroups: string[],
): string[] {
  const selected = new Set(selectedGroups.map(groupKey));
  const all = distinctGroups(analytes);
  return all.every((g) => selected.has(groupKey(g))) ? [] : selectedGroups;
}

/**
 * Filter analytes to a selected group set for PREVIEW/COUNTS (the full panel is still what gets
 * sent — the backend applies the authoritative filter). Empty selection = keep all.
 */
export function applyGroupFilter(
  analytes: StructuredAnalyte[],
  selectedGroups: string[],
): StructuredAnalyte[] {
  if (selectedGroups.length === 0) return analytes;
  const sel = new Set(selectedGroups.map(groupKey));
  return analytes.filter((a) => a.group !== undefined && sel.has(groupKey(a.group)));
}

/** Distinct-name count of a (possibly group-filtered) set — the R18 soft-warn basis. */
export function distinctNameCount(analytes: StructuredAnalyte[]): number {
  return new Set(analytes.map((a) => a.name.toLowerCase())).size;
}

const FORMULA_PREFIXES = ["=", "+", "-", "@"];

/**
 * True if a cell value looks like a spreadsheet formula-injection payload (R4/Unit 4 flag).
 * The UI badges these and renders them as text nodes (never dangerouslySetInnerHTML).
 */
export function isFormulaInjection(value: string | undefined): boolean {
  if (!value) return false;
  return FORMULA_PREFIXES.includes(value.trimStart().charAt(0));
}

/** Per-field limit enforced by the server (ANALYTE_FIELD_MAX_LEN). */
export const ANALYTE_FIELD_MAX_LEN = 512;
// C0/C1 control characters (Unicode category Cc), matching the server's check.
const CONTROL_CHARS = /[\u0000-\u001F\u007F-\u009F]/;

/**
 * Mirror of the server's per-field validation (analyte_ingest.py), so a panel it would reject is
 * caught before sending and the staged upload is kept. Returns an error message, or null if valid.
 */
export function validatePanelFields(analytes: StructuredAnalyte[]): string | null {
  for (let i = 0; i < analytes.length; i++) {
    const a = analytes[i];
    for (const [label, raw] of [["name", a.name], ["group", a.group], ["type", a.type]] as const) {
      const val = raw?.trim();
      if (!val) continue;
      // Count code points like Python's len(), not UTF-16 units (astral chars would count as 2).
      if (Array.from(val).length > ANALYTE_FIELD_MAX_LEN) {
        return `Analyte ${label} at row ${i + 1} is longer than ${ANALYTE_FIELD_MAX_LEN} characters.`;
      }
      if (CONTROL_CHARS.test(val)) {
        return `Analyte ${label} at row ${i + 1} contains control characters.`;
      }
    }
  }
  return null;
}
