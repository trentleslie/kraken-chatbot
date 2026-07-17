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

export type MappingTarget = "analyte" | "group" | "type" | "kme" | "kim";

/** User's column→target assignment. Values are header names; undefined = unmapped. */
export interface ColumnMapping {
  analyte?: string;
  group?: string;
  type?: string;
  /** Signed module-eigengene correlation column (Axis A). */
  kme?: string;
  /** Raw intramodular connectivity (kWithin) column (Axis A). */
  kim?: string;
}

export interface StructuredAnalyte {
  name: string;
  group?: string;
  type?: AnalyteType;
  /**
   * Signed kME ∈ [-1, 1]. A `number` when the mapped cell parsed cleanly; the raw string is
   * preserved (verbatim) for a present-but-non-numeric cell so the backend R19 gate can REJECT
   * the bad weight instead of the client silently dropping it (which would unweight the member
   * and change module-spine coverage without surfacing the bad upload). Absent when unmapped/blank.
   */
  kme?: number | string;
  /** Raw kIM (kWithin) >= 0. `number` when parsed; raw string preserved for backend rejection. */
  kim?: number | string;
}

/** A per-module eigengene→outcome direction row (Axis A). Optional; backend authoritative. */
export interface ModuleDirectionInput {
  group: string;
  /**
   * `number` when the mapped correlation cell parsed; the raw string is preserved for a
   * present-but-non-numeric (or blank) cell so the backend R19 gate REJECTS the malformed
   * direction instead of it silently vanishing from the module spine.
   */
  eigengene_trait_correlation: number | string;
  trait_label: string;
}

/** Column assignment for the (separate) ME-trait direction table. */
export interface DirectionMapping {
  group?: string;
  correlation?: string;
  trait?: string;
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

  const dataRows = matrix.slice(1);
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
// Unambiguous kME/kIM headers (Axis A). kME variants start with "kme"; kIM variants start with
// "kim" or spell out "kwithin"/"k_within" (raw intramodular connectivity). A bare "value" or
// "correlation" header is intentionally NOT matched (R6 unambiguous-only).
const KME_HEADER_PATTERNS = [/^k[_ ]?me\b/i, /^kme/i];
const KIM_HEADER_PATTERNS = [/^k[_ ]?im\b/i, /^kim/i, /^k[_ ]?within/i];

/**
 * Auto-suggest a column mapping from header names, restricted to UNAMBIGUOUS matches (R6).
 * Ambiguous headers (`name`, `type`, `class`, `value`) are intentionally left unset rather than
 * risk a silent mis-map. First matching header wins per target.
 */
export function suggestMapping(headers: string[]): ColumnMapping {
  const mapping: ColumnMapping = {};
  for (const header of headers) {
    if (!mapping.analyte && ANALYTE_HEADER_PATTERNS.some((p) => p.test(header))) {
      mapping.analyte = header;
    } else if (!mapping.kme && KME_HEADER_PATTERNS.some((p) => p.test(header))) {
      mapping.kme = header;
    } else if (!mapping.kim && KIM_HEADER_PATTERNS.some((p) => p.test(header))) {
      mapping.kim = header;
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

/**
 * Classify a mapped weight/correlation cell, mirroring the backend R19 `_coerce_weight` states:
 *   - `absent`  — missing/blank cell (member simply carries no weight);
 *   - `value`   — a finite number was parsed;
 *   - `invalid` — present but non-numeric/non-finite (the raw string is carried through so the
 *                 backend R19 gate — authoritative — can REJECT it; the client must NOT drop it).
 */
type WeightCell =
  | { status: "absent" }
  | { status: "value"; value: number }
  | { status: "invalid"; raw: string };

function coerceWeightCell(raw: string | undefined): WeightCell {
  if (raw === undefined) return { status: "absent" };
  const s = raw.trim();
  if (s === "") return { status: "absent" };
  const n = Number(s);
  return Number.isFinite(n) ? { status: "value", value: n } : { status: "invalid", raw: s };
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
    // Signed weights (Axis A): when the column is mapped, a numeric cell rides through as a
    // number and a present-but-non-numeric cell is preserved VERBATIM so the backend R19 gate
    // rejects the malformed weight. Dropping the invalid cell here would silently unweight the
    // member and change module-spine coverage without surfacing the bad upload. Blank → omitted.
    if (mapping.kme) {
      const cell = coerceWeightCell(row[mapping.kme]);
      if (cell.status === "value") analyte.kme = cell.value;
      else if (cell.status === "invalid") analyte.kme = cell.raw;
    }
    if (mapping.kim) {
      const cell = coerceWeightCell(row[mapping.kim]);
      if (cell.status === "value") analyte.kim = cell.value;
      else if (cell.status === "invalid") analyte.kim = cell.raw;
    }
    analytes.push(analyte);
  }

  return { analytes, rowsRead, rowsKept: analytes.length };
}

/**
 * Build per-module direction rows from a (separately-exported) ME-trait table (Axis A). Returns
 * [] unless all three columns (group, correlation, trait) are mapped. Every row with ANY content
 * in a mapped column is preserved VERBATIM (a numeric correlation as a number, otherwise the raw
 * string) so the backend R19 gate re-validates + range-checks and REJECTS a malformed direction —
 * dropping it here would let a typo'd correlation or blank trait silently vanish from the module
 * spine. Only fully-empty rows (blank group, correlation, and trait) are skipped as filler.
 */
export function buildModuleDirections(
  rows: Record<string, string>[],
  mapping: DirectionMapping,
): ModuleDirectionInput[] {
  if (!mapping.group || !mapping.correlation || !mapping.trait) return [];
  const out: ModuleDirectionInput[] = [];
  for (const row of rows) {
    const group = (row[mapping.group] ?? "").trim();
    const trait = (row[mapping.trait] ?? "").trim();
    const rawCorr = (row[mapping.correlation] ?? "").trim();
    if (!group && !rawCorr && !trait) continue; // fully-empty row = filler, not a direction
    const corr = coerceWeightCell(rawCorr);
    out.push({
      group,
      // number when parsed; otherwise the raw cell (or "" for a blank) so the backend rejects it.
      eigengene_trait_correlation: corr.status === "value" ? corr.value : rawCorr,
      trait_label: trait,
    });
  }
  return out;
}

/** Distinct group values present in a built analyte list, in first-seen order. */
export function distinctGroups(analytes: StructuredAnalyte[]): string[] {
  const groups: string[] = [];
  const seen = new Set<string>();
  for (const a of analytes) {
    if (a.group && !seen.has(a.group)) {
      seen.add(a.group);
      groups.push(a.group);
    }
  }
  return groups;
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
  const sel = new Set(selectedGroups);
  return analytes.filter((a) => a.group !== undefined && sel.has(a.group));
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
