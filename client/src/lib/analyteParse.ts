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
  // Signed-weight data spine (Axis A): optional numeric columns. kME (module-eigengene
  // correlation, [-1,1]) and kIM (raw intramodular connectivity kWithin, unbounded ≥ 0).
  kme?: string;
  kim?: string;
}

export interface StructuredAnalyte {
  name: string;
  group?: string;
  type?: AnalyteType;
  // Numeric signed weights (Axis A); undefined = not supplied for this row.
  kme?: number;
  kim?: number;
}

/** One per-module eigengene→outcome direction sent alongside the panel (Axis A). */
export interface ModuleDirectionInput {
  group: string;
  eigengene_trait_correlation: number;
  trait_label: string;
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
// Signed-weight headers (Axis A) — unambiguous only (R6). kME = module-eigengene correlation;
// kIM = raw intramodular connectivity (kWithin). Ambiguous headers like `value`/`weight` are
// intentionally NOT matched.
const KME_HEADER_PATTERNS = [
  /^k[._ ]?me$/i,
  /^signed[_ ]?kme$/i,
  /^module[_ ]?membership$/i,
];
const KIM_HEADER_PATTERNS = [
  /^k[._ ]?im$/i,
  /^k[._ ]?within$/i,
  /^intramodular[_ ]?connectivity$/i,
];

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
    } else if (!mapping.kme && KME_HEADER_PATTERNS.some((p) => p.test(header))) {
      mapping.kme = header;
    } else if (!mapping.kim && KIM_HEADER_PATTERNS.some((p) => p.test(header))) {
      mapping.kim = header;
    }
  }
  return mapping;
}

/** Result of coercing one numeric weight cell (mirrors the backend three-way outcome). */
export type WeightCell =
  | { status: "absent" }
  | { status: "value"; value: number }
  | { status: "invalid" };

function coerceFinite(raw: string | undefined): WeightCell {
  if (raw === undefined) return { status: "absent" };
  const s = raw.trim();
  if (s === "") return { status: "absent" };
  const n = Number(s);
  if (!Number.isFinite(n)) return { status: "invalid" };
  return { status: "value", value: n };
}

// kME / eigengene-trait correlations are bounded to [-1, 1], but a serialized WGCNA export can
// round to e.g. 1.0000002. Mirror the backend (analyte_ingest.py `_KME_EPSILON`): tolerate
// ±epsilon on the bound and clamp a within-epsilon value back inside, rather than rejecting a
// rounded export the shared ingest gate would ACCEPT.
export const KME_EPSILON = 1e-6;

/** Clamp a correlation to [-1, 1] (used after the epsilon-tolerant bound check passes). */
function clampCorrelation(v: number): number {
  return Math.max(-1, Math.min(1, v));
}

/**
 * Coerce a kME cell. kME is a correlation → valid only when finite AND within [-1, 1] up to
 * ±KME_EPSILON (mirrors the backend's epsilon-tolerant-clamp rule). A within-epsilon value is
 * clamped back inside so the payload sent to the backend is already in-bounds; only values beyond
 * the epsilon are rejected. Blank = absent (member simply unweighted).
 */
export function parseKmeCell(raw: string | undefined): WeightCell {
  const c = coerceFinite(raw);
  if (c.status !== "value") return c;
  if (c.value < -1 - KME_EPSILON || c.value > 1 + KME_EPSILON) return { status: "invalid" };
  return { status: "value", value: clampCorrelation(c.value) };
}

/**
 * Coerce a kIM cell. kIM is raw intramodular connectivity (kWithin): unbounded, non-negative — do
 * NOT bound to [-1, 1]. Valid only when finite AND `>= 0`. Blank = absent.
 */
export function parseKimCell(raw: string | undefined): WeightCell {
  const c = coerceFinite(raw);
  if (c.status !== "value") return c;
  if (c.value < 0) return { status: "invalid" };
  return c;
}

/**
 * Count mapped-but-invalid kME/kIM cells so the UI can badge them and BLOCK Continue (the client
 * twin of "reject don't clip"): a mapped kME column that is nearly all invalid must not be
 * indistinguishable from a clean no-kME run. Absent (blank) cells are not invalid.
 */
export function countInvalidWeightCells(
  rows: Record<string, string>[],
  mapping: ColumnMapping,
): { kme: number; kim: number } {
  let kme = 0;
  let kim = 0;
  for (const row of rows) {
    // Only count rows with an analyte name (the panel's real rows).
    if (mapping.analyte && !(row[mapping.analyte] ?? "").trim()) continue;
    if (mapping.kme && parseKmeCell(row[mapping.kme]).status === "invalid") kme += 1;
    if (mapping.kim && parseKimCell(row[mapping.kim]).status === "invalid") kim += 1;
  }
  return { kme, kim };
}

/**
 * Build the outgoing per-module direction rows from a group→correlation map + one shared trait
 * label (WGCNA runs study a single outcome). Blank/non-numeric correlations are skipped; an empty
 * trait label or all-blank correlations yield [] ("no direction supplied", R3).
 */
export function buildModuleDirections(
  correlationByGroup: Record<string, string>,
  traitLabel: string,
): ModuleDirectionInput[] {
  const label = traitLabel.trim();
  if (!label) return [];
  const out: ModuleDirectionInput[] = [];
  for (const [group, raw] of Object.entries(correlationByGroup)) {
    const c = coerceFinite(raw);
    if (c.status !== "value") continue;
    // Clamp a within-epsilon rounded export back inside [-1, 1] (mirrors the backend) so the
    // payload is already in-bounds. A value beyond the epsilon is left as-is here; it is flagged
    // by countInvalidDirections and the Continue gate blocks the upload before this is sent.
    const corr =
      c.value >= -1 - KME_EPSILON && c.value <= 1 + KME_EPSILON
        ? clampCorrelation(c.value)
        : c.value;
    out.push({ group, eigengene_trait_correlation: corr, trait_label: label });
  }
  return out;
}

/**
 * Count per-module direction correlations that are finite but OUTSIDE [-1, 1] (a correlation, same
 * bound as kME). The backend REJECTS these; because the WS layer clears the staged upload the
 * instant it sends, a server-side rejection loses the upload with no recovery. The Continue gate
 * reads this count so an out-of-range direction is caught client-side instead. Blank / non-numeric
 * entries are NOT counted (buildModuleDirections drops them; they are simply "no direction"). When
 * `groups` is given, only those groups are considered so a stale value for a no-longer-shown group
 * cannot invisibly block Continue.
 */
export function countInvalidDirections(
  correlationByGroup: Record<string, string>,
  groups?: string[],
): number {
  const keys = groups ?? Object.keys(correlationByGroup);
  let n = 0;
  for (const g of keys) {
    const c = coerceFinite(correlationByGroup[g]);
    // Epsilon-tolerant bound (mirrors backend): a within-epsilon rounded export is VALID (and gets
    // clamped in buildModuleDirections); only values beyond the epsilon are counted as invalid.
    if (c.status === "value" && (c.value < -1 - KME_EPSILON || c.value > 1 + KME_EPSILON)) n += 1;
  }
  return n;
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
  /**
   * Count of (name, group) rows whose kME/kIM DISAGREE with an earlier row for the same key.
   * The display dedup collapses these to the first value, so the backend — which receives the
   * already-deduped panel — can never see the conflict its `validate_and_normalize` is designed to
   * REJECT. Detected here so the UI can block Continue instead of silently coercing to row one.
   */
  weightConflicts: number;
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
    return { analytes: [], rowsRead: rows.length, rowsKept: 0, weightConflicts: 0 };
  }
  const analytes: StructuredAnalyte[] = [];
  const seen = new Set<string>();
  // First-seen VALID weight pair per (name, group) key, for conflict detection independent of the
  // display dedup below (mirrors the backend, which admits a member to the spine via a valid kME).
  const weightByKey = new Map<string, { kme: number; kim: number | undefined }>();
  let rowsRead = 0;
  let weightConflicts = 0;

  for (const row of rows) {
    const name = (row[mapping.analyte] ?? "").trim();
    if (!name) continue; // R9: skip empty-name rows
    rowsRead += 1;

    const group = mapping.group ? (row[mapping.group] ?? "").trim() || undefined : undefined;
    const type = mapping.type ? normalizeType(row[mapping.type]) : undefined;

    // Parse signed weights once (used for both conflict detection and the display attach). Only a
    // VALID (finite, in-range) cell yields a number; invalid/absent stays undefined.
    const kmeCell = mapping.kme ? parseKmeCell(row[mapping.kme]) : undefined;
    const kimCell = mapping.kim ? parseKimCell(row[mapping.kim]) : undefined;
    const kmeVal = kmeCell?.status === "value" ? kmeCell.value : undefined;
    const kimVal = kimCell?.status === "value" ? kimCell.value : undefined;

    const dedupKey = `${name.toLowerCase()} ${group ?? ""}`;
    // Conflict detection (mirrors backend validate_and_normalize): a member enters the signed spine
    // only via a valid kME. Two rows for the same (name, group) that BOTH carry a valid kME but
    // disagree on (kME, kIM) are a mis-map/duplicate footgun the backend REJECTS. Detected here,
    // BEFORE the display collapse, so a conflicting file cannot be silently coerced to the first
    // value and slipped past the (now blind) server-side check.
    if (kmeVal !== undefined) {
      const prior = weightByKey.get(dedupKey);
      if (prior === undefined) {
        weightByKey.set(dedupKey, { kme: kmeVal, kim: kimVal });
      } else if (prior.kme !== kmeVal || prior.kim !== kimVal) {
        weightConflicts += 1;
      }
    }

    if (seen.has(dedupKey)) continue; // (name, group) collapse
    seen.add(dedupKey);

    const analyte: StructuredAnalyte = { name };
    if (group) analyte.group = group;
    if (type) analyte.type = type;
    // Signed weights (Axis A): attach only VALID cells. An invalid (out-of-range / non-numeric)
    // cell is left unset here — never silently coerced — and surfaced separately via
    // countInvalidWeightCells so the UI badges it and blocks Continue.
    if (kmeVal !== undefined) analyte.kme = kmeVal;
    if (kimVal !== undefined) analyte.kim = kimVal;
    analytes.push(analyte);
  }

  return { analytes, rowsRead, rowsKept: analytes.length, weightConflicts };
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
