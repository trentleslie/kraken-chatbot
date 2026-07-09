import { describe, it, expect } from "vitest";
import {
  parseDelimitedText,
  suggestMapping,
  buildAnalytes,
  distinctGroups,
  applyGroupFilter,
  distinctNameCount,
  isFormulaInjection,
  MAX_ROWS,
} from "./analyteParse";

describe("parseDelimitedText", () => {
  it("parses CSV with headers and rows", () => {
    const res = parseDelimitedText("analyte,module\nglucose,Brown\nIL6,Blue\n", "x.csv");
    expect(res.ok).toBe(true);
    if (!res.ok) return;
    expect(res.data.headers).toEqual(["analyte", "module"]);
    expect(res.data.rows).toEqual([
      { analyte: "glucose", module: "Brown" },
      { analyte: "IL6", module: "Blue" },
    ]);
  });

  it("detects TSV delimiter from filename", () => {
    const res = parseDelimitedText("analyte\tmodule\nglucose\tBrown\n", "x.tsv");
    expect(res.ok).toBe(true);
    if (!res.ok) return;
    expect(res.data.headers).toEqual(["analyte", "module"]);
    expect(res.data.rows[0]).toEqual({ analyte: "glucose", module: "Brown" });
  });

  it("limits preview to 5 rows but keeps all rows", () => {
    const lines = ["analyte"];
    for (let i = 0; i < 8; i++) lines.push(`m${i}`);
    const res = parseDelimitedText(lines.join("\n"), "x.csv");
    expect(res.ok).toBe(true);
    if (!res.ok) return;
    expect(res.data.previewRows).toHaveLength(5);
    expect(res.data.rows).toHaveLength(8);
  });

  it("errors on empty file", () => {
    expect(parseDelimitedText("", "x.csv")).toMatchObject({ ok: false });
    expect(parseDelimitedText("   \n  ", "x.csv")).toMatchObject({ ok: false });
  });

  it("errors on header-only (zero data rows)", () => {
    const res = parseDelimitedText("analyte,module\n", "x.csv");
    expect(res.ok).toBe(false);
  });

  it("errors on duplicate headers", () => {
    const res = parseDelimitedText("name,name\na,b\n", "x.csv");
    expect(res.ok).toBe(false);
  });

  it("errors on blank header cell", () => {
    const res = parseDelimitedText("analyte,,module\na,b,c\n", "x.csv");
    expect(res.ok).toBe(false);
  });

  it("errors when row count exceeds MAX_ROWS", () => {
    const lines = ["analyte"];
    for (let i = 0; i < MAX_ROWS + 1; i++) lines.push(`m${i}`);
    const res = parseDelimitedText(lines.join("\n"), "x.csv");
    expect(res.ok).toBe(false);
  });
});

describe("suggestMapping", () => {
  it("auto-maps unambiguous headers", () => {
    expect(suggestMapping(["metabolite", "module", "omic"])).toEqual({
      analyte: "metabolite",
      group: "module",
      type: "omic",
    });
  });

  it("leaves ambiguous headers unset", () => {
    // name/type/class are ambiguous — must NOT be auto-mapped (R6).
    expect(suggestMapping(["name", "type", "class"])).toEqual({});
  });

  it("maps analyte/group but leaves type unset when absent", () => {
    expect(suggestMapping(["gene", "cluster"])).toEqual({
      analyte: "gene",
      group: "cluster",
    });
  });
});

describe("buildAnalytes", () => {
  const rows = [
    { analyte: "glucose", module: "Brown", omic: "metabolite" },
    { analyte: "IL6", module: "Blue", omic: "protein" },
  ];

  it("builds the structured list from a mapping", () => {
    const res = buildAnalytes(rows, { analyte: "analyte", group: "module", type: "omic" });
    expect(res.analytes).toEqual([
      { name: "glucose", group: "Brown", type: "metabolite" },
      { name: "IL6", group: "Blue", type: "protein" },
    ]);
    expect(res.rowsRead).toBe(2);
    expect(res.rowsKept).toBe(2);
  });

  it("skips empty-name rows and trims", () => {
    const res = buildAnalytes(
      [
        { analyte: "  ", module: "Brown" },
        { analyte: "  glucose  ", module: "  Brown  " },
      ],
      { analyte: "analyte", group: "module" },
    );
    expect(res.analytes).toEqual([{ name: "glucose", group: "Brown" }]);
    expect(res.rowsRead).toBe(1);
  });

  it("keeps same name in two groups as two entries", () => {
    const res = buildAnalytes(
      [
        { analyte: "glucose", module: "Brown" },
        { analyte: "glucose", module: "Blue" },
      ],
      { analyte: "analyte", group: "module" },
    );
    expect(res.analytes).toEqual([
      { name: "glucose", group: "Brown" },
      { name: "glucose", group: "Blue" },
    ]);
  });

  it("collapses same name+group (case-insensitive), counts read vs kept", () => {
    const res = buildAnalytes(
      [
        { analyte: "glucose", module: "Brown" },
        { analyte: "GLUCOSE", module: "Brown" },
      ],
      { analyte: "analyte", group: "module" },
    );
    expect(res.analytes).toEqual([{ name: "glucose", group: "Brown" }]);
    expect(res.rowsRead).toBe(2);
    expect(res.rowsKept).toBe(1);
  });

  it("maps unknown type to undefined but keeps the row", () => {
    const res = buildAnalytes([{ analyte: "glucose", omic: "lipid" }], {
      analyte: "analyte",
      type: "omic",
    });
    expect(res.analytes).toEqual([{ name: "glucose" }]);
  });

  it("returns empty when analyte column unmapped", () => {
    const res = buildAnalytes(rows, { group: "module" });
    expect(res.analytes).toEqual([]);
  });
});

describe("group filtering + counts", () => {
  const analytes = [
    { name: "glucose", group: "Brown" },
    { name: "IL6", group: "Blue" },
    { name: "KIF6", group: "Blue" },
  ];

  it("distinctGroups returns first-seen order", () => {
    expect(distinctGroups(analytes)).toEqual(["Brown", "Blue"]);
  });

  it("applyGroupFilter keeps only chosen groups", () => {
    expect(applyGroupFilter(analytes, ["Blue"]).map((a) => a.name)).toEqual(["IL6", "KIF6"]);
  });

  it("applyGroupFilter with empty selection keeps all", () => {
    expect(applyGroupFilter(analytes, [])).toHaveLength(3);
  });

  it("distinctNameCount is case-insensitive", () => {
    expect(
      distinctNameCount([
        { name: "glucose", group: "Brown" },
        { name: "GLUCOSE", group: "Blue" },
      ]),
    ).toBe(1);
  });
});

describe("isFormulaInjection", () => {
  it("flags cells starting with formula prefixes", () => {
    expect(isFormulaInjection("=SUM(A1)")).toBe(true);
    expect(isFormulaInjection("+cmd")).toBe(true);
    expect(isFormulaInjection("-2")).toBe(true);
    expect(isFormulaInjection("@x")).toBe(true);
  });

  it("does not flag normal names", () => {
    expect(isFormulaInjection("glucose")).toBe(false);
    expect(isFormulaInjection("")).toBe(false);
    expect(isFormulaInjection(undefined)).toBe(false);
  });
});
