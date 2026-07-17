import { describe, it, expect } from "vitest";
import {
  parseDelimitedText,
  suggestMapping,
  suggestDirectionMapping,
  buildAnalytes,
  buildModuleDirections,
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

describe("suggestMapping — kME/kIM (Axis A)", () => {
  it("auto-maps kME and kIM headers", () => {
    expect(suggestMapping(["analyte", "module", "kME", "kIM"])).toEqual({
      analyte: "analyte",
      group: "module",
      kme: "kME",
      kim: "kIM",
    });
  });

  it("maps case/spacing variants of kME/kIM", () => {
    const m = suggestMapping(["feature", "kme_signed", "k_within"]);
    expect(m.kme).toBe("kme_signed");
    expect(m.kim).toBe("k_within");
  });

  it("does not auto-map an ambiguous 'value' header to kME", () => {
    expect(suggestMapping(["analyte", "value"]).kme).toBeUndefined();
  });
});

describe("buildAnalytes — kME/kIM (Axis A)", () => {
  it("parses numeric kME/kIM cells onto each analyte", () => {
    const res = buildAnalytes(
      [
        { analyte: "glucose", module: "Brown", kME: "0.82", kIM: "40" },
        { analyte: "IL6", module: "Brown", kME: "-0.4", kIM: "" },
      ],
      { analyte: "analyte", group: "module", kme: "kME", kim: "kIM" },
    );
    expect(res.analytes[0]).toEqual({ name: "glucose", group: "Brown", kme: 0.82, kim: 40 });
    // blank kIM → undefined (member without kIM)
    expect(res.analytes[1]).toEqual({ name: "IL6", group: "Brown", kme: -0.4 });
  });

  it("blank kME cell → no kme field on the analyte", () => {
    const res = buildAnalytes([{ analyte: "glucose", kME: "" }], {
      analyte: "analyte",
      kme: "kME",
    });
    expect(res.analytes[0]).toEqual({ name: "glucose" });
  });

  it("non-numeric kME cell → raw string preserved so the backend R19 gate can reject it", () => {
    const res = buildAnalytes([{ analyte: "glucose", kME: "high" }], {
      analyte: "analyte",
      kme: "kME",
    });
    // Preserved verbatim (NOT dropped): dropping it would silently unweight the member and
    // change module-spine coverage instead of surfacing the bad upload.
    expect(res.analytes[0].kme).toBe("high");
  });

  it("non-numeric kIM cell → raw string preserved for backend rejection", () => {
    const res = buildAnalytes([{ analyte: "glucose", kME: "0.5", kIM: "n/a" }], {
      analyte: "analyte",
      kme: "kME",
      kim: "kIM",
    });
    expect(res.analytes[0]).toEqual({ name: "glucose", kme: 0.5, kim: "n/a" });
  });

  it("kME column unmapped → analytes built exactly as before (no regression)", () => {
    const res = buildAnalytes([{ analyte: "glucose", module: "Brown" }], {
      analyte: "analyte",
      group: "module",
    });
    expect(res.analytes[0]).toEqual({ name: "glucose", group: "Brown" });
  });
});

describe("suggestDirectionMapping (Axis A)", () => {
  it("auto-maps the pinned brown_module_directions.csv headers", () => {
    expect(
      suggestDirectionMapping(["module", "eigengene_trait_correlation", "trait_label"]),
    ).toEqual({
      group: "module",
      correlation: "eigengene_trait_correlation",
      trait: "trait_label",
    });
  });

  it("maps common shorthand headers (cor / trait / cluster)", () => {
    expect(suggestDirectionMapping(["cluster", "cor", "trait"])).toEqual({
      group: "cluster",
      correlation: "cor",
      trait: "trait",
    });
  });

  it("leaves targets unset when a column is missing", () => {
    // No correlation column present → correlation stays undefined (panel keeps Confirm disabled).
    expect(suggestDirectionMapping(["module", "trait"])).toEqual({
      group: "module",
      trait: "trait",
    });
  });
});

describe("buildModuleDirections (Axis A)", () => {
  const rows = [
    { module: "Brown", cor: "0.6", trait: "frailty index" },
    { module: "Blue", cor: "-0.3", trait: "frailty index" },
  ];

  it("builds per-group direction rows from mapped columns", () => {
    const out = buildModuleDirections(rows, {
      group: "module",
      correlation: "cor",
      trait: "trait",
    });
    expect(out).toEqual([
      { group: "Brown", eigengene_trait_correlation: 0.6, trait_label: "frailty index" },
      { group: "Blue", eigengene_trait_correlation: -0.3, trait_label: "frailty index" },
    ]);
  });

  it("returns [] when the mapping is incomplete", () => {
    expect(buildModuleDirections(rows, { group: "module", correlation: "cor" })).toEqual([]);
  });

  it("preserves malformed rows VERBATIM so the backend R19 gate can reject them", () => {
    const out = buildModuleDirections(
      [
        { module: "", cor: "0.6", trait: "t" }, // blank group → backend warns + drops
        { module: "Brown", cor: "n/a", trait: "t" }, // non-numeric corr → backend rejects
        { module: "Green", cor: "", trait: "t" }, // blank corr → backend rejects
        { module: "Blue", cor: "0.2", trait: "t" }, // valid → number
      ],
      { group: "module", correlation: "cor", trait: "trait" },
    );
    expect(out).toEqual([
      { group: "", eigengene_trait_correlation: 0.6, trait_label: "t" },
      { group: "Brown", eigengene_trait_correlation: "n/a", trait_label: "t" },
      { group: "Green", eigengene_trait_correlation: "", trait_label: "t" },
      { group: "Blue", eigengene_trait_correlation: 0.2, trait_label: "t" },
    ]);
  });

  it("skips only fully-empty rows (no group, correlation, or trait)", () => {
    const out = buildModuleDirections(
      [
        { module: "", cor: "", trait: "" }, // filler → skipped
        { module: "Blue", cor: "0.2", trait: "t" },
      ],
      { group: "module", correlation: "cor", trait: "trait" },
    );
    expect(out).toEqual([{ group: "Blue", eigengene_trait_correlation: 0.2, trait_label: "t" }]);
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
