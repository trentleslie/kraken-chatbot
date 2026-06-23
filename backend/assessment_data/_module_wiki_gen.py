"""Generate Outline-ready .outline.md for a module run (discovery output OR perf report).

Templated per-module wiki docs for the multi-module batch: reads a run artifact JSON
(brown_c1_pilot_e2e output, which carries _meta + coverage + synthesis_report) and the perf
report .md, and emits a wrapper (provenance / intro + Related links) with the report embedded
(headings demoted one level so they nest under the doc H1). Prose stays em-dash-free.

Usage:
  python -m assessment_data._module_wiki_gen --module Blue --json <run.json> --perf <perf.md> \
      --kind discovery [--perf-url URL] --out ../docs/wiki/blue-module-discovery-output.outline.md
  python -m assessment_data._module_wiki_gen --module Blue --json <run.json> --perf <perf.md> \
      --kind perf [--discovery-url URL] --out ../docs/wiki/blue-module-performance-report.outline.md
"""
import argparse
import json
from pathlib import Path

INDEX_URL = "https://phwiki.phenoma.ai/doc/biomapper-collection-index-UcTfYSTchm"


def _demote(md: str) -> str:
    return "\n".join(("#" + ln) if ln.startswith("#") else ln for ln in md.splitlines())


def discovery_doc(module, j, perf_url=None, model_label=None, compare_url=None):
    m, c = j["_meta"], j["coverage"]
    tb, rbm = c["triage_buckets"], c["resolved_by_method"]
    date = m["generated"][:10]
    rbm_str = ", ".join(f"{v} {k}" for k, v in rbm.items())
    mf = c.get("triage_measurement_failures", 0)
    on = f" on {model_label}" if model_label else ""
    model_note = (
        f" The SDK-backed pipeline nodes ran on the {model_label} model rather than the default "
        f"Sonnet; this run is published for direct model comparison."
        if model_label else ""
    )
    lines = [
        f"# {module} Module Run{on}: Discovery Output ({m['n_submitted']}-analyte, dev, {date})",
        "",
        f"This document presents the discovery output of a full-module run of the **{module}** WGCNA "
        f"module through the Kraken 12-node discovery pipeline, executed on the dev environment on "
        f"{date} (commit `{m['git_sha'][:7]}`) with BioMapper entity resolution enabled.{model_note} "
        f"It is one of a per-module series over the Frailty multi-omics WGCNA modules; see the "
        f"[BioMapper: Collection Index]({INDEX_URL}).",
        "",
        "## Run Provenance",
        "",
        f"The run submitted {m['n_submitted']} named analytes, parsed {m['n_parsed']} at intake, and "
        f"resolved {c['resolved_entities_count']} distinct entities ({rbm_str}) to {c['distinct_curies']} "
        f"distinct CURIEs. Triage classified {tb['well_characterized']} well-characterized, "
        f"{tb['moderate']} moderate, {tb['sparse']} sparse, and {tb['cold_start']} cold-start, with "
        f"{mf} edge-count measurement failures. Downstream analysis produced {c['direct_findings']} "
        f"direct-KG findings, {c['cold_start_findings']} cold-start findings, {c['biological_themes']} "
        f"biological themes, {c['bridges']} cross-entity bridges ({c['grounded_bridges']} evidence-grounded), "
        f"and {c['hypotheses']} hypotheses supported by {c['literature_support_total']} literature references. "
        f"Synthesis emitted a {c['synthesis_report_chars']}-character report. The run completed in "
        f"approximately {m['wall_seconds']} s of wall-clock time (status {m['status']}, {c['errors']} errors). "
        f"All counts are taken from the reducer-merged pipeline state.",
        "",
        "| Stage | Result |",
        "|---|---|",
        f"| Submitted | {m['n_submitted']} named analytes |",
        f"| Intake | {m['n_parsed']} parsed |",
        f"| Entity resolution | {c['resolved_entities_count']} resolved ({rbm_str}) to {c['distinct_curies']} distinct CURIEs |",
        f"| Triage | {tb['well_characterized']} well-characterized, {tb['moderate']} moderate, {tb['sparse']} sparse, {tb['cold_start']} cold-start ({mf} measurement failures) |",
        f"| Direct KG | {c['direct_findings']} findings |",
        f"| Cold-start | {c['cold_start_findings']} findings, {c.get('cold_start_skipped_count', 0)} skipped |",
        f"| Pathway enrichment | {c['biological_themes']} biological themes |",
        f"| Integration | {c['bridges']} bridges ({c['grounded_bridges']} evidence-grounded) |",
        f"| Literature grounding | {c['literature_support_total']} papers |",
        f"| Synthesis | {c['hypotheses']} hypotheses, {c['synthesis_report_chars']}-character report |",
        f"| Run total | ~{m['wall_seconds']} s wall-clock, status {m['status']}, {c['errors']} errors |",
        "",
        "## Related",
        "",
    ]
    if perf_url:
        lines.append(f"- Companion run metrics: [{module} Module Run{on}: Pipeline Performance Report]({perf_url})")
    if compare_url:
        lines.append(f"- Model comparison baseline (Sonnet): [{module} Module Run: Discovery Output]({compare_url})")
    lines.append(f"- [BioMapper: Collection Index]({INDEX_URL})")
    lines += ["", "---", "", _demote(j["synthesis_report"])]
    return "\n".join(lines)


def perf_doc(module, j, perf_md, discovery_url=None, model_label=None, compare_url=None):
    m = j["_meta"]
    date = m["generated"][:10]
    on = f" on {model_label}" if model_label else ""
    model_note = (
        f" All SDK-backed nodes ran on the {model_label} model; this report is published for direct "
        f"model comparison against the Sonnet baseline."
        if model_label else ""
    )
    lines = [
        f"# {module} Module Run{on}: Pipeline Performance Report ({m['n_submitted']}-analyte, dev, {date})",
        "",
        f"This is the per-node performance report emitted automatically at the end of the full-module "
        f"**{module}** discovery run on dev (commit `{m['git_sha'][:7]}`).{model_note} The Kraken "
        f"performance-reporter (a terminal graph node) attributes wall-clock, tokens, estimated cost, "
        f"output counts, and errors across the twelve pipeline nodes, and the Context management section "
        f"reports how synthesis compresses the accumulated evidence into a bounded context.",
        "",
        "## Related",
        "",
    ]
    if discovery_url:
        lines.append(f"- Discovery analysis from this run: [{module} Module Run{on}: Discovery Output]({discovery_url})")
    if compare_url:
        lines.append(f"- Model comparison baseline (Sonnet): [{module} Module Run: Pipeline Performance Report]({compare_url})")
    lines.append(f"- [BioMapper: Collection Index]({INDEX_URL})")
    lines += ["", "---", "", _demote(perf_md)]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--perf", required=True)
    ap.add_argument("--kind", choices=["discovery", "perf"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--perf-url", default=None)
    ap.add_argument("--discovery-url", default=None)
    ap.add_argument("--model-label", default=None, help="e.g. 'Opus 4.8' — labels a non-default-model run")
    ap.add_argument("--compare-url", default=None, help="URL of the baseline (Sonnet) doc to cross-link")
    args = ap.parse_args()

    j = json.loads(Path(args.json).read_text())
    perf_md = Path(args.perf).read_text()
    if args.kind == "discovery":
        text = discovery_doc(args.module, j, perf_url=args.perf_url,
                             model_label=args.model_label, compare_url=args.compare_url)
    else:
        text = perf_doc(args.module, j, perf_md, discovery_url=args.discovery_url,
                        model_label=args.model_label, compare_url=args.compare_url)
    assert "—" not in text.split("---", 1)[0], "em-dash in authored wrapper prose"
    Path(args.out).write_text(text)
    print(f"wrote {args.out} ({len(text)} chars)")


if __name__ == "__main__":
    main()
