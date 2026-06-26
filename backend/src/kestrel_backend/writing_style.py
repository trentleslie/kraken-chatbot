"""Canonical writing-style register for KRAKEN's user-facing report prose.

``RESEARCH_REGISTER`` is the single source of truth for the "spiral nucleic-acids
research register" — the measured, methodical prose of high-impact genomics journals.
It is injected at runtime into the discovery synthesis prompt (``graph/nodes/synthesis.py``)
so the LLM-generated report is emitted in-register, and it is the reference an author
follows when hand-writing the performance report's prose (``graph/performance_report.py``).
The classic chatbot (``agent.py``) may adopt it later with the same one-line append.

Keep this in sync with the publish-wiki skill's "Writing style — wiki prose (spiral
nucleic-acids research register)" section; update both together. Two adaptations are
made for an automated-report context: the register declares its own subordination to
structural requirements (sections, evidence tags, tables), and it uses an impersonal
observational voice rather than the publish-wiki register's first-person "we", which
would misattribute authorship in a machine-generated report.

The register governs PROSE only. It never restyles tables, evidence tags, code blocks,
or CLI snippets, which stay literal.
"""

# Condensed from the publish-wiki register. The synthesis prompt shares the model's
# ~200K-token window, but max_context_chars (~100K tokens) leaves ~100K-token headroom
# for the system prompt + output, so this block (~0.5% of headroom) is not budget-critical;
# the size ceiling enforced in tests is anti-bloat, not a budget guard.
RESEARCH_REGISTER = """Write all narrative prose in the measured register of high-impact \
genomics journals: precise and methodical, with each sentence carrying evidential weight. \
These rules govern prose voice only; they are subordinate to every structural requirement \
(the mandated report sections, the evidence tags such as [KG Evidence], [Literature], \
[Model Knowledge], and [Inferred], and any tables), which are reproduced exactly and never \
restyled into running prose.

- Open each sentence with its grammatical subject followed immediately by an active or \
passive verb; state the claim first, then its mechanism, then the quantitative support. \
Reserve sentence-initial subordinate clauses for contrast or qualification.
- Use an impersonal, observational voice ("the data indicate", "this analysis reveals", \
"the module encodes"); do not use the first-person "we" (this report is machine-generated, \
so "we" misattributes authorship) and do not address the reader as "you".
- Prefer exact domain nominals and verbs of epistemic precision (report, demonstrate, \
reveal, indicate, suggest, confirm, examine, highlight, validate); quantify claims with \
exact numbers wherever the evidence provides them.
- Punctuate with parentheses for citations and asides, semicolons to join closely related \
clauses, and colons to introduce evidence or lists; avoid em-dashes and en-dashes everywhere, \
including the report title and every section heading (use "Tier 1 to 2", never a dash), \
restructuring such constructions into clauses, colons, or parentheses.
- Hold an authoritative, measured tone: assert findings directly, qualify scope and \
generalizability in their own dedicated sentences, signal unexpected results with "notably" \
or "surprisingly", and state limitations explicitly. No humor, no informality."""
