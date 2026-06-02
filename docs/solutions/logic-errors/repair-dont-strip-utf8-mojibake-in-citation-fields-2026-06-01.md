---
title: "Repair, don't strip: UTF-8 mojibake in citation fields"
date: 2026-06-01
category: docs/solutions/logic-errors
module: kestrel_backend.graph
problem_type: logic_error
component: assistant
symptoms:
  - "Citation fields (title, authors, key_passage) rendered UTF-8 mojibake, e.g. â€™ for a curly apostrophe"
  - "Double-encoded sequences like Ã¢Â€Â\" appeared in place of em-dashes"
  - "The prior char-class strip deleted legitimate em-dashes and apostrophes from clean citations"
root_cause: logic_error
resolution_type: code_fix
severity: medium
related_components:
  - assistant
  - service_object
tags:
  - mojibake
  - utf-8
  - encoding
  - ftfy
  - pydantic-v2
  - field-validator
  - literature-grounding
  - citations
---

# Repair, don't strip: UTF-8 mojibake in citation fields

> `component: assistant` is the closest schema enum; the true component is the
> LangGraph discovery pipeline's literature-grounding node (`graph/state.py`,
> `graph/nodes/literature_grounding.py`), for which no enum value exists.

## Problem

Citation text fields (`title`, `authors`, `key_passage`) produced by the literature-grounding stage of the discovery pipeline rendered with UTF-8 mojibake — text where bytes from one encoding were interpreted under a different codec (GitHub issue #39). The source text arrives from external literature APIs (OpenAlex, Exa, Semantic Scholar, PubMed), which is where the bad encoding originates.

## Symptoms

- **Single-encoded** mojibake: `Rabbani's group` rendering as `Rabbaniâ€™s group`; corrupted em-dashes.
- **Double-encoded** mojibake (the harder case): `dose — response curve` rendering as `dose Ã¢Â€Â" response curve` — a UTF-8 em-dash mis-decoded as Latin-1, re-encoded as UTF-8, then mis-decoded again (observed in real citations, Rabbani 2021).
- Legitimate punctuation was *simultaneously being lost* by the old "fix" (see below).
- The garbage propagated from the markdown references table into the frontend.

## What Didn't Work

The prior code in `backend/src/kestrel_backend/graph/nodes/literature_grounding.py` cleaned the passage with a **character-class strip**:

```python
relevance = re.sub(r'[âÂ€™""—–]+', '', relevance)  # Strip UTF-8 mojibake artifacts
```

Two failures:

1. **It deleted legitimate characters.** The class included `—` (em-dash), `'` (apostrophe), and `–` (en-dash) — valid in real citation text. A genuine em-dash in `"dose — response curve"` got erased along with the corruption.
2. **It never decoded anything.** Stripping suspect bytes does not *repair* a wrong-codec interpretation. For the double-encoded `Ã¢Â€Â"` sequence, the regex only chewed off lead bytes, leaving mangled remnants rather than the intended `—`.

It also ran only on `key_passage` during table construction, leaving `title` and `authors` unremediated, and addressed nothing at the data boundary.

## Solution

Repair the text instead of stripping it, at the point of model construction so every downstream consumer sees clean data.

**Before** — lossy char-class strip inside the table builder (`literature_grounding.py`):

```python
relevance = re.sub(r'[âÂ€™""—–]+', '', relevance)  # deletes legit em-dashes/quotes, decodes nothing
```

**After** — strip removed from `literature_grounding.py`, with a comment recording why; mojibake is now repaired upstream at construction.

**After** — a Pydantic v2 `mode="before"` field validator on `LiteratureSupport` in `backend/src/kestrel_backend/graph/state.py`:

```python
import ftfy  # module-level: ImportError surfaces at import time, not per-construction

# Deeply double-encoded UTF-8 mojibake that ftfy cannot fully resolve on its own
# (observed in literature citations, e.g. Rabbani 2021). Applied BEFORE ftfy.fix_text.
_MOJIBAKE_REPLACEMENTS = {
    'Ã¢Â€Â"': "—",   # em-dash
    "Ã¢Â€Â™": "'",   # apostrophe
}

class LiteratureSupport(BaseModel):
    # ...
    @field_validator("title", "authors", "key_passage", mode="before")
    @classmethod
    def _repair_mojibake(cls, v: Any) -> Any:
        if not isinstance(v, str) or not v:
            return v
        for bad, good in _MOJIBAKE_REPLACEMENTS.items():
            v = v.replace(bad, good)
        # NOTE: fix_text defaults to unescape_html="auto", which decodes HTML
        # entities when no tags are present. Fine here (table builder escapes the
        # cell), but pass unescape_html=False if copying this into a context that
        # renders the value as HTML. See "Known follow-up caveat" below.
        return ftfy.fix_text(v)
```

**After** — dependency added in `backend/pyproject.toml`: `"ftfy>=6.3.1"`.

Design notes (reflecting review feedback):
- `import ftfy` is at **module level**, so an `ImportError` surfaces at import time, not on first model construction, and the module isn't re-imported per instantiation.
- The explicit `_MOJIBAKE_REPLACEMENTS` map runs **first** (double-encoded cases), **then** `ftfy.fix_text` cleans the remaining single-encoding cases.
- The non-`str`/empty guard passes `None`/non-string input through untouched.
- **The literal byte sequences in `_MOJIBAKE_REPLACEMENTS` (and in the test below) are byte-sensitive.** Pasting them through a non-UTF-8 editor, chat client, or re-encoding terminal can silently alter the bytes — at which point the key no longer matches and the repair no-ops. Verify these literals survived the paste byte-for-byte (e.g. `python -c "print('Ã¢Â€Â\"'.encode())"`).

## Why This Works

**Mojibake is a decoding problem, not a content problem.** The bytes are correct; they were interpreted under the wrong codec (UTF-8 read as Latin-1, sometimes twice). The fix *re-decodes / repairs* the text — reversing the bad interpretation — rather than deleting odd-looking bytes.

- `ftfy.fix_text` detects common single-encoding mojibake and reverses the mis-decode, restoring `'` and `—` while leaving genuinely-correct characters (real em-dashes, accents like `café`) intact. Because it repairs rather than strips, it is non-destructive.
- The double-encoded case `Ã¢Â€Â"` is too far gone for ftfy to unwind reliably (once bytes are encoded twice, charset detection alone can't recover them), so the explicit map repairs those exact sequences first.
- Stripping is inherently lossy: it cannot distinguish a corrupted em-dash from a legitimate one, so it must erase both or neither. Repairing the encoding sidesteps that false choice.
- Doing the repair in a `mode="before"` validator centralizes it at the data boundary — every `LiteratureSupport` is clean by construction, so no downstream consumer needs its own ad-hoc sanitizer.

## Prevention

The regression test asserts **both** that the bad bytes are gone *and* that the correct character is present (a positive assertion, not mere absence), and separately verifies legitimate punctuation/accents survive. From `backend/tests/test_literature_grounding.py`:

```python
def test_repairs_mojibake_in_text_fields(self):
    moj_em = "—".encode("utf-8").decode("latin-1")    # em-dash mojibake
    moj_apos = "’".encode("utf-8").decode("latin-1")  # apostrophe mojibake
    lit = LiteratureSupport(
        paper_id="p", year=2021, authors=f"Rabbani{moj_apos}s group",
        title=f"Lipid{moj_em}diabetes link",
        key_passage="real em-dash — and café",
    )
    assert lit.title == "Lipid—diabetes link"   # single-encoding mojibake repaired
    assert moj_em not in lit.title              # negative pair: bad bytes gone
    assert "Rabbani" in lit.authors and moj_apos not in lit.authors
    assert "café" in lit.key_passage            # legitimate accented text preserved
    assert "—" in lit.key_passage               # legitimate em-dash preserved (not stripped)

    lit2 = LiteratureSupport(paper_id="p2", year=2021, authors="A",
                             title='dose Ã¢Â€Â" response curve')
    assert "Ã¢" not in lit2.title   # double-mojibake lead bytes removed
    assert "—" in lit2.title        # replaced with the correct em-dash
```

Generalizable rules:
- **Assert the positive, not just the negative.** `assert "—" in title` would fail against the old strip-based code; an absence-only check (`assert "Ã¢" not in title`) would pass it. Pair both to pin down "repaired correctly," not just "garbage removed."
- **Include a preservation case** (`café`, a real `—`) so a future over-aggressive sanitizer that re-introduces stripping fails the test.
- **Cover both corruption tiers** — single-encoded (`.encode("utf-8").decode("latin-1")`) and the literal double-encoded sequence from the real issue.

**Known follow-up caveat (session history):** `ftfy.fix_text` runs with default `unescape_html="auto"`, which decodes HTML entities (`&lt;`, `&gt;`, …) when the input contains no HTML tags — a transformation the old char-class strip never performed. Since `build_references_table()` does not HTML-escape the title/citation column (it escapes only `\n`, `\r`, `|`), an entity-encoded title could in principle reach the `synthesis_report` markdown shipped to the frontend. Severity is bounded by the frontend renderer's HTML policy. If hardening: pass `unescape_html=False` to `fix_text`, or HTML-escape the citation column in the table builder.

## Related Issues

- GitHub issue #39 — UTF-8 mojibake in literature citations (fixed in PR #59)
- `../best-practices/verify-temporal-provenance-before-kg-holdout-eval-2026-05-29.md` — touches the same file (`literature_grounding.py`), orthogonal concern (date-filtering for temporal eval)
