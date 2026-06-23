"""Unit 1: the canonical RESEARCH_REGISTER constant.

The register is the single source of truth injected into the synthesis prompt and
followed when authoring the performance-report prose. These tests guard its presence,
its load-bearing directives, and a size ceiling so a careless edit cannot silently
bloat the synthesis system prompt or drop a core rule.
"""

from kestrel_backend.writing_style import RESEARCH_REGISTER

# Anti-bloat ceiling. The synthesis prompt shares the ~200K-token window, but
# SynthesisConfig.max_context_chars leaves ~100K-token headroom for system prompt +
# output, so this ceiling is not budget-critical — it exists to stop the register from
# growing unbounded. 1800 chars ~= 500 tokens, well within headroom.
MAX_REGISTER_CHARS = 1800


def test_register_is_non_empty_str():
    assert isinstance(RESEARCH_REGISTER, str)
    assert RESEARCH_REGISTER.strip()


def test_register_under_size_ceiling():
    assert len(RESEARCH_REGISTER) <= MAX_REGISTER_CHARS, (
        f"RESEARCH_REGISTER is {len(RESEARCH_REGISTER)} chars, over the "
        f"{MAX_REGISTER_CHARS}-char anti-bloat ceiling"
    )


def test_register_carries_core_directives():
    # The em-dash rule (a concrete, checkable register directive).
    assert "avoid em-dashes" in RESEARCH_REGISTER
    # The subordination clause: style is subordinate to structural requirements.
    assert "subordinate to" in RESEARCH_REGISTER
    # The impersonal-voice adaptation (no first-person "we" in machine-generated reports).
    assert "impersonal" in RESEARCH_REGISTER
    assert 'do not use the first-person "we"' in RESEARCH_REGISTER


def test_register_practices_what_it_preaches_on_em_dashes():
    # The register tells authors to avoid em-dashes; it must not contain one itself.
    assert "—" not in RESEARCH_REGISTER
