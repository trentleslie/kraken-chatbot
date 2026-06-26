"""Unit 2: the research register is injected into the synthesis prompt, budget-safe.

These guard that (1) SYNTHESIS_PROMPT carries the register without losing its existing
structure/evidence-tag contract, and (2) the enlarged system prompt still fits the
model's context window alongside the assembled context (SynthesisConfig.max_context_chars)
with output headroom. The prose-quality effect is LLM-driven and validated by a sample
run, not by unit assertions; the synthesis fallback-marker behavior is covered by
tests/test_synthesis_fallback_marker.py (unchanged by this unit).
"""

from kestrel_backend.graph.nodes.synthesis import SYNTHESIS_PROMPT
from kestrel_backend.graph.pipeline_config import get_pipeline_config
from kestrel_backend.writing_style import RESEARCH_REGISTER


def test_prompt_contains_register():
    assert RESEARCH_REGISTER in SYNTHESIS_PROMPT
    assert "## Writing register" in SYNTHESIS_PROMPT


def test_prompt_retains_existing_contract():
    # Appending the register must not truncate or replace the base prompt: its section
    # scaffolding and evidence-tag rules must survive intact.
    assert "Executive Summary" in SYNTHESIS_PROMPT
    assert "Evidence Attribution (REQUIRED)" in SYNTHESIS_PROMPT
    assert "[KG Evidence]" in SYNTHESIS_PROMPT
    assert "Generate a clear, scientific report in markdown format." in SYNTHESIS_PROMPT


def test_register_appended_after_base_prompt():
    # The register is the closing section, so it does not interrupt the report contract.
    assert SYNTHESIS_PROMPT.rstrip().endswith(RESEARCH_REGISTER.rstrip())


def test_enlarged_prompt_fits_window_with_headroom():
    # Affirmative budget gate against the real mechanism. There is no "system-prompt
    # reserve" field; the guard is this arithmetic against SynthesisConfig.max_context_chars.
    WINDOW_TOKENS = 200_000          # model input window (the real ceiling)
    CHARS_PER_TOKEN = 3.5            # measured for this CURIE-dense content (conservative)
    OUTPUT_RESERVE_TOKENS = 32_000   # headroom reserved for the generated report

    max_context_chars = get_pipeline_config().synthesis.max_context_chars
    system_tokens = len(SYNTHESIS_PROMPT) / CHARS_PER_TOKEN
    context_tokens = max_context_chars / CHARS_PER_TOKEN

    assert system_tokens + context_tokens + OUTPUT_RESERVE_TOKENS < WINDOW_TOKENS, (
        f"system prompt ({system_tokens:.0f}t) + max context ({context_tokens:.0f}t) + "
        f"output reserve ({OUTPUT_RESERVE_TOKENS}t) exceeds the {WINDOW_TOKENS}t window; "
        f"lower SynthesisConfig.max_context_chars to compensate for the register"
    )
