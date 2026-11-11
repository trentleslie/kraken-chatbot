"""Pipeline LLM calls must run with Claude Code's built-in tools disabled.

`allowed_tools=[]` only lists tools that skip approval; it does not remove any. With
`permission_mode="bypassPermissions"` every built-in tool stays usable. Prompts here carry
untrusted text (paper abstracts, KG names, uploaded analyte names), so each call must pass
`tools=[]` and must not bypass permissions. Classic mode (agent.py) keeps its tools by design
and is excluded.
"""

import ast
from pathlib import Path

import pytest

from kestrel_backend.graph import sdk_utils

SRC = Path(__file__).resolve().parents[1] / "src" / "kestrel_backend"
EXCLUDED = {SRC / "agent.py"}  # classic mode: tools are intentional (separate decision)


def _option_calls():
    """Every ClaudeAgentOptions(...) call in the backend, with its keyword names and values."""
    for path in SRC.rglob("*.py"):
        if path in EXCLUDED:
            continue
        tree = ast.parse(path.read_text())
        dict_kwargs = {}  # name -> dict literal, for ClaudeAgentOptions(**name)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Dict):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        dict_kwargs[target.id] = node.value
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "ClaudeAgentOptions":
                kw = {k.arg: k.value for k in node.keywords if k.arg}
                for k in node.keywords:
                    if k.arg is None and isinstance(k.value, ast.Name) and k.value.id in dict_kwargs:
                        d = dict_kwargs[k.value.id]
                        kw.update({key.value: val for key, val in zip(d.keys, d.values)
                                   if isinstance(key, ast.Constant)})
                yield path.relative_to(SRC), node.lineno, kw


def test_every_options_construction_disables_builtin_tools():
    offenders = []
    for path, line, kw in _option_calls():
        tools = kw.get("tools")
        if not (isinstance(tools, ast.List) and not tools.elts):
            offenders.append(f"{path}:{line} missing tools=[]")
        mode = kw.get("permission_mode")
        if isinstance(mode, ast.Constant) and mode.value == "bypassPermissions":
            offenders.append(f"{path}:{line} uses bypassPermissions")
    assert not offenders, offenders


def test_scan_finds_the_known_call_sites():
    # Guard against a scanner that silently matches nothing.
    assert len(list(_option_calls())) >= 8


@pytest.mark.skipif(not sdk_utils.HAS_SDK, reason="claude_agent_sdk not installed")
def test_create_agent_options_disables_builtin_tools():
    opts = sdk_utils.create_agent_options(system_prompt="x", allowed_tools=[], max_turns=1)
    assert opts.tools == []
    assert opts.permission_mode != "bypassPermissions"


@pytest.mark.skipif(not sdk_utils.HAS_SDK, reason="claude_agent_sdk not installed")
async def test_query_with_usage_locks_options_built_elsewhere(monkeypatch):
    seen = {}

    async def fake_query(prompt, options):
        seen["options"] = options
        return
        yield  # pragma: no cover  (makes this an async generator)

    monkeypatch.setattr(sdk_utils, "query", fake_query)
    loose = sdk_utils.ClaudeAgentOptions(
        system_prompt="x", allowed_tools=[], max_turns=1, permission_mode="bypassPermissions"
    )
    await sdk_utils.query_with_usage("hi", loose, node_name="t")
    assert seen["options"].tools == []
    assert seen["options"].permission_mode != "bypassPermissions"
