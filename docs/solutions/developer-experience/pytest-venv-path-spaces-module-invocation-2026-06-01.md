---
title: "Run pytest as a module: venv-path spaces break console-script shebangs"
date: 2026-06-01
category: docs/solutions/developer-experience
module: backend/testing
problem_type: developer_experience
component: testing_framework
severity: high
applies_when:
  - "A uv/Poetry/venv project is checked out under a path containing spaces (Google Drive, My Drive, macOS Library/Mobile Documents, OneDrive, Dropbox)"
  - "pytest/ruff/mypy/alembic raise ModuleNotFoundError for a dependency that IS installed"
  - "sys.executable resolves to system Python (/usr/bin/python3) instead of the project .venv"
  - "A test stubs a package via sys.modules and unrelated test files fail during collection"
root_cause: config_error
resolution_type: workflow_improvement
related_components:
  - documentation
  - tooling
tags:
  - pytest
  - uv
  - venv
  - shebang
  - path-with-spaces
  - module-invocation
  - test-isolation
  - sys-modules-stub
---

# Run pytest as a module: venv-path spaces break console-script shebangs

## Context

The kraken-chatbot backend lives under a cloud-synced checkout whose absolute path contains spaces:

```
/home/trentleslie/trentleslie@gmail.com/Google Drive/projects/kraken-chatbot/backend
```

The venv (`backend/.venv`) is managed by `uv`. The canonical test command documented in the project `CLAUDE.md` is:

```bash
cd backend && uv run pytest tests/ -v -m "not integration"
```

That form fails confusingly: `ModuleNotFoundError` for dependencies that are demonstrably installed (`pydantic`, `langgraph`, `ftfy`), or pytest silently runs under system Python `/usr/bin/python3` instead of the venv.

**Root cause — fragile shebang in console-script launchers.** When `uv` installs a console-script entry point like `pytest`, it writes `.venv/bin/pytest` as a tiny launcher whose first line is a shebang pointing at the absolute interpreter path:

```
#!/home/trentleslie/.../Google Drive/projects/kraken-chatbot/backend/.venv/bin/python
```

The kernel's `execve`/shebang (`#!`) handling does not robustly support spaces in the interpreter path — on Linux it takes the path up to the first space as the interpreter and treats the remainder as a single argument, so a space in the venv directory means it looks for a truncated, non-existent interpreter. The launch falls through to system Python, which lacks the project's dependencies, producing the misleading `ModuleNotFoundError`. The packages aren't missing — the wrong interpreter is running.

> Note: in this particular checkout the baked-in shebang was *also* a stale path from a previous sync root (`Insync/...`). A console-script shebang is hard-coded at venv-creation time, so it can be stale **or** space-containing — either way the conclusion is the same: never rely on the `.venv/bin/<tool>` shebang; invoke the tool as a module.

**A second, independent trap in the same investigation (issue #40).** `backend/tests/test_literature_grounding.py` installs fake stub modules into `sys.modules` so its unit tests don't import heavy real dependencies. The original stub for the `kestrel_backend` package was an empty module with no `__path__`. Because that bare module shadowed the real installed package, Python could no longer resolve *real* submodules that other test files import (e.g. `kestrel_backend.logging_config`, used by `test_logging.py`). The result was a pytest **collection** error across unrelated test modules — not an assertion failure inside the literature tests — which is far harder to attribute to its source.

## Guidance

**Run the tool as a module, not as a console script**, so interpreter selection bypasses the shebang entirely:

```bash
cd backend && uv run python -m pytest tests/ -v -m "not integration"
```

`uv run python` resolves the venv interpreter directly (no shebang parsed), and `-m pytest` imports the `pytest` package inside that already-correct interpreter. The space-in-path problem never arises. The same applies to every other console-script tool: prefer `uv run python -m ruff`, `... -m mypy`, `... -m alembic`, etc.

**Diagnose which interpreter actually ran** before trusting any pass/fail result:

```bash
# Which interpreter does the module form use? (should be the .venv python)
cd backend && uv run python -c "import sys; print(sys.executable)"

# Compare the two invocation forms — the script form may launch the wrong python (or fail):
cd backend && uv run python -m pytest --version   # module form (correct)
cd backend && uv run pytest --version             # script form (fragile shebang)

# Inspect the offending launcher's shebang directly:
cd backend && head -1 .venv/bin/pytest
```

If `sys.executable` points anywhere other than `.../backend/.venv/bin/python`, you are not running against the project venv and any `ModuleNotFoundError` is a false signal.

**For sys.modules stubbing**, when you fake a *package* (not a leaf module) to dodge heavy imports, give the stub a real `__path__` so the package's genuine submodules stay importable, and complete the stubbed surface every importer touches:

```python
import sys, types

# Stub EVERY package level down to the real submodule, each with a real __path__.
# A one-level stub is not enough: faking only `kestrel_backend` still yields
# "'kestrel_backend.graph' is not a package" because the intermediate node also
# needs __path__.
for name, path in [
    ("kestrel_backend", "src/kestrel_backend"),
    ("kestrel_backend.graph", "src/kestrel_backend/graph"),
    ("kestrel_backend.graph.nodes", "src/kestrel_backend/graph/nodes"),
]:
    mod = types.ModuleType(name)
    mod.__path__ = [path]          # keep real submodules resolvable
    sys.modules[name] = mod
```

A direct `sys.modules["pkg.sub"] = fake` entry is consulted *before* `__path__`-based resolution, so explicitly-faked submodules remain authoritative while everything you didn't fake falls through to the real source.

## Why This Matters

- **The failure lies about its cause.** A `ModuleNotFoundError` for an installed package sends developers down a dependency-reinstall rabbit hole (`uv sync`, deleting `.venv`, pinning versions) when the real fix is changing how the runner is launched. (session history) The first attempt here was to add `pythonpath = ["src"]` to `pyproject.toml`; it fixed pytest's *import path* but immediately surfaced `ModuleNotFoundError: No module named 'langgraph'`, which finally exposed that pytest was running under `/usr/bin/python3`, not the venv.
- **The canonical doc steers people into the trap.** `CLAUDE.md` documents the `uv run pytest` (script) form. Every developer and agent who follows the docs faithfully hits the bug. A wrong-by-default command is worse than no command — it manufactures confidence in a broken path.
- **Cloud-synced dev directories make this common, not exotic.** "Google Drive", "My Drive", macOS "Library/Mobile Documents", OneDrive, and Dropbox all put spaces in paths. Any uv/Poetry/venv project checked out under one is exposed.
- **sys.modules stubbing has spooky action at a distance.** A bad stub in *one* test file breaks *collection* of *other, unrelated* test files. The `__path__` discipline prevents a whole class of hard-to-localize collection failures.

## When to Apply

Use the **`python -m` module form** whenever a uv/Poetry/venv project sits under a spaced path, or when a console-script tool raises `ModuleNotFoundError` for a confirmed-installed dependency, or when `sys.executable` resolves to system Python. It is the safe default for *all* console-script entry points, not just pytest.

Use the **stub `__path__` discipline** whenever a test injects fake *packages* into `sys.modules` and other test files import real submodules of that package, or when you see pytest **collection** errors (e.g. `'<pkg>' is not a package`) in modules unrelated to the one doing the stubbing.

## Examples

**Before — script form, fragile shebang (documented but broken):**

```bash
$ cd backend && uv run pytest tests/ -v -m "not integration"
ModuleNotFoundError: No module named 'langgraph'
# ...yet the package is installed:
$ cd backend && uv run python -c "import langgraph; print('ok')"
ok        # it IS there — wrong interpreter was launched

$ head -1 backend/.venv/bin/pytest
#!/home/trentleslie/.../Google Drive/projects/kraken-chatbot/backend/.venv/bin/python
#                      ^^^^^^^^^^^^ spaces here break kernel shebang parsing
```

**After — module form, shebang bypassed:**

```bash
$ cd backend && uv run python -m pytest tests/ -v -m "not integration"
# collects and runs against the venv interpreter; deps resolve correctly

$ cd backend && uv run python -c "import sys; print(sys.executable)"
/home/.../Google Drive/projects/kraken-chatbot/backend/.venv/bin/python   # correct venv
```

**Before — empty package stub shadows the real package (breaks unrelated collection):**

```python
import sys, types

# BAD: no __path__ → real submodules (kestrel_backend.logging_config, etc.)
# become unresolvable; test_logging.py fails at COLLECTION, not assertion.
kestrel_backend = types.ModuleType("kestrel_backend")
sys.modules["kestrel_backend"] = kestrel_backend
```

**After — real `__path__` plus a complete stubbed surface (from `backend/tests/test_literature_grounding.py`):**

```python
import sys, types

# Give EACH package level a real __path__ so submodules this file does NOT fake
# (e.g. kestrel_backend.logging_config, imported by test_logging.py) still resolve
# to the real source instead of failing collection with "'<pkg>' is not a package".
kestrel_backend = types.ModuleType("kestrel_backend")
kestrel_backend.__path__ = ["src/kestrel_backend"]
kestrel_backend_graph = types.ModuleType("kestrel_backend.graph")
kestrel_backend_graph.__path__ = ["src/kestrel_backend/graph"]
kestrel_backend_graph_nodes = types.ModuleType("kestrel_backend.graph.nodes")
kestrel_backend_graph_nodes.__path__ = ["src/kestrel_backend/graph/nodes"]
sys.modules["kestrel_backend"] = kestrel_backend
sys.modules["kestrel_backend.graph"] = kestrel_backend_graph
sys.modules["kestrel_backend.graph.nodes"] = kestrel_backend_graph_nodes

# Then complete the surface every importer touches. The real node imports
# classify_relationship_llm (issue #40), so the leaf stub must define it too:
mock_semantic_scholar = types.ModuleType("kestrel_backend.semantic_scholar")
mock_semantic_scholar.classify_relationship = None
mock_semantic_scholar.classify_relationship_llm = None   # issue #40: real node imports this
sys.modules["kestrel_backend.semantic_scholar"] = mock_semantic_scholar
```

**Doc fix (the meta-learning):** the `CLAUDE.md` test command should use `uv run python -m pytest ...` so the canonical command no longer steers developers into the broken path.

## Related

- `../best-practices/langgraph-json-src-layout-import-2026-05-06.md` — sibling Python import/module-resolution lesson (different root cause; "see also")
- GitHub issue #40 — test_logging.py collection error from the `sys.modules` stub
