---
title: "langgraph.json: use dotted-import format with src/ layout"
date: 2026-05-06
category: best-practices
module: discovery-pipeline
problem_type: best_practice
component: tooling
severity: high
applies_when:
  - "Using langgraph-cli with a Python src/ directory layout"
  - "Build system is hatchling (or similar) that remaps src/package → package"
  - "Graph builder module uses relative imports (from .state import ...)"
tags:
  - langgraph
  - langgraph-cli
  - langgraph-studio
  - src-layout
  - hatchling
  - python-imports
---

# langgraph.json: use dotted-import format with src/ layout

## Context

When setting up LangGraph Studio (`langgraph dev`) for local pipeline development, the graph reference format in `langgraph.json` determines how `langgraph-cli` locates and imports the graph builder function. With a `src/` directory layout and hatchling build system, the file-path format fails because `langgraph-cli` loads the file directly via `importlib`, bypassing the package's import system. This breaks all relative imports within the module tree.

The planning phase identified both formats as viable options but did not resolve which one to use — a feasibility reviewer flagged the ambiguity. The answer was discovered at execution time when the file-path format failed immediately. (session history)

## Guidance

Use the **dotted-import format** in `langgraph.json`, not file-path format:

```json
{
  "dependencies": ["."],
  "graphs": {
    "kraken_discovery": "kestrel_backend.graph.builder:build_discovery_graph"
  },
  "env": ".env"
}
```

The `"dependencies": ["."]` entry is load-bearing — it tells `langgraph-cli` to install the current directory as a package (via the build system) before attempting imports. Without it, neither format works. (session history)

## Why This Matters

File-path references cause `langgraph-cli` to load the module as a standalone script rather than as part of an installed package. Any relative import in the module tree (e.g., `from .state import DiscoveryState`) fails with:

```
ImportError: attempted relative import with no known parent package
```

This blocks `langgraph dev` from starting entirely — the graph cannot be loaded, and Studio shows no topology.

## When to Apply

- Any LangGraph project with a Python `src/` directory layout
- Projects using hatchling, setuptools with `src` layout, or any build system that remaps `src/package/` to `package/` at install time
- When `pyproject.toml` has a wheel target like `packages = ["src/kestrel_backend"]`

## Examples

**Failing configuration** (file-path format):
```json
{
  "dependencies": ["."],
  "graphs": {
    "kraken_discovery": "./src/kestrel_backend/graph/builder.py:build_discovery_graph"
  },
  "env": ".env"
}
```

Error: `ImportError: attempted relative import with no known parent package`

**Working configuration** (dotted-import format):
```json
{
  "dependencies": ["."],
  "graphs": {
    "kraken_discovery": "kestrel_backend.graph.builder:build_discovery_graph"
  },
  "env": ".env"
}
```

**Supporting `pyproject.toml` context** (hatchling):
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/kestrel_backend"]
```

This configuration tells hatchling to install `src/kestrel_backend` as `kestrel_backend` at the top level, making the dotted-import path `kestrel_backend.graph.builder` resolvable.

## Related

- `docs/solutions/best-practices/langgraph-pipeline-production-formalization.md` — broader pipeline architecture and configuration patterns
- `docs/plans/2026-05-06-002-feat-langgraph-studio-setup-plan.md` — the Studio setup plan where this was discovered
- GitHub Issue #40 — related `src/` layout import error in `test_logging.py`
