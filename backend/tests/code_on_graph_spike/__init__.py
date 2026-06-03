"""Code-on-Graph go/no-go spike harness (throwaway).

Phase-0 kill-test: does an LLM iterative query-refinement loop beat the static
query plan at endpoint-to-endpoint bridge discovery? Hits Kestrel REST /api
directly (the /mcp server drops X-API-Key intermittently — see plan finding #1).

See docs/plans/2026-06-03-002-feat-code-on-graph-spike-plan.md.
"""
