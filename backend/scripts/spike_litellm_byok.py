"""Phase-0 spike (GATE) — confirm claude-agent-sdk routes through the LiteLLM proxy
with per-session BYOK before building Tasks 3-7. See
docs/plans/2026-07-13-litellm-proxy-byok.md (Task 2) and the design spec's §4.

Run against a LOCAL litellm proxy started from deploy/litellm.config.yaml:

    # terminal 1
    export LITELLM_MASTER_KEY=sk-proxy-local
    litellm --config deploy/litellm.config.yaml --port 4000 --detailed_debug

    # terminal 2
    export LITELLM_MASTER_KEY=sk-proxy-local
    export USER_ANTHROPIC_KEY=sk-ant-...        # a REAL Anthropic key
    uv run python backend/scripts/spike_litellm_byok.py

Confirm in the proxy's --detailed_debug log:
  1. inbound request carries BOTH `authorization` AND `x-api-key` (neither clobbers the other)
  2. request reaches the proxy (no 403, no direct-to-Anthropic bypass)
  3. upstream call uses the FORWARDED x-api-key (the user key), not an operator key
  4. tool-use / streaming intact; turn completes

Negative check: re-run with USER_ANTHROPIC_KEY=sk-ant-INVALID and confirm the failure
originates at Anthropic (401 upstream), not at the proxy — proving the user key transited.
"""
import asyncio
import os
import shutil

from claude_agent_sdk import ClaudeAgentOptions, query


async def main() -> None:
    cli = shutil.which("claude")
    print(f"[spike] cli_path (system claude binary): {cli}")
    opts = ClaudeAgentOptions(
        model="claude-sonnet-4",
        cli_path=cli,  # bundled SDK binary ignores ANTHROPIC_BASE_URL (#677/#1089)
        max_turns=1,
        env={
            "ANTHROPIC_BASE_URL": "http://127.0.0.1:4000",
            "ANTHROPIC_AUTH_TOKEN": os.environ["LITELLM_MASTER_KEY"],  # -> Authorization (proxy-auth, stripped)
            "ANTHROPIC_API_KEY": os.environ["USER_ANTHROPIC_KEY"],     # -> x-api-key (forwarded upstream)
        },
    )
    async for msg in query(prompt="Reply with the single word: pong.", options=opts):
        print(type(msg).__name__, getattr(msg, "content", msg))


if __name__ == "__main__":
    asyncio.run(main())
