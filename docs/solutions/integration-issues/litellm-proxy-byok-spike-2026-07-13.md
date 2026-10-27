---
title: "LiteLLM-proxy BYOK spike (claude-agent-sdk through a DB-less proxy with per-session keys)"
date: 2026-07-13
last_updated: 2026-10-27
category: integration-issues
module: byok
tags: [litellm, proxy, byok, claude-agent-sdk, anthropic]
problem_type: integration_issue
---

# LiteLLM-proxy BYOK spike — 2026-07-13

**Result: ✅ GO.** claude-agent-sdk routes through a DB-less LiteLLM proxy with per-session BYOK; the user's own key is forwarded to Anthropic and is what bills. Tasks 3–7 of `docs/plans/2026-07-13-litellm-proxy-byok.md` are cleared to build.

## Update 2026-10-27: pins superseded; proxy mode must also meet the model's CLI minimum

The spike's findings (header mapping, `cli_path` requirement, user key bills) still hold and are what ships. What changed:

- **Version pins are superseded.** The app now runs `claude-agent-sdk` **0.2.170** (bundled CLI **2.1.306**) and the default model `claude-opus-5-5` (#103). New models are gated by Claude Code CLI version: Opus 5.5 rejects older CLIs with `400 claude_code_version_too_old` ("version 2.1.280 or newer is required").
- **Proxy mode uses the *system* `claude`, which needs that minimum too.** Because `cli_path=shutil.which("claude")` bypasses the SDK's bundled binary, bumping the SDK alone doesn't help BYOK/proxy turns. On the Lightsail box the system CLI was 2.1.237 and had to be upgraded (`sudo npm i -g @anthropic-ai/claude-code`; the version ceiling is gated by the Node major).
- **The "stale model id" follow-up below is resolved.** `sdk_utils.DEFAULT_MODEL_NAME` now derives from `config.DEFAULT_CLAUDE_MODEL`, and every SDK call is pinned at `query_with_usage` (`_pin_model`). The same retired-ID trap later hit the key-validation probe (`claude-3-5-haiku-latest`, now `claude-haiku-4-5`, #112).
- **`deploy/litellm.env` is a hard prerequisite.** When it was missing, the proxy unit crash-looped and the deploy exited before `uv sync`/restart. Dev ran stale code for months.
- The ~$0.13-per-trivial-turn overhead (~34k cached tokens of Claude Code system prompt and tool definitions) still applies, and is higher at Opus prices.

Full upgrade checklist: the vault learning `integration-issues/agent-sdk-model-upgrade-bundled-cli-version-gate-2026-10-27.md`.

## Versions pinned (at spike time, superseded; see update above)
- `claude` CLI (Claude Code): **2.1.201**
- `claude-agent-sdk`: **0.1.31**
- `litellm`: **1.92.0**
- Model exercised: `claude-sonnet-4-5-20250929` (via alias `claude-sonnet-4`)

## Setup exercised
- Proxy: `litellm --config deploy/litellm.config.yaml --port 4000 --detailed_debug`, DB-less, `general_settings.forward_client_headers_to_llm_api: true` + `forward_llm_provider_auth_headers: true`, `master_key=os.environ/LITELLM_MASTER_KEY`, **no operator provider key configured**.
- Client: `ClaudeAgentOptions(cli_path=shutil.which("claude"), env={ANTHROPIC_BASE_URL=http://127.0.0.1:4000, ANTHROPIC_AUTH_TOKEN=<master key>, ANTHROPIC_API_KEY=<user key>})`. Driver: `backend/scripts/spike_litellm_byok.py`.

## Criteria — all confirmed
1. **Both auth headers coexist.** `ANTHROPIC_AUTH_TOKEN`→`Authorization` (proxy-auth) and `ANTHROPIC_API_KEY`→`x-api-key` (user key) were both emitted; the documented "both set is version-dependent" caution did **not** bite on CLI 2.1.201 / SDK 0.1.31. Positive run returned `200 OK` (proxy-auth validated) and litellm logged: `Setting client-provided x-api-key as api_key parameter (will override deployment key)`.
2. **base_url honored via `cli_path`.** Request reached the proxy (`POST /v1/messages?beta=true 200 OK`) — no #677/#1089 bypass, no local 403. `cli_path`=system `claude` binary was necessary/sufficient.
3. **User key is what bills.** Proxy has no operator key, so upstream auth could only be the forwarded `x-api-key`. **Negative check** clinches it: `ANTHROPIC_API_KEY=sk-ant-INVALID` → `401 authentication_error "invalid x-api-key"` **from `https://api.anthropic.com/v1/messages`** (upstream), not from the proxy.
4. **Agentic behavior intact.** Streaming thinking blocks + text; `result='pong'`, `is_error=False`, `num_turns=1`.

## Gotchas found (feed into the build)
- **Stale model id.** `anthropic/claude-sonnet-4-20250514` (from the Task-1 config, and also kraken's `sdk_utils.DEFAULT_MODEL_NAME`) is **retired** → Anthropic `404 not_found_error`. Config fixed to `claude-sonnet-4-5-20250929`. **Pre-existing risk:** `sdk_utils.py:DEFAULT_MODEL_NAME` still hard-codes the retired id — out of scope for this pilot but worth a follow-up. Models available to the key (2026-07-13): `claude-sonnet-5`, `claude-sonnet-4-6`, `claude-sonnet-4-5-20250929`, `claude-opus-4-8/4-7/4-6`, `claude-haiku-4-5-20251001`, … Final pilot model is Trent's call.
- **Cost note.** One "pong" turn cost ~$0.13 — dominated by ~33.8k cache-creation tokens from the Claude Code system prompt the SDK injects, not the reply. Expected SDK overhead, not a proxy artifact.
- **`ANTHROPIC_API_KEY` set → claude.ai connectors disabled** (SDK warns). Harmless for server use; the SDK correctly reports `apiKeySource: 'ANTHROPIC_API_KEY'`.

## Implication for the build
The design's env→header mapping and `cli_path` requirement are **confirmed**, not assumed. `byok.build_agent_env()` (Task 3) should emit exactly: `ANTHROPIC_API_KEY`=user key, `ANTHROPIC_AUTH_TOKEN`=`LITELLM_MASTER_KEY`, `ANTHROPIC_BASE_URL`=proxy URL; and every proxy-targeting `ClaudeAgentOptions` sets `cli_path=shutil.which("claude")`.
