# LiteLLM-Proxy Pivot for Kraken BYOK — Design

- **Date:** 2026-07-13
- **Status:** Design (approved for spec write; pending user review before planning)
- **Branch:** `litellm-proxy-byok` (off `dev`)
- **Builds on:** BYOK per-session pilot (PR trentleslie/kraken-chatbot#93, merged 2026-07-11)
- **Related:** `docs/brainstorms/2026-07-11-byok-per-session-requirements.md`, `docs/plans/2026-07-11-byok-per-session-kraken.md`

## 1. Problem & goal

The merged BYOK pilot (#93) is **Anthropic-only**: it resolves a per-session key and injects
`ANTHROPIC_API_KEY` into `ClaudeAgentOptions(env=…)` via a `ContextVar`, so every LLM call goes
straight to Anthropic with the caller's key. The long-term goal is **multi-provider BYOK** — a
public user brings *any* provider's key (OpenAI, Gemini, …) and pays their own inference
(cost-offload) — without re-architecting kraken per provider.

We pivot to routing kraken's LLM traffic through a **LiteLLM proxy** (github.com/BerriAI/litellm),
which sits in front of the Claude Agent SDK and, in the future, translates the SDK's Anthropic-format
requests to whatever provider the user's key belongs to.

### Decisions locked during brainstorming

| Decision | Choice | Rationale |
|---|---|---|
| Integration shape | **Proxy in front** | Keeps the Claude Agent SDK + MCP tools + bash sandbox intact; the proxy is the routing/translation seam. |
| Who pays | **User pays (cost-offload)** | Keeps the original driver. No virtual keys, no budgets, **no Postgres**. |
| Multi-provider | **Yes — eventual goal** | Justifies the proxy over a plain per-call key. |
| Pilot provider scope | **Anthropic-only now** | De-risk: no cross-provider translation in the pilot. Adding providers later is config + a translation spike, not a re-architecture. |
| Deployment | **systemd sibling of the backend** | Matches existing infra: AWS Lightsail VM, systemd + nginx, no Docker. |

### Honest scoping note

With Anthropic-only **and** Langfuse already wired for observability, the proxy earns its keep in
*this pilot* mainly as **future-proofing** — laying the pipe so adding OpenAI/Gemini later is
config-only. This is a deliberate investment, not accidental scope. If the Phase-0 spike looks
shaky, the fallback (below) is to defer the proxy entirely.

## 2. Architecture

```
                    per-request env (ClaudeAgentOptions.env), never os.environ
  ┌────────────────────────────────────────────────────────────────────┐
  │ kraken-backend (FastAPI, systemd)                                    │
  │   resolve_effective_key_and_provider(byok_key, verified_email)       │
  │        → (key, provider="anthropic", source)                         │
  │   3 call boundaries build ClaudeAgentOptions(env={                    │
  │        ANTHROPIC_BASE_URL = http://127.0.0.1:4000,                    │
  │        <proxy-auth + forwarded provider key headers>  })              │
  └───────────────┬────────────────────────────────────────────────────┘
                  │ Anthropic /v1/messages format
                  ▼
  ┌────────────────────────────────────────────────────────────────────┐
  │ litellm proxy (systemd, 127.0.0.1:4000, DB-less)                     │
  │   general_settings:                                                  │
  │     forward_client_headers_to_llm_api: true                          │
  │     forward_llm_provider_auth_headers: true   # BYOK                 │
  │   Authorization: Bearer <proxy-auth>   → stripped (proxy auth only)  │
  │   x-api-key: sk-ant-…                   → FORWARDED upstream          │
  └───────────────┬────────────────────────────────────────────────────┘
                  │ forwards user's key
                  ▼
             Anthropic API      (future: OpenAI / Gemini via translation)
```

### 2.1 Two key layers (the conceptual crux)

Kept strictly distinct — this is what makes it cost-offload BYOK rather than operator-pays:

1. **Proxy-auth key** — a static internal secret (`LITELLM_MASTER_KEY`), backend → proxy only.
   Rides the `Authorization` header. **Stripped by the proxy, never forwarded.** (On a
   loopback-only DB-less proxy this is effectively an "accept any key" gate; we still set it so the
   proxy isn't wide open if the bind ever changes.)
2. **Provider key** — the per-session user Anthropic key. Rides `x-api-key`. **Forwarded upstream**
   so Anthropic bills the *user*, not Phenome.

Confirmed against LiteLLM docs: with `forward_client_headers_to_llm_api` +
`forward_llm_provider_auth_headers`, the client sends `Authorization: Bearer <proxy-auth>`
(stripped) **and** `x-api-key: sk-ant-…` (forwarded); the proxy's Authorization header is never
forwarded to providers.

**Concrete env→header mapping (from Claude Code auth docs — the spike confirms, doesn't discover):**

| Env var (per-request `ClaudeAgentOptions.env`) | HTTP header emitted | Our use |
|---|---|---|
| `ANTHROPIC_AUTH_TOKEN` | `Authorization: Bearer …` | **proxy-auth** = `LITELLM_MASTER_KEY` (stripped by proxy) |
| `ANTHROPIC_API_KEY` | `x-api-key: …` | **user's key** (forwarded upstream) — *unchanged from #93* |

This is a minimal delta from #93 (which already injects the user key via `ANTHROPIC_API_KEY`): we
**add** `ANTHROPIC_AUTH_TOKEN` (proxy-auth) + `ANTHROPIC_BASE_URL` to the same per-request env.
⚠️ **Documented caution:** setting *both* `ANTHROPIC_AUTH_TOKEN` and `ANTHROPIC_API_KEY` is
version-dependent — "which credential wins depends on the endpoint and tool version." The spike's
job is to confirm the installed SDK/CLI version emits **both** headers simultaneously (not one
clobbering the other). If it doesn't, fallback in §4.

### 2.2 Components

- **`deploy/litellm.config.yaml`** — model list (Anthropic entries), `general_settings` with the two
  BYOK flags. DB-less (no `database_url` / `DATABASE_URL`).
- **`deploy/litellm-proxy.service`** (+ `deploy/dev/litellm-proxy-dev.service`) — systemd unit
  running `litellm --config …`, bound to `127.0.0.1:4000`, mirroring `kraken-backend.service`.
- **`.github/workflows/deploy.yml`** — add install/restart of the proxy service. (Actual VM apply is
  Trent's to execute per project policy — this produces units + a PR, not a direct deploy.)
- **`byok.py`** — extend the single chokepoint (see §3).
- **3 call boundaries** — `agent.py:build_agent_options`, `graph/sdk_utils.py:query_with_usage`
  (used by `entity_resolution` + `integration` nodes), `semantic_scholar.py`'s direct query. Each
  adds `ANTHROPIC_BASE_URL` + `ANTHROPIC_AUTH_TOKEN` to its per-request `ClaudeAgentOptions(env=…)`
  (user key stays in `ANTHROPIC_API_KEY`), **and sets `cli_path` to the system `claude` binary** —
  `shutil.which("claude")` — because the *bundled* SDK binary ignores `ANTHROPIC_BASE_URL` from
  `env=…` (live issues #677/#1089). Design with `cli_path` from the start; don't wait to hit the 403.

## 3. The `byok.py` chokepoint extension

Preserve the single-chokepoint discipline #93 established (nodes never do precedence logic):

```python
current_api_key: ContextVar[str | None]  = ContextVar("current_api_key", default=None)
current_provider: ContextVar[str | None] = ContextVar("current_provider", default=None)  # NEW

def resolve_effective_key_and_provider(
    byok_key: str | None, verified_email: str | None
) -> tuple[str, str, str]:            # (key, provider, source)
    key, source = resolve_effective_key(byok_key, verified_email)  # existing precedence
    provider = "anthropic"            # pilot: fixed. Future: derive from key prefix / user choice.
    return key, provider, source
```

Precedence, trusted `phenomehealth.org` fallback, and fail-closed (`NeedsKeyError`) are unchanged.
`provider` is threaded through now (defaulted) so adding providers later touches config + this one
function, not the call sites.

## 4. Phase-0 spike (GATE — before any build)

The two load-bearing unknowns now have **documented expected behavior** (see §2.1 + refs), so the
spike is a *confirmation against a starting config*, not open exploration. Run on the **dev VM**.

**Start from this config (the expectation), don't rediscover it:**
```python
options = ClaudeAgentOptions(
    model="claude-sonnet-4-...",
    cli_path=shutil.which("claude"),          # system binary honors ANTHROPIC_BASE_URL (#677/#1089)
    env={
        "ANTHROPIC_BASE_URL":  "http://127.0.0.1:4000",
        "ANTHROPIC_AUTH_TOKEN": LITELLM_MASTER_KEY,   # → Authorization: Bearer (proxy-auth, stripped)
        "ANTHROPIC_API_KEY":    USER_ANTHROPIC_KEY,   # → x-api-key (forwarded upstream)
    },
)
```
litellm `general_settings`: `forward_client_headers_to_llm_api: true`,
`forward_llm_provider_auth_headers: true`.

**Spike exit criteria (all must pass):**
1. **Both headers emitted.** Confirm the installed SDK/CLI version sends `Authorization: Bearer`
   (proxy-auth) **and** `x-api-key` (user key) *simultaneously* — the documented "both set is
   version-dependent" caution is the #1 risk. Inspect at the proxy (log inbound headers) to prove
   neither clobbers the other.
2. **base_url honored via `cli_path`.** The turn reaches the proxy (no 403 / no direct-to-Anthropic
   bypass). If it bypasses, confirm `cli_path` → system binary fixes it; capture the failing case.
3. **User key is what bills.** Use a BYOK key the operator env does *not* have; the call succeeds via
   the forwarded `x-api-key`, and a deliberately-wrong key fails at Anthropic (not at the proxy) —
   proving the *user's* key reached upstream, not an operator key.
4. **Agentic behavior intact.** Tool-use + streaming survive through the proxy; check subagents still
   get the model (claude-code#5680) if the classic path uses them.
5. Request visible in proxy logs; **per-request `ClaudeAgentOptions(env=…)`** only — never
   `os.environ` (the naive tutorial's approach would break multi-session isolation).

**Version pin:** record the exact `claude-agent-sdk`, bundled-CLI, and `litellm` versions — the
header/base_url behavior is version-sensitive (cf. AssetOpsBench#275 "Extra inputs are not permitted").

**If the spike fails:** fall back to pointing `ANTHROPIC_BASE_URL` straight at Anthropic (or add a
thin pre-call hook) and **defer the proxy** — without sinking build cost into a dead path. Record the
outcome regardless (SOP: expensive-to-reproduce findings persist).

## 5. Config & secrets

| Name | Where | Purpose |
|---|---|---|
| `LITELLM_MASTER_KEY` | proxy env + backend env | Internal proxy-auth (stripped upstream). |
| `KRAKEN_LLM_BASE_URL` | backend env | Backend → proxy (`http://127.0.0.1:4000`). |
| `deploy/litellm.config.yaml` | repo | Model list + BYOK `general_settings`. |
| `SERVER_ANTHROPIC_API_KEY` | backend env | **Unchanged** — trusted-user fallback, now flows through the proxy. |

## 6. Testing

- **Unit** — `resolve_effective_key_and_provider`: provider defaults to `anthropic`; BYOK-key,
  trusted-fallback, and fail-closed paths all still hold (mirror existing `test_clerk_identity.py` /
  byok tests).
- **Integration smoke** — backend pointed at a local proxy completes one turn; reuse the existing
  pre-flight key-validation endpoint through the proxy path.
- **Spike artifact** — Phase-0 results written to disk (dev VM), pinned with the litellm + SDK
  versions used.

## 7. Out of scope (YAGNI)

- Virtual keys, budgets, spend UI, Postgres.
- Non-Anthropic providers (config-extensible **seam** only in the pilot).
- ddharmon (standing instruction: no PRs without explicit ask).
- The EITL port — that's the *next* app once this pattern proves out.

## 8. Risk register

| Risk | Mitigation |
|---|---|
| Bundled CLI ignores `ANTHROPIC_BASE_URL` / 403 (live issues #677, #1089) | **Design in `cli_path` → system binary from the start** (§2.2); spike criterion #2 confirms. |
| Header collision: proxy-auth vs forwarded key; "both set is version-dependent" | Expected mapping known (§2.1): `ANTHROPIC_AUTH_TOKEN`=proxy-auth, `ANTHROPIC_API_KEY`=user key; spike criterion #1 confirms both headers emit. |
| Version-sensitive header/base_url behavior | Pin `claude-agent-sdk` + bundled-CLI + `litellm` versions in the spike artifact (§4). |
| Proxy adds a failure point on the request path | Loopback-only bind; health check; fail-closed on proxy-down. |
| Per-session isolation broken by process-global env (naive tutorial uses `os.environ`) | Per-request `ClaudeAgentOptions(env=…)` only — the #93 discipline; never `os.environ`. |
| Proxy earns keep mostly as future-proofing in Anthropic-only pilot | Accepted deliberately; fallback defers proxy if spike is shaky. |

## References

- Claude Agent SDK with LiteLLM — https://docs.litellm.ai/docs/tutorials/claude_agent_sdk
- Forward Client Headers to LLM API (BYOK) — https://docs.litellm.ai/docs/proxy/forward_client_headers
- LiteLLM production / DB-less — https://docs.litellm.ai/docs/proxy/prod
- claude-agent-sdk-python#677 — https://github.com/anthropics/claude-agent-sdk-python/issues/677
- claude-code-action#1089 — https://github.com/anthropics/claude-code-action/issues/1089
- Claude Code Authentication (env→header) — https://code.claude.com/docs/en/authentication
- ANTHROPIC_API_KEY vs ANTHROPIC_AUTH_TOKEN / custom base URL — https://www.coderouter.io/blog/claude-code-401-custom-base-url-fix
- Subagents not getting custom model — https://github.com/anthropics/claude-code/issues/5680
