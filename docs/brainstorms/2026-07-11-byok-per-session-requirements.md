# BYOK (Bring Your Own Key) — Per-Session Design

**Date:** 2026-07-11
**Pilot app:** kraken-chatbot
**Reuse targets:** expert-in-the-loop (web module); ddharmon (env-var note only)
**Status:** Design approved, pre-implementation

---

## 1. Motivation

Let people other than Trent use the LLM-backed apps without Trent paying for their
inference and without provisioning a key for each person.

Two drivers, in priority order:

1. **Cost offload** — the public / collaborators pay for their own inference.
2. **Distribution** — hand the apps to others without per-person key provisioning.

Explicitly **not** a driver: compliance / data-residency. Users' prompts do not need to
flow through their own provider account for legal reasons. This lowers the security bar:
we must not be reckless with the key, but we are not obligated to protect it at rest.

## 2. Scope

- **Pilot:** kraken-chatbot (most public-facing, richest LLM surface, real cost pressure).
- **Reuse:** extract an app-agnostic web BYOK module and later apply to expert-in-the-loop.
- **ddharmon:** out of the web pattern. It is a Python library/CLI — "BYOK" there is just
  reading `ANTHROPIC_API_KEY` from caller-supplied env/config. Documented in one paragraph,
  no UI, no session handling.

## 3. Key decisions (settled during brainstorm)

| # | Decision | Choice |
|---|----------|--------|
| 1 | Motivation | Cost offload + distribution (no compliance) |
| 2 | Persistence | **Per-session only** — no accounts, no DB, no at-rest encryption, no KMS |
| 3 | Rollout | Reusable web module; **kraken is the pilot**; port to EITL after |
| 4 | Provider | **Single Anthropic key** (forced by the Claude Agent SDK path) |
| 5 | No-key fallback | **Trusted users ride Trent's key; everyone else must BYOK** |
| 6 | Trusted definition | Any Clerk user with a **verified** `phenomehealth.org` email |

### Why single Anthropic key (decision 4)

kraken's synthesis pipeline runs on the **Claude Agent SDK** (`claude_agent_sdk.query`
with `ClaudeAgentOptions`; it spawns Claude Code as a subprocess reading creds from process
env). That path is Anthropic-specific and will not cleanly route through OpenRouter, so a
provider-aggregator key is not viable. OpenAI is **not** a meaningful cost center here —
embeddings / vector search run through Kestrel/SPOKE server-side, not OpenAI — so a single
Anthropic key covers the entire cost driver.

## 4. Architecture

```
Browser (key held in memory only — never localStorage/sessionStorage)
   │  POST /api/session/key   (TLS)
   ▼
Backend: validate key → store in server-side session (keyed to existing cookie)
   │        frontend discards its copy; the server session is the source of truth
   ▼
Every LLM request → resolve_api_key(request) → inject → Anthropic
```

No database, no at-rest encryption, no accounts. The key lives in the server session and
dies with it (TTL or explicit clear).

### 4.1 The one chokepoint: `resolve_api_key(request)`

A single function all LLM-invoking code must call. Precedence:

1. **Session BYOK key** present → use it. Source = `"byok"`.
2. else **Clerk identity has a verified `phenomehealth.org` email** → server default key.
   Source = `"server"`.
3. else → `401 needs-key`.

After this change, **no LLM path reads the key from process env directly** — they all ask
`resolve_api_key`.

> Security note on decision 6: the domain check must use Clerk's **verified primary email**,
> not any user-editable field, or the trust gate is spoofable.

### 4.2 Injection into kraken's two LLM paths (the real implementation lift)

- **Claude Agent SDK:** pass `env={"ANTHROPIC_API_KEY": key}` into `ClaudeAgentOptions`
  **per query**. Never mutate global `os.environ` — that races across concurrent sessions.
- **LangChain:** `ChatAnthropic(api_key=key)` per request (currently in
  `src/kestrel_backend/local_tools.py`).

Both currently read from process env / settings, so the bulk of the work is **threading a
per-request key context down into the pipeline** to reach these two call sites.

### 4.3 Reusable module boundary

- **App-agnostic (the module):**
  - frontend key-entry component + in-memory key handling
  - backend session-key store
  - `resolve_api_key` with the trusted-fallback rule
  - key validation
- **App-specific (thin adapter):** how each app *injects* the resolved key. kraken =
  SDK + LangChain; EITL = its own client. The module never knows how the key is used.

## 5. Security posture (matched to per-session, non-compliance)

- Key never in browser `localStorage` / `sessionStorage`. In memory during entry, then the
  server session.
- **Scrubbed from logs and Langfuse traces.** Langfuse is already wired into kraken
  (`sdk_utils.py`) — this is a concrete leak risk and must be explicitly prevented and tested.
- TLS required.
- Session TTL + explicit "clear key" button.
- No at-rest persistence → nothing to encrypt.

## 6. Validation & error UX

- **On submit:** one cheap Anthropic ping (e.g. a 1-token message) to confirm the key works
  before accepting it → instant feedback.
- **No key + untrusted user:** chat disabled, inline "enter your Anthropic key to start."
- **Key revoked / over-quota mid-session:** surface the provider error, clear the stored
  session key, re-prompt.
- **"Whose key am I on?" indicator:** show source (`your key` vs `server`) so trusted users
  know when they are spending Trent's budget.

## 7. Testing

- **Precedence unit tests** for `resolve_api_key`: byok > trusted-fallback > deny; verified
  vs unverified email; wrong domain.
- **Concurrency test:** two sessions with two different keys, assert no SDK-subprocess env
  bleed between them.
- **Log/trace scrub assertion:** the key never appears in logs or Langfuse payloads.
- **Validation happy/sad path.**
- **End-to-end:** paste → validate → synthesize → clear.

## 8. ddharmon note

ddharmon reads `ANTHROPIC_API_KEY` from caller-supplied environment or config. No settings
UI, no session handling. This is the entirety of its "BYOK" story and is already how a
library should behave — document the convention, add nothing.

## 9. Out of scope (YAGNI)

Accounts · remembered keys · at-rest encryption / KMS · metering / free tier ·
OpenRouter / multi-provider · OpenAI BYOK.

These are deliberately excluded for the first cut. If a "remembered key" experience is
wanted later, that is a separate design (it reintroduces accounts + encrypted storage).
