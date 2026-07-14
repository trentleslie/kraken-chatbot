# LiteLLM-Proxy BYOK Pivot — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route kraken's Claude Agent SDK traffic through a DB-less LiteLLM proxy so per-session BYOK keys are forwarded upstream (cost-offload), laying an Anthropic-only seam that later extends to other providers by config.

**Architecture:** A LiteLLM proxy runs as a systemd sibling of `kraken-backend` on the Lightsail VM (127.0.0.1:4000, no Postgres). The backend points the SDK at it via per-request `ClaudeAgentOptions(env=…)` + `cli_path`→system binary. Proxy-auth rides `ANTHROPIC_AUTH_TOKEN`→`Authorization` (stripped); the user's key rides `ANTHROPIC_API_KEY`→`x-api-key` (forwarded). All three env-injection sites collapse onto one `byok.build_agent_env()` helper.

**Tech Stack:** Python 3.11+, FastAPI, `claude-agent-sdk`, `litellm[proxy]`, systemd, nginx, pytest.

**Spec:** `docs/brainstorms/2026-07-13-litellm-proxy-byok-design.md`

## Global Constraints

- **Never mutate `os.environ`** for keys/base-url — per-request `ClaudeAgentOptions(env=…)` + `ContextVar` only.
- **Single chokepoint:** all key/provider/env-building logic lives in `byok.py`; call sites never re-derive it.
- **`cli_path=shutil.which("claude")`** on every `ClaudeAgentOptions` that targets the proxy (bundled binary ignores `ANTHROPIC_BASE_URL` — issues #677/#1089).
- **Env→header mapping (verify in Task 2, then rely on):** `ANTHROPIC_AUTH_TOKEN`→`Authorization: Bearer` (proxy-auth, stripped); `ANTHROPIC_API_KEY`→`x-api-key` (user key, forwarded).
- **DB-less proxy:** no `DATABASE_URL`, no virtual keys, no budgets. Anthropic-only for the pilot; `provider` threaded but fixed to `"anthropic"`.
- **PR routing:** branch `litellm-proxy-byok` → PR base `dev` on `trentleslie/kraken-chatbot`.
- **Never deploy directly** — CI/systemd changes ship via PR; the VM apply is Trent's.
- Secrets never printed/committed (gitleaks pre-commit is active).

---

### Task 1: LiteLLM proxy config + local run (spike needs something to hit)

**Files:**
- Create: `deploy/litellm.config.yaml`
- Create: `deploy/README-litellm.md` (how to run locally + on the VM)
- Test: `backend/tests/test_litellm_config.py`

**Interfaces:**
- Produces: a runnable proxy at `http://127.0.0.1:4000` with BYOK forwarding enabled; config path `deploy/litellm.config.yaml`.

- [ ] **Step 1: Write the failing test** — assert the config declares BYOK forwarding, is DB-less, and lists an Anthropic model.

```python
# backend/tests/test_litellm_config.py
from pathlib import Path
import yaml

CONFIG = Path(__file__).resolve().parents[2] / "deploy" / "litellm.config.yaml"

def test_config_enables_byok_forwarding_and_is_dbless():
    cfg = yaml.safe_load(CONFIG.read_text())
    gs = cfg["general_settings"]
    assert gs["forward_client_headers_to_llm_api"] is True
    assert gs["forward_llm_provider_auth_headers"] is True   # BYOK
    assert "database_url" not in gs and "database_url" not in cfg  # DB-less
    names = [m["model_name"] for m in cfg["model_list"]]
    assert any("claude" in n for n in names)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_litellm_config.py -v`
Expected: FAIL — `FileNotFoundError` (config not created yet).

- [ ] **Step 3: Create the config**

```yaml
# deploy/litellm.config.yaml — DB-less Anthropic-only BYOK gateway
model_list:
  - model_name: claude-sonnet-4
    litellm_params:
      model: anthropic/claude-sonnet-4-20250514
      # No api_key here: the forwarded client x-api-key supplies it (BYOK).
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY      # proxy-auth (Authorization), stripped upstream
  forward_client_headers_to_llm_api: true
  forward_llm_provider_auth_headers: true         # 👈 forward client x-api-key to Anthropic (BYOK)
litellm_settings:
  drop_params: true
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_litellm_config.py -v`
Expected: PASS.

- [ ] **Step 5: Write the run doc** — `deploy/README-litellm.md`:

```markdown
## Run the LiteLLM proxy locally
    uv tool install 'litellm[proxy]'
    export LITELLM_MASTER_KEY=sk-proxy-local
    litellm --config deploy/litellm.config.yaml --port 4000
Health check: `curl -s http://127.0.0.1:4000/health/liveliness` → `"I'm alive!"`.
DB-less: no DATABASE_URL is set, so virtual keys / spend UI are intentionally unavailable.
```

- [ ] **Step 6: Commit**

```bash
git add deploy/litellm.config.yaml deploy/README-litellm.md backend/tests/test_litellm_config.py
git commit -m "feat(byok): add DB-less LiteLLM proxy config with client-key forwarding"
```

---

### Task 2: Phase-0 spike — GATE (confirm header behavior; go/no-go)

> **This task gates Tasks 3–7.** If it fails, STOP and escalate the fallback in the spec (§4) — do not build on a dead path.

**Files:**
- Create: `backend/scripts/spike_litellm_byok.py` (throwaway verification driver — committed for reproducibility)
- Create: `docs/solutions/integration-issues/litellm-proxy-byok-spike-2026-07-13.md` (outcome artifact — persisted by default per SOP)

**Interfaces:**
- Produces: a go/no-go decision + the *confirmed* env→header behavior + pinned versions, consumed as assumptions by Tasks 3–7.

- [ ] **Step 1: Start the proxy** (Task 1 config) with a master key, in a second terminal.

Run: `export LITELLM_MASTER_KEY=sk-proxy-local && litellm --config deploy/litellm.config.yaml --port 4000 --detailed_debug`
Expected: boots; `curl -s http://127.0.0.1:4000/health/liveliness` returns alive.

- [ ] **Step 2: Write the spike driver** — one real turn through the proxy with both keys set.

```python
# backend/scripts/spike_litellm_byok.py
import asyncio, os, shutil
from claude_agent_sdk import ClaudeAgentOptions, query

async def main():
    opts = ClaudeAgentOptions(
        model="claude-sonnet-4",
        cli_path=shutil.which("claude"),                      # dodge bundled-binary base_url bypass
        max_turns=1,
        env={
            "ANTHROPIC_BASE_URL":  "http://127.0.0.1:4000",
            "ANTHROPIC_AUTH_TOKEN": os.environ["LITELLM_MASTER_KEY"],  # → Authorization (proxy-auth)
            "ANTHROPIC_API_KEY":    os.environ["USER_ANTHROPIC_KEY"],  # → x-api-key (forwarded)
        },
    )
    async for msg in query(prompt="Reply with the single word: pong.", options=opts):
        print(type(msg).__name__, getattr(msg, "content", msg))

asyncio.run(main())
```

- [ ] **Step 3: Run it and read the proxy's `--detailed_debug` log**

Run: `USER_ANTHROPIC_KEY=sk-ant-… uv run python backend/scripts/spike_litellm_byok.py`
Confirm ALL of:
  1. Proxy log shows inbound **both** `authorization` and `x-api-key` headers (neither clobbers the other).
  2. Request reaches the proxy (no 403, no direct-to-Anthropic bypass). If bypassed → confirm `cli_path` is the system binary.
  3. Upstream call uses the **forwarded** `x-api-key` (the user key), not an operator key.
  4. Turn completes with tool-use/streaming intact.

- [ ] **Step 4: Negative check — prove the user key is what bills**

Run the driver with `USER_ANTHROPIC_KEY=sk-ant-INVALID`.
Expected: failure originates **at Anthropic** (401 upstream), not at the proxy — proving the user's key transited.

- [ ] **Step 5: Write the outcome artifact** (persist regardless of result)

```markdown
# LiteLLM-proxy BYOK spike — 2026-07-13
- Result: GO / NO-GO
- Versions: claude-agent-sdk==X, bundled CLI==Y, litellm==Z
- Confirmed: [both headers emit? cli_path needed? user key forwarded?]
- If NO-GO: which criterion failed + chosen fallback.
```
Frontmatter: `module: byok`, `tags: [litellm, proxy, byok, claude-agent-sdk]`, `problem_type: integration_issue`.

- [ ] **Step 6: Commit**

```bash
git add backend/scripts/spike_litellm_byok.py docs/solutions/integration-issues/litellm-proxy-byok-spike-2026-07-13.md
git commit -m "chore(byok): Phase-0 spike driver + outcome artifact (proxy header confirmation)"
```

**GATE:** proceed to Task 3 only if the artifact says GO.

---

### Task 3: `byok.py` — provider ContextVar + shared env/cli helpers

**Files:**
- Modify: `backend/src/kestrel_backend/byok.py`
- Test: `backend/tests/test_byok.py` (extend)

**Interfaces:**
- Consumes: existing `resolve_effective_key(byok_key, verified_email) -> (key, source)`, `current_api_key`.
- Produces:
  - `current_provider: ContextVar[str | None]`
  - `resolve_effective_key_and_provider(byok_key, verified_email) -> tuple[str, str, str]  # (key, provider, source)`
  - `build_agent_env() -> dict[str, str]` — the single source of truth for SDK env injection.
  - `system_cli_path() -> str | None`

- [ ] **Step 1: Write failing tests**

```python
# add to backend/tests/test_byok.py
import shutil
from kestrel_backend import byok

def test_resolve_returns_provider_anthropic():
    key, provider, source = byok.resolve_effective_key_and_provider("sk-user", "t@phenomehealth.org")
    assert (key, provider, source) == ("sk-user", "anthropic", "byok")

def test_build_agent_env_empty_when_no_key():
    byok.current_api_key.set(None)
    assert byok.build_agent_env() == {}

def test_build_agent_env_maps_keys(monkeypatch):
    s = byok.get_settings()
    monkeypatch.setattr(s, "litellm_master_key", "sk-proxy", raising=False)
    monkeypatch.setattr(s, "kraken_llm_base_url", "http://127.0.0.1:4000", raising=False)
    byok.current_api_key.set("sk-user")
    env = byok.build_agent_env()
    assert env["ANTHROPIC_API_KEY"] == "sk-user"          # user key → x-api-key (forwarded)
    assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-proxy"      # proxy-auth → Authorization (stripped)
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:4000"

def test_build_agent_env_direct_mode_when_no_base_url(monkeypatch):
    s = byok.get_settings()
    monkeypatch.setattr(s, "kraken_llm_base_url", "", raising=False)
    byok.current_api_key.set("sk-user")
    env = byok.build_agent_env()
    assert env == {"ANTHROPIC_API_KEY": "sk-user"}        # no proxy → legacy direct behavior

def test_system_cli_path_is_string_or_none():
    assert byok.system_cli_path() in (None, shutil.which("claude"))
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd backend && uv run pytest tests/test_byok.py -v`
Expected: FAIL — `AttributeError: module 'kestrel_backend.byok' has no attribute 'resolve_effective_key_and_provider'`.

- [ ] **Step 3: Implement in `byok.py`**

```python
import shutil
from contextvars import ContextVar

current_provider: ContextVar[str | None] = ContextVar("current_provider", default=None)

def resolve_effective_key_and_provider(
    byok_key: str | None, verified_email: str | None
) -> tuple[str, str, str]:
    key, source = resolve_effective_key(byok_key, verified_email)
    return key, "anthropic", source   # pilot: fixed provider; future: derive here

def system_cli_path() -> str | None:
    """System `claude` binary — the bundled SDK binary ignores ANTHROPIC_BASE_URL (#677/#1089)."""
    return shutil.which("claude")

def build_agent_env() -> dict[str, str]:
    """Single source of truth for SDK env injection. Empty when no key is set
    (classic/internal callers keep ambient env). When a proxy base URL is
    configured, route through it (user key → x-api-key forwarded; master key →
    Authorization/proxy-auth). Otherwise fall back to legacy direct-to-Anthropic."""
    key = current_api_key.get()
    if not key:
        return {}
    s = get_settings()
    base_url = getattr(s, "kraken_llm_base_url", "") or ""
    if not base_url:
        return {"ANTHROPIC_API_KEY": key}
    env = {"ANTHROPIC_API_KEY": key, "ANTHROPIC_BASE_URL": base_url}
    master = getattr(s, "litellm_master_key", "") or ""
    if master:
        env["ANTHROPIC_AUTH_TOKEN"] = master
    return env
```

Add settings fields in `config.py` `Settings`: `kraken_llm_base_url: str = ""` and `litellm_master_key: str = ""` (env: `KRAKEN_LLM_BASE_URL`, `LITELLM_MASTER_KEY`).

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && uv run pytest tests/test_byok.py -v`
Expected: PASS (all, including the pre-existing tests).

- [ ] **Step 5: Commit**

```bash
git add backend/src/kestrel_backend/byok.py backend/src/kestrel_backend/config.py backend/tests/test_byok.py
git commit -m "feat(byok): provider contextvar + shared build_agent_env/system_cli_path helpers"
```

---

### Task 4: Collapse the 3 injection sites onto the shared helper

**Files:**
- Modify: `backend/src/kestrel_backend/agent.py:359-383` (`build_agent_options`)
- Modify: `backend/src/kestrel_backend/graph/sdk_utils.py` (`create_agent_options`, `_apply_byok_env`)
- Modify: `backend/src/kestrel_backend/semantic_scholar.py:214` (inline `ClaudeAgentOptions`)
- Test: `backend/tests/test_byok_injection_classic.py`, `backend/tests/test_byok_injection_pipeline.py` (extend)

**Interfaces:**
- Consumes: `byok.build_agent_env()`, `byok.system_cli_path()` (Task 3).
- Produces: every `ClaudeAgentOptions` targeting the proxy carries the full env + `cli_path`.

- [ ] **Step 1: Extend the injection tests** — assert BASE_URL/AUTH_TOKEN + cli_path now flow.

```python
# test_byok_injection_classic.py — with proxy configured
def test_classic_options_route_through_proxy(monkeypatch):
    from kestrel_backend import byok, agent
    s = byok.get_settings()
    monkeypatch.setattr(s, "kraken_llm_base_url", "http://127.0.0.1:4000", raising=False)
    monkeypatch.setattr(s, "litellm_master_key", "sk-proxy", raising=False)
    byok.current_api_key.set("sk-user")
    opts = agent.build_agent_options()
    assert opts.env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:4000"
    assert opts.env["ANTHROPIC_AUTH_TOKEN"] == "sk-proxy"
    assert opts.env["ANTHROPIC_API_KEY"] == "sk-user"
    assert opts.cli_path == byok.system_cli_path()
```
Mirror an equivalent `test_pipeline_options_route_through_proxy` calling `sdk_utils.create_agent_options(...)` and asserting `_apply_byok_env` produces the same env.

- [ ] **Step 2: Run to verify they fail**

Run: `cd backend && uv run pytest tests/test_byok_injection_classic.py tests/test_byok_injection_pipeline.py -v`
Expected: FAIL — `KeyError: 'ANTHROPIC_BASE_URL'` (sites still hard-code only `ANTHROPIC_API_KEY`).

- [ ] **Step 3: Replace hard-coded env in all three sites with the helper**

`agent.py:build_agent_options` — replace the `key = current_api_key.get(); if key: options_kwargs["env"] = {...}` block:
```python
    env = build_agent_env()          # from kestrel_backend.byok
    if env:
        options_kwargs["env"] = env
    cli = system_cli_path()
    if cli:
        options_kwargs["cli_path"] = cli
```
`sdk_utils.create_agent_options` — same replacement for its `kwargs["env"]` block, plus `kwargs["cli_path"] = system_cli_path()` when set.
`sdk_utils._apply_byok_env` — replace body:
```python
def _apply_byok_env(options):
    env = build_agent_env()
    if env and options is not None:
        options.env = {**(getattr(options, "env", None) or {}), **env}
        cli = system_cli_path()
        if cli and not getattr(options, "cli_path", None):
            options.cli_path = cli
    return options
```
`semantic_scholar.py` — after building its `ClaudeAgentOptions`, pass it through `_apply_byok_env(options)` (import from `.graph.sdk_utils`) instead of setting env inline.
Add the imports `from .byok import build_agent_env, system_cli_path` (and in sdk_utils, from `..byok`).

- [ ] **Step 4: Run to verify pass** (new + all pre-existing injection tests)

Run: `cd backend && uv run pytest tests/test_byok_injection_classic.py tests/test_byok_injection_pipeline.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/kestrel_backend/agent.py backend/src/kestrel_backend/graph/sdk_utils.py backend/src/kestrel_backend/semantic_scholar.py backend/tests/
git commit -m "refactor(byok): route all SDK options through build_agent_env + cli_path (DRY the 3 sites)"
```

---

### Task 5: Thread `current_provider` at the WS call site

**Files:**
- Modify: `backend/src/kestrel_backend/main.py:1004` (resolution block)
- Test: `backend/tests/test_ws_byok_wiring.py` (extend)

**Interfaces:**
- Consumes: `byok.resolve_effective_key_and_provider`, `byok.current_provider`.

- [ ] **Step 1: Write failing test** — the WS handler sets `current_provider` alongside `current_api_key`.

```python
def test_ws_sets_provider_contextvar(monkeypatch):
    # drive the resolution branch with a trusted verified email + server key,
    # assert byok.current_provider.get() == "anthropic" after resolution.
    ...
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && uv run pytest tests/test_ws_byok_wiring.py -v`
Expected: FAIL — provider contextvar unset (`None`).

- [ ] **Step 3: Update the call site**

```python
key, provider, source = resolve_effective_key_and_provider(
    connection_api_keys.get(connection_id), verified)
...
tok = current_api_key.set(key)
prov_tok = current_provider.set(provider)
```
Reset `current_provider` in the same `finally` that resets `current_api_key` (`current_provider.reset(prov_tok)`).

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && uv run pytest tests/test_ws_byok_wiring.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/kestrel_backend/main.py backend/tests/test_ws_byok_wiring.py
git commit -m "feat(byok): set current_provider contextvar at the WS resolution boundary"
```

---

### Task 6: systemd unit + deploy wiring

**Files:**
- Create: `deploy/litellm-proxy.service`, `deploy/dev/litellm-proxy-dev.service`
- Modify: `.github/workflows/deploy.yml`, `.github/workflows/deploy-dev.yml`

**Interfaces:**
- Produces: a proxy managed like `kraken-backend.service`, started before the backend.

- [ ] **Step 1: Write the prod unit** (mirror `deploy/kraken-backend.service`)

```ini
# deploy/litellm-proxy.service
[Unit]
Description=LiteLLM proxy (kraken BYOK gateway)
After=network.target
[Service]
Type=simple
WorkingDirectory=/home/ubuntu/kraken-chatbot
EnvironmentFile=/home/ubuntu/kraken-chatbot/deploy/litellm.env      # LITELLM_MASTER_KEY only
ExecStart=/home/ubuntu/.local/bin/litellm --config deploy/litellm.config.yaml --port 4000
Restart=always
RestartSec=3
[Install]
WantedBy=multi-user.target
```
Dev variant: port 4000 on the dev host, `WorkingDirectory` per `deploy/dev/` convention. (Confirm the real deploy user/paths against the existing `kraken-backend.service` before finalizing.)

- [ ] **Step 2: Add install/restart to `deploy.yml`** — inside the SSH heredoc, before the backend restart:

```bash
sudo cp deploy/litellm-proxy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now litellm-proxy
until curl -sf http://127.0.0.1:4000/health/liveliness >/dev/null; do sleep 1; done
```
Set `KRAKEN_LLM_BASE_URL=http://127.0.0.1:4000` in the backend's environment file step. Mirror in `deploy-dev.yml`.

- [ ] **Step 3: Sanity-check the unit locally**

Run: `systemd-analyze verify deploy/litellm-proxy.service` (or `systemctl --user` dry parse)
Expected: no syntax errors.

- [ ] **Step 4: Commit**

```bash
git add deploy/litellm-proxy.service deploy/dev/litellm-proxy-dev.service .github/workflows/deploy.yml .github/workflows/deploy-dev.yml
git commit -m "feat(byok): run LiteLLM proxy as a systemd sibling; wire deploy to start it before backend"
```

---

### Task 7: Integration smoke — backend → local proxy → one turn

**Files:**
- Test: `backend/tests/test_litellm_integration_smoke.py` (marked `@pytest.mark.integration`, skipped without `RUN_LITELLM_SMOKE=1`)

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Write the smoke test** (opt-in; needs a running proxy + a real key)

```python
import os, pytest
pytestmark = pytest.mark.skipif(os.environ.get("RUN_LITELLM_SMOKE") != "1",
                                reason="requires local litellm proxy + real key")

def test_pipeline_turn_through_proxy(monkeypatch):
    from kestrel_backend import byok
    monkeypatch.setattr(byok.get_settings(), "kraken_llm_base_url", "http://127.0.0.1:4000", raising=False)
    monkeypatch.setattr(byok.get_settings(), "litellm_master_key", os.environ["LITELLM_MASTER_KEY"], raising=False)
    byok.current_api_key.set(os.environ["USER_ANTHROPIC_KEY"])
    # invoke the existing pre-flight key-validation endpoint / a single node and assert success
    ...
```

- [ ] **Step 2: Run it against a live local proxy**

Run: `RUN_LITELLM_SMOKE=1 LITELLM_MASTER_KEY=sk-proxy-local USER_ANTHROPIC_KEY=sk-ant-… uv run pytest tests/test_litellm_integration_smoke.py -v`
Expected: PASS; proxy log shows the forwarded `x-api-key`.

- [ ] **Step 3: Verify the default suite still skips it**

Run: `cd backend && uv run pytest -q`
Expected: full suite green; smoke test reported skipped.

- [ ] **Step 4: Commit + open PR**

```bash
git add backend/tests/test_litellm_integration_smoke.py
git commit -m "test(byok): opt-in integration smoke for backend->proxy->Anthropic"
git push -u origin litellm-proxy-byok
gh pr create --base dev --repo trentleslie/kraken-chatbot \
  --title "BYOK via LiteLLM proxy (Anthropic-only pilot, multi-provider seam)" \
  --body "Implements docs/plans/2026-07-13-litellm-proxy-byok.md. Phase-0 spike artifact: docs/solutions/integration-issues/litellm-proxy-byok-spike-2026-07-13.md."
```

---

## Self-Review

- **Spec coverage:** §2.2 components → Tasks 1,3,4,5,6; §2.1 two-key mapping → Tasks 1+3+4; §3 chokepoint → Task 3; §4 spike → Task 2 (gate); §5 config/secrets → Tasks 1,3,6; §6 testing → Tasks 3,4,5,7; §7 out-of-scope respected (no virtual keys/Postgres/other providers/ddharmon/EITL). ✅
- **Placeholder scan:** the only intentionally-open items are Task 5/7 test *bodies* driving existing WS/endpoint fixtures (the executor wires them to `test_ws_byok_wiring.py`'s existing harness) and the deploy user/paths in Task 6 (explicitly "confirm against `kraken-backend.service`"). All code-bearing steps carry real code.
- **Type consistency:** `resolve_effective_key_and_provider -> (key, provider, source)`, `build_agent_env() -> dict`, `system_cli_path() -> str|None`, `current_provider` used identically across Tasks 3–5. Settings fields `kraken_llm_base_url` / `litellm_master_key` consistent across Tasks 3,4,6,7. ✅
