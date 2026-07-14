"""Per-session BYOK key resolution. No persistence, no os.environ mutation."""
import shutil
from contextvars import ContextVar
from .config import get_settings

# The resolved effective key for the active request (byok or server). Option
# builders in agent.py and sdk_utils.py read this; nodes never do precedence logic.
current_api_key: ContextVar[str | None] = ContextVar("current_api_key", default=None)

# The resolved effective provider for the active request. Fixed to "anthropic" for this
# pilot; threaded through now so a future multi-provider pivot doesn't need a second plumbing
# pass through call sites.
current_provider: ContextVar[str | None] = ContextVar("current_provider", default=None)


class NeedsKeyError(Exception):
    """Untrusted user made an LLM request without providing a BYOK key."""


def is_trusted_email(verified_email: str | None) -> bool:
    if not verified_email or "@" not in verified_email:
        return False
    domain = verified_email.rsplit("@", 1)[-1].lower()
    trusted = {d.lower() for d in get_settings().byok_trusted_email_domains}
    return domain in trusted


def resolve_effective_key(byok_key: str | None, verified_email: str | None) -> tuple[str, str]:
    if byok_key:
        return byok_key, "byok"
    if is_trusted_email(verified_email):
        server_key = get_settings().server_anthropic_api_key
        if not server_key:
            raise NeedsKeyError("Trusted user but no server key configured.")
        return server_key, "server"
    raise NeedsKeyError("No API key provided and user is not trusted.")


def resolve_effective_key_and_provider(
    byok_key: str | None, verified_email: str | None
) -> tuple[str, str, str]:
    """Wraps resolve_effective_key with the (currently fixed) provider.

    Precedence, trusted fallback, and fail-closed behavior are unchanged — this only adds
    the provider slot for future multi-provider support (pilot: fixed to "anthropic").
    """
    key, source = resolve_effective_key(byok_key, verified_email)
    return key, "anthropic", source  # pilot: fixed provider; future: derive here


def system_cli_path() -> str | None:
    """System `claude` binary — the bundled SDK binary ignores ANTHROPIC_BASE_URL (#677/#1089).

    Used directly by the spike driver. SDK call sites should use `agent_cli_path()` instead,
    which only applies this override in proxy mode (keeping the legacy no-proxy path
    equivalent to the merged #93 behavior).
    """
    return shutil.which("claude")


def agent_cli_path() -> str | None:
    """cli_path override for SDK call sites — proxy mode only.

    The system `claude` binary override exists solely to make ANTHROPIC_BASE_URL work
    (#677/#1089). In the legacy no-proxy path there's no base URL to honor, so the bundled
    SDK binary is fine and we return None to keep that path equivalent to the merged #93
    behavior. In proxy mode, raises if the system `claude` binary is unavailable, since the
    bundled SDK binary ignores ANTHROPIC_BASE_URL and would silently bypass the proxy.
    """
    s = get_settings()
    if not (s.kraken_llm_base_url or ""):
        return None
    cli = system_cli_path()
    if not cli:
        raise RuntimeError(
            "KRAKEN_LLM_BASE_URL is set but the system `claude` binary is unavailable on PATH; "
            "the bundled SDK binary ignores ANTHROPIC_BASE_URL and would bypass the proxy."
        )
    return cli


def build_agent_env() -> dict[str, str]:
    """Single source of truth for SDK env injection.

    Empty when no key is set (classic/internal callers keep ambient env). When a proxy base
    URL is configured, route through it: the user's key is forwarded upstream as `x-api-key`
    via ANTHROPIC_API_KEY, while the proxy-auth key rides ANTHROPIC_AUTH_TOKEN (Authorization:
    Bearer) and is stripped by the proxy before forwarding. Otherwise falls back to the legacy
    direct-to-Anthropic shape (backward-compatible with the merged #93 behavior).

    Raises RuntimeError if proxy mode is configured (base_url set) but the master key is
    missing — silently omitting ANTHROPIC_AUTH_TOKEN would make every request 401 against
    the proxy, which is worse than failing fast at startup/request time.
    """
    key = current_api_key.get()
    if not key:
        return {}
    s = get_settings()
    base_url = s.kraken_llm_base_url or ""
    if not base_url:
        return {"ANTHROPIC_API_KEY": key}
    master = s.litellm_master_key or ""
    if not master:
        raise RuntimeError(
            "KRAKEN_LLM_BASE_URL is set but LITELLM_MASTER_KEY is missing — proxy mode "
            "requires the proxy-auth master key."
        )
    return {"ANTHROPIC_API_KEY": key, "ANTHROPIC_BASE_URL": base_url, "ANTHROPIC_AUTH_TOKEN": master}
