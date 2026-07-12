"""Per-session BYOK key resolution. No persistence, no os.environ mutation."""
from contextvars import ContextVar
from .config import get_settings

# The resolved effective key for the active request (byok or server). Option
# builders in agent.py and sdk_utils.py read this; nodes never do precedence logic.
current_api_key: ContextVar[str | None] = ContextVar("current_api_key", default=None)


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
