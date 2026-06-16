"""Configuration management for Kestrel Backend."""

import os
from functools import lru_cache
from pydantic import BaseModel, model_validator
from dotenv import load_dotenv


class Settings(BaseModel):
    """Application settings loaded from environment variables."""

    # Server settings
    host: str = "127.0.0.1"
    port: int = 8000

    # CORS
    allowed_origins: list[str] = ["https://kraken.expertintheloop.io"]

    # Rate limiting
    rate_limit_per_minute: int = 10

    # Claude model (None = use SDK default)
    model: str | None = None

    # Clerk authentication
    clerk_auth_enabled: bool = False
    clerk_secret_key: str | None = None
    clerk_jwks_url: str | None = None
    clerk_issuer: str | None = None
    clerk_proxy_url: str | None = None  # e.g., https://dev-kraken.expertintheloop.io/api/__clerk
    clerk_allowed_email_domains: list[str] = []
    clerk_allowed_emails: list[str] = []

    # Langfuse observability
    langfuse_enabled: bool = True
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_base_url: str = "https://us.cloud.langfuse.com"

    # Biomapper2 entity resolution (R7: secrets via env; flag/policy live in pipeline_config).
    # biomapper_base_url=None means "use the biomapper client default" (prod HTTPS instance).
    # biomapper_dev_base_url backs the prod/dev API toggle (mirrors biomapper-ui env routing).
    biomapper_api_key: str | None = None
    biomapper_base_url: str | None = None
    biomapper_dev_base_url: str | None = None

    # Logging configuration
    log_level: str = "INFO"
    log_format: str = "text"  # "json" or "text"
    log_module_levels: dict[str, str] = {}  # Module-specific log levels

    @model_validator(mode="after")
    def _enforce_biomapper_https(self) -> "Settings":
        """Never transmit the Biomapper API key over plaintext HTTP.

        Reject an http:// base URL when a key is set so the credential is never sent in the
        clear. A None base URL (client default) and HTTPS URLs are both fine.
        """
        if self.biomapper_api_key:
            for field, url in (
                ("BIOMAPPER_BASE_URL", self.biomapper_base_url),
                ("BIOMAPPER_DEV_BASE_URL", self.biomapper_dev_base_url),
            ):
                if url and url.lower().startswith("http://"):
                    raise ValueError(
                        f"{field} must use HTTPS when BIOMAPPER_API_KEY is set "
                        "(refusing to send the API key over plaintext HTTP)."
                    )
        return self


@lru_cache
def get_settings() -> Settings:
    """Load settings from environment, cached for performance."""
    load_dotenv()

    # Reconcile Langfuse host: the v3 client (get_client()) reads LANGFUSE_HOST from the
    # environment, but our config historically exposed LANGFUSE_BASE_URL. Without this,
    # LANGFUSE_BASE_URL is dead config and the client falls back to the EU default host.
    # If LANGFUSE_HOST is unset, derive it from LANGFUSE_BASE_URL (default: US cloud) so the
    # correct region is always used. An explicitly set LANGFUSE_HOST always wins.
    if not os.getenv("LANGFUSE_HOST"):
        os.environ["LANGFUSE_HOST"] = os.getenv(
            "LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com"
        )

    origins_str = os.getenv("ALLOWED_ORIGINS", "https://kraken.expertintheloop.io")
    origins = [o.strip() for o in origins_str.split(",") if o.strip()]

    # Parse module-specific log levels from env var (format: "module1:DEBUG,module2:INFO")
    module_levels = {}
    module_levels_str = os.getenv("LOG_MODULE_LEVELS", "")
    if module_levels_str:
        for item in module_levels_str.split(","):
            if ":" in item:
                module, level = item.split(":", 1)
                module_levels[module.strip()] = level.strip()

    # Parse Clerk allowed email domains and emails
    clerk_domains = []
    clerk_domains_str = os.getenv("ALLOWED_EMAIL_DOMAINS", "")
    if clerk_domains_str:
        clerk_domains = [d.strip().lower() for d in clerk_domains_str.split(",") if d.strip()]

    clerk_emails = []
    clerk_emails_str = os.getenv("ALLOWED_EMAILS", "")
    if clerk_emails_str:
        clerk_emails = [e.strip().lower() for e in clerk_emails_str.split(",") if e.strip()]

    # Clerk auth: fail-closed design. If CLERK_AUTH_ENABLED=true but CLERK_SECRET_KEY is missing,
    # we log a critical warning at startup. The app will start but auth will reject all requests.
    clerk_auth_enabled = os.getenv("CLERK_AUTH_ENABLED", "false").lower() == "true"
    clerk_secret_key = os.getenv("CLERK_SECRET_KEY")

    return Settings(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        allowed_origins=origins,
        rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "10")),
        model=os.getenv("CLAUDE_MODEL"),  # None = use SDK default
        clerk_auth_enabled=clerk_auth_enabled,
        clerk_secret_key=clerk_secret_key,
        clerk_jwks_url=os.getenv("CLERK_JWKS_URL"),
        clerk_issuer=os.getenv("CLERK_ISSUER"),
        clerk_proxy_url=os.getenv("CLERK_PROXY_URL"),
        clerk_allowed_email_domains=clerk_domains,
        clerk_allowed_emails=clerk_emails,
        langfuse_enabled=os.getenv("LANGFUSE_ENABLED", "true").lower() == "true",
        langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        langfuse_base_url=os.getenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com"),
        biomapper_api_key=os.getenv("BIOMAPPER_API_KEY"),
        biomapper_base_url=os.getenv("BIOMAPPER_BASE_URL"),
        biomapper_dev_base_url=os.getenv("BIOMAPPER_DEV_BASE_URL"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_format=os.getenv("LOG_FORMAT", "text"),
        log_module_levels=module_levels,
    )


_VALID_BIOMAPPER_ENVS = {"production", "dev"}


def resolve_biomapper_base_url(env: str | None, settings: "Settings") -> str | None:
    """Resolve the biomapper2 base URL for a requested environment (the prod/dev toggle).

    Mirrors biomapper-ui's env routing. Returns the base URL, where None means "use the biomapper
    client default" (the prod instance). ``env`` is the per-request toggle value (``"production"``
    default, or ``"dev"``).

    Raises ValueError on an unknown env name, or when ``dev`` is requested but
    ``BIOMAPPER_DEV_BASE_URL`` is not configured — the caller maps this to a user-facing error.
    """
    name = (env or "production").strip().lower()
    if name not in _VALID_BIOMAPPER_ENVS:
        raise ValueError(f"Invalid biomapper env '{env}'. Valid values: production, dev")
    if name == "dev":
        if not settings.biomapper_dev_base_url:
            raise ValueError("Dev biomapper API is not configured (BIOMAPPER_DEV_BASE_URL not set)")
        return settings.biomapper_dev_base_url
    return settings.biomapper_base_url  # None = client default (prod)


def biomapper_misconfig_reason(enabled: bool, api_key: str | None) -> str | None:
    """Return a CRITICAL-log reason if the Biomapper pre-resolver is misconfigured, else None.

    The enable flag lives in pipeline_config (tuning) and the secret in Settings (deployment),
    so this pure helper cross-checks both without coupling the two modules. Mirrors the Clerk
    fail-closed guard: enabled-but-unconfigured surfaces loudly at startup instead of as a
    BioMapperAuthError on every live request. Returns None (no-op) when the flag is off.
    """
    if enabled and not api_key:
        return (
            "biomapper.enabled=true but BIOMAPPER_API_KEY is unset. Every live Biomapper "
            "resolution will raise BioMapperAuthError and fall back to Kestrel. "
            "Set BIOMAPPER_API_KEY or disable the biomapper flag."
        )
    return None
