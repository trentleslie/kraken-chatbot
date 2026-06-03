"""Configuration management for Kestrel Backend."""

import os
from functools import lru_cache
from pydantic import BaseModel
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

    # Logging configuration
    log_level: str = "INFO"
    log_format: str = "text"  # "json" or "text"
    log_module_levels: dict[str, str] = {}  # Module-specific log levels


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
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_format=os.getenv("LOG_FORMAT", "text"),
        log_module_levels=module_levels,
    )
