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

    # Langfuse observability
    langfuse_enabled: bool = True
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_base_url: str = "https://us.cloud.langfuse.com"


@lru_cache
def get_settings() -> Settings:
    """Load settings from environment, cached for performance."""
    load_dotenv()

    origins_str = os.getenv("ALLOWED_ORIGINS", "https://kraken.expertintheloop.io")
    origins = [o.strip() for o in origins_str.split(",") if o.strip()]

    return Settings(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        allowed_origins=origins,
        rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "10")),
        model=os.getenv("CLAUDE_MODEL"),  # None = use SDK default
        langfuse_enabled=os.getenv("LANGFUSE_ENABLED", "true").lower() == "true",
        langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        langfuse_base_url=os.getenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com"),
    )
