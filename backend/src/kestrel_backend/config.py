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

    # Claude model
    model: str = "claude-sonnet-4-5-20250514"


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
        model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250514"),
    )
