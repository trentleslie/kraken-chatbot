"""Alembic migration environment using synchronous psycopg2.

Uses psycopg2 (sync) instead of asyncpg (async) for migrations because:
1. Migrations are one-off DDL operations that don't need async performance
2. asyncpg can have issues with event loop creation in SSH/CI environments
3. Sync migrations are simpler and more reliable for deployment pipelines
"""
import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# Import the SQLAlchemy Base and models
import sys
from pathlib import Path

# Add the src directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir / "src"))

from kestrel_backend.models import Base

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# Get database URL from environment variable
database_url = os.environ.get("DATABASE_URL")
if database_url:
    # Use psycopg2 (sync) for migrations - more reliable in CI/SSH environments
    # Replace any async driver specification with psycopg2
    if "+asyncpg" in database_url:
        database_url = database_url.replace("+asyncpg", "+psycopg2")
    elif database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    config.set_main_option("sqlalchemy.url", database_url)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode using synchronous psycopg2."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
