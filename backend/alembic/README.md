# KRAKEN Database Migrations

This directory contains Alembic database migrations for the KRAKEN chatbot application.

## Overview

The migration system uses Alembic with async PostgreSQL support (asyncpg) to manage database schema changes. The baseline migration uses `IF NOT EXISTS` clauses to ensure idempotency, which is important for production deployments where tables may already exist.

## Structure

```
alembic/
├── README.md                    # This file
├── env.py                       # Alembic environment configuration (async support)
├── script.py.mako               # Template for new migration files
└── versions/                    # Migration version files
    └── 001_baseline_schema.py   # Initial schema (conversations, turns, tool_calls)
```

## Database Schema

The schema tracks KRAKEN conversation metadata, user/assistant turns, and tool call details:

### Tables

1. **kraken_conversations** - Top-level conversation metadata
   - `id` (UUID, PK)
   - `session_id` (VARCHAR, indexed)
   - `model` (VARCHAR)
   - `system_prompt_hash` (VARCHAR(16))
   - `agent_version` (VARCHAR)
   - `status` (VARCHAR, default: 'active')
   - `started_at` (TIMESTAMP, default: NOW())
   - `ended_at` (TIMESTAMP, nullable)
   - `total_turns` (INTEGER, default: 0)
   - `total_tokens` (INTEGER, default: 0)
   - `total_cost_usd` (NUMERIC(10,6), default: 0)

2. **kraken_turns** - Individual conversation turns
   - `id` (UUID, PK)
   - `conversation_id` (UUID, FK → kraken_conversations)
   - `turn_number` (INTEGER)
   - `user_query` (TEXT)
   - `assistant_response` (TEXT)
   - `input_tokens` (INTEGER, default: 0)
   - `output_tokens` (INTEGER, default: 0)
   - `cost_usd` (NUMERIC(10,6), default: 0)
   - `duration_ms` (INTEGER, default: 0)
   - `tool_calls_count` (INTEGER, default: 0)
   - `cache_creation_tokens` (INTEGER, default: 0)
   - `cache_read_tokens` (INTEGER, default: 0)
   - `model` (VARCHAR)
   - `created_at` (TIMESTAMP, default: NOW())

3. **kraken_tool_calls** - Tool calls within turns
   - `id` (UUID, PK)
   - `turn_id` (UUID, FK → kraken_turns)
   - `tool_name` (VARCHAR)
   - `tool_args` (JSONB)
   - `tool_result` (JSONB)
   - `result_truncated` (BOOLEAN, default: FALSE)
   - `sequence_order` (INTEGER)

## Usage

### Running Migrations

Migrations are automatically run during application startup via `database.py:run_migrations()`. You can also run them manually:

```bash
# Apply all pending migrations
cd backend
uv run alembic upgrade head

# Check current migration status
uv run alembic current

# View migration history
uv run alembic history

# Rollback one migration
uv run alembic downgrade -1
```

### Creating New Migrations

To create a new migration:

```bash
cd backend
uv run alembic revision -m "description of changes"
```

For auto-generated migrations based on model changes:

```bash
uv run alembic revision --autogenerate -m "description of changes"
```

**Note**: Always review auto-generated migrations before applying them.

### Environment Variables

Migrations require the `DATABASE_URL` environment variable:

```bash
export DATABASE_URL="postgresql://user:password@localhost:5432/kraken_db"
```

The migration system automatically converts `postgresql://` to `postgresql+asyncpg://` for async support.

## Production Deployment

The GitHub Actions deployment workflow automatically runs migrations before restarting the backend service:

```bash
~/.local/bin/uv run alembic upgrade head
```

The baseline migration's use of `IF NOT EXISTS` ensures migrations are safe to run on existing production databases.

## Testing

Migration tests are in `tests/test_migration_setup.py` and `tests/test_migrations.py`:

```bash
# Run migration setup tests (no database required)
cd backend
uv run pytest tests/test_migration_setup.py -v

# Run database migration tests (requires DATABASE_URL)
uv run pytest tests/test_migrations.py -v
```

## Troubleshooting

### Migration fails with "alembic command not found"

Ensure dependencies are installed:
```bash
cd backend
uv sync
```

### Migration fails with "Expected string or URL object, got None"

Set the `DATABASE_URL` environment variable:
```bash
export DATABASE_URL="postgresql://user:password@localhost:5432/kraken_db"
```

### Tables already exist

This is expected. The baseline migration uses `CREATE TABLE IF NOT EXISTS` to handle existing tables gracefully.

## Further Reading

- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [SQLAlchemy 2.0 Documentation](https://docs.sqlalchemy.org/)
- [asyncpg Documentation](https://magicstack.github.io/asyncpg/)
