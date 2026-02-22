# PR: Add Alembic Database Migration System (#22)

## Summary
Implements a complete Alembic migration system for KRAKEN database schema management with async PostgreSQL support, idempotent migrations, and automated deployment integration.

## Changes

### Core Implementation
- **SQLAlchemy Models** (`backend/src/kestrel_backend/models.py`)
  - Defined ORM models for `kraken_conversations`, `kraken_turns`, `kraken_tool_calls`
  - Uses SQLAlchemy 2.0 modern API with proper relationships and foreign keys

- **Alembic Configuration** (`backend/alembic/`)
  - `alembic.ini` - Configuration with logging setup
  - `env.py` - Async environment with PostgreSQL+asyncpg support
  - `script.py.mako` - Migration template
  - `versions/001_baseline_schema.py` - Idempotent baseline migration using `IF NOT EXISTS`

- **Migration Runner** (`backend/src/kestrel_backend/database.py`)
  - Added `run_migrations()` function that executes `alembic upgrade head`
  - Integrated into `init_db()` for automatic migration on startup
  - Graceful error handling (doesn't fail startup if migrations fail)

### Deployment Integration
- **GitHub Actions** (`.github/workflows/deploy.yml`)
  - Added `uv run alembic upgrade head` step before service restart
  - Ensures schema is always up-to-date on deployment

### Dependencies
- **pyproject.toml**
  - Added `alembic>=1.13.0` and `sqlalchemy>=2.0.0`
  - Added dev dependencies: `pytest>=8.0.0`, `pytest-asyncio>=0.23.0`

### Tests
- **test_migration_setup.py** - 12 tests validating migration structure (no DB required)
- **test_migrations.py** - Database integration tests for schema validation

### Documentation
- **alembic/README.md** - Comprehensive migration system documentation
- **.env.example** - Added `DATABASE_URL` configuration example

## Key Features

### Idempotent Migrations
The baseline migration uses `CREATE TABLE IF NOT EXISTS` and `CREATE INDEX IF NOT EXISTS`, making it safe to run on:
- Fresh databases (tables will be created)
- Existing production databases (tables already exist, no errors)

### Async Support
- Uses `postgresql+asyncpg://` dialect for async operations
- Matches the async pattern used throughout the application
- Alembic env.py runs migrations in async mode

### Automatic Execution
Migrations run automatically via:
1. GitHub Actions during deployment
2. Application startup as backup (via `init_db()`)

## Database Schema

Creates three tables with proper indexes and foreign keys:

1. **kraken_conversations** - Conversation metadata (session, model, tokens, cost)
2. **kraken_turns** - Individual turns (user query, assistant response, metrics)
3. **kraken_tool_calls** - Tool calls within turns (name, args, results)

## Testing

All migration setup tests pass:
```bash
cd backend
uv run pytest tests/test_migration_setup.py -v
# 12 passed in 0.34s
```

Tests validate:
- Configuration files exist and are valid
- Models import successfully
- Baseline migration structure is correct
- Idempotency (6+ IF NOT EXISTS clauses)
- SQLAlchemy metadata matches expected schema

## Production Safety

- **Idempotent**: Can be run multiple times without errors
- **Non-breaking**: If DATABASE_URL is not set, migrations are skipped with a warning
- **Graceful degradation**: Migration failures don't prevent app startup
- **Backward compatible**: Existing database.py functions unchanged

## Usage

### Development
```bash
cd backend
uv run alembic upgrade head      # Apply migrations
uv run alembic current          # Check status
uv run alembic history          # View migration history
```

### Production
Migrations run automatically during deployment. No manual intervention required.

## Files Changed

**New Files** (10):
- `backend/alembic.ini`
- `backend/alembic/env.py`
- `backend/alembic/script.py.mako`
- `backend/alembic/README.md`
- `backend/alembic/versions/001_baseline_schema.py`
- `backend/src/kestrel_backend/models.py`
- `backend/tests/test_migration_setup.py`
- `backend/tests/test_migrations.py`
- `MIGRATION_IMPLEMENTATION.md`
- `PR_DESCRIPTION.md`

**Modified Files** (4):
- `backend/pyproject.toml`
- `backend/src/kestrel_backend/database.py`
- `backend/.env.example`
- `.github/workflows/deploy.yml`

## Next Steps

After merge:
1. Deployment workflow will automatically run migrations on production
2. Future schema changes can be managed via `alembic revision -m "description"`
3. Consider migrating from raw asyncpg to SQLAlchemy ORM for queries (optional future enhancement)

## Related Issue

Closes #22
