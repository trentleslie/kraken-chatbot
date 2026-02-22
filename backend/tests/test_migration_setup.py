"""Tests for Alembic migration setup (no database required)."""
import os
from pathlib import Path
import pytest


def test_alembic_ini_exists():
    """Test that alembic.ini configuration file exists."""
    backend_dir = Path(__file__).parent.parent
    alembic_ini = backend_dir / "alembic.ini"
    assert alembic_ini.exists(), "alembic.ini not found"


def test_alembic_env_exists():
    """Test that alembic/env.py exists."""
    backend_dir = Path(__file__).parent.parent
    env_py = backend_dir / "alembic" / "env.py"
    assert env_py.exists(), "alembic/env.py not found"


def test_alembic_script_template_exists():
    """Test that alembic script template exists."""
    backend_dir = Path(__file__).parent.parent
    template = backend_dir / "alembic" / "script.py.mako"
    assert template.exists(), "alembic/script.py.mako not found"


def test_baseline_migration_exists():
    """Test that baseline migration file exists."""
    backend_dir = Path(__file__).parent.parent
    versions_dir = backend_dir / "alembic" / "versions"
    assert versions_dir.exists(), "alembic/versions directory not found"

    # Check for baseline migration
    baseline_migration = versions_dir / "001_baseline_schema.py"
    assert baseline_migration.exists(), "Baseline migration not found"


def test_models_file_exists():
    """Test that SQLAlchemy models file exists."""
    backend_dir = Path(__file__).parent.parent
    models_file = backend_dir / "src" / "kestrel_backend" / "models.py"
    assert models_file.exists(), "models.py not found"


def test_models_import():
    """Test that models can be imported without errors."""
    try:
        from kestrel_backend.models import Base, KrakenConversation, KrakenTurn, KrakenToolCall
        assert Base is not None
        assert KrakenConversation is not None
        assert KrakenTurn is not None
        assert KrakenToolCall is not None
    except ImportError as e:
        pytest.fail(f"Failed to import models: {e}")


def test_baseline_migration_has_upgrade():
    """Test that baseline migration has upgrade function."""
    backend_dir = Path(__file__).parent.parent
    baseline_migration = backend_dir / "alembic" / "versions" / "001_baseline_schema.py"

    content = baseline_migration.read_text()
    assert "def upgrade()" in content, "upgrade() function not found in baseline migration"
    assert "CREATE TABLE IF NOT EXISTS kraken_conversations" in content, "kraken_conversations table creation not found"
    assert "CREATE TABLE IF NOT EXISTS kraken_turns" in content, "kraken_turns table creation not found"
    assert "CREATE TABLE IF NOT EXISTS kraken_tool_calls" in content, "kraken_tool_calls table creation not found"


def test_baseline_migration_has_downgrade():
    """Test that baseline migration has downgrade function."""
    backend_dir = Path(__file__).parent.parent
    baseline_migration = backend_dir / "alembic" / "versions" / "001_baseline_schema.py"

    content = baseline_migration.read_text()
    assert "def downgrade()" in content, "downgrade() function not found in baseline migration"
    assert "drop_table" in content, "Table drops not found in downgrade"


def test_baseline_migration_is_idempotent():
    """Test that baseline migration uses IF NOT EXISTS for idempotency."""
    backend_dir = Path(__file__).parent.parent
    baseline_migration = backend_dir / "alembic" / "versions" / "001_baseline_schema.py"

    content = baseline_migration.read_text()
    # Count IF NOT EXISTS clauses
    if_not_exists_count = content.count("IF NOT EXISTS")

    # Should have IF NOT EXISTS for 3 tables + 3 indexes = 6 total
    assert if_not_exists_count >= 6, f"Expected at least 6 'IF NOT EXISTS' clauses, found {if_not_exists_count}"


def test_database_py_has_run_migrations():
    """Test that database.py has run_migrations function."""
    backend_dir = Path(__file__).parent.parent
    database_file = backend_dir / "src" / "kestrel_backend" / "database.py"

    content = database_file.read_text()
    assert "async def run_migrations()" in content, "run_migrations() function not found in database.py"
    assert "alembic upgrade head" in content, "alembic upgrade head command not found"


def test_dependencies_installed():
    """Test that required dependencies are available."""
    try:
        import alembic
        import sqlalchemy
        assert alembic is not None
        assert sqlalchemy is not None
    except ImportError as e:
        pytest.fail(f"Required dependencies not installed: {e}")


def test_sqlalchemy_models_metadata():
    """Test that SQLAlchemy models have proper metadata."""
    from kestrel_backend.models import Base, KrakenConversation, KrakenTurn, KrakenToolCall

    # Check that all models are in the metadata
    table_names = {table.name for table in Base.metadata.tables.values()}
    assert "kraken_conversations" in table_names
    assert "kraken_turns" in table_names
    assert "kraken_tool_calls" in table_names

    # Check that models have correct table names
    assert KrakenConversation.__tablename__ == "kraken_conversations"
    assert KrakenTurn.__tablename__ == "kraken_turns"
    assert KrakenToolCall.__tablename__ == "kraken_tool_calls"
