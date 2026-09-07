"""Exercise migration dispatch; PostgreSQL schema semantics need DB tests."""

from unittest.mock import Mock, patch

from alembic import op
from pytest import raises

from avalan.task.stores.pgsql_migrations.versions import (
    v20260907_0002_triggers as migration,
)


def test_trigger_migration_executes_every_statement_in_order() -> None:
    bind = Mock()
    with patch.object(op, "get_bind", return_value=bind):
        migration.upgrade()
    assert (
        tuple(call.args[0] for call in bind.exec_driver_sql.call_args_list)
        == migration.TASK_SCHEMA_STATEMENTS
    )


def test_trigger_migration_does_not_swallow_database_failure() -> None:
    bind = Mock()
    bind.exec_driver_sql.side_effect = RuntimeError("database failure")
    with (
        patch.object(op, "get_bind", return_value=bind),
        raises(RuntimeError, match="database failure"),
    ):
        migration.upgrade()
    assert bind.exec_driver_sql.call_count == 1


def test_trigger_migration_cannot_drop_deduplication_history() -> None:
    with raises(NotImplementedError, match="forward-only"):
        migration.downgrade()
