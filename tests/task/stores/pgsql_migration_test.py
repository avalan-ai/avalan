from collections.abc import Mapping
from os import environ
from typing import Any
from unittest import IsolatedAsyncioTestCase, TestCase, main
from uuid import uuid4

from pytest import importorskip

from avalan.task import TaskArtifactPurpose, TaskAttemptState, TaskRunState
from avalan.task.store import TaskStoreConflictError
from avalan.task.stores import (
    TASK_PGSQL_MIGRATIONS,
    PgsqlTaskMigration,
    PgsqlTaskMigrationRunner,
    task_pgsql_schema_statements,
)


class FakeDatabase:
    def __init__(self) -> None:
        self.migrations: dict[int, dict[str, object]] = {}
        self.executed: list[str] = []

    def connection(self) -> "FakeConnectionContext":
        return FakeConnectionContext(self)


class FakeConnectionContext:
    def __init__(self, database: FakeDatabase) -> None:
        self.database = database

    async def __aenter__(self) -> "FakeConnection":
        return FakeConnection(self.database)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool:
        return False


class FakeConnection:
    def __init__(self, database: FakeDatabase) -> None:
        self.database = database

    def cursor(self) -> "FakeCursorContext":
        return FakeCursorContext(self.database)


class FakeCursorContext:
    def __init__(self, database: FakeDatabase) -> None:
        self.database = database

    async def __aenter__(self) -> "FakeCursor":
        return FakeCursor(self.database)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool:
        return False


class FakeCursor:
    def __init__(self, database: FakeDatabase) -> None:
        self.database = database
        self.row: Mapping[str, object] | None = None

    async def execute(
        self,
        query: str,
        parameters: tuple[object, ...] | None = None,
    ) -> None:
        self.database.executed.append(query)
        if "SELECT" in query and "task_schema_migrations" in query:
            assert parameters is not None
            self.row = self.database.migrations.get(int(parameters[0]))
            return
        if "INSERT INTO" in query and "task_schema_migrations" in query:
            assert parameters is not None
            version = int(parameters[0])
            self.database.migrations[version] = {
                "version": version,
                "name": parameters[1],
                "checksum": parameters[2],
            }
            self.row = None
            return
        self.row = None

    async def fetchone(self) -> Mapping[str, object] | None:
        return self.row


class PgsqlMigrationSchemaTest(TestCase):
    def test_lifecycle_schema_has_expected_tables_and_constraints(
        self,
    ) -> None:
        schema = "\n".join(task_pgsql_schema_statements())

        for table_name in (
            "task_schema_migrations",
            "task_definitions",
            "task_runs",
            "task_run_transitions",
            "task_attempts",
            "task_attempt_transitions",
            "task_artifacts",
        ):
            self.assertIn(f'"{table_name}"', schema)

        self.assertIn('"uq_task_definitions_identity"', schema)
        self.assertIn('"uq_task_attempts_run_order"', schema)
        self.assertIn('"uq_task_attempts_one_active_per_run"', schema)
        self.assertIn(
            "WHERE \"state\" NOT IN ('succeeded', 'failed', 'abandoned')",
            schema,
        )
        self.assertIn('"task_reject_terminal_run_state_change"', schema)
        self.assertIn('"tr_task_runs_terminal_state"', schema)

    def test_schema_covers_current_state_and_artifact_vocabularies(
        self,
    ) -> None:
        schema = "\n".join(task_pgsql_schema_statements())

        for state in TaskRunState:
            self.assertIn(f"'{state.value}'", schema)
        for state in TaskAttemptState:
            self.assertIn(f"'{state.value}'", schema)
        for purpose in TaskArtifactPurpose:
            self.assertIn(f"'{purpose.value}'", schema)

    def test_migration_metadata_is_stable(self) -> None:
        self.assertEqual(
            [migration.version for migration in TASK_PGSQL_MIGRATIONS],
            [1],
        )
        self.assertEqual(TASK_PGSQL_MIGRATIONS[0].name, "task_lifecycle")
        self.assertEqual(len(TASK_PGSQL_MIGRATIONS[0].checksum), 64)


class PgsqlMigrationRunnerTest(IsolatedAsyncioTestCase):
    async def test_applies_pending_migrations_once(self) -> None:
        database = FakeDatabase()
        runner = PgsqlTaskMigrationRunner(database)

        applied = await runner.apply()
        first_execute_count = len(database.executed)
        applied_again = await runner.apply()

        self.assertEqual(applied, TASK_PGSQL_MIGRATIONS)
        self.assertEqual(applied_again, ())
        self.assertEqual(database.migrations[1]["name"], "task_lifecycle")
        self.assertEqual(len(database.executed), first_execute_count + 2)

    async def test_rejects_changed_applied_migration(self) -> None:
        migration = PgsqlTaskMigration(
            version=1,
            name="changed",
            statements=("SELECT 1;",),
        )
        database = FakeDatabase()
        database.migrations[1] = {
            "version": 1,
            "name": "changed",
            "checksum": "stale",
        }

        with self.assertRaises(TaskStoreConflictError):
            await PgsqlTaskMigrationRunner(
                database,
                migrations=(migration,),
            ).apply()

    async def test_applies_to_real_postgresql_when_configured(self) -> None:
        dsn = environ.get("AVALAN_TASK_TEST_POSTGRESQL_DSN")
        if not dsn:
            self.skipTest("AVALAN_TASK_TEST_POSTGRESQL_DSN is not set")
        psycopg_pool = importorskip("psycopg_pool")
        rows = importorskip("psycopg.rows")
        schema = f"avalan_task_test_{uuid4().hex}"

        async def configure(connection: Any) -> None:
            connection.row_factory = rows.dict_row

        pool = psycopg_pool.AsyncConnectionPool(
            conninfo=dsn,
            configure=configure,
            open=False,
        )
        await pool.open()
        try:
            async with pool.connection() as connection:
                await connection.execute(f'CREATE SCHEMA "{schema}"')
            await PgsqlTaskMigrationRunner(
                SearchPathDatabase(pool, schema)
            ).apply()
            async with pool.connection() as connection:
                await connection.execute(f'SET search_path TO "{schema}"')
                row = await connection.execute(
                    """
                    SELECT COUNT(*) AS table_count
                    FROM information_schema.tables
                    WHERE table_schema = %s
                    AND table_name LIKE 'task_%';
                    """,
                    (schema,),
                )
                count = await row.fetchone()
            self.assertIsNotNone(count)
            self.assertGreaterEqual(count["table_count"], 7)
        finally:
            async with pool.connection() as connection:
                await connection.execute(
                    f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'
                )
            await pool.close()


class SearchPathDatabase:
    def __init__(self, pool: Any, schema: str) -> None:
        self.pool = pool
        self.schema = schema

    def connection(self) -> "SearchPathConnectionContext":
        return SearchPathConnectionContext(self.pool, self.schema)


class SearchPathConnectionContext:
    def __init__(self, pool: Any, schema: str) -> None:
        self.pool = pool
        self.schema = schema
        self._context: Any | None = None
        self._connection: Any | None = None

    async def __aenter__(self) -> Any:
        self._context = self.pool.connection()
        self._connection = await self._context.__aenter__()
        await self._connection.execute(f'SET search_path TO "{self.schema}"')
        return self._connection

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool:
        assert self._context is not None
        return await self._context.__aexit__(exc_type, exc, traceback)


if __name__ == "__main__":
    main()
