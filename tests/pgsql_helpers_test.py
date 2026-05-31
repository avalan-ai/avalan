from asyncio import CancelledError
from types import SimpleNamespace
from typing import cast
from unittest import IsolatedAsyncioTestCase, TestCase, main
from urllib.parse import parse_qsl, urlsplit

from avalan.pgsql import (
    PgsqlDatabaseClosedError,
    PgsqlFailureCategory,
    PgsqlOperationError,
    PgsqlUnitOfWork,
    PsycopgAsyncDatabase,
    PsycopgPoolSettings,
    classify_pgsql_error,
    normalize_pgsql_dsn,
    quote_pgsql_identifier,
    run_pgsql_transaction,
)


class FakePool:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.opened = False
        self.closed = False
        self.open_count = 0
        self.close_count = 0
        self.connections = 0

    def connection(self) -> "FakeConnectionContext":
        self.connections += 1
        return FakeConnectionContext()

    async def open(self) -> None:
        self.opened = True
        self.open_count += 1

    async def close(self) -> None:
        self.closed = True
        self.close_count += 1


class RecordingPool(FakePool):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.context: FakeConnectionContext | None = None

    def connection(self) -> "FakeConnectionContext":
        self.connections += 1
        self.context = FakeConnectionContext()
        return self.context


class FakeConnectionContext:
    def __init__(self) -> None:
        self.exit_type: type[BaseException] | None = None

    async def __aenter__(self) -> str:
        return "connection"

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool:
        self.exit_type = exc_type
        return False


class FakeConfigureCursorContext:
    def __init__(self, cursor: "FakeConfigureCursor") -> None:
        self.cursor = cursor
        self.exit_type: type[BaseException] | None = None

    async def __aenter__(self) -> "FakeConfigureCursor":
        return self.cursor

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool:
        self.exit_type = exc_type
        return False


class FakeConfigureCursor:
    def __init__(self) -> None:
        self.executed: list[tuple[str, object]] = []

    async def execute(
        self,
        query: str,
        parameters: object | None = None,
    ) -> None:
        self.executed.append((query, parameters))


class FakeConfigureConnection:
    def __init__(self) -> None:
        self.cursor_value = FakeConfigureCursor()
        self.cursor_context: FakeConfigureCursorContext | None = None
        self.row_factory: object | None = None
        self.autocommit_values: list[bool] = []

    def cursor(self) -> FakeConfigureCursorContext:
        self.cursor_context = FakeConfigureCursorContext(self.cursor_value)
        return self.cursor_context

    async def set_autocommit(self, value: bool) -> None:
        self.autocommit_values.append(value)


class FakeTransactionCursorContext:
    def __init__(self, cursor: "FakeTransactionCursor") -> None:
        self.cursor = cursor
        self.exit_type: type[BaseException] | None = None

    async def __aenter__(self) -> "FakeTransactionCursor":
        return self.cursor

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool:
        self.exit_type = exc_type
        return False


class FakeTransactionCursor:
    def __init__(self) -> None:
        self.executed: list[tuple[str, object | None]] = []

    async def execute(
        self,
        query: str,
        parameters: object | None = None,
    ) -> None:
        self.executed.append((query, parameters))

    async def executemany(
        self,
        query: str,
        parameters_seq: object,
    ) -> None:
        self.executed.append((query, parameters_seq))

    async def fetchone(self) -> dict[str, object] | None:
        return {"ok": True}

    async def fetchall(self) -> tuple[dict[str, object], ...]:
        return ({"ok": True},)

    async def close(self) -> None:
        return None


class FakeTransactionContext:
    def __init__(self) -> None:
        self.exit_type: type[BaseException] | None = None
        self.entered = False

    async def __aenter__(self) -> object:
        self.entered = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool:
        self.exit_type = exc_type
        return False


class FakeTransactionConnection:
    def __init__(self) -> None:
        self.cursor_value = FakeTransactionCursor()
        self.cursor_context: FakeTransactionCursorContext | None = None
        self.transaction_context = FakeTransactionContext()
        self.row_factory: object | None = None

    def cursor(self) -> FakeTransactionCursorContext:
        self.cursor_context = FakeTransactionCursorContext(self.cursor_value)
        return self.cursor_context

    def transaction(self) -> FakeTransactionContext:
        return self.transaction_context

    async def set_autocommit(self, value: bool) -> None:
        return None


class FakeTransactionConnectionContext:
    def __init__(self, connection: FakeTransactionConnection) -> None:
        self.connection = connection
        self.exit_type: type[BaseException] | None = None

    async def __aenter__(self) -> FakeTransactionConnection:
        return self.connection

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool:
        self.exit_type = exc_type
        return False


class FakeTransactionDatabase:
    def __init__(self) -> None:
        self.connection_value = FakeTransactionConnection()
        self.connection_context: FakeTransactionConnectionContext | None = None

    def connection(self) -> FakeTransactionConnectionContext:
        self.connection_context = FakeTransactionConnectionContext(
            self.connection_value
        )
        return self.connection_context


class CancelledConfigureCursor(FakeConfigureCursor):
    async def execute(
        self,
        query: str,
        parameters: object | None = None,
    ) -> None:
        await super().execute(query, parameters)
        raise CancelledError()


class CancelledConfigureConnection(FakeConfigureConnection):
    def __init__(self) -> None:
        super().__init__()
        self.cursor_value = CancelledConfigureCursor()


class PgsqlHelpersTest(TestCase):
    def test_normalizes_dsn_without_leaking_driver_dependency(self) -> None:
        self.assertEqual(
            normalize_pgsql_dsn("user@host/database"),
            "postgresql://user@host/database",
        )
        self.assertEqual(
            normalize_pgsql_dsn("postgresql+psycopg://host/database"),
            "postgresql+psycopg://host/database",
        )
        self.assertEqual(
            normalize_pgsql_dsn(
                "postgresql://user@host/database?sslmode=require",
                connect_timeout_seconds=3,
                application_name="avalan-task",
                connection_parameters={"target_session_attrs": "read-write"},
            ),
            "postgresql://user@host/database?"
            "sslmode=require&target_session_attrs=read-write&"
            "connect_timeout=3&application_name=avalan-task",
        )

    def test_hosted_dsn_fixtures_preserve_pool_options(self) -> None:
        cases = (
            (
                (
                    "postgresql://user:pass@rds.amazonaws.com/tasks?"
                    "sslmode=require"
                ),
                {"target_session_attrs": "read-write"},
            ),
            (
                (
                    "postgresql://user:pass@server.postgres.database.azure.com/"
                    "tasks?sslmode=require"
                ),
                {"channel_binding": "require"},
            ),
            (
                (
                    "postgresql://user:pass@aws-0-us-east-1.pooler.supabase.com/"
                    "postgres?sslmode=require"
                ),
                {"options": "-c statement_timeout=5000"},
            ),
            (
                (
                    "postgresql://user:pass@ep-example.neon.tech/tasks?"
                    "sslmode=require"
                ),
                {"target_session_attrs": "read-write"},
            ),
        )

        for dsn, parameters in cases:
            with self.subTest(dsn=dsn):
                normalized = normalize_pgsql_dsn(
                    dsn,
                    connect_timeout_seconds=3,
                    application_name="avalan-task",
                    connection_parameters=parameters,
                )
                query = dict(parse_qsl(urlsplit(normalized).query))

                self.assertEqual(query["sslmode"], "require")
                self.assertEqual(query["connect_timeout"], "3")
                self.assertEqual(query["application_name"], "avalan-task")
                for key, value in parameters.items():
                    self.assertEqual(query[key], value)

    def test_quotes_only_safe_identifiers(self) -> None:
        self.assertEqual(quote_pgsql_identifier("tenant_1"), '"tenant_1"')
        with self.assertRaises(AssertionError):
            quote_pgsql_identifier("tenant;drop")
        with self.assertRaises(AssertionError):
            quote_pgsql_identifier("1tenant")

    def test_pool_settings_reject_invalid_options(self) -> None:
        with self.assertRaises(AssertionError):
            PsycopgPoolSettings(
                dsn="postgresql://host/database",
                pool_timeout_seconds=0,
            )
        with self.assertRaises(AssertionError):
            PsycopgPoolSettings(
                dsn="postgresql://host/database",
                connect_timeout_seconds=True,
            )
        with self.assertRaises(AssertionError):
            PsycopgPoolSettings(
                dsn="postgresql://host/database",
                application_name="",
            )
        with self.assertRaises(AssertionError):
            PsycopgPoolSettings(
                dsn="postgresql://host/database",
                connection_parameters=cast(
                    dict[str, str],
                    {"sslmode": 1},
                ),
            )

    def test_classifies_postgresql_failures_without_raw_payloads(self) -> None:
        cases = (
            (_sqlstate_error("40001", "private input"), "serialization", True),
            (_sqlstate_error("40P01", "private lock"), "deadlock", True),
            (
                _sqlstate_error("23505", "private key"),
                "unique_conflict",
                False,
            ),
            (
                _sqlstate_error("42501", "private role"),
                "insufficient_privilege",
                False,
            ),
            (_sqlstate_error("08006", "private host"), "connection", True),
            (
                _named_error("OperationalError", "private host"),
                "connection",
                True,
            ),
            (_named_error("PoolTimeout", "private pool"), "timeout", True),
            (TimeoutError("private timeout"), "timeout", True),
            (_migration_error("private revision"), "migration", False),
            (_invalid_sqlstate_error("private code"), "unknown", False),
        )

        for error, category, retryable in cases:
            with self.subTest(category=category):
                failure = classify_pgsql_error(error, operation="append")

                self.assertEqual(failure.category.value, category)
                self.assertEqual(failure.retryable, retryable)
                self.assertEqual(failure.operation, "append")
                self.assertNotIn("private", str(failure))


class PsycopgAsyncDatabaseTest(IsolatedAsyncioTestCase):
    async def test_injected_pool_lifecycle_is_borrowed(self) -> None:
        pool = FakePool()
        database = PsycopgAsyncDatabase(
            PsycopgPoolSettings(
                pool=pool,
                module_importer=self._unexpected_import,
            )
        )

        async with database.connection() as connection:
            self.assertEqual(connection, "connection")
        await database.open()
        await database.open()
        await database.aclose()
        await database.aclose()

        self.assertEqual(pool.connections, 1)
        self.assertFalse(pool.opened)
        self.assertFalse(pool.closed)
        self.assertEqual(pool.open_count, 0)
        self.assertEqual(pool.close_count, 0)

    async def test_cancelled_connection_body_releases_pool_context(
        self,
    ) -> None:
        pool = RecordingPool()
        database = PsycopgAsyncDatabase(PsycopgPoolSettings(pool=pool))

        with self.assertRaises(CancelledError):
            async with database.connection():
                raise CancelledError()

        assert pool.context is not None
        self.assertIs(pool.context.exit_type, CancelledError)

    async def test_cancelled_connection_configuration_closes_cursor(
        self,
    ) -> None:
        dict_row = object()
        modules = {"psycopg.rows": SimpleNamespace(dict_row=dict_row)}
        database = PsycopgAsyncDatabase(
            PsycopgPoolSettings(
                pool=FakePool(),
                statement_timeout_milliseconds=1000,
                module_importer=modules.__getitem__,
            )
        )
        connection = CancelledConfigureConnection()

        with self.assertRaises(CancelledError):
            await database._configure_connection(connection)

        assert connection.cursor_context is not None
        self.assertIs(connection.cursor_context.exit_type, CancelledError)
        self.assertEqual(connection.autocommit_values, [True])

    async def test_connection_configuration_without_settings_skips_cursor(
        self,
    ) -> None:
        dict_row = object()
        modules = {"psycopg.rows": SimpleNamespace(dict_row=dict_row)}
        database = PsycopgAsyncDatabase(
            PsycopgPoolSettings(
                pool=FakePool(),
                module_importer=modules.__getitem__,
            )
        )
        connection = FakeConfigureConnection()

        await database._configure_connection(connection)
        await database.close()

        self.assertIs(connection.cursor_context, None)
        self.assertIs(connection.row_factory, dict_row)

    async def test_transaction_helper_wraps_failures_safely(self) -> None:
        database = FakeTransactionDatabase()

        async def callback(unit: PgsqlUnitOfWork) -> object:
            await unit.cursor.execute("INSERT", ("secret",))
            raise _sqlstate_error("23505", "duplicate private payload")

        with self.assertRaises(PgsqlOperationError) as caught:
            await run_pgsql_transaction(
                database,
                operation="register_definition",
                callback=callback,
            )

        failure = caught.exception.failure
        self.assertEqual(
            failure.category,
            PgsqlFailureCategory.UNIQUE_CONFLICT,
        )
        self.assertFalse(failure.retryable)
        self.assertNotIn("duplicate private payload", str(caught.exception))
        self.assertEqual(
            database.connection_value.transaction_context.exit_type.__name__,
            "SqlstateError",
        )
        assert database.connection_value.cursor_context is not None
        self.assertEqual(
            database.connection_value.cursor_context.exit_type.__name__,
            "SqlstateError",
        )

    async def test_transaction_helper_returns_successful_callback_value(
        self,
    ) -> None:
        database = FakeTransactionDatabase()

        async def callback(unit: PgsqlUnitOfWork) -> object:
            await unit.cursor.execute("SELECT 1")
            return {"ok": True}

        result = await run_pgsql_transaction(
            database,
            operation="inspect",
            callback=callback,
        )

        self.assertEqual(result, {"ok": True})
        self.assertIs(
            database.connection_value.transaction_context.exit_type,
            None,
        )
        assert database.connection_value.cursor_context is not None
        self.assertIs(database.connection_value.cursor_context.exit_type, None)

    async def test_transaction_helper_preserves_cancellation(self) -> None:
        database = FakeTransactionDatabase()

        async def callback(unit: PgsqlUnitOfWork) -> object:
            raise CancelledError("private cancellation")

        with self.assertRaises(CancelledError):
            await run_pgsql_transaction(
                database,
                operation="claim",
                callback=callback,
            )

        self.assertIs(
            database.connection_value.transaction_context.exit_type,
            CancelledError,
        )

    async def test_close_before_pool_creation_is_noop(self) -> None:
        database = PsycopgAsyncDatabase(
            PsycopgPoolSettings(
                dsn="postgresql://host/database",
                module_importer=self._unexpected_import,
            )
        )

        await database.aclose()
        await database.aclose()

        with self.assertRaises(PgsqlDatabaseClosedError):
            database.connection()

    async def test_owned_pool_opens_and_closes_once(self) -> None:
        pool_instance = FakePool()
        pool_cls = SimpleNamespace(
            AsyncConnectionPool=lambda **kwargs: pool_instance
        )
        database = PsycopgAsyncDatabase(
            PsycopgPoolSettings(
                dsn="postgresql://host/database",
                module_importer={"psycopg_pool": pool_cls}.__getitem__,
            )
        )

        await database.open()
        await database.open()
        await database.aclose()
        await database.aclose()

        self.assertEqual(pool_instance.open_count, 1)
        self.assertEqual(pool_instance.close_count, 1)

    async def test_constructor_open_pool_is_not_reopened(self) -> None:
        pool_instance = FakePool()
        pool_cls = SimpleNamespace(
            AsyncConnectionPool=lambda **kwargs: pool_instance
        )
        database = PsycopgAsyncDatabase(
            PsycopgPoolSettings(
                dsn="postgresql://host/database",
                open=True,
                module_importer={"psycopg_pool": pool_cls}.__getitem__,
            )
        )

        self.assertIs(database.pool, pool_instance)
        await database.open()
        await database.aclose()

        self.assertEqual(pool_instance.open_count, 0)
        self.assertEqual(pool_instance.close_count, 1)

    async def test_async_context_manager_owns_pool_lifecycle(self) -> None:
        pool_instance = FakePool()
        pool_cls = SimpleNamespace(
            AsyncConnectionPool=lambda **kwargs: pool_instance
        )
        database = PsycopgAsyncDatabase(
            PsycopgPoolSettings(
                dsn="postgresql://host/database",
                module_importer={"psycopg_pool": pool_cls}.__getitem__,
            )
        )

        async with database as opened:
            self.assertIs(opened, database)
            self.assertEqual(pool_instance.open_count, 1)

        self.assertEqual(pool_instance.close_count, 1)

    async def test_lazy_pool_configures_dict_rows_and_search_path(
        self,
    ) -> None:
        created: dict[str, FakePool] = {}
        dict_row = object()
        configured: list[FakeConfigureConnection] = []

        def pool_factory(**kwargs: object) -> FakePool:
            pool = FakePool(**kwargs)
            created["pool"] = pool
            return pool

        async def configure(connection: object) -> None:
            configured.append(cast(FakeConfigureConnection, connection))

        modules = {
            "psycopg.rows": SimpleNamespace(dict_row=dict_row),
            "psycopg_pool": SimpleNamespace(AsyncConnectionPool=pool_factory),
        }
        database = PsycopgAsyncDatabase(
            PsycopgPoolSettings(
                dsn="user@host/database",
                pool_minimum=2,
                pool_maximum=5,
                pool_timeout_seconds=4.5,
                connect_timeout_seconds=3,
                statement_timeout_milliseconds=1000,
                lock_timeout_milliseconds=250,
                idle_in_transaction_session_timeout_milliseconds=5000,
                application_name="avalan-task",
                connection_parameters={"sslmode": "require"},
                schema="tenant_1",
                module_importer=modules.__getitem__,
                configure_connection=configure,
            )
        )

        pool = cast(FakePool, database.pool)
        connection = FakeConfigureConnection()
        configure_connection = cast(object, pool.kwargs["configure"])
        assert callable(configure_connection)
        await configure_connection(connection)

        self.assertEqual(
            pool.kwargs["conninfo"],
            "postgresql://user@host/database?sslmode=require&"
            "connect_timeout=3&application_name=avalan-task",
        )
        self.assertEqual(pool.kwargs["min_size"], 2)
        self.assertEqual(pool.kwargs["max_size"], 5)
        self.assertEqual(pool.kwargs["timeout"], 4.5)
        self.assertFalse(pool.kwargs["open"])
        self.assertIs(connection.row_factory, dict_row)
        self.assertEqual(connection.autocommit_values, [True])
        self.assertEqual(
            connection.cursor_value.executed,
            [
                ("SET statement_timeout TO %s", (1000,)),
                ("SET lock_timeout TO %s", (250,)),
                (
                    "SET idle_in_transaction_session_timeout TO %s",
                    (5000,),
                ),
                ('SET search_path TO "tenant_1"', None),
            ],
        )
        self.assertEqual(configured, [connection])
        self.assertIs(created["pool"], pool)

    @staticmethod
    def _unexpected_import(module: str) -> object:
        raise AssertionError(f"unexpected import: {module}")


def _sqlstate_error(sqlstate: str, message: str) -> BaseException:
    class SqlstateError(RuntimeError):
        pass

    error = SqlstateError(message)
    error.sqlstate = sqlstate  # type: ignore[attr-defined]
    return error


def _migration_error(message: str) -> BaseException:
    class CommandError(RuntimeError):
        pass

    CommandError.__module__ = "alembic.util.exc"
    return CommandError(message)


def _named_error(name: str, message: str) -> BaseException:
    error_type = type(name, (RuntimeError,), {})
    return error_type(message)


def _invalid_sqlstate_error(message: str) -> BaseException:
    error = RuntimeError(message)
    error.sqlstate = object()  # type: ignore[attr-defined]
    return error


if __name__ == "__main__":
    main()
