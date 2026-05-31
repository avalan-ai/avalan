from types import SimpleNamespace
from typing import cast
from unittest import IsolatedAsyncioTestCase, TestCase, main

from avalan.pgsql import (
    PsycopgAsyncDatabase,
    PsycopgPoolSettings,
    normalize_pgsql_dsn,
    quote_pgsql_identifier,
)


class FakePool:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.opened = False
        self.closed = False
        self.connections = 0

    def connection(self) -> "FakeConnectionContext":
        self.connections += 1
        return FakeConnectionContext()

    async def open(self) -> None:
        self.opened = True

    async def close(self) -> None:
        self.closed = True


class FakeConnectionContext:
    async def __aenter__(self) -> str:
        return "connection"

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool:
        return False


class FakeConfigureCursorContext:
    def __init__(self, cursor: "FakeConfigureCursor") -> None:
        self.cursor = cursor

    async def __aenter__(self) -> "FakeConfigureCursor":
        return self.cursor

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool:
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
        self.row_factory: object | None = None
        self.autocommit_values: list[bool] = []

    def cursor(self) -> FakeConfigureCursorContext:
        return FakeConfigureCursorContext(self.cursor_value)

    async def set_autocommit(self, value: bool) -> None:
        self.autocommit_values.append(value)


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

    def test_quotes_only_safe_identifiers(self) -> None:
        self.assertEqual(quote_pgsql_identifier("tenant_1"), '"tenant_1"')
        with self.assertRaises(AssertionError):
            quote_pgsql_identifier("tenant;drop")
        with self.assertRaises(AssertionError):
            quote_pgsql_identifier("1tenant")


class PsycopgAsyncDatabaseTest(IsolatedAsyncioTestCase):
    async def test_injected_pool_does_not_import_psycopg(self) -> None:
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
        await database.aclose()

        self.assertEqual(pool.connections, 1)
        self.assertTrue(pool.opened)
        self.assertTrue(pool.closed)

    async def test_close_before_pool_creation_is_noop(self) -> None:
        database = PsycopgAsyncDatabase(
            PsycopgPoolSettings(
                dsn="postgresql://host/database",
                module_importer=self._unexpected_import,
            )
        )

        await database.aclose()

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
            "postgresql://user@host/database",
        )
        self.assertEqual(pool.kwargs["min_size"], 2)
        self.assertEqual(pool.kwargs["max_size"], 5)
        self.assertFalse(pool.kwargs["open"])
        self.assertIs(connection.row_factory, dict_row)
        self.assertEqual(connection.autocommit_values, [True])
        self.assertEqual(
            connection.cursor_value.executed,
            [('SET search_path TO "tenant_1"', None)],
        )
        self.assertEqual(configured, [connection])
        self.assertIs(created["pool"], pool)

    @staticmethod
    def _unexpected_import(module: str) -> object:
        raise AssertionError(f"unexpected import: {module}")


if __name__ == "__main__":
    main()
