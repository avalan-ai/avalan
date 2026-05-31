"""Provide shared async PostgreSQL protocol helpers."""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from inspect import isawaitable
from re import fullmatch
from typing import Any, AsyncContextManager, Protocol, cast

PgsqlParameters = tuple[object, ...] | Mapping[str, object] | None
PgsqlRow = Mapping[str, object]
ModuleImporter = Callable[[str], object]


class PgsqlTransaction(Protocol):
    async def __aenter__(self) -> object: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool | None: ...


class PgsqlCursor(Protocol):
    async def execute(
        self,
        query: str,
        parameters: PgsqlParameters = None,
    ) -> object | None: ...

    async def executemany(
        self,
        query: str,
        parameters_seq: Sequence[PgsqlParameters],
    ) -> object | None: ...

    async def fetchone(self) -> PgsqlRow | None: ...

    async def fetchall(self) -> Sequence[PgsqlRow]: ...

    async def close(self) -> None: ...


class PgsqlConnection(Protocol):
    row_factory: object

    def cursor(self) -> AsyncContextManager[PgsqlCursor]: ...

    def transaction(self) -> AsyncContextManager[PgsqlTransaction]: ...

    async def set_autocommit(self, value: bool) -> None: ...


class PgsqlDatabase(Protocol):
    def connection(self) -> AsyncContextManager[PgsqlConnection]: ...

    async def open(self) -> None: ...


PgsqlConnectionConfigurer = Callable[[PgsqlConnection], Awaitable[None]]


@dataclass(frozen=True, slots=True, kw_only=True)
class PsycopgPoolSettings:
    dsn: str | None = None
    pool: PgsqlDatabase | None = None
    pool_minimum: int = 1
    pool_maximum: int = 10
    schema: str | None = None
    autocommit: bool | None = True
    open: bool = False
    module_importer: ModuleImporter = import_module
    configure_connection: PgsqlConnectionConfigurer | None = None

    def __post_init__(self) -> None:
        assert (self.dsn is None) != (
            self.pool is None
        ), "dsn or pool is required"
        if self.dsn is not None:
            _assert_non_empty_string(self.dsn, "dsn")
        if self.pool is not None:
            assert hasattr(self.pool, "connection")
        assert isinstance(self.pool_minimum, int)
        assert not isinstance(self.pool_minimum, bool)
        assert self.pool_minimum > 0
        assert isinstance(self.pool_maximum, int)
        assert not isinstance(self.pool_maximum, bool)
        assert self.pool_maximum >= self.pool_minimum
        if self.schema is not None:
            assert_pgsql_identifier(self.schema, "schema")
        if self.autocommit is not None:
            assert isinstance(self.autocommit, bool)
        assert isinstance(self.open, bool)
        assert callable(self.module_importer)
        if self.configure_connection is not None:
            assert callable(self.configure_connection)


class PsycopgAsyncDatabase:
    def __init__(self, settings: PsycopgPoolSettings) -> None:
        assert isinstance(settings, PsycopgPoolSettings)
        self._settings = settings
        self._pool = settings.pool

    @property
    def pool(self) -> PgsqlDatabase:
        return self._require_pool()

    def connection(self) -> AsyncContextManager[PgsqlConnection]:
        return self._require_pool().connection()

    async def open(self) -> None:
        pool = self._require_pool()
        await _maybe_await(pool.open())

    async def aclose(self) -> None:
        pool = self._pool
        if pool is None:
            return
        close = getattr(pool, "close", None)
        if close is not None:
            await _maybe_await(close())

    def _require_pool(self) -> PgsqlDatabase:
        if self._pool is None:
            self._pool = self._create_pool()
        return self._pool

    def _create_pool(self) -> PgsqlDatabase:
        pool_module = cast(
            Any,
            self._settings.module_importer("psycopg_pool"),
        )
        pool = pool_module.AsyncConnectionPool(
            conninfo=normalize_pgsql_dsn(cast(str, self._settings.dsn)),
            min_size=self._settings.pool_minimum,
            max_size=self._settings.pool_maximum,
            configure=self._configure_connection,
            open=self._settings.open,
        )
        return cast(PgsqlDatabase, pool)

    async def _configure_connection(
        self,
        connection: PgsqlConnection,
    ) -> None:
        rows_module = cast(
            Any,
            self._settings.module_importer("psycopg.rows"),
        )
        connection.row_factory = rows_module.dict_row
        if self._settings.autocommit is not None:
            await connection.set_autocommit(self._settings.autocommit)
        if self._settings.schema is not None:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "SET search_path TO "
                    f"{quote_pgsql_identifier(self._settings.schema)}"
                )
        if self._settings.configure_connection is not None:
            await self._settings.configure_connection(connection)


def normalize_pgsql_dsn(dsn: str) -> str:
    _assert_non_empty_string(dsn, "dsn")
    if "://" in dsn:
        return dsn
    return f"postgresql://{dsn}"


def assert_pgsql_identifier(value: str, field_name: str) -> None:
    _assert_non_empty_string(value, field_name)
    assert fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]{0,62}",
        value,
    ), f"{field_name} must be a PostgreSQL identifier"


def quote_pgsql_identifier(value: str) -> str:
    assert_pgsql_identifier(value, "identifier")
    return f'"{value}"'


async def _maybe_await(value: object) -> None:
    if isawaitable(value):
        await value


def _assert_non_empty_string(value: object, field_name: str) -> None:
    assert isinstance(value, str), f"{field_name} must be a string"
    assert value.strip(), f"{field_name} must not be empty"
