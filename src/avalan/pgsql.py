"""Provide shared async PostgreSQL protocol helpers."""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from inspect import isawaitable
from re import fullmatch
from typing import Any, AsyncContextManager, Protocol, cast
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

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

    async def aclose(self) -> None: ...


class PgsqlDatabaseClosedError(RuntimeError):
    pass


PgsqlConnectionConfigurer = Callable[[PgsqlConnection], Awaitable[None]]


@dataclass(frozen=True, slots=True, kw_only=True)
class PsycopgPoolSettings:
    dsn: str | None = None
    pool: PgsqlDatabase | None = None
    pool_minimum: int = 1
    pool_maximum: int = 10
    pool_timeout_seconds: float | None = None
    connect_timeout_seconds: int | None = None
    statement_timeout_milliseconds: int | None = None
    lock_timeout_milliseconds: int | None = None
    idle_in_transaction_session_timeout_milliseconds: int | None = None
    application_name: str | None = None
    connection_parameters: Mapping[str, str] | None = None
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
        _assert_optional_positive_number(
            self.pool_timeout_seconds,
            "pool_timeout_seconds",
        )
        _assert_optional_positive_int(
            self.connect_timeout_seconds,
            "connect_timeout_seconds",
        )
        _assert_optional_positive_int(
            self.statement_timeout_milliseconds,
            "statement_timeout_milliseconds",
        )
        _assert_optional_positive_int(
            self.lock_timeout_milliseconds,
            "lock_timeout_milliseconds",
        )
        _assert_optional_positive_int(
            self.idle_in_transaction_session_timeout_milliseconds,
            "idle_in_transaction_session_timeout_milliseconds",
        )
        if self.application_name is not None:
            _assert_non_empty_string(self.application_name, "application_name")
        if self.connection_parameters is not None:
            assert isinstance(self.connection_parameters, Mapping)
            for key, value in self.connection_parameters.items():
                _assert_non_empty_string(key, "connection parameter name")
                assert isinstance(value, str)
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
        self._owned_pool = settings.pool is None
        self._opened = False
        self._closed = False

    @property
    def pool(self) -> PgsqlDatabase:
        return self._require_pool()

    def connection(self) -> AsyncContextManager[PgsqlConnection]:
        self._raise_if_closed()
        return self._require_pool().connection()

    async def open(self) -> None:
        self._raise_if_closed()
        if self._opened:
            return
        pool = self._require_pool()
        if self._owned_pool:
            await _maybe_await(pool.open())
        self._opened = True

    async def aclose(self) -> None:
        if self._closed:
            return
        pool = self._pool
        self._closed = True
        if pool is not None and self._owned_pool:
            close = getattr(pool, "close", None)
            if close is not None:
                await _maybe_await(close())

    async def __aenter__(self) -> "PsycopgAsyncDatabase":
        await self.open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool | None:
        await self.aclose()
        return None

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise PgsqlDatabaseClosedError("PostgreSQL database is closed")

    def _require_pool(self) -> PgsqlDatabase:
        self._raise_if_closed()
        if self._pool is None:
            self._pool = self._create_pool()
        return self._pool

    def _create_pool(self) -> PgsqlDatabase:
        pool_module = cast(
            Any,
            self._settings.module_importer("psycopg_pool"),
        )
        kwargs: dict[str, object] = {
            "conninfo": normalize_pgsql_dsn(
                cast(str, self._settings.dsn),
                connect_timeout_seconds=(
                    self._settings.connect_timeout_seconds
                ),
                application_name=self._settings.application_name,
                connection_parameters=self._settings.connection_parameters,
            ),
            "min_size": self._settings.pool_minimum,
            "max_size": self._settings.pool_maximum,
            "configure": self._configure_connection,
            "open": self._settings.open,
        }
        if self._settings.pool_timeout_seconds is not None:
            kwargs["timeout"] = self._settings.pool_timeout_seconds
        pool = pool_module.AsyncConnectionPool(**kwargs)
        if self._settings.open:
            self._opened = True
        return cast(PgsqlDatabase, pool)

    async def _execute_connection_settings(
        self,
        connection: PgsqlConnection,
    ) -> None:
        statements = _connection_setting_statements(self._settings)
        if not statements:
            return
        async with connection.cursor() as cursor:
            for statement, parameters in statements:
                await cursor.execute(statement, parameters)

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
        await self._execute_connection_settings(connection)
        if self._settings.configure_connection is not None:
            await self._settings.configure_connection(connection)

    async def close(self) -> None:
        await self.aclose()


def normalize_pgsql_dsn(
    dsn: str,
    *,
    connect_timeout_seconds: int | None = None,
    application_name: str | None = None,
    connection_parameters: Mapping[str, str] | None = None,
) -> str:
    _assert_non_empty_string(dsn, "dsn")
    normalized = dsn if "://" in dsn else f"postgresql://{dsn}"
    parameters: dict[str, str] = dict(connection_parameters or {})
    if connect_timeout_seconds is not None:
        _assert_optional_positive_int(
            connect_timeout_seconds,
            "connect_timeout_seconds",
        )
        parameters["connect_timeout"] = str(connect_timeout_seconds)
    if application_name is not None:
        _assert_non_empty_string(application_name, "application_name")
        parameters["application_name"] = application_name
    if not parameters:
        return normalized
    parts = urlsplit(normalized)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.update(parameters)
    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            urlencode(query),
            parts.fragment,
        )
    )


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


def _connection_setting_statements(
    settings: PsycopgPoolSettings,
) -> tuple[tuple[str, PgsqlParameters], ...]:
    statements: list[tuple[str, PgsqlParameters]] = []
    if settings.statement_timeout_milliseconds is not None:
        statements.append(
            (
                "SET statement_timeout TO %s",
                (settings.statement_timeout_milliseconds,),
            )
        )
    if settings.lock_timeout_milliseconds is not None:
        statements.append(
            (
                "SET lock_timeout TO %s",
                (settings.lock_timeout_milliseconds,),
            )
        )
    if settings.idle_in_transaction_session_timeout_milliseconds is not None:
        statements.append(
            (
                "SET idle_in_transaction_session_timeout TO %s",
                (settings.idle_in_transaction_session_timeout_milliseconds,),
            )
        )
    if settings.schema is not None:
        statements.append(
            (
                (
                    "SET search_path TO"
                    f" {quote_pgsql_identifier(settings.schema)}"
                ),
                None,
            )
        )
    return tuple(statements)


def _assert_optional_positive_int(
    value: object | None,
    field_name: str,
) -> None:
    if value is None:
        return
    assert isinstance(value, int), f"{field_name} must be an integer"
    assert not isinstance(value, bool), f"{field_name} must be an integer"
    assert value > 0, f"{field_name} must be positive"


def _assert_optional_positive_number(
    value: object | None,
    field_name: str,
) -> None:
    if value is None:
        return
    assert isinstance(value, int | float), f"{field_name} must be numeric"
    assert not isinstance(value, bool), f"{field_name} must be numeric"
    assert value > 0, f"{field_name} must be positive"


def _assert_non_empty_string(value: object, field_name: str) -> None:
    assert isinstance(value, str), f"{field_name} must be a string"
    assert value.strip(), f"{field_name} must not be empty"
