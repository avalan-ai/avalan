"""Provide shared async PostgreSQL protocol helpers."""

from .types import (
    assert_non_empty_string as _assert_non_empty_string,
)
from .types import (
    assert_optional_positive_int as _assert_optional_positive_int,
)
from .types import (
    assert_optional_positive_number as _assert_optional_positive_number,
)

from asyncio import CancelledError
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
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


class PgsqlFailureCategory(StrEnum):
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    SERIALIZATION = "serialization"
    DEADLOCK = "deadlock"
    UNIQUE_CONFLICT = "unique_conflict"
    INSUFFICIENT_PRIVILEGE = "insufficient_privilege"
    MIGRATION = "migration"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True, kw_only=True)
class PgsqlFailure:
    category: PgsqlFailureCategory
    code: str
    retryable: bool
    operation: str | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.category, PgsqlFailureCategory)
        _assert_non_empty_string(self.code, "code")
        assert isinstance(self.retryable, bool)
        if self.operation is not None:
            _assert_non_empty_string(self.operation, "operation")


class PgsqlOperationError(RuntimeError):
    def __init__(self, failure: PgsqlFailure) -> None:
        assert isinstance(failure, PgsqlFailure)
        self.failure = failure
        super().__init__(
            "PostgreSQL operation failed: "
            f"category={failure.category.value}, "
            f"code={failure.code}, retryable={failure.retryable}"
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class PgsqlUnitOfWork:
    connection: PgsqlConnection
    cursor: PgsqlCursor

    def __post_init__(self) -> None:
        assert hasattr(self.connection, "cursor")
        assert hasattr(self.cursor, "execute")


PgsqlConnectionConfigurer = Callable[[PgsqlConnection], Awaitable[None]]
PgsqlUnitOfWorkCallback = Callable[[PgsqlUnitOfWork], Awaitable[object]]


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


async def run_pgsql_transaction(
    database: PgsqlDatabase,
    *,
    operation: str,
    callback: PgsqlUnitOfWorkCallback,
) -> object:
    assert hasattr(database, "connection")
    _assert_non_empty_string(operation, "operation")
    assert callable(callback)
    try:
        async with database.connection() as connection:
            async with connection.transaction():
                async with connection.cursor() as cursor:
                    return await callback(
                        PgsqlUnitOfWork(
                            connection=connection,
                            cursor=cursor,
                        )
                    )
    except (KeyboardInterrupt, SystemExit, CancelledError):
        raise
    except BaseException as error:
        failure = classify_pgsql_error(error, operation=operation)
        raise PgsqlOperationError(failure) from None


def classify_pgsql_error(
    error: BaseException,
    *,
    operation: str | None = None,
) -> PgsqlFailure:
    assert isinstance(error, BaseException)
    if operation is not None:
        _assert_non_empty_string(operation, "operation")
    sqlstate = _error_sqlstate(error)
    name = error.__class__.__name__
    module = error.__class__.__module__
    lower_name = name.lower()
    if sqlstate in {"57014", "55P03", "53300"} or "timeout" in lower_name:
        return PgsqlFailure(
            category=PgsqlFailureCategory.TIMEOUT,
            code=sqlstate or "pgsql.timeout",
            retryable=True,
            operation=operation,
        )
    if sqlstate is not None and sqlstate.startswith("08"):
        return PgsqlFailure(
            category=PgsqlFailureCategory.CONNECTION,
            code=sqlstate,
            retryable=True,
            operation=operation,
        )
    if "operationalerror" == lower_name or "pooltimeout" in lower_name:
        return PgsqlFailure(
            category=PgsqlFailureCategory.CONNECTION,
            code=sqlstate or "pgsql.connection",
            retryable=True,
            operation=operation,
        )
    if sqlstate == "40001":
        return PgsqlFailure(
            category=PgsqlFailureCategory.SERIALIZATION,
            code=sqlstate,
            retryable=True,
            operation=operation,
        )
    if sqlstate == "40P01":
        return PgsqlFailure(
            category=PgsqlFailureCategory.DEADLOCK,
            code=sqlstate,
            retryable=True,
            operation=operation,
        )
    if sqlstate == "23505":
        return PgsqlFailure(
            category=PgsqlFailureCategory.UNIQUE_CONFLICT,
            code=sqlstate,
            retryable=False,
            operation=operation,
        )
    if sqlstate == "42501":
        return PgsqlFailure(
            category=PgsqlFailureCategory.INSUFFICIENT_PRIVILEGE,
            code=sqlstate,
            retryable=False,
            operation=operation,
        )
    if (
        module.startswith("alembic")
        or "migration" in lower_name
        or "revision" in lower_name
        or lower_name == "commanderror"
    ):
        return PgsqlFailure(
            category=PgsqlFailureCategory.MIGRATION,
            code=sqlstate or "pgsql.migration",
            retryable=False,
            operation=operation,
        )
    return PgsqlFailure(
        category=PgsqlFailureCategory.UNKNOWN,
        code=sqlstate or "pgsql.unknown",
        retryable=False,
        operation=operation,
    )


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


def _error_sqlstate(error: BaseException) -> str | None:
    value = getattr(error, "sqlstate", None)
    if value is None:
        value = getattr(error, "pgcode", None)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    return value
