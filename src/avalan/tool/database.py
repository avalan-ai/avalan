from . import Tool, ToolSet
from ..compat import override
from ..entities import ToolCallContext
from abc import ABC
from asyncio import sleep
from contextlib import AsyncExitStack
from dataclasses import dataclass
from re import compile as regex_compile
from typing import Literal
from sqlalchemy import MetaData, func, inspect, select
from sqlalchemy import Table as SATable
from sqlalchemy.engine import Connection
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


@dataclass(frozen=True, kw_only=True, slots=True)
class ForeignKey:
    field: str
    ref_table: str
    ref_field: str


@dataclass(frozen=True, kw_only=True, slots=True)
class Table:
    name: str
    columns: dict[str, str]
    foreign_keys: list[ForeignKey]


@dataclass(frozen=True, kw_only=True, slots=True)
class DatabaseToolSettings:
    dsn: str
    delay_secs: int | None = None
    identifier_case: Literal["preserve", "lower", "upper"] = "preserve"


class IdentifierCaseNormalizer:
    __slots__ = ("_mode", "_token_pattern")

    def __init__(self, mode: Literal["preserve", "lower", "upper"]) -> None:
        self._mode = mode
        self._token_pattern = regex_compile(r"[0-9A-Za-z_]+(?:\.[0-9A-Za-z_]+)?")

    def normalize(self, identifier: str) -> str:
        if self._mode == "lower":
            return identifier.lower()
        if self._mode == "upper":
            return identifier.upper()
        return identifier

    def normalize_token(self, identifier: str) -> str:
        if "." not in identifier:
            return self.normalize(identifier)
        schema, table = identifier.split(".", 1)
        table_normalized = self.normalize(table)
        return f"{schema}.{table_normalized}"

    def iter_tokens(self, sql: str) -> list[tuple[str, int, int]]:
        return [
            (match.group(0), match.start(), match.end())
            for match in self._token_pattern.finditer(sql)
        ]


class DatabaseTool(Tool, ABC):
    _engine: AsyncEngine
    _settings: DatabaseToolSettings
    _normalizer: IdentifierCaseNormalizer | None
    _table_cache: dict[str | None, dict[str, str]]

    def __init__(
        self,
        engine: AsyncEngine,
        settings: DatabaseToolSettings,
        *,
        normalizer: IdentifierCaseNormalizer | None = None,
        table_cache: dict[str | None, dict[str, str]] | None = None,
    ) -> None:
        self._engine = engine
        self._settings = settings
        if settings.identifier_case == "preserve":
            self._normalizer = None
        else:
            self._normalizer = normalizer or IdentifierCaseNormalizer(
                settings.identifier_case
            )
        self._table_cache = table_cache if table_cache is not None else {}
        super().__init__()

    def _register_table_names(
        self, schema: str | None, table_names: list[str]
    ) -> None:
        if self._normalizer is None:
            return
        cache = self._table_cache.setdefault(schema, {})
        for name in table_names:
            cache[self._normalizer.normalize(name)] = name

    def _denormalize_table_name(
        self,
        connection: Connection,
        schema: str | None,
        table_name: str,
    ) -> str:
        if self._normalizer is None:
            return table_name

        cache = self._table_cache.get(schema)
        normalized = self._normalizer.normalize(table_name)

        if cache is None or normalized not in cache:
            inspector = inspect(connection)
            actual_tables = inspector.get_table_names(schema=schema)
            cache = {
                self._normalizer.normalize(name): name for name in actual_tables
            }
            self._table_cache[schema] = cache

        return cache.get(normalized, table_name)

    def _normalize_table_for_output(self, table_name: str) -> str:
        if self._normalizer is None:
            return table_name
        return self._normalizer.normalize(table_name)

    def _apply_identifier_case(
        self, connection: Connection, sql: str
    ) -> str:
        if self._normalizer is None:
            return sql

        inspector = inspect(connection)
        _, schemas = self._schemas(connection, inspector)

        for schema in schemas:
            if schema in self._table_cache:
                continue
            actual_tables = inspector.get_table_names(schema=schema)
            self._register_table_names(schema, actual_tables)

        replacements: dict[str, str] = {}
        for schema, table_map in self._table_cache.items():
            for normalized, actual in table_map.items():
                replacements[normalized] = actual
                if schema is not None:
                    replacements[
                        f"{schema}.{normalized}"
                    ] = f"{schema}.{actual}"

        if not replacements:
            return sql

        tokens = self._normalizer.iter_tokens(sql)
        if not tokens:
            return sql

        result_parts: list[str] = []
        last_end = 0
        sql_length = len(sql)
        for token, start, end in tokens:
            result_parts.append(sql[last_end:start])
            if start > 0 and sql[start - 1] in {'"', "'", "`"}:
                result_parts.append(token)
                last_end = end
                continue
            if end < sql_length and sql[end:end + 1] in {'"', "'", "`"}:
                result_parts.append(token)
                last_end = end
                continue

            normalized = self._normalizer.normalize_token(token)
            replacement = replacements.get(normalized)
            if replacement is None:
                result_parts.append(token)
            else:
                result_parts.append(replacement)
            last_end = end

        result_parts.append(sql[last_end:])
        return "".join(result_parts)

    @staticmethod
    def _schemas(
        connection: Connection, inspector: Inspector
    ) -> tuple[str | None, list[str | None]]:
        default_schema = inspector.default_schema_name
        dialect = connection.dialect.name

        if dialect == "postgresql":
            sys = {"information_schema", "pg_catalog"}
            schemas = [
                s
                for s in inspector.get_schema_names()
                if s not in sys and not (s or "").startswith("pg_")
            ]
            if default_schema and default_schema not in schemas:
                schemas.append(default_schema)
            return default_schema, schemas

        # Non-PostgreSQL: try to enumerate all schemas, filter known schemas
        all_schemas = inspector.get_schema_names() or (
            [default_schema] if default_schema is not None else [None]
        )

        sys_filters = {
            "mysql": {
                "information_schema",
                "performance_schema",
                "mysql",
                "sys",
            },
            "mariadb": {
                "information_schema",
                "performance_schema",
                "mysql",
                "sys",
            },
            "mssql": {"INFORMATION_SCHEMA", "sys"},
            "oracle": {"SYS", "SYSTEM"},
            "sqlite": set(),  # typically "main"/"temp" only
        }
        sys = sys_filters.get(dialect, set())
        schemas = [s for s in all_schemas if s not in sys]

        if not schemas:
            schemas = (
                [default_schema] if default_schema is not None else [None]
            )

        # De-duplicate while preserving order
        seen: set[str | None] = set()
        uniq: list[str | None] = []
        for s in schemas:
            if s not in seen:
                uniq.append(s)
                seen.add(s)

        # Ensure default schema is included
        if default_schema not in seen:
            uniq.append(default_schema)

        return default_schema, uniq

    @override
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: BaseException | None,
    ) -> bool:
        return await super().__aexit__(exc_type, exc_value, traceback)


class DatabaseCountTool(DatabaseTool):
    """
    Count rows in the given table.

    Args:
        table_name: Table to count rows from (optionally schema-qualified,
                    e.g. "public.users").

    Returns:
        Number of rows in the table.
    """

    def __init__(
        self,
        engine: AsyncEngine,
        settings: DatabaseToolSettings,
        *,
        normalizer: IdentifierCaseNormalizer | None = None,
        table_cache: dict[str | None, dict[str, str]] | None = None,
    ) -> None:
        super().__init__(
            engine,
            settings,
            normalizer=normalizer,
            table_cache=table_cache,
        )
        self.__name__ = "count"

    @staticmethod
    def _split_schema_and_table(qualified: str) -> tuple[str | None, str]:
        # Basic split; handles "schema.table". If no dot, schema is None.
        if "." in qualified:
            sch, tbl = qualified.split(".", 1)
            return (sch or None), tbl
        return None, qualified

    async def __call__(
        self, table_name: str, *, context: ToolCallContext
    ) -> int:
        assert table_name, "table_name must not be empty"
        async with self._engine.connect() as conn:
            if self._settings.delay_secs:
                await sleep(self._settings.delay_secs)

            schema, tbl_name = self._split_schema_and_table(table_name)
            actual_name = await conn.run_sync(
                self._denormalize_table_name, schema, tbl_name
            )
            tbl = SATable(actual_name, MetaData(), schema=schema)
            stmt = select(func.count()).select_from(tbl)

            result = await conn.execute(stmt)
            return int(result.scalar_one())


class DatabaseInspectTool(DatabaseTool):
    """
    Gets the schema for the given tables using introspection.

    It returns the table column names, types, and foreign keys.

    Args:
        table_names: tables to get schemas from.
        schema: optional schema the tables belong to, default schema if none.

    Returns:
        The list of table schemas.
    """

    def __init__(
        self,
        engine: AsyncEngine,
        settings: DatabaseToolSettings,
        *,
        normalizer: IdentifierCaseNormalizer | None = None,
        table_cache: dict[str | None, dict[str, str]] | None = None,
    ) -> None:
        super().__init__(
            engine,
            settings,
            normalizer=normalizer,
            table_cache=table_cache,
        )
        self.__name__ = "inspect"

    async def __call__(
        self,
        table_names: list[str],
        schema: str | None = None,
        *,
        context: ToolCallContext,
    ) -> list[Table]:
        assert table_names, "table_names must not be empty"
        async with self._engine.connect() as conn:
            if self._settings.delay_secs:
                await sleep(self._settings.delay_secs)
            result = await conn.run_sync(
                self._collect,
                schema=schema,
                table_names=table_names,
            )
            return result

    def _collect(
        self,
        connection: Connection,
        *,
        schema: str | None,
        table_names: list[str],
    ) -> list[Table]:
        inspector = inspect(connection)
        default_schema, _ = self._schemas(connection, inspector)
        sch = schema or default_schema

        tables: list[Table] = []
        for table_name in table_names:
            actual_table = self._denormalize_table_name(
                connection, sch, table_name
            )
            # If the table doesn't exist, skip it (instead of failing all).
            try:
                column_info = inspector.get_columns(actual_table, schema=sch)
            except NoSuchTableError:
                continue

            columns = {c["name"]: str(c["type"]) for c in column_info}

            fkeys: list[ForeignKey] = []
            try:
                fks = inspector.get_foreign_keys(actual_table, schema=sch)
            except NoSuchTableError:
                fks = []

            for fk in fks or []:
                ref_schema = fk.get("referred_schema")
                ref_table = (
                    f"{ref_schema}.{self._normalize_table_for_output(fk['referred_table'])}"
                    if ref_schema
                    else self._normalize_table_for_output(fk["referred_table"])
                )
                for source, target in zip(
                    fk.get("constrained_columns", []),
                    fk.get("referred_columns", []),
                ):
                    fkeys.append(
                        ForeignKey(
                            field=source, ref_table=ref_table, ref_field=target
                        )
                    )

            table_display = self._normalize_table_for_output(actual_table)
            name = (
                table_display
                if sch in (None, default_schema)
                else f"{sch}.{table_display}"
            )
            tables.append(
                Table(name=name, columns=columns, foreign_keys=fkeys)
            )

        return tables


class DatabaseRunTool(DatabaseTool):
    """
    Runs the given SQL statement on the database and gets results.

    Args:
        sql: Valid SQL statement to run.

    Returns:
        The SQL execution results.
    """

    def __init__(
        self,
        engine: AsyncEngine,
        settings: DatabaseToolSettings,
        *,
        normalizer: IdentifierCaseNormalizer | None = None,
        table_cache: dict[str | None, dict[str, str]] | None = None,
    ) -> None:
        super().__init__(
            engine,
            settings,
            normalizer=normalizer,
            table_cache=table_cache,
        )
        self.__name__ = "run"

    async def __call__(
        self, sql: str, *, context: ToolCallContext
    ) -> list[dict]:
        async with self._engine.begin() as conn:
            if self._settings.delay_secs:
                await sleep(self._settings.delay_secs)

            sql_to_run = await conn.run_sync(self._apply_identifier_case, sql)
            result = await conn.exec_driver_sql(sql_to_run)

            if result.returns_rows:
                return [dict(row) for row in result.mappings().all()]
            return []


class DatabaseTablesTool(DatabaseTool):
    """
    Gets the list of table names on the database for all schemas.

    Returns:
        A list of table names indexed by schema.
    """

    def __init__(
        self,
        engine: AsyncEngine,
        settings: DatabaseToolSettings,
        *,
        normalizer: IdentifierCaseNormalizer | None = None,
        table_cache: dict[str | None, dict[str, str]] | None = None,
    ) -> None:
        super().__init__(
            engine,
            settings,
            normalizer=normalizer,
            table_cache=table_cache,
        )
        self.__name__ = "tables"

    async def __call__(
        self, *, context: ToolCallContext
    ) -> dict[str | None, list[str]]:
        async with self._engine.connect() as conn:
            if self._settings.delay_secs:
                await sleep(self._settings.delay_secs)
            return await conn.run_sync(self._collect)

    def _collect(self, connection: Connection) -> dict[str | None, list[str]]:
        inspector = inspect(connection)
        _, schemas = self._schemas(connection, inspector)
        result: dict[str | None, list[str]] = {}
        for schema in schemas:
            actual_tables = inspector.get_table_names(schema=schema)
            self._register_table_names(schema, actual_tables)
            result[schema] = [
                self._normalize_table_for_output(name) for name in actual_tables
            ]
        return result


class DatabaseToolSet(ToolSet):
    _engine: AsyncEngine
    _settings: DatabaseToolSettings

    @override
    def __init__(
        self,
        settings: DatabaseToolSettings,
        *,
        exit_stack: AsyncExitStack | None = None,
        namespace: str | None = None,
    ):
        self._settings = settings
        self._engine = create_async_engine(
            self._settings.dsn, pool_pre_ping=True
        )

        normalizer = (
            IdentifierCaseNormalizer(settings.identifier_case)
            if settings.identifier_case != "preserve"
            else None
        )
        table_cache: dict[str | None, dict[str, str]] = {}

        tools = [
            DatabaseCountTool(
                self._engine,
                settings,
                normalizer=normalizer,
                table_cache=table_cache,
            ),
            DatabaseInspectTool(
                self._engine,
                settings,
                normalizer=normalizer,
                table_cache=table_cache,
            ),
            DatabaseRunTool(
                self._engine,
                settings,
                normalizer=normalizer,
                table_cache=table_cache,
            ),
            DatabaseTablesTool(
                self._engine,
                settings,
                normalizer=normalizer,
                table_cache=table_cache,
            ),
        ]
        super().__init__(
            exit_stack=exit_stack, namespace=namespace, tools=tools
        )

    @override
    async def __aexit__(self, exc_type, exc, tb):
        try:
            if self._engine is not None:
                await self._engine.dispose()
        finally:
            return await super().__aexit__(exc_type, exc, tb)
