from . import Tool, ToolSet
from ..compat import override
from ..entities import ToolCallContext
from abc import ABC
from asyncio import sleep
from contextlib import AsyncExitStack
from dataclasses import dataclass
from sqlalchemy import inspect, func, select, MetaData
from sqlalchemy import Table as SATable
from sqlalchemy.engine import Connection
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.exc import NoSuchTableError


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


class DatabaseTool(Tool, ABC):
    def __init__(
        self, engine: AsyncEngine, settings: DatabaseToolSettings
    ) -> None:
        self._engine = engine
        self._settings = settings
        super().__init__()

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
        self, engine: AsyncEngine, settings: DatabaseToolSettings
    ) -> None:
        super().__init__(engine, settings)
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
            tbl = SATable(tbl_name, MetaData(), schema=schema)
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
        self, engine: AsyncEngine, settings: DatabaseToolSettings
    ) -> None:
        super().__init__(engine, settings)
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
                DatabaseInspectTool._collect,
                schema=schema,
                table_names=table_names,
            )
            return result

    @staticmethod
    def _collect(
        connection: Connection,
        *,
        schema: str | None,
        table_names: list[str],
    ) -> list[Table]:
        inspector = inspect(connection)
        default_schema, _ = DatabaseTool._schemas(connection, inspector)
        sch = schema or default_schema

        tables: list[Table] = []
        for table_name in table_names:
            # If the table doesn't exist, skip it (instead of failing all).
            try:
                column_info = inspector.get_columns(table_name, schema=sch)
            except NoSuchTableError:
                continue

            columns = {c["name"]: str(c["type"]) for c in column_info}

            fkeys: list[ForeignKey] = []
            try:
                fks = inspector.get_foreign_keys(table_name, schema=sch)
            except NoSuchTableError:
                fks = []

            for fk in fks or []:
                ref_schema = fk.get("referred_schema")
                ref_table = (
                    f"{ref_schema}.{fk['referred_table']}"
                    if ref_schema
                    else fk["referred_table"]
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

            name = (
                table_name
                if sch in (None, default_schema)
                else f"{sch}.{table_name}"
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
        self, engine: AsyncEngine, settings: DatabaseToolSettings
    ) -> None:
        super().__init__(engine, settings)
        self.__name__ = "run"

    async def __call__(
        self, sql: str, *, context: ToolCallContext
    ) -> list[dict]:
        async with self._engine.begin() as conn:
            if self._settings.delay_secs:
                await sleep(self._settings.delay_secs)

            result = await conn.exec_driver_sql(sql)

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
        self, engine: AsyncEngine, settings: DatabaseToolSettings
    ) -> None:
        super().__init__(engine, settings)
        self.__name__ = "tables"

    async def __call__(
        self, *, context: ToolCallContext
    ) -> dict[str | None, list[str]]:
        async with self._engine.connect() as conn:
            if self._settings.delay_secs:
                await sleep(self._settings.delay_secs)
            return await conn.run_sync(DatabaseTablesTool._collect)

    @staticmethod
    def _collect(connection: Connection) -> dict[str | None, list[str]]:
        inspector = inspect(connection)
        _, schemas = DatabaseTool._schemas(connection, inspector)
        return {
            schema: inspector.get_table_names(schema=schema)
            for schema in schemas
        }


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

        tools = [
            DatabaseCountTool(self._engine, settings),
            DatabaseInspectTool(self._engine, settings),
            DatabaseRunTool(self._engine, settings),
            DatabaseTablesTool(self._engine, settings),
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
