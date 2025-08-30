from . import Tool, ToolSet
from ..compat import override
from abc import ABC
from contextlib import AsyncExitStack
from dataclasses import dataclass
from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection
from sqlalchemy.engine.reflection import Inspector
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


class DatabaseTool(ABC):
    @staticmethod
    def _schemas(
        connection: Connection, inspector: Inspector
    ) -> tuple[str | None, list[str | None]]:
        default_schema = inspector.default_schema_name
        if connection.dialect.name == "postgresql":
            sys = {"information_schema", "pg_catalog"}
            schemas = [
                s
                for s in inspector.get_schema_names()
                if s not in sys and not s.startswith("pg_")
            ]
            if default_schema and default_schema not in schemas:
                schemas.append(default_schema)
        else:
            schemas = [default_schema]
        return default_schema, schemas


class DatabaseInspectTool(Tool):
    """
    Gets the schema for a given table using introspection.

    It returns the table column names, types, and foreign keys.

    Args:
        table_name: table to get schema from.
        schema: optional schema the table belongs to, default schema if none.

    Returns:
        The table schema.
    """

    def __init__(self, settings: DatabaseToolSettings) -> None:
        super().__init__()
        self._settings = settings
        self.__name__ = "inspect"

    async def __call__(
        self, table_name: str, schema: str | None = None
    ) -> Table:
        engine: AsyncEngine | None = getattr(self, "_client", None)
        own_engine = engine is None
        if own_engine:
            engine = create_async_engine(
                self._settings.dsn, pool_pre_ping=True
            )
        try:
            async with engine.connect() as conn:
                return await conn.run_sync(
                    DatabaseInspectTool._collect,
                    schema=schema,
                    table_name=table_name,
                )
        finally:
            if own_engine:
                await engine.dispose()

    @staticmethod
    def _collect(
        connection: Connection, *, schema: str | None, table_name: str
    ) -> Table:
        inspector = inspect(connection)
        default_schema, _ = DatabaseTool._schemas(connection, inspector)
        sch = schema or default_schema

        columns = {
            c["name"]: str(c["type"])
            for c in inspector.get_columns(table_name, schema=sch)
        }

        fkeys: list[ForeignKey] = []
        for fk in inspector.get_foreign_keys(table_name, schema=sch):
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
        return Table(name=name, columns=columns, foreign_keys=fkeys)


class DatabaseRunTool(Tool):
    """
    Runs the given SQL statement on the database and gets results.

    Args:
        sql: Valid SQL statement to run.

    Returns:
        The SQL execution results.
    """

    def __init__(self, settings: DatabaseToolSettings) -> None:
        super().__init__()
        self._settings = settings
        self.__name__ = "run"

    async def __call__(self, sql: str) -> list[dict]:
        engine: AsyncEngine | None = getattr(self, "_client", None)
        own_engine = engine is None
        if own_engine:
            engine = create_async_engine(
                self._settings.dsn, pool_pre_ping=True
            )
        try:
            async with engine.begin() as conn:
                result = await conn.execute(text(sql))
                if result.returns_rows:
                    return [dict(row) for row in result.mappings().all()]
                return []
        finally:
            if own_engine:
                await engine.dispose()


class DatabaseTablesTool(Tool):
    """
    Gets the list of table names on the database for all schemas.

    Returns:
        A list of table names indexed by schema.
    """

    def __init__(self, settings: DatabaseToolSettings) -> None:
        super().__init__()
        self._settings = settings
        self.__name__ = "tables"

    async def __call__(self) -> dict[str | None, list[str]]:
        engine: AsyncEngine | None = getattr(self, "_client", None)
        own_engine = engine is None
        if own_engine:
            engine = create_async_engine(
                self._settings.dsn, pool_pre_ping=True
            )
        try:
            async with engine.connect() as conn:
                return await conn.run_sync(DatabaseTablesTool._collect)
        finally:
            if own_engine:
                await engine.dispose()

    @staticmethod
    def _collect(connection: Connection) -> dict[str | None, list[str]]:
        inspector = inspect(connection)
        _, schemas = DatabaseTool._schemas(connection, inspector)
        return {
            schema: inspector.get_table_names(schema=schema)
            for schema in schemas
        }


class DatabaseToolSet(ToolSet):
    @override
    def __init__(
        self,
        settings: DatabaseToolSettings,
        *,
        exit_stack: AsyncExitStack | None = None,
        namespace: str | None = None,
    ):
        self._settings = settings
        self._engine: AsyncEngine | None = None
        tools = [
            DatabaseInspectTool(settings),
            DatabaseRunTool(settings),
            DatabaseTablesTool(settings),
        ]
        super().__init__(
            exit_stack=exit_stack, namespace=namespace, tools=tools
        )

    @override
    async def __aenter__(self) -> "DatabaseToolSet":
        self._engine = create_async_engine(
            self._settings.dsn, pool_pre_ping=True
        )
        for i, tool in enumerate(self._tools):
            self._tools[i] = tool.with_client(self._engine)
        return await super().__aenter__()

    @override
    async def __aexit__(self, exc_type, exc, tb):
        try:
            if self._engine is not None:
                await self._engine.dispose()
        finally:
            return await super().__aexit__(exc_type, exc, tb)
