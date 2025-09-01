from . import Tool, ToolSet
from ..compat import override
from ..entities import ToolCallContext
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
        table_name: Table to count rows from.

    Returns:
        Number of rows in the table.
    """

    def __init__(
        self, engine: AsyncEngine, settings: DatabaseToolSettings
    ) -> None:
        super().__init__(engine, settings)
        self.__name__ = "count"

    async def __call__(
        self, table_name: str, *, context: ToolCallContext
    ) -> int:
        assert table_name, "table_name must not be empty"
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(f"SELECT COUNT(*) FROM {table_name}")
            )
            return result.scalar_one()


class DatabaseInspectTool(DatabaseTool):
    """
    Gets the schema for a given table using introspection.

    It returns the table column names, types, and foreign keys.

    Args:
        table_name: table to get schema from.
        schema: optional schema the table belongs to, default schema if none.

    Returns:
        The table schema.
    """

    def __init__(
        self, engine: AsyncEngine, settings: DatabaseToolSettings
    ) -> None:
        super().__init__(engine, settings)
        self.__name__ = "inspect"

    async def __call__(
        self,
        table_name: str,
        schema: str | None = None,
        *,
        context: ToolCallContext,
    ) -> Table:
        async with self._engine.connect() as conn:
            result = await conn.run_sync(
                DatabaseInspectTool._collect,
                schema=schema,
                table_name=table_name,
            )
            return result

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
            result = await conn.execute(text(sql))
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
