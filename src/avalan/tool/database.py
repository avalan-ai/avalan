from . import Tool, ToolSet
from ..compat import override
from abc import ABC
from contextlib import AsyncExitStack
from dataclasses import dataclass
from sqlalchemy import inspect, text
from sqlalchemy.engine.base import Connection
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.ext.asyncio import create_async_engine


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
class DatabaseToolSettings(dict):
    dsn: str

    def __post_init__(self):
        self["dsn"] = self.dsn


class DatabaseTool(ABC):
    @staticmethod
    def _schemas(
        connection: Connection, inspector: Inspector
    ) -> tuple[str, list[str]]:
        default_schema = inspector.default_schema_name
        dialect = connection.dialect.name

        if dialect == "postgresql":
            # all non-system schemas in this DB
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

    def __init__(
        self,
        settings: DatabaseToolSettings,
    ) -> None:
        super().__init__()
        self._settings = settings
        self.__name__ = "inspect"

    async def __call__(
        self,
        table_name: str,
        schema: str | None = None
    ) -> Table:
        engine = create_async_engine(self._settings.dsn)

        try:
            async with engine.connect() as conn:
                tables = await conn.run_sync(
                    DatabaseInspectTool._collect,
                    schema=schema,
                    table_name=table_name
                )
        finally:
            await engine.dispose()
        return tables

    @staticmethod
    def _collect(
        connection: Connection,
        *,
        schema: str | None,
        table_name: str
    ):
        inspector = inspect(connection)
        default_schema, schemas = DatabaseTool._schemas(connection, inspector)
        if not schema:
            schema = default_schema

        columns = {
            c["name"]: str(c["type"])
            for c in inspector.get_columns(table_name, schema=schema)
        }

        fkeys: list[ForeignKey] = []
        for fk in inspector.get_foreign_keys(table_name, schema=schema):
            schema = fk.get("referred_schema")
            table = fk["referred_table"]
            ref_table = (
                f"{schema}.{table}"
                if schema
                else None
            )
            for source, target in zip(
                fk.get("constrained_columns", []),
                fk.get("referred_columns", []),
            ):
                fkeys.append(
                    ForeignKey(
                        field=source,
                        ref_table=ref_table,
                        ref_field=target,
                    )
                )

        name = (
            table_name
            if schema in (None, default_schema)
            else f"{schema}.{table_name}"
        )
        table = Table(name=name, columns=columns, foreign_keys=fkeys)
        return table


class DatabaseRunTool(Tool):
    """
    Runs the given SQL statement on the database and gets results.

    Args:
        sql: Valid SQL statement to run.

    Returns:
        The SQL execution results.
    """

    def __init__(
        self,
        settings: DatabaseToolSettings,
    ) -> None:
        super().__init__()
        self._settings = settings
        self.__name__ = "run"

    async def __call__(self, sql: str) -> list[dict]:
        engine = create_async_engine(self._settings.dsn)

        try:
            async with engine.connect() as conn:
                result = await conn.execute(text(sql))
                rows = result.mappings().all()
                return [dict(row) for row in rows]
        finally:
            await engine.dispose()


class DatabaseTablesTool(Tool):
    """
    Gets the list of table names on the database for all schemas.

    Returns:
        A list of table names indexed by schema.
    """

    def __init__(
        self,
        settings: DatabaseToolSettings,
    ) -> None:
        super().__init__()
        self._settings = settings
        self.__name__ = "tables"

    async def __call__(self) -> dict[str, list[str]]:
        engine = create_async_engine(self._settings.dsn)

        try:
            async with engine.connect() as conn:
                tables = await conn.run_sync(DatabaseTablesTool._collect)
        finally:
            await engine.dispose()
        return tables

    @staticmethod
    def _collect(connection: Connection) -> dict[str, list[str]]:
        inspector = inspect(connection)
        _, schemas = DatabaseTool._schemas(connection, inspector)

        tables: dict[str, list[str]] = {}
        for schema in schemas:
            if schema not in tables:
                tables[schema] = []
            tables[schema].extend([
                table_name
                for table_name in inspector.get_table_names(schema=schema)
            ])
        return tables


class DatabaseToolSet(ToolSet):
    @override
    def __init__(
        self,
        settings: DatabaseToolSettings,
        *,
        exit_stack: AsyncExitStack | None = None,
        namespace: str | None = None,
    ):
        assert settings
        tools = [
            DatabaseInspectTool(settings),
            DatabaseRunTool(settings),
            DatabaseTablesTool(settings)
        ]
        return super().__init__(
            exit_stack=exit_stack, namespace=namespace, tools=tools
        )

    @override
    async def __aenter__(self) -> "DatabaseToolSet":
        self._client = await self._exit_stack.enter_async_context(self._client)
        for i, tool in enumerate(self._tools):
            self._tools[i] = tool.with_client(self._client)
        return await super().__aenter__()
