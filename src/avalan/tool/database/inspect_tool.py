from importlib import import_module

from ...entities import ToolCallContext
from . import (
    DatabaseTool,
    DatabaseToolSettings,
    ForeignKey,
    IdentifierCaseNormalizer,
    Table,
)

from sqlalchemy.engine import Connection
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.ext.asyncio import AsyncEngine

_database = import_module(__package__)


class DatabaseInspectTool(DatabaseTool):
    """Inspect tables to retrieve column schemas and foreign keys.

    Args:
        table_names: Tables to inspect.
        schema: Optional schema the tables belong to; defaults to the current schema.

    Returns:
        Schemas describing the requested tables.
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
        if self._settings.delay_secs:
            await _database.sleep(self._settings.delay_secs)

        async with self._engine.connect() as conn:
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
        inspector = _database.inspect(connection)
        default_schema, _ = self._schemas(connection, inspector)
        sch = schema or default_schema

        tables: list[Table] = []
        for table_name in table_names:
            actual_table = self._denormalize_table_name(
                connection, sch, table_name
            )
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

