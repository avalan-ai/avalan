from importlib import import_module

from ...entities import ToolCallContext
from . import (
    DatabaseTool,
    DatabaseToolSettings,
    IdentifierCaseNormalizer,
)

from sqlalchemy import MetaData, func, select
from sqlalchemy import Table as SATable
from sqlalchemy.ext.asyncio import AsyncEngine

_database = import_module(__package__)


class DatabaseCountTool(DatabaseTool):
    """Count rows in the given table.

    Args:
        table_name: Table to count rows from (optionally schema-qualified, e.g. 'public.users').

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
        if "." in qualified:
            sch, tbl = qualified.split(".", 1)
            return (sch or None), tbl
        return None, qualified

    async def __call__(
        self, table_name: str, *, context: ToolCallContext
    ) -> int:
        assert table_name, "table_name must not be empty"
        if self._settings.delay_secs:
            await _database.sleep(self._settings.delay_secs)

        async with self._engine.connect() as conn:
            schema, tbl_name = self._split_schema_and_table(table_name)
            actual_name = await conn.run_sync(
                self._denormalize_table_name, schema, tbl_name
            )
            tbl = SATable(actual_name, MetaData(), schema=schema)
            stmt = select(func.count()).select_from(tbl)

            result = await conn.execute(stmt)
            return int(result.scalar_one())

