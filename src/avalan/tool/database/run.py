from ...entities import ToolCallContext
from . import (
    AsyncEngine,
    DatabaseTool,
    DatabaseToolSettings,
    IdentifierCaseNormalizer,
)

from typing import Any


class DatabaseRunTool(DatabaseTool):
    """Run the given SQL statement on the database and return result rows.

    Args:
        sql: SQL statement to execute.

    Returns:
        Rows returned by the SQL statement, if any.
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
    ) -> list[dict[str, Any]]:
        await self._sleep_if_configured()

        async with self._engine.begin() as conn:
            normalized_sql = self._prepare_sql_for_execution(sql)
            sql_to_run = await conn.run_sync(
                self._apply_identifier_case, normalized_sql
            )
            result = await conn.exec_driver_sql(sql_to_run)

            if result.returns_rows:
                return [dict(row) for row in result.mappings().all()]
        return []
