from importlib import import_module

from ...entities import ToolCallContext
from . import (
    DatabaseTool,
    DatabaseToolSettings,
    IdentifierCaseNormalizer,
)

from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine

_database = import_module(__package__)


class DatabaseTablesTool(DatabaseTool):
    """List table names available in the database grouped by schema.

    Args:
        None.

    Returns:
        Mapping of schema names to the tables they contain.
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
        if self._settings.delay_secs:
            await _database.sleep(self._settings.delay_secs)

        async with self._engine.connect() as conn:
            result = await conn.run_sync(self._collect)
        return result

    def _collect(self, connection: Connection) -> dict[str | None, list[str]]:
        inspector = _database.inspect(connection)
        _, schemas = self._schemas(connection, inspector)
        result: dict[str | None, list[str]] = {}
        for schema in schemas:
            actual_tables = inspector.get_table_names(schema=schema)
            self._register_table_names(schema, actual_tables)
            result[schema] = [
                self._normalize_table_for_output(name)
                for name in actual_tables
            ]
        return result

