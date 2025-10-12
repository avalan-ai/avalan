from importlib import import_module

from ...entities import ToolCallContext
from . import (
    DatabaseTool,
    DatabaseToolSettings,
    IdentifierCaseNormalizer,
    QueryPlan,
)

from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.sql.elements import TextClause

_database = import_module(__package__)


class DatabasePlanTool(DatabaseTool):
    """Explain how the database will execute the provided SQL statement.

    Args:
        sql: SQL statement to analyze.

    Returns:
        Query plan describing the execution strategy for the SQL statement.
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
        self.__name__ = "plan"

    async def __call__(self, sql: str, *, context: ToolCallContext) -> QueryPlan:
        if self._settings.delay_secs:
            await _database.sleep(self._settings.delay_secs)

        async with self._engine.connect() as conn:
            normalized_sql = self._prepare_sql_for_execution(sql)
            rewritten_sql = await conn.run_sync(
                self._apply_identifier_case, normalized_sql
            )
            statement, dialect = await conn.run_sync(
                self._statement_for_plan, rewritten_sql
            )
            result = await conn.execute(statement)

            if not result.returns_rows:
                return QueryPlan(dialect=dialect, steps=[])

            rows = [dict(row) for row in result.mappings().all()]
            return QueryPlan(dialect=dialect, steps=rows)

    def _statement_for_plan(
        self, connection: Connection, sql: str
    ) -> tuple[TextClause, str]:
        dialect = connection.dialect.name

        if dialect == "sqlite":
            prefix = "EXPLAIN QUERY PLAN "
        elif dialect == "postgresql":
            prefix = "EXPLAIN (FORMAT TEXT) "
        elif dialect in {"mysql", "mariadb"}:
            prefix = "EXPLAIN FORMAT=JSON "
        else:
            prefix = "EXPLAIN "

        return text(f"{prefix}{sql}"), dialect

