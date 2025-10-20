from ...entities import ToolCallContext
from . import (
    DatabaseTool,
    DatabaseToolSettings,
    IdentifierCaseNormalizer,
)

from typing import Any, Iterable, Sequence

from sqlalchemy import MetaData, Table as SATable, select, text
from sqlalchemy.sql import Select
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.ext.asyncio import AsyncEngine


class DatabaseSampleTool(DatabaseTool):
    """Return a limited sample of rows from the given table.

    Args:
        table_name: Table to sample rows from (optionally schema-qualified).
        columns: Optional list of column names to include in the result.
        conditions: Optional SQL expression to filter returned rows.
        order: Optional mapping of column names to sort direction ('asc' or 'desc').
        count: Optional maximum number of rows to return.

    Returns:
        Sampled rows represented as dictionaries keyed by column name.
    """

    _DEFAULT_COUNT = 10

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
        self.__name__ = "sample"

    @staticmethod
    def _split_schema_and_table(qualified: str) -> tuple[str | None, str]:
        if "." in qualified:
            schema, table = qualified.split(".", 1)
            return (schema or None), table
        return None, qualified

    async def __call__(
        self,
        table_name: str,
        *,
        context: ToolCallContext,
        columns: Sequence[str] | None = None,
        conditions: str | None = None,
        order: dict[str, str] | None = None,
        count: int | None = None,
    ) -> list[dict[str, object]]:
        assert table_name, "table_name must not be empty"
        if columns is not None:
            assert all(columns), "columns must contain valid names"
        if order is not None:
            assert order, "order must not be empty when provided"
        if count is not None:
            assert count > 0, "count must be a positive integer"

        await self._sleep_if_configured()

        async with self._engine.connect() as conn:
            schema, table = self._split_schema_and_table(table_name)
            actual_table = await conn.run_sync(
                self._denormalize_table_name, schema, table
            )
            statement = await conn.run_sync(
                self._build_select_statement,
                schema=schema,
                actual_table=actual_table,
                requested_columns=columns,
                conditions=conditions,
                order=order,
                limit=count,
            )
            result = await conn.execute(statement)
            rows = result.mappings().all()
        return [dict(row) for row in rows]

    def _build_select_statement(
        self,
        connection,
        *,
        schema: str | None,
        actual_table: str,
        requested_columns: Sequence[str] | None,
        conditions: str | None,
        order: dict[str, str] | None,
        limit: int | None,
    ) -> Select:
        table = self._reflect_table(connection, schema, actual_table)

        if requested_columns:
            selection = self._resolve_columns(table, requested_columns)
        else:
            selection = list(table.c)

        stmt = select(*selection)

        if conditions:
            stmt = stmt.where(text(conditions))

        if order:
            orderings = self._build_ordering(table, order)
            stmt = stmt.order_by(*orderings)

        row_limit = limit if limit is not None else self._DEFAULT_COUNT
        stmt = stmt.limit(row_limit)
        return stmt

    def _build_ordering(
        self, table: SATable, order: dict[str, str]
    ) -> Iterable[ColumnElement[Any]]:
        clauses: list[ColumnElement[Any]] = []
        for column_name, direction in order.items():
            column = self._resolve_column(table, column_name)
            direction_value = str(direction).lower()
            assert direction_value in {"asc", "desc"}, (
                "order directions must be 'asc' or 'desc'"
            )
            if direction_value == "desc":
                clauses.append(column.desc())
            else:
                clauses.append(column.asc())
        return clauses

    def _resolve_columns(
        self, table: SATable, columns: Sequence[str]
    ) -> list[ColumnElement[Any]]:
        return [self._resolve_column(table, name) for name in columns]

    def _resolve_column(
        self, table: SATable, name: str
    ) -> ColumnElement[Any]:
        lookup = {col.name: col for col in table.c}
        if self._normalizer is not None:
            for col in table.c:
                normalized = self._normalizer.normalize(col.name)
                lookup.setdefault(normalized, col)

        column = lookup.get(name)
        if column is None and self._normalizer is not None:
            column = lookup.get(self._normalizer.normalize(name))
        if column is None:
            raise ValueError(f"Column '{name}' does not exist on table '{table.name}'")
        return column

    def _reflect_table(
        self, connection, schema: str | None, table_name: str
    ) -> SATable:
        return SATable(
            table_name,
            MetaData(),
            schema=schema,
            autoload_with=connection,
        )


