from ...entities import ToolCallContext
from . import (
    AsyncEngine,
    Connection,
    DatabaseTool,
    DatabaseToolSettings,
    IdentifierCaseNormalizer,
    TableKey,
)


class DatabaseKeysTool(DatabaseTool):
    """List primary and unique keys defined on a table.

    Args:
        table_name: Table to inspect for key definitions.
        schema: Optional schema containing the table; defaults to the
            active schema.

    Returns:
        Key definitions describing the table's primary and unique constraints.
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
        self.__name__ = "keys"

    async def __call__(
        self,
        table_name: str,
        schema: str | None = None,
        *,
        context: ToolCallContext,
    ) -> list[TableKey]:
        assert table_name, "table_name must not be empty"
        await self._sleep_if_configured()

        async with self._engine.connect() as conn:
            return await conn.run_sync(
                self._collect,
                schema=schema,
                table_name=table_name,
            )

    def _collect(
        self,
        connection: Connection,
        *,
        schema: str | None,
        table_name: str,
    ) -> list[TableKey]:
        inspector = self._inspect_connection(connection)
        default_schema, _ = self._schemas(connection, inspector)
        resolved_schema = schema or default_schema

        actual_table = self._denormalize_table_name(
            connection, resolved_schema, table_name
        )

        keys: list[TableKey] = []

        pk = (
            inspector.get_pk_constraint(actual_table, schema=resolved_schema)
            or {}
        )
        pk_columns = tuple(
            pk.get("constrained_columns") or pk.get("column_names") or []
        )
        if pk_columns:
            keys.append(
                TableKey(
                    type="primary",
                    name=pk.get("name"),
                    columns=pk_columns,
                )
            )

        unique_constraints = (
            inspector.get_unique_constraints(
                actual_table,
                schema=resolved_schema,
            )
            or []
        )

        for constraint in unique_constraints:
            columns = tuple(
                constraint.get("column_names")
                or constraint.get("constrained_columns")
                or []
            )
            if not columns:
                continue
            keys.append(
                TableKey(
                    type="unique",
                    name=constraint.get("name"),
                    columns=columns,
                )
            )

        return keys
