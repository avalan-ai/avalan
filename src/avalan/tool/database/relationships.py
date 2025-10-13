from ...entities import ToolCallContext
from . import (
    DatabaseTool,
    DatabaseToolSettings,
    IdentifierCaseNormalizer,
    TableRelationship,
)

from sqlalchemy.engine import Connection
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.ext.asyncio import AsyncEngine


class DatabaseRelationshipsTool(DatabaseTool):
    """List incoming and outgoing table relationships.

    Args:
        table_name: Table to inspect for foreign key relationships.
        schema: Optional schema containing the table; defaults to the active schema.

    Returns:
        Relationships describing how the table links to other tables.
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
        self.__name__ = "relationships"

    async def __call__(
        self,
        table_name: str,
        *,
        schema: str | None = None,
        context: ToolCallContext,
    ) -> list[TableRelationship]:
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
    ) -> list[TableRelationship]:
        inspector = self._inspect_connection(connection)
        default_schema, schemas = self._schemas(connection, inspector)
        actual_schema = schema or default_schema
        actual_table = self._denormalize_table_name(
            connection, actual_schema, table_name
        )

        try:
            inspector.get_columns(actual_table, schema=actual_schema)
        except NoSuchTableError:
            return []

        outgoing = self._collect_outgoing(
            inspector,
            actual_table,
            actual_schema,
            default_schema,
        )
        incoming = self._collect_incoming(
            inspector,
            actual_table,
            actual_schema,
            default_schema,
            schemas,
        )
        return [*outgoing, *incoming]

    def _collect_outgoing(
        self,
        inspector: Inspector,
        table_name: str,
        schema: str | None,
        default_schema: str | None,
    ) -> list[TableRelationship]:
        try:
            foreign_keys = inspector.get_foreign_keys(table_name, schema=schema) or []
        except NoSuchTableError:
            return []

        relationships: list[TableRelationship] = []
        for fk in foreign_keys:
            referred_table = fk.get("referred_table")
            if not referred_table:
                continue
            target_schema = fk.get("referred_schema") or schema
            relationships.append(
                TableRelationship(
                    direction="outgoing",
                    local_columns=tuple(
                        fk.get("constrained_columns") or ()
                    ),
                    related_table=self._format_related_table(
                        referred_table,
                        target_schema,
                        default_schema,
                    ),
                    related_columns=tuple(
                        fk.get("referred_columns") or ()
                    ),
                    constraint_name=fk.get("name"),
                )
            )
        return relationships

    def _collect_incoming(
        self,
        inspector: Inspector,
        table_name: str,
        schema: str | None,
        default_schema: str | None,
        schemas: list[str | None],
    ) -> list[TableRelationship]:
        relationships: list[TableRelationship] = []
        local_schema = self._normalize_schema(schema, default_schema)

        for schema_name in schemas:
            try:
                table_names = inspector.get_table_names(schema=schema_name) or []
            except Exception:
                continue

            tables = list(table_names)
            self._register_table_names(schema_name, tables)

            for candidate in tables:
                candidate_schema = self._normalize_schema(
                    schema_name, default_schema
                )
                if (
                    candidate == table_name
                    and candidate_schema == local_schema
                ):
                    continue

                try:
                    foreign_keys = (
                        inspector.get_foreign_keys(candidate, schema=schema_name) or []
                    )
                except NoSuchTableError:
                    continue

                for fk in foreign_keys:
                    referred_table = fk.get("referred_table")
                    if not referred_table:
                        continue

                    target_schema = fk.get("referred_schema") or schema_name
                    if referred_table != table_name:
                        continue

                    target_normalized = self._normalize_schema(
                        target_schema, default_schema
                    )
                    if target_normalized != local_schema:
                        continue

                    relationships.append(
                        TableRelationship(
                            direction="incoming",
                            local_columns=tuple(
                                fk.get("referred_columns") or ()
                            ),
                            related_table=self._format_related_table(
                                candidate,
                                schema_name,
                                default_schema,
                            ),
                            related_columns=tuple(
                                fk.get("constrained_columns") or ()
                            ),
                            constraint_name=fk.get("name"),
                        )
                    )

        return relationships

    @staticmethod
    def _normalize_schema(
        schema: str | None, default_schema: str | None
    ) -> str | None:
        if schema is not None:
            return schema
        return default_schema

    def _format_related_table(
        self,
        table_name: str,
        schema: str | None,
        default_schema: str | None,
    ) -> str:
        display = self._normalize_table_for_output(table_name)
        schema_name = self._normalize_schema(schema, default_schema)
        if schema_name is None or schema_name == default_schema:
            return display
        return f"{schema_name}.{display}"
