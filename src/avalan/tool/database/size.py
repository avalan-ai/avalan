from ...entities import ToolCallContext
from . import (
    DatabaseTool,
    DatabaseToolSettings,
    IdentifierCaseNormalizer,
    TableSize,
    TableSizeMetric,
)

from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine


class DatabaseSizeTool(DatabaseTool):
    """Summarize storage usage for a database table.

    Args:
        table_name: Table to analyze (optionally schema-qualified).

    Returns:
        Size metrics describing how much space the table occupies.
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
        self.__name__ = "size"

    @staticmethod
    def _split_schema_and_table(qualified: str) -> tuple[str | None, str]:
        if "." in qualified:
            schema, table = qualified.split(".", 1)
            return (schema or None), table
        return None, qualified

    async def __call__(
        self, table_name: str, *, context: ToolCallContext
    ) -> TableSize:
        assert table_name, "table_name must not be empty"
        await self._sleep_if_configured()

        async with self._engine.connect() as connection:
            schema, tbl = self._split_schema_and_table(table_name)
            return await connection.run_sync(
                self._collect,
                schema=schema,
                table_name=tbl,
            )

    def _collect(
        self,
        connection: Connection,
        *,
        schema: str | None,
        table_name: str,
    ) -> TableSize:
        inspector = self._inspect_connection(connection)
        default_schema, _ = self._schemas(connection, inspector)
        effective_schema = schema or default_schema
        actual_table = self._denormalize_table_name(
            connection, effective_schema, table_name
        )

        metrics = self._metrics_for_dialect(
            connection, effective_schema, actual_table
        )

        display_table = self._normalize_table_for_output(actual_table)
        schema_display = self._normalize_schema_for_output(effective_schema)

        if effective_schema and effective_schema != default_schema:
            if schema_display:
                name = f"{schema_display}.{display_table}"
            else:
                name = f"{effective_schema}.{display_table}"
        else:
            name = display_table

        return TableSize(name=name, schema=schema_display, metrics=tuple(metrics))

    def _metrics_for_dialect(
        self,
        connection: Connection,
        schema: str | None,
        table_name: str,
    ) -> list[TableSizeMetric]:
        dialect = getattr(connection.dialect, "name", "") or ""

        if dialect == "postgresql":
            return self._collect_postgresql(connection, schema, table_name)
        if dialect in {"mysql", "mariadb"}:
            return self._collect_mysql(connection, schema, table_name)
        if dialect == "sqlite":
            return self._collect_sqlite(connection, table_name)
        if dialect == "oracle":
            return self._collect_oracle(connection, schema, table_name)
        if dialect.startswith("mssql"):
            return self._collect_mssql(connection, schema, table_name)
        return []

    def _collect_postgresql(
        self,
        connection: Connection,
        schema: str | None,
        table_name: str,
    ) -> list[TableSizeMetric]:
        statement = text(
            """
            SELECT
                pg_table_size(c.oid) AS data_bytes,
                pg_indexes_size(c.oid) AS index_bytes,
                pg_total_relation_size(c.oid) AS total_bytes
            FROM pg_class AS c
            JOIN pg_namespace AS n ON n.oid = c.relnamespace
            WHERE c.relname = :table_name
              AND (:schema IS NULL OR n.nspname = :schema)
            LIMIT 1
            """
        )

        row = connection.execute(
            statement,
            {"table_name": table_name, "schema": schema},
        ).mappings().first()

        if not row:
            return []

        data_bytes = self._int_or_none(row.get("data_bytes"))
        index_bytes = self._int_or_none(row.get("index_bytes"))
        total_bytes = self._int_or_none(row.get("total_bytes"))
        if total_bytes is None:
            total_bytes = self._combine_bytes(data_bytes, index_bytes)

        metrics = [self._metric("data", data_bytes)]
        metrics.append(self._metric("indexes", index_bytes))
        metrics.append(self._metric("total", total_bytes))
        return [metric for metric in metrics if metric.bytes is not None]

    def _collect_mysql(
        self,
        connection: Connection,
        schema: str | None,
        table_name: str,
    ) -> list[TableSizeMetric]:
        statement = text(
            """
            SELECT
                DATA_LENGTH AS data_bytes,
                INDEX_LENGTH AS index_bytes,
                DATA_FREE AS free_bytes
            FROM information_schema.tables
            WHERE TABLE_NAME = :table_name
              AND (:schema IS NULL OR TABLE_SCHEMA = :schema)
            LIMIT 1
            """
        )

        row = connection.execute(
            statement,
            {"table_name": table_name, "schema": schema},
        ).mappings().first()

        if not row:
            return []

        data_bytes = self._int_or_none(row.get("data_bytes"))
        index_bytes = self._int_or_none(row.get("index_bytes"))
        free_bytes = self._int_or_none(row.get("free_bytes"))
        total_bytes = self._combine_bytes(data_bytes, index_bytes)

        metrics = [self._metric("data", data_bytes)]
        metrics.append(self._metric("indexes", index_bytes))
        metrics.append(self._metric("free", free_bytes))
        metrics.append(self._metric("total", total_bytes))
        return [metric for metric in metrics if metric.bytes is not None]

    def _collect_sqlite(
        self, connection: Connection, table_name: str
    ) -> list[TableSizeMetric]:
        try:
            table_row = connection.execute(
                text(
                    "SELECT SUM(pgsize) AS size FROM dbstat WHERE name = :table_name"
                ),
                {"table_name": table_name},
            ).mappings().first()
        except Exception:
            return self._collect_sqlite_via_pages(connection)

        data_bytes = self._int_or_none(table_row.get("size") if table_row else None)

        index_rows: list[dict[str, Any]] = []
        try:
            pragma = table_name.replace("'", "''")
            index_rows = connection.execute(
                text(f"PRAGMA index_list('{pragma}')")
            ).mappings().all()
        except Exception:
            index_rows = []

        index_values: list[int] = []
        for row in index_rows:
            index_name = row.get("name")
            if not index_name:
                continue
            index_row = connection.execute(
                text(
                    "SELECT SUM(pgsize) AS size FROM dbstat WHERE name = :index_name"
                ),
                {"index_name": index_name},
            ).mappings().first()
            index_value = self._int_or_none(
                index_row.get("size") if index_row else None
            )
            if index_value is not None:
                index_values.append(index_value)

        index_bytes = sum(index_values) if index_values else None
        total_bytes = self._combine_bytes(data_bytes, index_bytes)

        metrics = []
        if data_bytes is not None:
            metrics.append(self._metric("data", data_bytes))
        if index_bytes is not None:
            metrics.append(self._metric("indexes", index_bytes))
        if total_bytes is not None:
            metrics.append(self._metric("total", total_bytes))
        return metrics

    def _collect_sqlite_via_pages(
        self, connection: Connection
    ) -> list[TableSizeMetric]:
        page_count = self._int_or_none(
            connection.execute(text("PRAGMA page_count")).scalar()
        )
        page_size = self._int_or_none(
            connection.execute(text("PRAGMA page_size")).scalar()
        )
        total_bytes = self._multiply(page_count, page_size)
        if total_bytes is None:
            return []
        return [self._metric("total", total_bytes)]

    def _collect_oracle(
        self,
        connection: Connection,
        schema: str | None,
        table_name: str,
    ) -> list[TableSizeMetric]:
        owner = schema.upper() if schema else None
        table_key = table_name.upper()

        data_row = connection.execute(
            text(
                """
                SELECT SUM(bytes) AS bytes
                FROM all_segments
                WHERE segment_name = :table_name
                  AND segment_type LIKE 'TABLE%'
                  AND (:owner IS NULL OR owner = :owner)
                """
            ),
            {"table_name": table_key, "owner": owner},
        ).mappings().first()

        data_bytes = self._int_or_none(data_row.get("bytes") if data_row else None)

        index_rows = connection.execute(
            text(
                """
                SELECT index_name
                FROM all_indexes
                WHERE table_name = :table_name
                  AND (:owner IS NULL OR owner = :owner)
                """
            ),
            {"table_name": table_key, "owner": owner},
        ).mappings().all()

        index_total = 0
        index_found = False
        for row in index_rows:
            index_name = row.get("index_name")
            if not index_name:
                continue
            seg_row = connection.execute(
                text(
                    """
                    SELECT SUM(bytes) AS bytes
                    FROM all_segments
                    WHERE segment_name = :index_name
                      AND (:owner IS NULL OR owner = :owner)
                    """
                ),
                {"index_name": index_name, "owner": owner},
            ).mappings().first()
            seg_value = self._int_or_none(
                seg_row.get("bytes") if seg_row else None
            )
            if seg_value is not None:
                index_found = True
                index_total += seg_value

        index_bytes = index_total if index_found else None
        total_bytes = self._combine_bytes(data_bytes, index_bytes)

        metrics = []
        if data_bytes is not None:
            metrics.append(self._metric("data", data_bytes))
        if index_bytes is not None:
            metrics.append(self._metric("indexes", index_bytes))
        if total_bytes is not None:
            metrics.append(self._metric("total", total_bytes))
        return metrics

    def _collect_mssql(
        self,
        connection: Connection,
        schema: str | None,
        table_name: str,
    ) -> list[TableSizeMetric]:
        statement = text(
            """
            SELECT
                SUM(CASE WHEN i.index_id <= 1 THEN ps.used_page_count ELSE 0 END)
                    * 8192 AS data_bytes,
                SUM(CASE WHEN i.index_id > 1 THEN ps.used_page_count ELSE 0 END)
                    * 8192 AS index_bytes
            FROM sys.dm_db_partition_stats AS ps
            JOIN sys.indexes AS i
                ON ps.object_id = i.object_id AND ps.index_id = i.index_id
            JOIN sys.tables AS t ON t.object_id = i.object_id
            WHERE t.name = :table_name
              AND (:schema IS NULL OR SCHEMA_NAME(t.schema_id) = :schema)
            """
        )

        row = connection.execute(
            statement,
            {"table_name": table_name, "schema": schema},
        ).mappings().first()

        if not row:
            return []

        data_bytes = self._int_or_none(row.get("data_bytes"))
        index_bytes = self._int_or_none(row.get("index_bytes"))
        total_bytes = self._combine_bytes(data_bytes, index_bytes)

        metrics = []
        if data_bytes is not None:
            metrics.append(self._metric("data", data_bytes))
        if index_bytes is not None:
            metrics.append(self._metric("indexes", index_bytes))
        if total_bytes is not None:
            metrics.append(self._metric("total", total_bytes))
        return metrics

    def _metric(self, category: str, value: int | None) -> TableSizeMetric:
        return TableSizeMetric(
            category=category,
            bytes=value,
            human_readable=self._format_bytes(value),
        )

    def _normalize_schema_for_output(self, schema: str | None) -> str | None:
        if schema is None:
            return None
        if self._normalizer is None:
            return schema
        return self._normalizer.normalize(schema)

    @staticmethod
    def _combine_bytes(*values: int | None) -> int | None:
        total = 0
        has_value = False
        for value in values:
            if value is None:
                continue
            total += value
            has_value = True
        return total if has_value else None

    @staticmethod
    def _multiply(a: int | None, b: int | None) -> int | None:
        if a is None or b is None:
            return None
        return a * b

    @staticmethod
    def _int_or_none(value: Any) -> int | None:
        if value is None:
            return None
        try:
            number = int(value)
        except (TypeError, ValueError):
            return None
        if number < 0:
            return None
        return number

    @staticmethod
    def _format_bytes(value: int | None) -> str | None:
        if value is None:
            return None
        if value < 0:
            return None
        units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
        size = float(value)
        index = 0
        while size >= 1024 and index < len(units) - 1:
            size /= 1024
            index += 1
        if index == 0:
            return f"{int(size)} {units[index]}"
        return f"{size:.1f} {units[index]}"
