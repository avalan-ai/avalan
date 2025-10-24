from ...entities import ToolCallContext
from . import (
    AsyncEngine,
    Connection,
    DatabaseLock,
    DatabaseTool,
    DatabaseToolSettings,
    IdentifierCaseNormalizer,
    SQLAlchemyError,
    text,
)

from typing import Any


class DatabaseLocksTool(DatabaseTool):
    """List locks currently visible in supported database engines.

    Args:
        None.

    Returns:
        Locks currently held or awaited on PostgreSQL and MySQL connections.
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
        self.__name__ = "locks"

    async def __call__(
        self,
        *,
        context: ToolCallContext,
    ) -> list[DatabaseLock]:
        await self._sleep_if_configured()

        async with self._engine.connect() as connection:
            locks = await connection.run_sync(self._collect)
        return locks

    def _collect(self, connection: Connection) -> list[DatabaseLock]:
        dialect_name = getattr(connection.dialect, "name", "")

        if dialect_name == "postgresql":
            return self._collect_postgresql(connection)
        if dialect_name in {"mysql", "mariadb"}:
            return self._collect_mysql(connection)
        return []

    def _collect_postgresql(
        self, connection: Connection
    ) -> list[DatabaseLock]:
        statement = text(
            """
            SELECT
                a.pid::text AS pid,
                a.usename AS user_name,
                l.locktype,
                l.mode,
                l.granted,
                a.state,
                a.query,
                COALESCE(
                    NULLIF(n.nspname, '') || '.' || c.relname,
                    c.relname,
                    d.datname,
                    CAST(l.relation AS TEXT),
                    l.locktype
                ) AS lock_target,
                pg_blocking_pids(a.pid) AS blocking_pids
            FROM pg_locks AS l
            LEFT JOIN pg_stat_activity AS a ON a.pid = l.pid
            LEFT JOIN pg_class AS c ON c.oid = l.relation
            LEFT JOIN pg_namespace AS n ON n.oid = c.relnamespace
            LEFT JOIN pg_database AS d ON d.oid = l.database
            WHERE a.pid IS NOT NULL
            ORDER BY a.pid, l.locktype, l.mode
            """
        )

        result = connection.execute(statement)
        locks: list[DatabaseLock] = []
        for row in result.mappings().all():
            blocking = self._normalize_blocking(row.get("blocking_pids"))
            lock_target = self._string_or_none(row.get("lock_target"))
            query = self._normalize_query(row.get("query"))
            locks.append(
                DatabaseLock(
                    pid=self._string_or_none(row.get("pid")),
                    user=self._string_or_none(row.get("user_name")),
                    lock_type=self._string_or_none(row.get("locktype")),
                    lock_target=lock_target,
                    mode=self._string_or_none(row.get("mode")),
                    granted=self._bool_or_none(row.get("granted")),
                    blocking=blocking,
                    state=self._string_or_none(row.get("state")),
                    query=query,
                )
            )
        return locks

    def _collect_mysql(self, connection: Connection) -> list[DatabaseLock]:
        statement = text(
            """
            SELECT
                t.PROCESSLIST_ID AS pid,
                t.PROCESSLIST_USER AS user_name,
                dl.OBJECT_SCHEMA AS lock_schema,
                dl.OBJECT_NAME AS lock_name,
                dl.LOCK_TYPE AS lock_type,
                dl.LOCK_MODE AS lock_mode,
                dl.LOCK_STATUS AS lock_status,
                dl.LOCK_DATA AS lock_data,
                t.PROCESSLIST_STATE AS state,
                t.PROCESSLIST_INFO AS query,
                GROUP_CONCAT(
                    DISTINCT tb.PROCESSLIST_ID
                    ORDER BY tb.PROCESSLIST_ID SEPARATOR ','
                ) AS blocking_pids
            FROM performance_schema.data_locks AS dl
            JOIN performance_schema.threads AS t ON t.THREAD_ID = dl.THREAD_ID
            LEFT JOIN performance_schema.data_lock_waits AS dw
                ON dw.REQUESTING_ENGINE_LOCK_ID = dl.ENGINE_LOCK_ID
            LEFT JOIN performance_schema.data_locks AS bl
                ON bl.ENGINE_LOCK_ID = dw.BLOCKING_ENGINE_LOCK_ID
            LEFT JOIN performance_schema.threads AS tb
                ON tb.THREAD_ID = bl.THREAD_ID
            GROUP BY
                t.PROCESSLIST_ID,
                t.PROCESSLIST_USER,
                dl.OBJECT_SCHEMA,
                dl.OBJECT_NAME,
                dl.LOCK_TYPE,
                dl.LOCK_MODE,
                dl.LOCK_STATUS,
                dl.LOCK_DATA,
                t.PROCESSLIST_STATE,
                t.PROCESSLIST_INFO
            """
        )

        try:
            result = connection.execute(statement)
        except SQLAlchemyError:
            return []

        locks: list[DatabaseLock] = []
        for row in result.mappings().all():
            blocking = self._normalize_blocking(row.get("blocking_pids"))
            lock_target = self._normalize_mysql_target(
                row.get("lock_schema"),
                row.get("lock_name"),
                row.get("lock_data"),
            )
            query = self._normalize_query(row.get("query"))
            locks.append(
                DatabaseLock(
                    pid=self._string_or_none(row.get("pid")),
                    user=self._string_or_none(row.get("user_name")),
                    lock_type=self._string_or_none(row.get("lock_type")),
                    lock_target=lock_target,
                    mode=self._string_or_none(row.get("lock_mode")),
                    granted=self._mysql_granted(row.get("lock_status")),
                    blocking=blocking,
                    state=self._string_or_none(row.get("state")),
                    query=query,
                )
            )
        return locks

    @staticmethod
    def _normalize_blocking(value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            items = [item.strip() for item in value.split(",") if item.strip()]
            return tuple(items)
        if isinstance(value, (list, tuple, set)):
            items = [
                str(item)
                for item in value
                if item is not None and str(item).strip()
            ]
            return tuple(items)
        text_value = str(value).strip("{} ")
        if not text_value:
            return ()
        return tuple(
            part.strip() for part in text_value.split(",") if part.strip()
        )

    @staticmethod
    def _normalize_query(value: Any) -> str | None:
        if value is None:
            return None
        text_value = str(value).strip()
        return text_value or None

    @staticmethod
    def _string_or_none(value: Any) -> str | None:
        if value is None:
            return None
        text_value = str(value)
        return text_value if text_value else None

    @staticmethod
    def _bool_or_none(value: Any) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"t", "true", "1", "yes", "y"}:
                return True
            if lowered in {"f", "false", "0", "no", "n"}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return None

    @staticmethod
    def _normalize_mysql_target(
        schema: Any,
        name: Any,
        data: Any,
    ) -> str | None:
        schema_text = DatabaseLocksTool._string_or_none(schema)
        name_text = DatabaseLocksTool._string_or_none(name)
        data_text = DatabaseLocksTool._string_or_none(data)

        if schema_text and name_text:
            return f"{schema_text}.{name_text}"
        if name_text:
            return name_text
        return data_text

    @staticmethod
    def _mysql_granted(value: Any) -> bool | None:
        if value is None:
            return None
        text_value = str(value).strip().lower()
        if not text_value:
            return None
        if text_value == "granted":
            return True
        if text_value == "waiting":
            return False
        return None
