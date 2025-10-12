from importlib import import_module
from typing import Any

from ...entities import ToolCallContext
from . import (
    DatabaseTask,
    DatabaseTool,
    DatabaseToolSettings,
    IdentifierCaseNormalizer,
)

from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine

_database = import_module(__package__)

class DatabaseTasksTool(DatabaseTool):
    """List killable database tasks from supported engines.

    Args:
        running_for: Minimum number of seconds a task must be running to be
            included. Provide ``None`` to return all tasks regardless of
            duration.

    Returns:
        Tasks that are currently running on PostgreSQL or MySQL connections.
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
        self.__name__ = "tasks"

    async def __call__(
        self,
        *,
        running_for: int | None = None,
        context: ToolCallContext,
    ) -> list[DatabaseTask]:
        if running_for is not None:
            assert running_for >= 0, "running_for must be zero or greater"
        if self._settings.delay_secs:
            await _database.sleep(self._settings.delay_secs)

        async with self._engine.connect() as conn:
            result = await conn.run_sync(
                self._collect, running_for=running_for
            )
        return result

    def _collect(
        self, connection: Connection, *, running_for: int | None = None
    ) -> list[DatabaseTask]:
        dialect_name = getattr(connection.dialect, "name", None)

        if dialect_name == "postgresql":
            return self._collect_postgresql(connection, running_for=running_for)
        if dialect_name in {"mysql", "mariadb"}:
            return self._collect_mysql(connection, running_for=running_for)
        return []

    def _collect_postgresql(
        self, connection: Connection, *, running_for: int | None = None
    ) -> list[DatabaseTask]:
        statement = text(
            """
            select pid::text as id,
                   usename as user_name,
                   state,
                   query,
                   CAST(EXTRACT(EPOCH FROM clock_timestamp() - query_start) AS BIGINT)
                       AS duration
            from pg_stat_activity
            where pid <> pg_backend_pid()
              and query is not null
              and state is not null
            """
        )
        result = connection.execute(statement)
        tasks: list[DatabaseTask] = []
        for row in result.mappings().all():
            query = (row.get("query") or "").strip()
            if not query:
                continue
            duration = self._normalize_duration(row.get("duration"))
            if running_for is not None and (
                duration is None or duration < running_for
            ):
                continue
            tasks.append(
                DatabaseTask(
                    id=str(row.get("id")),
                    user=row.get("user_name"),
                    state=row.get("state"),
                    query=query,
                    duration=duration,
                )
            )
        return tasks

    def _collect_mysql(
        self, connection: Connection, *, running_for: int | None = None
    ) -> list[DatabaseTask]:
        current_id = connection.scalar(text("SELECT CONNECTION_ID()"))
        result = connection.execute(text("SHOW FULL PROCESSLIST"))
        tasks: list[DatabaseTask] = []

        for row in result.mappings().all():
            identifier = row.get("Id")
            if identifier is None:
                continue
            if current_id is not None and identifier == current_id:
                continue

            command = (row.get("Command") or "").strip().lower()
            if command == "sleep":
                continue

            info = row.get("Info")
            query = (str(info).strip() if info is not None else "")
            if not query:
                continue

            duration = self._normalize_duration(row.get("Time"))
            if running_for is not None and (
                duration is None or duration < running_for
            ):
                continue

            tasks.append(
                DatabaseTask(
                    id=str(identifier),
                    user=row.get("User"),
                    state=row.get("State") or row.get("Command"),
                    query=query,
                    duration=duration,
                )
            )
        return tasks

    @staticmethod
    def _normalize_duration(value: Any) -> int | None:
        if value is None:
            return None
        try:
            duration = int(value)
        except (TypeError, ValueError):
            return None
        if duration < 0:
            return 0
        return duration

