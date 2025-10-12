from ...entities import ToolCallContext
from . import (
    DatabaseTool,
    DatabaseToolSettings,
    IdentifierCaseNormalizer,
)

from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine


class DatabaseKillTool(DatabaseTool):
    """Cancel a running database task by identifier.

    Args:
        task_id: Identifier of the task to cancel.

    Returns:
        True when cancellation succeeds on supported engines; otherwise False.
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
        self.__name__ = "kill"

    async def __call__(
        self,
        task_id: str,
        *,
        context: ToolCallContext,
    ) -> bool:
        assert task_id, "task_id must not be empty"
        await self._sleep_if_configured()

        async with self._engine.begin() as conn:
            return await conn.run_sync(self._kill, task_id=task_id)

    def _kill(self, connection: Connection, *, task_id: str) -> bool:
        dialect_name = getattr(connection.dialect, "name", None)

        if dialect_name == "postgresql":
            return self._kill_postgresql(connection, task_id)
        if dialect_name in {"mysql", "mariadb"}:
            return self._kill_mysql(connection, task_id)

        raise RuntimeError(
            "Killing tasks is not supported for "
            f"{dialect_name or 'unknown'} databases."
        )

    def _kill_postgresql(self, connection: Connection, task_id: str) -> bool:
        pid = self._parse_integer_task_id(task_id)
        statement = text(
            "SELECT pg_cancel_backend(:pid) AS cancelled"
        )
        result = connection.execute(statement, {"pid": pid})
        cancelled = result.scalar()
        return bool(cancelled)

    def _kill_mysql(self, connection: Connection, task_id: str) -> bool:
        pid = self._parse_integer_task_id(task_id)
        connection.execute(text("KILL :pid"), {"pid": pid})
        return True

    @staticmethod
    def _parse_integer_task_id(task_id: str) -> int:
        try:
            value = int(task_id)
        except ValueError as error:
            raise RuntimeError("Task identifier must be an integer value.") from error

        if value < 0:
            raise RuntimeError("Task identifier must be a positive integer.")
        return value

