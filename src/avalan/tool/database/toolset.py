from importlib import import_module

from . import (
    DatabaseCountTool,
    DatabaseInspectTool,
    DatabaseKillTool,
    DatabasePlanTool,
    DatabaseRunTool,
    DatabaseTablesTool,
    DatabaseTasksTool,
    DatabaseTool,
    DatabaseToolSettings,
    IdentifierCaseNormalizer,
)
from .. import ToolSet
from ...compat import override

from contextlib import AsyncExitStack

from sqlalchemy.ext.asyncio import AsyncEngine


_database = import_module(__package__)


class DatabaseToolSet(ToolSet):
    _engine: AsyncEngine
    _settings: DatabaseToolSettings

    @override
    def __init__(
        self,
        settings: DatabaseToolSettings,
        *,
        exit_stack: AsyncExitStack | None = None,
        namespace: str | None = None,
    ):
        self._settings = settings
        self._engine = _database.create_async_engine(
            self._settings.dsn, pool_pre_ping=True
        )
        DatabaseTool._configure_read_only_engine(
            self._engine, self._settings.read_only
        )

        normalizer = (
            IdentifierCaseNormalizer(settings.identifier_case)
            if settings.identifier_case != "preserve"
            else None
        )
        table_cache: dict[str | None, dict[str, str]] = {}

        tools = [
            DatabaseCountTool(
                self._engine,
                settings,
                normalizer=normalizer,
                table_cache=table_cache,
            ),
            DatabaseInspectTool(
                self._engine,
                settings,
                normalizer=normalizer,
                table_cache=table_cache,
            ),
            DatabasePlanTool(
                self._engine,
                settings,
                normalizer=normalizer,
                table_cache=table_cache,
            ),
            DatabaseRunTool(
                self._engine,
                settings,
                normalizer=normalizer,
                table_cache=table_cache,
            ),
            DatabaseTablesTool(
                self._engine,
                settings,
                normalizer=normalizer,
                table_cache=table_cache,
            ),
            DatabaseTasksTool(
                self._engine,
                settings,
                normalizer=normalizer,
                table_cache=table_cache,
            ),
            DatabaseKillTool(
                self._engine,
                settings,
                normalizer=normalizer,
                table_cache=table_cache,
            ),
        ]
        super().__init__(
            exit_stack=exit_stack, namespace=namespace, tools=tools
        )

    @override
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: BaseException | None,
    ) -> bool:
        try:
            if self._engine is not None:
                await self._engine.dispose()
        finally:
            return await super().__aexit__(exc_type, exc, tb)

