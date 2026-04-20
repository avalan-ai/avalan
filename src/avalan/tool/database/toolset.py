from ...compat import override
from .. import ToolSet
from . import AsyncEngine, DatabaseTool, IdentifierCaseNormalizer
from .count import DatabaseCountTool
from .inspect import DatabaseInspectTool
from .keys import DatabaseKeysTool
from .kill import DatabaseKillTool
from .locks import DatabaseLocksTool
from .plan import DatabasePlanTool
from .relationships import DatabaseRelationshipsTool
from .run import DatabaseRunTool
from .sample import DatabaseSampleTool
from .settings import DatabaseToolSettings
from .size import DatabaseSizeTool
from .tables import DatabaseTablesTool
from .tasks import DatabaseTasksTool

from contextlib import AsyncExitStack
from types import TracebackType


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
        self._engine = DatabaseTool._create_engine(
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
            DatabaseKeysTool(
                self._engine,
                settings,
                normalizer=normalizer,
                table_cache=table_cache,
            ),
            DatabaseRelationshipsTool(
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
            DatabaseSampleTool(
                self._engine,
                settings,
                normalizer=normalizer,
                table_cache=table_cache,
            ),
            DatabaseSizeTool(
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
            DatabaseLocksTool(
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
        tb: TracebackType | None,
    ) -> bool | None:
        try:
            if self._engine is not None:
                await self._engine.dispose()
        finally:
            return await super().__aexit__(exc_type, exc, tb)
