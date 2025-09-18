from .. import di_get_logger, di_set
from ..entities import EngineRequest, OrchestratorContext
from ...agent.loader import OrchestratorLoader
from ...agent.orchestrator import Orchestrator
from ...tool.context import ToolSettingsContext
from ...tool.database import DatabaseToolSettings
from contextlib import AbstractAsyncContextManager
from dataclasses import replace
from fastapi import APIRouter, Depends, Request
from logging import Logger
from uuid import UUID

router = APIRouter()


@router.post("/engine")
async def set_engine(
    request: Request,
    engine: EngineRequest,
    logger: Logger = Depends(di_get_logger),
) -> dict[str, str | None]:
    """Reload orchestrator with a new engine URI."""
    stack = request.app.state.stack
    await stack.aclose()
    ctx: OrchestratorContext = request.app.state.ctx
    loader: OrchestratorLoader = request.app.state.loader
    agent_id: UUID | None = getattr(request.app.state, "agent_id", None)

    if hasattr(request.app.state, "orchestrator"):
        delattr(request.app.state, "orchestrator")

    tool_settings = _merge_tool_settings(ctx.tool_settings, engine.database)

    try:
        orchestrator_cm, new_ctx = await _load_orchestrator(
            loader=loader,
            ctx=ctx,
            agent_id=agent_id,
            uri=engine.uri,
            tool_settings=tool_settings,
        )
        orchestrator = await stack.enter_async_context(orchestrator_cm)
    except Exception as error:
        request.app.state.ctx = ctx
        request.app.state.agent_id = agent_id
        try:
            restore_cm, restore_ctx = await _load_orchestrator(
                loader=loader,
                ctx=ctx,
                agent_id=agent_id,
                uri=None,
                tool_settings=ctx.tool_settings,
            )
            orchestrator = await stack.enter_async_context(restore_cm)
        except Exception as restore_error:
            raise restore_error from error
        else:
            request.app.state.ctx = restore_ctx
            request.app.state.agent_id = orchestrator.id
            di_set(request.app, logger=logger, orchestrator=orchestrator)
            raise error.with_traceback(error.__traceback__)

    request.app.state.ctx = new_ctx
    request.app.state.agent_id = orchestrator.id
    di_set(request.app, logger=logger, orchestrator=orchestrator)
    return {"uri": new_ctx.settings.uri if new_ctx.settings else engine.uri}


def _merge_tool_settings(
    tool_settings: ToolSettingsContext | None, database_uri: str | None
) -> ToolSettingsContext | None:
    if database_uri is None:
        return tool_settings
    db_settings = DatabaseToolSettings(dsn=database_uri)
    if tool_settings:
        return replace(tool_settings, database=db_settings)
    return ToolSettingsContext(database=db_settings)


async def _load_orchestrator(
    *,
    loader: OrchestratorLoader,
    ctx: OrchestratorContext,
    agent_id: UUID | None,
    uri: str | None,
    tool_settings: ToolSettingsContext | None,
) -> tuple[AbstractAsyncContextManager[Orchestrator], OrchestratorContext]:
    if ctx.specs_path:
        orchestrator_cm = await loader.from_file(
            ctx.specs_path,
            agent_id=agent_id,
            uri=uri,
            tool_settings=tool_settings,
        )
        new_ctx = OrchestratorContext(
            participant_id=ctx.participant_id,
            specs_path=ctx.specs_path,
            settings=ctx.settings,
            tool_settings=tool_settings,
        )
    else:
        assert ctx.settings
        settings = (
            replace(ctx.settings, uri=uri)
            if uri is not None
            else ctx.settings
        )
        orchestrator_cm = await loader.from_settings(
            settings, tool_settings=tool_settings
        )
        new_ctx = OrchestratorContext(
            participant_id=ctx.participant_id,
            specs_path=ctx.specs_path,
            settings=settings,
            tool_settings=tool_settings,
        )
    return orchestrator_cm, new_ctx
