from .. import di_get_logger, di_set
from ..entities import EngineRequest, OrchestratorContext
from ...tool.context import ToolSettingsContext
from ...tool.database import DatabaseToolSettings
from dataclasses import replace
from fastapi import APIRouter, Depends, Request
from logging import Logger

router = APIRouter()


@router.post("/engine")
async def set_engine(
    request: Request,
    engine: EngineRequest,
    logger: Logger = Depends(di_get_logger),
) -> dict[str, str]:
    """Reload orchestrator with a new engine URI."""
    stack = request.app.state.stack
    await stack.aclose()
    ctx: OrchestratorContext = request.app.state.ctx
    loader = request.app.state.loader
    tool_settings = ctx.tool_settings
    if engine.database is not None:
        db_settings = DatabaseToolSettings(dsn=engine.database)
        tool_settings = (
            replace(tool_settings, database=db_settings)
            if tool_settings
            else ToolSettingsContext(database=db_settings)
        )
    if ctx.specs_path:
        orchestrator_cm = await loader.from_file(
            ctx.specs_path,
            agent_id=request.app.state.agent_id,
            uri=engine.uri,
            tool_settings=tool_settings,
        )
        ctx = OrchestratorContext(
            participant_id=ctx.participant_id,
            specs_path=ctx.specs_path,
            settings=ctx.settings,
            tool_settings=tool_settings,
        )
    else:
        assert ctx.settings
        settings = replace(ctx.settings, uri=engine.uri)
        orchestrator_cm = await loader.from_settings(
            settings, tool_settings=tool_settings
        )
        ctx = OrchestratorContext(
            participant_id=ctx.participant_id,
            specs_path=ctx.specs_path,
            settings=settings,
            tool_settings=tool_settings,
        )
    request.app.state.ctx = ctx
    orchestrator = await stack.enter_async_context(orchestrator_cm)
    request.app.state.agent_id = orchestrator.id
    di_set(request.app, logger=logger, orchestrator=orchestrator)
    return {"uri": engine.uri}
