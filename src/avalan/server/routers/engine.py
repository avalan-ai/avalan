from .. import di_get_logger, di_set
from ..entities import EngineRequest, OrchestratorContext
from ...agent.loader import OrchestratorLoader
from contextlib import AsyncExitStack
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
    stack: AsyncExitStack = request.app.state.stack
    await stack.aclose()
    ctx: OrchestratorContext = request.app.state.ctx
    new_stack = AsyncExitStack()
    loader = OrchestratorLoader(
        hub=ctx.hub,
        logger=logger,
        participant_id=ctx.participant_id,
        stack=new_stack,
    )
    if ctx.specs_path:
        orchestrator_cm = await loader.from_file(
            ctx.specs_path,
            agent_id=request.app.state.agent_id,
            uri=engine.uri,
        )
        ctx = OrchestratorContext(
            loader=loader,
            hub=ctx.hub,
            participant_id=ctx.participant_id,
            specs_path=ctx.specs_path,
            settings=ctx.settings,
            browser_settings=ctx.browser_settings,
            database_settings=ctx.database_settings,
        )
    else:
        assert ctx.settings
        settings = replace(ctx.settings, uri=engine.uri)
        orchestrator_cm = await loader.from_settings(
            settings,
            browser_settings=ctx.browser_settings,
            database_settings=ctx.database_settings,
        )
        ctx = OrchestratorContext(
            loader=loader,
            hub=ctx.hub,
            participant_id=ctx.participant_id,
            specs_path=ctx.specs_path,
            settings=settings,
            browser_settings=ctx.browser_settings,
            database_settings=ctx.database_settings,
        )
    orchestrator = await new_stack.enter_async_context(orchestrator_cm)
    request.app.state.ctx = ctx
    request.app.state.stack = new_stack
    request.app.state.agent_id = orchestrator.id
    di_set(request.app, logger=logger, orchestrator=orchestrator)
    return {"uri": engine.uri}
