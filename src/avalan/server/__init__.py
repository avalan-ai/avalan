from ..agent.loader import OrchestratorLoader
from ..agent.orchestrator import Orchestrator
from ..entities import OrchestratorSettings
from ..model.hubs.huggingface import HuggingfaceHub
from ..tool.context import ToolSettingsContext
from ..utils import logger_replace
from .entities import OrchestratorContext
from .routers import mcp as mcp_router
from contextlib import AsyncExitStack, asynccontextmanager
from importlib import import_module
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from logging import Logger
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from uvicorn import Server


def agents_server(
    hub: HuggingfaceHub,
    name: str,
    version: str,
    host: str,
    port: int,
    reload: bool,
    specs_path: str | None,
    settings: OrchestratorSettings | None,
    tool_settings: ToolSettingsContext | None,
    mcp_prefix: str,
    openai_prefix: str,
    mcp_name: str,
    mcp_description: str | None,
    logger: Logger,
    agent_id: UUID | None = None,
    participant_id: UUID | None = None,
    allow_origins: list[str] | None = None,
    allow_origin_regex: str | None = None,
    allow_methods: list[str] | None = None,
    allow_headers: list[str] | None = None,
    allow_credentials: bool = False,
) -> "Server":
    """Build a configured Uvicorn server for Avalan agents."""
    assert (specs_path is None) ^ (
        settings is None
    ), "Provide either specs_path or settings, but not both"

    from os import environ
    from uvicorn import Config, Server

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Initializing app lifespan")
        environ["TOKENIZERS_PARALLELISM"] = "false"
        async with AsyncExitStack() as stack:
            logger.info("Loading OrchestratorLoader in app lifespan")
            pid = participant_id or uuid4()
            loader = OrchestratorLoader(
                hub=hub,
                logger=logger,
                participant_id=pid,
                stack=stack,
            )
            tool_ctx = tool_settings
            ctx = OrchestratorContext(
                participant_id=pid,
                specs_path=specs_path,
                settings=settings,
                tool_settings=tool_ctx,
            )
            app.state.ctx = ctx
            app.state.stack = stack
            app.state.loader = loader
            app.state.logger = logger
            app.state.agent_id = agent_id
            app.state.mcp_resource_store = mcp_router.MCPResourceStore()
            app.state.mcp_resource_base_path = mcp_prefix
            app.state.mcp_tool_name = mcp_name or "run"
            if mcp_description:
                app.state.mcp_tool_description = mcp_description
            yield

    logger.debug("Creating %s server", name)
    app = FastAPI(title=name, version=version, lifespan=lifespan)

    if any(
        [
            allow_origins,
            allow_origin_regex,
            allow_methods,
            allow_headers,
            allow_credentials,
        ]
    ):
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allow_origins or [],
            allow_origin_regex=allow_origin_regex,
            allow_credentials=allow_credentials,
            allow_methods=allow_methods or ["*"],
            allow_headers=allow_headers or ["*"],
        )

    logger.debug("Adding routes to %s server", name)
    chat_router_module = import_module("avalan.server.routers.chat")
    responses_router_module = import_module("avalan.server.routers.responses")
    engine_router_module = import_module("avalan.server.routers.engine")
    app.include_router(chat_router_module.router, prefix=openai_prefix)
    app.include_router(responses_router_module.router, prefix=openai_prefix)
    app.include_router(engine_router_module.router)

    logger.debug("Creating MCP HTTP stream router")
    mcp_http_router = mcp_router.create_router()
    app.include_router(mcp_http_router, prefix=mcp_prefix)

    logger.debug("Starting %s server at %s:%d", name, host, port)
    config = Config(app, host=host, port=port, reload=reload)
    server = Server(config)
    logger_replace(
        logger,
        [
            "uvicorn",
            "uvicorn.error",
            "uvicorn.access",
            "uvicorn.asgi",
            "uvicorn.lifespan",
        ],
    )
    return server


def di_set(app: FastAPI, logger: Logger, orchestrator: Orchestrator) -> None:
    """Store dependencies on the application state."""
    assert logger is not None
    assert orchestrator is not None
    app.state.logger = logger
    app.state.orchestrator = orchestrator


def di_get_logger(request: Request) -> Logger:
    """Retrieve the application logger from the request."""
    assert hasattr(request.app.state, "logger")
    logger = request.app.state.logger
    assert isinstance(logger, Logger)
    return logger


async def di_get_orchestrator(request: Request) -> Orchestrator:
    """Retrieve the orchestrator from the request."""
    if not hasattr(request.app.state, "orchestrator"):
        ctx: OrchestratorContext = request.app.state.ctx
        loader: OrchestratorLoader = request.app.state.loader
        stack: AsyncExitStack = request.app.state.stack
        if ctx.specs_path:
            orchestrator_cm = await loader.from_file(
                ctx.specs_path,
                agent_id=request.app.state.agent_id,
                tool_settings=ctx.tool_settings,
            )
        else:
            assert ctx.settings
            orchestrator_cm = await loader.from_settings(
                ctx.settings,
                tool_settings=ctx.tool_settings,
            )
        orchestrator = await stack.enter_async_context(orchestrator_cm)
        request.app.state.orchestrator = orchestrator
        request.app.state.agent_id = orchestrator.id
    orchestrator = request.app.state.orchestrator
    assert orchestrator is not None
    return orchestrator
