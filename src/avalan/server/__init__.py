from ..agent.loader import OrchestratorLoader
from ..agent.orchestrator import Orchestrator
from ..entities import OrchestratorSettings
from ..model.hubs.huggingface import HuggingfaceHub
from ..tool.context import ToolSettingsContext
from ..utils import logger_replace
from .entities import OrchestratorContext
from contextlib import AsyncExitStack, asynccontextmanager
from fastapi import APIRouter, FastAPI, Request
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
    prefix_mcp: str,
    prefix_openai: str,
    logger: Logger,
    agent_id: UUID | None = None,
    participant_id: UUID | None = None,
    allow_origins: list[str] | None = None,
    allow_origin_regex: str | None = None,
    allow_methods: list[str] | None = None,
    allow_headers: list[str] | None = None,
    allow_credentials: bool = False,
) -> "Server":
    """Build a configured Uvicorn server for Avalan agents.

    Exactly one of ``specs_path`` or ``settings`` must be provided to
    construct the orchestrator.

    Args:
        hub: Model hub used to load model data.
        name: Human readable server name.
        version: Server version string.
        host: Host address to bind.
        port: Port to listen on.
        reload: Whether Uvicorn should reload on changes.
        specs_path: Optional path to an agent specification file.
        settings: Optional in-memory orchestrator settings.
        tool_settings: Optional tool configuration context.
        prefix_mcp: URL prefix for MCP endpoints.
        prefix_openai: URL prefix for OpenAI-compatible endpoints.
        logger: Application logger.
        agent_id: Optional agent identifier.
        participant_id: Optional participant identifier.
        allow_origins: Optional list of allowed CORS origins.
        allow_origin_regex: Optional regex for allowed CORS origins.
        allow_methods: Optional list of allowed CORS methods.
        allow_headers: Optional list of allowed CORS headers.
        allow_credentials: Whether to allow CORS credentials.

    Returns:
        Configured Uvicorn server instance.
    """
    assert (specs_path is None) ^ (
        settings is None
    ), "Provide either specs_path or settings, but not both"

    from ..server.routers import chat
    from ..server.routers import engine
    from ..server.routers import responses
    from mcp.server.lowlevel.server import Server as MCPServer
    from mcp.server.sse import SseServerTransport
    from mcp.types import EmbeddedResource, ImageContent, TextContent, Tool
    from os import environ
    from starlette.requests import Request
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
            if ctx.specs_path:
                orchestrator_cm = await loader.from_file(
                    ctx.specs_path,
                    agent_id=agent_id or uuid4(),
                    tool_settings=ctx.tool_settings,
                )
            else:
                orchestrator_cm = await loader.from_settings(
                    ctx.settings,
                    tool_settings=ctx.tool_settings,
                )
            orchestrator = await stack.enter_async_context(orchestrator_cm)
            di_set(app, logger=logger, orchestrator=orchestrator)
            app.state.agent_id = orchestrator.id
            logger.info(
                "Agent loaded from %s in app lifespan",
                ctx.specs_path if ctx.specs_path else "inline settings",
            )
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
    app.include_router(chat.router, prefix=prefix_openai)
    app.include_router(responses.router, prefix=prefix_openai)
    app.include_router(engine.router)

    logger.debug("Creating MCP server with SSE")
    mcp_server = MCPServer(name=name)
    sse = SseServerTransport(f"{prefix_mcp}/messages/")
    mcp_router = APIRouter()

    @mcp_router.get("/sse/")
    async def mcp_sse_handler(request: Request) -> None:
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await mcp_server.run(
                streams[0],
                streams[1],
                mcp_server.create_initialization_options(),
            )

    @mcp_server.list_tools()
    async def mcp_list_tools_handler() -> list[Tool]:
        return [
            Tool(
                name="calculate_sum",
                description="Add two numbers together",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "required": ["a", "b"],
                },
            )
        ]

    @mcp_server.call_tool()
    async def call_tool(
        name: str, arguments: dict
    ) -> list[TextContent | ImageContent | EmbeddedResource]:
        if name == "calculate_sum":
            a = arguments["a"]
            b = arguments["b"]
            result = a + b
            return [TextContent(type="text", text=str(result))]
        raise ValueError(f"Tool not found: {name}")

    app.mount(f"{prefix_mcp}/messages/", app=sse.handle_post_message)
    app.include_router(mcp_router, prefix=prefix_mcp)

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


def di_get_orchestrator(request: Request) -> Orchestrator:
    """Retrieve the orchestrator from the request."""
    assert hasattr(request.app.state, "orchestrator")
    orchestrator = request.app.state.orchestrator
    assert orchestrator is not None
    return orchestrator
