from ..agent.loader import OrchestratorLoader
from ..agent.orchestrator import Orchestrator
from ..entities import OrchestratorSettings
from ..model.hubs.huggingface import HuggingfaceHub
from ..tool.browser import BrowserToolSettings
from ..utils import logger_replace
from contextlib import AsyncExitStack, asynccontextmanager
from fastapi import APIRouter, FastAPI, Request
from logging import Logger
from uuid import uuid4


def agents_server(
    hub: HuggingfaceHub,
    name: str,
    version: str,
    host: str,
    port: int,
    reload: bool,
    specs_path: str | None,
    settings: OrchestratorSettings | None,
    browser_settings: BrowserToolSettings | None,
    prefix_mcp: str,
    prefix_openai: str,
    logger: Logger,
):
    from ..server.routers import chat
    from mcp.server.lowlevel.server import Server as MCPServer
    from mcp.server.sse import SseServerTransport
    from mcp.types import EmbeddedResource, ImageContent, TextContent, Tool
    from starlette.requests import Request
    from uvicorn import Config, Server
    from os import environ

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Initializing app lifespan")
        environ["TOKENIZERS_PARALLELISM"] = "false"
        async with AsyncExitStack() as stack:
            logger.info("Loading OrchestratorLoader in app lifespan")
            loader = OrchestratorLoader(
                hub=hub,
                logger=logger,
                participant_id=uuid4(),
                stack=stack,
            )
            if specs_path:
                orchestrator = await loader.from_file(
                    specs_path,
                    agent_id=uuid4(),
                )
            else:
                orchestrator = await loader.from_settings(
                    settings,
                    browser_settings=browser_settings,
                )
            orchestrator = await stack.enter_async_context(orchestrator)
            di_set(app, logger=logger, orchestrator=orchestrator)
            logger.info(
                "Agent loaded from %s in app lifespan",
                specs_path if specs_path else "inline settings",
            )
            yield

    logger.debug("Creating %s server", name)
    app = FastAPI(title=name, version=version, lifespan=lifespan)

    logger.debug("Adding routes to %s server", name)
    app.include_router(chat.router, prefix=prefix_openai)

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
    app.state.logger = logger
    app.state.orchestrator = orchestrator


def di_get_logger(request: Request) -> Logger:
    return request.app.state.logger


def di_get_orchestrator(request: Request) -> Orchestrator:
    return request.app.state.orchestrator
