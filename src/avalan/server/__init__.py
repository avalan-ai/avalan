from ..agent.orchestrator import Orchestrator
from ..utils import logger_replace
from logging import Logger


def agents_server(
    name: str,
    version: str,
    orchestrators: list[Orchestrator],
    host: str,
    port: int,
    reload: bool,
    prefix_mcp: str,
    prefix_openai: str,
    logger: Logger,
):
    from ..server.routers import chat
    from fastapi import FastAPI, APIRouter
    from mcp.server.lowlevel.server import Server as MCPServer
    from mcp.server.sse import SseServerTransport
    from mcp.types import EmbeddedResource, ImageContent, TextContent, Tool
    from starlette.requests import Request
    from uvicorn import Config, Server

    logger.debug("Creating %s server", name)
    app = FastAPI(title=name, version=version)
    app.state.orchestrator = orchestrators[0]

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
