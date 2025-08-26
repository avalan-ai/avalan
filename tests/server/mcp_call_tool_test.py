from avalan.server import agents_server
from logging import Logger
import sys
from types import ModuleType
from unittest import IsolatedAsyncioTestCase
from unittest.mock import MagicMock, patch


class MCPCallToolTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)

        FastAPI = MagicMock()
        APIRouter = MagicMock()
        fastapi_mod = ModuleType("fastapi")
        fastapi_mod.FastAPI = FastAPI
        fastapi_mod.APIRouter = APIRouter

        MCPServer = MagicMock()
        SseServerTransport = MagicMock()
        Config = MagicMock()
        Server = MagicMock()

        mcp_server_mod = ModuleType("mcp.server.lowlevel.server")
        mcp_server_mod.Server = MCPServer
        sse_mod = ModuleType("mcp.server.sse")
        sse_mod.SseServerTransport = SseServerTransport
        types_mod = ModuleType("mcp.types")

        class TextContent:
            def __init__(self, type=None, text=None):
                self.type = type
                self.text = text

            def __eq__(self, other):
                return (
                    isinstance(other, TextContent)
                    and self.type == other.type
                    and self.text == other.text
                )

        class EmbeddedResource:
            pass

        class ImageContent:
            pass

        class Tool:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        types_mod.EmbeddedResource = EmbeddedResource
        types_mod.ImageContent = ImageContent
        types_mod.TextContent = TextContent
        types_mod.Tool = Tool

        uvicorn_mod = ModuleType("uvicorn")
        uvicorn_mod.Config = Config
        uvicorn_mod.Server = Server

        starlette_requests_mod = ModuleType("starlette.requests")
        starlette_requests_mod.Request = object

        chat_module = ModuleType("avalan.server.routers.chat")
        chat_module.router = MagicMock()

        mcp_mod = ModuleType("mcp")
        server_pkg = ModuleType("mcp.server")
        lowlevel_pkg = ModuleType("mcp.server.lowlevel")
        mcp_mod.server = server_pkg
        server_pkg.lowlevel = lowlevel_pkg
        lowlevel_pkg.server = mcp_server_mod
        server_pkg.sse = sse_mod

        modules = {
            "fastapi": fastapi_mod,
            "mcp": mcp_mod,
            "mcp.server": server_pkg,
            "mcp.server.lowlevel": lowlevel_pkg,
            "mcp.server.lowlevel.server": mcp_server_mod,
            "mcp.server.sse": sse_mod,
            "mcp.types": types_mod,
            "uvicorn": uvicorn_mod,
            "starlette.requests": starlette_requests_mod,
            "avalan.server.routers.chat": chat_module,
        }

        captured = {}
        with patch.dict(sys.modules, modules):
            logger = MagicMock(spec=Logger)
            logger.handlers = []
            logger.level = 0
            logger.propagate = False
            app = MagicMock()
            FastAPI.return_value = app
            mcp_router = MagicMock()
            mcp_router.get.return_value = lambda f: f
            APIRouter.return_value = mcp_router
            sse_instance = MagicMock()
            sse_instance.handle_post_message = MagicMock()
            SseServerTransport.return_value = sse_instance
            mcp_server = MagicMock()
            mcp_server.list_tools.return_value = lambda f: f

            def call_tool():
                def decorator(func):
                    captured["fn"] = func
                    return func

                return decorator

            mcp_server.call_tool.side_effect = call_tool
            MCPServer.return_value = mcp_server
            Config.return_value = MagicMock()
            Server.return_value = MagicMock()

            with patch("avalan.server.logger_replace"):
                  agents_server(
                      hub=MagicMock(),
                      name="srv",
                      version="v",
                      host="h",
                      port=1,
                      reload=False,
                      specs_path=None,
                      settings=MagicMock(),
                      browser_settings=None,
                      prefix_mcp="/m",
                      prefix_openai="/o",
                      logger=logger,
                  )

        self.call_tool = captured["fn"]
        self.TextContent = TextContent

    async def test_calculate_sum_tool(self):
        result = await self.call_tool("calculate_sum", {"a": 2, "b": 3})
        self.assertEqual(result, [self.TextContent(type="text", text="5")])

    async def test_missing_tool_raises(self):
        with self.assertRaises(ValueError) as ctx:
            await self.call_tool("missing", {})
        self.assertEqual(str(ctx.exception), "Tool not found: missing")
