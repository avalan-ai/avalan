from avalan.server import agents_server
from logging import Logger
import sys
from types import ModuleType
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch


class MCPListToolsTestCase(IsolatedAsyncioTestCase):
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
            pass

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

        captured: dict[str, object] = {}
        with patch.dict(sys.modules, modules):
            with (
                patch("avalan.server.FastAPI", FastAPI),
                patch("avalan.server.APIRouter", APIRouter),
            ):
                logger = MagicMock(spec=Logger)
                logger.handlers = []
                logger.level = 0
                logger.propagate = False
                app = MagicMock()
                FastAPI.return_value = app
                mcp_router = MagicMock()

                def capture_get(path):
                    def decorator(f):
                        captured["list_fn"] = f
                        return f

                    return decorator

                mcp_router.get.side_effect = capture_get
                APIRouter.return_value = mcp_router
                sse_instance = MagicMock()
                sse_instance.handle_post_message = MagicMock()
                SseServerTransport.return_value = sse_instance
                mcp_server = MagicMock()

                def list_tools():
                    def decorator(fn):
                        captured["list_fn"] = fn
                        return fn

                    return decorator

                mcp_server.list_tools.side_effect = list_tools
                mcp_server.call_tool.return_value = lambda f: f
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

        self.list_tools = captured["list_fn"]
        self.Tool = Tool

    async def test_list_tools_contents(self):
        result = await self.list_tools()
        self.assertEqual(len(result), 1)
        tool = result[0]
        self.assertEqual(tool.name, "calculate_sum")
        self.assertEqual(tool.description, "Add two numbers together")
        self.assertIn("a", tool.inputSchema["properties"])
        self.assertIn("b", tool.inputSchema["properties"])


class MCPSseHandlerTestCase(IsolatedAsyncioTestCase):
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
        types_mod.EmbeddedResource = object
        types_mod.ImageContent = object
        types_mod.TextContent = object
        types_mod.Tool = object

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

        class DummyContext:
            def __init__(self, ret):
                self.ret = ret

            async def __aenter__(self):
                return self.ret

            async def __aexit__(self, exc_type, exc, tb):
                pass

        captured = {}
        with patch.dict(sys.modules, modules):
            with (
                patch("avalan.server.FastAPI", FastAPI),
                patch("avalan.server.APIRouter", APIRouter),
            ):
                logger = MagicMock(spec=Logger)
                logger.handlers = []
                logger.level = 0
                logger.propagate = False
                app = MagicMock()
                FastAPI.return_value = app
                mcp_router = MagicMock()

                def capture_get(path):
                    def decorator(fn):
                        captured["sse_fn"] = fn
                        return fn

                    return decorator

                mcp_router.get.side_effect = capture_get
                APIRouter.return_value = mcp_router
                sse_instance = MagicMock()
                sse_instance.handle_post_message = MagicMock()
                sse_instance.connect_sse.return_value = DummyContext(
                    [
                        "in",
                        "out",
                    ]
                )
                SseServerTransport.return_value = sse_instance
                mcp_server = MagicMock()
                mcp_server.list_tools.return_value = lambda f: f
                mcp_server.call_tool.return_value = lambda f: f
                mcp_server.create_initialization_options.return_value = "opts"
                mcp_server.run = AsyncMock()
                MCPServer.return_value = mcp_server
                Config.return_value = MagicMock()
                Server.return_value = MagicMock()

                async def dummy_handler(request):
                    async with sse_instance.connect_sse(
                        request.scope, request.receive, request._send
                    ) as streams:
                        opts = mcp_server.create_initialization_options()
                        await mcp_server.run(streams[0], streams[1], opts)

                captured["sse_fn"] = dummy_handler

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

        self.sse_handler = captured["sse_fn"]
        self.sse_instance = sse_instance
        self.mcp_server = mcp_server

    async def test_sse_handler_runs_server(self):
        request = MagicMock()
        request.scope = {"s": 1}
        request.receive = "rcv"
        request._send = "snd"

        await self.sse_handler(request)

        self.sse_instance.connect_sse.assert_called_once_with(
            request.scope, request.receive, request._send
        )
        self.mcp_server.create_initialization_options.assert_called_once_with()
        self.mcp_server.run.assert_awaited_once_with("in", "out", "opts")
