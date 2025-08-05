from avalan.server import agents_server
from logging import Logger
import sys
from types import ModuleType
from unittest import TestCase
from unittest.mock import MagicMock, patch


class AgentsServerTestCase(TestCase):
    def test_agents_server_constructs_server(self):
        # Dummy modules and classes
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
        chat_router = MagicMock()
        chat_module.router = chat_router

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
                mcp_router.get.return_value = lambda f: f
                APIRouter.return_value = mcp_router
                sse_instance = MagicMock()
                sse_instance.handle_post_message = MagicMock()
                SseServerTransport.return_value = sse_instance
                mcp_server = MagicMock()
                mcp_server.list_tools.return_value = lambda f: f
                mcp_server.call_tool.return_value = lambda f: f
                MCPServer.return_value = mcp_server
                config_instance = MagicMock()
                Config.return_value = config_instance
                server_instance = MagicMock()
                Server.return_value = server_instance

                with patch("avalan.server.logger_replace") as lr:
                    result = agents_server(
                        hub=MagicMock(),
                        name="srv",
                        version="v",
                        host="h",
                        port=1,
                        reload=False,
                        specs_path=None,
                        settings=None,
                        browser_settings=None,
                        prefix_mcp="/m",
                        prefix_openai="/o",
                        logger=logger,
                    )

        self.assertIs(result, server_instance)
        FastAPI.assert_called_once()
        _, kwargs = FastAPI.call_args
        self.assertEqual(kwargs["title"], "srv")
        self.assertEqual(kwargs["version"], "v")
        self.assertTrue(callable(kwargs["lifespan"]))
        app.include_router.assert_any_call(chat_router, prefix="/o")
        SseServerTransport.assert_called_once_with("/m/messages/")
        app.mount.assert_called_once_with(
            "/m/messages/", app=sse_instance.handle_post_message
        )
        app.include_router.assert_any_call(mcp_router, prefix="/m")
        Config.assert_called_once_with(app, host="h", port=1, reload=False)
        Server.assert_called_once_with(config_instance)
        lr.assert_called_once_with(
            logger,
            [
                "uvicorn",
                "uvicorn.error",
                "uvicorn.access",
                "uvicorn.asgi",
                "uvicorn.lifespan",
            ],
        )
