from avalan.server import agents_server
from logging import Logger
import sys
from types import ModuleType, SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch


def make_modules():
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

    chat_mod = ModuleType("avalan.server.routers.chat")
    chat_mod.router = MagicMock()

    modules = {
        "fastapi": fastapi_mod,
        "mcp.server.lowlevel.server": mcp_server_mod,
        "mcp.server.sse": sse_mod,
        "mcp.types": types_mod,
        "uvicorn": uvicorn_mod,
        "starlette.requests": starlette_requests_mod,
        "avalan.server.routers.chat": chat_mod,
    }

    return (
        modules,
        FastAPI,
        APIRouter,
        MCPServer,
        SseServerTransport,
        Config,
        Server,
    )


class AgentsServerLifespanTestCase(IsolatedAsyncioTestCase):
    async def test_lifespan_builds_orchestrator_from_file(self) -> None:
        (
            modules,
            FastAPI,
            APIRouter,
            MCPServer,
            SseServerTransport,
            Config,
            Server,
        ) = make_modules()

        with patch.dict(sys.modules, modules):
            with (
                patch("avalan.server.FastAPI", FastAPI),
                patch("avalan.server.APIRouter", APIRouter),
                patch("avalan.server.OrchestratorLoader") as Loader,
            ):
                loader = MagicMock()
                Loader.return_value = loader

                orchestrator = MagicMock()
                orchestrator_cm = MagicMock()
                orchestrator_cm.__aenter__ = AsyncMock(
                    return_value=orchestrator
                )
                orchestrator_cm.__aexit__ = AsyncMock(return_value=False)
                loader.from_file = AsyncMock(return_value=orchestrator_cm)
                loader.from_settings = AsyncMock()

                logger = MagicMock(spec=Logger)
                logger.handlers = []
                logger.level = 0
                logger.propagate = False
                app = MagicMock()
                app.state = SimpleNamespace()
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
                        specs_path="path.json",
                        settings=None,
                        browser_settings=None,
                        prefix_mcp="/m",
                        prefix_openai="/o",
                        logger=logger,
                    )

                lifespan = FastAPI.call_args.kwargs["lifespan"]

                self.assertFalse(hasattr(app.state, "orchestrator"))

                async with lifespan(app):
                    pass

                loader.from_file.assert_awaited_once()
                loader.from_settings.assert_not_called()
                args, kwargs = loader.from_file.await_args
                self.assertEqual(args[0], "path.json")
                self.assertIn("agent_id", kwargs)
                _, loader_kwargs = Loader.call_args
                self.assertIn("participant_id", loader_kwargs)
                orchestrator_cm.__aenter__.assert_awaited_once()
                orchestrator_cm.__aexit__.assert_awaited_once()
                self.assertIs(app.state.orchestrator, orchestrator)
                self.assertIs(app.state.logger, logger)

    async def test_lifespan_builds_orchestrator_from_settings(self) -> None:
        (
            modules,
            FastAPI,
            APIRouter,
            MCPServer,
            SseServerTransport,
            Config,
            Server,
        ) = make_modules()

        with patch.dict(sys.modules, modules):
            with (
                patch("avalan.server.FastAPI", FastAPI),
                patch("avalan.server.APIRouter", APIRouter),
                patch("avalan.server.OrchestratorLoader") as Loader,
            ):
                loader = MagicMock()
                Loader.return_value = loader

                orchestrator = MagicMock()
                orchestrator_cm = MagicMock()
                orchestrator_cm.__aenter__ = AsyncMock(
                    return_value=orchestrator
                )
                orchestrator_cm.__aexit__ = AsyncMock(return_value=False)
                loader.from_settings = AsyncMock(return_value=orchestrator_cm)
                loader.from_file = AsyncMock()

                logger = MagicMock(spec=Logger)
                logger.handlers = []
                logger.level = 0
                logger.propagate = False
                settings = MagicMock()
                browser_settings = MagicMock()
                app = MagicMock()
                app.state = SimpleNamespace()
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
                        settings=settings,
                        browser_settings=browser_settings,
                        prefix_mcp="/m",
                        prefix_openai="/o",
                        logger=logger,
                    )

                lifespan = FastAPI.call_args.kwargs["lifespan"]

                self.assertFalse(hasattr(app.state, "orchestrator"))

                async with lifespan(app):
                    pass

                loader.from_settings.assert_awaited_once()
                loader.from_file.assert_not_called()
                orchestrator_cm.__aenter__.assert_awaited_once()
                orchestrator_cm.__aexit__.assert_awaited_once()
                self.assertIs(app.state.orchestrator, orchestrator)
                self.assertIs(app.state.logger, logger)
                args, kwargs = loader.from_settings.await_args
                self.assertEqual(args[0], settings)
                self.assertEqual(kwargs["browser_settings"], browser_settings)
                _, loader_kwargs = Loader.call_args
                self.assertIn("participant_id", loader_kwargs)
