from avalan.server import agents_server
from logging import Logger
import sys
from types import ModuleType
from unittest import TestCase
from unittest.mock import MagicMock, patch


class AgentsServerHttpTestCase(TestCase):
    def setUp(self) -> None:
        self.addCleanup(patch.stopall)

    def _install_modules(self) -> dict[str, ModuleType]:
        FastAPI = MagicMock()
        fastapi_mod = ModuleType("fastapi")
        fastapi_mod.FastAPI = FastAPI
        fastapi_mod.HTTPException = Exception

        uvicorn_mod = ModuleType("uvicorn")
        uvicorn_mod.Config = MagicMock()
        uvicorn_mod.Server = MagicMock()

        chat_mod = ModuleType("avalan.server.routers.chat")
        chat_mod.router = MagicMock(name="chat_router")
        engine_mod = ModuleType("avalan.server.routers.engine")
        engine_mod.router = MagicMock(name="engine_router")
        responses_mod = ModuleType("avalan.server.routers.responses")
        responses_mod.router = MagicMock(name="responses_router")

        modules = {
            "fastapi": fastapi_mod,
            "uvicorn": uvicorn_mod,
            "avalan.server.routers.chat": chat_mod,
            "avalan.server.routers.engine": engine_mod,
            "avalan.server.routers.responses": responses_mod,
        }

        self.FastAPI = FastAPI
        self.Config = uvicorn_mod.Config
        self.Server = uvicorn_mod.Server
        self.chat_router = chat_mod.router
        self.engine_router = engine_mod.router
        self.responses_router = responses_mod.router

        return modules

    def test_agents_server_constructs_server(self) -> None:
        modules = self._install_modules()

        logger = MagicMock(spec=Logger)
        logger.handlers = []
        logger.level = 0
        logger.propagate = False

        app = MagicMock()
        app.include_router = MagicMock()
        app.add_middleware = MagicMock()
        self.FastAPI.return_value = app

        mcp_router = MagicMock(name="mcp_router")

        config_instance = MagicMock()
        server_instance = MagicMock()
        self.Config.return_value = config_instance
        self.Server.return_value = server_instance

        with patch.dict(sys.modules, modules):
            with (
                patch("avalan.server.FastAPI", self.FastAPI),
                patch("avalan.server.mcp_router.create_router", return_value=mcp_router) as create_router,
                patch("avalan.server.logger_replace"),
            ):
                result = agents_server(
                    hub=MagicMock(),
                    name="srv",
                    version="v",
                    host="h",
                    port=8080,
                    reload=False,
                    specs_path=None,
                    settings=MagicMock(),
                    tool_settings=None,
                    prefix_mcp="/m",
                    prefix_openai="/o",
                    logger=logger,
                )

        self.assertIs(result, self.Server.return_value)
        self.FastAPI.assert_called_once()
        create_router.assert_called_once_with()
        app.include_router.assert_any_call(self.chat_router, prefix="/o")
        app.include_router.assert_any_call(self.responses_router, prefix="/o")
        app.include_router.assert_any_call(self.engine_router)
        app.include_router.assert_any_call(mcp_router, prefix="/m")
        self.Config.assert_called_once_with(app, host="h", port=8080, reload=False)
        self.Server.assert_called_once_with(config_instance)

    def test_agents_server_cors_options(self) -> None:
        modules = self._install_modules()

        logger = MagicMock(spec=Logger)
        logger.handlers = []
        logger.level = 0
        logger.propagate = False

        app = MagicMock()
        app.include_router = MagicMock()
        app.add_middleware = MagicMock()
        self.FastAPI.return_value = app

        with patch.dict(sys.modules, modules):
            with (
                patch("avalan.server.FastAPI", self.FastAPI),
                patch("avalan.server.mcp_router.create_router", return_value=MagicMock()),
                patch("avalan.server.logger_replace"),
            ):
                agents_server(
                    hub=MagicMock(),
                    name="srv",
                    version="v",
                    host="h",
                    port=1,
                    reload=False,
                    specs_path=None,
                    settings=MagicMock(),
                    tool_settings=None,
                    prefix_mcp="/m",
                    prefix_openai="/o",
                    logger=logger,
                    allow_origins=["*"],
                    allow_methods=["GET"],
                    allow_headers=["X"],
                    allow_credentials=True,
                )

        app.add_middleware.assert_called_once()
