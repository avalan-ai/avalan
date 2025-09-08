from avalan.server import (
    agents_server,
    di_get_logger,
    di_get_orchestrator,
    di_set,
)
from logging import Logger
import sys
from types import ModuleType, SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import MagicMock, patch


class DiHelpersTestCase(TestCase):
    def test_di_set_and_get(self) -> None:
        app = SimpleNamespace(state=SimpleNamespace())
        logger = MagicMock(spec=Logger)
        logger.handlers = []
        logger.level = 0
        logger.propagate = False
        orch = MagicMock()
        di_set(app, logger, orch)
        request = SimpleNamespace(app=app)
        self.assertIs(di_get_logger(request), logger)
        self.assertIs(di_get_orchestrator(request), orch)


class AgentsServerValidationTestCase(TestCase):
    def test_requires_configuration(self) -> None:
        logger = MagicMock(spec=Logger)
        logger.handlers = []
        logger.level = 0
        logger.propagate = False
        with self.assertRaises(AssertionError):
            agents_server(
                hub=MagicMock(),
                name="srv",
                version="v",
                host="h",
                port=1,
                reload=False,
                specs_path=None,
                settings=None,
                browser_settings=None,
                database_settings=None,
                prefix_mcp="/m",
                prefix_openai="/o",
                logger=logger,
            )


class CallToolTestCase(IsolatedAsyncioTestCase):
    async def test_call_tool_handler(self) -> None:
        FastAPI = MagicMock()
        APIRouter = MagicMock()
        fastapi_mod = ModuleType("fastapi")
        fastapi_mod.FastAPI = FastAPI
        fastapi_mod.APIRouter = APIRouter

        class DummyTool:
            def __init__(
                self, name: str, description: str, inputSchema: dict
            ) -> None:
                self.name = name
                self.description = description
                self.inputSchema = inputSchema

        class DummyContent:
            def __init__(self, type: str, text: str) -> None:
                self.type = type
                self.text = text

        types_mod = ModuleType("mcp.types")
        types_mod.EmbeddedResource = object
        types_mod.ImageContent = object
        types_mod.TextContent = DummyContent
        types_mod.Tool = DummyTool

        class DummyServer:
            def __init__(self, *_, **__):
                pass

            def list_tools(self):
                def decorator(func):
                    self.list_func = func
                    return func

                return decorator

            def call_tool(self):
                def decorator(func):
                    self.call_func = func
                    return func

                return decorator

            async def run(self, *_: object) -> None:
                return None

            def create_initialization_options(self) -> dict:
                return {}

        MCPServer = MagicMock(return_value=DummyServer())
        SseServerTransport = MagicMock()
        sse_instance = MagicMock()
        sse_instance.handle_post_message = MagicMock()
        SseServerTransport.return_value = sse_instance
        Config = MagicMock()
        Server = MagicMock()

        chat_module = ModuleType("avalan.server.routers.chat")
        chat_module.router = MagicMock()
        engine_module = ModuleType("avalan.server.routers.engine")
        engine_module.router = MagicMock()
        responses_module = ModuleType("avalan.server.routers.responses")
        responses_module.router = MagicMock()

        modules = {
            "fastapi": fastapi_mod,
            "mcp.server.lowlevel.server": ModuleType(
                "mcp.server.lowlevel.server"
            ),
            "mcp.server.sse": ModuleType("mcp.server.sse"),
            "mcp.types": types_mod,
            "uvicorn": ModuleType("uvicorn"),
            "avalan.server.routers.chat": chat_module,
            "avalan.server.routers.engine": engine_module,
            "avalan.server.routers.responses": responses_module,
        }
        modules["mcp.server.lowlevel.server"].Server = MCPServer
        modules["mcp.server.sse"].SseServerTransport = SseServerTransport
        modules["uvicorn"].Config = Config
        modules["uvicorn"].Server = Server

        with patch.dict(sys.modules, modules):
            logger = MagicMock(spec=Logger)
            logger.handlers = []
            logger.level = 0
            logger.propagate = False
            app_inst = MagicMock()
            app_inst.state = SimpleNamespace()
            FastAPI.return_value = app_inst
            mcp_router = MagicMock()
            mcp_router.get.return_value = lambda f: f
            APIRouter.return_value = mcp_router
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
                    database_settings=None,
                    prefix_mcp="/m",
                    prefix_openai="/o",
                    logger=logger,
                )
            dummy_server: DummyServer = MCPServer.return_value

        tools = await dummy_server.list_func()
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].name, "calculate_sum")

        result = await dummy_server.call_func(
            "calculate_sum", {"a": 1, "b": 2}
        )
        self.assertEqual(result[0].text, "3")

        with self.assertRaises(ValueError):
            await dummy_server.call_func("missing", {})
