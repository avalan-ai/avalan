from avalan.server import (
    agents_server,
    di_get_logger,
    di_get_orchestrator,
    di_set,
)
from contextlib import asynccontextmanager
from logging import Logger
import asyncio
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
        self.assertIs(asyncio.run(di_get_orchestrator(request)), orch)


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
                tool_settings=None,
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
                    tool_settings=None,
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


class SseHandlerCoverageTestCase(IsolatedAsyncioTestCase):
    async def test_sse_handler_streams(self) -> None:
        for name in list(sys.modules):
            if name == "avalan.server" or name.startswith("avalan.server."):
                sys.modules.pop(name)

        import importlib

        server_mod = importlib.import_module("avalan.server")

        created: dict[str, object] = {}

        class StubApp:
            def __init__(self, *_, **__):
                self.state = SimpleNamespace()
                self.include_calls: list[tuple[object, str]] = []
                self.mount_calls: list[tuple[str, object]] = []
                self.middleware_calls: list[tuple[tuple[object, ...], dict]] = []

            def include_router(self, router: object, prefix: str = "") -> None:
                self.include_calls.append((router, prefix))

            def mount(self, path: str, app: object) -> None:
                self.mount_calls.append((path, app))

            def add_middleware(self, *args: object, **kwargs: object) -> None:
                self.middleware_calls.append((args, kwargs))

        class StubRouter:
            def __init__(self) -> None:
                self.routes: dict[str, object] = {}

            def get(self, path: str):
                def decorator(func):
                    self.routes[path] = func
                    return func

                return decorator

        class DummyMCPServer:
            def __init__(self, *_: object, **__: object) -> None:
                self.run_calls: list[tuple[object, object, object]] = []
                self.list_handler = None
                self.call_handler = None
                created["mcp"] = self

            def list_tools(self):
                def decorator(func):
                    self.list_handler = func
                    return func

                return decorator

            def call_tool(self):
                def decorator(func):
                    self.call_handler = func
                    return func

                return decorator

            def create_initialization_options(self) -> dict[str, str]:
                return {"value": "init"}

            async def run(
                self, stream_in: object, stream_out: object, options: object
            ) -> None:
                self.run_calls.append((stream_in, stream_out, options))

        class DummySSETransport:
            def __init__(self, path: str) -> None:
                self.path = path
                self.handle_post_message = object()
                self.calls: list[tuple[object, object, object]] = []
                created["sse"] = self

            @asynccontextmanager
            async def connect_sse(
                self, scope: object, receive: object, send: object
            ):
                self.calls.append((scope, receive, send))
                yield ("incoming", "outgoing")

        class DummyConfig:
            def __init__(self, app: object, host: str, port: int, reload: bool) -> None:
                self.app = app
                self.host = host
                self.port = port
                self.reload = reload

        class DummyServer:
            def __init__(self, config: DummyConfig) -> None:
                self.config = config

        class DummyTool:
            def __init__(self, **kwargs: object) -> None:
                self.__dict__.update(kwargs)

        class DummyTextContent:
            def __init__(self, **kwargs: object) -> None:
                self.__dict__.update(kwargs)

        class DummyEmbeddedResource:
            pass

        class DummyImageContent:
            pass

        modules = {
            "mcp.server.lowlevel.server": ModuleType("mcp.server.lowlevel.server"),
            "mcp.server.sse": ModuleType("mcp.server.sse"),
            "mcp.types": ModuleType("mcp.types"),
            "uvicorn": ModuleType("uvicorn"),
            "starlette.requests": ModuleType("starlette.requests"),
        }
        modules["mcp.server.lowlevel.server"].Server = DummyMCPServer
        modules["mcp.server.sse"].SseServerTransport = DummySSETransport
        modules["mcp.types"].Tool = DummyTool
        modules["mcp.types"].TextContent = DummyTextContent
        modules["mcp.types"].EmbeddedResource = DummyEmbeddedResource
        modules["mcp.types"].ImageContent = DummyImageContent
        modules["uvicorn"].Config = DummyConfig
        modules["uvicorn"].Server = DummyServer
        modules["starlette.requests"].Request = object

        logger_calls: list[tuple[object, list[str]]] = []

        with patch.dict(sys.modules, modules):
            with (
                patch.object(server_mod, "FastAPI", StubApp),
                patch.object(server_mod, "APIRouter", StubRouter),
                patch.object(
                    server_mod,
                    "logger_replace",
                    lambda logger, names: logger_calls.append((logger, names)),
                ),
            ):
                logger = MagicMock(spec=Logger)
                logger.handlers = []
                logger.level = 0
                logger.propagate = False

                server = server_mod.agents_server(
                    hub=MagicMock(),
                    name="srv",
                    version="v",
                    host="h",
                    port=1,
                    reload=False,
                    specs_path=None,
                    settings=SimpleNamespace(),
                    tool_settings=None,
                    prefix_mcp="/m",
                    prefix_openai="/o",
                    logger=logger,
                )

        app: StubApp = server.config.app  # type: ignore[assignment]
        mcp_router = next(
            router for router, prefix in app.include_calls if prefix == "/m"
        )
        handler = mcp_router.routes["/sse/"]
        request = SimpleNamespace(scope={"type": "http"}, receive="rcv", _send="snd")

        await handler(request)

        sse: DummySSETransport = created["sse"]  # type: ignore[assignment]
        self.assertEqual(sse.calls, [(request.scope, request.receive, request._send)])

        mcp: DummyMCPServer = created["mcp"]  # type: ignore[assignment]
        self.assertEqual(
            mcp.run_calls,
            [("incoming", "outgoing", {"value": "init"})],
        )
        self.assertTrue(logger_calls)
