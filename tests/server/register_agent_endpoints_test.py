import os
import sys
from contextlib import asynccontextmanager
from logging import Logger
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest

from avalan.server import register_agent_endpoints

MODULE = register_agent_endpoints.__module__
SERVER_MODULE = sys.modules[MODULE]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_register_agent_endpoints_wraps_existing_lifespan() -> None:
    logger = MagicMock(spec=Logger)
    resource_store_instance = MagicMock(name="resource_store")
    hub = MagicMock(name="hub")
    generated_participant_id = UUID(int=5)
    existing_events: list[str] = []

    @asynccontextmanager
    async def existing_lifespan(app):
        existing_events.append("enter")
        yield
        existing_events.append("exit")

    app = SimpleNamespace(
        state=SimpleNamespace(),
        include_router=MagicMock(),
        add_middleware=MagicMock(),
        router=SimpleNamespace(lifespan_context=existing_lifespan),
    )

    loader_instance = MagicMock(name="loader_instance")

    with (
        patch.object(
            SERVER_MODULE.mcp_router,
            "MCPResourceStore",
            return_value=resource_store_instance,
        ) as resource_store_cls,
        patch.object(
            SERVER_MODULE, "uuid4", return_value=generated_participant_id
        ),
        patch.object(
            SERVER_MODULE, "OrchestratorLoader", return_value=loader_instance
        ) as loader_cls,
        patch.dict(os.environ, {}, clear=True),
    ):
        register_agent_endpoints(
            app,
            hub=hub,
            logger=logger,
            specs_path="agent.yaml",
            settings=None,
            tool_settings="tools",
            mcp_prefix="/mcp",
            openai_prefix="/openai",
            mcp_name="run",
        )

        assert app.router.lifespan_context is not None

        async with app.router.lifespan_context(app):
            assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
            ctx = app.state.ctx
            assert ctx.specs_path == "agent.yaml"
            assert ctx.settings is None
            assert ctx.tool_settings == "tools"
            loader = app.state.loader
            assert loader is loader_instance
            loader_cls.assert_called_once()
            assert loader_cls.call_args.kwargs["hub"] is hub
            assert loader_cls.call_args.kwargs["logger"] is logger
            assert (
                loader_cls.call_args.kwargs["participant_id"]
                == generated_participant_id
            )
            assert loader_cls.call_args.kwargs["stack"] is app.state.stack
            assert app.state.logger is logger
            assert app.state.agent_id is None
            assert app.state.mcp_resource_store is resource_store_instance
            assert app.state.mcp_resource_base_path == "/mcp"
            assert app.state.mcp_tool_name == "run"

    assert existing_events == ["enter", "exit"]
    resource_store_cls.assert_called_once_with()


def test_register_agent_endpoints_normalizes_protocols() -> None:
    logger = MagicMock(spec=Logger)
    hub = MagicMock(name="hub")

    app = SimpleNamespace(
        state=SimpleNamespace(),
        include_router=MagicMock(),
        add_middleware=MagicMock(),
        router=SimpleNamespace(lifespan_context=None),
    )

    modules: dict[str, object] = {}

    def import_module(name: str):
        module = SimpleNamespace(router=f"router:{name}")
        modules[name] = module
        return module

    sys.modules.pop("avalan.server.routers.chat", None)
    sys.modules.pop("avalan.server.routers.responses", None)
    sys.modules.pop("avalan.server.routers.engine", None)

    with (
        patch.object(SERVER_MODULE.mcp_router, "MCPResourceStore"),
        patch.object(
            SERVER_MODULE.mcp_router, "create_router"
        ) as create_router,
        patch.object(SERVER_MODULE, "OrchestratorLoader"),
        patch.object(
            SERVER_MODULE, "import_module", side_effect=import_module
        ) as importer,
    ):
        register_agent_endpoints(
            app,
            hub=hub,
            logger=logger,
            specs_path=None,
            settings="settings",
            tool_settings=None,
            mcp_prefix="/mcp",
            openai_prefix="/openai",
            mcp_name="run",
            protocols={"openai": {"RESPONSES"}, "mcp": set()},
        )

        called_module_names = [
            call.args[0] for call in importer.call_args_list
        ]
        assert "avalan.server.routers.responses" in called_module_names
        assert "avalan.server.routers.engine" in called_module_names
        assert "avalan.server.routers.chat" not in called_module_names
        create_router.assert_called_once_with()
        app.include_router.assert_any_call(
            modules["avalan.server.routers.engine"].router
        )
        app.include_router.assert_any_call(
            modules["avalan.server.routers.responses"].router, prefix="/openai"
        )
