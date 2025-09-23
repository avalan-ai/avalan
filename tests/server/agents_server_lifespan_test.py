import os
import sys
from contextlib import AsyncExitStack
from logging import Logger
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest

from avalan.server import agents_server


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_agents_server_lifespan_initializes_state() -> None:
    logger = MagicMock(spec=Logger)
    loader_instance = MagicMock(name="loader_instance")
    context_instance = MagicMock(name="context_instance")
    resource_store_instance = MagicMock(name="resource_store")
    config_instance = MagicMock(name="config_instance")
    server_instance = MagicMock(name="server_instance")
    generated_participant_id = UUID(int=1)
    agent_identifier = UUID(int=2)
    loader_kwargs: dict[str, object] = {}
    captured_lifespan: dict[str, object] = {}
    hub = MagicMock(name="hub")

    def build_fastapi(*args, **kwargs):
        app = SimpleNamespace(
            state=SimpleNamespace(),
            include_router=MagicMock(),
            add_middleware=MagicMock(),
        )
        captured_lifespan["app"] = app
        captured_lifespan["lifespan"] = kwargs["lifespan"]
        return app

    def build_loader(*args, **kwargs):
        loader_kwargs.update(kwargs)
        return loader_instance

    uvicorn_module = ModuleType("uvicorn")
    config_mock = MagicMock(return_value=config_instance)
    server_mock = MagicMock(return_value=server_instance)
    uvicorn_module.Config = config_mock
    uvicorn_module.Server = server_mock

    with patch.dict(sys.modules, {"uvicorn": uvicorn_module}):
        with (
            patch(
                "avalan.server.FastAPI", side_effect=build_fastapi
            ) as fastapi_mock,
            patch(
                "avalan.server.OrchestratorLoader", side_effect=build_loader
            ) as loader_cls,
            patch(
                "avalan.server.OrchestratorContext",
                return_value=context_instance,
            ) as context_cls,
            patch(
                "avalan.server.mcp_router.MCPResourceStore",
                return_value=resource_store_instance,
            ) as resource_store_cls,
            patch("avalan.server.logger_replace") as logger_replace,
            patch.dict(os.environ, {}, clear=True),
            patch(
                "avalan.server.uuid4", return_value=generated_participant_id
            ) as uuid4_mock,
        ):
            server = agents_server(
                hub=hub,
                name="srv",
                version="v1",
                host="0.0.0.0",
                port=1234,
                reload=False,
                specs_path="agent.yaml",
                settings=None,
                tool_settings="tools",
                mcp_prefix="/mcp",
                openai_prefix="/openai",
                mcp_name="run",
                mcp_description=None,
                logger=logger,
                agent_id=agent_identifier,
                participant_id=None,
            )

            assert server is server_instance
            fastapi_mock.assert_called_once()
            config_mock.assert_called_once()
            server_mock.assert_called_once_with(config_instance)
            logger_replace.assert_called_once()

            lifespan = captured_lifespan["lifespan"]
            app = captured_lifespan["app"]

            async with lifespan(app):
                assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
                assert app.state.ctx is context_instance
                assert app.state.stack is loader_kwargs["stack"]
                assert app.state.loader is loader_instance
                assert app.state.logger is logger
                assert app.state.agent_id == agent_identifier
                assert app.state.mcp_resource_store is resource_store_instance
                assert app.state.mcp_resource_base_path == "/mcp"
                assert app.state.mcp_tool_name == "run"

            loader_cls.assert_called_once()
            context_cls.assert_called_once_with(
                participant_id=generated_participant_id,
                specs_path="agent.yaml",
                settings=None,
                tool_settings="tools",
            )
            resource_store_cls.assert_called_once_with()
            uuid4_mock.assert_called_once_with()

    assert isinstance(loader_kwargs["stack"], AsyncExitStack)
    assert loader_kwargs["hub"] is hub
    assert loader_kwargs["logger"] is logger
    assert loader_kwargs["participant_id"] == generated_participant_id


@pytest.mark.anyio
async def test_agents_server_lifespan_sets_mcp_description() -> None:
    logger = MagicMock(spec=Logger)
    loader_instance = MagicMock(name="loader_instance")
    resource_store_instance = MagicMock(name="resource_store")
    config_instance = MagicMock(name="config_instance")
    server_instance = MagicMock(name="server_instance")
    captured_lifespan: dict[str, object] = {}
    hub = MagicMock(name="hub")

    def build_fastapi(*args, **kwargs):
        app = SimpleNamespace(
            state=SimpleNamespace(),
            include_router=MagicMock(),
            add_middleware=MagicMock(),
        )
        captured_lifespan["app"] = app
        captured_lifespan["lifespan"] = kwargs["lifespan"]
        return app

    uvicorn_module = ModuleType("uvicorn")
    uvicorn_module.Config = MagicMock(return_value=config_instance)
    uvicorn_module.Server = MagicMock(return_value=server_instance)

    with patch.dict(sys.modules, {"uvicorn": uvicorn_module}):
        with (
            patch("avalan.server.FastAPI", side_effect=build_fastapi),
            patch(
                "avalan.server.OrchestratorLoader",
                return_value=loader_instance,
            ),
            patch(
                "avalan.server.mcp_router.MCPResourceStore",
                return_value=resource_store_instance,
            ),
            patch("avalan.server.logger_replace"),
            patch.dict(os.environ, {}, clear=True),
            patch("avalan.server.uuid4", return_value=UUID(int=3)),
        ):
            server = agents_server(
                hub=hub,
                name="srv",
                version="v1",
                host="0.0.0.0",
                port=4321,
                reload=False,
                specs_path="agent.yaml",
                settings=None,
                tool_settings=None,
                mcp_prefix="/mcp",
                openai_prefix="/openai",
                mcp_name="run",
                mcp_description="Describe run tool",
                logger=logger,
                agent_id=None,
                participant_id=None,
            )

            assert server is server_instance

            lifespan = captured_lifespan["lifespan"]
            app = captured_lifespan["app"]

            async with lifespan(app):
                assert app.state.mcp_tool_description == "Describe run tool"


@pytest.mark.anyio
async def test_agents_server_lifespan_sets_a2a_description() -> None:
    logger = MagicMock(spec=Logger)
    loader_instance = MagicMock(name="loader_instance")
    resource_store_instance = MagicMock(name="resource_store")
    config_instance = MagicMock(name="config_instance")
    server_instance = MagicMock(name="server_instance")
    captured_lifespan: dict[str, object] = {}
    hub = MagicMock(name="hub")

    def build_fastapi(*args, **kwargs):
        app = SimpleNamespace(
            state=SimpleNamespace(),
            include_router=MagicMock(),
            add_middleware=MagicMock(),
        )
        captured_lifespan["app"] = app
        captured_lifespan["lifespan"] = kwargs["lifespan"]
        return app

    uvicorn_module = ModuleType("uvicorn")
    uvicorn_module.Config = MagicMock(return_value=config_instance)
    uvicorn_module.Server = MagicMock(return_value=server_instance)

    with patch.dict(sys.modules, {"uvicorn": uvicorn_module}):
        with (
            patch("avalan.server.FastAPI", side_effect=build_fastapi),
            patch(
                "avalan.server.OrchestratorLoader",
                return_value=loader_instance,
            ),
            patch(
                "avalan.server.mcp_router.MCPResourceStore",
                return_value=resource_store_instance,
            ),
            patch("avalan.server.logger_replace"),
            patch.dict(os.environ, {}, clear=True),
            patch("avalan.server.uuid4", return_value=UUID(int=4)),
        ):
            server = agents_server(
                hub=hub,
                name="srv",
                version="v1",
                host="0.0.0.0",
                port=9876,
                reload=False,
                specs_path="agent.yaml",
                settings=None,
                tool_settings=None,
                mcp_prefix="/mcp",
                openai_prefix="/openai",
                mcp_name="run",
                mcp_description=None,
                a2a_tool_name="execute",
                a2a_tool_description="Execute the orchestrated agent",
                logger=logger,
                agent_id=None,
                participant_id=None,
            )

            assert server is server_instance

            lifespan = captured_lifespan["lifespan"]
            app = captured_lifespan["app"]

            async with lifespan(app):
                assert app.state.a2a_tool_name == "execute"
                assert (
                    app.state.a2a_tool_description
                    == "Execute the orchestrated agent"
                )
