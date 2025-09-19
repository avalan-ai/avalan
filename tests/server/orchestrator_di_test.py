from avalan.server import di_get_logger, di_get_orchestrator, di_set
from avalan.server.entities import OrchestratorContext
from logging import Logger
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import patch
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock


class DummyStack:
    def __init__(self) -> None:
        self.enter_async_context = AsyncMock()


class DummyOrchestrator:
    def __init__(self) -> None:
        self.id = uuid4()


class OrchestratorDiTestCase(IsolatedAsyncioTestCase):
    async def test_di_get_orchestrator_from_file(self) -> None:
        orchestrator = DummyOrchestrator()
        loader = MagicMock()
        context_manager = AsyncMock()
        context_manager.__aenter__.return_value = orchestrator
        context_manager.__aexit__.return_value = None
        loader.from_file = AsyncMock(return_value=context_manager)

        ctx = OrchestratorContext(
            participant_id=MagicMock(),
            specs_path="agent.yaml",
            settings=None,
            tool_settings=None,
        )
        stack = DummyStack()
        stack.enter_async_context.return_value = orchestrator

        state = SimpleNamespace(
            ctx=ctx,
            loader=loader,
            stack=stack,
            logger=MagicMock(spec=Logger),
            agent_id=None,
        )
        request = SimpleNamespace(app=SimpleNamespace(state=state))

        with patch("avalan.server.Orchestrator", DummyOrchestrator):
            result = await di_get_orchestrator(request)
        self.assertIs(result, orchestrator)
        self.assertIs(request.app.state.orchestrator, orchestrator)
        loader.from_file.assert_called_once()

    async def test_di_get_orchestrator_from_settings(self) -> None:
        orchestrator = DummyOrchestrator()
        loader = MagicMock()
        context_manager = AsyncMock()
        context_manager.__aenter__.return_value = orchestrator
        context_manager.__aexit__.return_value = None
        loader.from_settings = AsyncMock(return_value=context_manager)

        ctx = OrchestratorContext(
            participant_id=MagicMock(),
            specs_path=None,
            settings=MagicMock(),
            tool_settings=None,
        )
        stack = DummyStack()
        stack.enter_async_context.return_value = orchestrator

        state = SimpleNamespace(
            ctx=ctx,
            loader=loader,
            stack=stack,
            logger=MagicMock(spec=Logger),
            agent_id=None,
        )
        request = SimpleNamespace(app=SimpleNamespace(state=state))

        with patch("avalan.server.Orchestrator", DummyOrchestrator):
            result = await di_get_orchestrator(request)
        self.assertIs(result, orchestrator)
        loader.from_settings.assert_called_once()

    async def test_di_get_logger(self) -> None:
        logger = MagicMock(spec=Logger)
        app_state = SimpleNamespace(logger=logger)
        request = SimpleNamespace(app=SimpleNamespace(state=app_state))
        self.assertIs(di_get_logger(request), logger)

    async def test_di_set_assigns_dependencies(self) -> None:
        app_state = SimpleNamespace()
        app = SimpleNamespace(state=app_state)
        logger = MagicMock(spec=Logger)
        orchestrator = MagicMock()
        di_set(app, logger, orchestrator)
        self.assertIs(app.state.logger, logger)
        self.assertIs(app.state.orchestrator, orchestrator)
