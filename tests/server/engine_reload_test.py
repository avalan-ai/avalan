from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from avalan.server.entities import EngineRequest, OrchestratorContext


class EngineReloadTestCase(IsolatedAsyncioTestCase):
    async def test_reload_from_file(self) -> None:
        import avalan.server.routers.engine as eng

        hub = MagicMock()
        pid = uuid4()
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    ctx=OrchestratorContext(
                        loader=SimpleNamespace(hub=hub, participant_id=pid),
                        hub=hub,
                        participant_id=pid,
                        specs_path="agent.toml",
                    ),
                    stack=SimpleNamespace(aclose=AsyncMock()),
                    agent_id=uuid4(),
                )
            )
        )
        old_stack = request.app.state.stack
        loader = MagicMock()
        orchestrator_cm = MagicMock()
        orchestrator = MagicMock()
        orchestrator.id = uuid4()
        orchestrator_cm.__aenter__ = AsyncMock(return_value=orchestrator)
        loader.from_file = AsyncMock(return_value=orchestrator_cm)
        new_stack = SimpleNamespace(
            enter_async_context=AsyncMock(return_value=orchestrator)
        )
        with (
            patch.object(
                eng, "OrchestratorLoader", return_value=loader
            ) as Loader,
            patch.object(eng, "AsyncExitStack", return_value=new_stack),
            patch.object(eng, "di_set") as di_set,
        ):
            logger = MagicMock()
            await eng.set_engine(request, EngineRequest(uri="new"), logger)
        old_stack.aclose.assert_called_once()
        Loader.assert_called_once()
        di_set.assert_called_once_with(
            request.app, logger=logger, orchestrator=orchestrator
        )
        self.assertIs(request.app.state.stack, new_stack)
        self.assertEqual(request.app.state.agent_id, orchestrator.id)
        self.assertIs(request.app.state.ctx.loader, loader)

    async def test_reload_from_settings(self) -> None:
        import avalan.server.routers.engine as eng

        settings = MagicMock()
        hub = MagicMock()
        pid = uuid4()
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    ctx=OrchestratorContext(
                        loader=SimpleNamespace(hub=hub, participant_id=pid),
                        hub=hub,
                        participant_id=pid,
                        settings=settings,
                        browser_settings=MagicMock(),
                        database_settings=MagicMock(),
                    ),
                    stack=SimpleNamespace(aclose=AsyncMock()),
                )
            )
        )
        old_stack = request.app.state.stack
        loader = MagicMock()
        orchestrator_cm = MagicMock()
        orchestrator = MagicMock()
        orchestrator.id = uuid4()
        orchestrator_cm.__aenter__ = AsyncMock(return_value=orchestrator)
        loader.from_settings = AsyncMock(return_value=orchestrator_cm)
        new_stack = SimpleNamespace(
            enter_async_context=AsyncMock(return_value=orchestrator)
        )
        new_settings = MagicMock()
        with (
            patch.object(eng, "OrchestratorLoader", return_value=loader),
            patch.object(eng, "AsyncExitStack", return_value=new_stack),
            patch.object(eng, "replace", return_value=new_settings) as repl,
            patch.object(eng, "di_set") as di_set,
        ):
            logger = MagicMock()
            await eng.set_engine(request, EngineRequest(uri="new"), logger)
        old_stack.aclose.assert_called_once()
        repl.assert_called_once_with(settings, uri="new")
        di_set.assert_called_once_with(
            request.app, logger=logger, orchestrator=orchestrator
        )
        self.assertIs(request.app.state.stack, new_stack)
        self.assertEqual(request.app.state.agent_id, orchestrator.id)
        self.assertIs(request.app.state.ctx.settings, new_settings)
        self.assertIs(request.app.state.ctx.loader, loader)
