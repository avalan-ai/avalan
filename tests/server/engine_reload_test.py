from avalan.server.entities import EngineRequest, OrchestratorContext
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4


class EngineReloadTestCase(IsolatedAsyncioTestCase):
    async def test_reload_from_file(self) -> None:
        import avalan.server.routers.engine as eng

        pid = uuid4()
        orchestrator = MagicMock()
        orchestrator.id = uuid4()
        orchestrator_cm = MagicMock()
        loader = SimpleNamespace(
            from_file=AsyncMock(return_value=orchestrator_cm)
        )
        stack = SimpleNamespace(
            aclose=AsyncMock(),
            enter_async_context=AsyncMock(return_value=orchestrator),
        )
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    ctx=OrchestratorContext(
                        participant_id=pid,
                        specs_path="agent.toml",
                    ),
                    stack=stack,
                    loader=loader,
                    agent_id=uuid4(),
                )
            )
        )
        with patch.object(eng, "di_set") as di_set:
            logger = MagicMock()
            original_agent_id = request.app.state.agent_id
            await eng.set_engine(request, EngineRequest(uri="new"), logger)
        stack.aclose.assert_called_once()
        loader.from_file.assert_called_once_with(
            "agent.toml",
            agent_id=original_agent_id,
            uri="new",
        )
        stack.enter_async_context.assert_called_once_with(orchestrator_cm)
        di_set.assert_called_once_with(
            request.app, logger=logger, orchestrator=orchestrator
        )
        self.assertEqual(request.app.state.agent_id, orchestrator.id)

    async def test_reload_from_settings(self) -> None:
        import avalan.server.routers.engine as eng

        settings = MagicMock()
        pid = uuid4()
        orchestrator = MagicMock()
        orchestrator.id = uuid4()
        orchestrator_cm = MagicMock()
        browser_settings = MagicMock()
        database_settings = MagicMock()
        loader = SimpleNamespace(
            from_settings=AsyncMock(return_value=orchestrator_cm)
        )
        stack = SimpleNamespace(
            aclose=AsyncMock(),
            enter_async_context=AsyncMock(return_value=orchestrator),
        )
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    ctx=OrchestratorContext(
                        participant_id=pid,
                        settings=settings,
                        browser_settings=browser_settings,
                        database_settings=database_settings,
                    ),
                    stack=stack,
                    loader=loader,
                )
            )
        )
        new_settings = MagicMock()
        with (
            patch.object(eng, "replace", return_value=new_settings) as repl,
            patch.object(eng, "di_set") as di_set,
        ):
            logger = MagicMock()
            await eng.set_engine(request, EngineRequest(uri="new"), logger)
        stack.aclose.assert_called_once()
        repl.assert_called_once_with(settings, uri="new")
        loader.from_settings.assert_called_once_with(
            new_settings,
            browser_settings=browser_settings,
            database_settings=database_settings,
        )
        stack.enter_async_context.assert_called_once_with(orchestrator_cm)
        di_set.assert_called_once_with(
            request.app, logger=logger, orchestrator=orchestrator
        )
        self.assertEqual(request.app.state.agent_id, orchestrator.id)
        self.assertIs(request.app.state.ctx.settings, new_settings)
