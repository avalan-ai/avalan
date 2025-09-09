from avalan.server.entities import EngineRequest, OrchestratorContext
from avalan.tool.context import ToolSettingsContext
from avalan.tool.database import DatabaseToolSettings
from dataclasses import dataclass
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
            tool_settings=None,
        )
        stack.enter_async_context.assert_called_once_with(orchestrator_cm)
        di_set.assert_called_once_with(
            request.app, logger=logger, orchestrator=orchestrator
        )
        self.assertEqual(request.app.state.agent_id, orchestrator.id)

    async def test_reload_from_settings(self) -> None:
        import avalan.server.routers.engine as eng

        @dataclass
        class DummySettings:
            uri: str

        settings = DummySettings(uri="old")
        pid = uuid4()
        orchestrator = MagicMock()
        orchestrator.id = uuid4()
        orchestrator_cm = MagicMock()
        browser_settings = MagicMock()
        database_settings = MagicMock()
        tool_settings = ToolSettingsContext(
            browser=browser_settings, database=database_settings
        )
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
                        tool_settings=tool_settings,
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
            await eng.set_engine(
                request,
                EngineRequest(uri="new?max_new_tokens=1"),
                logger,
            )
        stack.aclose.assert_called_once()
        repl.assert_called_once_with(settings, uri="new?max_new_tokens=1")
        loader.from_settings.assert_called_once_with(
            new_settings, tool_settings=tool_settings
        )
        stack.enter_async_context.assert_called_once_with(orchestrator_cm)
        di_set.assert_called_once_with(
            request.app, logger=logger, orchestrator=orchestrator
        )
        self.assertEqual(request.app.state.agent_id, orchestrator.id)
        self.assertIs(request.app.state.ctx.settings, new_settings)

    async def test_reload_updates_database_settings(self) -> None:
        import avalan.server.routers.engine as eng

        @dataclass
        class DummySettings:
            uri: str

        settings = DummySettings(uri="old")
        pid = uuid4()
        orchestrator = MagicMock()
        orchestrator.id = uuid4()
        orchestrator_cm = MagicMock()
        database_settings = DatabaseToolSettings(dsn="old")
        tool_settings = ToolSettingsContext(database=database_settings)
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
                        tool_settings=tool_settings,
                    ),
                    stack=stack,
                    loader=loader,
                )
            )
        )
        with patch.object(eng, "di_set"):
            logger = MagicMock()
            await eng.set_engine(
                request,
                EngineRequest(uri="new", database="postgres://db"),
                logger,
            )
        stack.aclose.assert_called_once()
        loader.from_settings.assert_called_once()
        passed_tool_settings = loader.from_settings.call_args.kwargs[
            "tool_settings"
        ]
        self.assertEqual(passed_tool_settings.database.dsn, "postgres://db")
        self.assertEqual(
            request.app.state.ctx.tool_settings.database.dsn, "postgres://db"
        )
        self.assertEqual(request.app.state.ctx.settings.uri, "new")
