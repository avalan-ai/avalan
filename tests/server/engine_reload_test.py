from avalan.server.entities import EngineRequest, OrchestratorContext
from avalan.tool.context import ToolSettingsContext
from avalan.tool.database import DatabaseToolSettings
from dataclasses import dataclass
from pydantic import ValidationError
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, call, patch
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

    async def test_reload_creates_tool_settings_when_missing(self) -> None:
        import avalan.server.routers.engine as eng

        @dataclass
        class DummySettings:
            uri: str

        settings = DummySettings(uri="old")
        pid = uuid4()
        orchestrator = MagicMock()
        orchestrator.id = uuid4()
        orchestrator_cm = MagicMock()
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
                        tool_settings=None,
                    ),
                    stack=stack,
                    loader=loader,
                )
            )
        )
        with patch.object(eng, "di_set") as di_set:
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
        self.assertIsNotNone(request.app.state.ctx.tool_settings)
        self.assertEqual(
            request.app.state.ctx.tool_settings.database.dsn, "postgres://db"
        )
        self.assertEqual(request.app.state.ctx.settings.uri, "new")
        di_set.assert_called_once_with(
            request.app, logger=logger, orchestrator=orchestrator
        )

    async def test_reload_with_only_database(self) -> None:
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
                request, EngineRequest(database="postgres://db"), logger
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
        self.assertEqual(request.app.state.ctx.settings.uri, "old")

    async def test_reload_removes_existing_orchestrator(self) -> None:
        import avalan.server.routers.engine as eng

        @dataclass
        class DummySettings:
            uri: str

        class TrackingNamespace(SimpleNamespace):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                object.__setattr__(self, "deleted_orchestrator", False)

            def __delattr__(self, name: str) -> None:
                if name == "orchestrator":
                    object.__setattr__(self, "deleted_orchestrator", True)
                super().__delattr__(name)

        settings = DummySettings(uri="old")
        pid = uuid4()
        orchestrator = MagicMock()
        orchestrator.id = uuid4()
        orchestrator_cm = MagicMock()
        loader = SimpleNamespace(
            from_settings=AsyncMock(return_value=orchestrator_cm)
        )
        stack = SimpleNamespace(
            aclose=AsyncMock(),
            enter_async_context=AsyncMock(return_value=orchestrator),
        )
        state = TrackingNamespace(
            ctx=OrchestratorContext(
                participant_id=pid,
                settings=settings,
            ),
            stack=stack,
            loader=loader,
            agent_id=None,
            orchestrator=MagicMock(),
        )
        request = SimpleNamespace(app=SimpleNamespace(state=state))
        with patch.object(eng, "di_set") as di_set:
            logger = MagicMock()
            await eng.set_engine(
                request,
                EngineRequest(uri="new"),
                logger,
            )
        stack.aclose.assert_called_once()
        loader.from_settings.assert_called_once()
        stack.enter_async_context.assert_called_once_with(orchestrator_cm)
        di_set.assert_called_once_with(
            request.app, logger=logger, orchestrator=orchestrator
        )
        self.assertTrue(state.deleted_orchestrator)
        self.assertFalse(hasattr(state, "orchestrator"))
        self.assertEqual(request.app.state.ctx.settings.uri, "new")

    async def test_reload_restores_previous_state_when_loader_fails(self) -> None:
        import avalan.server.routers.engine as eng

        @dataclass
        class DummySettings:
            uri: str

        settings = DummySettings(uri="old")
        pid = uuid4()
        orchestrator = MagicMock()
        orchestrator.id = uuid4()
        orchestrator_cm = MagicMock()
        loader = SimpleNamespace(
            from_settings=AsyncMock(
                side_effect=[Exception("boom"), orchestrator_cm]
            )
        )
        stack = SimpleNamespace(
            aclose=AsyncMock(),
            enter_async_context=AsyncMock(return_value=orchestrator),
        )
        agent_id = uuid4()
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    ctx=OrchestratorContext(
                        participant_id=pid,
                        settings=settings,
                    ),
                    stack=stack,
                    loader=loader,
                    agent_id=agent_id,
                )
            )
        )
        logger = MagicMock()
        with patch.object(eng, "di_set") as di_set:
            with self.assertRaises(Exception):
                await eng.set_engine(request, EngineRequest(uri="new"), logger)
        stack.aclose.assert_called_once()
        self.assertEqual(loader.from_settings.await_count, 2)
        stack.enter_async_context.assert_called_once_with(orchestrator_cm)
        di_set.assert_called_once_with(
            request.app, logger=logger, orchestrator=orchestrator
        )
        self.assertIs(request.app.state.ctx.settings, settings)
        self.assertEqual(request.app.state.agent_id, orchestrator.id)

    async def test_reload_raises_restore_error_when_recovery_fails(self) -> None:
        import avalan.server.routers.engine as eng

        @dataclass
        class DummySettings:
            uri: str

        settings = DummySettings(uri="old")
        pid = uuid4()
        stack = SimpleNamespace(
            aclose=AsyncMock(),
            enter_async_context=AsyncMock(),
        )
        state = SimpleNamespace(
            ctx=OrchestratorContext(
                participant_id=pid,
                settings=settings,
            ),
            stack=stack,
            loader=SimpleNamespace(),
            agent_id=uuid4(),
        )
        request = SimpleNamespace(app=SimpleNamespace(state=state))
        logger = MagicMock()
        primary_error = RuntimeError("primary failure")
        restore_error = RuntimeError("restore failure")
        load_mock = AsyncMock(side_effect=[primary_error, restore_error])
        with (
            patch.object(eng, "_load_orchestrator", load_mock),
            patch.object(eng, "di_set") as di_set,
        ):
            with self.assertRaises(RuntimeError) as exc_info:
                await eng.set_engine(request, EngineRequest(uri="new"), logger)
        stack.aclose.assert_called_once()
        stack.enter_async_context.assert_not_called()
        di_set.assert_not_called()
        self.assertIs(exc_info.exception, restore_error)
        self.assertIs(exc_info.exception.__cause__, primary_error)
        self.assertEqual(load_mock.await_count, 2)
        self.assertIs(state.ctx.settings, settings)
        self.assertEqual(state.agent_id, request.app.state.agent_id)

    async def test_reload_restores_previous_state_when_entering_fails(
        self,
    ) -> None:
        import avalan.server.routers.engine as eng

        @dataclass
        class DummySettings:
            uri: str

        settings = DummySettings(uri="old")
        pid = uuid4()
        orchestrator = MagicMock()
        orchestrator.id = uuid4()
        new_cm = MagicMock()
        restore_cm = MagicMock()
        loader = SimpleNamespace(
            from_settings=AsyncMock(side_effect=[new_cm, restore_cm])
        )
        stack = SimpleNamespace(
            aclose=AsyncMock(),
            enter_async_context=AsyncMock(
                side_effect=[Exception("boom"), orchestrator]
            ),
        )
        agent_id = uuid4()
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    ctx=OrchestratorContext(
                        participant_id=pid,
                        settings=settings,
                    ),
                    stack=stack,
                    loader=loader,
                    agent_id=agent_id,
                )
            )
        )
        logger = MagicMock()
        with patch.object(eng, "di_set") as di_set:
            with self.assertRaises(Exception):
                await eng.set_engine(request, EngineRequest(uri="new"), logger)
        stack.aclose.assert_called_once()
        self.assertEqual(loader.from_settings.await_count, 2)
        self.assertEqual(
            stack.enter_async_context.await_args_list,
            [call(new_cm), call(restore_cm)],
        )
        di_set.assert_called_once_with(
            request.app, logger=logger, orchestrator=orchestrator
        )
        self.assertIs(request.app.state.ctx.settings, settings)
        self.assertEqual(request.app.state.agent_id, orchestrator.id)

    def test_engine_request_requires_uri_or_database(self) -> None:
        with self.assertRaises(ValidationError):
            EngineRequest()
