from avalan.agent.loader import OrchestratorLoader
from avalan.entities import OrchestratorSettings, ToolFormat
from avalan.event import Event, EventType
from avalan.model.hubs.huggingface import HuggingfaceHub
from contextlib import AsyncExitStack
from logging import DEBUG, INFO, Logger
from os import chmod, geteuid
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any, Callable
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from avalan.tool.browser import BrowserToolSettings
from avalan.tool.context import ToolSettingsContext
from avalan.tool.database import DatabaseToolSettings


class LoaderPropertyTestCase(IsolatedAsyncioTestCase):
    async def test_hub_and_participant_id(self):
        stack = AsyncExitStack()
        hub = MagicMock(spec=HuggingfaceHub)
        logger = MagicMock(spec=Logger)
        participant_id = uuid4()
        loader = OrchestratorLoader(
            hub=hub, logger=logger, participant_id=participant_id, stack=stack
        )
        self.assertIs(loader.hub, hub)
        self.assertEqual(loader.participant_id, participant_id)
        await stack.aclose()


class LoaderFromFileTestCase(IsolatedAsyncioTestCase):
    async def test_file_not_found(self):
        stack = AsyncExitStack()
        loader = OrchestratorLoader(
            hub=MagicMock(spec=HuggingfaceHub),
            logger=MagicMock(spec=Logger),
            participant_id=uuid4(),
            stack=stack,
        )
        with self.assertRaises(FileNotFoundError):
            await loader.from_file("missing.toml", agent_id=uuid4())
        await stack.aclose()

    async def test_permission_error(self):
        if geteuid() == 0:
            self.skipTest("Running as root; permission error won't occur")
        with NamedTemporaryFile(delete=False) as tmp:
            path = tmp.name
        chmod(path, 0)
        stack = AsyncExitStack()
        try:
            loader = OrchestratorLoader(
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=stack,
            )
            with self.assertRaises(PermissionError):
                await loader.from_file(path, agent_id=uuid4())
        finally:
            chmod(path, 0o644)
        await stack.aclose()

    async def test_permission_error_when_access_denied(self):
        with NamedTemporaryFile() as tmp:
            path = tmp.name
        stack = AsyncExitStack()
        with (
            patch("avalan.agent.loader.exists", return_value=True),
            patch("avalan.agent.loader.access", return_value=False),
        ):
            loader = OrchestratorLoader(
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=stack,
            )
            with self.assertRaises(PermissionError):
                await loader.from_file(path, agent_id=uuid4())
        await stack.aclose()

    async def test_load_default_orchestrator(self):
        config = """
[agent]
role = \"assistant\"
task = \"do\"
instructions = \"how\"

[engine]
uri = \"ai://local/model\"
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            hub = MagicMock(spec=HuggingfaceHub)
            logger = MagicMock(spec=Logger)
            stack = AsyncExitStack()

            sentence_model = MagicMock()
            sentence_model.__enter__.return_value = sentence_model

            model_manager = MagicMock()
            model_manager.__enter__.return_value = model_manager
            model_manager.parse_uri.return_value = "uri_obj"
            model_manager.get_engine_settings.return_value = "settings_obj"

            memory = MagicMock()
            tool = MagicMock()
            browser_toolset = MagicMock()
            event_manager = MagicMock()

            with (
                patch(
                    "avalan.agent.loader.SentenceTransformerModel",
                    return_value=sentence_model,
                ),
                patch("avalan.agent.loader.TextPartitioner"),
                patch(
                    "avalan.agent.loader.MemoryManager.create_instance",
                    new=AsyncMock(return_value=memory),
                ) as mm_patch,
                patch(
                    "avalan.agent.loader.ModelManager",
                    return_value=model_manager,
                ) as model_patch,
                patch(
                    "avalan.agent.loader.DefaultOrchestrator",
                    return_value="orch",
                ) as orch_patch,
                patch(
                    "avalan.agent.loader.ToolManager.create_instance",
                    return_value=tool,
                ),
                patch(
                    "avalan.agent.loader.BrowserToolSet",
                    return_value=browser_toolset,
                ) as bts_patch,
                patch(
                    "avalan.agent.loader.EventManager",
                    return_value=event_manager,
                ),
            ):
                loader = OrchestratorLoader(
                    hub=hub,
                    logger=logger,
                    participant_id=uuid4(),
                    stack=stack,
                )
                result = await loader.from_file(
                    path,
                    agent_id=uuid4(),
                    disable_memory=True,
                )

                self.assertEqual(result, "orch")
                orch_patch.assert_called_once()
                model_patch.assert_called_once_with(
                    hub, logger, event_manager=event_manager
                )
                mm_patch.assert_awaited_once()
                bts = bts_patch.call_args.kwargs["settings"]
                self.assertIsInstance(bts, BrowserToolSettings)
                self.assertEqual(bts.engine, "firefox")
            await stack.aclose()

    async def test_load_tool_settings(self):
        config = """
[agent]
role = \"assistant\"

[engine]
uri = \"ai://local/model\"

[tool.browser.open]
engine = \"webkit\"
debug = true
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            hub = MagicMock(spec=HuggingfaceHub)
            logger = MagicMock(spec=Logger)
            stack = AsyncExitStack()

            sentence_model = MagicMock()
            sentence_model.__enter__.return_value = sentence_model

            model_manager = MagicMock()
            model_manager.__enter__.return_value = model_manager
            model_manager.parse_uri.return_value = "uri_obj"
            model_manager.get_engine_settings.return_value = "settings_obj"

            memory = MagicMock()
            tool = MagicMock()
            browser_toolset = MagicMock()
            event_manager = MagicMock()

            with (
                patch(
                    "avalan.agent.loader.SentenceTransformerModel",
                    return_value=sentence_model,
                ),
                patch("avalan.agent.loader.TextPartitioner"),
                patch(
                    "avalan.agent.loader.MemoryManager.create_instance",
                    new=AsyncMock(return_value=memory),
                ),
                patch(
                    "avalan.agent.loader.ModelManager",
                    return_value=model_manager,
                ),
                patch(
                    "avalan.agent.loader.DefaultOrchestrator",
                    return_value="orch",
                ),
                patch(
                    "avalan.agent.loader.ToolManager.create_instance",
                    return_value=tool,
                ),
                patch(
                    "avalan.agent.loader.BrowserToolSet",
                    return_value=browser_toolset,
                ) as bts_patch,
                patch(
                    "avalan.agent.loader.EventManager",
                    return_value=event_manager,
                ),
            ):
                loader = OrchestratorLoader(
                    hub=hub,
                    logger=logger,
                    participant_id=uuid4(),
                    stack=stack,
                )
                await loader.from_file(
                    path,
                    agent_id=uuid4(),
                    disable_memory=True,
                )

                bs = bts_patch.call_args.kwargs["settings"]
                self.assertIsInstance(bs, BrowserToolSettings)
                self.assertEqual(bs.engine, "webkit")
                self.assertTrue(bs.debug)
            await stack.aclose()

    async def test_load_old_tool_settings_section(self):
        config = """
[agent]
role = \"assistant\"

[engine]
uri = \"ai://local/model\"

[tool.browser]
engine = \"chromium\"
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            hub = MagicMock(spec=HuggingfaceHub)
            logger = MagicMock(spec=Logger)
            stack = AsyncExitStack()

            sentence_model = MagicMock()
            sentence_model.__enter__.return_value = sentence_model

            model_manager = MagicMock()
            model_manager.__enter__.return_value = model_manager
            model_manager.parse_uri.return_value = "uri_obj"
            model_manager.get_engine_settings.return_value = "settings_obj"

            memory = MagicMock()
            tool = MagicMock()
            browser_toolset = MagicMock()
            event_manager = MagicMock()

            with (
                patch(
                    "avalan.agent.loader.SentenceTransformerModel",
                    return_value=sentence_model,
                ),
                patch("avalan.agent.loader.TextPartitioner"),
                patch(
                    "avalan.agent.loader.MemoryManager.create_instance",
                    new=AsyncMock(return_value=memory),
                ),
                patch(
                    "avalan.agent.loader.ModelManager",
                    return_value=model_manager,
                ),
                patch(
                    "avalan.agent.loader.DefaultOrchestrator",
                    return_value="orch",
                ),
                patch(
                    "avalan.agent.loader.ToolManager.create_instance",
                    return_value=tool,
                ),
                patch(
                    "avalan.agent.loader.BrowserToolSet",
                    return_value=browser_toolset,
                ) as bts_patch,
                patch(
                    "avalan.agent.loader.EventManager",
                    return_value=event_manager,
                ),
            ):
                loader = OrchestratorLoader(
                    hub=hub,
                    logger=logger,
                    participant_id=uuid4(),
                    stack=stack,
                )
                await loader.from_file(
                    path,
                    agent_id=uuid4(),
                    disable_memory=True,
                )

                bs = bts_patch.call_args.kwargs["settings"]
                self.assertIsInstance(bs, BrowserToolSettings)
                self.assertEqual(bs.engine, "chromium")
            await stack.aclose()

    async def test_load_database_tool_settings(self):
        config = """
[agent]
role = \"assistant\"

[engine]
uri = \"ai://local/model\"

[tool.database]
dsn = \"sqlite:///db.sqlite\"
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            hub = MagicMock(spec=HuggingfaceHub)
            logger = MagicMock(spec=Logger)
            stack = AsyncExitStack()

            with patch.object(
                OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as lfs_patch:
                loader = OrchestratorLoader(
                    hub=hub,
                    logger=logger,
                    participant_id=uuid4(),
                    stack=stack,
                )
                await loader.from_file(
                    path,
                    agent_id=uuid4(),
                )

                tool_settings = lfs_patch.call_args.kwargs["tool_settings"]
                dbs = tool_settings.database
                self.assertIsInstance(dbs, DatabaseToolSettings)
                self.assertEqual(dbs.dsn, "sqlite:///db.sqlite")
            await stack.aclose()

    async def test_tool_format_setting(self):
        config = """
[agent]
role = \"assistant\"

[engine]
uri = \"ai://local/model\"

[tool]
format = \"react\"
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            hub = MagicMock(spec=HuggingfaceHub)
            logger = MagicMock(spec=Logger)
            stack = AsyncExitStack()

            sentence_model = MagicMock()
            sentence_model.__enter__.return_value = sentence_model

            model_manager = MagicMock()
            model_manager.__enter__.return_value = model_manager
            model_manager.parse_uri.return_value = "uri_obj"
            model_manager.get_engine_settings.return_value = "settings_obj"

            memory = MagicMock()
            tool = MagicMock()
            browser_toolset = MagicMock()
            event_manager = MagicMock()

            with (
                patch(
                    "avalan.agent.loader.SentenceTransformerModel",
                    return_value=sentence_model,
                ),
                patch("avalan.agent.loader.TextPartitioner"),
                patch(
                    "avalan.agent.loader.MemoryManager.create_instance",
                    new=AsyncMock(return_value=memory),
                ),
                patch(
                    "avalan.agent.loader.ModelManager",
                    return_value=model_manager,
                ),
                patch(
                    "avalan.agent.loader.DefaultOrchestrator",
                    return_value="orch",
                ),
                patch(
                    "avalan.agent.loader.ToolManager.create_instance",
                    return_value=tool,
                ) as tm_patch,
                patch(
                    "avalan.agent.loader.BrowserToolSet",
                    return_value=browser_toolset,
                ),
                patch(
                    "avalan.agent.loader.EventManager",
                    return_value=event_manager,
                ),
            ):
                loader = OrchestratorLoader(
                    hub=hub,
                    logger=logger,
                    participant_id=uuid4(),
                    stack=stack,
                )
                await loader.from_file(path, agent_id=uuid4())

                settings = tm_patch.call_args.kwargs["settings"]
                self.assertEqual(settings.tool_format, ToolFormat.REACT)
            await stack.aclose()

    async def test_tool_settings_argument(self):
        config = """
[agent]
role = \"assistant\"

[engine]
uri = \"ai://local/model\"
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            hub = MagicMock(spec=HuggingfaceHub)
            logger = MagicMock(spec=Logger)
            stack = AsyncExitStack()

            tool_settings = ToolSettingsContext(
                browser=BrowserToolSettings(engine="webkit"),
                database=DatabaseToolSettings(dsn="sqlite:///db.sqlite"),
                extra={"x": 1},
            )

            with patch.object(
                OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as fs_patch:
                loader = OrchestratorLoader(
                    hub=hub,
                    logger=logger,
                    participant_id=uuid4(),
                    stack=stack,
                )
                result = await loader.from_file(
                    path, agent_id=uuid4(), tool_settings=tool_settings
                )
                self.assertEqual(result, "orch")
                fs_patch.assert_awaited_once()
                passed = fs_patch.call_args.kwargs["tool_settings"]
                self.assertEqual(passed.browser.engine, "webkit")
                self.assertEqual(passed.database.dsn, "sqlite:///db.sqlite")
                self.assertEqual(passed.extra, {"x": 1})
            await stack.aclose()

    async def test_load_json_orchestrator(self):
        config = """
[agent]
type = \"json\"
role = \"assistant\"
task = \"do\"
instructions = \"how\"

[engine]
uri = \"ai://local/model\"

[json]
value = { type = \"string\", description = \"d\" }
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            hub = MagicMock(spec=HuggingfaceHub)
            logger = MagicMock(spec=Logger)
            stack = AsyncExitStack()

            sentence_model = MagicMock()
            sentence_model.__enter__.return_value = sentence_model

            model_manager = MagicMock()
            model_manager.__enter__.return_value = model_manager
            model_manager.parse_uri.return_value = "uri_obj"
            model_manager.get_engine_settings.return_value = "settings_obj"

            memory = MagicMock()
            tool = MagicMock()
            event_manager = MagicMock()

            with (
                patch(
                    "avalan.agent.loader.SentenceTransformerModel",
                    return_value=sentence_model,
                ),
                patch("avalan.agent.loader.TextPartitioner"),
                patch(
                    "avalan.agent.loader.MemoryManager.create_instance",
                    new=AsyncMock(return_value=memory),
                ),
                patch(
                    "avalan.agent.loader.ModelManager",
                    return_value=model_manager,
                ),
                patch.object(
                    OrchestratorLoader,
                    "_load_json_orchestrator",
                    return_value="json_orch",
                ) as json_patch,
                patch(
                    "avalan.agent.loader.ToolManager.create_instance",
                    return_value=tool,
                ),
                patch(
                    "avalan.agent.loader.EventManager",
                    return_value=event_manager,
                ),
            ):
                loader = OrchestratorLoader(
                    hub=hub,
                    logger=logger,
                    participant_id=uuid4(),
                    stack=stack,
                )
                result = await loader.from_file(
                    path,
                    agent_id=uuid4(),
                    disable_memory=True,
                )

                self.assertEqual(result, "json_orch")
                json_patch.assert_called_once()
            await stack.aclose()

    async def test_unknown_type(self):
        config = """
[agent]
type = \"foo\"
role = \"assistant\"

[engine]
uri = \"ai://local/model\"
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            stack = AsyncExitStack()
            loader = OrchestratorLoader(
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=stack,
            )
            with self.assertRaises(AssertionError):
                await loader.from_file(path, agent_id=uuid4())
        await stack.aclose()

    async def test_from_file_overrides_uri(self):
        config = """
[agent]
role = \"assistant\"

[engine]
uri = \"ai://old/model\"
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            hub = MagicMock(spec=HuggingfaceHub)
            logger = MagicMock(spec=Logger)
            stack = AsyncExitStack()
            loader = OrchestratorLoader(
                hub=hub,
                logger=logger,
                participant_id=uuid4(),
                stack=stack,
            )
            with patch.object(
                loader,
                "from_settings",
                AsyncMock(return_value=MagicMock()),
            ) as fs:
                await loader.from_file(
                    path, agent_id=uuid4(), uri="ai://new/model"
                )
            settings_arg = fs.await_args.args[0]
            self.assertEqual(settings_arg.uri, "ai://new/model")
        await stack.aclose()

    async def test_sentence_model_engine_config(self):
        config = """
[agent]
role = \"assistant\"

[engine]
uri = \"ai://local/model\"

["memory.engine"]
model_id = "smodel"
max_tokens = 300
overlap_size = 60
window_size = 120
backend = "onnx"
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            hub = MagicMock(spec=HuggingfaceHub)
            logger = MagicMock(spec=Logger)
            stack = AsyncExitStack()

            with patch.object(
                OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as lfs_patch:
                loader = OrchestratorLoader(
                    hub=hub,
                    logger=logger,
                    participant_id=uuid4(),
                    stack=stack,
                )
                result = await loader.from_file(
                    path,
                    agent_id=uuid4(),
                )

                self.assertEqual(result, "orch")
                lfs_patch.assert_awaited_once()
                settings = lfs_patch.call_args.args[0]
                self.assertEqual(settings.sentence_model_id, "smodel")
                self.assertEqual(settings.sentence_model_max_tokens, 300)
                self.assertEqual(settings.sentence_model_overlap_size, 60)
                self.assertEqual(settings.sentence_model_window_size, 120)
                self.assertEqual(
                    settings.sentence_model_engine_config, {"backend": "onnx"}
                )
            await stack.aclose()


class LoaderTomlVariantsTestCase(IsolatedAsyncioTestCase):
    async def _run_loader(self, config: str) -> dict:
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            hub = MagicMock(spec=HuggingfaceHub)
            logger = MagicMock(spec=Logger)
            stack = AsyncExitStack()

            sentence_model = MagicMock()
            sentence_model.__enter__.return_value = sentence_model

            model_manager = MagicMock()
            model_manager.__enter__.return_value = model_manager
            model_manager.parse_uri.return_value = "uri_obj"
            model_manager.get_engine_settings.return_value = "settings_obj"

            memory = MagicMock()
            tool = MagicMock()
            browser_toolset = MagicMock()
            event_manager = MagicMock()

            with (
                patch(
                    "avalan.agent.loader.SentenceTransformerModel",
                    return_value=sentence_model,
                ),
                patch("avalan.agent.loader.TextPartitioner"),
                patch(
                    "avalan.agent.loader.MemoryManager.create_instance",
                    new=AsyncMock(return_value=memory),
                ),
                patch(
                    "avalan.agent.loader.ModelManager",
                    return_value=model_manager,
                ),
                patch(
                    "avalan.agent.loader.DefaultOrchestrator",
                    return_value="orch",
                ) as orch_patch,
                patch(
                    "avalan.agent.loader.ToolManager.create_instance",
                    return_value=tool,
                ),
                patch(
                    "avalan.agent.loader.BrowserToolSet",
                    return_value=browser_toolset,
                ),
                patch(
                    "avalan.agent.loader.EventManager",
                    return_value=event_manager,
                ),
            ):
                loader = OrchestratorLoader(
                    hub=hub,
                    logger=logger,
                    participant_id=uuid4(),
                    stack=stack,
                )
                await loader.from_file(
                    path,
                    agent_id=uuid4(),
                    disable_memory=True,
                )

                kwargs = orch_patch.call_args.kwargs
            await stack.aclose()
            return kwargs

    async def test_agent_system_only(self):
        config = """
[agent]
system = \"sys\"

[engine]
uri = \"ai://local/model\"
"""
        kwargs = await self._run_loader(config)
        self.assertEqual(kwargs["system"], "sys")
        self.assertIsNone(kwargs["role"])
        self.assertIsNone(kwargs["task"])
        self.assertIsNone(kwargs["instructions"])
        self.assertIsNone(kwargs["rules"])

    async def test_agent_role_task_only(self):
        config = """
[agent]
role = \"assistant\"
task = \"do\"

[engine]
uri = \"ai://local/model\"
"""
        kwargs = await self._run_loader(config)
        self.assertEqual(kwargs["role"], "assistant")
        self.assertEqual(kwargs["task"], "do")
        self.assertIsNone(kwargs["instructions"])
        self.assertIsNone(kwargs["system"])

    async def test_agent_full_definition(self):
        config = """
[agent]
role = \"assistant\"
task = \"do\"
instructions = \"how\"
rules = [\"r1\", \"r2\"]

[engine]
uri = \"ai://local/model\"
"""
        kwargs = await self._run_loader(config)
        self.assertEqual(kwargs["role"], "assistant")
        self.assertEqual(kwargs["task"], "do")
        self.assertEqual(kwargs["instructions"], "how")
        self.assertEqual(kwargs["rules"], ["r1", "r2"])
        self.assertIsNone(kwargs["system"])

    async def test_agent_user_only(self):
        config = """
[agent]
user = \"hello {{input}}\"

[engine]
uri = \"ai://local/model\"
"""
        kwargs = await self._run_loader(config)
        self.assertEqual(kwargs["user"], "hello {{input}}")
        self.assertIsNone(kwargs.get("user_template"))

    async def test_agent_user_template_only(self):
        config = """
[agent]
user_template = \"u.md\"

[engine]
uri = \"ai://local/model\"
"""
        kwargs = await self._run_loader(config)
        self.assertEqual(kwargs["user_template"], "u.md")
        self.assertIsNone(kwargs.get("user"))


class LoadJsonOrchestratorVariantsTestCase(IsolatedAsyncioTestCase):
    def test_system_only(self):
        agent_id = uuid4()
        engine_uri = MagicMock()
        engine_settings = MagicMock()
        logger = MagicMock()
        model_manager = MagicMock()
        memory = MagicMock()
        tool = MagicMock()
        event_manager = MagicMock()

        config = {"json": {"value": {"type": "string", "description": "v"}}}
        agent_config = {"system": "sys"}

        with patch("avalan.agent.loader.JsonOrchestrator") as orch_patch:
            OrchestratorLoader._load_json_orchestrator(
                agent_id=agent_id,
                engine_uri=engine_uri,
                engine_settings=engine_settings,
                logger=logger,
                model_manager=model_manager,
                memory=memory,
                tool=tool,
                event_manager=event_manager,
                config=config,
                agent_config=agent_config,
                call_options=None,
                template_vars=None,
            )
            kwargs = orch_patch.call_args.kwargs
            self.assertEqual(kwargs["system"], "sys")
            self.assertIsNone(kwargs["task"])
            self.assertIsNone(kwargs["instructions"])

    async def test_browser_debug_source(self):
        with TemporaryDirectory() as tmp:
            debug_path = f"{tmp}/debug.txt"
            with open(debug_path, "w", encoding="utf-8") as fh:
                fh.write("debug")

            config = f"""
[agent]
role = \"assistant\"

[engine]
uri = \"ai://local/model\"

[tool.browser.open]
debug = true
debug_source = \"{debug_path}\"
"""
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            hub = MagicMock(spec=HuggingfaceHub)
            logger = MagicMock(spec=Logger)
            stack = AsyncExitStack()

            with patch.object(
                OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as lfs_patch:
                loader = OrchestratorLoader(
                    hub=hub,
                    logger=logger,
                    participant_id=uuid4(),
                    stack=stack,
                )
                result = await loader.from_file(
                    path,
                    agent_id=uuid4(),
                )

                self.assertEqual(result, "orch")
                lfs_patch.assert_awaited_once()
                tool_settings = lfs_patch.call_args.kwargs["tool_settings"]
                browser_settings = tool_settings.browser
                self.assertTrue(browser_settings.debug)
                self.assertIsNotNone(browser_settings.debug_source)
                self.assertEqual(browser_settings.debug_source.read(), "debug")
                browser_settings.debug_source.close()
                dbs = tool_settings.database
                self.assertIsNone(dbs)
            await stack.aclose()

    async def test_json_settings_provided(self):
        config = """
[agent]
role = \"assistant\"
task = \"do\"
instructions = \"ins\"

[engine]
uri = \"ai://local/model\"

[json]
value = { type = \"string\", description = \"d\" }
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            hub = MagicMock(spec=HuggingfaceHub)
            logger = MagicMock(spec=Logger)
            stack = AsyncExitStack()

            with patch.object(
                OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as lfs_patch:
                loader = OrchestratorLoader(
                    hub=hub,
                    logger=logger,
                    participant_id=uuid4(),
                    stack=stack,
                )
                result = await loader.from_file(
                    path,
                    agent_id=uuid4(),
                )

                self.assertEqual(result, "orch")
                lfs_patch.assert_awaited_once()
                settings = lfs_patch.call_args.args[0]
                self.assertEqual(
                    settings.json_config,
                    {"value": {"type": "string", "description": "d"}},
                )
        await stack.aclose()

    async def test_run_chat_settings_from_file(self):
        config = """
[agent]
role = \"assistant\"

[engine]
uri = \"ai://local/model\"

[run.chat]
enable_thinking = true
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            hub = MagicMock(spec=HuggingfaceHub)
            logger = MagicMock(spec=Logger)
            stack = AsyncExitStack()

            with patch.object(
                OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as lfs_patch:
                loader = OrchestratorLoader(
                    hub=hub,
                    logger=logger,
                    participant_id=uuid4(),
                    stack=stack,
                )
                result = await loader.from_file(
                    path,
                    agent_id=uuid4(),
                )

                self.assertEqual(result, "orch")
                lfs_patch.assert_awaited_once()
                settings = lfs_patch.call_args.args[0]
                self.assertTrue(
                    settings.call_options["chat_settings"]["enable_thinking"]
                )
            await stack.aclose()

    async def test_permanent_memory_from_file(self):
        config_tmpl = """
[agent]
role = \"assistant\"
task = \"do\"
instructions = \"how\"

[engine]
uri = \"ai://local/model\"

[memory]
permanent = {{ {entries} }}
"""
        cases = [
            {"code": "dsn1"},
            {"code": "dsn1", "docs": "dsn2"},
            {
                "code": "dsn1",
                "docs": "dsn2",
                "more": "dsn3",
            },
        ]

        for case in cases:
            with self.subTest(case=case):
                entries = ", ".join(f'{k} = "{v}"' for k, v in case.items())
                config = config_tmpl.format(entries=entries)
                with TemporaryDirectory() as tmp:
                    path = f"{tmp}/agent.toml"
                    with open(path, "w", encoding="utf-8") as fh:
                        fh.write(config)

                    stack = AsyncExitStack()
                    with patch.object(
                        OrchestratorLoader,
                        "from_settings",
                        new=AsyncMock(return_value="orch"),
                    ) as lfs_patch:
                        loader = OrchestratorLoader(
                            hub=MagicMock(spec=HuggingfaceHub),
                            logger=MagicMock(spec=Logger),
                            participant_id=uuid4(),
                            stack=stack,
                        )
                        result = await loader.from_file(
                            path,
                            agent_id=uuid4(),
                        )

                        self.assertEqual(result, "orch")
                        lfs_patch.assert_awaited_once()
                        settings = lfs_patch.call_args.args[0]
                        self.assertEqual(settings.permanent_memory, case)
                    await stack.aclose()

    async def test_engine_only_generates_id(self):
        config = """
[agent]

[engine]
uri = \"ai://local/model\"
"""
        uid = uuid4()
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            with (
                patch.object(
                    OrchestratorLoader,
                    "from_settings",
                    new=AsyncMock(return_value="orch"),
                ) as lfs_patch,
                patch("avalan.agent.loader.uuid4", return_value=uid),
            ):
                stack = AsyncExitStack()
                loader = OrchestratorLoader(
                    hub=MagicMock(spec=HuggingfaceHub),
                    logger=MagicMock(spec=Logger),
                    participant_id=uuid4(),
                    stack=stack,
                )
                result = await loader.from_file(path, agent_id=None)

                self.assertEqual(result, "orch")
                lfs_patch.assert_awaited_once()
                settings = lfs_patch.call_args.args[0]
                self.assertEqual(settings.agent_id, uid)
                self.assertEqual(settings.uri, "ai://local/model")
            await stack.aclose()

    async def test_engine_with_id(self):
        uid = uuid4()
        config = f"""
[agent]
id = \"{uid}\"

[engine]
uri = \"ai://local/model\"
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            with patch.object(
                OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as lfs_patch:
                stack = AsyncExitStack()
                loader = OrchestratorLoader(
                    hub=MagicMock(spec=HuggingfaceHub),
                    logger=MagicMock(spec=Logger),
                    participant_id=uuid4(),
                    stack=stack,
                )
                result = await loader.from_file(path, agent_id=None)

                self.assertEqual(result, "orch")
                lfs_patch.assert_awaited_once()
                settings = lfs_patch.call_args.args[0]
                self.assertEqual(settings.agent_id, str(uid))
            await stack.aclose()

    async def test_engine_with_name(self):
        config = """
[agent]
name = \"Agent\"

[engine]
uri = \"ai://local/model\"
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            with patch.object(
                OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as lfs_patch:
                stack = AsyncExitStack()
                loader = OrchestratorLoader(
                    hub=MagicMock(spec=HuggingfaceHub),
                    logger=MagicMock(spec=Logger),
                    participant_id=uuid4(),
                    stack=stack,
                )
                result = await loader.from_file(path, agent_id=None)

                self.assertEqual(result, "orch")
                lfs_patch.assert_awaited_once()
                settings = lfs_patch.call_args.args[0]
                self.assertEqual(settings.agent_config.get("name"), "Agent")
            await stack.aclose()

    async def test_engine_generation_settings(self):
        config = """
[agent]

[engine]
uri = \"ai://local/model\"

[run]
temperature = 0.5
top_p = 0.9
top_k = 5
max_new_tokens = 42
"""
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            with patch.object(
                OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value="orch"),
            ) as lfs_patch:
                stack = AsyncExitStack()
                loader = OrchestratorLoader(
                    hub=MagicMock(spec=HuggingfaceHub),
                    logger=MagicMock(spec=Logger),
                    participant_id=uuid4(),
                    stack=stack,
                )
                result = await loader.from_file(path, agent_id=None)

                self.assertEqual(result, "orch")
                lfs_patch.assert_awaited_once()
                settings = lfs_patch.call_args.args[0]
                self.assertEqual(settings.call_options["temperature"], 0.5)
                self.assertEqual(settings.call_options["top_p"], 0.9)
                self.assertEqual(settings.call_options["top_k"], 5)
                self.assertEqual(settings.call_options["max_new_tokens"], 42)
            await stack.aclose()


class LoaderFromSettingsTestCase(IsolatedAsyncioTestCase):
    async def test_load_default_orchestrator_from_settings(self):
        hub = MagicMock(spec=HuggingfaceHub)
        logger = MagicMock(spec=Logger)
        stack = AsyncExitStack()

        sentence_model = MagicMock()
        sentence_model.__enter__.return_value = sentence_model

        model_manager = MagicMock()
        model_manager.__enter__.return_value = model_manager
        model_manager.parse_uri.return_value = "uri_obj"
        model_manager.get_engine_settings.return_value = "settings_obj"

        memory = MagicMock()
        tool = MagicMock()
        event_manager = MagicMock()

        settings = OrchestratorSettings(
            agent_id=uuid4(),
            orchestrator_type=None,
            agent_config={"role": "assistant"},
            uri="ai://local/model",
            engine_config={},
            tools=None,
            call_options=None,
            template_vars=None,
            memory_permanent_message=None,
            permanent_memory=None,
            memory_recent=False,
            sentence_model_id=OrchestratorLoader.DEFAULT_SENTENCE_MODEL_ID,
            sentence_model_engine_config=None,
            sentence_model_max_tokens=500,
            sentence_model_overlap_size=125,
            sentence_model_window_size=250,
            json_config=None,
            log_events=True,
        )

        with (
            patch(
                "avalan.agent.loader.SentenceTransformerModel",
                return_value=sentence_model,
            ),
            patch("avalan.agent.loader.TextPartitioner"),
            patch(
                "avalan.agent.loader.MemoryManager.create_instance",
                new=AsyncMock(return_value=memory),
            ) as mm_patch,
            patch(
                "avalan.agent.loader.ModelManager", return_value=model_manager
            ) as model_patch,
            patch(
                "avalan.agent.loader.DefaultOrchestrator", return_value="orch"
            ) as orch_patch,
            patch(
                "avalan.agent.loader.ToolManager.create_instance",
                return_value=tool,
            ),
            patch(
                "avalan.agent.loader.EventManager", return_value=event_manager
            ),
        ):
            loader = OrchestratorLoader(
                hub=hub,
                logger=logger,
                participant_id=uuid4(),
                stack=stack,
            )
            result = await loader.from_settings(settings)

            self.assertEqual(result, "orch")
            orch_patch.assert_called_once()
            model_patch.assert_called_once_with(
                hub, logger, event_manager=event_manager
            )
            mm_patch.assert_awaited_once()
        await stack.aclose()

    async def test_permanent_memory_from_settings(self):
        base_settings = dict(
            agent_id=uuid4(),
            orchestrator_type=None,
            agent_config={"role": "assistant"},
            uri="ai://local/model",
            engine_config={},
            tools=None,
            call_options=None,
            template_vars=None,
            memory_permanent_message=None,
            memory_recent=False,
            sentence_model_id=OrchestratorLoader.DEFAULT_SENTENCE_MODEL_ID,
            sentence_model_engine_config=None,
            sentence_model_max_tokens=500,
            sentence_model_overlap_size=125,
            sentence_model_window_size=250,
            json_config=None,
            log_events=True,
        )

        cases = [
            {"code": "dsn1"},
            {"code": "dsn1", "docs": "dsn2"},
            {
                "code": "dsn1",
                "docs": "dsn2",
                "more": "dsn3",
            },
        ]

        for case in cases:
            with self.subTest(case=case):
                hub = MagicMock(spec=HuggingfaceHub)
                logger = MagicMock(spec=Logger)
                stack = AsyncExitStack()

                sentence_model = MagicMock()
                sentence_model.__enter__.return_value = sentence_model

                model_manager = MagicMock()
                model_manager.__enter__.return_value = model_manager
                model_manager.parse_uri.return_value = "uri_obj"
                model_manager.get_engine_settings.return_value = "settings_obj"

                memory = MagicMock()
                tool = MagicMock()
                event_manager = MagicMock()

                settings = OrchestratorSettings(
                    **base_settings,
                    permanent_memory=case,
                )

                with (
                    patch(
                        "avalan.agent.loader.SentenceTransformerModel",
                        return_value=sentence_model,
                    ),
                    patch("avalan.agent.loader.TextPartitioner"),
                    patch(
                        "avalan.agent.loader.MemoryManager.create_instance",
                        new=AsyncMock(return_value=memory),
                    ),
                    patch(
                        "avalan.agent.loader.ModelManager",
                        return_value=model_manager,
                    ),
                    patch(
                        "avalan.agent.loader.PgsqlRawMemory.create_instance",
                        new=AsyncMock(side_effect=[MagicMock() for _ in case]),
                    ) as pg_patch,
                    patch(
                        "avalan.agent.loader.DefaultOrchestrator",
                        return_value="orch",
                    ),
                    patch(
                        "avalan.agent.loader.ToolManager.create_instance",
                        return_value=tool,
                    ),
                    patch(
                        "avalan.agent.loader.EventManager",
                        return_value=event_manager,
                    ),
                ):
                    loader = OrchestratorLoader(
                        hub=hub,
                        logger=logger,
                        participant_id=uuid4(),
                        stack=stack,
                    )
                    await loader.from_settings(settings)

                    self.assertEqual(pg_patch.await_count, len(case))
                    self.assertEqual(
                        memory.add_permanent_memory.call_count, len(case)
                    )
                await stack.aclose()

    async def test_database_tool_from_settings(self):
        hub = MagicMock(spec=HuggingfaceHub)
        logger = MagicMock(spec=Logger)
        stack = AsyncExitStack()

        sentence_model = MagicMock()
        sentence_model.__enter__.return_value = sentence_model

        model_manager = MagicMock()
        model_manager.__enter__.return_value = model_manager
        model_manager.parse_uri.return_value = "uri_obj"
        model_manager.get_engine_settings.return_value = "settings_obj"

        memory = MagicMock()
        tool = MagicMock()
        tool.__aenter__.return_value = tool
        event_manager = MagicMock()

        settings = OrchestratorSettings(
            agent_id=uuid4(),
            orchestrator_type=None,
            agent_config={"role": "assistant"},
            uri="ai://local/model",
            engine_config={},
            tools=None,
            call_options=None,
            template_vars=None,
            memory_permanent_message=None,
            permanent_memory=None,
            memory_recent=False,
            sentence_model_id=OrchestratorLoader.DEFAULT_SENTENCE_MODEL_ID,
            sentence_model_engine_config=None,
            sentence_model_max_tokens=500,
            sentence_model_overlap_size=125,
            sentence_model_window_size=250,
            json_config=None,
            log_events=True,
        )

        db_settings = DatabaseToolSettings(dsn="sqlite:///db.sqlite")

        browser_tool = MagicMock()
        code_tool = MagicMock()
        math_tool = MagicMock()
        memory_tool = MagicMock()
        db_tool = MagicMock()

        with (
            patch(
                "avalan.agent.loader.SentenceTransformerModel",
                return_value=sentence_model,
            ),
            patch("avalan.agent.loader.TextPartitioner"),
            patch(
                "avalan.agent.loader.MemoryManager.create_instance",
                new=AsyncMock(return_value=memory),
            ),
            patch(
                "avalan.agent.loader.ModelManager", return_value=model_manager
            ),
            patch(
                "avalan.agent.loader.DefaultOrchestrator", return_value="orch"
            ),
            patch(
                "avalan.agent.loader.ToolManager.create_instance",
                return_value=tool,
            ) as tm_patch,
            patch(
                "avalan.agent.loader.EventManager", return_value=event_manager
            ),
            patch(
                "avalan.agent.loader.BrowserToolSet", return_value=browser_tool
            ),
            patch("avalan.agent.loader.CodeToolSet", return_value=code_tool),
            patch("avalan.agent.loader.MathToolSet", return_value=math_tool),
            patch(
                "avalan.agent.loader.MemoryToolSet", return_value=memory_tool
            ),
            patch(
                "avalan.agent.loader.DatabaseToolSet",
                return_value=db_tool,
            ) as db_patch,
        ):
            loader = OrchestratorLoader(
                hub=hub,
                logger=logger,
                participant_id=uuid4(),
                stack=stack,
            )
            result = await loader.from_settings(
                settings,
                tool_settings=ToolSettingsContext(database=db_settings),
            )

            self.assertEqual(result, "orch")
            db_patch.assert_called_once_with(
                settings=db_settings, namespace="database"
            )
            available = tm_patch.call_args.kwargs["available_toolsets"]
            self.assertIn(db_tool, available)
        await stack.aclose()

    async def test_tool_settings_logged(self):
        hub = MagicMock(spec=HuggingfaceHub)
        logger = MagicMock(spec=Logger)
        stack = AsyncExitStack()

        sentence_model = MagicMock()
        sentence_model.__enter__.return_value = sentence_model

        model_manager = MagicMock()
        model_manager.__enter__.return_value = model_manager
        model_manager.parse_uri.return_value = "uri_obj"
        model_manager.get_engine_settings.return_value = "settings_obj"

        memory = MagicMock()
        tool = MagicMock()
        tool.__aenter__ = AsyncMock(return_value=tool)
        event_manager = MagicMock()

        settings = OrchestratorSettings(
            agent_id=uuid4(),
            orchestrator_type=None,
            agent_config={"role": "assistant"},
            uri="ai://local/model",
            engine_config={},
            tools=None,
            call_options=None,
            template_vars=None,
            memory_permanent_message=None,
            permanent_memory=None,
            memory_recent=False,
            sentence_model_id=OrchestratorLoader.DEFAULT_SENTENCE_MODEL_ID,
            sentence_model_engine_config=None,
            sentence_model_max_tokens=500,
            sentence_model_overlap_size=125,
            sentence_model_window_size=250,
            json_config=None,
            log_events=True,
        )

        browser_settings = BrowserToolSettings()
        db_settings = DatabaseToolSettings(dsn="sqlite:///db.sqlite")

        with (
            patch(
                "avalan.agent.loader.SentenceTransformerModel",
                return_value=sentence_model,
            ),
            patch("avalan.agent.loader.TextPartitioner"),
            patch(
                "avalan.agent.loader.MemoryManager.create_instance",
                new=AsyncMock(return_value=memory),
            ),
            patch(
                "avalan.agent.loader.ModelManager",
                return_value=model_manager,
            ),
            patch(
                "avalan.agent.loader.DefaultOrchestrator",
                return_value="orch",
            ),
            patch(
                "avalan.agent.loader.ToolManager.create_instance",
                return_value=tool,
            ),
            patch(
                "avalan.agent.loader.EventManager", return_value=event_manager
            ),
            patch("avalan.agent.loader.BrowserToolSet"),
            patch("avalan.agent.loader.CodeToolSet"),
            patch("avalan.agent.loader.MathToolSet"),
            patch("avalan.agent.loader.MemoryToolSet"),
            patch("avalan.agent.loader.DatabaseToolSet"),
        ):
            loader = OrchestratorLoader(
                hub=hub,
                logger=logger,
                participant_id=uuid4(),
                stack=stack,
            )
            result = await loader.from_settings(
                settings,
                tool_settings=ToolSettingsContext(
                    browser=browser_settings, database=db_settings
                ),
            )

            self.assertEqual(result, "orch")
            logger.log.assert_any_call(
                DEBUG,
                "<OrchestratorLoader> Tool settings: browser=%s, database=%s",
                browser_settings,
                db_settings,
            )
        await stack.aclose()

    async def test_event_logging_listener(self):
        hub = MagicMock(spec=HuggingfaceHub)
        logger = MagicMock(spec=Logger)
        stack = AsyncExitStack()

        sentence_model = MagicMock()
        sentence_model.__enter__.return_value = sentence_model

        model_manager = MagicMock()
        model_manager.__enter__.return_value = model_manager
        model_manager.parse_uri.return_value = "uri_obj"
        model_manager.get_engine_settings.return_value = "settings_obj"

        memory = MagicMock()
        tool = MagicMock()
        tool.__aenter__ = AsyncMock(return_value=tool)
        tool.__aexit__ = AsyncMock(return_value=None)

        class DummyEventManager:
            def __init__(self) -> None:
                self.listeners: list[Callable[[Event], Any]] = []

            def add_listener(
                self,
                listener: Callable[[Event], Any],
                event_types: Any | None = None,
            ) -> None:
                self.listeners.append(listener)

        event_manager = DummyEventManager()

        settings = OrchestratorSettings(
            agent_id=uuid4(),
            orchestrator_type=None,
            agent_config={"role": "assistant"},
            uri="ai://local/model",
            engine_config={},
            tools=None,
            call_options=None,
            template_vars=None,
            memory_permanent_message=None,
            permanent_memory=None,
            memory_recent=False,
            sentence_model_id=OrchestratorLoader.DEFAULT_SENTENCE_MODEL_ID,
            sentence_model_engine_config=None,
            sentence_model_max_tokens=500,
            sentence_model_overlap_size=125,
            sentence_model_window_size=250,
            json_config=None,
            log_events=True,
        )

        with (
            patch(
                "avalan.agent.loader.SentenceTransformerModel",
                return_value=sentence_model,
            ),
            patch("avalan.agent.loader.TextPartitioner"),
            patch(
                "avalan.agent.loader.MemoryManager.create_instance",
                new=AsyncMock(return_value=memory),
            ),
            patch(
                "avalan.agent.loader.ModelManager", return_value=model_manager
            ),
            patch(
                "avalan.agent.loader.DefaultOrchestrator", return_value="orch"
            ),
            patch(
                "avalan.agent.loader.ToolManager.create_instance",
                return_value=tool,
            ),
            patch(
                "avalan.agent.loader.EventManager", return_value=event_manager
            ),
        ):
            loader = OrchestratorLoader(
                hub=hub,
                logger=logger,
                participant_id=uuid4(),
                stack=stack,
            )
            result = await loader.from_settings(settings)

        self.assertEqual(result, "orch")
        self.assertEqual(len(event_manager.listeners), 1)

        listener = event_manager.listeners[0]
        logger.log.reset_mock()

        tool_event = Event(
            type=EventType.TOOL_PROCESS,
            payload={"message": "tool running"},
        )
        listener(tool_event)
        logger.log.assert_called_once()
        level, message, payload = logger.log.call_args.args
        self.assertEqual(level, INFO)
        self.assertEqual(
            message, "<Event tool_process @ OrchestratorLoader> %s"
        )
        self.assertEqual(payload, tool_event.payload)

        logger.log.reset_mock()

        debug_event = Event(
            type=EventType.START,
            payload={"message": "start"},
        )
        listener(debug_event)
        logger.log.assert_called_once()
        level, message, payload = logger.log.call_args.args
        self.assertEqual(level, DEBUG)
        self.assertEqual(message, "<Event start @ OrchestratorLoader> %s")
        self.assertEqual(payload, debug_event.payload)

        await stack.aclose()

    async def test_load_json_orchestrator_from_settings(self):
        hub = MagicMock(spec=HuggingfaceHub)
        logger = MagicMock(spec=Logger)
        stack = AsyncExitStack()

        sentence_model = MagicMock()
        sentence_model.__enter__.return_value = sentence_model

        model_manager = MagicMock()
        model_manager.__enter__.return_value = model_manager
        model_manager.parse_uri.return_value = "uri_obj"
        model_manager.get_engine_settings.return_value = "settings_obj"

        memory = MagicMock()
        tool = MagicMock()
        event_manager = MagicMock()

        settings = OrchestratorSettings(
            agent_id=uuid4(),
            orchestrator_type="json",
            agent_config={
                "role": "assistant",
                "task": "do",
                "instructions": "how",
            },
            uri="ai://local/model",
            engine_config={},
            tools=None,
            call_options=None,
            template_vars=None,
            memory_permanent_message=None,
            permanent_memory=None,
            memory_recent=False,
            sentence_model_id=OrchestratorLoader.DEFAULT_SENTENCE_MODEL_ID,
            sentence_model_engine_config=None,
            sentence_model_max_tokens=500,
            sentence_model_overlap_size=125,
            sentence_model_window_size=250,
            json_config={"value": {"type": "string", "description": "d"}},
            log_events=True,
        )

        with (
            patch(
                "avalan.agent.loader.SentenceTransformerModel",
                return_value=sentence_model,
            ),
            patch("avalan.agent.loader.TextPartitioner"),
            patch(
                "avalan.agent.loader.MemoryManager.create_instance",
                new=AsyncMock(return_value=memory),
            ),
            patch(
                "avalan.agent.loader.ModelManager", return_value=model_manager
            ),
            patch.object(
                OrchestratorLoader,
                "_load_json_orchestrator",
                return_value="json_orch",
            ) as json_patch,
            patch(
                "avalan.agent.loader.ToolManager.create_instance",
                return_value=tool,
            ),
            patch(
                "avalan.agent.loader.EventManager", return_value=event_manager
            ),
        ):
            loader = OrchestratorLoader(
                hub=hub,
                logger=logger,
                participant_id=uuid4(),
                stack=stack,
            )
            result = await loader.from_settings(settings)

            self.assertEqual(result, "json_orch")
            json_patch.assert_called_once()
        await stack.aclose()

    async def test_load_json_orchestrator_properties(self):
        agent_id = uuid4()
        engine_uri = MagicMock()
        engine_settings = MagicMock()
        logger = MagicMock()
        model_manager = MagicMock()
        memory = MagicMock()
        tool = MagicMock()
        event_manager = MagicMock()

        config = {
            "json": {
                "name": {"type": "string", "description": "n"},
                "age": {"type": "integer", "description": "a"},
            }
        }

        agent_config = {
            "role": "assistant",
            "task": "do",
            "instructions": "how",
        }

        with patch("avalan.agent.loader.JsonOrchestrator") as orch_patch:
            OrchestratorLoader._load_json_orchestrator(
                agent_id=agent_id,
                engine_uri=engine_uri,
                engine_settings=engine_settings,
                logger=logger,
                model_manager=model_manager,
                memory=memory,
                tool=tool,
                event_manager=event_manager,
                config=config,
                agent_config=agent_config,
                call_options=None,
                template_vars=None,
            )

            orch_patch.assert_called_once()
            properties = orch_patch.call_args.args[6]
            self.assertEqual(len(properties), 2)
            self.assertEqual(properties[0].name, "name")
            self.assertEqual(properties[1].data_type, "integer")


if __name__ == "__main__":
    main()
