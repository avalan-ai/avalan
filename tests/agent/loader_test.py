from avalan.agent.loader import OrchestratorLoader
from avalan.entities import OrchestratorSettings
from avalan.model.hubs.huggingface import HuggingfaceHub
from contextlib import AsyncExitStack
from logging import Logger
from tempfile import NamedTemporaryFile, TemporaryDirectory
from os import chmod, geteuid
from uuid import uuid4
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import AsyncMock, MagicMock, patch
from avalan.tool.browser import BrowserToolSettings


class LoaderFromFileTestCase(IsolatedAsyncioTestCase):
    async def test_file_not_found(self):
        stack = AsyncExitStack()
        with self.assertRaises(FileNotFoundError):
            await OrchestratorLoader.from_file(
                "missing.toml",
                agent_id=uuid4(),
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=stack,
            )
        await stack.aclose()

    async def test_permission_error(self):
        if geteuid() == 0:
            self.skipTest("Running as root; permission error won't occur")
        with NamedTemporaryFile(delete=False) as tmp:
            path = tmp.name
        chmod(path, 0)
        stack = AsyncExitStack()
        try:
            with self.assertRaises(PermissionError):
                await OrchestratorLoader.from_file(
                    path,
                    agent_id=uuid4(),
                    hub=MagicMock(spec=HuggingfaceHub),
                    logger=MagicMock(spec=Logger),
                    participant_id=uuid4(),
                    stack=stack,
                )
        finally:
            chmod(path, 0o644)
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
                result = await OrchestratorLoader.from_file(
                    path,
                    agent_id=uuid4(),
                    hub=hub,
                    logger=logger,
                    participant_id=uuid4(),
                    stack=stack,
                    disable_memory=True,
                )

                self.assertEqual(result, "orch")
                orch_patch.assert_called_once()
                model_patch.assert_called_once_with(hub, logger)
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
                await OrchestratorLoader.from_file(
                    path,
                    agent_id=uuid4(),
                    hub=hub,
                    logger=logger,
                    participant_id=uuid4(),
                    stack=stack,
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
                await OrchestratorLoader.from_file(
                    path,
                    agent_id=uuid4(),
                    hub=hub,
                    logger=logger,
                    participant_id=uuid4(),
                    stack=stack,
                    disable_memory=True,
                )

                bs = bts_patch.call_args.kwargs["settings"]
                self.assertIsInstance(bs, BrowserToolSettings)
                self.assertEqual(bs.engine, "chromium")
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
                result = await OrchestratorLoader.from_file(
                    path,
                    agent_id=uuid4(),
                    hub=hub,
                    logger=logger,
                    participant_id=uuid4(),
                    stack=stack,
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
            with self.assertRaises(AssertionError):
                await OrchestratorLoader.from_file(
                    path,
                    agent_id=uuid4(),
                    hub=MagicMock(spec=HuggingfaceHub),
                    logger=MagicMock(spec=Logger),
                    participant_id=uuid4(),
                    stack=stack,
                )
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
                "load_from_settings",
                new=AsyncMock(return_value="orch"),
            ) as lfs_patch:
                result = await OrchestratorLoader.from_file(
                    path,
                    agent_id=uuid4(),
                    hub=hub,
                    logger=logger,
                    participant_id=uuid4(),
                    stack=stack,
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
                "load_from_settings",
                new=AsyncMock(return_value="orch"),
            ) as lfs_patch:
                result = await OrchestratorLoader.from_file(
                    path,
                    agent_id=uuid4(),
                    hub=hub,
                    logger=logger,
                    participant_id=uuid4(),
                    stack=stack,
                )

                self.assertEqual(result, "orch")
                lfs_patch.assert_awaited_once()
                browser_settings = lfs_patch.call_args.kwargs[
                    "browser_settings"
                ]
                self.assertTrue(browser_settings.debug)
                self.assertIsNotNone(browser_settings.debug_source)
                self.assertEqual(browser_settings.debug_source.read(), "debug")
                browser_settings.debug_source.close()
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
                "load_from_settings",
                new=AsyncMock(return_value="orch"),
            ) as lfs_patch:
                result = await OrchestratorLoader.from_file(
                    path,
                    agent_id=uuid4(),
                    hub=hub,
                    logger=logger,
                    participant_id=uuid4(),
                    stack=stack,
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
                "load_from_settings",
                new=AsyncMock(return_value="orch"),
            ) as lfs_patch:
                result = await OrchestratorLoader.from_file(
                    path,
                    agent_id=uuid4(),
                    hub=hub,
                    logger=logger,
                    participant_id=uuid4(),
                    stack=stack,
                )

                self.assertEqual(result, "orch")
                lfs_patch.assert_awaited_once()
                settings = lfs_patch.call_args.args[0]
                self.assertTrue(
                    settings.call_options["chat_template_settings"][
                        "enable_thinking"
                    ]
                )
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
            memory_permanent=None,
            memory_recent=False,
            sentence_model_id=OrchestratorLoader.DEFAULT_SENTENCE_MODEL_ID,
            sentence_model_engine_config=None,
            sentence_model_max_tokens=500,
            sentence_model_overlap_size=125,
            sentence_model_window_size=250,
            json_config=None,
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
            result = await OrchestratorLoader.load_from_settings(
                settings,
                hub=hub,
                logger=logger,
                participant_id=uuid4(),
                stack=stack,
            )

            self.assertEqual(result, "orch")
            orch_patch.assert_called_once()
            model_patch.assert_called_once_with(hub, logger)
            mm_patch.assert_awaited_once()
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
            memory_permanent=None,
            memory_recent=False,
            sentence_model_id=OrchestratorLoader.DEFAULT_SENTENCE_MODEL_ID,
            sentence_model_engine_config=None,
            sentence_model_max_tokens=500,
            sentence_model_overlap_size=125,
            sentence_model_window_size=250,
            json_config={"value": {"type": "string", "description": "d"}},
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
            result = await OrchestratorLoader.load_from_settings(
                settings,
                hub=hub,
                logger=logger,
                participant_id=uuid4(),
                stack=stack,
            )

            self.assertEqual(result, "json_orch")
            json_patch.assert_called_once()
        await stack.aclose()


if __name__ == "__main__":
    main()
