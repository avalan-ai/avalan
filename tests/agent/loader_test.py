from avalan.agent.loader import Loader
from avalan.model.hubs.huggingface import HuggingfaceHub
from contextlib import AsyncExitStack
from logging import Logger
from tempfile import NamedTemporaryFile, TemporaryDirectory
from os import chmod
from uuid import uuid4
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import AsyncMock, MagicMock, patch

class LoaderFromFileTestCase(IsolatedAsyncioTestCase):
    async def test_file_not_found(self):
        stack = AsyncExitStack()
        with self.assertRaises(FileNotFoundError):
            await Loader.from_file(
                "missing.toml",
                agent_id=uuid4(),
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=stack,
            )
        await stack.aclose()

    async def test_permission_error(self):
        with NamedTemporaryFile(delete=False) as tmp:
            path = tmp.name
        chmod(path, 0)
        stack = AsyncExitStack()
        try:
            with self.assertRaises(PermissionError):
                await Loader.from_file(
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
            event_manager = MagicMock()

            with patch("avalan.agent.loader.SentenceTransformerModel", return_value=sentence_model), \
                 patch("avalan.agent.loader.TextPartitioner"), \
                 patch("avalan.agent.loader.MemoryManager.create_instance", new=AsyncMock(return_value=memory)) as mm_patch, \
                 patch("avalan.agent.loader.ModelManager", return_value=model_manager) as model_patch, \
                 patch("avalan.agent.loader.DefaultOrchestrator", return_value="orch") as orch_patch, \
                 patch("avalan.agent.loader.ToolManager.create_instance", return_value=tool) as tool_patch, \
                 patch("avalan.agent.loader.EventManager", return_value=event_manager):
                result = await Loader.from_file(
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
                tool_patch.assert_called_once_with(enable_tools=None)
                mm_patch.assert_awaited_once()
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

            with patch("avalan.agent.loader.SentenceTransformerModel", return_value=sentence_model), \
                 patch("avalan.agent.loader.TextPartitioner"), \
                 patch("avalan.agent.loader.MemoryManager.create_instance", new=AsyncMock(return_value=memory)), \
                 patch("avalan.agent.loader.ModelManager", return_value=model_manager), \
                 patch.object(Loader, "_load_json_orchestrator", return_value="json_orch") as json_patch, \
                 patch("avalan.agent.loader.ToolManager.create_instance", return_value=tool), \
                 patch("avalan.agent.loader.EventManager", return_value=event_manager):
                result = await Loader.from_file(
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
                await Loader.from_file(
                    path,
                    agent_id=uuid4(),
                    hub=MagicMock(spec=HuggingfaceHub),
                    logger=MagicMock(spec=Logger),
                    participant_id=uuid4(),
                    stack=stack,
                )
            await stack.aclose()

if __name__ == "__main__":
    main()
