from avalan.agent.loader import OrchestratorLoader
from avalan.entities import PermanentMemoryStoreSettings
from avalan.model.hubs.huggingface import HuggingfaceHub
from contextlib import AsyncExitStack
from logging import Logger
from tempfile import NamedTemporaryFile, TemporaryDirectory
from uuid import uuid4
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch


class OrchestratorLoaderCoverageTestCase(IsolatedAsyncioTestCase):
    async def test_run_chat_memory_and_debug_source(self) -> None:
        with NamedTemporaryFile() as debug_file, TemporaryDirectory() as tmp:
            config = f"""
[agent]
role = \"assistant\"

[engine]
uri = \"ai://local/model\"

[run.chat]
enable_thinking = true

[memory.permanent]
ns1 = \"dsn1\"
ns2 = \"dsn2\"

[tool.browser]
debug_source = \"{debug_file.name}\"
"""
            path = f"{tmp}/agent.toml"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config)

            hub = MagicMock(spec=HuggingfaceHub)
            logger = MagicMock(spec=Logger)
            stack = AsyncExitStack()

            with (
                patch.object(
                    OrchestratorLoader,
                    "from_settings",
                    new=AsyncMock(return_value="orch"),
                ) as from_settings,
                patch("avalan.agent.loader.BrowserToolSettings") as bts_patch,
            ):
                loader = OrchestratorLoader(
                    hub=hub,
                    logger=logger,
                    participant_id=uuid4(),
                    stack=stack,
                )
                result = await loader.from_file(path, agent_id=uuid4())

                self.assertEqual(result, "orch")
                from_settings.assert_awaited_once()
                settings = from_settings.call_args.args[0]
                self.assertTrue(
                    settings.call_options["chat_settings"]["enable_thinking"]
                )
                self.assertEqual(
                    settings.permanent_memory,
                    {
                        "ns1": PermanentMemoryStoreSettings(
                            dsn="dsn1", description=None
                        ),
                        "ns2": PermanentMemoryStoreSettings(
                            dsn="dsn2", description=None
                        ),
                    },
                )
                bts_patch.assert_called_once()
                self.assertEqual(
                    bts_patch.call_args.kwargs["debug_source"].name,
                    debug_file.name,
                )
            await stack.aclose()
