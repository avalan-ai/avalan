from contextlib import AsyncExitStack
from logging import Logger
from tempfile import NamedTemporaryFile, TemporaryDirectory
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, call, patch
from uuid import uuid4

from avalan.agent import loader as loader_module
from avalan.agent.loader import OrchestratorLoader
from avalan.entities import PermanentMemoryStoreSettings
from avalan.model.hubs.huggingface import HuggingfaceHub


class OrchestratorLoaderCoverageTestCase(IsolatedAsyncioTestCase):
    async def test_lazy_model_and_partitioner_types_import_when_uncached(
        self,
    ) -> None:
        class LoadedSentenceTransformerModel:
            pass

        class LoadedTextPartitioner:
            pass

        def load_module(name: str) -> SimpleNamespace:
            if name == "avalan.model.nlp.sentence":
                return SimpleNamespace(
                    SentenceTransformerModel=LoadedSentenceTransformerModel
                )
            if name == "avalan.memory.partitioner.text":
                return SimpleNamespace(TextPartitioner=LoadedTextPartitioner)
            raise AssertionError(name)

        with (
            patch.object(loader_module, "SentenceTransformerModel", None),
            patch.object(loader_module, "TextPartitioner", None),
            patch.object(
                loader_module,
                "import_module",
                side_effect=load_module,
            ) as import_module,
        ):
            self.assertIs(
                OrchestratorLoader._sentence_transformer_model_type(),
                LoadedSentenceTransformerModel,
            )
            self.assertIs(
                OrchestratorLoader._text_partitioner_type(),
                LoadedTextPartitioner,
            )
            self.assertIs(
                OrchestratorLoader._sentence_transformer_model_type(),
                LoadedSentenceTransformerModel,
            )
            self.assertIs(
                OrchestratorLoader._text_partitioner_type(),
                LoadedTextPartitioner,
            )

        import_module.assert_has_calls(
            [
                call("avalan.model.nlp.sentence"),
                call("avalan.memory.partitioner.text"),
            ]
        )
        self.assertEqual(import_module.call_count, 2)

    async def test_lazy_pgsql_raw_memory_type_imports_when_uncached(
        self,
    ) -> None:
        class LoadedPgsqlRawMemory:
            pass

        with (
            patch.object(loader_module, "PgsqlRawMemory", None),
            patch.object(
                loader_module,
                "import_module",
                return_value=SimpleNamespace(
                    PgsqlRawMemory=LoadedPgsqlRawMemory
                ),
            ) as import_module,
        ):
            self.assertIs(
                OrchestratorLoader._pgsql_raw_memory_type(),
                LoadedPgsqlRawMemory,
            )
            self.assertIs(
                OrchestratorLoader._pgsql_raw_memory_type(),
                LoadedPgsqlRawMemory,
            )

        import_module.assert_called_once_with(
            "avalan.memory.permanent.pgsql.raw"
        )

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
                debug_source = bts_patch.call_args.kwargs["debug_source"]
                try:
                    self.assertEqual(debug_source.name, debug_file.name)
                finally:
                    debug_source.close()
            await stack.aclose()
