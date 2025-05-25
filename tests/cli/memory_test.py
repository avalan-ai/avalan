from argparse import Namespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID
import numpy as np
import sys
import types
import importlib.machinery

# Stub faiss before importing the module under test
faiss_stub = types.ModuleType("faiss")
faiss_stub.IndexFlatL2 = MagicMock()
faiss_stub.__spec__ = importlib.machinery.ModuleSpec("faiss", loader=None)
sys.modules.setdefault("faiss", faiss_stub)

# Stub httpx before importing the module under test
httpx_stub = types.ModuleType("httpx")
httpx_stub.AsyncClient = object
httpx_stub.Response = object
httpx_stub.__spec__ = importlib.machinery.ModuleSpec("httpx", loader=None)
sys.modules.setdefault("httpx", httpx_stub)

# Stub markitdown before importing the module under test
md_stub = types.ModuleType("markitdown")
md_stub.MarkItDown = MagicMock()
md_stub.DocumentConverterResult = object
md_stub.__spec__ = importlib.machinery.ModuleSpec("markitdown", loader=None)
sys.modules.setdefault("markitdown", md_stub)

from avalan.cli.commands import memory as memory_cmds
from avalan.memory.permanent import MemoryType, VectorFunction
from avalan.memory.partitioner.text import TextPartition


class CliMemoryDocumentIndexTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.args = Namespace(
            model="m",
            source="file.txt",
            partition_max_tokens=10,
            partition_overlap=2,
            partition_window=5,
            partitioner="text",
            encoding="utf-8",
            language=None,
            dsn="dsn",
            participant="11111111-1111-1111-1111-111111111111",
            namespace="ns",
            identifier=None,
            display_partitions=1,
            no_display_partitions=False,
        )
        self.console = MagicMock()
        self.theme = MagicMock()
        self.theme._ = lambda s: s
        self.theme.icons = {}
        self.theme.memory_partitions.return_value = "panel"
        self.hub = MagicMock()
        self.logger = MagicMock()

    async def test_index_file(self):
        partition = TextPartition(data="d", embeddings=np.array([1.0]), total_tokens=1)
        manager = MagicMock()
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        model = MagicMock()
        load_cm = MagicMock()
        load_cm.__enter__.return_value = model
        load_cm.__exit__.return_value = False
        manager.load.return_value = load_cm

        memory_store = MagicMock()
        memory_store.append_with_partitions = AsyncMock()

        tp_inst = AsyncMock(return_value=[partition])

        with patch.object(memory_cmds, "get_model_settings", return_value={}) as gms_patch, \
             patch.object(memory_cmds, "ModelManager", return_value=manager) as mm_patch, \
             patch.object(memory_cmds.Path, "read_text", return_value="content") as read_patch, \
             patch.object(memory_cmds, "TextPartitioner", return_value=tp_inst) as tp_patch, \
             patch.object(memory_cmds.PgsqlRawMemory, "create_instance", AsyncMock(return_value=memory_store)) as mem_patch, \
             patch.object(memory_cmds, "model_display"):
            await memory_cmds.memory_document_index(
                self.args, self.console, self.theme, self.hub, self.logger
            )

        gms_patch.assert_called_once_with(
            self.args, self.hub, self.logger, self.args.model, is_sentence_transformer=True
        )
        tp_patch.assert_called_once_with(
            model,
            self.logger,
            max_tokens=self.args.partition_max_tokens,
            window_size=self.args.partition_window,
            overlap_size=self.args.partition_overlap,
        )
        tp_inst.assert_awaited_once_with("content")
        mem_patch.assert_awaited_once_with(dsn=self.args.dsn)
        memory_store.append_with_partitions.assert_awaited_once_with(
            self.args.namespace,
            UUID(self.args.participant),
            memory_type=MemoryType.FILE,
            data="content",
            identifier=str(memory_cmds.Path(self.args.source).resolve()),
            partitions=[partition],
            symbols={},
            model_id=self.args.model,
        )
        self.console.print.assert_called_once_with("panel")


class CliMemoryEmbeddingsTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.args = Namespace(
            model="m",
            compare=["c"],
            search=None,
            search_k=1,
            sort=None,
            partition=False,
            no_repl=False,
            quiet=False,
            display_partitions=1,
            no_display_partitions=False,
        )
        self.console = MagicMock()
        self.theme = MagicMock()
        self.theme._ = lambda s: s
        self.theme.icons = {"user_input": ">"}
        self.theme.memory_embeddings.return_value = "emb"
        self.theme.memory_embeddings_comparison.return_value = "cmp"
        self.hub = MagicMock()
        self.logger = MagicMock()

    async def test_embeddings_compare(self):
        emb1 = np.array([1.0, 0.0])
        emb2 = np.array([0.0, 1.0])
        manager = MagicMock()
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        model = AsyncMock(return_value=[emb1, emb2])
        model.token_count = MagicMock(return_value=2)
        load_cm = MagicMock()
        load_cm.__enter__.return_value = model
        load_cm.__exit__.return_value = False
        manager.load.return_value = load_cm

        with patch.object(memory_cmds, "get_input", return_value="text") as gi_patch, \
             patch.object(memory_cmds, "get_model_settings", return_value={}) as gms_patch, \
             patch.object(memory_cmds, "ModelManager", return_value=manager) as mm_patch, \
             patch.object(memory_cmds, "model_display"):
            await memory_cmds.memory_embeddings(
                self.args, self.console, self.theme, self.hub, self.logger
            )

        gi_patch.assert_called_once_with(
            self.console,
            self.theme.icons["user_input"] + " ",
            echo_stdin=not self.args.no_repl,
            is_quiet=self.args.quiet,
        )
        gms_patch.assert_called_once_with(
            self.args, self.hub, self.logger, self.args.model, is_sentence_transformer=True
        )
        model.assert_awaited_once_with(["text", *self.args.compare])
        self.assertEqual(len(self.console.print.call_args_list), 2)
        self.assertEqual(
            self.console.print.call_args_list[0].args[0],
            "emb",
        )
        self.assertEqual(
            self.console.print.call_args_list[1].args[0],
            "cmp",
        )


class CliMemorySearchTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.args = Namespace(
            model="m",
            dsn="dsn",
            participant="11111111-1111-1111-1111-111111111111",
            namespace="ns",
            function=VectorFunction.L2_DISTANCE,
            limit=2,
            partition_max_tokens=10,
            partition_overlap=2,
            partition_window=5,
            no_repl=False,
            quiet=False,
        )
        self.console = MagicMock()
        self.theme = MagicMock()
        self.theme._ = lambda s: s
        self.theme.icons = {"user_input": ">"}
        self.theme.memory_search_matches.return_value = "search"
        self.hub = MagicMock()
        self.logger = MagicMock()

    async def test_memory_search(self):
        partition = TextPartition(data="d", embeddings=np.array([1.0]), total_tokens=1)
        manager = MagicMock()
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        model = MagicMock()
        load_cm = MagicMock()
        load_cm.__enter__.return_value = model
        load_cm.__exit__.return_value = False
        manager.load.return_value = load_cm

        tp_inst = AsyncMock(return_value=[partition])

        memory_store = MagicMock()
        memory_store.search_memories = AsyncMock(return_value=["m"])

        with patch.object(memory_cmds, "get_input", return_value="query") as gi_patch, \
             patch.object(memory_cmds, "get_model_settings", return_value={}) as gms_patch, \
             patch.object(memory_cmds, "ModelManager", return_value=manager) as mm_patch, \
             patch.object(memory_cmds, "TextPartitioner", return_value=tp_inst) as tp_patch, \
             patch.object(memory_cmds.PgsqlRawMemory, "create_instance", AsyncMock(return_value=memory_store)) as mem_patch, \
             patch.object(memory_cmds, "model_display"):
            await memory_cmds.memory_search(
                self.args, self.console, self.theme, self.hub, self.logger
            )

        gi_patch.assert_called_once_with(
            self.console,
            self.theme.icons["user_input"] + " ",
            echo_stdin=not self.args.no_repl,
            is_quiet=self.args.quiet,
        )
        tp_patch.assert_called_once_with(
            model,
            self.logger,
            max_tokens=self.args.partition_max_tokens,
            window_size=self.args.partition_window,
            overlap_size=self.args.partition_overlap,
        )
        tp_inst.assert_awaited_once_with("query")
        mem_patch.assert_awaited_once_with(dsn=self.args.dsn)
        memory_store.search_memories.assert_awaited_once_with(
            search_partitions=[partition],
            participant_id=UUID(self.args.participant),
            namespace=self.args.namespace,
            function=self.args.function,
            limit=self.args.limit,
        )
        self.console.print.assert_called_once_with("search")

