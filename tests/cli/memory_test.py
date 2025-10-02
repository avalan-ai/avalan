from avalan.cli.commands import memory as memory_cmds
from avalan.entities import DistanceType, Modality
from avalan.memory.permanent import MemoryType, VectorFunction
from avalan.memory.partitioner.text import TextPartition
from argparse import Namespace
from unittest import IsolatedAsyncioTestCase, TestCase
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
            title="Knowledge Base",
            description="Knowledge base",
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
        partition = TextPartition(
            data="d", embeddings=np.array([1.0]), total_tokens=1
        )
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

        manager.parse_uri = MagicMock(return_value="engine_uri")
        with (
            patch.object(
                memory_cmds, "get_model_settings", return_value={}
            ) as gms_patch,
            patch.object(memory_cmds, "ModelManager", return_value=manager),
            patch.object(
                memory_cmds.Path, "read_text", return_value="content"
            ),
            patch.object(
                memory_cmds, "TextPartitioner", return_value=tp_inst
            ) as tp_patch,
            patch.object(
                memory_cmds.PgsqlRawMemory,
                "create_instance",
                AsyncMock(return_value=memory_store),
            ) as mem_patch,
            patch.object(memory_cmds, "model_display"),
        ):
            await memory_cmds.memory_document_index(
                self.args, self.console, self.theme, self.hub, self.logger
            )

        manager.parse_uri.assert_called_once_with(self.args.model)
        gms_patch.assert_called_once_with(
            self.args,
            self.hub,
            self.logger,
            "engine_uri",
            modality=Modality.EMBEDDING,
        )
        tp_patch.assert_called_once_with(
            model,
            self.logger,
            max_tokens=self.args.partition_max_tokens,
            window_size=self.args.partition_window,
            overlap_size=self.args.partition_overlap,
        )
        tp_inst.assert_awaited_once_with("content")
        mem_patch.assert_awaited_once_with(
            dsn=self.args.dsn, logger=self.logger
        )
        memory_store.append_with_partitions.assert_awaited_once_with(
            self.args.namespace,
            UUID(self.args.participant),
            memory_type=MemoryType.FILE,
            data="content",
            identifier=str(memory_cmds.Path(self.args.source).resolve()),
            partitions=[partition],
            symbols={},
            model_id=self.args.model,
            title=self.args.title,
            description=self.args.description,
        )
        self.console.print.assert_called_once_with("panel")

    async def test_index_url(self):
        self.args.source = "https://example.com"
        partition = TextPartition(
            data="d", embeddings=np.array([1.0]), total_tokens=1
        )
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
        document = types.SimpleNamespace(
            title="Fetched title",
            description="desc",
            markdown="content",
        )
        memory_source = MagicMock()
        memory_source.__aenter__ = AsyncMock(return_value=memory_source)
        memory_source.__aexit__ = AsyncMock(return_value=False)
        memory_source.fetch = AsyncMock(return_value=document)

        with (
            patch.object(memory_cmds, "get_model_settings", return_value={}),
            patch.object(memory_cmds, "ModelManager", return_value=manager),
            patch.object(
                memory_cmds, "MemorySource", return_value=memory_source
            ) as ms_patch,
            patch.object(
                memory_cmds, "TextPartitioner", return_value=tp_inst
            ) as tp_patch,
            patch.object(
                memory_cmds.PgsqlRawMemory,
                "create_instance",
                AsyncMock(return_value=memory_store),
            ) as mem_patch,
            patch.object(memory_cmds.Path, "read_text") as read_patch,
            patch.object(memory_cmds, "model_display"),
        ):
            await memory_cmds.memory_document_index(
                self.args, self.console, self.theme, self.hub, self.logger
            )

        read_patch.assert_not_called()
        ms_patch.assert_called_once_with()
        memory_source.fetch.assert_awaited_once_with(self.args.source)
        tp_patch.assert_called_once_with(
            model,
            self.logger,
            max_tokens=self.args.partition_max_tokens,
            window_size=self.args.partition_window,
            overlap_size=self.args.partition_overlap,
        )
        tp_inst.assert_awaited_once_with("content")
        mem_patch.assert_awaited_once_with(
            dsn=self.args.dsn, logger=self.logger
        )
        memory_store.append_with_partitions.assert_awaited_once_with(
            self.args.namespace,
            UUID(self.args.participant),
            memory_type=MemoryType.URL,
            data="content",
            identifier=self.args.source,
            partitions=[partition],
            symbols={},
            model_id=self.args.model,
            title=self.args.title,
            description=self.args.description,
        )
        self.console.print.assert_called_once_with("panel")

    async def test_index_url_transform_called(self):
        self.args.source = "https://example.com"
        partition = TextPartition(
            data="d", embeddings=np.array([1.0]), total_tokens=1
        )
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
        document = types.SimpleNamespace(
            description="Fetched description", markdown="html"
        )
        memory_source = MagicMock()
        memory_source.__aenter__ = AsyncMock(return_value=memory_source)
        memory_source.__aexit__ = AsyncMock(return_value=False)
        memory_source.fetch = AsyncMock(return_value=document)
        self.args.description = None

        with (
            patch.object(memory_cmds, "get_model_settings", return_value={}),
            patch.object(memory_cmds, "ModelManager", return_value=manager),
            patch.object(
                memory_cmds, "MemorySource", return_value=memory_source
            ),
            patch.object(memory_cmds, "TextPartitioner", return_value=tp_inst),
            patch.object(
                memory_cmds.PgsqlRawMemory,
                "create_instance",
                AsyncMock(return_value=memory_store),
            ),
            patch.object(memory_cmds, "model_display"),
        ):
            await memory_cmds.memory_document_index(
                self.args, self.console, self.theme, self.hub, self.logger
            )

        memory_source.fetch.assert_awaited_once_with(self.args.source)
        tp_inst.assert_awaited_once_with("html")
        call = memory_store.append_with_partitions.await_args
        self.assertEqual(call.kwargs["description"], "Fetched description")
        self.console.print.assert_called_once()

    async def test_index_url_missing_title_defaults_to_netloc(self):
        self.args.source = "https://example.com/path"
        self.args.title = None
        partition = TextPartition(
            data="d", embeddings=np.array([1.0]), total_tokens=1
        )
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
        document = types.SimpleNamespace(
            title=None,
            description="desc",
            markdown="content",
        )
        memory_source = MagicMock()
        memory_source.__aenter__ = AsyncMock(return_value=memory_source)
        memory_source.__aexit__ = AsyncMock(return_value=False)
        memory_source.fetch = AsyncMock(return_value=document)

        with (
            patch.object(memory_cmds, "get_model_settings", return_value={}),
            patch.object(memory_cmds, "ModelManager", return_value=manager),
            patch.object(
                memory_cmds, "MemorySource", return_value=memory_source
            ),
            patch.object(memory_cmds, "TextPartitioner", return_value=tp_inst),
            patch.object(
                memory_cmds.PgsqlRawMemory,
                "create_instance",
                AsyncMock(return_value=memory_store),
            ),
            patch.object(memory_cmds, "model_display"),
        ):
            await memory_cmds.memory_document_index(
                self.args, self.console, self.theme, self.hub, self.logger
            )

        call = memory_store.append_with_partitions.await_args
        self.assertEqual(call.kwargs["title"], "example.com")

    async def test_index_pdf_discovers_metadata_without_cli_options(self):
        self.args.source = "doc.pdf"
        self.args.title = None
        self.args.description = None
        partition = TextPartition(
            data="d", embeddings=np.array([1.0]), total_tokens=1
        )
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
        document = types.SimpleNamespace(
            title="PDF Title",
            description="PDF Description",
            markdown="pdf markdown",
        )
        memory_source = MagicMock()
        memory_source.__aenter__ = AsyncMock(return_value=memory_source)
        memory_source.__aexit__ = AsyncMock(return_value=False)
        memory_source.from_bytes = AsyncMock(return_value=document)

        with (
            patch.object(memory_cmds, "get_model_settings", return_value={}),
            patch.object(memory_cmds, "ModelManager", return_value=manager),
            patch.object(
                memory_cmds.Path, "read_bytes", return_value=b"pdf"
            ) as read_bytes,
            patch.object(
                memory_cmds.Path, "read_text", MagicMock()
            ) as read_text,
            patch.object(
                memory_cmds, "MemorySource", return_value=memory_source
            ) as ms_patch,
            patch.object(memory_cmds, "TextPartitioner", return_value=tp_inst),
            patch.object(
                memory_cmds.PgsqlRawMemory,
                "create_instance",
                AsyncMock(return_value=memory_store),
            ),
            patch.object(memory_cmds, "model_display"),
        ):
            await memory_cmds.memory_document_index(
                self.args, self.console, self.theme, self.hub, self.logger
            )

        read_bytes.assert_called_once()
        read_text.assert_not_called()
        ms_patch.assert_called_once_with()
        memory_source.from_bytes.assert_awaited_once_with(
            memory_cmds.Path(self.args.source).resolve().as_uri(),
            "application/pdf",
            b"pdf",
        )
        tp_inst.assert_awaited_once_with("pdf markdown")
        call = memory_store.append_with_partitions.await_args
        self.assertEqual(call.kwargs["title"], "PDF Title")
        self.assertEqual(call.kwargs["description"], "PDF Description")

    async def test_index_pdf_metadata_missing_falls_back(self):
        self.args.source = "doc.pdf"
        self.args.title = None
        self.args.description = None
        partition = TextPartition(
            data="d", embeddings=np.array([1.0]), total_tokens=1
        )
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
        document = types.SimpleNamespace(
            title=None,
            description=None,
            markdown="pdf markdown",
        )
        memory_source = MagicMock()
        memory_source.__aenter__ = AsyncMock(return_value=memory_source)
        memory_source.__aexit__ = AsyncMock(return_value=False)
        memory_source.from_bytes = AsyncMock(return_value=document)

        with (
            patch.object(memory_cmds, "get_model_settings", return_value={}),
            patch.object(memory_cmds, "ModelManager", return_value=manager),
            patch.object(memory_cmds.Path, "read_bytes", return_value=b"pdf"),
            patch.object(memory_cmds.Path, "read_text", MagicMock()),
            patch.object(
                memory_cmds, "MemorySource", return_value=memory_source
            ),
            patch.object(memory_cmds, "TextPartitioner", return_value=tp_inst),
            patch.object(
                memory_cmds.PgsqlRawMemory,
                "create_instance",
                AsyncMock(return_value=memory_store),
            ),
            patch.object(memory_cmds, "model_display"),
        ):
            await memory_cmds.memory_document_index(
                self.args, self.console, self.theme, self.hub, self.logger
            )

        call = memory_store.append_with_partitions.await_args
        self.assertEqual(call.kwargs["title"], "doc.pdf")
        self.assertIsNone(call.kwargs["description"])

    async def test_index_markdown_discovers_metadata_without_cli_options(self):
        self.args.source = "doc.md"
        self.args.title = None
        self.args.description = None
        partition = TextPartition(
            data="d", embeddings=np.array([1.0]), total_tokens=1
        )
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
        document = types.SimpleNamespace(
            title="Markdown Title",
            description="Markdown Description",
            markdown="md markdown",
        )
        memory_source = MagicMock()
        memory_source.__aenter__ = AsyncMock(return_value=memory_source)
        memory_source.__aexit__ = AsyncMock(return_value=False)
        memory_source.from_bytes = AsyncMock(return_value=document)

        with (
            patch.object(memory_cmds, "get_model_settings", return_value={}),
            patch.object(memory_cmds, "ModelManager", return_value=manager),
            patch.object(memory_cmds.Path, "read_bytes", return_value=b"md"),
            patch.object(memory_cmds.Path, "read_text", MagicMock()),
            patch.object(
                memory_cmds, "MemorySource", return_value=memory_source
            ),
            patch.object(memory_cmds, "TextPartitioner", return_value=tp_inst),
            patch.object(
                memory_cmds.PgsqlRawMemory,
                "create_instance",
                AsyncMock(return_value=memory_store),
            ),
            patch.object(memory_cmds, "model_display"),
        ):
            await memory_cmds.memory_document_index(
                self.args, self.console, self.theme, self.hub, self.logger
            )

        call = memory_store.append_with_partitions.await_args
        self.assertEqual(call.kwargs["title"], "Markdown Title")
        self.assertEqual(call.kwargs["description"], "Markdown Description")

    async def test_index_markdown_metadata_missing_falls_back(self):
        self.args.source = "doc.md"
        self.args.title = None
        self.args.description = None
        partition = TextPartition(
            data="d", embeddings=np.array([1.0]), total_tokens=1
        )
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
        document = types.SimpleNamespace(
            title=None,
            description=None,
            markdown="md markdown",
        )
        memory_source = MagicMock()
        memory_source.__aenter__ = AsyncMock(return_value=memory_source)
        memory_source.__aexit__ = AsyncMock(return_value=False)
        memory_source.from_bytes = AsyncMock(return_value=document)

        with (
            patch.object(memory_cmds, "get_model_settings", return_value={}),
            patch.object(memory_cmds, "ModelManager", return_value=manager),
            patch.object(memory_cmds.Path, "read_bytes", return_value=b"md"),
            patch.object(memory_cmds.Path, "read_text", MagicMock()),
            patch.object(
                memory_cmds, "MemorySource", return_value=memory_source
            ),
            patch.object(memory_cmds, "TextPartitioner", return_value=tp_inst),
            patch.object(
                memory_cmds.PgsqlRawMemory,
                "create_instance",
                AsyncMock(return_value=memory_store),
            ),
            patch.object(memory_cmds, "model_display"),
        ):
            await memory_cmds.memory_document_index(
                self.args, self.console, self.theme, self.hub, self.logger
            )

        call = memory_store.append_with_partitions.await_args
        self.assertEqual(call.kwargs["title"], "doc.md")
        self.assertIsNone(call.kwargs["description"])

    async def test_index_text_defaults_title_to_filename(self):
        self.args.source = "doc.txt"
        self.args.title = None
        partition = TextPartition(
            data="d", embeddings=np.array([1.0]), total_tokens=1
        )
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

        with (
            patch.object(memory_cmds, "get_model_settings", return_value={}),
            patch.object(memory_cmds, "ModelManager", return_value=manager),
            patch.object(
                memory_cmds.Path, "read_text", return_value="content"
            ),
            patch.object(memory_cmds, "TextPartitioner", return_value=tp_inst),
            patch.object(
                memory_cmds.PgsqlRawMemory,
                "create_instance",
                AsyncMock(return_value=memory_store),
            ),
            patch.object(memory_cmds, "model_display"),
        ):
            await memory_cmds.memory_document_index(
                self.args, self.console, self.theme, self.hub, self.logger
            )

        tp_inst.assert_awaited_once_with("content")
        call = memory_store.append_with_partitions.await_args
        self.assertEqual(call.kwargs["title"], "doc.txt")

    async def test_index_html_discovers_metadata_without_cli_options(self):
        self.args.source = "doc.html"
        self.args.title = None
        self.args.description = None
        partition = TextPartition(
            data="d", embeddings=np.array([1.0]), total_tokens=1
        )
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
        document = types.SimpleNamespace(
            title="HTML Title",
            description="HTML Description",
            markdown="html markdown",
        )
        memory_source = MagicMock()
        memory_source.__aenter__ = AsyncMock(return_value=memory_source)
        memory_source.__aexit__ = AsyncMock(return_value=False)
        memory_source.from_bytes = AsyncMock(return_value=document)

        with (
            patch.object(memory_cmds, "get_model_settings", return_value={}),
            patch.object(memory_cmds, "ModelManager", return_value=manager),
            patch.object(memory_cmds.Path, "read_bytes", return_value=b"html"),
            patch.object(memory_cmds.Path, "read_text", MagicMock()),
            patch.object(
                memory_cmds, "MemorySource", return_value=memory_source
            ),
            patch.object(memory_cmds, "TextPartitioner", return_value=tp_inst),
            patch.object(
                memory_cmds.PgsqlRawMemory,
                "create_instance",
                AsyncMock(return_value=memory_store),
            ),
            patch.object(memory_cmds, "model_display"),
        ):
            await memory_cmds.memory_document_index(
                self.args, self.console, self.theme, self.hub, self.logger
            )

        call = memory_store.append_with_partitions.await_args
        self.assertEqual(call.kwargs["title"], "HTML Title")
        self.assertEqual(call.kwargs["description"], "HTML Description")

    async def test_index_html_metadata_missing_falls_back(self):
        self.args.source = "doc.html"
        self.args.title = None
        self.args.description = None
        partition = TextPartition(
            data="d", embeddings=np.array([1.0]), total_tokens=1
        )
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
        document = types.SimpleNamespace(
            title=None,
            description=None,
            markdown="html markdown",
        )
        memory_source = MagicMock()
        memory_source.__aenter__ = AsyncMock(return_value=memory_source)
        memory_source.__aexit__ = AsyncMock(return_value=False)
        memory_source.from_bytes = AsyncMock(return_value=document)

        with (
            patch.object(memory_cmds, "get_model_settings", return_value={}),
            patch.object(memory_cmds, "ModelManager", return_value=manager),
            patch.object(memory_cmds.Path, "read_bytes", return_value=b"html"),
            patch.object(memory_cmds.Path, "read_text", MagicMock()),
            patch.object(
                memory_cmds, "MemorySource", return_value=memory_source
            ),
            patch.object(memory_cmds, "TextPartitioner", return_value=tp_inst),
            patch.object(
                memory_cmds.PgsqlRawMemory,
                "create_instance",
                AsyncMock(return_value=memory_store),
            ),
            patch.object(memory_cmds, "model_display"),
        ):
            await memory_cmds.memory_document_index(
                self.args, self.console, self.theme, self.hub, self.logger
            )

        call = memory_store.append_with_partitions.await_args
        self.assertEqual(call.kwargs["title"], "doc.html")
        self.assertIsNone(call.kwargs["description"])

    async def test_index_code_partitioner(self):
        self.args.partitioner = "code"
        partition = types.SimpleNamespace(data="d")
        manager = MagicMock()
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        model = AsyncMock()
        model.token_count = MagicMock(return_value=1)
        model.return_value = np.array([1.0])
        load_cm = MagicMock()
        load_cm.__enter__.return_value = model
        load_cm.__exit__.return_value = False
        manager.load.return_value = load_cm

        memory_store = MagicMock()
        memory_store.append_with_partitions = AsyncMock()

        cp_inst = MagicMock()
        with (
            patch.object(memory_cmds, "get_model_settings", return_value={}),
            patch.object(memory_cmds, "ModelManager", return_value=manager),
            patch.object(memory_cmds.Path, "read_text", return_value="code"),
            patch.object(
                memory_cmds, "CodePartitioner", return_value=cp_inst
            ) as cp_patch,
            patch.object(
                memory_cmds,
                "to_thread",
                AsyncMock(return_value=([partition], None)),
            ) as tt_patch,
            patch.object(
                memory_cmds.PgsqlRawMemory,
                "create_instance",
                AsyncMock(return_value=memory_store),
            ) as mem_patch,
            patch.object(memory_cmds, "model_display"),
        ):
            await memory_cmds.memory_document_index(
                self.args, self.console, self.theme, self.hub, self.logger
            )

        cp_patch.assert_called_once_with(self.logger)
        tt_patch.assert_awaited_once_with(
            cp_inst.partition,
            self.args.language or "python",
            "code",
            self.args.encoding,
            self.args.partition_max_tokens,
        )
        mem_patch.assert_awaited_once_with(
            dsn=self.args.dsn, logger=self.logger
        )
        call = memory_store.append_with_partitions.await_args
        self.assertEqual(call.kwargs["memory_type"], MemoryType.CODE)
        self.assertEqual(len(call.kwargs["partitions"]), 1)


class CliMemoryEmbeddingsTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.args = Namespace(
            model="m",
            compare=["c"],
            search=None,
            search_k=1,
            sort=None,
            partition=False,
            partition_max_tokens=10,
            partition_window=5,
            partition_overlap=2,
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
        self.theme.memory_embeddings_search.return_value = "search"
        self.theme.memory_partitions.return_value = "parts"
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

        with (
            patch.object(
                memory_cmds, "get_input", return_value="text"
            ) as gi_patch,
            patch.object(
                memory_cmds, "get_model_settings", return_value={}
            ) as gms_patch,
            patch.object(
                memory_cmds, "ModelManager", return_value=manager
            ) as mm_patch,
            patch.object(memory_cmds, "model_display"),
        ):
            await memory_cmds.memory_embeddings(
                self.args, self.console, self.theme, self.hub, self.logger
            )

        gi_patch.assert_called_once_with(
            self.console,
            self.theme.icons["user_input"] + " ",
            echo_stdin=not self.args.no_repl,
            is_quiet=self.args.quiet,
            tty_path="/dev/tty",
        )
        mm_patch.parse_uri.assert_called_once_with(self.args.model)
        gms_patch.assert_called_once_with(
            self.args,
            self.hub,
            self.logger,
            mm_patch.parse_uri.return_value,
            modality=Modality.EMBEDDING,
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

    async def test_embeddings_no_input(self):
        self.args.partition = True
        with (
            patch.object(
                memory_cmds, "get_input", return_value=None
            ) as gi_patch,
            patch.object(memory_cmds, "ModelManager"),
            patch.object(memory_cmds, "get_model_settings", return_value={}),
            patch.object(memory_cmds, "model_display"),
        ):
            await memory_cmds.memory_embeddings(
                self.args, self.console, self.theme, self.hub, self.logger
            )

        gi_patch.assert_called_once()
        self.console.print.assert_not_called()

    async def test_embeddings_search(self):
        self.args.search = ["q"]
        self.args.compare = None
        self.args.sort = DistanceType.L1
        emb = np.array([1.0, 0.0])
        search_emb = np.array([0.5, 0.5])
        manager = MagicMock()
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        model = AsyncMock(side_effect=[emb, [search_emb]])
        model.token_count = MagicMock(return_value=2)
        load_cm = MagicMock()
        load_cm.__enter__.return_value = model
        load_cm.__exit__.return_value = False
        manager.load.return_value = load_cm

        index = MagicMock()
        index.search = MagicMock(
            return_value=(np.array([[0.1]]), np.array([[0]]))
        )
        index.add = MagicMock()

        with (
            patch.object(
                memory_cmds, "get_input", return_value="text"
            ) as gi_patch,
            patch.object(memory_cmds, "get_model_settings", return_value={}),
            patch.object(memory_cmds, "ModelManager", return_value=manager),
            patch.object(
                memory_cmds, "IndexFlatL2", return_value=index
            ) as idx_patch,
            patch.object(memory_cmds, "model_display"),
        ):
            await memory_cmds.memory_embeddings(
                self.args, self.console, self.theme, self.hub, self.logger
            )

        gi_patch.assert_called_once()
        idx_patch.assert_called_once_with(emb.shape[0])
        index.add.assert_called()
        index.search.assert_called()
        self.assertEqual(
            [c.args[0] for c in self.console.print.call_args_list],
            ["emb", "search"],
        )

    async def test_embeddings_search_with_partitioner(self):
        self.args.search = ["q"]
        self.args.compare = None
        self.args.partition = True
        emb = np.array([1.0])
        partition = TextPartition(data="d", embeddings=emb, total_tokens=1)
        manager = MagicMock()
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        model = AsyncMock(side_effect=[emb, emb])
        model.token_count = MagicMock(return_value=1)
        load_cm = MagicMock()
        load_cm.__enter__.return_value = model
        load_cm.__exit__.return_value = False
        manager.load.return_value = load_cm

        tp_inst = AsyncMock(return_value=[partition])
        index = MagicMock()
        index.search = MagicMock(
            return_value=(np.array([[0.1]]), np.array([[0]]))
        )
        index.add = MagicMock()

        with (
            patch.object(memory_cmds, "get_input", return_value="text"),
            patch.object(memory_cmds, "get_model_settings", return_value={}),
            patch.object(memory_cmds, "ModelManager", return_value=manager),
            patch.object(
                memory_cmds, "TextPartitioner", return_value=tp_inst
            ) as tp_patch,
            patch.object(memory_cmds, "IndexFlatL2", return_value=index),
            patch.object(memory_cmds, "model_display"),
        ):
            await memory_cmds.memory_embeddings(
                self.args, self.console, self.theme, self.hub, self.logger
            )

        tp_patch.assert_called_once_with(
            model,
            self.logger,
            max_tokens=self.args.partition_max_tokens,
            window_size=self.args.partition_window,
            overlap_size=self.args.partition_overlap,
        )
        self.console.print.assert_any_call(
            self.theme.memory_partitions.return_value
        )

    async def test_embeddings_search_skips_empty_match(self):
        self.args.search = ["q"]
        self.args.compare = None
        index = MagicMock()
        index.search = MagicMock(
            return_value=(np.array([[0.1]]), np.array([[1]]))
        )
        index.add = MagicMock()
        manager = MagicMock()
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        model = AsyncMock(side_effect=[np.array([1.0]), [np.array([0.5])]])
        model.token_count = MagicMock(return_value=1)
        load_cm = MagicMock()
        load_cm.__enter__.return_value = model
        load_cm.__exit__.return_value = False
        manager.load.return_value = load_cm

        with (
            patch.object(memory_cmds, "get_input", return_value="text"),
            patch.object(memory_cmds, "get_model_settings", return_value={}),
            patch.object(memory_cmds, "ModelManager", return_value=manager),
            patch.object(memory_cmds, "IndexFlatL2", return_value=index),
            patch.object(memory_cmds, "model_display"),
        ):
            await memory_cmds.memory_embeddings(
                self.args, self.console, self.theme, self.hub, self.logger
            )

        # search result skipped -> only embeddings output printed
        self.assertEqual(
            [c.args[0] for c in self.console.print.call_args_list],
            ["emb", "search"],
        )

    async def test_embeddings_with_partitioner_display(self):
        self.args.compare = None
        self.args.partition = True
        emb = np.array([1.0])
        partition = TextPartition(data="d", embeddings=emb, total_tokens=1)
        manager = MagicMock()
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        model = AsyncMock(return_value=emb)
        model.token_count = MagicMock(return_value=1)
        load_cm = MagicMock()
        load_cm.__enter__.return_value = model
        load_cm.__exit__.return_value = False
        manager.load.return_value = load_cm
        tp_inst = AsyncMock(return_value=[partition])
        with (
            patch.object(memory_cmds, "get_input", return_value="text") as gi,
            patch.object(memory_cmds, "get_model_settings", return_value={}),
            patch.object(memory_cmds, "ModelManager", return_value=manager),
            patch.object(
                memory_cmds, "TextPartitioner", return_value=tp_inst
            ) as tp,
            patch.object(memory_cmds, "model_display"),
        ):
            await memory_cmds.memory_embeddings(
                self.args, self.console, self.theme, self.hub, self.logger
            )

        gi.assert_called_once()
        tp.assert_called_once_with(
            model,
            self.logger,
            max_tokens=self.args.partition_max_tokens,
            window_size=self.args.partition_window,
            overlap_size=self.args.partition_overlap,
        )
        tp_inst.assert_awaited_once_with("text")
        self.console.print.assert_called_once_with(
            self.theme.memory_partitions.return_value
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
        partition = TextPartition(
            data="d", embeddings=np.array([1.0]), total_tokens=1
        )
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

        with (
            patch.object(
                memory_cmds, "get_input", return_value="query"
            ) as gi_patch,
            patch.object(memory_cmds, "get_model_settings", return_value={}),
            patch.object(memory_cmds, "ModelManager", return_value=manager),
            patch.object(
                memory_cmds, "TextPartitioner", return_value=tp_inst
            ) as tp_patch,
            patch.object(
                memory_cmds.PgsqlRawMemory,
                "create_instance",
                AsyncMock(return_value=memory_store),
            ) as mem_patch,
            patch.object(memory_cmds, "model_display"),
        ):
            await memory_cmds.memory_search(
                self.args, self.console, self.theme, self.hub, self.logger
            )

        gi_patch.assert_called_once_with(
            self.console,
            self.theme.icons["user_input"] + " ",
            echo_stdin=not self.args.no_repl,
            is_quiet=self.args.quiet,
            tty_path="/dev/tty",
        )
        tp_patch.assert_called_once_with(
            model,
            self.logger,
            max_tokens=self.args.partition_max_tokens,
            window_size=self.args.partition_window,
            overlap_size=self.args.partition_overlap,
        )
        tp_inst.assert_awaited_once_with("query")
        mem_patch.assert_awaited_once_with(
            dsn=self.args.dsn, logger=self.logger
        )
        memory_store.search_memories.assert_awaited_once_with(
            search_partitions=[partition],
            participant_id=UUID(self.args.participant),
            namespace=self.args.namespace,
            function=self.args.function,
            limit=self.args.limit,
        )
        self.console.print.assert_called_once_with("search")

    async def test_memory_search_no_input(self):
        with (
            patch.object(
                memory_cmds, "get_input", return_value=None
            ) as gi_patch,
            patch.object(memory_cmds, "ModelManager") as mm_patch,
        ):
            await memory_cmds.memory_search(
                self.args, self.console, self.theme, self.hub, self.logger
            )

        gi_patch.assert_called_once()
        mm_patch.assert_not_called()
        self.console.print.assert_not_called()


class CliMemoryDocumentTransformTestCase(TestCase):
    def test_transform_helper_converts_bytes(self) -> None:
        transform_code = memory_cmds.memory_document_index.__code__.co_consts[
            3
        ]
        transform = types.FunctionType(
            transform_code, memory_cmds.memory_document_index.__globals__
        )
        convert_stream = MagicMock(return_value="converted")
        mark_it_down = MagicMock(convert_stream=convert_stream)

        with patch.object(
            memory_cmds, "MarkItDown", return_value=mark_it_down
        ):
            result = transform(b"html")

        self.assertEqual(result, "converted")
        convert_stream.assert_called_once()
        argument = convert_stream.call_args.args[0]
        self.assertEqual(argument.getvalue(), b"html")
