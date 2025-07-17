import sys
import types
import importlib.machinery
from unittest.mock import MagicMock

# Stub elasticsearch before importing the module under test
es_stub = types.ModuleType("elasticsearch")
es_stub.AsyncElasticsearch = MagicMock()
es_stub.__spec__ = importlib.machinery.ModuleSpec("elasticsearch", loader=None)
sys.modules.setdefault("elasticsearch", es_stub)

from avalan.memory.partitioner.text import TextPartition  # noqa: E402
from avalan.memory.permanent import MemoryType  # noqa: E402
from avalan.memory.permanent.elasticsearch.raw import (  # noqa: E402
    ElasticsearchRawMemory,
)
from datetime import datetime, timezone  # noqa: E402
import numpy as np  # noqa: E402
from uuid import UUID, uuid4  # noqa: E402
from unittest import IsolatedAsyncioTestCase  # noqa: E402
from unittest.mock import AsyncMock, patch  # noqa: E402


class ElasticsearchRawMemoryTestCase(IsolatedAsyncioTestCase):
    async def test_create_instance(self):
        client = MagicMock()
        memory = await ElasticsearchRawMemory.create_instance(
            index="idx", logger=MagicMock(), es_client=client
        )
        self.assertIsInstance(memory, ElasticsearchRawMemory)
        self.assertIs(memory._client, client)

    async def test_create_instance_no_client(self):
        client = MagicMock()
        with patch(
            "avalan.memory.permanent.elasticsearch.raw.AsyncElasticsearch",
            return_value=client,
        ) as es_patch:
            memory = await ElasticsearchRawMemory.create_instance(
                index="idx", logger=MagicMock()
            )
        es_patch.assert_called_once_with()
        self.assertIsInstance(memory, ElasticsearchRawMemory)
        self.assertIs(memory._client, client)

    async def test_append_with_partitions(self):
        memory = ElasticsearchRawMemory(
            index="idx", client=AsyncMock(), logger=MagicMock()
        )
        part1 = TextPartition(
            data="a", embeddings=np.array([0.1]), total_tokens=1
        )
        part2 = TextPartition(
            data="b", embeddings=np.array([0.2]), total_tokens=1
        )
        mem_id = UUID("11111111-1111-1111-1111-111111111111")
        with (
            patch(
                "avalan.memory.permanent.elasticsearch.raw.uuid4",
                return_value=mem_id,
            ),
            patch(
                "avalan.memory.permanent.elasticsearch.raw.to_thread",
                AsyncMock(side_effect=lambda fn, **kw: fn(**kw)),
            ),
            patch(
                "avalan.memory.permanent.elasticsearch.to_thread",
                AsyncMock(side_effect=lambda fn, **kw: fn(**kw)),
            ),
        ):
            await memory.append_with_partitions(
                "ns",
                uuid4(),
                memory_type=MemoryType.RAW,
                data="d",
                identifier="id",
                partitions=[part1, part2],
                symbols={},
                model_id="m",
            )
        self.assertTrue(memory._client.index.called)
        self.assertEqual(memory._client.index_vector.call_count, 2)

    async def test_search_memories(self):
        mem_id = UUID("11111111-1111-1111-1111-111111111111")
        part = TextPartition(
            data="x", embeddings=np.array([0.3]), total_tokens=1
        )
        client = MagicMock()
        client.query_vector.return_value = {
            "Items": [{"Metadata": {"memory_id": str(mem_id)}}]
        }
        client.get.return_value = {
            "_source": {
                "id": str(mem_id),
                "model_id": "m",
                "type": "raw",
                "participant_id": str(mem_id),
                "namespace": "ns",
                "identifier": "id",
                "data": "d",
                "partitions": 1,
                "symbols": {},
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        }
        memory = ElasticsearchRawMemory(
            index="idx", client=client, logger=MagicMock()
        )
        with (
            patch(
                "avalan.memory.permanent.elasticsearch.raw.to_thread",
                AsyncMock(side_effect=lambda fn, **kw: fn(**kw)),
            ),
            patch(
                "avalan.memory.permanent.elasticsearch.to_thread",
                AsyncMock(side_effect=lambda fn, **kw: fn(**kw)),
            ),
        ):
            result = await memory.search_memories(
                search_partitions=[part],
                participant_id=mem_id,
                namespace="ns",
                function=MemoryType.RAW,  # type: ignore[arg-type]
                limit=1,
            )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, mem_id)

    async def test_search_memories_missing_id(self):
        part = TextPartition(
            data="x", embeddings=np.array([0.4]), total_tokens=1
        )
        client = MagicMock()
        client.query_vector.return_value = {"Items": [{"Metadata": {}}]}
        memory = ElasticsearchRawMemory(
            index="idx", client=client, logger=MagicMock()
        )
        with (
            patch(
                "avalan.memory.permanent.elasticsearch.raw.to_thread",
                AsyncMock(side_effect=lambda fn, **kw: fn(**kw)),
            ),
            patch(
                "avalan.memory.permanent.elasticsearch.to_thread",
                AsyncMock(side_effect=lambda fn, **kw: fn(**kw)),
            ),
        ):
            result = await memory.search_memories(
                search_partitions=[part],
                participant_id=uuid4(),
                namespace="ns",
                function=MemoryType.RAW,  # type: ignore[arg-type]
                limit=1,
            )
        self.assertEqual(result, [])
        client.get.assert_not_called()

    async def test_search_memories_missing_source(self):
        mem_id_valid = uuid4()
        mem_id_no_source = uuid4()
        part = TextPartition(
            data="y", embeddings=np.array([0.5]), total_tokens=1
        )
        client = MagicMock()
        client.query_vector.return_value = {
            "Items": [
                {},
                {"Metadata": {}},
                {"Metadata": {"memory_id": str(mem_id_no_source)}},
                {"Metadata": {"memory_id": str(mem_id_valid)}},
            ]
        }
        client.get.side_effect = [
            {},
            {
                "_source": {
                    "id": str(mem_id_valid),
                    "model_id": "m",
                    "type": "raw",
                    "participant_id": str(mem_id_valid),
                    "namespace": "ns",
                    "identifier": "id",
                    "data": "d",
                    "partitions": 1,
                    "symbols": {},
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            },
        ]
        memory = ElasticsearchRawMemory(
            index="idx", client=client, logger=MagicMock()
        )
        with (
            patch(
                "avalan.memory.permanent.elasticsearch.raw.to_thread",
                AsyncMock(side_effect=lambda fn, **kw: fn(**kw)),
            ),
            patch(
                "avalan.memory.permanent.elasticsearch.to_thread",
                AsyncMock(side_effect=lambda fn, **kw: fn(**kw)),
            ),
        ):
            result = await memory.search_memories(
                search_partitions=[part],
                participant_id=mem_id_valid,
                namespace="ns",
                function=MemoryType.RAW,  # type: ignore[arg-type]
                limit=4,
            )
        self.assertEqual(len(result), 1)
        self.assertEqual(client.get.call_count, 2)

    async def test_search_not_implemented(self):
        memory = ElasticsearchRawMemory(
            index="idx", client=MagicMock(), logger=MagicMock()
        )
        with self.assertRaises(NotImplementedError):
            await memory.search("q")
