import importlib.machinery
import sys
import types
from unittest.mock import MagicMock

# Stub elasticsearch before importing the module under test
es_stub = types.ModuleType("elasticsearch")
es_stub.AsyncElasticsearch = MagicMock()
es_stub.__spec__ = importlib.machinery.ModuleSpec("elasticsearch", loader=None)
sys.modules.setdefault("elasticsearch", es_stub)

from datetime import datetime, timezone  # noqa: E402
from unittest import IsolatedAsyncioTestCase  # noqa: E402
from unittest.mock import AsyncMock, patch  # noqa: E402
from uuid import UUID, uuid4  # noqa: E402

import numpy as np  # noqa: E402

from avalan.memory.partitioner.text import TextPartition  # noqa: E402
from avalan.memory.permanent import (  # noqa: E402
    MemoryType,
    PermanentMemoryPartition,
    VectorFunction,
)
from avalan.memory.permanent.elasticsearch.raw import (  # noqa: E402
    ElasticsearchRawMemory,
)


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
                title="title",
                description="desc",
            )
        self.assertTrue(memory._client.index.called)
        self.assertEqual(memory._client.index_vector.call_count, 2)
        document = memory._client.index.await_args.kwargs["document"]
        self.assertEqual(document["description"], "desc")
        self.assertEqual(document["title"], "title")

    async def test_search_memories(self):
        mem_id = UUID("11111111-1111-1111-1111-111111111111")
        part = TextPartition(
            data="x", embeddings=np.array([0.3]), total_tokens=1
        )
        client = MagicMock()
        created_at = datetime.now(timezone.utc)
        client.query_vector.return_value = {
            "Items": [
                {
                    "Metadata": {
                        "memory_id": str(mem_id),
                        "participant_id": str(mem_id),
                        "namespace": "ns",
                        "partition": 0,
                        "data": "partition-data",
                        "embedding": [0.3],
                        "created_at": created_at.isoformat(),
                    }
                }
            ]
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
                function=VectorFunction.L2_DISTANCE,
                limit=1,
            )
        self.assertEqual(len(result), 1)
        partition = result[0]
        self.assertIsInstance(partition, PermanentMemoryPartition)
        self.assertEqual(partition.memory_id, mem_id)
        self.assertEqual(partition.data, "partition-data")
        self.assertTrue(np.array_equal(partition.embedding, np.array([0.3])))
        self.assertEqual(partition.created_at, created_at)

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
                function=VectorFunction.L2_DISTANCE,
                limit=1,
            )
        self.assertEqual(result, [])

    async def test_list_memories(self):
        participant_id = uuid4()
        created_at = datetime.now(timezone.utc)
        client = AsyncMock()
        client.search.return_value = {
            "hits": {
                "hits": [
                    {},
                    {"_source": None},
                    {
                        "_source": {
                            "id": str(uuid4()),
                            "model_id": "model",
                            "type": MemoryType.RAW.value,
                            "participant_id": str(participant_id),
                            "namespace": "ns",
                            "identifier": "id",
                            "data": "data",
                            "partitions": 1,
                            "symbols": {"a": 1},
                            "created_at": created_at.isoformat(),
                            "title": "title",
                            "description": "desc",
                        }
                    },
                ]
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
            memories = await memory.list_memories(
                participant_id=participant_id,
                namespace="ns",
            )

        client.search.assert_awaited_once()
        self.assertEqual(len(memories), 1)
        memory_entry = memories[0]
        self.assertEqual(memory_entry.description, "desc")
        self.assertEqual(memory_entry.title, "title")
        self.assertEqual(memory_entry.type, MemoryType.RAW)
        self.assertEqual(memory_entry.partitions, 1)

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
                {
                    "Metadata": {
                        "memory_id": str(mem_id_valid),
                        "participant_id": str(mem_id_valid),
                        "namespace": "ns",
                        "partition": 0,
                        "data": "partition-data",
                        "embedding": [0.5],
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                },
            ]
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
                participant_id=mem_id_valid,
                namespace="ns",
                function=VectorFunction.L2_DISTANCE,
                limit=4,
            )
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], PermanentMemoryPartition)

    async def test_search_not_implemented(self):
        memory = ElasticsearchRawMemory(
            index="idx", client=MagicMock(), logger=MagicMock()
        )
        with self.assertRaises(NotImplementedError):
            await memory.search("q")
