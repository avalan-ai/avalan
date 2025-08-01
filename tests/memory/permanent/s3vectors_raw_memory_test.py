from avalan.memory.partitioner.text import TextPartition
from avalan.memory.permanent import MemoryType
from avalan.memory.permanent.s3vectors.raw import S3VectorsRawMemory
from datetime import datetime, timezone
import numpy as np
from uuid import uuid4, UUID
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch


class S3VectorsRawMemoryTestCase(IsolatedAsyncioTestCase):
    async def test_create_instance(self):
        client = MagicMock()
        memory = await S3VectorsRawMemory.create_instance(
            bucket="b", collection="c", logger=MagicMock(), aws_client=client
        )
        self.assertIsInstance(memory, S3VectorsRawMemory)
        self.assertIs(memory._client._client, client)

    async def test_create_instance_no_client(self):
        client = MagicMock()
        with patch(
            "avalan.memory.permanent.s3vectors.raw.boto_client",
            return_value=client,
        ) as boto_patch:
            memory = await S3VectorsRawMemory.create_instance(
                bucket="b", collection="c", logger=MagicMock()
            )
        boto_patch.assert_called_once_with("s3vectors")
        self.assertIsInstance(memory, S3VectorsRawMemory)
        self.assertIs(memory._client._client, client)

    async def test_append_with_partitions(self):
        memory = S3VectorsRawMemory(
            bucket="b", collection="c", client=AsyncMock(), logger=MagicMock()
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
                "avalan.memory.permanent.s3vectors.raw.uuid4",
                return_value=mem_id,
            ),
            patch(
                "avalan.memory.permanent.s3vectors.raw.to_thread",
                AsyncMock(side_effect=lambda fn, **kw: fn(**kw)),
            ),
            patch(
                "avalan.memory.permanent.s3vectors.to_thread",
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
        self.assertTrue(memory._client.put_object.called)
        self.assertEqual(memory._client.put_vector.call_count, 2)

    async def test_search_memories(self):
        mem_id = UUID("11111111-1111-1111-1111-111111111111")
        part = TextPartition(
            data="x", embeddings=np.array([0.3]), total_tokens=1
        )
        client = MagicMock()
        client.query_vector.return_value = {
            "Items": [{"Metadata": {"memory_id": str(mem_id)}}]
        }
        client.get_object.return_value = {
            "Body": MagicMock(
                read=MagicMock(
                    return_value=(
                        b'{"id": "%s", "model_id": "m", "type": "raw", '
                        b'"participant_id": "%s", "namespace": "ns", '
                        b'"identifier": "id", "data": "d", "partitions": 1, '
                        b'"symbols": {}, "created_at": "%s"}'
                    )
                    % (
                        str(mem_id).encode(),
                        str(mem_id).encode(),
                        datetime.now(timezone.utc).isoformat().encode(),
                    )
                )
            )
        }
        memory = S3VectorsRawMemory(
            bucket="b", collection="c", client=client, logger=MagicMock()
        )
        with (
            patch(
                "avalan.memory.permanent.s3vectors.raw.to_thread",
                AsyncMock(side_effect=lambda fn, **kw: fn(**kw)),
            ),
            patch(
                "avalan.memory.permanent.s3vectors.to_thread",
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
        memory = S3VectorsRawMemory(
            bucket="b", collection="c", client=client, logger=MagicMock()
        )
        with (
            patch(
                "avalan.memory.permanent.s3vectors.raw.to_thread",
                AsyncMock(side_effect=lambda fn, **kw: fn(**kw)),
            ),
            patch(
                "avalan.memory.permanent.s3vectors.to_thread",
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
        client.get_object.assert_not_called()

    async def test_search_not_implemented(self):
        memory = S3VectorsRawMemory(
            bucket="b", collection="c", client=MagicMock(), logger=MagicMock()
        )
        with self.assertRaises(NotImplementedError):
            await memory.search("q")
