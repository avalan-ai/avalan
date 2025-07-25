from avalan.memory.partitioner.text import TextPartition
from avalan.memory.permanent.s3vectors.message import S3VectorsMessageMemory
from avalan.entities import EngineMessage, Message, MessageRole
from avalan.memory.permanent import VectorFunction
from uuid import uuid4, UUID
from datetime import datetime, timezone
import numpy as np
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch


class S3VectorsMessageMemoryTestCase(IsolatedAsyncioTestCase):
    async def test_create_instance(self):
        client = MagicMock()
        memory = await S3VectorsMessageMemory.create_instance(
            bucket="b", collection="c", logger=MagicMock(), aws_client=client
        )
        self.assertIsInstance(memory, S3VectorsMessageMemory)
        self.assertIs(memory._client._client, client)

    async def test_create_instance_no_client(self):
        client = MagicMock()
        with patch(
            "avalan.memory.permanent.s3vectors.message.boto_client",
            return_value=client,
        ) as boto_patch:
            memory = await S3VectorsMessageMemory.create_instance(
                bucket="b", collection="c", logger=MagicMock()
            )
        boto_patch.assert_called_once_with("s3vectors")
        self.assertIsInstance(memory, S3VectorsMessageMemory)
        self.assertIs(memory._client._client, client)

    async def test_append_with_partitions(self):
        memory = S3VectorsMessageMemory(
            bucket="b", collection="c", client=AsyncMock(), logger=MagicMock()
        )
        memory._session_id = uuid4()
        engine_message = EngineMessage(
            agent_id=uuid4(),
            model_id="m",
            message=Message(role=MessageRole.USER, content="hi"),
        )
        part1 = TextPartition(
            data="a", embeddings=np.array([0.1]), total_tokens=1
        )
        part2 = TextPartition(
            data="b", embeddings=np.array([0.2]), total_tokens=1
        )
        msg_id = UUID("11111111-1111-1111-1111-111111111111")
        with (
            patch(
                "avalan.memory.permanent.s3vectors.message.uuid4",
                return_value=msg_id,
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
                engine_message, partitions=[part1, part2]
            )
        self.assertTrue(memory._client.put_object.called)
        self.assertEqual(memory._client.put_vector.call_count, 2)

    async def test_get_recent_messages(self):
        client = MagicMock()
        session_id = uuid4()
        client.list_objects_v2.return_value = {
            "Contents": [
                {
                    "Key": f"c/{session_id}/m1.json",
                    "LastModified": datetime(2024, 1, 1, tzinfo=timezone.utc),
                },
                {
                    "Key": f"c/{session_id}/m2.json",
                    "LastModified": datetime(2024, 1, 2, tzinfo=timezone.utc),
                },
            ]
        }
        client.get_object.return_value = {
            "Body": MagicMock(
                read=MagicMock(
                    return_value=b'{"agent_id": "'
                    + str(uuid4()).encode()
                    + b'", "model_id": "m", "author": "user", "data": "hi"}'
                )
            )
        }
        memory = S3VectorsMessageMemory(
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
            result = await memory.get_recent_messages(
                session_id=session_id, participant_id=uuid4(), limit=1
            )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].message.content, "hi")

    async def test_search_messages(self):
        client = MagicMock()
        session_id = uuid4()
        msg_id = uuid4()
        part = TextPartition(
            data="q", embeddings=np.array([0.3]), total_tokens=1
        )
        client.query_vector.return_value = {
            "Items": [{"Metadata": {"message_id": str(msg_id)}, "Score": 0.5}]
        }
        client.get_object.return_value = {
            "Body": MagicMock(
                read=MagicMock(
                    return_value=b'{"agent_id": "'
                    + str(uuid4()).encode()
                    + b'", "model_id": "m", "author": "user", "data": "hi"}'
                )
            )
        }
        memory = S3VectorsMessageMemory(
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
            result = await memory.search_messages(
                search_partitions=[part],
                agent_id=uuid4(),
                session_id=session_id,
                participant_id=uuid4(),
                function=VectorFunction.L2_DISTANCE,
                limit=1,
                search_user_messages=True,
                exclude_session_id=None,
            )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].score, 0.5)

    async def test_create_session(self):
        memory = S3VectorsMessageMemory(
            bucket="b", collection="c", client=MagicMock(), logger=MagicMock()
        )
        sid = UUID("22222222-2222-2222-2222-222222222222")
        with patch(
            "avalan.memory.permanent.s3vectors.message.uuid4",
            return_value=sid,
        ):
            result = await memory.create_session(
                agent_id=uuid4(), participant_id=uuid4()
            )
        self.assertEqual(result, sid)

    async def test_continue_session_and_get_id(self):
        memory = S3VectorsMessageMemory(
            bucket="b", collection="c", client=MagicMock(), logger=MagicMock()
        )
        sid = uuid4()
        result = await memory.continue_session_and_get_id(
            agent_id=uuid4(), participant_id=uuid4(), session_id=sid
        )
        self.assertEqual(result, sid)

    async def test_search_messages_missing_metadata(self):
        client = MagicMock()
        msg_id1 = uuid4()
        msg_id2 = uuid4()
        part = TextPartition(
            data="x", embeddings=np.array([0.1]), total_tokens=1
        )
        client.query_vector.return_value = {
            "Items": [
                {},
                {"Metadata": {}},
                {"Metadata": {"message_id": str(msg_id1)}, "Score": 0.1},
                {"Metadata": {"message_id": str(msg_id2)}, "Score": 0.2},
            ]
        }
        client.get_object.return_value = {
            "Body": MagicMock(
                read=MagicMock(
                    return_value=b'{"agent_id": "'
                    + str(uuid4()).encode()
                    + b'", "model_id": "m", "author": "user", "data": "hi"}'
                )
            )
        }
        memory = S3VectorsMessageMemory(
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
            result = await memory.search_messages(
                search_partitions=[part],
                agent_id=uuid4(),
                session_id=None,
                participant_id=uuid4(),
                function=VectorFunction.L2_DISTANCE,
                limit=4,
                search_user_messages=True,
                exclude_session_id=None,
            )
        self.assertEqual(len(result), 2)
        self.assertEqual(client.get_object.call_count, 2)
