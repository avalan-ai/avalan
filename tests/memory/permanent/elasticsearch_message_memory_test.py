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
from avalan.memory.permanent.elasticsearch.message import (  # noqa: E402
    ElasticsearchMessageMemory,
)
from avalan.entities import EngineMessage, Message, MessageRole  # noqa: E402
from avalan.memory.permanent import VectorFunction  # noqa: E402
from uuid import UUID, uuid4  # noqa: E402
import numpy as np  # noqa: E402
from unittest import IsolatedAsyncioTestCase  # noqa: E402
from unittest.mock import AsyncMock, patch  # noqa: E402


class ElasticsearchMessageMemoryTestCase(IsolatedAsyncioTestCase):
    async def test_create_instance(self):
        client = MagicMock()
        memory = await ElasticsearchMessageMemory.create_instance(
            index="idx", logger=MagicMock(), es_client=client
        )
        self.assertIsInstance(memory, ElasticsearchMessageMemory)
        self.assertIs(memory._client, client)

    async def test_create_instance_no_client(self):
        client = MagicMock()
        with patch(
            "avalan.memory.permanent.elasticsearch.message.AsyncElasticsearch",
            return_value=client,
        ) as es_patch:
            memory = await ElasticsearchMessageMemory.create_instance(
                index="idx", logger=MagicMock()
            )
        es_patch.assert_called_once_with()
        self.assertIsInstance(memory, ElasticsearchMessageMemory)
        self.assertIs(memory._client, client)

    async def test_append_with_partitions(self):
        memory = ElasticsearchMessageMemory(
            index="idx", client=AsyncMock(), logger=MagicMock()
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
                "avalan.memory.permanent.elasticsearch.message.uuid4",
                return_value=msg_id,
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
                engine_message, partitions=[part1, part2]
            )
        self.assertTrue(memory._client.index.called)
        self.assertEqual(memory._client.index_vector.call_count, 2)

    async def test_get_recent_messages(self):
        client = MagicMock()
        session_id = uuid4()
        client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "agent_id": str(uuid4()),
                            "model_id": "m",
                            "author": "user",
                            "data": "hi",
                        },
                        "_id": f"{session_id}/m1.json",
                    }
                ]
            }
        }
        memory = ElasticsearchMessageMemory(
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
        client.get.return_value = {
            "_source": {
                "agent_id": str(uuid4()),
                "model_id": "m",
                "author": "user",
                "data": "hi",
            }
        }
        memory = ElasticsearchMessageMemory(
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
        memory = ElasticsearchMessageMemory(
            index="idx", client=MagicMock(), logger=MagicMock()
        )
        sid = UUID("22222222-2222-2222-2222-222222222222")
        with patch(
            "avalan.memory.permanent.uuid4",
            return_value=sid,
        ):
            result = await memory.create_session(
                agent_id=uuid4(), participant_id=uuid4()
            )
        self.assertEqual(result, sid)
        self.assertTrue(memory._client.index.called)

    async def test_continue_session_and_get_id(self):
        memory = ElasticsearchMessageMemory(
            index="idx", client=MagicMock(), logger=MagicMock()
        )
        sid = uuid4()
        agent_id = uuid4()
        participant_id = uuid4()
        memory._client.get.return_value = {
            "_source": {
                "agent_id": str(agent_id),
                "participant_id": str(participant_id),
            }
        }
        with patch(
            "avalan.memory.permanent.elasticsearch.to_thread",
            AsyncMock(side_effect=lambda fn, **kw: fn(**kw)),
        ):
            result = await memory.continue_session_and_get_id(
                agent_id=agent_id,
                participant_id=participant_id,
                session_id=sid,
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
        client.get.return_value = {
            "_source": {
                "agent_id": str(uuid4()),
                "model_id": "m",
                "author": "user",
                "data": "hi",
            }
        }
        memory = ElasticsearchMessageMemory(
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
        self.assertEqual(client.get.call_count, 2)

    async def test_get_recent_messages_missing_source(self):
        client = MagicMock()
        session_id = uuid4()
        client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "agent_id": str(uuid4()),
                            "model_id": "m",
                            "author": "user",
                            "data": "hi",
                        },
                        "_id": f"{session_id}/m1.json",
                    },
                    {"_id": f"{session_id}/m2.json"},
                ]
            }
        }
        memory = ElasticsearchMessageMemory(
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
            result = await memory.get_recent_messages(
                session_id=session_id, participant_id=uuid4(), limit=2
            )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].message.content, "hi")

    async def test_search_messages_missing_source(self):
        client = MagicMock()
        msg_id_valid = uuid4()
        msg_id_no_source = uuid4()
        part = TextPartition(
            data="y", embeddings=np.array([0.5]), total_tokens=1
        )
        client.query_vector.return_value = {
            "Items": [
                {},
                {"Metadata": {}},
                {"Metadata": {"message_id": str(msg_id_no_source)}},
                {"Metadata": {"message_id": str(msg_id_valid)}, "Score": 0.4},
            ]
        }
        client.get.side_effect = [
            {},
            {
                "_source": {
                    "agent_id": str(uuid4()),
                    "model_id": "m",
                    "author": "user",
                    "data": "hi",
                }
            },
        ]
        memory = ElasticsearchMessageMemory(
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
        self.assertEqual(len(result), 1)
        self.assertEqual(client.get.call_count, 2)
