import importlib
import sys
from logging import getLogger
from pathlib import Path
from types import ModuleType
from unittest import IsolatedAsyncioTestCase
from unittest.mock import ANY, AsyncMock, MagicMock, patch
from uuid import UUID

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from psycopg import AsyncConnection, AsyncCursor
from psycopg_pool import AsyncConnectionPool

from avalan.agent.orchestrator import Orchestrator
from avalan.entities import EngineMessage, Message, MessageRole, TextPartition
from avalan.memory.manager import MemoryManager
from avalan.memory.permanent.pgsql.message import PgsqlMessageMemory
from avalan.model import TextGenerationResponse

AGENT_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
PARTICIPANT_ID = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
MESSAGE_ID = UUID("11111111-1111-1111-1111-111111111111")


class PgsqlChatCompletionTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        # Set up FastAPI components and import chat router
        # without running server __init__
        server_pkg = ModuleType("avalan.server")
        server_pkg.__path__ = [str(Path("src/avalan/server").resolve())]

        from fastapi import Request

        def di_get_logger(request: Request):
            return request.app.state.logger

        def di_get_orchestrator(request: Request):
            return request.app.state.orchestrator

        server_pkg.di_get_logger = di_get_logger
        server_pkg.di_get_orchestrator = di_get_orchestrator
        sys.modules["avalan.server"] = server_pkg
        self.chat = importlib.import_module("avalan.server.routers.chat")
        self.FastAPI = FastAPI
        self.TestClient = TestClient

    def tearDown(self):
        sys.modules.pop("avalan.server.routers.chat", None)
        sys.modules.pop("avalan.server", None)

    async def test_syncs_messages_to_pgsql(self):
        pool, connection, cursor, _ = self.mock_pgsql()
        memory_store = await PgsqlMessageMemory.create_instance_from_pool(
            pool=pool, logger=getLogger()
        )
        partitioner = AsyncMock(
            return_value=[
                TextPartition(
                    data="ok", embeddings=np.array([0.1]), total_tokens=1
                )
            ]
        )
        memory = MemoryManager(
            agent_id=AGENT_ID,
            participant_id=PARTICIPANT_ID,
            permanent_message_memory=memory_store,
            recent_message_memory=None,
            text_partitioner=partitioner,
            logger=getLogger(),
        )

        class MemoryOrchestrator(Orchestrator):
            def __init__(self, memory):
                self._memory = memory

            async def __call__(self, messages, settings=None):
                return TextGenerationResponse(
                    lambda: "ok", logger=getLogger(), use_async_generator=False
                )

            async def sync_messages(self):
                await self._memory.append_message(
                    EngineMessage(
                        agent_id=AGENT_ID,
                        model_id="model",
                        message=Message(
                            role=MessageRole.ASSISTANT, content="ok"
                        ),
                    )
                )

        orchestrator = MemoryOrchestrator(memory)
        app = self.FastAPI()
        app.state.logger = getLogger()
        app.state.orchestrator = orchestrator
        app.include_router(self.chat.router)

        client = self.TestClient(app)
        payload = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        }
        with patch(
            "avalan.memory.permanent.pgsql.message.uuid4",
            return_value=MESSAGE_ID,
        ):
            resp = client.post("/chat/completions", json=payload)
        self.assertEqual(resp.status_code, 200)

        partitioner.assert_awaited_once_with("ok")
        connection.transaction.assert_called_once()

        insert_query = """
                        INSERT INTO "messages"(
                            "id",
                            "agent_id",
                            "model_id",
                            "session_id",
                            "author",
                            "data",
                            "partitions",
                            "created_at"
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s
                        )
                    """
        execute_call = cursor.execute.await_args_list[0]
        self.assertEqual(
            self._normalize(execute_call.args[0]),
            self._normalize(insert_query),
        )
        self.assertEqual(
            execute_call.args[1],
            (
                str(MESSAGE_ID),
                str(AGENT_ID),
                "model",
                None,
                str(MessageRole.ASSISTANT),
                "ok",
                1,
                ANY,
            ),
        )

        partitions_query = """
                        INSERT INTO "message_partitions"(
                            "agent_id",
                            "session_id",
                            "message_id",
                            "partition",
                            "data",
                            "embedding",
                            "created_at"
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s
                        )
                    """
        executemany_call = cursor.executemany.await_args_list[0]
        self.assertEqual(
            self._normalize(executemany_call.args[0]),
            self._normalize(partitions_query),
        )
        params = executemany_call.args[1]
        self.assertEqual(len(params), 1)
        self.assertEqual(
            params[0][0:5],
            (
                str(AGENT_ID),
                None,
                str(MESSAGE_ID),
                1,
                "ok",
            ),
        )
        self.assertIsNotNone(params[0][5])
        self.assertIsNotNone(params[0][6])
        cursor.close.assert_awaited_once()

    @staticmethod
    def mock_pgsql():
        cursor = AsyncMock(spec=AsyncCursor)
        cursor.__aenter__.return_value = cursor
        cursor.executemany = AsyncMock()
        transaction = AsyncMock()
        transaction.__aenter__.return_value = transaction
        connection = AsyncMock(spec=AsyncConnection)
        connection.cursor.return_value = cursor
        connection.transaction = MagicMock(return_value=transaction)
        connection.__aenter__.return_value = connection
        pool = MagicMock(spec=AsyncConnectionPool)
        pool.connection.return_value = connection
        pool.__aenter__.return_value = pool
        return pool, connection, cursor, transaction

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.split())
