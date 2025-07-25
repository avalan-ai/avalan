from avalan.server.entities import ChatCompletionRequest, ChatMessage
from avalan.agent.orchestrator import Orchestrator
from avalan.entities import MessageRole
from avalan.model import TextGenerationResponse
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch
import importlib
import sys


class DummyOrchestrator(Orchestrator):
    async def __call__(self, messages, settings=None):
        return TextGenerationResponse(lambda: "ok", use_async_generator=False)


class ChatRouterUnitTest(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        sys.modules.pop("avalan.server.routers.chat", None)
        self.chat = importlib.import_module("avalan.server.routers.chat")

    def tearDown(self) -> None:
        sys.modules.pop("avalan.server.routers.chat", None)

    async def test_create_chat_completion_non_stream(self) -> None:
        orch = AsyncMock(spec=DummyOrchestrator)
        orch.return_value = TextGenerationResponse(
            lambda: "ok", use_async_generator=False
        )
        req = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role=MessageRole.USER, content="hi")],
        )
        with patch("avalan.server.routers.chat.time", return_value=1):
            resp = await self.chat.create_chat_completion(req, orch)
        self.assertEqual(resp.object, "chat.completion")
        self.assertEqual(resp.choices[0].message.content, "ok")
        orch.assert_awaited_once()

    async def test_dependency_get_orchestrator(self) -> None:
        req = type(
            "Req",
            (),
            {
                "app": type(
                    "App", (), {"state": type("S", (), {"orchestrator": 1})()}
                )()
            },
        )()
        self.assertEqual(self.chat.dependency_get_orchestrator(req), 1)

    async def test_create_chat_completion_stream(self) -> None:
        async def output_gen():
            yield "a"
            yield "b"

        orch = AsyncMock(spec=DummyOrchestrator)
        orch.return_value = TextGenerationResponse(
            lambda: output_gen(), use_async_generator=True
        )
        req = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role=MessageRole.USER, content="hi")],
            stream=True,
        )
        with patch("avalan.server.routers.chat.time", return_value=1):
            resp = await self.chat.create_chat_completion(req, orch)
        chunks = [chunk async for chunk in resp.body_iterator]
        self.assertIn('"content":"a"', chunks[0])
        self.assertIn('"content":"b"', chunks[1])
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")
        orch.assert_awaited_once()

    async def test_streaming_skips_event_tokens(self) -> None:
        from avalan.event import Event, EventType

        async def output_gen():
            yield "a"
            yield Event(type=EventType.TOOL_RESULT, payload={})
            yield "b"

        orch = AsyncMock(spec=DummyOrchestrator)
        orch.return_value = output_gen()
        req = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role=MessageRole.USER, content="hi")],
            stream=True,
        )
        with patch("avalan.server.routers.chat.time", return_value=1):
            resp = await self.chat.create_chat_completion(req, orch)
        chunks = [chunk async for chunk in resp.body_iterator]
        self.assertIn('"content":"a"', chunks[0])
        self.assertIn('"content":"b"', chunks[1])
        self.assertEqual(len(chunks), 3)
        orch.assert_awaited_once()
