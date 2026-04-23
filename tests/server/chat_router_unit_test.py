import importlib
import sys
from logging import Logger, getLogger
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from avalan.agent.orchestrator import Orchestrator
from avalan.entities import (
    MessageContentFile,
    MessageContentImage,
    MessageContentText,
    MessageRole,
    ReasoningEffort,
    ReasoningToken,
    ToolCallToken,
)
from avalan.model import TextGenerationResponse
from avalan.server.entities import (
    ChatCompletionRequest,
    ChatMessage,
    ContentFile,
    ContentImage,
    ContentText,
    ReasoningConfig,
    ResponsesRequest,
)


class DummyOrchestrator(Orchestrator):
    async def __call__(self, messages, settings=None):
        return TextGenerationResponse(
            lambda: "ok", logger=getLogger(), use_async_generator=False
        )


class ChatRouterUnitTest(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        sys.modules.pop("avalan.server.routers.chat", None)
        sys.modules.pop("avalan.server.routers.responses", None)
        self.chat = importlib.import_module("avalan.server.routers.chat")
        self.responses = importlib.import_module(
            "avalan.server.routers.responses"
        )

    def tearDown(self) -> None:
        sys.modules.pop("avalan.server.routers.chat", None)
        sys.modules.pop("avalan.server.routers.responses", None)

    async def test_create_chat_completion_non_stream(self) -> None:
        logger = AsyncMock(spec=Logger)
        orch = AsyncMock(spec=DummyOrchestrator)
        orch.return_value = TextGenerationResponse(
            lambda: "ok", logger=getLogger(), use_async_generator=False
        )
        req = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role=MessageRole.USER, content="hi")],
        )
        with patch("avalan.server.routers.time", return_value=1):
            resp = await self.chat.create_chat_completion(req, logger, orch)
        self.assertEqual(resp.object, "chat.completion")
        self.assertEqual(resp.choices[0].message.content, "ok")
        orch.assert_awaited_once()

    async def test_create_chat_completion_multiple(self) -> None:
        logger = AsyncMock(spec=Logger)
        orch = AsyncMock(spec=DummyOrchestrator)
        orch.return_value = TextGenerationResponse(
            lambda: "ok", logger=getLogger(), use_async_generator=False
        )
        req = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role=MessageRole.USER, content="hi")],
            n=3,
        )
        with patch("avalan.server.routers.time", return_value=1):
            resp = await self.chat.create_chat_completion(req, logger, orch)
        self.assertEqual(len(resp.choices), 3)
        self.assertEqual([c.index for c in resp.choices], [0, 1, 2])
        orch.assert_awaited_once()
        settings = orch.await_args.kwargs["settings"]
        self.assertEqual(settings.num_return_sequences, 3)

    async def test_create_chat_completion_forwards_reasoning_effort(
        self,
    ) -> None:
        logger = AsyncMock(spec=Logger)
        orch = AsyncMock(spec=DummyOrchestrator)
        orch.return_value = TextGenerationResponse(
            lambda: "ok", logger=getLogger(), use_async_generator=False
        )
        req = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role=MessageRole.USER, content="hi")],
            reasoning_effort=ReasoningEffort.XHIGH,
        )
        with patch("avalan.server.routers.time", return_value=1):
            await self.chat.create_chat_completion(req, logger, orch)

        settings = orch.await_args.kwargs["settings"]
        self.assertEqual(settings.reasoning.effort, ReasoningEffort.XHIGH)

    async def test_create_chat_completion_uses_orchestrator_model(
        self,
    ) -> None:
        logger = AsyncMock(spec=Logger)
        orch = AsyncMock(spec=DummyOrchestrator)
        orch.model_ids = {"server-model"}
        orch.return_value = TextGenerationResponse(
            lambda: "ok", logger=getLogger(), use_async_generator=False
        )
        req = ChatCompletionRequest(
            messages=[ChatMessage(role=MessageRole.USER, content="hi")],
        )
        with patch("avalan.server.routers.time", return_value=1):
            resp = await self.chat.create_chat_completion(req, logger, orch)

        self.assertEqual(resp.model, "server-model")

    async def test_create_chat_completion_prefers_request_model(
        self,
    ) -> None:
        logger = AsyncMock(spec=Logger)
        orch = AsyncMock(spec=DummyOrchestrator)
        orch.model_ids = {"alpha", "server-model"}
        orch.return_value = TextGenerationResponse(
            lambda: "ok", logger=getLogger(), use_async_generator=False
        )
        req = ChatCompletionRequest(
            model="request-model",
            messages=[ChatMessage(role=MessageRole.USER, content="hi")],
        )
        with patch("avalan.server.routers.time", return_value=1):
            resp = await self.chat.create_chat_completion(req, logger, orch)

        self.assertEqual(resp.model, "request-model")

    async def test_create_response_forwards_reasoning_effort(self) -> None:
        logger = AsyncMock(spec=Logger)
        orch = AsyncMock(spec=DummyOrchestrator)
        orch.return_value = TextGenerationResponse(
            lambda: "ok", logger=getLogger(), use_async_generator=False
        )
        req = ResponsesRequest(
            model="m",
            input=[ChatMessage(role=MessageRole.USER, content="hi")],
            reasoning=ReasoningConfig(effort=ReasoningEffort.HIGH),
        )
        with patch("avalan.server.routers.time", return_value=1):
            await self.responses.create_response(req, logger, orch)

        settings = orch.await_args.kwargs["settings"]
        self.assertEqual(settings.reasoning.effort, ReasoningEffort.HIGH)

    async def test_create_response_uses_orchestrator_model(self) -> None:
        logger = AsyncMock(spec=Logger)
        orch = AsyncMock(spec=DummyOrchestrator)
        orch.model_ids = {"server-model"}
        orch.return_value = TextGenerationResponse(
            lambda: "ok", logger=getLogger(), use_async_generator=False
        )
        req = ResponsesRequest(
            input=[ChatMessage(role=MessageRole.USER, content="hi")],
        )
        with patch("avalan.server.routers.time", return_value=1):
            resp = await self.responses.create_response(req, logger, orch)

        self.assertEqual(resp["model"], "server-model")

    async def test_create_response_prefers_request_model(self) -> None:
        logger = AsyncMock(spec=Logger)
        orch = AsyncMock(spec=DummyOrchestrator)
        orch.model_ids = {"alpha", "server-model"}
        orch.return_value = TextGenerationResponse(
            lambda: "ok", logger=getLogger(), use_async_generator=False
        )
        req = ResponsesRequest(
            model="request-model",
            input=[ChatMessage(role=MessageRole.USER, content="hi")],
        )
        with patch("avalan.server.routers.time", return_value=1):
            resp = await self.responses.create_response(req, logger, orch)

        assert isinstance(resp, dict)
        self.assertEqual(resp["model"], "request-model")

    def test_resolve_model_id_prefers_request_model(self) -> None:
        routers = importlib.import_module("avalan.server.routers")

        model_id = routers.resolve_model_id(
            SimpleNamespace(model_ids={"alpha", "beta"}),
            "request-model",
        )

        self.assertEqual(model_id, "request-model")

    def test_resolve_model_id_falls_back_to_default(self) -> None:
        routers = importlib.import_module("avalan.server.routers")

        model_id = routers.resolve_model_id(SimpleNamespace())

        self.assertEqual(model_id, routers.MODEL_FALLBACK)

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
        self.assertEqual(await self.chat.di_get_orchestrator(req), 1)

    async def test_create_chat_completion_stream(self) -> None:
        async def output_gen():
            yield "a"
            yield "b"

        logger = AsyncMock(spec=Logger)
        orch = AsyncMock(spec=DummyOrchestrator)
        orch.return_value = TextGenerationResponse(
            lambda: output_gen(), logger=getLogger(), use_async_generator=True
        )
        req = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role=MessageRole.USER, content="hi")],
            stream=True,
        )
        with patch("avalan.server.routers.time", return_value=1):
            resp = await self.chat.create_chat_completion(req, logger, orch)
        chunks = [chunk async for chunk in resp.body_iterator]
        self.assertIn('"content":"a"', chunks[0])
        self.assertIn('"content":"b"', chunks[1])
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")
        orch.assert_awaited_once()

    async def test_create_chat_completion_stream_multiple_raises(self) -> None:
        logger = AsyncMock(spec=Logger)
        orch = AsyncMock(spec=DummyOrchestrator)
        req = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role=MessageRole.USER, content="hi")],
            stream=True,
            n=2,
        )
        with self.assertRaises(HTTPException):
            await self.chat.create_chat_completion(req, logger, orch)
        orch.assert_not_awaited()

    async def test_streaming_skips_event_tokens(self) -> None:
        from avalan.event import Event, EventType

        async def output_gen():
            yield "a"
            yield Event(type=EventType.TOOL_RESULT, payload={})
            yield "b"

        logger = AsyncMock(spec=Logger)
        orch = AsyncMock(spec=DummyOrchestrator)
        orch.return_value = output_gen()
        req = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role=MessageRole.USER, content="hi")],
            stream=True,
        )
        with patch("avalan.server.routers.time", return_value=1):
            resp = await self.chat.create_chat_completion(req, logger, orch)
        chunks = [chunk async for chunk in resp.body_iterator]
        self.assertIn('"content":"a"', chunks[0])
        self.assertIn('"content":"b"', chunks[1])
        self.assertEqual(len(chunks), 3)
        orch.assert_awaited_once()

    async def test_streaming_includes_reasoning_tokens(self) -> None:
        async def output_gen():
            yield "a"
            yield ReasoningToken(token="r")
            yield "b"

        logger = AsyncMock(spec=Logger)
        orch = AsyncMock(spec=DummyOrchestrator)
        orch.return_value = output_gen()
        req = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role=MessageRole.USER, content="hi")],
            stream=True,
        )
        with patch("avalan.server.routers.time", return_value=1):
            resp = await self.chat.create_chat_completion(req, logger, orch)
        chunks = [chunk async for chunk in resp.body_iterator]
        self.assertIn('"content":"a"', chunks[0])
        self.assertIn('"content":"r"', chunks[1])
        self.assertIn('"content":"b"', chunks[2])
        self.assertEqual(len(chunks), 4)
        orch.assert_awaited_once()

    async def test_streaming_includes_tool_call_tokens(self) -> None:
        async def output_gen():
            yield "a"
            yield ToolCallToken(token="t")
            yield "b"

        logger = AsyncMock(spec=Logger)
        orch = AsyncMock(spec=DummyOrchestrator)
        orch.return_value = output_gen()
        req = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role=MessageRole.USER, content="hi")],
            stream=True,
        )
        with patch("avalan.server.routers.time", return_value=1):
            resp = await self.chat.create_chat_completion(req, logger, orch)
        chunks = [chunk async for chunk in resp.body_iterator]
        self.assertIn('"content":"a"', chunks[0])
        self.assertIn('"content":"t"', chunks[1])
        self.assertIn('"content":"b"', chunks[2])
        self.assertEqual(len(chunks), 4)
        orch.assert_awaited_once()

    async def test_message_content_string(self) -> None:
        logger = AsyncMock(spec=Logger)
        orch = AsyncMock(spec=DummyOrchestrator)
        orch.return_value = TextGenerationResponse(
            lambda: "ok",
            logger=getLogger(),
            use_async_generator=False,
        )
        req = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role=MessageRole.USER, content="hi")],
        )
        with patch("avalan.server.routers.time", return_value=1):
            await self.chat.create_chat_completion(req, logger, orch)

        orch.assert_awaited_once()
        messages = orch.await_args.args[0]
        self.assertEqual(len(messages), 1)
        msg = messages[0]
        self.assertIsInstance(msg.content, MessageContentText)
        self.assertEqual(msg.content.text, "hi")

    async def test_message_content_image(self) -> None:
        logger = AsyncMock(spec=Logger)
        orch = AsyncMock(spec=DummyOrchestrator)
        orch.return_value = TextGenerationResponse(
            lambda: "ok",
            logger=getLogger(),
            use_async_generator=False,
        )
        req = ChatCompletionRequest(
            model="m",
            messages=[
                ChatMessage(
                    role=MessageRole.USER,
                    content=[
                        ContentImage(
                            type="image_url", image_url={"url": "img"}
                        )
                    ],
                )
            ],
        )
        with patch("avalan.server.routers.time", return_value=1):
            await self.chat.create_chat_completion(req, logger, orch)

        orch.assert_awaited_once()
        msg = orch.await_args.args[0][0]
        self.assertIsInstance(msg.content, list)
        self.assertEqual(len(msg.content), 1)
        self.assertIsInstance(msg.content[0], MessageContentImage)
        self.assertEqual(msg.content[0].image_url, {"url": "img"})

    async def test_message_content_text_object(self) -> None:
        logger = AsyncMock(spec=Logger)
        orch = AsyncMock(spec=DummyOrchestrator)
        orch.return_value = TextGenerationResponse(
            lambda: "ok",
            logger=getLogger(),
            use_async_generator=False,
        )
        req = ChatCompletionRequest(
            model="m",
            messages=[
                ChatMessage(
                    role=MessageRole.USER,
                    content=[ContentText(type="text", text="hello")],
                )
            ],
        )
        with patch("avalan.server.routers.time", return_value=1):
            await self.chat.create_chat_completion(req, logger, orch)

        msg = orch.await_args.args[0][0]
        self.assertIsInstance(msg.content, list)
        self.assertEqual(len(msg.content), 1)
        self.assertIsInstance(msg.content[0], MessageContentText)
        self.assertEqual(msg.content[0].text, "hello")

    async def test_response_message_content_input_text_object(self) -> None:
        logger = AsyncMock(spec=Logger)
        orch = AsyncMock(spec=DummyOrchestrator)
        orch.return_value = TextGenerationResponse(
            lambda: "ok",
            logger=getLogger(),
            use_async_generator=False,
        )
        req = ResponsesRequest(
            model="m",
            input=[
                ChatMessage(
                    role=MessageRole.USER,
                    content=[ContentText(type="input_text", text="hello")],
                )
            ],
        )
        with patch("avalan.server.routers.time", return_value=1):
            await self.responses.create_response(req, logger, orch)

        msg = orch.await_args.args[0][0]
        self.assertIsInstance(msg.content, list)
        self.assertEqual(len(msg.content), 1)
        self.assertIsInstance(msg.content[0], MessageContentText)
        self.assertEqual(msg.content[0].type, "text")
        self.assertEqual(msg.content[0].text, "hello")

    async def test_message_content_list(self) -> None:
        logger = AsyncMock(spec=Logger)
        orch = AsyncMock(spec=DummyOrchestrator)
        orch.return_value = TextGenerationResponse(
            lambda: "ok",
            logger=getLogger(),
            use_async_generator=False,
        )
        req = ChatCompletionRequest(
            model="m",
            messages=[
                ChatMessage(
                    role=MessageRole.USER,
                    content=[
                        ContentText(type="text", text="hi"),
                        ContentImage(
                            type="image_url", image_url={"url": "img"}
                        ),
                    ],
                )
            ],
        )
        with patch("avalan.server.routers.time", return_value=1):
            await self.chat.create_chat_completion(req, logger, orch)

        msg = orch.await_args.args[0][0]
        self.assertIsInstance(msg.content, list)
        self.assertIsInstance(msg.content[0], MessageContentText)
        self.assertEqual(msg.content[0].text, "hi")
        self.assertIsInstance(msg.content[1], MessageContentImage)
        self.assertEqual(msg.content[1].image_url, {"url": "img"})

    async def test_message_content_file_object(self) -> None:
        logger = AsyncMock(spec=Logger)
        orch = AsyncMock(spec=DummyOrchestrator)
        orch.return_value = TextGenerationResponse(
            lambda: "ok",
            logger=getLogger(),
            use_async_generator=False,
        )
        req = ChatCompletionRequest(
            model="m",
            messages=[
                ChatMessage(
                    role=MessageRole.USER,
                    content=[
                        ContentFile(
                            type="file",
                            file={"file_id": "file-1", "title": "Doc"},
                        )
                    ],
                )
            ],
        )
        with patch("avalan.server.routers.time", return_value=1):
            await self.chat.create_chat_completion(req, logger, orch)

        msg = orch.await_args.args[0][0]
        self.assertIsInstance(msg.content, list)
        self.assertEqual(len(msg.content), 1)
        self.assertIsInstance(msg.content[0], MessageContentFile)
        self.assertEqual(
            msg.content[0].file, {"file_id": "file-1", "title": "Doc"}
        )

    async def test_message_content_input_file_object(self) -> None:
        logger = AsyncMock(spec=Logger)
        orch = AsyncMock(spec=DummyOrchestrator)
        orch.return_value = TextGenerationResponse(
            lambda: "ok",
            logger=getLogger(),
            use_async_generator=False,
        )
        req = ChatCompletionRequest(
            model="m",
            messages=[
                ChatMessage(
                    role=MessageRole.USER,
                    content=[
                        ContentFile(
                            type="input_file",
                            file_id="file-2",
                            filename="report.pdf",
                        )
                    ],
                )
            ],
        )
        with patch("avalan.server.routers.time", return_value=1):
            await self.chat.create_chat_completion(req, logger, orch)

        msg = orch.await_args.args[0][0]
        self.assertIsInstance(msg.content, list)
        self.assertEqual(len(msg.content), 1)
        self.assertIsInstance(msg.content[0], MessageContentFile)
        self.assertEqual(
            msg.content[0].file,
            {"file_id": "file-2", "filename": "report.pdf"},
        )

    def test_to_message_content_preserves_top_level_file_fields(self) -> None:
        from avalan.server.routers import to_message_content

        content = to_message_content(
            ContentFile(
                type="input_file",
                file_url="https://example.com/report.pdf",
                file_data="YWJj",
                filename="report.pdf",
            )
        )

        self.assertIsInstance(content, MessageContentFile)
        self.assertEqual(
            content.file,
            {
                "file_url": "https://example.com/report.pdf",
                "file_data": "YWJj",
                "filename": "report.pdf",
            },
        )

    async def test_to_message_content_invalid_type(self) -> None:
        from avalan.server.routers import to_message_content

        with self.assertRaises(TypeError):
            to_message_content(123)
