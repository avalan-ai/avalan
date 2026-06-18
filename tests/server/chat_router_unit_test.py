import asyncio
import importlib
import sys
from collections.abc import AsyncIterator
from json import loads
from logging import Logger, getLogger
from types import SimpleNamespace
from typing import Any, cast
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
    ToolCallToken,
)
from avalan.event import Event, EventType
from avalan.event.manager import EventManager, EventManagerMode
from avalan.model import TextGenerationResponse
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemCorrelation,
    StreamItemKind,
    StreamTerminalOutcome,
    StreamValidationError,
    project_canonical_stream_item,
)
from avalan.server.entities import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkChoiceDelta,
    ChatCompletionRequest,
    ChatMessage,
    ContentFile,
    ContentImage,
    ContentText,
    ReasoningConfig,
    ResponsesRequest,
)


async def _canonical_answer_gen(
    *deltas: str,
) -> AsyncIterator[CanonicalStreamItem]:
    sequence = 0
    yield CanonicalStreamItem(
        stream_session_id="chat-test-stream",
        run_id="chat-test-run",
        turn_id="chat-test-turn",
        sequence=sequence,
        kind=StreamItemKind.STREAM_STARTED,
        channel=StreamChannel.CONTROL,
    )
    sequence += 1
    for delta in deltas:
        yield CanonicalStreamItem(
            stream_session_id="chat-test-stream",
            run_id="chat-test-run",
            turn_id="chat-test-turn",
            sequence=sequence,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta=delta,
        )
        sequence += 1
    if deltas:
        yield CanonicalStreamItem(
            stream_session_id="chat-test-stream",
            run_id="chat-test-run",
            turn_id="chat-test-turn",
            sequence=sequence,
            kind=StreamItemKind.ANSWER_DONE,
            channel=StreamChannel.ANSWER,
        )
        sequence += 1
    yield CanonicalStreamItem(
        stream_session_id="chat-test-stream",
        run_id="chat-test-run",
        turn_id="chat-test-turn",
        sequence=sequence,
        kind=StreamItemKind.STREAM_COMPLETED,
        channel=StreamChannel.CONTROL,
        usage=cast(Any, {"input_tokens": 0, "output_tokens": len(deltas)}),
        terminal_outcome=StreamTerminalOutcome.COMPLETED,
    )
    sequence += 1
    yield CanonicalStreamItem(
        stream_session_id="chat-test-stream",
        run_id="chat-test-run",
        turn_id="chat-test-turn",
        sequence=sequence,
        kind=StreamItemKind.STREAM_CLOSED,
        channel=StreamChannel.CONTROL,
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

    async def test_create_response_forwards_text_format(self) -> None:
        logger = AsyncMock(spec=Logger)
        orch = AsyncMock(spec=DummyOrchestrator)
        orch.return_value = TextGenerationResponse(
            lambda: "ok", logger=getLogger(), use_async_generator=False
        )
        req = ResponsesRequest(
            model="m",
            input=[ChatMessage(role=MessageRole.USER, content="hi")],
            text={
                "format": {"type": "json_object"},
                "stop": ["DONE"],
            },
        )
        with patch("avalan.server.routers.time", return_value=1):
            await self.responses.create_response(req, logger, orch)

        settings = orch.await_args.kwargs["settings"]
        self.assertEqual(settings.response_format, {"type": "json_object"})
        self.assertEqual(settings.stop_strings, ["DONE"])

    async def test_create_response_forwards_instructions_separately(
        self,
    ) -> None:
        logger = AsyncMock(spec=Logger)
        orch = AsyncMock(spec=DummyOrchestrator)
        orch.return_value = TextGenerationResponse(
            lambda: "ok", logger=getLogger(), use_async_generator=False
        )
        req = ResponsesRequest(
            model="m",
            instructions="top-level",
            input=[ChatMessage(role=MessageRole.USER, content="hi")],
        )

        with patch("avalan.server.routers.time", return_value=1):
            await self.responses.create_response(req, logger, orch)

        self.assertEqual(orch.await_args.kwargs["instructions"], "top-level")
        messages = orch.await_args.args[0]
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].role, MessageRole.USER)
        self.assertNotIn("top-level", str(messages))

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

    def test_chat_chunk_envelope_matches_model_json(self) -> None:
        envelope = self.chat._ChatCompletionChunkEnvelope(
            response_id='response-"id"',
            timestamp=7,
            model_id="model-雪",
        )
        content = 'line\nsnow 雪 "quoted"'
        expected = ChatCompletionChunk(
            id='response-"id"',
            created=7,
            model="model-雪",
            choices=[
                ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkChoiceDelta(content=content)
                )
            ],
        ).model_dump_json()

        self.assertEqual(envelope.chunk_json(content), expected)
        self.assertEqual(
            envelope.message("x"),
            f"data: {envelope.chunk_json('x')}\n\n",
        )
        final_payload = loads(envelope.final_chunk_json())
        self.assertEqual(final_payload["choices"][0]["delta"], {})
        self.assertEqual(
            final_payload["choices"][0]["finish_reason"],
            "stop",
        )

    def test_chat_chunk_envelope_does_not_leak_mutable_state(self) -> None:
        envelope = self.chat._ChatCompletionChunkEnvelope(
            response_id="response-id",
            timestamp=7,
            model_id="model-id",
        )

        first = loads(envelope.chunk_json("first"))
        first["id"] = "changed"
        first["choices"][0]["delta"]["content"] = "changed"
        second = loads(envelope.chunk_json("second"))

        self.assertEqual(second["id"], "response-id")
        self.assertEqual(second["choices"][0]["delta"]["content"], "second")

    def test_chat_chunk_envelope_rejects_invalid_state(self) -> None:
        with self.assertRaises(AssertionError):
            self.chat._ChatCompletionChunkEnvelope(
                response_id=object(),
                timestamp=7,
                model_id="model-id",
            )
        with self.assertRaises(AssertionError):
            self.chat._ChatCompletionChunkEnvelope(
                response_id="response-id",
                timestamp=True,
                model_id="model-id",
            )
        envelope = self.chat._ChatCompletionChunkEnvelope(
            response_id="response-id",
            timestamp=7,
            model_id="model-id",
        )
        with self.assertRaises(AssertionError):
            envelope.chunk_json(object())  # type: ignore[arg-type]

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
        logger = AsyncMock(spec=Logger)
        orch = AsyncMock(spec=DummyOrchestrator)
        orch.return_value = TextGenerationResponse(
            lambda: _canonical_answer_gen('a\nsnow 雪 "quoted"', "b"),
            logger=getLogger(),
            use_async_generator=True,
            provider_family="transformers",
        )
        req = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role=MessageRole.USER, content="hi")],
            stream=True,
        )
        with patch("avalan.server.routers.time", return_value=1):
            resp = await self.chat.create_chat_completion(req, logger, orch)
        chunks = [chunk async for chunk in resp.body_iterator]
        first_payload = loads(chunks[0][6:])
        self.assertEqual(
            first_payload["choices"][0]["delta"]["content"],
            'a\nsnow 雪 "quoted"',
        )
        self.assertIn('"content":"b"', chunks[1])
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")
        orch.assert_awaited_once()

    async def test_chat_empty_stream_rejects_missing_terminal(
        self,
    ) -> None:
        class EmptyResponse:
            def __init__(self) -> None:
                self.close_count = 0

            def __aiter__(self) -> AsyncIterator[object]:
                return self

            async def __anext__(self) -> object:
                raise StopAsyncIteration

            async def aclose(self) -> None:
                self.close_count += 1

        logger = AsyncMock(spec=Logger)
        orch = AsyncMock(spec=DummyOrchestrator)
        orch.sync_messages = AsyncMock()
        response = EmptyResponse()
        orch.return_value = response
        req = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role=MessageRole.USER, content="hi")],
            stream=True,
        )
        with patch("avalan.server.routers.time", return_value=1):
            resp = await self.chat.create_chat_completion(req, logger, orch)

        with self.assertRaisesRegex(
            StreamValidationError,
            "stream missing terminal outcome",
        ):
            async for _chunk in resp.body_iterator:
                pass
        self.assertEqual(response.close_count, 1)
        orch.assert_awaited_once()
        orch.sync_messages.assert_not_awaited()

    async def test_repeated_chat_stream_requests_bound_no_listener_state(
        self,
    ) -> None:
        class EventfulChatOrchestrator(Orchestrator):
            def __init__(self) -> None:
                self._event_manager = EventManager(
                    mode=EventManagerMode.SERVER
                )
                self.sync_count = 0

            async def __call__(self, messages, settings=None):  # type: ignore[no-untyped-def]
                _ = messages, settings
                for _ in range(10):
                    await self.event_manager.trigger(
                        Event(type=EventType.START)
                    )

                return TextGenerationResponse(
                    lambda: _canonical_answer_gen("bounded"),
                    logger=getLogger(),
                    use_async_generator=True,
                    provider_family="transformers",
                )

            async def sync_messages(self) -> None:  # type: ignore[override]
                self.sync_count += 1

        logger = AsyncMock(spec=Logger)
        orchestrator = EventfulChatOrchestrator()
        req = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role=MessageRole.USER, content="hi")],
            stream=True,
        )

        for _ in range(3):
            resp = await self.chat.create_chat_completion(
                req, logger, orchestrator
            )
            chunks = [chunk async for chunk in resp.body_iterator]
            self.assertIn('"content":"bounded"', chunks[0])
            final_chunk = loads(chunks[-2][6:])
            self.assertEqual(final_chunk["choices"][0]["delta"], {})
            self.assertEqual(
                final_chunk["choices"][0]["finish_reason"],
                "stop",
            )
            self.assertEqual(chunks[-1], "data: [DONE]\n\n")

        self.assertEqual(orchestrator.sync_count, 3)
        self.assertEqual(orchestrator.event_manager.history, [])
        self.assertEqual(orchestrator.event_manager._history_bytes, 0)
        self.assertEqual(orchestrator.event_manager._delivery_queue.qsize(), 0)
        self.assertEqual(orchestrator.event_manager.stats.published, 0)
        self.assertEqual(orchestrator.event_manager.stats.queue_depth, 0)
        self.assertEqual(orchestrator.event_manager.stats.dropped, 0)

    async def test_create_chat_completion_stream_uses_response_projections(
        self,
    ) -> None:
        class ProjectionResponse:
            def __init__(self) -> None:
                self.close_count = 0
                self.consumer_kwargs: dict[str, str] | None = None

            def __aiter__(self) -> AsyncIterator[str]:
                raise AssertionError("raw iterator used")

            def consumer_projections(
                self,
                *,
                stream_session_id: str,
                run_id: str,
                turn_id: str,
            ) -> AsyncIterator[object]:
                self.consumer_kwargs = {
                    "stream_session_id": stream_session_id,
                    "run_id": run_id,
                    "turn_id": turn_id,
                }

                async def gen() -> AsyncIterator[object]:
                    for item in (
                        CanonicalStreamItem(
                            stream_session_id="s",
                            run_id="r",
                            turn_id="t",
                            sequence=0,
                            kind=StreamItemKind.STREAM_STARTED,
                            channel=StreamChannel.CONTROL,
                        ),
                        CanonicalStreamItem(
                            stream_session_id="s",
                            run_id="r",
                            turn_id="t",
                            sequence=1,
                            kind=StreamItemKind.ANSWER_DELTA,
                            channel=StreamChannel.ANSWER,
                            text_delta="projected",
                        ),
                        CanonicalStreamItem(
                            stream_session_id="s",
                            run_id="r",
                            turn_id="t",
                            sequence=2,
                            kind=StreamItemKind.ANSWER_DONE,
                            channel=StreamChannel.ANSWER,
                        ),
                        CanonicalStreamItem(
                            stream_session_id="s",
                            run_id="r",
                            turn_id="t",
                            sequence=3,
                            kind=StreamItemKind.STREAM_COMPLETED,
                            channel=StreamChannel.CONTROL,
                            usage={},
                            terminal_outcome=StreamTerminalOutcome.COMPLETED,
                        ),
                    ):
                        yield project_canonical_stream_item(item)

                return gen()

            async def aclose(self) -> None:
                self.close_count += 1

        response = ProjectionResponse()
        logger = AsyncMock(spec=Logger)
        orch = AsyncMock(spec=DummyOrchestrator)
        orch.return_value = response
        req = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role=MessageRole.USER, content="hi")],
            stream=True,
        )

        with patch("avalan.server.routers.time", return_value=1):
            resp = await self.chat.create_chat_completion(req, logger, orch)
        chunks = [chunk async for chunk in resp.body_iterator]
        first_chunk = loads(chunks[0][6:])

        self.assertIn('"content":"projected"', chunks[0])
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")
        self.assertEqual(
            response.consumer_kwargs,
            {
                "stream_session_id": "chat-sse-stream",
                "run_id": first_chunk["id"],
                "turn_id": "chat-sse-turn",
            },
        )
        self.assertEqual(response.close_count, 1)
        orch.sync_messages.assert_awaited_once()

    async def test_create_chat_completion_stream_closes_source(self) -> None:
        class Stream:
            def __init__(self) -> None:
                self.close_count = 0
                self.cancel_count = 0
                self._items = iter(
                    [
                        CanonicalStreamItem(
                            stream_session_id="s",
                            run_id="r",
                            turn_id="t",
                            sequence=0,
                            kind=StreamItemKind.STREAM_STARTED,
                            channel=StreamChannel.CONTROL,
                        ),
                        CanonicalStreamItem(
                            stream_session_id="s",
                            run_id="r",
                            turn_id="t",
                            sequence=1,
                            kind=StreamItemKind.ANSWER_DELTA,
                            channel=StreamChannel.ANSWER,
                            text_delta="a",
                        ),
                        CanonicalStreamItem(
                            stream_session_id="s",
                            run_id="r",
                            turn_id="t",
                            sequence=2,
                            kind=StreamItemKind.ANSWER_DONE,
                            channel=StreamChannel.ANSWER,
                        ),
                        CanonicalStreamItem(
                            stream_session_id="s",
                            run_id="r",
                            turn_id="t",
                            sequence=3,
                            kind=StreamItemKind.USAGE_COMPLETED,
                            channel=StreamChannel.USAGE,
                            usage={},
                        ),
                        CanonicalStreamItem(
                            stream_session_id="s",
                            run_id="r",
                            turn_id="t",
                            sequence=4,
                            kind=StreamItemKind.STREAM_COMPLETED,
                            channel=StreamChannel.CONTROL,
                            terminal_outcome=StreamTerminalOutcome.COMPLETED,
                        ),
                    ]
                )

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    return next(self._items)
                except StopIteration as exc:
                    raise StopAsyncIteration from exc

            async def cancel(self) -> None:
                self.cancel_count += 1

            async def aclose(self) -> None:
                self.close_count += 1

        stream = Stream()
        logger = AsyncMock(spec=Logger)
        orch = AsyncMock(spec=DummyOrchestrator)
        orch.return_value = stream
        req = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role=MessageRole.USER, content="hi")],
            stream=True,
        )

        with patch("avalan.server.routers.time", return_value=1):
            resp = await self.chat.create_chat_completion(req, logger, orch)
        chunks = [chunk async for chunk in resp.body_iterator]

        self.assertEqual(chunks[-1], "data: [DONE]\n\n")
        self.assertEqual(stream.close_count, 1)
        self.assertEqual(stream.cancel_count, 0)
        orch.sync_messages.assert_awaited_once()

    async def test_create_chat_completion_stream_disconnect_closes_source(
        self,
    ) -> None:
        class Stream:
            def __init__(self) -> None:
                self.close_count = 0
                self.cancel_count = 0
                self._items = iter(
                    [
                        CanonicalStreamItem(
                            stream_session_id="s",
                            run_id="r",
                            turn_id="t",
                            sequence=0,
                            kind=StreamItemKind.STREAM_STARTED,
                            channel=StreamChannel.CONTROL,
                        ),
                        CanonicalStreamItem(
                            stream_session_id="s",
                            run_id="r",
                            turn_id="t",
                            sequence=1,
                            kind=StreamItemKind.ANSWER_DELTA,
                            channel=StreamChannel.ANSWER,
                            text_delta="a",
                        ),
                        CanonicalStreamItem(
                            stream_session_id="s",
                            run_id="r",
                            turn_id="t",
                            sequence=2,
                            kind=StreamItemKind.ANSWER_DELTA,
                            channel=StreamChannel.ANSWER,
                            text_delta="b",
                        ),
                    ]
                )

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    return next(self._items)
                except StopIteration as exc:
                    raise StopAsyncIteration from exc

            async def cancel(self) -> None:
                self.cancel_count += 1

            async def aclose(self) -> None:
                self.close_count += 1

        stream = Stream()
        logger = AsyncMock(spec=Logger)
        orch = AsyncMock(spec=DummyOrchestrator)
        orch.return_value = stream
        req = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role=MessageRole.USER, content="hi")],
            stream=True,
        )

        with patch("avalan.server.routers.time", return_value=1):
            resp = await self.chat.create_chat_completion(req, logger, orch)
        iterator = resp.body_iterator
        first = await anext(iterator)
        await iterator.aclose()

        self.assertIn('"content":"a"', first)
        self.assertEqual(stream.close_count, 1)
        self.assertEqual(stream.cancel_count, 0)
        orch.sync_messages.assert_not_awaited()

    async def test_create_chat_completion_stream_cancellation_cancels_source(
        self,
    ) -> None:
        class Stream:
            def __init__(self) -> None:
                self.close_count = 0
                self.cancel_count = 0
                self.started = asyncio.Event()

            def __aiter__(self):
                return self

            async def __anext__(self):
                self.started.set()
                await asyncio.Event().wait()
                raise StopAsyncIteration

            async def cancel(self) -> None:
                self.cancel_count += 1

            async def aclose(self) -> None:
                self.close_count += 1

        stream = Stream()
        logger = AsyncMock(spec=Logger)
        orch = AsyncMock(spec=DummyOrchestrator)
        orch.return_value = stream
        req = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role=MessageRole.USER, content="hi")],
            stream=True,
        )

        with patch("avalan.server.routers.time", return_value=1):
            resp = await self.chat.create_chat_completion(req, logger, orch)
        task = asyncio.create_task(anext(resp.body_iterator))
        await stream.started.wait()
        task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(stream.cancel_count, 1)
        self.assertEqual(stream.close_count, 1)
        orch.sync_messages.assert_not_awaited()

    async def test_create_chat_completion_streams_canonical_items(
        self,
    ) -> None:
        async def output_gen():
            yield CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            )
            yield CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=1,
                kind=StreamItemKind.REASONING_DELTA,
                channel=StreamChannel.REASONING,
                text_delta="plan",
            )
            yield CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=2,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="answer",
            )
            yield CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=3,
                kind=StreamItemKind.REASONING_DONE,
                channel=StreamChannel.REASONING,
            )
            yield CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=4,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            )
            yield CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=5,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                usage={"input_tokens": 2, "output_tokens": 1},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            )

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

        self.assertNotIn("plan", "".join(chunks))
        self.assertIn('"content":"answer"', chunks[0])
        usage_chunk = loads(chunks[1][6:])
        self.assertEqual(usage_chunk["choices"], [])
        self.assertEqual(
            usage_chunk["usage"],
            {
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "total_tokens": 3,
            },
        )
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")
        final_chunk = loads(chunks[-2][6:])
        self.assertEqual(final_chunk["choices"][0]["delta"], {})
        self.assertEqual(final_chunk["choices"][0]["finish_reason"], "stop")
        self.assertEqual(len(chunks), 4)

    async def test_create_chat_completion_stream_preserves_terminal_failure(
        self,
    ) -> None:
        cases = (
            (
                StreamItemKind.STREAM_CANCELLED,
                StreamTerminalOutcome.CANCELLED,
                "chat.completion.cancelled",
                None,
            ),
            (
                StreamItemKind.STREAM_ERRORED,
                StreamTerminalOutcome.ERRORED,
                "chat.completion.failed",
                {
                    "error_type": "RuntimeError",
                    "message": "provider failed",
                },
            ),
        )

        for (
            terminal_kind,
            terminal_outcome,
            event_name,
            terminal_data,
        ) in cases:
            with self.subTest(event=event_name):

                async def output_gen():
                    yield CanonicalStreamItem(
                        stream_session_id="s",
                        run_id="r",
                        turn_id="t",
                        sequence=0,
                        kind=StreamItemKind.STREAM_STARTED,
                        channel=StreamChannel.CONTROL,
                    )
                    yield CanonicalStreamItem(
                        stream_session_id="s",
                        run_id="r",
                        turn_id="t",
                        sequence=1,
                        kind=StreamItemKind.USAGE_COMPLETED,
                        channel=StreamChannel.USAGE,
                        usage={"input_tokens": 3, "output_tokens": 2},
                    )
                    yield CanonicalStreamItem(
                        stream_session_id="s",
                        run_id="r",
                        turn_id="t",
                        sequence=2,
                        kind=terminal_kind,
                        channel=StreamChannel.CONTROL,
                        data=terminal_data,
                        terminal_outcome=terminal_outcome,
                    )

                logger = AsyncMock(spec=Logger)
                orch = AsyncMock(spec=DummyOrchestrator)
                orch.return_value = output_gen()
                req = ChatCompletionRequest(
                    model="m",
                    messages=[
                        ChatMessage(role=MessageRole.USER, content="hi")
                    ],
                    stream=True,
                )
                with patch("avalan.server.routers.time", return_value=1):
                    resp = await self.chat.create_chat_completion(
                        req, logger, orch
                    )
                chunks = [chunk async for chunk in resp.body_iterator]
                text = "".join(chunks)
                usage_chunk = loads(chunks[0][6:])

                self.assertEqual(
                    usage_chunk["usage"],
                    {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "total_tokens": 5,
                    },
                )
                self.assertEqual(usage_chunk["choices"], [])
                self.assertIn(f"event: {event_name}", chunks[1])
                terminal_chunk = loads(chunks[1].split("data: ")[1])
                self.assertEqual(terminal_chunk["sequence_number"], 2)
                if terminal_data is None:
                    self.assertNotIn("error", terminal_chunk)
                else:
                    self.assertEqual(terminal_chunk["error"], terminal_data)
                self.assertIn(f"event: {event_name}", text)
                self.assertNotIn("[DONE]", text)
                orch.sync_messages.assert_not_awaited()

    async def test_create_chat_completion_stream_preserves_projection_usage(
        self,
    ) -> None:
        async def output_gen():
            for item in (
                CanonicalStreamItem(
                    stream_session_id="s",
                    run_id="r",
                    turn_id="t",
                    sequence=0,
                    kind=StreamItemKind.STREAM_STARTED,
                    channel=StreamChannel.CONTROL,
                ),
                CanonicalStreamItem(
                    stream_session_id="s",
                    run_id="r",
                    turn_id="t",
                    sequence=1,
                    kind=StreamItemKind.ANSWER_DELTA,
                    channel=StreamChannel.ANSWER,
                    text_delta="answer",
                ),
                CanonicalStreamItem(
                    stream_session_id="s",
                    run_id="r",
                    turn_id="t",
                    sequence=2,
                    kind=StreamItemKind.ANSWER_DONE,
                    channel=StreamChannel.ANSWER,
                ),
                CanonicalStreamItem(
                    stream_session_id="s",
                    run_id="r",
                    turn_id="t",
                    sequence=3,
                    kind=StreamItemKind.USAGE_COMPLETED,
                    channel=StreamChannel.USAGE,
                    usage={"input_tokens": 4, "output_tokens": 3},
                ),
                CanonicalStreamItem(
                    stream_session_id="s",
                    run_id="r",
                    turn_id="t",
                    sequence=4,
                    kind=StreamItemKind.STREAM_COMPLETED,
                    channel=StreamChannel.CONTROL,
                    terminal_outcome=StreamTerminalOutcome.COMPLETED,
                ),
            ):
                yield project_canonical_stream_item(item)

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

        self.assertIn('"content":"answer"', chunks[0])
        usage_chunk = loads(chunks[1][6:])
        self.assertEqual(
            usage_chunk["usage"],
            {
                "prompt_tokens": 4,
                "completion_tokens": 3,
                "total_tokens": 7,
            },
        )
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")
        final_chunk = loads(chunks[-2][6:])
        self.assertEqual(final_chunk["choices"][0]["delta"], {})
        self.assertEqual(final_chunk["choices"][0]["finish_reason"], "stop")
        self.assertEqual(len(chunks), 4)
        orch.sync_messages.assert_awaited_once()

    async def test_create_chat_completion_stream_preserves_projection_failure(
        self,
    ) -> None:
        async def output_gen():
            for item in (
                CanonicalStreamItem(
                    stream_session_id="s",
                    run_id="r",
                    turn_id="t",
                    sequence=0,
                    kind=StreamItemKind.STREAM_STARTED,
                    channel=StreamChannel.CONTROL,
                ),
                CanonicalStreamItem(
                    stream_session_id="s",
                    run_id="r",
                    turn_id="t",
                    sequence=1,
                    kind=StreamItemKind.STREAM_ERRORED,
                    channel=StreamChannel.CONTROL,
                    data={
                        "error_type": "RuntimeError",
                        "message": "provider failed",
                    },
                    terminal_outcome=StreamTerminalOutcome.ERRORED,
                ),
            ):
                yield project_canonical_stream_item(item)

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
        terminal_chunk = loads(chunks[0].split("data: ")[1])

        self.assertIn("event: chat.completion.failed", chunks[0])
        self.assertEqual(terminal_chunk["sequence_number"], 1)
        self.assertEqual(
            terminal_chunk["error"],
            {"error_type": "RuntimeError", "message": "provider failed"},
        )
        self.assertNotIn("[DONE]", "".join(chunks))
        orch.sync_messages.assert_not_awaited()

    async def test_create_chat_completion_stream_rejects_late_content(
        self,
    ) -> None:
        async def output_gen():
            yield CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            )
            yield CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=1,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                usage={},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            )
            yield CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=2,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="late",
            )

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

        with self.assertRaises(StreamValidationError):
            async for _chunk in resp.body_iterator:
                pass

    async def test_create_chat_completion_stream_rejects_late_projection(
        self,
    ) -> None:
        async def output_gen():
            for item in (
                CanonicalStreamItem(
                    stream_session_id="s",
                    run_id="r",
                    turn_id="t",
                    sequence=0,
                    kind=StreamItemKind.STREAM_STARTED,
                    channel=StreamChannel.CONTROL,
                ),
                CanonicalStreamItem(
                    stream_session_id="s",
                    run_id="r",
                    turn_id="t",
                    sequence=1,
                    kind=StreamItemKind.STREAM_COMPLETED,
                    channel=StreamChannel.CONTROL,
                    usage={},
                    terminal_outcome=StreamTerminalOutcome.COMPLETED,
                ),
                CanonicalStreamItem(
                    stream_session_id="s",
                    run_id="r",
                    turn_id="t",
                    sequence=2,
                    kind=StreamItemKind.ANSWER_DELTA,
                    channel=StreamChannel.ANSWER,
                    text_delta="late",
                ),
            ):
                yield project_canonical_stream_item(item)

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

        with self.assertRaises(StreamValidationError):
            async for _chunk in resp.body_iterator:
                pass

    async def test_create_chat_completion_stream_rejects_mixed_surfaces(
        self,
    ) -> None:
        canonical_item = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=0,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
        )
        projection = project_canonical_stream_item(canonical_item)
        cases = (
            (
                ("legacy", canonical_item),
                "unsupported stream item for Chat SSE projection",
            ),
            (
                ("legacy", projection),
                "unsupported stream item for Chat SSE projection",
            ),
            (
                (canonical_item, "legacy"),
                "unsupported stream item for Chat SSE projection",
            ),
            (
                (projection, "legacy"),
                "unsupported stream item for Chat SSE projection",
            ),
        )

        for items, message in cases:
            with self.subTest(message=message, first=type(items[0]).__name__):

                async def output_gen():
                    for item in items:
                        yield item

                logger = AsyncMock(spec=Logger)
                orch = AsyncMock(spec=DummyOrchestrator)
                orch.return_value = output_gen()
                req = ChatCompletionRequest(
                    model="m",
                    messages=[
                        ChatMessage(role=MessageRole.USER, content="hi")
                    ],
                    stream=True,
                )
                with patch("avalan.server.routers.time", return_value=1):
                    resp = await self.chat.create_chat_completion(
                        req, logger, orch
                    )

                with self.assertRaisesRegex(StreamValidationError, message):
                    async for _chunk in resp.body_iterator:
                        pass
                orch.sync_messages.assert_not_awaited()

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

    async def test_streaming_rejects_legacy_event_tokens(self) -> None:
        from avalan.event import Event, EventType

        async def output_gen():
            yield CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            )
            yield CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=1,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="a",
            )
            yield Event(type=EventType.TOOL_RESULT, payload={})

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
        chunks = []
        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported stream item for Chat SSE projection",
        ):
            async for chunk in resp.body_iterator:
                chunks.append(chunk)
        self.assertIn('"content":"a"', chunks[0])
        self.assertEqual(len(chunks), 1)
        orch.sync_messages.assert_not_awaited()

    async def test_streaming_rejects_legacy_event_before_text(self) -> None:
        from avalan.event import Event, EventType

        async def output_gen():
            yield Event(
                type=EventType.TOOL_DIAGNOSTIC,
                payload={"diagnostic": "ignored"},
            )
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
        chunks = []
        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported stream item for Chat SSE projection",
        ):
            async for chunk in resp.body_iterator:
                chunks.append(chunk)
        self.assertEqual(chunks, [])
        orch.sync_messages.assert_not_awaited()

    async def test_streaming_skips_reasoning_tokens(self) -> None:
        async def output_gen():
            for item in (
                CanonicalStreamItem(
                    stream_session_id="s",
                    run_id="r",
                    turn_id="t",
                    sequence=0,
                    kind=StreamItemKind.STREAM_STARTED,
                    channel=StreamChannel.CONTROL,
                ),
                CanonicalStreamItem(
                    stream_session_id="s",
                    run_id="r",
                    turn_id="t",
                    sequence=1,
                    kind=StreamItemKind.REASONING_DELTA,
                    channel=StreamChannel.REASONING,
                    text_delta="r",
                ),
                CanonicalStreamItem(
                    stream_session_id="s",
                    run_id="r",
                    turn_id="t",
                    sequence=2,
                    kind=StreamItemKind.REASONING_DONE,
                    channel=StreamChannel.REASONING,
                ),
                CanonicalStreamItem(
                    stream_session_id="s",
                    run_id="r",
                    turn_id="t",
                    sequence=3,
                    kind=StreamItemKind.ANSWER_DELTA,
                    channel=StreamChannel.ANSWER,
                    text_delta="a",
                ),
                CanonicalStreamItem(
                    stream_session_id="s",
                    run_id="r",
                    turn_id="t",
                    sequence=4,
                    kind=StreamItemKind.ANSWER_DELTA,
                    channel=StreamChannel.ANSWER,
                    text_delta="b",
                ),
                CanonicalStreamItem(
                    stream_session_id="s",
                    run_id="r",
                    turn_id="t",
                    sequence=5,
                    kind=StreamItemKind.ANSWER_DONE,
                    channel=StreamChannel.ANSWER,
                ),
                CanonicalStreamItem(
                    stream_session_id="s",
                    run_id="r",
                    turn_id="t",
                    sequence=6,
                    kind=StreamItemKind.USAGE_COMPLETED,
                    channel=StreamChannel.USAGE,
                    usage={},
                ),
                CanonicalStreamItem(
                    stream_session_id="s",
                    run_id="r",
                    turn_id="t",
                    sequence=7,
                    kind=StreamItemKind.STREAM_COMPLETED,
                    channel=StreamChannel.CONTROL,
                    terminal_outcome=StreamTerminalOutcome.COMPLETED,
                ),
            ):
                yield item

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
        self.assertNotIn('"content":"r"', "".join(chunks))
        self.assertEqual(len(chunks), 5)
        orch.assert_awaited_once()

    async def test_streaming_skips_tool_call_tokens(self) -> None:
        async def output_gen():
            correlation = StreamItemCorrelation(tool_call_id="call-1")
            for item in (
                CanonicalStreamItem(
                    stream_session_id="s",
                    run_id="r",
                    turn_id="t",
                    sequence=0,
                    kind=StreamItemKind.STREAM_STARTED,
                    channel=StreamChannel.CONTROL,
                ),
                CanonicalStreamItem(
                    stream_session_id="s",
                    run_id="r",
                    turn_id="t",
                    sequence=1,
                    kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    channel=StreamChannel.TOOL_CALL,
                    correlation=correlation,
                    text_delta="t",
                ),
                CanonicalStreamItem(
                    stream_session_id="s",
                    run_id="r",
                    turn_id="t",
                    sequence=2,
                    kind=StreamItemKind.TOOL_CALL_READY,
                    channel=StreamChannel.TOOL_CALL,
                    correlation=correlation,
                ),
                CanonicalStreamItem(
                    stream_session_id="s",
                    run_id="r",
                    turn_id="t",
                    sequence=3,
                    kind=StreamItemKind.TOOL_CALL_DONE,
                    channel=StreamChannel.TOOL_CALL,
                    correlation=correlation,
                ),
                CanonicalStreamItem(
                    stream_session_id="s",
                    run_id="r",
                    turn_id="t",
                    sequence=4,
                    kind=StreamItemKind.ANSWER_DELTA,
                    channel=StreamChannel.ANSWER,
                    text_delta="a",
                ),
                CanonicalStreamItem(
                    stream_session_id="s",
                    run_id="r",
                    turn_id="t",
                    sequence=5,
                    kind=StreamItemKind.ANSWER_DELTA,
                    channel=StreamChannel.ANSWER,
                    text_delta="b",
                ),
                CanonicalStreamItem(
                    stream_session_id="s",
                    run_id="r",
                    turn_id="t",
                    sequence=6,
                    kind=StreamItemKind.ANSWER_DONE,
                    channel=StreamChannel.ANSWER,
                ),
                CanonicalStreamItem(
                    stream_session_id="s",
                    run_id="r",
                    turn_id="t",
                    sequence=7,
                    kind=StreamItemKind.USAGE_COMPLETED,
                    channel=StreamChannel.USAGE,
                    usage={},
                ),
                CanonicalStreamItem(
                    stream_session_id="s",
                    run_id="r",
                    turn_id="t",
                    sequence=8,
                    kind=StreamItemKind.STREAM_COMPLETED,
                    channel=StreamChannel.CONTROL,
                    terminal_outcome=StreamTerminalOutcome.COMPLETED,
                ),
            ):
                yield item

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
        self.assertNotIn('"content":"t"', "".join(chunks))
        self.assertEqual(len(chunks), 5)
        orch.assert_awaited_once()

    def test_chat_stream_text_uses_consumer_projection(self) -> None:
        reasoning = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=1,
            kind=StreamItemKind.REASONING_DELTA,
            channel=StreamChannel.REASONING,
            text_delta="plan",
        )
        answer = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=2,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta="answer",
        )
        tool = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=3,
            kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
            channel=StreamChannel.TOOL_CALL,
            correlation=StreamItemCorrelation(tool_call_id="tool-1"),
            text_delta="tool",
        )
        done = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=4,
            kind=StreamItemKind.ANSWER_DONE,
            channel=StreamChannel.ANSWER,
        )

        reasoning_projection = project_canonical_stream_item(reasoning)
        answer_projection = project_canonical_stream_item(answer)
        tool_projection = project_canonical_stream_item(tool)
        done_projection = project_canonical_stream_item(done)

        self.assertIsNone(self.chat._stream_text(reasoning_projection))
        self.assertEqual(
            self.chat._stream_text(answer_projection),
            "answer",
        )
        self.assertIsNone(self.chat._stream_text(tool_projection))
        self.assertIsNone(self.chat._stream_text(done_projection))
        with self.assertRaises(AssertionError):
            self.chat._stream_text(object())  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            self.chat._stream_text(ToolCallToken(token="raw"))  # type: ignore[arg-type]

    def test_chat_terminal_event_preserves_non_completed_outcomes(
        self,
    ) -> None:
        self.assertIsNone(
            self.chat._chat_terminal_event(
                "response-id",
                1,
                "model-id",
                None,
            )
        )
        self.assertIsNone(
            self.chat._chat_terminal_event(
                "response-id",
                1,
                "model-id",
                StreamTerminalOutcome.COMPLETED,
            )
        )

        cancelled = self.chat._chat_terminal_event(
            "response-id",
            1,
            "model-id",
            StreamTerminalOutcome.CANCELLED,
        )
        assert cancelled is not None
        self.assertIn("event: chat.completion.cancelled", cancelled)
        self.assertEqual(
            loads(cancelled.split("data: ")[1])["type"],
            "chat.completion.cancelled",
        )

        failed = self.chat._chat_terminal_event(
            "response-id",
            1,
            "model-id",
            StreamTerminalOutcome.ERRORED,
        )
        assert failed is not None
        self.assertIn("event: chat.completion.failed", failed)
        self.assertEqual(
            loads(failed.split("data: ")[1])["type"],
            "chat.completion.failed",
        )

    def test_chat_terminal_event_preserves_error_data(self) -> None:
        item = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=4,
            kind=StreamItemKind.STREAM_ERRORED,
            channel=StreamChannel.CONTROL,
            data={"error_type": "RuntimeError", "message": "provider failed"},
            terminal_outcome=StreamTerminalOutcome.ERRORED,
        )

        event = self.chat._chat_terminal_event(
            "response-id",
            1,
            "model-id",
            project_canonical_stream_item(item),
        )

        assert event is not None
        data = loads(event.split("data: ")[1])
        self.assertEqual(data["type"], "chat.completion.failed")
        self.assertEqual(data["sequence_number"], 4)
        self.assertEqual(
            data["error"],
            {"error_type": "RuntimeError", "message": "provider failed"},
        )

    def test_chat_terminal_helpers_reject_bad_state(self) -> None:
        item = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=0,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta="answer",
        )

        with self.assertRaises(AssertionError):
            self.chat._chat_terminal_event(
                "response-id",
                1,
                "model-id",
                project_canonical_stream_item(item),
            )

    def test_chat_usage_handles_missing_and_malformed_usage(self) -> None:
        self.assertIsNone(self.chat._chat_usage(None))

        malformed = self.chat._chat_usage("usage")
        assert malformed is not None
        self.assertEqual(malformed.prompt_tokens, 0)
        self.assertEqual(malformed.completion_tokens, 0)
        self.assertEqual(malformed.total_tokens, 0)

        usage = self.chat._chat_usage(
            {
                "prompt_tokens": -1,
                "completion_tokens": True,
                "output_tokens": 4,
            }
        )
        assert usage is not None
        self.assertEqual(usage.prompt_tokens, 0)
        self.assertEqual(usage.completion_tokens, 4)
        self.assertEqual(usage.total_tokens, 4)

        canonical_usage = self.chat._chat_usage(
            {
                "totals": {
                    "input_tokens": 5,
                    "output_tokens": 6,
                    "total_tokens": 11,
                }
            }
        )
        assert canonical_usage is not None
        self.assertEqual(canonical_usage.prompt_tokens, 5)
        self.assertEqual(canonical_usage.completion_tokens, 6)
        self.assertEqual(canonical_usage.total_tokens, 11)

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
