from asyncio import (
    FIRST_COMPLETED,
    CancelledError,
    create_task,
    wait,
    wait_for,
)
from asyncio import Event as AsyncEvent
from collections.abc import AsyncIterator, Awaitable, Callable
from json import dumps
from logging import getLogger
from time import perf_counter
from types import SimpleNamespace
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from fastapi import Request

from avalan.agent.orchestrator import Orchestrator
from avalan.entities import (
    MessageRole,
    ToolCall,
    ToolCallContext,
    ToolCallDiagnostic,
    ToolCallDiagnosticCode,
    ToolExecutionStreamEvent,
    ToolExecutionStreamKind,
)
from avalan.event import Event, EventType
from avalan.event.manager import (
    EventDeliveryConfig,
    EventDeliveryPolicy,
    EventManager,
)
from avalan.model.stream import (
    CanonicalStreamItem,
    LocalTextStreamEventParser,
    StreamChannel,
    StreamItemKind,
    StreamPerformanceBudget,
    StreamProducerBackend,
    StreamProviderCapabilities,
    StreamProviderEvent,
    normalize_provider_stream,
)
from avalan.server.a2a.router import _cleanup_stream_sources_safely
from avalan.server.a2a.router import create_task as create_a2a_task
from avalan.server.a2a.store import TaskStore
from avalan.server.entities import (
    ChatCompletionRequest,
    ChatMessage,
    ResponsesRequest,
)
from avalan.server.routers import chat as chat_router
from avalan.server.routers import mcp as mcp_router
from avalan.server.routers import responses as responses_router
from avalan.server.routers.streaming import (
    cancellable_stream_iterator,
    cleanup_stream_sources,
)
from avalan.tool import Tool, ToolSet
from avalan.tool.manager import ToolManager

STREAM_LATENCY_TEST_TIMEOUT = 1.0


class _PendingProviderEvents:
    def __init__(self) -> None:
        self.started = AsyncEvent()
        self.cancelled = False
        self.closed = False
        self.read_count = 0

    def __aiter__(self) -> "_PendingProviderEvents":
        return self

    async def __anext__(self) -> StreamProviderEvent:
        self.read_count += 1
        self.started.set()
        try:
            await AsyncEvent().wait()
        except CancelledError:
            self.cancelled = True
            raise
        return StreamProviderEvent(
            kind=StreamItemKind.ANSWER_DELTA,
            text_delta="late",
        )

    async def aclose(self) -> None:
        self.closed = True


class _PendingLocalTokens:
    def __init__(self) -> None:
        self.started = AsyncEvent()
        self.cancelled = False
        self.closed = False
        self.read_count = 0

    def __aiter__(self) -> "_PendingLocalTokens":
        return self

    async def __anext__(self) -> str:
        self.read_count += 1
        self.started.set()
        try:
            await AsyncEvent().wait()
        except CancelledError:
            self.cancelled = True
            raise
        return "late"

    async def aclose(self) -> None:
        self.closed = True


async def _local_text_events(
    tokens: _PendingLocalTokens,
) -> AsyncIterator[StreamProviderEvent]:
    parser = LocalTextStreamEventParser()
    iterator = tokens.__aiter__()
    tokens_exhausted = False
    try:
        while True:
            try:
                token = await iterator.__anext__()
            except StopAsyncIteration:
                tokens_exhausted = True
                break
            for event in parser.push(token):
                yield event
        for event in parser.flush():
            yield event
    finally:
        if not tokens_exhausted:
            await iterator.aclose()


def _local_text_stream(
    tokens: _PendingLocalTokens,
    *,
    stream_session_id: str,
    run_id: str,
    turn_id: str,
) -> AsyncIterator[CanonicalStreamItem]:
    events_exhausted = False

    async def tracked_events() -> AsyncIterator[StreamProviderEvent]:
        nonlocal events_exhausted
        async for event in _local_text_events(tokens):
            yield event
        events_exhausted = True

    stream = normalize_provider_stream(
        tracked_events(),
        stream_session_id=stream_session_id,
        run_id=run_id,
        turn_id=turn_id,
        capabilities=StreamProviderCapabilities(
            backend=StreamProducerBackend.LOCAL,
            supports_reasoning=True,
            supports_tool_calls=True,
            supports_cancellation=True,
            max_queue_depth=StreamPerformanceBudget().max_queue_depth,
        ),
    )

    async def closeable_stream() -> AsyncIterator[CanonicalStreamItem]:
        try:
            async for item in stream:
                yield item
        finally:
            stream_aclose = getattr(stream, "aclose", None)
            if stream_aclose is not None:
                await stream_aclose()
            if not events_exhausted:
                await tokens.aclose()

    return closeable_stream()


class _CleanupProbe:
    def __init__(self) -> None:
        self.cancel_count = 0
        self.close_count = 0

    async def cancel(self) -> None:
        self.cancel_count += 1

    async def aclose(self) -> None:
        self.close_count += 1


class _LatencyResponse:
    input_token_count = 0
    output_token_count = 0

    def __init__(
        self,
        items: tuple[object, ...] = (),
        *,
        block_when_empty: bool = True,
    ) -> None:
        self.started = AsyncEvent()
        self.cancel_count = 0
        self.close_count = 0
        self.cancelled_by_pull = False
        self.read_count = 0
        self._items = list(items)
        self._block_when_empty = block_when_empty

    def __aiter__(self) -> "_LatencyResponse":
        return self

    async def __anext__(self) -> object:
        if self._items:
            self.read_count += 1
            return self._items.pop(0)
        if not self._block_when_empty:
            raise StopAsyncIteration
        self.started.set()
        try:
            await AsyncEvent().wait()
        except CancelledError:
            self.cancelled_by_pull = True
            raise
        raise StopAsyncIteration

    async def cancel(self) -> None:
        self.cancel_count += 1

    async def aclose(self) -> None:
        self.close_count += 1


class _CloseTrackingContext:
    def __init__(self) -> None:
        self.enter_count = 0
        self.close_count = 0

    async def __aenter__(self) -> "_CloseTrackingContext":
        self.enter_count += 1
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> bool | None:
        _ = exc_type, exc_value, traceback
        self.close_count += 1
        return None


class _StreamingCancellationTool(Tool):
    """Emit a stream event and honor cooperative cancellation.

    Args:
        None.

    Returns:
        A marker string when cancellation is not requested.
    """

    def __init__(self) -> None:
        super().__init__()
        self.__name__ = "streaming_cancel"

    async def __call__(self, context: ToolCallContext) -> str:
        if context.stream_event is not None:
            await context.stream_event(
                ToolExecutionStreamEvent(
                    kind=ToolExecutionStreamKind.STDOUT,
                    content="live output",
                )
            )
        if context.cancellation_checker is not None:
            await context.cancellation_checker()
        return "completed"


class _CloseTrackingTool(Tool):
    """Register an async close hook.

    Args:
        None.

    Returns:
        A marker string after the tool runs.
    """

    def __init__(self, resource: _CloseTrackingContext) -> None:
        super().__init__()
        self.__name__ = "close_tracking"
        self._resource = resource

    async def __aenter__(self) -> "_CloseTrackingTool":
        await super().__aenter__()
        await self._exit_stack.enter_async_context(self._resource)
        return self

    async def __call__(self) -> str:
        return "completed"


def _hosted_provider_capabilities() -> StreamProviderCapabilities:
    return StreamProviderCapabilities(
        backend=StreamProducerBackend.HOSTED,
        provider_family="openai",
        supports_cancellation=True,
        supports_terminal_events=True,
        max_queue_depth=StreamPerformanceBudget().max_queue_depth,
    )


def _canonical_answer_delta_items(
    text: str,
) -> tuple[CanonicalStreamItem, ...]:
    assert isinstance(text, str)
    assert text
    return (
        CanonicalStreamItem(
            stream_session_id="latency-stream",
            run_id="latency-run",
            turn_id="latency-turn",
            sequence=0,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
        ),
        CanonicalStreamItem(
            stream_session_id="latency-stream",
            run_id="latency-run",
            turn_id="latency-turn",
            sequence=1,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta=text,
        ),
    )


async def _elapsed_ms(action: Callable[[], Awaitable[None]]) -> float:
    started = perf_counter()
    await action()
    return (perf_counter() - started) * 1000


def _test_orchestrator() -> Orchestrator:
    orchestrator = cast(Any, Orchestrator.__new__(Orchestrator))
    orchestrator._model_ids = {"m"}
    orchestrator.sync_messages = AsyncMock()
    return cast(Orchestrator, orchestrator)


def _a2a_request() -> Request:
    payload = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    body = dumps(payload).encode("utf-8")
    consumed = False

    async def receive() -> dict[str, object]:
        nonlocal consumed
        if consumed:
            return {"type": "http.request", "body": b"", "more_body": False}
        consumed = True
        return {"type": "http.request", "body": body, "more_body": False}

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "path": "/tasks",
        "raw_path": b"/tasks",
        "headers": [(b"content-type", b"application/json")],
        "query_string": b"",
        "server": ("test", 80),
        "client": ("test", 1234),
        "scheme": "http",
        "root_path": "",
        "app": SimpleNamespace(state=SimpleNamespace()),
    }
    return Request(scope, receive)


class StreamingLatencyBudgetTestCase(IsolatedAsyncioTestCase):
    async def test_cancellation_and_close_latency_budget_across_surfaces(
        self,
    ) -> None:
        budget = StreamPerformanceBudget()

        cancellation_latencies = {
            "hosted_provider_normalization": (
                await self._hosted_provider_cancellation_latency()
            ),
            "local_generation_normalization": (
                await self._local_generation_cancellation_latency()
            ),
            "cooperative_tool_streaming": (
                await self._tool_stream_cancellation_latency()
            ),
            "mcp_pending_pull": await self._mcp_pending_pull_latency(),
            "http_chat_sse_adapter": (
                await self._chat_sse_cancellation_latency()
            ),
            "http_responses_sse_adapter": (
                await self._responses_sse_cancellation_latency()
            ),
            "mcp_stream_response_adapter": (
                await self._mcp_stream_response_cancellation_latency()
            ),
            "a2a_route_stream_adapter": (
                await self._a2a_route_cancellation_latency()
            ),
            "http_sse_adapter_cleanup": await self._http_cleanup_latency(
                cancelled=True
            ),
            "mcp_cleanup_wrapper": await self._mcp_cleanup_latency(
                cancelled=True
            ),
            "a2a_cleanup_wrapper": await self._a2a_cleanup_latency(
                cancelled=True
            ),
        }
        close_latencies = {
            "hosted_provider_normalization": (
                await self._hosted_provider_close_latency()
            ),
            "local_generation_normalization": (
                await self._local_generation_close_latency()
            ),
            "tool_manager_toolset_tool": (
                await self._tool_manager_close_latency()
            ),
            "fanout_subscribers": await self._fanout_close_latency(),
            "http_chat_sse_adapter": await self._chat_sse_close_latency(),
            "http_responses_sse_adapter": (
                await self._responses_sse_close_latency()
            ),
            "mcp_stream_response_adapter": (
                await self._mcp_stream_response_close_latency()
            ),
            "a2a_route_stream_adapter": await self._a2a_route_close_latency(),
            "http_sse_adapter_cleanup": await self._http_cleanup_latency(
                cancelled=False
            ),
            "mcp_cleanup_wrapper": await self._mcp_cleanup_latency(
                cancelled=False
            ),
            "a2a_cleanup_wrapper": await self._a2a_cleanup_latency(
                cancelled=False
            ),
        }

        print(
            "phase7 benchmark cancellation_close_latency "
            f"cancellation_ms={cancellation_latencies} "
            f"close_ms={close_latencies}"
        )
        for label, elapsed_ms in cancellation_latencies.items():
            with self.subTest(surface=label, budget="cancellation"):
                self.assertLessEqual(
                    elapsed_ms,
                    budget.cancellation_latency_ms,
                )
        for label, elapsed_ms in close_latencies.items():
            with self.subTest(surface=label, budget="close"):
                self.assertLessEqual(elapsed_ms, budget.close_latency_ms)

    async def _hosted_provider_cancellation_latency(self) -> float:
        events = _PendingProviderEvents()
        stream = normalize_provider_stream(
            events,
            stream_session_id="hosted-cancel-stream",
            run_id="hosted-cancel-run",
            turn_id="hosted-cancel-turn",
            capabilities=_hosted_provider_capabilities(),
        )
        started_item = await stream.__anext__()
        self.assertIs(started_item.kind, StreamItemKind.STREAM_STARTED)
        pull = create_task(stream.__anext__())
        await wait_for(
            events.started.wait(),
            STREAM_LATENCY_TEST_TIMEOUT,
        )
        elapsed_ms = await _elapsed_ms(lambda: self._cancel_pull(pull))
        await cast(Any, stream).aclose()

        self.assertTrue(events.cancelled)
        self.assertTrue(events.closed)
        return elapsed_ms

    async def _hosted_provider_close_latency(self) -> float:
        events = _PendingProviderEvents()
        stream = normalize_provider_stream(
            events,
            stream_session_id="hosted-close-stream",
            run_id="hosted-close-run",
            turn_id="hosted-close-turn",
            capabilities=_hosted_provider_capabilities(),
        )
        started_item = await stream.__anext__()
        self.assertIs(started_item.kind, StreamItemKind.STREAM_STARTED)
        elapsed_ms = await _elapsed_ms(lambda: cast(Any, stream).aclose())

        self.assertEqual(events.read_count, 0)
        self.assertTrue(events.closed)
        return elapsed_ms

    async def _local_generation_cancellation_latency(self) -> float:
        tokens = _PendingLocalTokens()
        stream = _local_text_stream(
            tokens,
            stream_session_id="local-cancel-stream",
            run_id="local-cancel-run",
            turn_id="local-cancel-turn",
        )
        started_item = await stream.__anext__()
        self.assertIs(started_item.kind, StreamItemKind.STREAM_STARTED)
        pull = create_task(stream.__anext__())
        await wait_for(
            tokens.started.wait(),
            STREAM_LATENCY_TEST_TIMEOUT,
        )
        elapsed_ms = await _elapsed_ms(lambda: self._cancel_pull(pull))
        await cast(Any, stream).aclose()

        self.assertTrue(tokens.cancelled)
        self.assertTrue(tokens.closed)
        return elapsed_ms

    async def _local_generation_close_latency(self) -> float:
        tokens = _PendingLocalTokens()
        stream = _local_text_stream(
            tokens,
            stream_session_id="local-close-stream",
            run_id="local-close-run",
            turn_id="local-close-turn",
        )
        started_item = await stream.__anext__()
        self.assertIs(started_item.kind, StreamItemKind.STREAM_STARTED)
        elapsed_ms = await _elapsed_ms(lambda: cast(Any, stream).aclose())

        self.assertEqual(tokens.read_count, 0)
        self.assertTrue(tokens.closed)
        return elapsed_ms

    async def _tool_stream_cancellation_latency(self) -> float:
        events: list[ToolExecutionStreamEvent] = []
        checks = 0

        async def record(event: ToolExecutionStreamEvent) -> None:
            events.append(event)

        async def cancel_after_dispatch() -> None:
            nonlocal checks
            checks += 1
            if checks == 4:
                raise CancelledError()

        manager = ToolManager.create_instance(
            enable_tools=["streaming_cancel"],
            available_toolsets=[ToolSet(tools=[_StreamingCancellationTool()])],
        )
        call = ToolCall(id="tool-call", name="streaming_cancel", arguments={})

        async def execute() -> None:
            outcome = await manager.execute_call(
                call,
                context=ToolCallContext(
                    cancellation_checker=cancel_after_dispatch,
                    stream_event=record,
                ),
            )
            self.assertIsInstance(outcome, ToolCallDiagnostic)
            diagnostic = cast(ToolCallDiagnostic, outcome)
            self.assertIs(diagnostic.code, ToolCallDiagnosticCode.CANCELLED)

        elapsed_ms = await _elapsed_ms(execute)

        self.assertEqual(checks, 4)
        self.assertEqual(len(events), 1)
        self.assertIs(events[0].kind, ToolExecutionStreamKind.STDOUT)
        return elapsed_ms

    async def _tool_manager_close_latency(self) -> float:
        resource = _CloseTrackingContext()
        manager = ToolManager.create_instance(
            enable_tools=["close_tracking"],
            available_toolsets=[ToolSet(tools=[_CloseTrackingTool(resource)])],
        )
        await manager.__aenter__()

        elapsed_ms = await _elapsed_ms(
            lambda: manager.__aexit__(None, None, None)
        )

        self.assertEqual(resource.enter_count, 1)
        self.assertEqual(resource.close_count, 1)
        return elapsed_ms

    async def _mcp_pending_pull_latency(self) -> float:
        source = _PendingProviderEvents()
        cancel_event = AsyncEvent()
        iterator = cancellable_stream_iterator(source, cancel_event)
        pull = create_task(anext(iterator))
        await wait_for(
            source.started.wait(),
            STREAM_LATENCY_TEST_TIMEOUT,
        )

        async def cancel() -> None:
            cancel_event.set()
            with self.assertRaises(StopAsyncIteration):
                await wait_for(pull, STREAM_LATENCY_TEST_TIMEOUT)

        elapsed_ms = await _elapsed_ms(cancel)

        self.assertTrue(source.cancelled)
        return elapsed_ms

    async def _fanout_close_latency(self) -> float:
        manager = EventManager()
        started = AsyncEvent()

        async def listener(event: Event) -> None:
            _ = event
            started.set()
            await AsyncEvent().wait()

        manager.add_listener(
            listener,
            [EventType.START],
            delivery_config=EventDeliveryConfig(
                policy=EventDeliveryPolicy.DROP,
                queue_limit=1,
            ),
        )
        await manager.trigger(Event(type=EventType.START))
        await wait_for(started.wait(), STREAM_LATENCY_TEST_TIMEOUT)
        subscriber = manager._subscriber_index[listener]
        assert subscriber.task is not None

        elapsed_ms = await _elapsed_ms(manager.aclose)

        self.assertTrue(subscriber.closed)
        self.assertTrue(subscriber.queue.empty())
        self.assertTrue(subscriber.task.cancelled())
        return elapsed_ms

    async def _chat_sse_cancellation_latency(self) -> float:
        source = _LatencyResponse()
        response = await self._chat_sse_response(source)
        iterator = cast(AsyncIterator[str], response.body_iterator)
        pull = create_task(anext(iterator))
        await wait_for(source.started.wait(), STREAM_LATENCY_TEST_TIMEOUT)

        elapsed_ms = await _elapsed_ms(lambda: self._cancel_task(pull))

        self.assertTrue(source.cancelled_by_pull)
        self.assertEqual(source.cancel_count, 1)
        self.assertEqual(source.close_count, 1)
        return elapsed_ms

    async def _chat_sse_close_latency(self) -> float:
        source = _LatencyResponse(_canonical_answer_delta_items("partial"))
        response = await self._chat_sse_response(source)
        iterator = cast(AsyncIterator[str], response.body_iterator)
        first = await anext(iterator)
        self.assertIn('"content":"partial"', first)

        elapsed_ms = await _elapsed_ms(lambda: cast(Any, iterator).aclose())

        self.assertEqual(source.cancel_count, 0)
        self.assertEqual(source.close_count, 1)
        return elapsed_ms

    async def _responses_sse_cancellation_latency(self) -> float:
        source = _LatencyResponse()
        response = await self._responses_sse_response(source)
        iterator = cast(AsyncIterator[str], response.body_iterator)
        created = await anext(iterator)
        self.assertIn("response.created", created)
        pull = create_task(anext(iterator))
        await wait_for(source.started.wait(), STREAM_LATENCY_TEST_TIMEOUT)

        elapsed_ms = await _elapsed_ms(lambda: self._cancel_task(pull))

        self.assertTrue(source.cancelled_by_pull)
        self.assertEqual(source.cancel_count, 1)
        self.assertEqual(source.close_count, 1)
        return elapsed_ms

    async def _responses_sse_close_latency(self) -> float:
        source = _LatencyResponse(_canonical_answer_delta_items("partial"))
        response = await self._responses_sse_response(source)
        iterator = cast(AsyncIterator[str], response.body_iterator)
        created = await anext(iterator)
        self.assertIn("response.created", created)
        partial = await anext(iterator)
        self.assertIn("response.output_item.added", partial)
        self.assertEqual(source.read_count, 2)

        elapsed_ms = await _elapsed_ms(lambda: cast(Any, iterator).aclose())

        self.assertEqual(source.cancel_count, 0)
        self.assertEqual(source.close_count, 1)
        return elapsed_ms

    async def _mcp_stream_response_cancellation_latency(self) -> float:
        source = _LatencyResponse()
        cancel_event = AsyncEvent()
        stream = self._mcp_stream_response(source, cancel_event=cancel_event)
        pull = create_task(anext(stream))
        await wait_for(source.started.wait(), STREAM_LATENCY_TEST_TIMEOUT)

        async def cancel() -> None:
            cancel_event.set()
            first = await wait_for(pull, STREAM_LATENCY_TEST_TIMEOUT)
            self.assertIn(b"Request cancelled", first)
            async for _chunk in stream:
                continue

        elapsed_ms = await _elapsed_ms(cancel)

        self.assertTrue(source.cancelled_by_pull)
        self.assertEqual(source.cancel_count, 1)
        self.assertEqual(source.close_count, 1)
        return elapsed_ms

    async def _mcp_stream_response_close_latency(self) -> float:
        source = _LatencyResponse(_canonical_answer_delta_items("partial"))
        stream = self._mcp_stream_response(source)
        first = await anext(stream)
        self.assertIn(b"answer.delta", first)

        elapsed_ms = await _elapsed_ms(lambda: stream.aclose())

        self.assertEqual(source.cancel_count, 1)
        self.assertEqual(source.close_count, 1)
        return elapsed_ms

    async def _a2a_route_cancellation_latency(self) -> float:
        source = _LatencyResponse()
        response = await self._a2a_streaming_response(source)
        iterator = cast(AsyncIterator[str], response.body_iterator)
        for _ in range(3):
            await anext(iterator)
        pull = create_task(anext(iterator))
        await wait_for(source.started.wait(), STREAM_LATENCY_TEST_TIMEOUT)

        elapsed_ms = await _elapsed_ms(
            lambda: self._cancel_a2a_pull(pull, iterator)
        )

        self.assertTrue(source.cancelled_by_pull)
        self.assertEqual(source.cancel_count, 1)
        self.assertEqual(source.close_count, 1)
        return elapsed_ms

    async def _a2a_route_close_latency(self) -> float:
        source = _LatencyResponse(_canonical_answer_delta_items("partial"))
        response = await self._a2a_streaming_response(source)
        iterator = cast(AsyncIterator[str], response.body_iterator)
        while source.read_count == 0:
            await wait_for(anext(iterator), STREAM_LATENCY_TEST_TIMEOUT)

        elapsed_ms = await _elapsed_ms(lambda: cast(Any, iterator).aclose())

        self.assertEqual(source.cancel_count, 1)
        self.assertEqual(source.close_count, 1)
        return elapsed_ms

    async def _chat_sse_response(self, source: _LatencyResponse) -> Any:
        request = ChatCompletionRequest(
            model="m",
            messages=[ChatMessage(role=MessageRole.USER, content="hi")],
            stream=True,
        )

        async def orchestrate_stub(
            request: ChatCompletionRequest,
            logger: Any,
            orchestrator: Orchestrator,
        ) -> tuple[_LatencyResponse, str, int]:
            _ = request, logger, orchestrator
            return source, "chat-latency", 1

        with patch.object(chat_router, "orchestrate", orchestrate_stub):
            return await chat_router.create_chat_completion(
                request,
                getLogger("tests.streaming_latency.chat"),
                _test_orchestrator(),
            )

    async def _responses_sse_response(self, source: _LatencyResponse) -> Any:
        request = ResponsesRequest(
            model="m",
            input=[ChatMessage(role=MessageRole.USER, content="hi")],
            stream=True,
        )

        async def orchestrate_stub(
            request: ResponsesRequest,
            logger: Any,
            orchestrator: Orchestrator,
        ) -> tuple[_LatencyResponse, str, int]:
            _ = request, logger, orchestrator
            return source, "responses-latency", 1

        with patch.object(responses_router, "orchestrate", orchestrate_stub):
            return await responses_router.create_response(
                request,
                getLogger("tests.streaming_latency.responses"),
                _test_orchestrator(),
            )

    def _mcp_stream_response(
        self,
        source: _LatencyResponse,
        *,
        cancel_event: AsyncEvent | None = None,
    ) -> AsyncIterator[bytes]:
        return mcp_router._stream_mcp_response(
            request_id="latency",
            request_model=ChatCompletionRequest(
                model="m",
                messages=[ChatMessage(role=MessageRole.USER, content="hi")],
                stream=True,
            ),
            response=cast(Any, source),
            response_id=uuid4(),
            timestamp=1,
            progress_token="progress",
            orchestrator=_test_orchestrator(),
            logger=getLogger("tests.streaming_latency.mcp"),
            resource_store=mcp_router.MCPResourceStore(),
            base_path="/resources",
            cancel_event=cancel_event or AsyncEvent(),
        )

    async def _a2a_streaming_response(self, source: _LatencyResponse) -> Any:
        async def orchestrate_stub(
            request: ChatCompletionRequest,
            logger: Any,
            orchestrator: Orchestrator,
        ) -> tuple[_LatencyResponse, object, int]:
            _ = request, logger, orchestrator
            return source, uuid4(), 1

        with patch("avalan.server.a2a.router.orchestrate", orchestrate_stub):
            return await create_a2a_task(
                _a2a_request(),
                logger=getLogger("tests.streaming_latency.a2a"),
                orchestrator=_test_orchestrator(),
                store=TaskStore(),
            )

    async def _http_cleanup_latency(self, *, cancelled: bool) -> float:
        source = _CleanupProbe()
        iterator = _CleanupProbe()
        elapsed_ms = await _elapsed_ms(
            lambda: cleanup_stream_sources(
                source,
                iterator,
                cancelled=cancelled,
            )
        )

        self.assertEqual(source.cancel_count, int(cancelled))
        self.assertEqual(iterator.cancel_count, int(cancelled))
        self.assertEqual(source.close_count, 1)
        self.assertEqual(iterator.close_count, 1)
        return elapsed_ms

    async def _mcp_cleanup_latency(self, *, cancelled: bool) -> float:
        source = _CleanupProbe()
        iterator = _CleanupProbe()
        elapsed_ms = await _elapsed_ms(
            lambda: mcp_router._cleanup_mcp_stream_sources(
                getLogger("tests.streaming_latency"),
                source,
                iterator,
                cancelled=cancelled,
            )
        )

        self.assertEqual(source.cancel_count, int(cancelled))
        self.assertEqual(iterator.cancel_count, int(cancelled))
        self.assertEqual(source.close_count, 1)
        self.assertEqual(iterator.close_count, 1)
        return elapsed_ms

    async def _a2a_cleanup_latency(self, *, cancelled: bool) -> float:
        source = _CleanupProbe()
        iterator = _CleanupProbe()

        async def cleanup() -> None:
            cleanup_error = await _cleanup_stream_sources_safely(
                source,
                iterator,
                cancelled=cancelled,
            )
            self.assertIsNone(cleanup_error)

        elapsed_ms = await _elapsed_ms(cleanup)

        self.assertEqual(source.cancel_count, int(cancelled))
        self.assertEqual(iterator.cancel_count, int(cancelled))
        self.assertEqual(source.close_count, 1)
        self.assertEqual(iterator.close_count, 1)
        return elapsed_ms

    async def _cancel_pull(self, pull: Any) -> None:
        pull.cancel()
        try:
            item = await wait_for(pull, STREAM_LATENCY_TEST_TIMEOUT)
        except CancelledError:
            return
        self.assertIs(item.kind, StreamItemKind.STREAM_CANCELLED)

    async def _cancel_task(self, task: Any) -> None:
        task.cancel()
        with self.assertRaises(CancelledError):
            await wait_for(task, STREAM_LATENCY_TEST_TIMEOUT)

    async def _cancel_a2a_pull(
        self,
        task: Any,
        iterator: AsyncIterator[str],
    ) -> None:
        task.cancel()
        try:
            await wait_for(task, STREAM_LATENCY_TEST_TIMEOUT)
        except CancelledError:
            return

        while True:
            pull = create_task(anext(iterator))
            done, _pending = await wait(
                {pull},
                timeout=STREAM_LATENCY_TEST_TIMEOUT,
                return_when=FIRST_COMPLETED,
            )
            self.assertIn(pull, done)
            try:
                await pull
            except (CancelledError, StopAsyncIteration):
                return


class StreamingLatencyBudgetValidationTestCase(TestCase):
    def test_cancellation_and_close_budgets_reject_non_positive_values(
        self,
    ) -> None:
        cases = (
            {"cancellation_latency_ms": 0},
            {"close_latency_ms": 0},
        )

        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(AssertionError):
                    StreamPerformanceBudget(**kwargs)
