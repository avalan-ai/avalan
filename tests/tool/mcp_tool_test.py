import sys
from logging import getLogger
from types import ModuleType
from typing import Any
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from fastapi import FastAPI
from httpx import ASGITransport

from avalan.entities import (
    ToolCall,
    ToolCallContext,
    ToolExecutionStreamEvent,
    ToolExecutionStreamKind,
    ToolFilter,
    ToolManagerSettings,
)
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemCorrelation,
    StreamItemKind,
    StreamTerminalOutcome,
)
from avalan.server.routers import mcp as mcp_router
from avalan.tool import mcp as mcp_module
from avalan.tool.manager import ToolManager
from avalan.tool.mcp import McpCallTool, McpToolSet


class _FakeResponse:
    def __init__(
        self,
        *,
        body: bytes | None = None,
        lines: list[str] | None = None,
        content_type: str,
    ) -> None:
        self._body = body or b""
        self._lines = lines or []
        self.headers = {"content-type": content_type}
        self.raise_for_status = MagicMock()

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, *args: object) -> bool:
        return False

    async def aread(self) -> bytes:
        return self._body

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeClient:
    def __init__(self, response: _FakeResponse, **kwargs: object) -> None:
        self.response = response
        self.kwargs = kwargs
        self.stream = MagicMock(return_value=response)

    async def __aenter__(self) -> "_FakeClient":
        return self

    async def __aexit__(self, *args: object) -> bool:
        return False


class McpCallToolTestCase(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.addCleanup(patch.stopall)
        self.response = _FakeResponse(
            body=(
                b'{"jsonrpc":"2.0","id":"req-1","result":'
                b'{"content":[{"type":"text","text":"result"}],'
                b'"structuredContent":{"toolCalls":[]}}}'
            ),
            content_type="application/json",
        )
        self.client = _FakeClient(self.response)
        self.AsyncClient = MagicMock(return_value=self.client)
        httpx_mod = ModuleType("httpx")
        httpx_mod.AsyncClient = self.AsyncClient
        patch.dict(sys.modules, {"httpx": httpx_mod}).start()
        patch("avalan.tool.mcp.uuid4", return_value="req-1").start()
        self.tool = McpCallTool(
            call_params={"request_id": "req-1", "timeout": 1}
        )

    async def test_call_with_arguments_returns_full_result(self):
        context = ToolCallContext()
        result = await self.tool(
            "http://host/mcp", "run", {"input_string": "hi"}, context=context
        )

        self.assertEqual(
            result,
            {
                "content": [{"type": "text", "text": "result"}],
                "structuredContent": {"toolCalls": []},
            },
        )
        self.AsyncClient.assert_called_once_with(
            headers={
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
            }
        )
        self.client.stream.assert_called_once()
        _, uri = self.client.stream.call_args.args
        self.assertEqual(uri, "http://host/mcp")
        request = self.client.stream.call_args.kwargs["json"]
        self.assertEqual(request["method"], "tools/call")
        self.assertEqual(request["params"]["name"], "run")
        self.assertEqual(
            request["params"]["arguments"], {"input_string": "hi"}
        )
        self.assertEqual(request["params"]["_meta"]["progressToken"], "req-1")
        self.assertEqual(self.client.stream.call_args.kwargs["timeout"], 1)

    async def test_call_without_arguments_sends_empty_arguments(self):
        context = ToolCallContext()
        await self.tool("http://host/mcp", "run", None, context=context)
        request = self.client.stream.call_args.kwargs["json"]
        self.assertEqual(request["params"]["arguments"], {})

    async def test_passes_client_params_and_preserves_headers(self):
        tool = McpCallTool(
            client_params={"headers": {"Authorization": "Bearer token"}},
            call_params={"request_id": "req-1"},
        )
        context = ToolCallContext()
        await tool("http://host/mcp", "run", None, context=context)
        self.AsyncClient.assert_called_once_with(
            headers={
                "Authorization": "Bearer token",
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
            }
        )

    async def test_streams_mcp_notifications_to_tool_context(self):
        self.response = _FakeResponse(
            content_type="text/event-stream",
            lines=[
                (
                    'data: {"jsonrpc":"2.0","method":"notifications/progress",'
                    '"params":{"progressToken":"req-1","progress":1,'
                    '"message":"{\\"type\\":\\"answer.delta\\",'
                    '\\"delta\\":\\"tok\\"}"}}'
                ),
                "",
                (
                    'data: {"jsonrpc":"2.0","method":"notifications/message",'
                    '"params":{"level":"info","data":{"type":"tool.call",'
                    '"toolCallId":"call-1","name":"shell.run"}}}'
                ),
                "",
                (
                    'data: {"jsonrpc":"2.0","method":'
                    '"notifications/resources/updated","params":{"resources":['
                    '{"uri":"mcp://resources/1","delta":{"set":'
                    '{"text":"stdout"}}}]}}'
                ),
                "",
                (
                    'data: {"jsonrpc":"2.0","id":"req-1","result":'
                    '{"content":[],"structuredContent":{"toolCalls":['
                    '{"id":"call-1","name":"shell.run"}]}}}'
                ),
                "",
            ],
        )
        self.client.response = self.response
        self.client.stream.return_value = self.response
        events: list[ToolExecutionStreamEvent] = []

        async def stream(event: ToolExecutionStreamEvent) -> None:
            events.append(event)

        result = await self.tool(
            "http://host/mcp",
            "run",
            {"input_string": "hi"},
            context=ToolCallContext(stream_event=stream),
        )

        self.assertEqual(
            result["structuredContent"],
            {"toolCalls": [{"id": "call-1", "name": "shell.run"}]},
        )
        self.assertEqual(events[0].kind, ToolExecutionStreamKind.STDOUT)
        self.assertEqual(events[0].content, "tok")
        self.assertEqual(events[1].kind, ToolExecutionStreamKind.LOG)
        self.assertIn('"type":"tool.call"', events[1].content or "")
        self.assertEqual(events[2].kind, ToolExecutionStreamKind.LOG)
        self.assertEqual(events[2].content, "stdout")

    async def test_error_response_raises(self):
        self.response = _FakeResponse(
            body=(
                b'{"jsonrpc":"2.0","id":"req-1","error":'
                b'{"code":-32603,"message":"broken"}}'
            ),
            content_type="application/json",
        )
        self.client.response = self.response
        self.client.stream.return_value = self.response

        with self.assertRaisesRegex(RuntimeError, "broken"):
            await self.tool(
                "http://host/mcp", "run", None, context=ToolCallContext()
            )

    async def test_invalid_json_response_raises(self):
        self.response = _FakeResponse(
            body=b"not-json", content_type="application/json"
        )
        self.client.response = self.response
        self.client.stream.return_value = self.response

        with self.assertRaisesRegex(RuntimeError, "Invalid MCP JSON-RPC"):
            await self.tool(
                "http://host/mcp", "run", None, context=ToolCallContext()
            )

    async def test_manager_filters_match_mcp_tool_not_remote_name(self):
        filtered_names: list[str] = []

        def filter_call(call: ToolCall, context: ToolCallContext):
            filtered_names.append(call.name)
            return (
                ToolCall(
                    id=call.id,
                    name=call.name,
                    arguments={
                        "uri": "http://host/mcp",
                        "name": "blocked",
                        "arguments": {},
                    },
                ),
                context,
            )

        manager = ToolManager.create_instance(
            enable_tools=["mcp.call"],
            available_toolsets=[
                McpToolSet(),
            ],
            settings=ToolManagerSettings(
                filters=[ToolFilter(func=filter_call, namespace="math")]
            ),
        )
        call = ToolCall(
            id="call-1",
            name="mcp.call",
            arguments={
                "uri": "http://host/mcp",
                "name": "math.calculator",
                "arguments": {"a": 1},
            },
        )

        result = await manager(call, context=ToolCallContext())

        self.assertEqual(
            result.result,
            {
                "content": [{"type": "text", "text": "result"}],
                "structuredContent": {"toolCalls": []},
            },
        )
        self.assertEqual(filtered_names, [])
        request = self.client.stream.call_args.kwargs["json"]
        self.assertEqual(request["params"]["name"], "math.calculator")
        self.assertEqual(request["params"]["arguments"], {"a": 1})

    async def test_tool_display_projector_returns_projection(self):
        projection = self.tool.tool_display_projector(
            ToolCall(
                id="call-1",
                name="mcp.call",
                arguments={"uri": "http://host/mcp", "name": "run"},
            )
        )

        self.assertIsNotNone(projection)

    async def test_exhausts_json_and_sse_response_iterators(self):
        json_response = _FakeResponse(
            body=b'{"jsonrpc":"2.0","id":"req-1","result":{}}',
            content_type="application/json",
        )
        json_messages = [
            message
            async for message in mcp_module._iter_mcp_response_messages(
                json_response
            )
        ]
        self.assertEqual(
            json_messages,
            [{"jsonrpc": "2.0", "id": "req-1", "result": {}}],
        )

        sse_response = _FakeResponse(
            content_type="text/event-stream",
            lines=[
                ": keepalive",
                (
                    '{"jsonrpc":"2.0","method":"notifications/message",'
                    '"params":{"level":"info","data":"raw"}}'
                ),
                'data: {"jsonrpc":"2.0","method":"notifications/message",',
                'data: "params":{"level":"info","data":"tail"}}',
            ],
        )
        sse_messages = [
            message
            async for message in mcp_module._iter_mcp_response_messages(
                sse_response
            )
        ]
        self.assertEqual(len(sse_messages), 2)
        self.assertEqual(sse_messages[0]["params"]["data"], "raw")
        self.assertEqual(sse_messages[1]["params"]["data"], "tail")

    async def test_non_object_json_response_raises(self):
        self.response = _FakeResponse(
            body=b'["not-object"]', content_type="application/json"
        )
        self.client.response = self.response
        self.client.stream.return_value = self.response

        with self.assertRaisesRegex(RuntimeError, "Invalid MCP JSON-RPC"):
            await self.tool(
                "http://host/mcp", "run", None, context=ToolCallContext()
            )

    async def test_response_without_terminal_result_raises(self):
        self.response = _FakeResponse(
            content_type="text/event-stream",
            lines=[
                (
                    'data: {"jsonrpc":"2.0","method":"notifications/message",'
                    '"params":{"level":"info","data":"still running"}}'
                ),
                "",
            ],
        )
        self.client.response = self.response
        self.client.stream.return_value = self.response

        with self.assertRaisesRegex(RuntimeError, "ended without a result"):
            await self.tool(
                "http://host/mcp", "run", None, context=ToolCallContext()
            )

    async def test_ignores_notifications_without_stream_or_params(self):
        await mcp_module._forward_mcp_notification(
            "notifications/message",
            {
                "jsonrpc": "2.0",
                "method": "notifications/message",
                "params": {"level": "info", "data": "ignored"},
            },
            ToolCallContext(),
        )

        events: list[ToolExecutionStreamEvent] = []

        async def stream(event: ToolExecutionStreamEvent) -> None:
            events.append(event)

        await mcp_module._forward_mcp_notification(
            "notifications/message",
            {"jsonrpc": "2.0", "method": "notifications/message"},
            ToolCallContext(stream_event=stream),
        )
        self.assertEqual(events, [])

    async def test_progress_and_message_edge_cases(self):
        events: list[ToolExecutionStreamEvent] = []

        async def stream(event: ToolExecutionStreamEvent) -> None:
            events.append(event)

        context = ToolCallContext(stream_event=stream)
        await mcp_module._forward_mcp_progress({}, context)
        await mcp_module._forward_mcp_progress({"message": 1}, context)
        await mcp_module._forward_mcp_progress(
            {"message": {"type": "answer.delta", "delta": ""}}, context
        )
        await mcp_module._forward_mcp_progress(
            {"message": {"type": "answer.completed"}}, context
        )
        await mcp_module._forward_mcp_progress(
            {"message": "not-json"}, context
        )
        await mcp_module._forward_mcp_message({"message": "legacy"}, context)
        await mcp_module._forward_mcp_message({}, context)

        self.assertEqual(events[0].kind, ToolExecutionStreamKind.PROGRESS)
        self.assertEqual(events[0].progress, 1)
        self.assertEqual(events[1].kind, ToolExecutionStreamKind.LOG)
        self.assertEqual(events[1].content, "legacy")
        self.assertEqual(len(events), 2)

    async def test_resource_notification_edge_cases(self):
        events: list[ToolExecutionStreamEvent] = []

        async def stream(event: ToolExecutionStreamEvent) -> None:
            events.append(event)

        context = ToolCallContext(stream_event=stream)
        for params in (
            {},
            {"resources": ["bad"]},
            {"resources": [{}]},
            {"resources": [{"delta": {}}]},
            {"resources": [{"delta": {"set": {}}}]},
        ):
            await mcp_module._forward_mcp_resources(params, context)

        self.assertEqual(events, [])

    async def test_cancellation_checker_runs_before_stream_emit(self):
        events: list[ToolExecutionStreamEvent] = []
        cancelled = False

        async def check_cancelled() -> None:
            nonlocal cancelled
            cancelled = True

        async def stream(event: ToolExecutionStreamEvent) -> None:
            events.append(event)

        await mcp_module._emit_mcp_stream_event(
            ToolCallContext(
                cancellation_checker=check_cancelled,
                stream_event=stream,
            ),
            kind=ToolExecutionStreamKind.LOG,
            content="log",
        )

        self.assertTrue(cancelled)
        self.assertEqual(events[0].content, "log")


class McpToolSetTestCase(TestCase):
    def test_default_namespace(self):
        toolset = McpToolSet()
        self.assertEqual(toolset.namespace, "mcp")
        self.assertEqual(len(toolset.tools), 1)
        self.assertEqual(toolset.tools[0].__name__, "call")


class McpCallToolHttpE2ETestCase(IsolatedAsyncioTestCase):
    async def test_calls_real_mcp_router_and_streams_remote_events(self):
        app = FastAPI()
        response = _CanonicalResponse(
            _mcp_e2e_items(
                answer="remote answer",
                tool_name="shell.run",
                tool_result={"ok": True},
                tool_output="stdout line\n",
            )
        )
        orchestrator = _E2EOrchestrator()
        app.state.logger = getLogger("test.mcp.tool.e2e")
        app.state.orchestrator = orchestrator
        app.include_router(mcp_router.create_router(), prefix="/mcp")
        events: list[ToolExecutionStreamEvent] = []

        async def stream(event: ToolExecutionStreamEvent) -> None:
            events.append(event)

        async def fake_orchestrate(*args: object, **kwargs: object):
            return response, uuid4(), 123

        with (
            patch.object(mcp_router, "Orchestrator", _E2EOrchestrator),
            patch.object(mcp_router, "resolve_model_id", return_value="gpt"),
            patch.object(mcp_router, "orchestrate", fake_orchestrate),
        ):
            tool = McpCallTool(
                client_params={
                    "transport": ASGITransport(app=app),
                    "base_url": "http://testserver",
                },
                call_params={"request_id": "req-1"},
            )
            result = await tool(
                "/mcp",
                "run",
                {"input_string": "Use a tool."},
                context=ToolCallContext(stream_event=stream),
            )

        self.assertEqual(
            result["content"],
            [{"type": "text", "text": "remote answer"}],
        )
        structured = result["structuredContent"]
        self.assertEqual(structured["model"], "gpt")
        self.assertEqual(
            structured["toolCalls"],
            [
                {
                    "id": "call-1",
                    "name": "shell.run",
                    "arguments": {},
                    "started": None,
                    "result": {"ok": True},
                    "resources": [
                        {
                            "uri": "mcp://resources/00000001",
                            "name": "stdout",
                        }
                    ],
                }
            ],
        )
        self.assertTrue(
            any(
                event.kind is ToolExecutionStreamKind.STDOUT
                and event.content == "remote answer"
                for event in events
            )
        )
        self.assertTrue(
            any(
                event.kind is ToolExecutionStreamKind.LOG
                and event.content == "stdout line\n"
                for event in events
            )
        )
        self.assertTrue(
            any(
                event.kind is ToolExecutionStreamKind.LOG
                and '"type":"tool.result"' in (event.content or "")
                for event in events
            )
        )
        orchestrator.sync_messages.assert_awaited_once()


class _E2EOrchestrator:
    sync_messages: AsyncMock

    def __init__(self) -> None:
        self.sync_messages = AsyncMock()


class _CanonicalResponse:
    input_token_count = 1
    output_token_count = 2
    _response_iterator = None

    def __init__(self, items: list[CanonicalStreamItem]) -> None:
        self._items = items

    async def to_str(self) -> str:
        return ""

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for item in self._items:
            yield item


def _mcp_e2e_items(
    *,
    answer: str,
    tool_name: str,
    tool_result: Any,
    tool_output: str,
) -> list[CanonicalStreamItem]:
    correlation = StreamItemCorrelation(tool_call_id="call-1")
    timings = {"started": None, "finished": None, "elapsed": None}
    return [
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
            text_delta=answer,
        ),
        CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=2,
            kind=StreamItemKind.TOOL_EXECUTION_STARTED,
            channel=StreamChannel.TOOL_EXECUTION,
            correlation=correlation,
            data={"name": tool_name, "arguments": {}, "timings": timings},
            metadata={"tool_name": tool_name},
        ),
        CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=3,
            kind=StreamItemKind.TOOL_EXECUTION_OUTPUT,
            channel=StreamChannel.TOOL_EXECUTION,
            correlation=correlation,
            text_delta=tool_output,
            data={"category": "stdout", "content": tool_output},
            metadata={"tool_name": tool_name},
        ),
        CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=4,
            kind=StreamItemKind.TOOL_EXECUTION_COMPLETED,
            channel=StreamChannel.TOOL_EXECUTION,
            correlation=correlation,
            data={
                "name": tool_name,
                "arguments": {},
                "result": tool_result,
                "timings": timings,
            },
            metadata={"tool_name": tool_name},
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
    ]
