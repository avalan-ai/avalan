import sys
from contextlib import AsyncExitStack
from json import dumps
from logging import getLogger
from types import ModuleType
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

from fastapi import FastAPI
from httpx import ASGITransport

from avalan.entities import (
    ToolCall,
    ToolCallContext,
    ToolCallResult,
    ToolExecutionStreamEvent,
    ToolExecutionStreamKind,
)
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemKind,
    StreamTerminalOutcome,
)
from avalan.server.entities import OrchestratorContext
from avalan.tool import a2a as a2a_module
from avalan.tool.a2a import A2ACallTool, A2AToolSet


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


class _IncompleteA2AHTTPResponse(a2a_module._A2AHTTPResponse):
    headers: dict[str, str] = {}


class A2ACallToolTestCase(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.addCleanup(patch.stopall)
        self.response = _FakeResponse(
            content_type="text/event-stream",
            lines=_completed_a2a_lines(answer="ok"),
        )
        self.client = _FakeClient(self.response)
        self.AsyncClient = MagicMock(return_value=self.client)
        httpx_mod = ModuleType("httpx")
        httpx_mod.AsyncClient = self.AsyncClient
        patch.dict(sys.modules, {"httpx": httpx_mod}).start()
        patch("avalan.tool.a2a.uuid4", return_value="req-1").start()
        self.tool = A2ACallTool(
            call_params={"request_id": "req-1", "timeout": 1}
        )

    async def test_incomplete_http_response_methods_raise(self):
        response = _IncompleteA2AHTTPResponse()

        with self.assertRaises(NotImplementedError):
            await response.aread()
        with self.assertRaises(NotImplementedError):
            response.aiter_lines()
        with self.assertRaises(NotImplementedError):
            response.raise_for_status()

    async def test_call_streams_answer_tool_and_status_events(self):
        self.response = _FakeResponse(
            content_type="text/event-stream",
            lines=_completed_a2a_lines(answer="Hello", tool_output="stderr"),
        )
        self.client.response = self.response
        self.client.stream.return_value = self.response
        events: list[ToolExecutionStreamEvent] = []

        async def stream(event: ToolExecutionStreamEvent) -> None:
            events.append(event)

        result = await self.tool(
            "http://host/a2a",
            "run",
            {"input_string": "Ping"},
            context=ToolCallContext(stream_event=stream),
        )

        self.assertEqual(
            result["content"], [{"type": "text", "text": "Hello"}]
        )
        structured = result["structuredContent"]
        self.assertEqual(structured["taskId"], "task-1")
        self.assertEqual(structured["state"], "TASK_STATE_COMPLETED")
        self.assertEqual(structured["artifacts"][0]["text"], "Hello")
        self.assertEqual(structured["artifacts"][1]["text"], "stderr")
        self.assertEqual(events[0].kind, ToolExecutionStreamKind.PROGRESS)
        self.assertIn("TASK_STATE_WORKING", events[0].content or "")
        self.assertEqual(events[1].kind, ToolExecutionStreamKind.STDOUT)
        self.assertEqual(events[1].content, "Hello")
        self.assertEqual(events[2].kind, ToolExecutionStreamKind.STDERR)
        self.assertEqual(events[2].content, "stderr")
        self.assertEqual(events[-1].progress, 1)

        self.AsyncClient.assert_called_once_with(
            headers={
                "A2A-Version": "1.0",
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
            }
        )
        self.client.stream.assert_called_once()
        _, uri = self.client.stream.call_args.args
        self.assertEqual(uri, "http://host/a2a")
        request = self.client.stream.call_args.kwargs["json"]
        self.assertEqual(request["method"], "SendStreamingMessage")
        self.assertEqual(
            request["params"]["message"]["parts"][0]["text"], "Ping"
        )
        self.assertEqual(request["params"]["metadata"]["skill"], "run")
        self.assertEqual(self.client.stream.call_args.kwargs["timeout"], 1)

    async def test_call_without_arguments_uses_skill_name_message(self):
        result = await self.tool(
            "http://host/a2a", "run", None, context=ToolCallContext()
        )

        request = self.client.stream.call_args.kwargs["json"]
        self.assertEqual(
            request["params"]["message"]["parts"], [{"text": "run"}]
        )
        self.assertEqual(result["content"], [{"type": "text", "text": "ok"}])

    async def test_call_serializes_non_text_arguments_and_merges_metadata(
        self,
    ):
        tool = A2ACallTool(
            call_params={
                "request_id": "req-1",
                "message_id": "msg-1",
                "metadata": {"call": "meta"},
            }
        )

        await tool(
            "http://host/a2a",
            "run",
            {"count": 2, "metadata": {"arg": "meta"}},
            context=ToolCallContext(),
        )

        request = self.client.stream.call_args.kwargs["json"]
        self.assertEqual(
            request["params"]["message"]["parts"][0]["text"],
            '{"count":2,"metadata":{"arg":"meta"}}',
        )
        self.assertEqual(request["params"]["message"]["messageId"], "msg-1")
        self.assertEqual(
            request["params"]["metadata"],
            {
                "skill": "run",
                "arg": "meta",
                "call": "meta",
                "arguments": {"count": 2, "metadata": {"arg": "meta"}},
            },
        )

    async def test_passes_client_params_and_preserves_headers(self):
        tool = A2ACallTool(
            client_params={"headers": {"Authorization": "Bearer token"}},
            call_params={"request_id": "req-1"},
        )

        await tool("http://host/a2a", "run", None, context=ToolCallContext())

        self.AsyncClient.assert_called_once_with(
            headers={
                "Authorization": "Bearer token",
                "A2A-Version": "1.0",
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
            }
        )
        self.assertIsNone(self.client.stream.call_args.kwargs["timeout"])

    async def test_json_rpc_error_response_raises(self):
        self.response = _FakeResponse(
            body=(
                b'{"jsonrpc":"2.0","id":"req-1","error":'
                b'{"code":-32000,"message":"broken"}}'
            ),
            content_type="application/json",
        )
        self.client.response = self.response
        self.client.stream.return_value = self.response

        with self.assertRaisesRegex(RuntimeError, "broken"):
            await self.tool(
                "http://host/a2a", "run", None, context=ToolCallContext()
            )

    async def test_json_response_success_returns_after_single_message(self):
        self.response = _FakeResponse(
            body=dumps(
                {
                    "jsonrpc": "2.0",
                    "id": "req-1",
                    "result": {
                        "statusUpdate": {
                            "taskId": "task-1",
                            "status": {"state": "TASK_STATE_COMPLETED"},
                        }
                    },
                },
                separators=(",", ":"),
            ).encode("utf-8"),
            content_type="application/json",
        )
        self.client.response = self.response
        self.client.stream.return_value = self.response

        result = await self.tool(
            "http://host/a2a", "run", None, context=ToolCallContext()
        )

        self.assertEqual(
            result["structuredContent"]["state"], "TASK_STATE_COMPLETED"
        )

    async def test_sse_comments_raw_json_and_trailing_data_are_consumed(self):
        self.response = _FakeResponse(
            content_type="text/event-stream",
            lines=[
                ": keepalive",
                dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": "other",
                        "result": {"statusUpdate": {}},
                    },
                    separators=(",", ":"),
                ),
                dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": "req-1",
                        "result": {"result": "ignored"},
                    },
                    separators=(",", ":"),
                ),
                _sse(
                    {
                        "jsonrpc": "2.0",
                        "id": "req-1",
                        "result": {
                            "statusUpdate": {
                                "task_id": "task-1",
                                "context_id": "ctx-1",
                                "status": {"state": "TASK_STATE_COMPLETED"},
                            }
                        },
                    }
                ),
            ],
        )
        self.client.response = self.response
        self.client.stream.return_value = self.response

        result = await self.tool(
            "http://host/a2a", "run", None, context=ToolCallContext()
        )

        self.assertEqual(
            result["structuredContent"]["state"], "TASK_STATE_COMPLETED"
        )

    async def test_invalid_json_response_raises(self):
        self.response = _FakeResponse(
            body=b"not-json", content_type="application/json"
        )
        self.client.response = self.response
        self.client.stream.return_value = self.response

        with self.assertRaisesRegex(RuntimeError, "Invalid A2A JSON-RPC"):
            await self.tool(
                "http://host/a2a", "run", None, context=ToolCallContext()
            )

    async def test_non_object_json_response_raises(self):
        self.response = _FakeResponse(
            body=b'["not-object"]', content_type="application/json"
        )
        self.client.response = self.response
        self.client.stream.return_value = self.response

        with self.assertRaisesRegex(RuntimeError, "Invalid A2A JSON-RPC"):
            await self.tool(
                "http://host/a2a", "run", None, context=ToolCallContext()
            )

    async def test_response_without_terminal_status_raises(self):
        self.response = _FakeResponse(
            content_type="text/event-stream",
            lines=[
                _sse(
                    {
                        "jsonrpc": "2.0",
                        "id": "req-1",
                        "result": {
                            "statusUpdate": {
                                "taskId": "task-1",
                                "status": {"state": "TASK_STATE_WORKING"},
                            }
                        },
                    }
                ),
                "",
            ],
        )
        self.client.response = self.response
        self.client.stream.return_value = self.response

        with self.assertRaisesRegex(RuntimeError, "terminal event"):
            await self.tool(
                "http://host/a2a", "run", None, context=ToolCallContext()
            )

    async def test_failed_terminal_status_raises(self):
        self.response = _FakeResponse(
            content_type="text/event-stream",
            lines=[
                _sse(
                    {
                        "jsonrpc": "2.0",
                        "id": "req-1",
                        "result": {
                            "statusUpdate": {
                                "taskId": "task-1",
                                "status": {"state": "TASK_STATE_FAILED"},
                                "final": True,
                            }
                        },
                    }
                ),
                "",
            ],
        )
        self.client.response = self.response
        self.client.stream.return_value = self.response

        with self.assertRaisesRegex(RuntimeError, "TASK_STATE_FAILED"):
            await self.tool(
                "http://host/a2a", "run", None, context=ToolCallContext()
            )

    async def test_ignores_stream_events_without_context_callback(self):
        result = await self.tool(
            "http://host/a2a", "run", None, context=ToolCallContext()
        )

        self.assertEqual(result["content"], [{"type": "text", "text": "ok"}])

    async def test_cancellation_checker_runs_before_stream_emit(self):
        events: list[ToolExecutionStreamEvent] = []
        cancelled = False

        async def check_cancelled() -> None:
            nonlocal cancelled
            cancelled = True

        async def stream(event: ToolExecutionStreamEvent) -> None:
            events.append(event)

        await a2a_module._emit_a2a_stream_event(
            ToolCallContext(
                cancellation_checker=check_cancelled,
                stream_event=stream,
            ),
            kind=ToolExecutionStreamKind.LOG,
            content="log",
        )

        self.assertTrue(cancelled)
        self.assertEqual(events[0].content, "log")

    async def test_tool_display_projector_returns_projection(self):
        projection = self.tool.tool_display_projector(
            ToolCall(
                id="call-1",
                name="a2a.call",
                arguments={"uri": "http://host/a2a", "name": "run"},
            )
        )

        self.assertIsNotNone(projection)

    async def test_tool_display_projector_returns_terminal_projection(self):
        call = ToolCall(
            id="call-1",
            name="a2a.call",
            arguments={"uri": "http://host/a2a", "name": "run"},
        )
        projection = self.tool.tool_display_projector(
            call,
            ToolCallResult(
                id="result-1",
                name=call.name,
                arguments=call.arguments,
                call=call,
                result={"content": [{"type": "text", "text": "25"}]},
            ),
        )

        self.assertIsNotNone(projection)
        self.assertEqual(projection.status, "completed")
        self.assertEqual(projection.scope, "A2A")

    async def test_state_records_task_message_snake_artifact_and_data_parts(
        self,
    ):
        state = a2a_module._A2AStreamState()
        await state.process(
            {
                "task": {
                    "id": "task-1",
                    "context_id": "ctx-1",
                    "status": {"state": ""},
                }
            },
            ToolCallContext(),
        )
        await state.process(
            {
                "message": {
                    "message_id": "msg-1",
                    "role": "ROLE_AGENT",
                    "parts": [
                        {"data": "plain"},
                        {"data": {"value": 1}},
                        "ignored",
                    ],
                    "metadata": [],
                }
            },
            ToolCallContext(),
        )
        await state.process(
            {
                "artifact_update": {
                    "task_id": "task-1",
                    "context_id": "ctx-1",
                    "artifact": {
                        "id": "answer",
                        "parts": [{"data": ["x"]}],
                    },
                    "last_chunk": True,
                }
            },
            ToolCallContext(),
        )
        result = state.result()

        self.assertEqual(
            result["content"], [{"type": "text", "text": '["x"]'}]
        )
        self.assertEqual(
            result["structuredContent"]["messages"][0]["text"],
            'plain{"value":1}',
        )
        self.assertTrue(
            result["structuredContent"]["artifacts"][0]["completed"]
        )

    async def test_task_snapshot_artifacts_and_history_are_consumed(self):
        state = a2a_module._A2AStreamState()
        events: list[ToolExecutionStreamEvent] = []

        async def stream(event: ToolExecutionStreamEvent) -> None:
            events.append(event)

        await state.process(
            {
                "task": {
                    "id": "task-1",
                    "contextId": "ctx-1",
                    "status": {"state": "TASK_STATE_COMPLETED"},
                    "artifacts": [
                        {
                            "artifactId": "answer",
                            "name": "Answer",
                            "parts": [{"text": "42"}],
                            "metadata": {"kind": "answer"},
                        }
                    ],
                    "history": [
                        {
                            "messageId": "msg-1",
                            "role": "ROLE_AGENT",
                            "parts": [{"text": "done"}],
                        }
                    ],
                }
            },
            ToolCallContext(stream_event=stream),
        )
        result = state.result()

        self.assertEqual(result["content"], [{"type": "text", "text": "42"}])
        self.assertEqual(
            result["structuredContent"]["messages"][0]["text"], "done"
        )
        self.assertEqual(events[0].kind, ToolExecutionStreamKind.STDOUT)

    async def test_direct_helpers_cover_ignored_payloads_and_fallbacks(self):
        state = a2a_module._A2AStreamState()
        await a2a_module._handle_a2a_message(
            {"id": "req-1", "result": "ignored"},
            state=state,
            request_id="req-1",
            context=ToolCallContext(),
        )

        self.assertEqual(a2a_module._part_text({"data": 1}), "")
        self.assertTrue(a2a_module._is_answer_artifact({"id": "answer"}))
        self.assertEqual(
            a2a_module._stream_kind({}), ToolExecutionStreamKind.LOG
        )

    async def test_emit_helpers_skip_missing_callbacks_and_map_stdout(self):
        await a2a_module._emit_status_update(
            {"state": "TASK_STATE_WORKING"}, ToolCallContext()
        )
        await a2a_module._emit_artifact_update(
            {"id": "artifact", "metadata": {"kind": "tool"}},
            [],
            ToolCallContext(stream_event=MagicMock()),
        )
        events: list[ToolExecutionStreamEvent] = []

        async def stream(event: ToolExecutionStreamEvent) -> None:
            events.append(event)

        await a2a_module._emit_artifact_update(
            {
                "id": "artifact",
                "metadata": {"kind": "tool", "category": "stdout"},
            },
            ["out"],
            ToolCallContext(stream_event=stream),
        )

        self.assertEqual(events[0].kind, ToolExecutionStreamKind.STDOUT)
        self.assertEqual(a2a_module._event_metadata([]), {})


class A2AToolSetTestCase(TestCase):
    def test_default_namespace(self):
        toolset = A2AToolSet()
        self.assertEqual(toolset.namespace, "a2a")
        self.assertEqual(len(toolset.tools), 1)
        self.assertEqual(toolset.tools[0].__name__, "call")


class A2ACallToolHttpE2ETestCase(IsolatedAsyncioTestCase):
    async def test_calls_sdk_v1_router_and_streams_status(self):
        try:
            from avalan.server.a2a import router as a2a_router
        except ImportError as exc:
            self.skipTest(f"A2A SDK routes unavailable: {exc}")

        app = FastAPI()
        app.state.logger = getLogger("test.a2a.tool.e2e")
        orchestrator = _E2EOrchestrator()
        loader = _E2ELoader(orchestrator)
        stack = AsyncExitStack()
        app.state.ctx = OrchestratorContext(
            participant_id=UUID(int=1),
            specs_path="agent.toml",
        )
        app.state.loader = loader
        app.state.stack = stack
        app.state.agent_id = None
        events: list[ToolExecutionStreamEvent] = []

        async def stream(event: ToolExecutionStreamEvent) -> None:
            events.append(event)

        async def fake_orchestrate(*args: object, **kwargs: object):
            return _CanonicalResponse(), uuid4(), 123

        a2a_router.install_a2a_routes(
            app,
            prefix="/a2a",
            name="run",
            description="Run the test agent.",
        )

        async with stack:
            with patch.object(a2a_router, "orchestrate", fake_orchestrate):
                tool = A2ACallTool(
                    client_params={
                        "transport": ASGITransport(app=app),
                        "base_url": "http://testserver",
                    },
                    call_params={"request_id": "req-1"},
                )
                result = await tool(
                    "/a2a",
                    "run",
                    {"input_string": "Ping"},
                    context=ToolCallContext(stream_event=stream),
                )

        self.assertEqual(loader.from_file_calls, 1)
        self.assertIs(app.state.orchestrator, orchestrator)
        self.assertEqual(orchestrator.sync_count, 1)

        self.assertEqual(result["content"], [{"type": "text", "text": "25"}])
        self.assertEqual(
            result["structuredContent"]["state"], "TASK_STATE_COMPLETED"
        )
        self.assertTrue(
            any(
                event.kind is ToolExecutionStreamKind.PROGRESS
                and "TASK_STATE_COMPLETED" in (event.content or "")
                for event in events
            )
        )


def _sse(payload: dict[str, object]) -> str:
    return f"data: {dumps(payload, separators=(',', ':'))}"


def _completed_a2a_lines(
    *, answer: str, tool_output: str | None = None
) -> list[str]:
    lines = [
        _sse(
            {
                "jsonrpc": "2.0",
                "id": "req-1",
                "result": {
                    "statusUpdate": {
                        "taskId": "task-1",
                        "contextId": "ctx-1",
                        "status": {"state": "TASK_STATE_WORKING"},
                        "metadata": {"phase": "start"},
                    }
                },
            }
        ),
        "",
        _sse(
            {
                "jsonrpc": "2.0",
                "id": "req-1",
                "result": {
                    "artifactUpdate": {
                        "taskId": "task-1",
                        "contextId": "ctx-1",
                        "artifact": {
                            "artifactId": "answer",
                            "name": "Answer",
                            "parts": [{"text": answer}],
                            "metadata": {
                                "kind": "answer",
                                "channel": "output",
                            },
                        },
                        "append": True,
                    }
                },
            }
        ),
        "",
    ]
    if tool_output is not None:
        lines.extend(
            [
                _sse(
                    {
                        "jsonrpc": "2.0",
                        "id": "req-1",
                        "result": {
                            "artifactUpdate": {
                                "taskId": "task-1",
                                "contextId": "ctx-1",
                                "artifact": {
                                    "artifactId": "call-1",
                                    "name": "shell.run",
                                    "parts": [{"text": tool_output}],
                                    "metadata": {
                                        "kind": "tool",
                                        "category": "stderr",
                                    },
                                },
                                "append": True,
                                "lastChunk": True,
                            }
                        },
                    }
                ),
                "",
            ]
        )
    lines.extend(
        [
            _sse(
                {
                    "jsonrpc": "2.0",
                    "id": "req-1",
                    "result": {
                        "statusUpdate": {
                            "taskId": "task-1",
                            "contextId": "ctx-1",
                            "status": {"state": "TASK_STATE_COMPLETED"},
                            "final": True,
                            "metadata": {"phase": "done"},
                        }
                    },
                }
            ),
            "",
        ]
    )
    return lines


class _E2EOrchestrator:
    model_ids = {"test-model"}

    def __init__(self) -> None:
        self.id = UUID(int=2)
        self.sync_count = 0

    async def sync_messages(self) -> None:
        self.sync_count += 1


class _E2ELoader:
    def __init__(self, orchestrator: _E2EOrchestrator) -> None:
        self.orchestrator = orchestrator
        self.from_file_calls = 0

    async def from_file(self, *args: object, **kwargs: object):
        self.from_file_calls += 1
        return _E2EOrchestratorContext(self.orchestrator)


class _E2EOrchestratorContext:
    def __init__(self, orchestrator: _E2EOrchestrator) -> None:
        self.orchestrator = orchestrator

    async def __aenter__(self) -> _E2EOrchestrator:
        return self.orchestrator

    async def __aexit__(self, *args: object) -> bool:
        return False


class _CanonicalResponse:
    input_token_count = 1
    output_token_count = 0
    _response_iterator = None

    async def to_str(self) -> str:
        return ""

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
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
            text_delta="25",
        )
        yield CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=2,
            kind=StreamItemKind.ANSWER_DONE,
            channel=StreamChannel.ANSWER,
        )
        yield CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=3,
            kind=StreamItemKind.USAGE_COMPLETED,
            channel=StreamChannel.USAGE,
            usage={},
        )
        yield CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=4,
            kind=StreamItemKind.STREAM_COMPLETED,
            channel=StreamChannel.CONTROL,
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
        )
