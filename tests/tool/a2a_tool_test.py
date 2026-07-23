from base64 import b64encode
from collections.abc import AsyncIterator, Sequence
from contextlib import AsyncExitStack
from importlib import import_module
from logging import getLogger
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

from a2a.types import a2a_pb2
from fastapi import FastAPI
from httpx import ASGITransport

from avalan.entities import (
    Message,
    MessageContentFile,
    MessageContentText,
    MessageRole,
    ToolCall,
    ToolCallContext,
    ToolCallResult,
    ToolExecutionStreamEvent,
    ToolExecutionStreamKind,
)
from avalan.model import ModelCapabilityCatalog
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemCorrelation,
    StreamItemKind,
    StreamTerminalOutcome,
)
from avalan.server.entities import (
    ChatCompletionRequest,
    ContentFile,
    ContentText,
    OrchestratorContext,
)
from avalan.tool import a2a as a2a_module
from avalan.tool.a2a import A2ACallTool, A2AToolSet
from avalan.tool.manager import ToolManager
from avalan.tool.shell.settings import ShellToolSettings
from avalan.tool.shell.toolset import ShellToolSet


class _FakeSdkClient:
    def __init__(self, responses: Sequence[object]) -> None:
        self.responses = responses
        self.requests: list[Any] = []
        self.contexts: list[Any] = []
        self.entered = 0
        self.exited = 0

    async def __aenter__(self) -> "_FakeSdkClient":
        self.entered += 1
        return self

    async def __aexit__(self, *args: object) -> bool:
        self.exited += 1
        return False

    async def send_message(
        self, request: object, *, context: object = None
    ) -> AsyncIterator[object]:
        self.requests.append(request)
        self.contexts.append(context)
        for response in self.responses:
            if isinstance(response, BaseException):
                raise response
            yield response


class A2ACallToolTestCase(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.addCleanup(patch.stopall)
        self.client = _FakeSdkClient(_completed_a2a_responses(answer="ok"))
        self.created_cards: list[Any] = []
        self.created_configs: list[Any] = []

        async def create_client(
            agent: object, *, client_config: object
        ) -> _FakeSdkClient:
            self.created_cards.append(agent)
            self.created_configs.append(client_config)
            return self.client

        self.AsyncClient = MagicMock(return_value="httpx-client")
        patch("httpx.AsyncClient", self.AsyncClient).start()
        patch("a2a.client.create_client", new=create_client).start()
        patch("avalan.tool.a2a.uuid4", return_value="req-1").start()
        self.tool = A2ACallTool(
            call_params={"request_id": "req-1", "timeout": 1}
        )

    async def test_call_streams_answer_tool_and_status_events(self) -> None:
        self.client.responses = _completed_a2a_responses(
            answer="Hello", tool_output="stderr"
        )
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
        structured = cast(dict[str, Any], result["structuredContent"])
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
            timeout=None,
            headers={
                "A2A-Version": "1.0",
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
            },
        )
        card = self.created_cards[0]
        self.assertEqual(card.supported_interfaces[0].url, "http://host/a2a")
        self.assertEqual(
            card.supported_interfaces[0].protocol_binding, "JSONRPC"
        )
        request = self.client.requests[0]
        self.assertEqual(
            request.message.parts[0].text,
            "Ping",
        )
        self.assertEqual(request.metadata["skill"], "run")
        self.assertEqual(self.client.contexts[0].timeout, 1)
        self.assertEqual(self.client.entered, 1)
        self.assertEqual(self.client.exited, 1)

    async def test_call_without_arguments_uses_skill_name_message(
        self,
    ) -> None:
        result = await self.tool(
            "http://host/a2a", "run", None, context=ToolCallContext()
        )

        request = self.client.requests[0]
        self.assertEqual(request.message.parts[0].text, "run")
        self.assertEqual(result["content"], [{"type": "text", "text": "ok"}])

    async def test_call_forwards_context_input_files_as_a2a_parts(
        self,
    ) -> None:
        file_data = b64encode(b"%PDF-1.7").decode("ascii")
        input_message = Message(
            role=MessageRole.USER,
            content=[
                MessageContentText(type="text", text="Read this"),
                MessageContentFile(
                    type="file",
                    file={
                        "file_data": file_data,
                        "filename": "report.pdf",
                        "local_path": "/workspace/report.pdf",
                        "mime_type": "application/pdf",
                    },
                ),
            ],
        )

        await self.tool(
            "http://host/a2a",
            "run",
            {"input_string": "Summarize the attached PDF."},
            context=ToolCallContext(input=input_message),
        )

        request = self.client.requests[0]
        parts = request.message.parts
        self.assertEqual(parts[0].text, "Summarize the attached PDF.")
        self.assertEqual(parts[1].raw, b"%PDF-1.7")
        self.assertEqual(parts[1].filename, "report.pdf")
        self.assertEqual(parts[1].media_type, "application/pdf")
        self.assertNotIn("local_path", parts[1].metadata)

    async def test_file_part_helpers_cover_urls_and_invalid_payloads(
        self,
    ) -> None:
        single_file = MessageContentFile(
            type="file",
            file={
                "data": "data:application/pdf;base64,JVBERi0xLjc=",
                "file_name": "single.pdf",
                "localPath": "/workspace/single.pdf",
                "mediaType": "application/pdf",
            },
        )
        url_file = MessageContentFile(
            type="file",
            file={
                "name": "remote.pdf",
                "url": "https://files.example/remote.pdf",
                "mimeType": "application/pdf",
            },
        )
        invalid_file = MessageContentFile(
            type="file",
            file={"file_data": "not base64!"},
        )
        text_message = Message(
            role=MessageRole.USER,
            content=MessageContentText(type="text", text="plain"),
        )

        url_part = a2a_module._file_part(a2a_pb2, url_file)
        raw_part = a2a_module._file_part(a2a_pb2, single_file)

        self.assertIsNotNone(url_part)
        self.assertEqual(url_part.url, "https://files.example/remote.pdf")
        self.assertEqual(url_part.filename, "remote.pdf")
        self.assertEqual(url_part.media_type, "application/pdf")
        self.assertIsNotNone(raw_part)
        self.assertEqual(raw_part.raw, b"%PDF-1.7")
        self.assertEqual(raw_part.filename, "single.pdf")
        self.assertNotIn("local_path", raw_part.metadata)
        self.assertIsNone(a2a_module._file_part(a2a_pb2, invalid_file))
        self.assertIsNone(a2a_module._decode_file_data(None))
        self.assertIsNone(a2a_module._decode_file_data(" "))
        self.assertIsNone(a2a_module._decode_file_data("data:broken"))
        self.assertIsNone(a2a_module._decode_file_data("not base64!"))
        self.assertEqual(
            a2a_module._iter_input_file_content(
                [
                    text_message,
                    Message(role=MessageRole.USER, content=single_file),
                ]
            ),
            [single_file],
        )
        self.assertEqual(
            a2a_module._message_file_content(text_message),
            [],
        )

    async def test_call_serializes_non_text_arguments_and_merges_metadata(
        self,
    ) -> None:
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

        request = self.client.requests[0]
        self.assertEqual(
            request.message.parts[0].text,
            '{"count":2,"metadata":{"arg":"meta"}}',
        )
        self.assertEqual(request.message.message_id, "msg-1")
        self.assertEqual(request.metadata["skill"], "run")
        self.assertEqual(request.metadata["arg"], "meta")
        self.assertEqual(request.metadata["call"], "meta")
        self.assertEqual(request.metadata["arguments"]["count"], 2)

    async def test_passes_client_params_and_preserves_headers(self) -> None:
        tool = A2ACallTool(
            client_params={
                "headers": {"Authorization": "Bearer token"},
                "http2": True,
                "verify": False,
            },
            call_params={
                "request_id": "req-1",
                "service_parameters": {"A2A-Extensions": "demo"},
                "state": {"tenant": "demo"},
            },
        )

        await tool("http://host/a2a", "run", None, context=ToolCallContext())

        self.AsyncClient.assert_called_once_with(
            timeout=None,
            http2=True,
            verify=False,
            headers={
                "Authorization": "Bearer token",
                "A2A-Version": "1.0",
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
            },
        )
        self.assertIsNone(self.client.contexts[0].timeout)
        self.assertEqual(self.client.contexts[0].state, {"tenant": "demo"})
        self.assertEqual(
            self.client.contexts[0].service_parameters,
            {"A2A-Extensions": "demo"},
        )

    async def test_does_not_close_injected_httpx_client(self) -> None:
        httpx_client = MagicMock()
        httpx_client.aclose = AsyncMock()
        tool = A2ACallTool(
            client_params={"httpx_client": httpx_client},
            call_params={"request_id": "req-1"},
        )

        await tool("http://host/a2a", "run", None, context=ToolCallContext())

        self.AsyncClient.assert_not_called()
        self.assertIs(self.created_configs[0].httpx_client, httpx_client)
        self.assertEqual(self.client.entered, 0)
        self.assertEqual(self.client.exited, 0)
        httpx_client.aclose.assert_not_called()

    async def test_sdk_client_error_response_raises(self) -> None:
        self.client.responses = [RuntimeError("broken")]

        with self.assertRaisesRegex(RuntimeError, "broken"):
            await self.tool(
                "http://host/a2a", "run", None, context=ToolCallContext()
            )

    async def test_task_response_success_returns_after_single_message(
        self,
    ) -> None:
        self.client.responses = [
            _task_response(
                task_id="task-1",
                state=a2a_pb2.TaskState.TASK_STATE_COMPLETED,
            )
        ]

        result = await self.tool(
            "http://host/a2a", "run", None, context=ToolCallContext()
        )
        structured = cast(dict[str, Any], result["structuredContent"])

        self.assertEqual(structured["state"], "TASK_STATE_COMPLETED")

    async def test_message_response_success_returns_answer_and_streams(
        self,
    ) -> None:
        self.client.responses = [_message_response(text="direct")]
        events: list[ToolExecutionStreamEvent] = []

        async def stream(event: ToolExecutionStreamEvent) -> None:
            events.append(event)

        result = await self.tool(
            "http://host/a2a",
            "run",
            None,
            context=ToolCallContext(stream_event=stream),
        )
        structured = cast(dict[str, Any], result["structuredContent"])

        self.assertEqual(
            result["content"], [{"type": "text", "text": "direct"}]
        )
        self.assertEqual(structured["state"], "TASK_STATE_COMPLETED")
        self.assertEqual(structured["messages"][0]["text"], "direct")
        self.assertEqual(events[0].kind, ToolExecutionStreamKind.STDOUT)
        self.assertEqual(events[0].content, "direct")
        self.assertEqual(events[0].progress, 1)

    async def test_response_without_terminal_status_raises(self) -> None:
        self.client.responses = [
            _status_response(
                state=a2a_pb2.TaskState.TASK_STATE_WORKING,
            )
        ]

        with self.assertRaisesRegex(RuntimeError, "terminal event"):
            await self.tool(
                "http://host/a2a", "run", None, context=ToolCallContext()
            )

    async def test_failed_terminal_status_raises(self) -> None:
        self.client.responses = [
            _status_response(
                state=a2a_pb2.TaskState.TASK_STATE_FAILED,
            )
        ]

        with self.assertRaisesRegex(RuntimeError, "TASK_STATE_FAILED"):
            await self.tool(
                "http://host/a2a", "run", None, context=ToolCallContext()
            )

    async def test_ignores_stream_events_without_context_callback(
        self,
    ) -> None:
        result = await self.tool(
            "http://host/a2a", "run", None, context=ToolCallContext()
        )

        self.assertEqual(result["content"], [{"type": "text", "text": "ok"}])

    async def test_cancellation_checker_runs_before_stream_emit(self) -> None:
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

    async def test_tool_display_projector_returns_projection(self) -> None:
        projection = self.tool.tool_display_projector(
            ToolCall(
                id="call-1",
                name="a2a.call",
                arguments={"uri": "http://host/a2a", "name": "run"},
            )
        )

        self.assertIsNotNone(projection)

    async def test_tool_display_projector_returns_terminal_projection(
        self,
    ) -> None:
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
        projection = cast(Any, projection)
        self.assertEqual(projection.status, "completed")
        self.assertEqual(projection.scope, "A2A")

    async def test_state_records_task_message_snake_artifact_and_data_parts(
        self,
    ) -> None:
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
        structured = cast(dict[str, Any], result["structuredContent"])

        self.assertEqual(
            result["content"],
            [{"type": "text", "text": 'plain{"value":1}["x"]'}],
        )
        self.assertEqual(
            structured["messages"][0]["text"],
            'plain{"value":1}',
        )
        self.assertTrue(structured["artifacts"][0]["completed"])

    async def test_task_snapshot_artifacts_and_history_are_consumed(
        self,
    ) -> None:
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
        structured = cast(dict[str, Any], result["structuredContent"])

        self.assertEqual(result["content"], [{"type": "text", "text": "42"}])
        self.assertEqual(structured["messages"][0]["text"], "done")
        self.assertEqual(events[0].kind, ToolExecutionStreamKind.STDOUT)

    async def test_direct_helpers_cover_ignored_payloads_and_fallbacks(
        self,
    ) -> None:
        state = a2a_module._A2AStreamState()
        await state.process(
            a2a_module._stream_response_payload(a2a_pb2.StreamResponse()),
            ToolCallContext(),
        )
        await state.process(
            a2a_module._stream_response_payload(
                a2a_pb2.StreamResponse(
                    message=a2a_pb2.Message(
                        message_id="msg-1",
                        role=a2a_pb2.Role.ROLE_AGENT,
                        parts=[a2a_pb2.Part(text="done")],
                    )
                )
            ),
            ToolCallContext(),
        )
        result = state.result()
        structured = cast(dict[str, Any], result["structuredContent"])

        self.assertEqual(structured["messages"][0]["text"], "done")
        self.assertEqual(a2a_module._part_text({"data": 1}), "")
        self.assertTrue(a2a_module._is_answer_artifact({"id": "answer"}))
        self.assertEqual(
            a2a_module._stream_kind({}), ToolExecutionStreamKind.LOG
        )

    async def test_emit_helpers_skip_missing_callbacks_and_map_stdout(
        self,
    ) -> None:
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
    def test_default_namespace(self) -> None:
        toolset = A2AToolSet()
        self.assertEqual(toolset.namespace, "a2a")
        self.assertEqual(len(toolset.tools), 1)
        self.assertEqual(cast(Any, toolset.tools[0]).__name__, "call")

    def test_a2a_call_schema_does_not_collide_with_shell_pipeline(
        self,
    ) -> None:
        manager = ToolManager.create_instance(
            available_toolsets=[
                A2AToolSet(),
                ShellToolSet(settings=ShellToolSettings(allow_pipelines=True)),
            ],
            enable_tools=["a2a.call", "shell.pipeline"],
        )

        names = {descriptor.name for descriptor in manager.list_tools()}
        schemas = (
            ModelCapabilityCatalog.create(
                manager.export_model_capability_seed()
            )
            .project()
            .schemas
        )
        self.assertEqual(names, {"a2a.call", "shell.pipeline"})
        self.assertEqual(
            len({schema["function"]["name"] for schema in schemas}),
            2,
        )

        a2a_descriptor = manager.describe_tool("a2a.call")
        pipeline_descriptor = manager.describe_tool("shell.pipeline")
        self.assertIsNotNone(a2a_descriptor)
        self.assertIsNotNone(pipeline_descriptor)
        assert a2a_descriptor is not None
        assert pipeline_descriptor is not None
        a2a_properties = a2a_descriptor.schema["function"]["parameters"][
            "properties"
        ]
        pipeline_properties = pipeline_descriptor.schema["function"][
            "parameters"
        ]["properties"]
        self.assertIn("uri", a2a_properties)
        self.assertIn("name", a2a_properties)
        self.assertNotIn("steps", a2a_properties)
        self.assertIn("steps", pipeline_properties)
        self.assertNotIn("uri", pipeline_properties)


class A2ACallToolHttpE2ETestCase(IsolatedAsyncioTestCase):
    async def test_calls_sdk_v1_router_and_streams_status(self) -> None:
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
        response = _CanonicalResponse()

        async def stream(event: ToolExecutionStreamEvent) -> None:
            events.append(event)

        async def fake_orchestrate(
            *args: object, **kwargs: object
        ) -> tuple["_CanonicalResponse", UUID, int]:
            return response, uuid4(), 123

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
        self.assertIs(orchestrator.synced_responses[0], response)

        self.assertEqual(result["content"], [{"type": "text", "text": "25"}])
        structured = cast(dict[str, Any], result["structuredContent"])
        self.assertEqual(structured["state"], "TASK_STATE_COMPLETED")
        self.assertTrue(
            any(
                event.kind is ToolExecutionStreamKind.PROGRESS
                and "TASK_STATE_COMPLETED" in (event.content or "")
                for event in events
            )
        )

    async def test_calls_sdk_v1_router_with_forwarded_input_file(self) -> None:
        try:
            from avalan.server.a2a import router as a2a_router
        except ImportError as exc:
            self.skipTest(f"A2A SDK routes unavailable: {exc}")

        app = FastAPI()
        app.state.logger = getLogger("test.a2a.tool.file.e2e")
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
        captured_requests: list[ChatCompletionRequest] = []
        response = _CanonicalResponse()

        async def fake_orchestrate(
            request: ChatCompletionRequest, *args: object, **kwargs: object
        ) -> tuple["_CanonicalResponse", UUID, int]:
            captured_requests.append(request)
            return response, uuid4(), 123

        a2a_router.install_a2a_routes(
            app,
            prefix="/a2a",
            name="run",
            description="Run the test agent.",
        )

        raw_pdf = b"%PDF-1.7\nfixture"
        encoded_pdf = b64encode(raw_pdf).decode("ascii")
        input_message = Message(
            role=MessageRole.USER,
            content=[
                MessageContentText(type="text", text="Caller prompt"),
                MessageContentFile(
                    type="file",
                    file={
                        "file_data": encoded_pdf,
                        "filename": "report.pdf",
                        "local_path": "/workspace/report.pdf",
                        "mime_type": "application/pdf",
                    },
                ),
            ],
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
                    {"input_string": "Summarize the attached PDF."},
                    context=ToolCallContext(input=input_message),
                )

        self.assertEqual(result["content"], [{"type": "text", "text": "25"}])
        self.assertEqual(orchestrator.sync_count, 1)
        self.assertIs(orchestrator.synced_responses[0], response)
        self.assertEqual(len(captured_requests), 1)
        message = captured_requests[0].messages[0]
        content = message.content
        self.assertIsInstance(content, list)
        content = cast(list[object], content)
        self.assertIsInstance(content[0], ContentText)
        self.assertEqual(
            cast(ContentText, content[0]).text,
            "Summarize the attached PDF.",
        )
        self.assertIsInstance(content[1], ContentFile)
        file_content = cast(ContentFile, content[1])
        self.assertEqual(file_content.file_data, encoded_pdf)
        self.assertEqual(file_content.filename, "report.pdf")
        self.assertEqual(
            file_content.file,
            {
                "filename": "report.pdf",
                "mime_type": "application/pdf",
            },
        )

    async def test_served_pipeline_enabled_agent_streams_over_a2a(
        self,
    ) -> None:
        try:
            from avalan.server.a2a import router as a2a_router
        except ImportError as exc:
            self.skipTest(f"A2A SDK routes unavailable: {exc}")

        served_manager = _a2a_pipeline_manager(allow_pipelines=True)
        self.assertIsNotNone(served_manager.describe_tool("shell.pipeline"))
        app, stack, loader, orchestrator = _a2a_e2e_app(
            "test.a2a.tool.pipeline",
            tool=served_manager,
        )
        events: list[ToolExecutionStreamEvent] = []
        response = _CanonicalResponse(
            _a2a_pipeline_items_for_served_orchestrator(orchestrator)
        )

        async def stream(event: ToolExecutionStreamEvent) -> None:
            events.append(event)

        async def fake_orchestrate(
            *args: object, **kwargs: object
        ) -> tuple["_CanonicalResponse", UUID, int]:
            self.assertIs(args[2], orchestrator)
            return response, uuid4(), 123

        a2a_router.install_a2a_routes(
            app,
            prefix="/a2a",
            name="run",
            description="Run the pipeline agent.",
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
                    {"input_string": "Use the configured pipeline tool."},
                    context=ToolCallContext(stream_event=stream),
                )

        self.assertEqual(loader.from_file_calls, 1)
        self.assertEqual(orchestrator.sync_count, 1)
        self.assertIs(orchestrator.synced_responses[0], response)
        structured = cast(dict[str, Any], result["structuredContent"])
        artifacts = cast(list[dict[str, Any]], structured["artifacts"])
        pipeline_artifacts = [
            artifact
            for artifact in artifacts
            if artifact.get("name") == "shell.pipeline"
        ]
        self.assertTrue(pipeline_artifacts)
        payload = str(result) + "".join(
            event.content or "" for event in events
        )
        self.assertIn("stage warning", payload)
        self.assertIn("counted 2 lines", payload)
        self.assertIn("2\n", payload)
        self.assertNotIn("INTERMEDIATE_STDOUT_SHOULD_NOT_LEAK", payload)
        self.assertTrue(
            any(
                event.kind is ToolExecutionStreamKind.STDERR
                for event in events
            )
        )

    async def test_default_denied_served_agent_reports_pipeline_over_a2a(
        self,
    ) -> None:
        try:
            from avalan.server.a2a import router as a2a_router
        except ImportError as exc:
            self.skipTest(f"A2A SDK routes unavailable: {exc}")

        served_manager = _a2a_pipeline_manager(allow_pipelines=False)
        self.assertIsNone(served_manager.describe_tool("shell.pipeline"))
        app, stack, loader, orchestrator = _a2a_e2e_app(
            "test.a2a.tool.pipeline.denied",
            tool=served_manager,
        )
        response = _CanonicalResponse(
            _a2a_pipeline_items_for_served_orchestrator(orchestrator)
        )

        async def fake_orchestrate(
            *args: object, **kwargs: object
        ) -> tuple["_CanonicalResponse", UUID, int]:
            self.assertIs(args[2], orchestrator)
            return response, uuid4(), 123

        a2a_router.install_a2a_routes(
            app,
            prefix="/a2a",
            name="run",
            description="Run the default-denied agent.",
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
                    {"input_string": "Try shell.pipeline."},
                    context=ToolCallContext(),
                )

        self.assertEqual(loader.from_file_calls, 1)
        self.assertEqual(orchestrator.sync_count, 1)
        self.assertIs(orchestrator.synced_responses[0], response)
        structured = cast(dict[str, Any], result["structuredContent"])
        artifacts = cast(list[dict[str, Any]], structured["artifacts"])
        diagnostic_artifact = next(
            artifact
            for artifact in artifacts
            if artifact.get("name") == "shell.pipeline"
        )
        self.assertIn("tool.disabled", diagnostic_artifact["text"])
        self.assertIn("allow_pipelines", diagnostic_artifact["text"])


def _struct(payload: dict[str, object]) -> Any:
    struct_pb2 = import_module("google.protobuf.struct_pb2")
    value = struct_pb2.Struct()
    value.update(payload)
    return value


def _status_response(
    *,
    state: Any,
    task_id: str = "task-1",
    context_id: str = "ctx-1",
    metadata: dict[str, object] | None = None,
) -> a2a_pb2.StreamResponse:
    return a2a_pb2.StreamResponse(
        status_update=a2a_pb2.TaskStatusUpdateEvent(
            task_id=task_id,
            context_id=context_id,
            status=a2a_pb2.TaskStatus(state=state),
            metadata=_struct(metadata or {}),
        )
    )


def _artifact_response(
    *,
    artifact_id: str,
    text: str,
    metadata: dict[str, object],
    name: str | None = None,
    append: bool = True,
    last_chunk: bool = False,
) -> a2a_pb2.StreamResponse:
    return a2a_pb2.StreamResponse(
        artifact_update=a2a_pb2.TaskArtifactUpdateEvent(
            task_id="task-1",
            context_id="ctx-1",
            artifact=a2a_pb2.Artifact(
                artifact_id=artifact_id,
                name=name or "",
                parts=[a2a_pb2.Part(text=text)],
                metadata=_struct(metadata),
            ),
            append=append,
            last_chunk=last_chunk,
        )
    )


def _task_response(
    *,
    task_id: str,
    state: Any,
) -> a2a_pb2.StreamResponse:
    return a2a_pb2.StreamResponse(
        task=a2a_pb2.Task(
            id=task_id,
            context_id="ctx-1",
            status=a2a_pb2.TaskStatus(state=state),
        )
    )


def _message_response(*, text: str) -> a2a_pb2.StreamResponse:
    return a2a_pb2.StreamResponse(
        message=a2a_pb2.Message(
            message_id="msg-1",
            role=a2a_pb2.Role.ROLE_AGENT,
            parts=[a2a_pb2.Part(text=text)],
        )
    )


def _completed_a2a_responses(
    *, answer: str, tool_output: str | None = None
) -> list[a2a_pb2.StreamResponse]:
    responses = [
        _status_response(
            state=a2a_pb2.TaskState.TASK_STATE_WORKING,
            metadata={"phase": "start"},
        ),
        _artifact_response(
            artifact_id="answer",
            name="Answer",
            text=answer,
            metadata={
                "kind": "answer",
                "channel": "output",
            },
        ),
    ]
    if tool_output is not None:
        responses.append(
            _artifact_response(
                artifact_id="call-1",
                name="shell.run",
                text=tool_output,
                metadata={
                    "kind": "tool",
                    "category": "stderr",
                },
                last_chunk=True,
            )
        )
    responses.append(
        _status_response(
            state=a2a_pb2.TaskState.TASK_STATE_COMPLETED,
            metadata={"phase": "done"},
        )
    )
    return responses


class _E2EOrchestrator:
    model_ids = {"test-model"}

    def __init__(self, tool: ToolManager | None = None) -> None:
        self.id = UUID(int=2)
        self.sync_count = 0
        self.synced_responses: list[object] = []
        self._tool = tool

    @property
    def tool(self) -> ToolManager | None:
        return self._tool

    async def sync_messages(self, response: object) -> None:
        self.synced_responses.append(response)
        self.sync_count += 1


class _E2ELoader:
    def __init__(self, orchestrator: _E2EOrchestrator) -> None:
        self.orchestrator = orchestrator
        self.from_file_calls = 0

    async def from_file(
        self, *args: object, **kwargs: object
    ) -> "_E2EOrchestratorContext":
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

    def __init__(
        self, items: Sequence[CanonicalStreamItem] | None = None
    ) -> None:
        self._items = items

    async def to_str(self) -> str:
        return ""

    def __aiter__(self) -> AsyncIterator[CanonicalStreamItem]:
        if self._items is not None:
            return _iter_canonical_items(self._items)
        return self._iter()

    async def _iter(self) -> AsyncIterator[CanonicalStreamItem]:
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


def _a2a_e2e_app(
    logger_name: str,
    *,
    tool: ToolManager | None = None,
) -> tuple[FastAPI, AsyncExitStack, _E2ELoader, _E2EOrchestrator]:
    app = FastAPI()
    app.state.logger = getLogger(logger_name)
    orchestrator = _E2EOrchestrator(tool)
    loader = _E2ELoader(orchestrator)
    stack = AsyncExitStack()
    app.state.ctx = OrchestratorContext(
        participant_id=UUID(int=1),
        specs_path="agent.toml",
    )
    app.state.loader = loader
    app.state.stack = stack
    app.state.agent_id = None
    return app, stack, loader, orchestrator


def _a2a_pipeline_manager(*, allow_pipelines: bool) -> ToolManager:
    return ToolManager.create_instance(
        available_toolsets=[
            ShellToolSet(
                settings=ShellToolSettings(allow_pipelines=allow_pipelines)
            )
        ],
        enable_tools=["shell.pipeline"],
    )


def _a2a_pipeline_items_for_served_orchestrator(
    orchestrator: object,
) -> list[CanonicalStreamItem]:
    tool = getattr(orchestrator, "tool", None)
    assert isinstance(tool, ToolManager)
    if tool.describe_tool("shell.pipeline") is None:
        return _a2a_pipeline_denied_items()
    return _a2a_pipeline_success_items()


def _a2a_pipeline_success_items() -> list[CanonicalStreamItem]:
    correlation = StreamItemCorrelation(tool_call_id="call-pipeline")
    arguments = {
        "steps": [
            {"id": "read", "command": "cat", "paths": ["input.txt"]},
            {
                "id": "count",
                "command": "wc",
                "options": {"lines": True},
                "stdin_from": {"step_id": "read", "stream": "stdout"},
            },
        ]
    }
    return [
        _a2a_item(0, StreamItemKind.STREAM_STARTED),
        _a2a_item(
            1,
            StreamItemKind.TOOL_CALL_READY,
            correlation=correlation,
            data={"name": "shell.pipeline", "arguments": arguments},
            metadata={"tool_name": "shell.pipeline"},
        ),
        _a2a_item(2, StreamItemKind.TOOL_CALL_DONE, correlation=correlation),
        _a2a_item(
            3,
            StreamItemKind.TOOL_EXECUTION_STARTED,
            correlation=correlation,
            data={"name": "shell.pipeline", "arguments": arguments},
            metadata={"tool_name": "shell.pipeline"},
        ),
        _a2a_item(
            4,
            StreamItemKind.TOOL_EXECUTION_OUTPUT,
            correlation=correlation,
            text_delta="stage warning\n",
            data={"category": "stderr", "content": "stage warning\n"},
            metadata={"tool_name": "shell.pipeline"},
        ),
        _a2a_item(
            5,
            StreamItemKind.TOOL_EXECUTION_PROGRESS,
            correlation=correlation,
            data={
                "category": "progress",
                "content": "counted 2 lines",
                "progress": 0.5,
                "metadata": {
                    "intermediate_stdout": (
                        "INTERMEDIATE_STDOUT_SHOULD_NOT_LEAK"
                    ),
                },
            },
            metadata={"tool_name": "shell.pipeline"},
        ),
        _a2a_item(
            6,
            StreamItemKind.TOOL_EXECUTION_OUTPUT,
            correlation=correlation,
            text_delta="2\n",
            data={"category": "stdout", "content": "2\n"},
            metadata={"tool_name": "shell.pipeline"},
        ),
        _a2a_item(
            7,
            StreamItemKind.TOOL_EXECUTION_COMPLETED,
            correlation=correlation,
            data={
                "name": "shell.pipeline",
                "arguments": arguments,
                "result": (
                    "tool: shell.pipeline\nstatus: completed\nstdout:\n2\n"
                ),
            },
            metadata={"tool_name": "shell.pipeline"},
        ),
        _a2a_item(8, StreamItemKind.ANSWER_DELTA, text_delta="done"),
        _a2a_item(9, StreamItemKind.ANSWER_DONE),
        _a2a_item(10, StreamItemKind.USAGE_COMPLETED, usage={}),
        _a2a_item(
            11,
            StreamItemKind.STREAM_COMPLETED,
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
        ),
    ]


def _a2a_pipeline_denied_items() -> list[CanonicalStreamItem]:
    correlation = StreamItemCorrelation(tool_call_id="call-pipeline")
    diagnostic = {
        "id": "diag-pipeline",
        "call_id": "call-pipeline",
        "code": "tool.disabled",
        "message": "shell.pipeline requires allow_pipelines=true.",
    }
    return [
        _a2a_item(0, StreamItemKind.STREAM_STARTED),
        _a2a_item(
            1,
            StreamItemKind.TOOL_EXECUTION_STARTED,
            correlation=correlation,
            data={
                "name": "shell.pipeline",
                "arguments": {"steps": []},
            },
            metadata={"tool_name": "shell.pipeline"},
        ),
        _a2a_item(
            2,
            StreamItemKind.TOOL_EXECUTION_ERROR,
            correlation=correlation,
            data={
                "name": "shell.pipeline",
                "arguments": {"steps": []},
                "error": f"{diagnostic['code']}: {diagnostic['message']}",
                "diagnostic": diagnostic,
            },
            metadata={"tool_name": "shell.pipeline"},
        ),
        _a2a_item(3, StreamItemKind.USAGE_COMPLETED, usage={}),
        _a2a_item(
            4,
            StreamItemKind.STREAM_COMPLETED,
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
        ),
    ]


def _a2a_item(
    sequence: int,
    kind: StreamItemKind,
    *,
    channel: StreamChannel | None = None,
    **kwargs: Any,
) -> CanonicalStreamItem:
    if channel is None:
        channel = {
            StreamItemKind.ANSWER_DELTA: StreamChannel.ANSWER,
            StreamItemKind.ANSWER_DONE: StreamChannel.ANSWER,
            StreamItemKind.TOOL_CALL_READY: StreamChannel.TOOL_CALL,
            StreamItemKind.TOOL_CALL_DONE: StreamChannel.TOOL_CALL,
            StreamItemKind.TOOL_EXECUTION_STARTED: (
                StreamChannel.TOOL_EXECUTION
            ),
            StreamItemKind.TOOL_EXECUTION_OUTPUT: StreamChannel.TOOL_EXECUTION,
            StreamItemKind.TOOL_EXECUTION_PROGRESS: (
                StreamChannel.TOOL_EXECUTION
            ),
            StreamItemKind.TOOL_EXECUTION_COMPLETED: (
                StreamChannel.TOOL_EXECUTION
            ),
            StreamItemKind.TOOL_EXECUTION_ERROR: StreamChannel.TOOL_EXECUTION,
            StreamItemKind.TOOL_EXECUTION_CANCELLED: (
                StreamChannel.TOOL_EXECUTION
            ),
            StreamItemKind.USAGE_COMPLETED: StreamChannel.USAGE,
        }.get(kind, StreamChannel.CONTROL)
    return CanonicalStreamItem(
        stream_session_id="s",
        run_id="r",
        turn_id="t",
        sequence=sequence,
        kind=kind,
        channel=channel,
        **kwargs,
    )


async def _iter_canonical_items(
    items: Sequence[CanonicalStreamItem],
) -> AsyncIterator[CanonicalStreamItem]:
    for item in items:
        yield item
