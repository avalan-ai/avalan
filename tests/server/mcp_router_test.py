from asyncio import (
    CancelledError,
    create_task,
    run,
)
from asyncio import (
    Event as AsyncEvent,
)
from contextlib import suppress
from json import dumps, loads
from logging import getLogger
from types import SimpleNamespace
from typing import Any, AsyncIterator
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

from avalan.entities import (
    ReasoningToken,
    Token,
    TokenDetail,
    ToolCall,
    ToolCallDiagnostic,
    ToolCallDiagnosticCode,
    ToolCallDiagnosticStage,
    ToolCallError,
    ToolCallResult,
    ToolCallToken,
)
from avalan.event import Event, EventType
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemCorrelation,
    StreamItemKind,
    StreamTerminalOutcome,
)
from avalan.server.entities import (
    ChatCompletionRequest,
    ChatMessage,
    MCPToolRequest,
)
from avalan.server.routers import mcp as mcp_router


class DummyRequest:
    def __init__(self, body: bytes | list[bytes]) -> None:
        if isinstance(body, list):
            self._body = body
        else:
            self._body = [body]
        self.state = SimpleNamespace()
        self.app = SimpleNamespace(state=SimpleNamespace())

    async def stream(self):
        for chunk in self._body:
            yield chunk


class DummyResponse:
    def __init__(self, items: list[Any]) -> None:
        self._items = items
        self._index = 0
        self.input_token_count = 3
        self.output_token_count = 2
        self._response_iterator = None
        self._closed = False
        self.text = ""

    def __aiter__(self):
        self._index = 0
        self._response_iterator = self
        return self

    async def __anext__(self):
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item() if callable(item) else item

    async def aclose(self) -> None:
        self._closed = True

    async def to_str(self) -> str:
        return self.text


class DummyOrchestrator(mcp_router.Orchestrator):
    def __init__(self, tool: Any) -> None:
        self._tool = tool

    @property
    def tool(self) -> Any:
        return self._tool


class StubTask:
    def __init__(self, coro: Any) -> None:
        self._coro = coro

    def cancel(self) -> None:
        with suppress(RuntimeError):
            self._coro.close()

    def __await__(self):  # type: ignore[override]
        async def done() -> None:
            return None

        return done().__await__()


def fake_create_task(coro: Any) -> StubTask:
    return StubTask(coro)


class OneTimeMatchStr(str):
    def __new__(cls, value: str) -> "OneTimeMatchStr":
        instance = super().__new__(cls, value)
        instance._first_compare = True  # type: ignore[attr-defined]
        return instance

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        if getattr(self, "_first_compare", False) and super().__eq__(other):
            self._first_compare = False  # type: ignore[attr-defined]
            return True
        return False

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(str(self))


class MCPResourceStoreTestCase(TestCase):
    def test_create_append_close(self) -> None:
        store = mcp_router.MCPResourceStore()
        resource = run(store.create(base_path="/m"))
        self.assertEqual(resource.text, "")
        updated = run(store.append(resource.id, "chunk"))
        self.assertEqual(updated.text, "chunk")
        closed = run(store.close(resource.id))
        self.assertTrue(closed.closed)
        fetched = run(store.get(resource.id))
        self.assertEqual(fetched.text, "chunk")

    def test_close_resource_notifications_closes_all(self) -> None:
        store = mcp_router.MCPResourceStore()
        first = run(store.create(base_path="/m", initial_text="one"))
        second = run(store.create(base_path="/m", initial_text="two"))

        notifications = run(
            mcp_router._close_mcp_resource_notifications(
                store, {"first": first, "second": second}
            )
        )

        self.assertEqual(
            run(mcp_router._close_mcp_resource_notifications(store, {})),
            (),
        )
        self.assertEqual(len(notifications), 2)
        self.assertTrue(run(store.get(first.id)).closed)
        self.assertTrue(run(store.get(second.id)).closed)
        self.assertTrue(notifications[0]["params"]["resources"][0]["closed"])
        self.assertTrue(notifications[1]["params"]["resources"][0]["closed"])

    def test_retains_only_configured_resource_history(self) -> None:
        store = mcp_router.MCPResourceStore(resource_item_limit=2)
        resource = run(store.create(base_path="/m", initial_text="one"))
        second = run(store.append(resource.id, "two"))
        third = run(store.append(resource.id, "three"))

        self.assertEqual(second.text, "onetwo")
        self.assertEqual(third.text, "twothree")
        self.assertEqual(run(store.history(resource.id)), ("two", "three"))
        self.assertEqual(third.revision, 3)

    def test_zero_resource_history_discards_text_but_keeps_resource(
        self,
    ) -> None:
        store = mcp_router.MCPResourceStore(resource_item_limit=0)
        resource = run(store.create(base_path="/m", initial_text="one"))
        updated = run(store.append(resource.id, "two"))
        fetched = run(store.get(resource.id))

        self.assertEqual(resource.text, "")
        self.assertEqual(updated.text, "")
        self.assertEqual(fetched.text, "")
        self.assertEqual(run(store.history(resource.id)), ())
        self.assertEqual(updated.revision, 2)

    def test_rejects_invalid_resource_item_limit(self) -> None:
        with self.assertRaises(AssertionError):
            mcp_router.MCPResourceStore(resource_item_limit=-1)
        with self.assertRaises(AssertionError):
            mcp_router.MCPResourceStore(  # type: ignore[arg-type]
                resource_item_limit=True
            )

    def test_extract_append_streams(self) -> None:
        streams = mcp_router._extract_append_streams(
            "call-1", {"stdout": "out", "stderr": "", "logs": "log"}
        )
        self.assertIn("call-1:stdout", streams)
        self.assertIn("call-1:logs", streams)

    def test_extract_append_streams_ignores_non_dict(self) -> None:
        self.assertEqual(
            mcp_router._extract_append_streams("call", [1, 2]), {}
        )

    def test_close_is_idempotent(self) -> None:
        store = mcp_router.MCPResourceStore()
        resource = run(store.create(base_path="/base"))
        first = run(store.close(resource.id))
        self.assertTrue(first.closed)
        second = run(store.close(resource.id))
        self.assertEqual(second.revision, first.revision)

    def test_ensure_raises_for_missing_resource(self) -> None:
        store = mcp_router.MCPResourceStore()
        with self.assertRaises(KeyError):
            run(store.append("missing", "text"))


class MCPUtilityTestCase(TestCase):
    def _request(self) -> DummyRequest:
        request = DummyRequest(b"")
        request.app.title = ""
        request.app.version = None
        return request

    def test_extract_call_arguments_invalid_tool(self) -> None:
        with self.assertRaises(mcp_router.HTTPException) as exc:
            mcp_router._extract_call_arguments(
                "tools/call",
                {"name": "invalid", "arguments": {}},
                allowed_tool_name="run",
            )
        self.assertIn("Unsupported tool", str(exc.exception.detail))

    def test_extract_call_arguments_missing_arguments(self) -> None:
        with self.assertRaises(mcp_router.HTTPException) as exc:
            mcp_router._extract_call_arguments(
                "tools/call",
                {"name": "run", "arguments": "value"},
                allowed_tool_name="run",
            )
        self.assertIn("Invalid tool arguments", str(exc.exception.detail))

    def test_extract_call_arguments_unsupported_method(self) -> None:
        with self.assertRaises(mcp_router.HTTPException):
            mcp_router._extract_call_arguments(
                "other", {}, allowed_tool_name="run"
            )

    def test_collect_tool_descriptions(self) -> None:
        req = self._request()
        req.app.state.mcp_tool_name = "run"
        req.app.state.mcp_tool_description = "Desc"
        descriptions = mcp_router._collect_tool_descriptions(req)
        self.assertEqual(len(descriptions), 1)
        self.assertEqual(descriptions[0]["name"], "run")
        self.assertEqual(descriptions[0]["description"], "Desc")
        self.assertEqual(
            descriptions[0]["inputSchema"],
            MCPToolRequest.model_json_schema(),
        )

    def test_default_model_id_selects_first_sorted_candidate(self) -> None:
        orchestrator = SimpleNamespace(model_ids={"beta", "alpha"})
        model_id = mcp_router._default_model_id(orchestrator)
        self.assertEqual(model_id, "alpha")

    def test_default_model_id_fallback_without_candidates(self) -> None:
        orchestrator = SimpleNamespace(model_ids=[])
        self.assertEqual(
            mcp_router._default_model_id(orchestrator),
            mcp_router.MODEL_FALLBACK,
        )

        missing = SimpleNamespace()
        self.assertEqual(
            mcp_router._default_model_id(missing),
            mcp_router.MODEL_FALLBACK,
        )

    def test_build_chat_request_uses_default_model(self) -> None:
        orchestrator = SimpleNamespace(model_ids={"zeta", "alpha"})
        tool_request = MCPToolRequest(input_string="ping")

        chat_request = mcp_router._build_chat_request(
            tool_request, orchestrator
        )

        self.assertIsInstance(chat_request, ChatCompletionRequest)
        self.assertTrue(chat_request.stream)
        self.assertEqual(chat_request.model, "alpha")
        self.assertEqual(len(chat_request.messages), 1)

        message = chat_request.messages[0]
        self.assertIsInstance(message, ChatMessage)
        self.assertEqual(message.role, "user")
        self.assertEqual(message.content, "ping")

    def test_token_text_variants(self) -> None:
        token = Token(token="a")
        detail = TokenDetail(token="b")
        control = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=0,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
        )
        answer = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=1,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta="answer",
        )
        self.assertEqual(mcp_router._token_text(token), "a")
        self.assertEqual(mcp_router._token_text(detail), "b")
        self.assertEqual(mcp_router._token_text(control), "")
        self.assertEqual(mcp_router._token_text(answer), "answer")
        self.assertEqual(mcp_router._token_text("text"), "text")
        self.assertEqual(mcp_router._token_text(123), "")

    def test_token_text_prefers_token_detail(self) -> None:
        detail = TokenDetail(token="detail")
        with patch.object(mcp_router, "Token", type("DifferentToken", (), {})):
            self.assertEqual(mcp_router._token_text(detail), "detail")

    def test_reasoning_delta_variants(self) -> None:
        token = ReasoningToken(token="legacy")
        control = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=0,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
        )
        answer = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=1,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta="answer",
        )

        self.assertEqual(mcp_router._reasoning_delta(token), "legacy")
        self.assertEqual(mcp_router._reasoning_delta(control), "")
        self.assertIsNone(mcp_router._reasoning_delta(answer))

    def test_tool_call_token_notification_variants(self) -> None:
        empty = ToolCallToken(token="")
        self.assertIsNone(mcp_router._tool_call_token_notification(empty))

        delta = ToolCallToken(token="chunk")
        message = mcp_router._tool_call_token_notification(delta)
        self.assertEqual(
            message["params"]["message"]["delta"],
            "chunk",
        )

        call = ToolCall(id="t", name="run", arguments={})
        token = ToolCallToken(token="ignored", call=call)
        notification = mcp_router._tool_call_token_notification(token)
        payload = notification["params"]["message"]
        self.assertEqual(payload["toolCallId"], "t")
        self.assertEqual(
            loads(payload["delta"]),
            {"id": "t", "name": "run", "arguments": {}},
        )

    def test_canonical_progress_notification_empty_answer_delta(
        self,
    ) -> None:
        item = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=0,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta="",
        )

        self.assertIsNone(
            mcp_router._canonical_progress_notification(item, "progress")
        )

    def test_canonical_error_message_defaults_without_message(self) -> None:
        accumulator = mcp_router.ProtocolStreamAccumulator()
        accumulator.add(
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            )
        )
        accumulator.add(
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=1,
                kind=StreamItemKind.STREAM_ERRORED,
                channel=StreamChannel.CONTROL,
                terminal_outcome=StreamTerminalOutcome.ERRORED,
            )
        )

        self.assertEqual(
            mcp_router._canonical_error_message(accumulator.snapshot()),
            "Stream errored.",
        )

    def test_resource_notification_variants(self) -> None:
        resource = mcp_router.MCPResource(
            id="1",
            uri="uri",
            http_uri="/res/1",
            mime_type="text/plain",
            text="payload",
            revision=1,
        )
        open_payload = mcp_router._resource_notification(resource)
        self.assertEqual(
            open_payload["params"]["resources"][0]["delta"]["set"]["text"],
            "payload",
        )

        resource.closed = True
        closed_payload = mcp_router._resource_notification(resource)
        self.assertTrue(closed_payload["params"]["resources"][0]["closed"])

    def test_append_tool_summary_resource_skips_malformed_entries(
        self,
    ) -> None:
        resources: list[Any] = ["malformed"]
        mcp_router._append_tool_summary_resource(
            resources, uri="mcp://resources/1", name="stdout"
        )
        mcp_router._append_tool_summary_resource(
            resources, uri="mcp://resources/1", name="stdout"
        )

        self.assertEqual(
            resources,
            [
                "malformed",
                {"uri": "mcp://resources/1", "name": "stdout"},
            ],
        )

    def test_tool_call_event_item_variants(self) -> None:
        call = ToolCall(id="c1", name="run", arguments={})
        error_event = Event(
            type=EventType.TOOL_RESULT,
            payload={
                "result": ToolCallError(
                    id="e1",
                    call=call,
                    name="run",
                    arguments={},
                    error=Exception("boom"),
                    message="boom",
                )
            },
        )
        error_item = mcp_router._tool_call_event_item(error_event)
        self.assertEqual(
            error_item["error"],
            {"type": "Exception", "message": "Tool call failed."},
        )
        self.assertNotIn("boom", str(error_item))

        result_event = Event(
            type=EventType.TOOL_RESULT,
            payload={
                "result": ToolCallResult(
                    id="r1",
                    call=call,
                    name="run",
                    arguments={},
                    result={"answer": 1},
                )
            },
        )
        result_item = mcp_router._tool_call_event_item(result_event)
        self.assertIn("result", result_item)

        list_event = Event(
            type=EventType.TOOL_PROCESS,
            payload=[call],
        )
        list_item = mcp_router._tool_call_event_item(list_event)
        self.assertEqual(list_item["id"], "c1")

        dict_event = Event(
            type=EventType.TOOL_PROCESS,
            payload={"call": call},
        )
        dict_item = mcp_router._tool_call_event_item(dict_event)
        self.assertEqual(dict_item["name"], "run")

        none_event = Event(type=EventType.TOOL_PROCESS, payload=None)
        self.assertIsNone(mcp_router._tool_call_event_item(none_event))

        null_call_event = Event(
            type=EventType.TOOL_PROCESS, payload={"call": None}
        )
        self.assertIsNone(mcp_router._tool_call_event_item(null_call_event))

        diagnostic = ToolCallDiagnostic(
            id="diag-1",
            call_id="c1",
            requested_name="run",
            code=ToolCallDiagnosticCode.UNKNOWN_TOOL,
            stage=ToolCallDiagnosticStage.RESOLVE,
            message="Unknown tool.",
        )
        diagnostic_event = Event(
            type=EventType.TOOL_DIAGNOSTIC,
            payload={"call": call, "diagnostic": diagnostic},
        )
        diagnostic_item = mcp_router._tool_call_event_item(diagnostic_event)
        self.assertEqual(diagnostic_item["id"], "c1")
        self.assertEqual(diagnostic_item["diagnostic"]["code"], "tool.unknown")

        result_diagnostic_event = Event(
            type=EventType.TOOL_RESULT,
            payload={"call": call, "result": diagnostic},
        )
        result_diagnostic_item = mcp_router._tool_call_event_item(
            result_diagnostic_event
        )
        self.assertEqual(result_diagnostic_item["id"], "c1")
        self.assertEqual(
            result_diagnostic_item["diagnostic"]["code"], "tool.unknown"
        )

        malformed_diagnostic_event = Event(
            type=EventType.TOOL_DIAGNOSTIC,
            payload={"diagnostics": ["bad"]},
        )
        self.assertIsNone(
            mcp_router._tool_call_event_item(malformed_diagnostic_event)
        )

        invalid_diagnostic_event = Event(
            type=EventType.TOOL_DIAGNOSTIC,
            payload={"diagnostic": "bad"},
        )
        self.assertIsNone(
            mcp_router._tool_call_event_item(invalid_diagnostic_event)
        )

    def test_get_resource_store_reuses_instance(self) -> None:
        request = self._request()
        first = mcp_router._get_resource_store(request)
        second = mcp_router._get_resource_store(request)
        self.assertIs(first, second)

    def test_server_info_defaults_and_state_version(self) -> None:
        request = self._request()
        info = mcp_router._server_info(request)
        self.assertEqual(info["name"], "avalan")
        self.assertEqual(info["version"], "0.0.0")

        request.app.state.version = "1.2.3"
        info_state = mcp_router._server_info(request)
        self.assertEqual(info_state["version"], "1.2.3")

    def test_server_info_app_version(self) -> None:
        request = self._request()
        request.app.title = "Test"
        request.app.version = "2.0"
        info = mcp_router._server_info(request)
        self.assertEqual(info, {"name": "Test", "version": "2.0"})

    def test_server_capabilities_with_tools(self) -> None:
        tool_manager = SimpleNamespace(is_empty=False)
        orchestrator = SimpleNamespace(tool=tool_manager)
        caps = mcp_router._server_capabilities(orchestrator)
        self.assertTrue(caps["tools"]["call"])

    def test_server_capabilities_without_tools(self) -> None:
        orchestrator = SimpleNamespace(tool=None)
        caps = mcp_router._server_capabilities(orchestrator)
        self.assertTrue(caps["tools"]["call"])
        self.assertTrue(caps["resources"]["subscribe"])

    def test_handle_list_tools_invalid_params(self) -> None:
        request = self._request()
        message = {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "tools/list",
            "params": "bad",
        }
        with self.assertRaises(mcp_router.HTTPException):
            mcp_router._handle_list_tools_message(
                request, MagicMock(), SimpleNamespace(tool=None), message
            )

    def test_handle_list_tools_message_includes_cursor(self) -> None:
        request = self._request()
        request.app.state.mcp_next_cursor = "cursor-token"
        message = {
            "jsonrpc": "2.0",
            "id": "cursor",
            "method": "tools/list",
            "params": {},
        }

        response = mcp_router._handle_list_tools_message(
            request,
            MagicMock(),
            SimpleNamespace(tool=None),
            message,
        )
        payload = loads(response.body.decode("utf-8"))
        self.assertEqual(payload["result"]["nextCursor"], "cursor-token")

    def test_handle_initialize_message_defaults(self) -> None:
        request = self._request()
        response = mcp_router._handle_initialize_message(
            request,
            MagicMock(),
            SimpleNamespace(tool=None),
            {"jsonrpc": "2.0"},
        )
        payload = loads(response.body.decode("utf-8"))
        self.assertEqual(payload["result"]["protocolVersion"], "1.0.0")

    def test_handle_ping_message_returns_empty_result(self) -> None:
        message = {
            "jsonrpc": "2.0",
            "id": "ping-1",
            "method": "ping",
            "params": {"message": "hello"},
        }

        response = mcp_router._handle_ping_message(MagicMock(), message)
        payload = loads(response.body.decode("utf-8"))
        self.assertEqual(payload["id"], "ping-1")
        self.assertEqual(payload["result"], {})

    def test_handle_ping_message_generates_id_when_missing(self) -> None:
        message = {"jsonrpc": "2.0", "method": "ping", "params": None}
        generated_id = UUID("00000000-0000-0000-0000-000000000042")

        with patch.object(mcp_router, "uuid4", return_value=generated_id):
            response = mcp_router._handle_ping_message(MagicMock(), message)

        payload = loads(response.body.decode("utf-8"))
        self.assertEqual(payload["id"], str(generated_id))
        self.assertEqual(payload["result"], {})

    def test_handle_ping_message_invalid_params(self) -> None:
        message = {
            "jsonrpc": "2.0",
            "method": "ping",
            "params": "invalid",
        }

        with self.assertRaises(mcp_router.HTTPException):
            mcp_router._handle_ping_message(MagicMock(), message)


class MCPJSONRPCMessageTestCase(IsolatedAsyncioTestCase):
    async def test_expect_jsonrpc_message_rejects_non_dict_from_state_iterator(
        self,
    ) -> None:
        async def iterator() -> AsyncIterator[object]:
            yield "invalid"  # type: ignore[misc]

        request = DummyRequest(b"")
        request.state._mcp_message_iter = iterator()

        with self.assertRaises(mcp_router.HTTPException) as exc:
            await mcp_router._expect_jsonrpc_message(request, {"initialize"})

        self.assertIn("Invalid MCP payload", str(exc.exception.detail))

    async def test_iter_jsonrpc_messages_rejects_non_dict_segment(
        self,
    ) -> None:
        request = DummyRequest([b"[1]\x1e"])

        with self.assertRaises(mcp_router.HTTPException) as exc:
            await anext(mcp_router._iter_jsonrpc_messages(request))

        self.assertIn("Invalid MCP payload", str(exc.exception.detail))


class MCPRouterAsyncTestCase(IsolatedAsyncioTestCase):
    def _get_route(self, path: str):
        router = mcp_router.create_router()
        for route in router.routes:
            if getattr(route, "path", None) == path:
                return route.endpoint
        raise AssertionError(f"Route {path} not found")

    async def test_stream_item_notifications_emit_canonical_answer_delta(
        self,
    ) -> None:
        state = mcp_router._MCPStreamProjectionState(
            accumulator=mcp_router.ProtocolStreamAccumulator(),
            legacy_answer_chunks=[],
            legacy_reasoning_chunks=[],
            tool_summaries={},
            resources={},
            resource_store=mcp_router.MCPResourceStore(),
            base_path="/mcp",
        )
        start = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=0,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
        )
        empty = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=1,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta="",
        )
        item = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=2,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta="answer",
        )
        done = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=3,
            kind=StreamItemKind.ANSWER_DONE,
            channel=StreamChannel.ANSWER,
        )

        start_notifications = await mcp_router._mcp_stream_item_notifications(
            start, state, "progress"
        )
        empty_notifications = await mcp_router._mcp_stream_item_notifications(
            empty, state, "progress"
        )
        notifications = await mcp_router._mcp_stream_item_notifications(
            item, state, "progress"
        )
        done_notifications = await mcp_router._mcp_stream_item_notifications(
            done, state, "progress"
        )

        self.assertEqual(start_notifications, [])
        self.assertEqual(empty_notifications, [])
        self.assertEqual(done_notifications, [])
        self.assertEqual(
            notifications,
            [
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/progress",
                    "params": {
                        "progressToken": "progress",
                        "progress": {
                            "type": "answer.delta",
                            "delta": "answer",
                        },
                    },
                }
            ],
        )
        self.assertEqual(state.accumulator.snapshot().answer_text, "answer")

    async def test_consume_call_request(self) -> None:
        message = {
            "jsonrpc": "2.0",
            "id": "call-1",
            "method": "tools/call",
            "params": {
                "name": "run",
                "arguments": {
                    "input_string": "Hello",
                },
            },
        }
        body = (dumps(message) + mcp_router.RS).encode("utf-8")
        request = DummyRequest(body)
        request.app.state.mcp_resource_base_path = "/m"

        request_id, req_model, token = await mcp_router._consume_call_request(
            request
        )
        self.assertEqual(request_id, "call-1")
        self.assertEqual(req_model.input_string, "Hello")
        self.assertTrue(token)
        remaining = []
        async for message in mcp_router._iter_jsonrpc_messages(request):
            remaining.append(message)
        self.assertFalse(remaining)

    async def test_stream_response_emits_notifications(self) -> None:
        tool_call = ToolCall(id="t1", name="tool", arguments={"a": 1})
        tool_result = ToolCallResult(
            id="res1",
            call=tool_call,
            name="tool",
            arguments={"a": 1},
            result={"stdout": "log"},
        )
        items: list[Any] = [
            ReasoningToken(token="thinking"),
            Event(
                type=EventType.TOOL_PROCESS, payload=[tool_call], started=1.0
            ),
            Token(token="Hello"),
            Event(
                type=EventType.TOOL_RESULT,
                payload={"result": tool_result},
                started=1.0,
                finished=2.0,
                elapsed=1.0,
            ),
            Token(token="!"),
            "text",
        ]
        response = DummyResponse(items)
        request_model = ChatCompletionRequest(
            model="gpt",
            messages=[ChatMessage(role="user", content="hi")],
            stream=True,
        )
        orchestrator = MagicMock()
        orchestrator.sync_messages = AsyncMock()
        cancel_event = AsyncEvent()
        store = mcp_router.MCPResourceStore()
        base_path = "/m"

        chunks = []
        async for chunk in mcp_router._stream_mcp_response(
            request_id="1",
            request_model=request_model,
            response=response,
            response_id=uuid4(),
            timestamp=123,
            progress_token="progress",
            orchestrator=orchestrator,
            logger=MagicMock(),
            resource_store=store,
            base_path=base_path,
            cancel_event=cancel_event,
        ):
            chunks.append(chunk.decode("utf-8"))

        orchestrator.sync_messages.assert_awaited()

        messages = [
            loads(part) for part in "".join(chunks).splitlines() if part
        ]
        dict_messages = [msg for msg in messages if isinstance(msg, dict)]
        methods = [
            msg.get("method") for msg in dict_messages if "method" in msg
        ]
        self.assertIn("notifications/message", methods)
        self.assertIn("notifications/progress", methods)
        self.assertIn("notifications/resources/updated", methods)
        result_messages = [msg for msg in dict_messages if msg.get("result")]
        self.assertTrue(result_messages)
        summary = result_messages[-1]["result"]["structuredContent"]
        self.assertEqual(summary["model"], "gpt")

    async def test_stream_response_emits_canonical_notifications(
        self,
    ) -> None:
        items: list[Any] = [
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
                text_delta="plan",
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=2,
                kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                channel=StreamChannel.TOOL_CALL,
                correlation=StreamItemCorrelation(tool_call_id="call-1"),
                text_delta='{"x"',
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=3,
                kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                channel=StreamChannel.TOOL_CALL,
                correlation=StreamItemCorrelation(tool_call_id="call-1"),
                text_delta="",
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=4,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="answer",
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
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                usage={},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        ]
        response = DummyResponse(items)
        request_model = ChatCompletionRequest(
            model="gpt",
            messages=[ChatMessage(role="user", content="hi")],
            stream=True,
        )
        orchestrator = MagicMock()
        orchestrator.sync_messages = AsyncMock()
        cancel_event = AsyncEvent()

        chunks = []
        async for chunk in mcp_router._stream_mcp_response(
            request_id="1",
            request_model=request_model,
            response=response,
            response_id=uuid4(),
            timestamp=123,
            progress_token="progress",
            orchestrator=orchestrator,
            logger=MagicMock(),
            resource_store=mcp_router.MCPResourceStore(),
            base_path="/m",
            cancel_event=cancel_event,
        ):
            chunks.append(chunk.decode("utf-8"))

        messages = [
            loads(part) for part in "".join(chunks).splitlines() if part
        ]
        notifications = [
            msg for msg in messages if msg.get("method") is not None
        ]
        notification_text = dumps(notifications)
        self.assertIn('"delta": "plan"', notification_text)
        self.assertIn('"type": "tool.input_delta"', notification_text)
        self.assertIn('"toolCallId": "call-1"', notification_text)
        stream_text = "".join(chunks)
        self.assertIn('"type":"answer.delta"', stream_text)
        self.assertEqual(stream_text.count('"type":"answer.completed"'), 1)
        result_messages = [msg for msg in messages if msg.get("result")]
        summary = result_messages[-1]["result"]["structuredContent"]
        self.assertEqual(summary["reasoning"], "plan")
        self.assertEqual(
            summary["toolCalls"],
            [{"id": "call-1", "name": None, "arguments": '{"x"'}],
        )
        self.assertEqual(
            result_messages[-1]["result"]["content"],
            [{"type": "text", "text": "answer"}],
        )
        orchestrator.sync_messages.assert_awaited()

    async def test_stream_response_maps_canonical_error_terminal_to_error(
        self,
    ) -> None:
        items: list[Any] = [
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
                text_delta="partial",
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=2,
                kind=StreamItemKind.STREAM_ERRORED,
                channel=StreamChannel.CONTROL,
                data={"message": "provider failed"},
                terminal_outcome=StreamTerminalOutcome.ERRORED,
            ),
        ]
        response = DummyResponse(items)
        request_model = ChatCompletionRequest(
            model="gpt",
            messages=[ChatMessage(role="user", content="hi")],
            stream=True,
        )
        orchestrator = MagicMock()
        orchestrator.sync_messages = AsyncMock()

        chunks = []
        async for chunk in mcp_router._stream_mcp_response(
            request_id="1",
            request_model=request_model,
            response=response,
            response_id=uuid4(),
            timestamp=123,
            progress_token="progress",
            orchestrator=orchestrator,
            logger=MagicMock(),
            resource_store=mcp_router.MCPResourceStore(),
            base_path="/m",
            cancel_event=AsyncEvent(),
        ):
            chunks.append(chunk.decode("utf-8"))

        messages = [
            loads(part) for part in "".join(chunks).splitlines() if part
        ]
        self.assertTrue(
            any(
                item.get("method") == "notifications/progress"
                and item["params"]["progress"]["type"] == "stream.errored"
                for item in messages
            )
        )
        errors = [item for item in messages if "error" in item]
        self.assertEqual(errors[-1]["error"]["code"], -32603)
        self.assertEqual(errors[-1]["error"]["message"], "provider failed")
        self.assertFalse(any("result" in item for item in messages))
        self.assertNotIn('"type":"answer.completed"', "".join(chunks))
        orchestrator.sync_messages.assert_awaited()

    async def test_stream_response_closes_resources_on_canonical_error(
        self,
    ) -> None:
        items: list[Any] = [
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
                kind=StreamItemKind.TOOL_EXECUTION_STARTED,
                channel=StreamChannel.TOOL_EXECUTION,
                correlation=StreamItemCorrelation(tool_call_id="call-1"),
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=2,
                kind=StreamItemKind.TOOL_EXECUTION_OUTPUT,
                channel=StreamChannel.TOOL_EXECUTION,
                correlation=StreamItemCorrelation(tool_call_id="call-1"),
                text_delta="partial",
                data={"category": "stdout", "content": "partial"},
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=3,
                kind=StreamItemKind.STREAM_ERRORED,
                channel=StreamChannel.CONTROL,
                data={"message": "provider failed"},
                terminal_outcome=StreamTerminalOutcome.ERRORED,
            ),
        ]
        response = DummyResponse(items)
        request_model = ChatCompletionRequest(
            model="gpt",
            messages=[ChatMessage(role="user", content="hi")],
            stream=True,
        )
        orchestrator = MagicMock()
        orchestrator.sync_messages = AsyncMock()
        store = mcp_router.MCPResourceStore()

        messages = []
        async for chunk in mcp_router._stream_mcp_response(
            request_id="1",
            request_model=request_model,
            response=response,
            response_id=uuid4(),
            timestamp=123,
            progress_token="progress",
            orchestrator=orchestrator,
            logger=MagicMock(),
            resource_store=store,
            base_path="/m",
            cancel_event=AsyncEvent(),
        ):
            messages.extend(
                loads(part)
                for part in chunk.decode("utf-8").splitlines()
                if part
            )

        resource_updates = [
            item
            for item in messages
            if item.get("method") == "notifications/resources/updated"
        ]
        self.assertEqual(len(resource_updates), 2)
        resource_id = resource_updates[0]["params"]["resources"][0][
            "uri"
        ].rsplit("/", 1)[-1]
        stored_resource = await store.get(resource_id)
        errors = [item for item in messages if "error" in item]

        self.assertTrue(response._closed)
        self.assertTrue(stored_resource.closed)
        self.assertTrue(
            resource_updates[-1]["params"]["resources"][0]["closed"]
        )
        self.assertEqual(errors[-1]["error"]["message"], "provider failed")
        self.assertFalse(any("result" in item for item in messages))
        orchestrator.sync_messages.assert_awaited()

    async def test_stream_response_maps_canonical_cancel_terminal_to_error(
        self,
    ) -> None:
        items: list[Any] = [
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
                kind=StreamItemKind.STREAM_CANCELLED,
                channel=StreamChannel.CONTROL,
                terminal_outcome=StreamTerminalOutcome.CANCELLED,
            ),
        ]
        response = DummyResponse(items)
        request_model = ChatCompletionRequest(
            model="gpt",
            messages=[ChatMessage(role="user", content="hi")],
            stream=True,
        )
        orchestrator = MagicMock()
        orchestrator.sync_messages = AsyncMock()

        chunks = []
        async for chunk in mcp_router._stream_mcp_response(
            request_id="1",
            request_model=request_model,
            response=response,
            response_id=uuid4(),
            timestamp=123,
            progress_token="progress",
            orchestrator=orchestrator,
            logger=MagicMock(),
            resource_store=mcp_router.MCPResourceStore(),
            base_path="/m",
            cancel_event=AsyncEvent(),
        ):
            chunks.append(chunk.decode("utf-8"))

        messages = [
            loads(part) for part in "".join(chunks).splitlines() if part
        ]
        self.assertTrue(
            any(
                item.get("method") == "notifications/progress"
                and item["params"]["progress"]["type"] == "stream.cancelled"
                for item in messages
            )
        )
        errors = [item for item in messages if "error" in item]
        self.assertEqual(errors[-1]["error"]["code"], -32000)
        self.assertFalse(any("result" in item for item in messages))
        orchestrator.sync_messages.assert_awaited()

    async def test_stream_response_closes_resources_on_canonical_cancel(
        self,
    ) -> None:
        items: list[Any] = [
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
                kind=StreamItemKind.TOOL_EXECUTION_STARTED,
                channel=StreamChannel.TOOL_EXECUTION,
                correlation=StreamItemCorrelation(tool_call_id="call-1"),
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=2,
                kind=StreamItemKind.TOOL_EXECUTION_OUTPUT,
                channel=StreamChannel.TOOL_EXECUTION,
                correlation=StreamItemCorrelation(tool_call_id="call-1"),
                text_delta="partial",
                data={"category": "stdout", "content": "partial"},
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=3,
                kind=StreamItemKind.TOOL_EXECUTION_CANCELLED,
                channel=StreamChannel.TOOL_EXECUTION,
                correlation=StreamItemCorrelation(tool_call_id="call-1"),
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=4,
                kind=StreamItemKind.STREAM_CANCELLED,
                channel=StreamChannel.CONTROL,
                terminal_outcome=StreamTerminalOutcome.CANCELLED,
            ),
        ]
        response = DummyResponse(items)
        request_model = ChatCompletionRequest(
            model="gpt",
            messages=[ChatMessage(role="user", content="hi")],
            stream=True,
        )
        orchestrator = MagicMock()
        orchestrator.sync_messages = AsyncMock()

        messages = []
        async for chunk in mcp_router._stream_mcp_response(
            request_id="1",
            request_model=request_model,
            response=response,
            response_id=uuid4(),
            timestamp=123,
            progress_token="progress",
            orchestrator=orchestrator,
            logger=MagicMock(),
            resource_store=mcp_router.MCPResourceStore(),
            base_path="/m",
            cancel_event=AsyncEvent(),
        ):
            messages.extend(
                loads(part)
                for part in chunk.decode("utf-8").splitlines()
                if part
            )

        resource_updates = [
            item
            for item in messages
            if item.get("method") == "notifications/resources/updated"
        ]
        self.assertEqual(len(resource_updates), 2)
        self.assertEqual(
            resource_updates[0]["params"]["resources"][0]["delta"]["set"][
                "text"
            ],
            "partial",
        )
        self.assertTrue(
            resource_updates[-1]["params"]["resources"][0]["closed"]
        )

    async def test_stream_response_rejects_missing_canonical_terminal(
        self,
    ) -> None:
        response = DummyResponse(
            [
                CanonicalStreamItem(
                    stream_session_id="s",
                    run_id="r",
                    turn_id="t",
                    sequence=0,
                    kind=StreamItemKind.STREAM_STARTED,
                    channel=StreamChannel.CONTROL,
                )
            ]
        )
        request_model = ChatCompletionRequest(
            model="gpt",
            messages=[ChatMessage(role="user", content="hi")],
            stream=True,
        )
        orchestrator = MagicMock()
        orchestrator.sync_messages = AsyncMock()
        logger = MagicMock()

        messages = []
        async for chunk in mcp_router._stream_mcp_response(
            request_id="1",
            request_model=request_model,
            response=response,
            response_id=uuid4(),
            timestamp=123,
            progress_token="progress",
            orchestrator=orchestrator,
            logger=logger,
            resource_store=mcp_router.MCPResourceStore(),
            base_path="/m",
            cancel_event=AsyncEvent(),
        ):
            messages.extend(
                loads(part)
                for part in chunk.decode("utf-8").splitlines()
                if part
            )

        errors = [item for item in messages if "error" in item]
        self.assertEqual(errors[-1]["error"]["code"], -32603)
        self.assertFalse(any("result" in item for item in messages))
        logger.exception.assert_called_once()
        orchestrator.sync_messages.assert_awaited()

    async def test_stream_response_closes_resources_on_validation_error(
        self,
    ) -> None:
        response = DummyResponse(
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
                    kind=StreamItemKind.TOOL_EXECUTION_STARTED,
                    channel=StreamChannel.TOOL_EXECUTION,
                    correlation=StreamItemCorrelation(tool_call_id="call-1"),
                ),
                CanonicalStreamItem(
                    stream_session_id="s",
                    run_id="r",
                    turn_id="t",
                    sequence=2,
                    kind=StreamItemKind.TOOL_EXECUTION_OUTPUT,
                    channel=StreamChannel.TOOL_EXECUTION,
                    correlation=StreamItemCorrelation(tool_call_id="call-1"),
                    text_delta="partial",
                    data={"category": "stdout", "content": "partial"},
                ),
            ]
        )
        request_model = ChatCompletionRequest(
            model="gpt",
            messages=[ChatMessage(role="user", content="hi")],
            stream=True,
        )
        orchestrator = MagicMock()
        orchestrator.sync_messages = AsyncMock()
        store = mcp_router.MCPResourceStore()

        messages = []
        async for chunk in mcp_router._stream_mcp_response(
            request_id="1",
            request_model=request_model,
            response=response,
            response_id=uuid4(),
            timestamp=123,
            progress_token="progress",
            orchestrator=orchestrator,
            logger=MagicMock(),
            resource_store=store,
            base_path="/m",
            cancel_event=AsyncEvent(),
        ):
            messages.extend(
                loads(part)
                for part in chunk.decode("utf-8").splitlines()
                if part
            )

        resource_updates = [
            item
            for item in messages
            if item.get("method") == "notifications/resources/updated"
        ]
        self.assertEqual(len(resource_updates), 2)
        resource_id = resource_updates[0]["params"]["resources"][0][
            "uri"
        ].rsplit("/", 1)[-1]
        stored_resource = await store.get(resource_id)

        self.assertTrue(response._closed)
        self.assertTrue(stored_resource.closed)
        self.assertTrue(
            resource_updates[-1]["params"]["resources"][0]["closed"]
        )
        self.assertEqual(messages[-1]["error"]["code"], -32603)
        self.assertFalse(any("result" in item for item in messages))
        orchestrator.sync_messages.assert_awaited()

    async def test_stream_response_final_result_uses_canonical_accumulator(
        self,
    ) -> None:
        items: list[Any] = [
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
                text_delta="canonical plan",
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
                text_delta="canonical answer",
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=4,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=5,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                usage={
                    "input_text_tokens": 1,
                    "output_text_tokens": 2,
                    "total_tokens": 3,
                },
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        ]
        response = DummyResponse(items)
        request_model = ChatCompletionRequest(
            model="gpt",
            messages=[ChatMessage(role="user", content="hi")],
            stream=True,
        )
        orchestrator = MagicMock()
        orchestrator.sync_messages = AsyncMock()

        chunks = []
        async for chunk in mcp_router._stream_mcp_response(
            request_id="1",
            request_model=request_model,
            response=response,
            response_id=uuid4(),
            timestamp=123,
            progress_token="progress",
            orchestrator=orchestrator,
            logger=MagicMock(),
            resource_store=mcp_router.MCPResourceStore(),
            base_path="/m",
            cancel_event=AsyncEvent(),
        ):
            chunks.append(chunk.decode("utf-8"))

        messages = [
            loads(part) for part in "".join(chunks).splitlines() if part
        ]
        result = [msg for msg in messages if msg.get("result")][-1]["result"]

        self.assertEqual(
            result["content"],
            [{"type": "text", "text": "canonical answer"}],
        )
        self.assertEqual(
            result["structuredContent"]["reasoning"], "canonical plan"
        )
        self.assertEqual(
            result["structuredContent"]["usage"],
            {
                "input_text_tokens": 1,
                "output_text_tokens": 2,
                "total_tokens": 3,
            },
        )

    async def test_stream_response_preserves_canonical_tool_ready_payload(
        self,
    ) -> None:
        items: list[Any] = [
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
                kind=StreamItemKind.TOOL_CALL_READY,
                channel=StreamChannel.TOOL_CALL,
                correlation=StreamItemCorrelation(tool_call_id="call-ready"),
                data={
                    "name": "lookup",
                    "arguments": {"query": "docs"},
                },
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=2,
                kind=StreamItemKind.TOOL_CALL_DONE,
                channel=StreamChannel.TOOL_CALL,
                correlation=StreamItemCorrelation(tool_call_id="call-ready"),
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=3,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                usage={
                    "input_text_tokens": 1,
                    "output_text_tokens": 0,
                    "total_tokens": 1,
                },
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        ]
        response = DummyResponse(items)
        request_model = ChatCompletionRequest(
            model="gpt",
            messages=[ChatMessage(role="user", content="hi")],
            stream=True,
        )
        orchestrator = MagicMock()
        orchestrator.sync_messages = AsyncMock()

        chunks = []
        async for chunk in mcp_router._stream_mcp_response(
            request_id="1",
            request_model=request_model,
            response=response,
            response_id=uuid4(),
            timestamp=123,
            progress_token="progress",
            orchestrator=orchestrator,
            logger=MagicMock(),
            resource_store=mcp_router.MCPResourceStore(),
            base_path="/m",
            cancel_event=AsyncEvent(),
        ):
            chunks.append(chunk.decode("utf-8"))

        messages = [
            loads(part) for part in "".join(chunks).splitlines() if part
        ]
        result = [msg for msg in messages if msg.get("result")][-1]["result"]

        self.assertEqual(
            result["structuredContent"]["toolCalls"],
            [
                {
                    "id": "call-ready",
                    "name": "lookup",
                    "arguments": {"query": "docs"},
                }
            ],
        )

    async def test_canonical_tool_ready_ignores_malformed_payload(
        self,
    ) -> None:
        state = mcp_router._MCPStreamProjectionState(
            accumulator=mcp_router.ProtocolStreamAccumulator(),
            legacy_answer_chunks=[],
            legacy_reasoning_chunks=[],
            tool_summaries={},
            resources={},
            resource_store=mcp_router.MCPResourceStore(),
            base_path="/mcp",
        )

        await mcp_router._mcp_stream_item_notifications(
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            ),
            state,
            "progress",
        )
        notifications = await mcp_router._mcp_stream_item_notifications(
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=1,
                kind=StreamItemKind.TOOL_CALL_READY,
                channel=StreamChannel.TOOL_CALL,
                correlation=StreamItemCorrelation(tool_call_id="call-ready"),
                data="bad",
            ),
            state,
            "progress",
        )

        self.assertEqual(notifications, [])
        self.assertEqual(
            state.tool_summaries,
            {
                "call-ready": {
                    "id": "call-ready",
                    "name": None,
                    "arguments": None,
                }
            },
        )

    async def test_stream_response_updates_resource_from_canonical_tool_output(
        self,
    ) -> None:
        items: list[Any] = [
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
                correlation=StreamItemCorrelation(tool_call_id="call-1"),
                text_delta="{}",
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=2,
                kind=StreamItemKind.TOOL_CALL_READY,
                channel=StreamChannel.TOOL_CALL,
                correlation=StreamItemCorrelation(tool_call_id="call-1"),
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=3,
                kind=StreamItemKind.TOOL_CALL_DONE,
                channel=StreamChannel.TOOL_CALL,
                correlation=StreamItemCorrelation(tool_call_id="call-1"),
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=4,
                kind=StreamItemKind.TOOL_EXECUTION_STARTED,
                channel=StreamChannel.TOOL_EXECUTION,
                correlation=StreamItemCorrelation(tool_call_id="call-1"),
                metadata={"tool_name": "shell"},
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=5,
                kind=StreamItemKind.TOOL_EXECUTION_OUTPUT,
                channel=StreamChannel.TOOL_EXECUTION,
                correlation=StreamItemCorrelation(tool_call_id="call-1"),
                text_delta="live\n",
                data={"category": "stdout", "content": "live\n"},
                metadata={"tool_name": "shell"},
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=6,
                kind=StreamItemKind.TOOL_EXECUTION_COMPLETED,
                channel=StreamChannel.TOOL_EXECUTION,
                correlation=StreamItemCorrelation(tool_call_id="call-1"),
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=7,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="done",
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=8,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=9,
                kind=StreamItemKind.USAGE_COMPLETED,
                channel=StreamChannel.USAGE,
                usage={
                    "input_text_tokens": 11,
                    "output_text_tokens": 22,
                    "total_tokens": 33,
                },
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=10,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        ]
        response = DummyResponse(items)
        request_model = ChatCompletionRequest(
            model="gpt",
            messages=[ChatMessage(role="user", content="hi")],
            stream=True,
        )
        orchestrator = MagicMock()
        orchestrator.sync_messages = AsyncMock()
        store = mcp_router.MCPResourceStore()

        chunks = []
        async for chunk in mcp_router._stream_mcp_response(
            request_id="1",
            request_model=request_model,
            response=response,
            response_id=uuid4(),
            timestamp=123,
            progress_token="progress",
            orchestrator=orchestrator,
            logger=MagicMock(),
            resource_store=store,
            base_path="/m",
            cancel_event=AsyncEvent(),
        ):
            chunks.append(chunk.decode("utf-8"))

        messages = [
            loads(part) for part in "".join(chunks).splitlines() if part
        ]
        resource_messages = [
            item
            for item in messages
            if item.get("method") == "notifications/resources/updated"
        ]
        self.assertEqual(len(resource_messages), 1)
        resource = resource_messages[0]["params"]["resources"][0]
        self.assertEqual(resource["delta"]["set"]["text"], "live\n")
        result_messages = [msg for msg in messages if msg.get("result")]
        result = result_messages[-1]["result"]
        self.assertEqual(result["content"], [{"type": "text", "text": "done"}])
        self.assertEqual(
            result["structuredContent"]["usage"],
            {
                "input_text_tokens": 11,
                "output_text_tokens": 22,
                "total_tokens": 33,
            },
        )
        self.assertEqual(
            result["structuredContent"]["toolCalls"][0]["resources"][0][
                "name"
            ],
            "stdout",
        )
        self.assertEqual(
            result["structuredContent"]["toolCalls"][0]["arguments"], "{}"
        )

    async def test_canonical_tool_resource_notifications_edge_cases(
        self,
    ) -> None:
        async def collect(
            item: CanonicalStreamItem,
            *,
            summaries: dict[str, dict[str, Any]] | None = None,
            resources: dict[str, mcp_router.MCPResource] | None = None,
        ) -> list[dict[str, Any]]:
            collected: list[dict[str, Any]] = []
            async for (
                notification
            ) in mcp_router._canonical_tool_resource_notifications(
                item=item,
                tool_summaries=summaries if summaries is not None else {},
                resources=resources if resources is not None else {},
                resource_store=store,
                base_path="/m",
            ):
                collected.append(notification)
            return collected

        store = mcp_router.MCPResourceStore()
        item = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=0,
            kind=StreamItemKind.TOOL_EXECUTION_OUTPUT,
            channel=StreamChannel.TOOL_EXECUTION,
            correlation=StreamItemCorrelation(tool_call_id="call-1"),
            text_delta="one",
            data={"category": "stdout", "content": "one"},
        )
        malformed_correlation = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=1,
            kind=StreamItemKind.TOOL_EXECUTION_OUTPUT,
            channel=StreamChannel.TOOL_EXECUTION,
            correlation=StreamItemCorrelation(tool_call_id="call-1"),
            text_delta="ignored",
            data={"category": "stdout", "content": "ignored"},
        )
        object.__setattr__(
            malformed_correlation, "correlation", StreamItemCorrelation()
        )
        no_data = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=2,
            kind=StreamItemKind.TOOL_EXECUTION_OUTPUT,
            channel=StreamChannel.TOOL_EXECUTION,
            correlation=StreamItemCorrelation(tool_call_id="call-1"),
            text_delta="ignored",
        )
        unsupported_category = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=3,
            kind=StreamItemKind.TOOL_EXECUTION_OUTPUT,
            channel=StreamChannel.TOOL_EXECUTION,
            correlation=StreamItemCorrelation(tool_call_id="call-1"),
            text_delta="ignored",
            data={"category": "progress", "content": "ignored"},
        )
        empty_content = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=4,
            kind=StreamItemKind.TOOL_EXECUTION_OUTPUT,
            channel=StreamChannel.TOOL_EXECUTION,
            correlation=StreamItemCorrelation(tool_call_id="call-1"),
            text_delta="",
            data={"category": "stdout", "content": ""},
        )
        unsupported_progress_category = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=5,
            kind=StreamItemKind.TOOL_EXECUTION_PROGRESS,
            channel=StreamChannel.TOOL_EXECUTION,
            correlation=StreamItemCorrelation(tool_call_id="call-1"),
            data={"category": "stdout", "progress": 0.5},
        )
        empty_progress_content = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=6,
            kind=StreamItemKind.TOOL_EXECUTION_PROGRESS,
            channel=StreamChannel.TOOL_EXECUTION,
            correlation=StreamItemCorrelation(tool_call_id="call-1"),
            data={"category": "progress"},
        )
        answer = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=7,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta="ignored",
        )

        self.assertEqual(await collect(malformed_correlation), [])
        self.assertEqual(await collect(no_data), [])
        self.assertEqual(await collect(unsupported_category), [])
        self.assertEqual(await collect(empty_content), [])
        self.assertEqual(await collect(unsupported_progress_category), [])
        self.assertEqual(await collect(empty_progress_content), [])
        self.assertEqual(
            mcp_router._canonical_tool_resource_content(answer), ""
        )

        summaries: dict[str, dict[str, Any]] = {}
        resources: dict[str, mcp_router.MCPResource] = {}
        first = await collect(item, summaries=summaries, resources=resources)
        second_item = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=5,
            kind=StreamItemKind.TOOL_EXECUTION_OUTPUT,
            channel=StreamChannel.TOOL_EXECUTION,
            correlation=StreamItemCorrelation(tool_call_id="call-1"),
            text_delta="two",
            data={"category": "stdout", "content": "two"},
        )
        second = await collect(
            second_item, summaries=summaries, resources=resources
        )

        self.assertEqual(
            first[0]["params"]["resources"][0]["delta"]["set"]["text"],
            "one",
        )
        self.assertEqual(
            second[0]["params"]["resources"][0]["delta"]["set"]["text"],
            "onetwo",
        )
        self.assertEqual(summaries["call-1"]["resources"][0]["name"], "stdout")
        self.assertEqual(len(summaries["call-1"]["resources"]), 1)

    async def test_canonical_tool_progress_updates_resource(self) -> None:
        store = mcp_router.MCPResourceStore(resource_item_limit=2)
        summaries: dict[str, dict[str, Any]] = {}
        resources: dict[str, mcp_router.MCPResource] = {}
        item = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=0,
            kind=StreamItemKind.TOOL_EXECUTION_PROGRESS,
            channel=StreamChannel.TOOL_EXECUTION,
            correlation=StreamItemCorrelation(tool_call_id="call-1"),
            data={"category": "progress", "progress": 0.5},
        )
        content_item = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=1,
            kind=StreamItemKind.TOOL_EXECUTION_PROGRESS,
            channel=StreamChannel.TOOL_EXECUTION,
            correlation=StreamItemCorrelation(tool_call_id="call-1"),
            data={"category": "progress", "content": "half"},
        )

        notifications: list[dict[str, Any]] = []
        async for (
            notification
        ) in mcp_router._canonical_tool_resource_notifications(
            item=item,
            tool_summaries=summaries,
            resources=resources,
            resource_store=store,
            base_path="/m",
        ):
            notifications.append(notification)
        async for (
            notification
        ) in mcp_router._canonical_tool_resource_notifications(
            item=content_item,
            tool_summaries=summaries,
            resources=resources,
            resource_store=store,
            base_path="/m",
        ):
            notifications.append(notification)

        self.assertEqual(len(notifications), 2)
        self.assertEqual(
            notifications[0]["params"]["resources"][0]["delta"]["set"]["text"],
            '{"progress": 0.5}',
        )
        self.assertEqual(
            notifications[1]["params"]["resources"][0]["delta"]["set"]["text"],
            '{"progress": 0.5}half',
        )
        self.assertEqual(
            summaries["call-1"]["resources"][-1]["name"], "progress"
        )
        self.assertEqual(len(summaries["call-1"]["resources"]), 1)

    async def test_stream_response_rejects_duplicate_canonical_terminal(
        self,
    ) -> None:
        items: list[Any] = [
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
                kind=StreamItemKind.USAGE_COMPLETED,
                channel=StreamChannel.USAGE,
                usage={},
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=2,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=3,
                kind=StreamItemKind.STREAM_ERRORED,
                channel=StreamChannel.CONTROL,
                terminal_outcome=StreamTerminalOutcome.ERRORED,
            ),
        ]
        response = DummyResponse(items)
        request_model = ChatCompletionRequest(
            model="gpt",
            messages=[ChatMessage(role="user", content="hi")],
            stream=True,
        )
        orchestrator = MagicMock()
        orchestrator.sync_messages = AsyncMock()
        cancel_event = AsyncEvent()

        payloads: list[dict[str, Any]] = []
        async for chunk in mcp_router._stream_mcp_response(
            request_id="id",
            request_model=request_model,
            response=response,
            response_id=uuid4(),
            timestamp=1,
            progress_token="tok",
            orchestrator=orchestrator,
            logger=MagicMock(),
            resource_store=mcp_router.MCPResourceStore(),
            base_path="/base",
            cancel_event=cancel_event,
        ):
            payloads.extend(
                loads(part)
                for part in chunk.decode("utf-8").splitlines()
                if part
            )

        self.assertTrue(cancel_event.is_set())
        self.assertEqual(payloads[-1]["error"]["code"], -32603)
        self.assertFalse(any("result" in item for item in payloads))

    async def test_stream_response_rejects_legacy_after_terminal(
        self,
    ) -> None:
        items: list[Any] = [
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
                kind=StreamItemKind.USAGE_COMPLETED,
                channel=StreamChannel.USAGE,
                usage={},
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=2,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
            Token(token="late"),
        ]
        response = DummyResponse(items)
        request_model = ChatCompletionRequest(
            model="gpt",
            messages=[ChatMessage(role="user", content="hi")],
            stream=True,
        )
        orchestrator = MagicMock()
        orchestrator.sync_messages = AsyncMock()
        cancel_event = AsyncEvent()

        payloads: list[dict[str, Any]] = []
        async for chunk in mcp_router._stream_mcp_response(
            request_id="id",
            request_model=request_model,
            response=response,
            response_id=uuid4(),
            timestamp=1,
            progress_token="tok",
            orchestrator=orchestrator,
            logger=MagicMock(),
            resource_store=mcp_router.MCPResourceStore(),
            base_path="/base",
            cancel_event=cancel_event,
        ):
            payloads.extend(
                loads(part)
                for part in chunk.decode("utf-8").splitlines()
                if part
            )

        self.assertTrue(cancel_event.is_set())
        self.assertEqual(payloads[-1]["error"]["code"], -32603)
        self.assertNotIn("late", dumps(payloads))
        self.assertFalse(any("result" in item for item in payloads))

    async def test_stream_response_rejects_mixed_surfaces(
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
        cases = (
            (
                "legacy-first",
                [Token(token="legacy"), canonical_item],
            ),
            (
                "canonical-first",
                [canonical_item, Token(token="legacy")],
            ),
        )

        for label, items in cases:
            with self.subTest(label=label):
                response = DummyResponse(items)
                request_model = ChatCompletionRequest(
                    model="gpt",
                    messages=[ChatMessage(role="user", content="hi")],
                    stream=True,
                )
                orchestrator = MagicMock()
                orchestrator.sync_messages = AsyncMock()
                cancel_event = AsyncEvent()

                payloads: list[dict[str, Any]] = []
                async for chunk in mcp_router._stream_mcp_response(
                    request_id="id",
                    request_model=request_model,
                    response=response,
                    response_id=uuid4(),
                    timestamp=1,
                    progress_token="tok",
                    orchestrator=orchestrator,
                    logger=MagicMock(),
                    resource_store=mcp_router.MCPResourceStore(),
                    base_path="/base",
                    cancel_event=cancel_event,
                ):
                    payloads.extend(
                        loads(part)
                        for part in chunk.decode("utf-8").splitlines()
                        if part
                    )

                self.assertTrue(cancel_event.is_set())
                self.assertEqual(payloads[-1]["error"]["code"], -32603)
                self.assertFalse(any("result" in item for item in payloads))

    async def test_stream_response_closes_resources_after_projection_error(
        self,
    ) -> None:
        call = ToolCall(id="cleanup", name="shell", arguments={})
        tool_result = ToolCallResult(
            id="result-cleanup",
            call=call,
            name="shell",
            arguments={},
            result={"stdout": "live"},
        )
        response = DummyResponse(
            [
                Event(
                    type=EventType.TOOL_RESULT,
                    payload={"result": tool_result},
                ),
                CanonicalStreamItem(
                    stream_session_id="s",
                    run_id="r",
                    turn_id="t",
                    sequence=0,
                    kind=StreamItemKind.STREAM_STARTED,
                    channel=StreamChannel.CONTROL,
                ),
            ]
        )
        request_model = ChatCompletionRequest(
            model="gpt",
            messages=[ChatMessage(role="user", content="hi")],
            stream=True,
        )
        orchestrator = MagicMock()
        orchestrator.sync_messages = AsyncMock()
        cancel_event = AsyncEvent()
        store = mcp_router.MCPResourceStore()

        payloads: list[dict[str, Any]] = []
        async for chunk in mcp_router._stream_mcp_response(
            request_id="id",
            request_model=request_model,
            response=response,
            response_id=uuid4(),
            timestamp=1,
            progress_token="tok",
            orchestrator=orchestrator,
            logger=MagicMock(),
            resource_store=store,
            base_path="/base",
            cancel_event=cancel_event,
        ):
            payloads.extend(
                loads(part)
                for part in chunk.decode("utf-8").splitlines()
                if part
            )

        resource_updates = [
            item
            for item in payloads
            if item.get("method") == "notifications/resources/updated"
        ]
        first_resource = resource_updates[0]["params"]["resources"][0]
        resource_id = first_resource["uri"].rsplit("/", 1)[-1]
        stored_resource = await store.get(resource_id)

        self.assertTrue(cancel_event.is_set())
        self.assertTrue(response._closed)
        self.assertTrue(stored_resource.closed)
        self.assertTrue(
            resource_updates[-1]["params"]["resources"][0]["closed"]
        )
        self.assertEqual(payloads[-1]["error"]["code"], -32603)
        self.assertFalse(any("result" in item for item in payloads))
        orchestrator.sync_messages.assert_awaited()

    async def test_stream_response_handles_iterator_creation_error(
        self,
    ) -> None:
        class BrokenResponse(DummyResponse):
            def __aiter__(self) -> AsyncIterator[Any]:
                raise RuntimeError("iterator failed")

        response = BrokenResponse([])
        request_model = ChatCompletionRequest(
            model="gpt",
            messages=[ChatMessage(role="user", content="hi")],
            stream=True,
        )
        orchestrator = MagicMock()
        orchestrator.sync_messages = AsyncMock()
        cancel_event = AsyncEvent()

        payloads: list[dict[str, Any]] = []
        async for chunk in mcp_router._stream_mcp_response(
            request_id="id",
            request_model=request_model,
            response=response,
            response_id=uuid4(),
            timestamp=1,
            progress_token="tok",
            orchestrator=orchestrator,
            logger=MagicMock(),
            resource_store=mcp_router.MCPResourceStore(),
            base_path="/base",
            cancel_event=cancel_event,
        ):
            payloads.extend(
                loads(part)
                for part in chunk.decode("utf-8").splitlines()
                if part
            )

        self.assertTrue(cancel_event.is_set())
        self.assertTrue(response._closed)
        self.assertEqual(payloads[-1]["error"]["code"], -32603)
        orchestrator.sync_messages.assert_awaited()

    async def test_stream_response_handles_cancellation(self) -> None:
        response = DummyResponse([Token(token="Hi")])
        request_model = ChatCompletionRequest(
            model="gpt",
            messages=[ChatMessage(role="user", content="hi")],
            stream=True,
        )
        orchestrator = MagicMock()
        orchestrator.sync_messages = AsyncMock()
        cancel_event = AsyncEvent()
        cancel_event.set()
        store = mcp_router.MCPResourceStore()

        chunks = []
        async for chunk in mcp_router._stream_mcp_response(
            request_id="1",
            request_model=request_model,
            response=response,
            response_id=uuid4(),
            timestamp=0,
            progress_token="progress",
            orchestrator=orchestrator,
            logger=MagicMock(),
            resource_store=store,
            base_path="/m",
            cancel_event=cancel_event,
        ):
            chunks.append(chunk.decode("utf-8"))

        messages = [
            loads(part) for part in "".join(chunks).splitlines() if part
        ]
        dict_messages = [msg for msg in messages if isinstance(msg, dict)]
        errors = [msg["error"] for msg in dict_messages if "error" in msg]
        self.assertTrue(errors)
        self.assertEqual(errors[-1]["code"], -32000)

    async def test_stream_response_handles_source_cancellation(self) -> None:
        class SourceCancelledResponse(DummyResponse):
            def __init__(self, items: list[Any]) -> None:
                super().__init__(items)
                self.cancel_count = 0
                self.close_count = 0

            async def cancel(self) -> None:
                self.cancel_count += 1

            async def aclose(self) -> None:
                self.close_count += 1
                await super().aclose()

        def cancel_source() -> object:
            raise CancelledError()

        call = ToolCall(id="call-1", name="run", arguments={})
        result = ToolCallResult(
            id="res-1",
            call=call,
            name="run",
            arguments={},
            result={"stdout": "partial"},
        )
        response = SourceCancelledResponse(
            [
                Event(
                    type=EventType.TOOL_RESULT,
                    payload={"result": result},
                    started=1.0,
                    finished=2.0,
                    elapsed=1.0,
                ),
                cancel_source,
            ]
        )
        orchestrator = MagicMock()
        orchestrator.sync_messages = AsyncMock()
        cancel_event = AsyncEvent()
        store = mcp_router.MCPResourceStore()

        payloads: list[dict[str, Any]] = []
        async for chunk in mcp_router._stream_mcp_response(
            request_id="id",
            request_model=ChatCompletionRequest(
                model="gpt",
                messages=[ChatMessage(role="user", content="hi")],
                stream=True,
            ),
            response=response,
            response_id=uuid4(),
            timestamp=1,
            progress_token="tok",
            orchestrator=orchestrator,
            logger=MagicMock(),
            resource_store=store,
            base_path="/base",
            cancel_event=cancel_event,
        ):
            payloads.extend(
                loads(part)
                for part in chunk.decode("utf-8").splitlines()
                if part
            )

        resource_updates = [
            item
            for item in payloads
            if item.get("method") == "notifications/resources/updated"
        ]
        resource_id = resource_updates[0]["params"]["resources"][0][
            "uri"
        ].rsplit("/", 1)[-1]

        self.assertTrue(cancel_event.is_set())
        self.assertEqual(response.cancel_count, 1)
        self.assertEqual(response.close_count, 1)
        self.assertTrue(response._closed)
        self.assertTrue((await store.get(resource_id)).closed)
        self.assertTrue(
            resource_updates[-1]["params"]["resources"][0]["closed"]
        )
        self.assertEqual(payloads[-1]["error"]["code"], -32000)
        self.assertEqual(payloads[-1]["error"]["message"], "Request cancelled")
        orchestrator.sync_messages.assert_awaited()

    async def test_stream_response_interrupts_pending_pull_on_cancellation(
        self,
    ) -> None:
        class PendingResponse:
            def __init__(self) -> None:
                self.input_token_count = 0
                self.output_token_count = 0
                self._response_iterator = None
                self.started = AsyncEvent()
                self.pull_cancelled = False
                self.cancel_count = 0
                self.close_count = 0

            def __aiter__(self) -> "PendingResponse":
                self._response_iterator = self
                return self

            async def __anext__(self) -> object:
                self.started.set()
                try:
                    await AsyncEvent().wait()
                except CancelledError:
                    self.pull_cancelled = True
                    raise
                return Token(token="late")

            async def cancel(self) -> None:
                self.cancel_count += 1

            async def aclose(self) -> None:
                self.close_count += 1

            async def to_str(self) -> str:
                return ""

        response = PendingResponse()
        request_model = ChatCompletionRequest(
            model="gpt",
            messages=[ChatMessage(role="user", content="hi")],
            stream=True,
        )
        orchestrator = MagicMock()
        orchestrator.sync_messages = AsyncMock()
        cancel_event = AsyncEvent()

        async def collect() -> list[dict[str, Any]]:
            payloads: list[dict[str, Any]] = []
            async for chunk in mcp_router._stream_mcp_response(
                request_id="id",
                request_model=request_model,
                response=response,
                response_id=uuid4(),
                timestamp=1,
                progress_token="tok",
                orchestrator=orchestrator,
                logger=MagicMock(),
                resource_store=mcp_router.MCPResourceStore(),
                base_path="/base",
                cancel_event=cancel_event,
            ):
                payloads.extend(
                    loads(part)
                    for part in chunk.decode("utf-8").splitlines()
                    if part
                )
            return payloads

        task = create_task(collect())
        await response.started.wait()
        cancel_event.set()
        payloads = await task

        self.assertTrue(response.pull_cancelled)
        self.assertEqual(response.cancel_count, 1)
        self.assertEqual(response.close_count, 1)
        self.assertEqual(payloads[-1]["error"]["code"], -32000)
        orchestrator.sync_messages.assert_awaited()

    async def test_stream_response_closes_resources_after_cancellation(
        self,
    ) -> None:
        call = ToolCall(id="call-1", name="run", arguments={})
        result = ToolCallResult(
            id="res-1",
            call=call,
            name="run",
            arguments={},
            result={"stdout": "partial"},
        )
        response = DummyResponse(
            [
                Event(
                    type=EventType.TOOL_RESULT,
                    payload={"result": result},
                    started=1.0,
                    finished=2.0,
                    elapsed=1.0,
                ),
                Token(token="late"),
            ]
        )
        request_model = ChatCompletionRequest(
            model="gpt",
            messages=[ChatMessage(role="user", content="hi")],
            stream=True,
        )
        orchestrator = MagicMock()
        orchestrator.sync_messages = AsyncMock()
        cancel_event = AsyncEvent()
        stream = mcp_router._stream_mcp_response(
            request_id="id",
            request_model=request_model,
            response=response,
            response_id=uuid4(),
            timestamp=1,
            progress_token="tok",
            orchestrator=orchestrator,
            logger=MagicMock(),
            resource_store=mcp_router.MCPResourceStore(),
            base_path="/base",
            cancel_event=cancel_event,
        )

        first = await anext(stream)
        first_payload = loads(first.decode("utf-8"))
        self.assertEqual(
            first_payload["method"], "notifications/resources/updated"
        )
        self.assertEqual(
            first_payload["params"]["resources"][0]["delta"]["set"]["text"],
            "partial",
        )

        cancel_event.set()
        payloads = [first_payload]
        async for chunk in stream:
            payloads.extend(
                loads(part)
                for part in chunk.decode("utf-8").splitlines()
                if part
            )

        closed_resources = [
            item
            for item in payloads
            if item.get("method") == "notifications/resources/updated"
            and item["params"]["resources"][0].get("closed") is True
        ]
        self.assertEqual(len(closed_resources), 1)
        self.assertEqual(payloads[-1]["error"]["code"], -32000)
        self.assertTrue(response._closed)

    async def test_stream_response_cleans_up_when_consumer_closes(
        self,
    ) -> None:
        class CleanupTrackingResponse(DummyResponse):
            def __init__(self, items: list[Any]) -> None:
                super().__init__(items)
                self.cancel_count = 0
                self.close_count = 0

            async def cancel(self) -> None:
                self.cancel_count += 1

            async def aclose(self) -> None:
                self.close_count += 1
                await super().aclose()

        call = ToolCall(id="call-1", name="run", arguments={})
        result = ToolCallResult(
            id="res-1",
            call=call,
            name="run",
            arguments={},
            result={"stdout": "partial"},
        )
        response = CleanupTrackingResponse(
            [
                Event(
                    type=EventType.TOOL_RESULT,
                    payload={"result": result},
                    started=1.0,
                    finished=2.0,
                    elapsed=1.0,
                ),
                Token(token="late"),
            ]
        )
        orchestrator = MagicMock()
        orchestrator.sync_messages = AsyncMock()
        cancel_event = AsyncEvent()
        resource_store = mcp_router.MCPResourceStore()
        stream = mcp_router._stream_mcp_response(
            request_id="id",
            request_model=ChatCompletionRequest(
                model="gpt",
                messages=[ChatMessage(role="user", content="hi")],
                stream=True,
            ),
            response=response,
            response_id=uuid4(),
            timestamp=1,
            progress_token="tok",
            orchestrator=orchestrator,
            logger=MagicMock(),
            resource_store=resource_store,
            base_path="/base",
            cancel_event=cancel_event,
        )

        first = await anext(stream)
        first_payload = loads(first.decode("utf-8"))
        resource_id = first_payload["params"]["resources"][0]["uri"].rsplit(
            "/", 1
        )[-1]
        await stream.aclose()

        self.assertTrue(cancel_event.is_set())
        self.assertEqual(response.cancel_count, 1)
        self.assertEqual(response.close_count, 1)
        self.assertTrue(response._closed)
        self.assertTrue((await resource_store.get(resource_id)).closed)
        orchestrator.sync_messages.assert_awaited()

    async def test_stream_response_honors_cancellation_after_pull(
        self,
    ) -> None:
        class CancellingResponse:
            def __init__(self, cancel_event: AsyncEvent) -> None:
                self.input_token_count = 0
                self.output_token_count = 0
                self._response_iterator = None
                self._cancel_event = cancel_event
                self._pulled = False
                self.cancel_count = 0
                self.close_count = 0

            def __aiter__(self) -> "CancellingResponse":
                self._response_iterator = self
                return self

            async def __anext__(self) -> object:
                if self._pulled:
                    raise StopAsyncIteration
                self._pulled = True
                self._cancel_event.set()
                return Token(token="late")

            async def cancel(self) -> None:
                self.cancel_count += 1

            async def aclose(self) -> None:
                self.close_count += 1

            async def to_str(self) -> str:
                return ""

        cancel_event = AsyncEvent()
        response = CancellingResponse(cancel_event)
        request_model = ChatCompletionRequest(
            model="gpt",
            messages=[ChatMessage(role="user", content="hi")],
            stream=True,
        )
        orchestrator = MagicMock()
        orchestrator.sync_messages = AsyncMock()

        payloads: list[dict[str, Any]] = []
        async for chunk in mcp_router._stream_mcp_response(
            request_id="id",
            request_model=request_model,
            response=response,
            response_id=uuid4(),
            timestamp=1,
            progress_token="tok",
            orchestrator=orchestrator,
            logger=MagicMock(),
            resource_store=mcp_router.MCPResourceStore(),
            base_path="/base",
            cancel_event=cancel_event,
        ):
            payloads.extend(
                loads(part)
                for part in chunk.decode("utf-8").splitlines()
                if part
            )

        self.assertEqual(response.cancel_count, 1)
        self.assertEqual(response.close_count, 1)
        self.assertEqual(payloads[-1]["error"]["code"], -32000)
        self.assertNotIn("late", dumps(payloads))

    async def test_stream_response_handles_cancel_racing_exhaustion(
        self,
    ) -> None:
        class ExhaustingResponse:
            def __init__(self, cancel_event: AsyncEvent) -> None:
                self.input_token_count = 0
                self.output_token_count = 0
                self._response_iterator = None
                self._cancel_event = cancel_event
                self._pulled = False
                self.cancel_count = 0
                self.close_count = 0

            def __aiter__(self) -> "ExhaustingResponse":
                self._response_iterator = self
                return self

            async def __anext__(self) -> object:
                if self._pulled:
                    raise StopAsyncIteration
                self._pulled = True
                self._cancel_event.set()
                raise StopAsyncIteration

            async def cancel(self) -> None:
                self.cancel_count += 1

            async def aclose(self) -> None:
                self.close_count += 1

            async def to_str(self) -> str:
                return ""

        cancel_event = AsyncEvent()
        response = ExhaustingResponse(cancel_event)
        orchestrator = MagicMock()
        orchestrator.sync_messages = AsyncMock()

        payloads: list[dict[str, Any]] = []
        async for chunk in mcp_router._stream_mcp_response(
            request_id="id",
            request_model=ChatCompletionRequest(
                model="gpt",
                messages=[ChatMessage(role="user", content="hi")],
                stream=True,
            ),
            response=response,
            response_id=uuid4(),
            timestamp=1,
            progress_token="tok",
            orchestrator=orchestrator,
            logger=MagicMock(),
            resource_store=mcp_router.MCPResourceStore(),
            base_path="/base",
            cancel_event=cancel_event,
        ):
            payloads.extend(
                loads(part)
                for part in chunk.decode("utf-8").splitlines()
                if part
            )

        self.assertEqual(response.cancel_count, 1)
        self.assertEqual(response.close_count, 1)
        self.assertEqual(payloads[-1]["error"]["code"], -32000)
        self.assertEqual(payloads[-1]["error"]["message"], "Request cancelled")
        orchestrator.sync_messages.assert_awaited()

    async def test_stream_response_handles_exception(self) -> None:
        def boom() -> None:
            raise RuntimeError("boom")

        response = DummyResponse([boom])
        request_model = ChatCompletionRequest(
            model="gpt",
            messages=[ChatMessage(role="user", content="hi")],
            stream=True,
        )
        orchestrator = MagicMock()
        orchestrator.sync_messages = AsyncMock()
        cancel_event = AsyncEvent()
        store = mcp_router.MCPResourceStore()

        payloads: list[dict[str, Any]] = []
        async for chunk in mcp_router._stream_mcp_response(
            request_id="id",
            request_model=request_model,
            response=response,
            response_id=uuid4(),
            timestamp=1,
            progress_token="tok",
            orchestrator=orchestrator,
            logger=MagicMock(),
            resource_store=store,
            base_path="/base",
            cancel_event=cancel_event,
        ):
            payloads.extend(
                loads(part)
                for part in chunk.decode("utf-8").splitlines()
                if part
            )

        orchestrator.sync_messages.assert_awaited()
        self.assertTrue(cancel_event.is_set())
        errors = [item for item in payloads if isinstance(item, dict)]
        self.assertTrue(errors)
        self.assertEqual(errors[-1]["error"]["code"], -32603)

    async def test_initialize_returns_server_info(self) -> None:
        endpoint = self._get_route("/initialize")
        message = {
            "jsonrpc": "2.0",
            "id": "init-1",
            "method": "initialize",
            "params": {
                "clientInfo": {"name": "tester", "version": "0.0.1"},
                "protocolVersion": "0.1.0",
            },
        }
        body = (dumps(message) + mcp_router.RS).encode("utf-8")
        request = DummyRequest(body)
        request.app.title = "Avalan MCP"
        request.app.version = "9.9.9"

        tool_manager = SimpleNamespace(
            is_empty=True, json_schemas=lambda: None
        )
        orchestrator = DummyOrchestrator(tool_manager)

        response = await endpoint(
            request,
            logger=getLogger("test.initialize"),
            orchestrator=orchestrator,
        )

        payload = loads(response.body.decode("utf-8"))
        result = payload["result"]
        self.assertEqual(
            result["serverInfo"], {"name": "Avalan MCP", "version": "9.9.9"}
        )
        self.assertIn("tools", result["capabilities"])
        self.assertTrue(result["capabilities"]["tools"]["call"])
        self.assertTrue(result["capabilities"]["resources"]["subscribe"])

    async def test_list_tools_returns_schemas(self) -> None:
        endpoint = self._get_route("/tools/list")
        message = {
            "jsonrpc": "2.0",
            "id": "list-1",
            "method": "tools/list",
            "params": {"limit": 10},
        }
        body = (dumps(message) + mcp_router.RS).encode("utf-8")
        request = DummyRequest(body)
        request.app.title = "Avalan MCP"
        request.app.version = "1.0.0"

        tool_manager = MagicMock()
        tool_manager.is_empty = False
        tool_manager.json_schemas.return_value = []
        orchestrator = DummyOrchestrator(tool_manager)

        response = await endpoint(
            request,
            logger=getLogger("test.list"),
            orchestrator=orchestrator,
        )

        payload = loads(response.body.decode("utf-8"))
        result = payload["result"]
        self.assertNotIn("nextCursor", result)
        self.assertEqual(len(result["tools"]), 1)
        run_tool = result["tools"][0]
        self.assertEqual(run_tool["name"], "run")
        self.assertEqual(
            run_tool["description"],
            "Execute the Avalan orchestrator run endpoint.",
        )
        self.assertEqual(
            run_tool["inputSchema"],
            MCPToolRequest.model_json_schema(),
        )
        tool_manager.json_schemas.assert_not_called()

    async def test_ping_endpoint_returns_empty_result(self) -> None:
        endpoint = self._get_route("/ping")
        message = {
            "jsonrpc": "2.0",
            "id": "ping-ep",
            "method": "ping",
            "params": {"message": "keepalive"},
        }
        body = (dumps(message) + mcp_router.RS).encode("utf-8")
        request = DummyRequest(body)

        response = await endpoint(
            request,
            logger=getLogger("test.ping"),
        )

        payload = loads(response.body.decode("utf-8"))
        self.assertEqual(payload["id"], "ping-ep")
        self.assertEqual(payload["result"], {})

    async def test_initialized_notification_endpoint_returns_no_content(
        self,
    ) -> None:
        endpoint = self._get_route("/notifications/initialized")
        message = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {"capabilities": {}},
        }
        body = (dumps(message) + mcp_router.RS).encode("utf-8")
        request = DummyRequest(body)

        response = await endpoint(
            request,
            logger=getLogger("test.initialized.endpoint"),
        )

        self.assertEqual(response.status_code, 204)
        self.assertEqual(response.body, b"")

    async def test_initialized_notification_endpoint_rejects_id(self) -> None:
        endpoint = self._get_route("/notifications/initialized")
        message = {
            "jsonrpc": "2.0",
            "id": "not-allowed",
            "method": "notifications/initialized",
        }
        body = (dumps(message) + mcp_router.RS).encode("utf-8")
        request = DummyRequest(body)

        with self.assertRaises(mcp_router.HTTPException) as exc:
            await endpoint(
                request,
                logger=getLogger("test.initialized.id"),
            )

        self.assertIn("cannot include an id", str(exc.exception.detail))

    async def test_initialized_notification_endpoint_rejects_invalid_params(
        self,
    ) -> None:
        endpoint = self._get_route("/notifications/initialized")
        message = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": "oops",
        }
        body = (dumps(message) + mcp_router.RS).encode("utf-8")
        request = DummyRequest(body)

        with self.assertRaises(mcp_router.HTTPException) as exc:
            await endpoint(
                request,
                logger=getLogger("test.initialized.params"),
            )

        self.assertIn("Missing MCP params", str(exc.exception.detail))

    async def test_base_rpc_initialize_dispatch(self) -> None:
        endpoint = self._get_route("/")
        message = {
            "jsonrpc": "2.0",
            "id": "init-2",
            "method": "initialize",
            "params": {"protocolVersion": "0.1.0"},
        }
        body = (dumps(message) + mcp_router.RS).encode("utf-8")
        request = DummyRequest(body)
        request.app.title = "Avalan MCP"
        request.app.version = "2.0.0"

        tool_manager = SimpleNamespace(
            is_empty=True, json_schemas=lambda: None
        )
        orchestrator = DummyOrchestrator(tool_manager)

        response = await endpoint(
            request,
            logger=getLogger("test.rpc.init"),
            orchestrator=orchestrator,
        )
        payload = loads(response.body.decode("utf-8"))
        result = payload["result"]
        self.assertEqual(
            result["serverInfo"], {"name": "Avalan MCP", "version": "2.0.0"}
        )

    async def test_base_rpc_tools_list_dispatch(self) -> None:
        endpoint = self._get_route("/")
        message = {
            "jsonrpc": "2.0",
            "id": "list-2",
            "method": "tools/list",
            "params": {"limit": 1},
        }
        body = (dumps(message) + mcp_router.RS).encode("utf-8")
        request = DummyRequest(body)
        request.app.title = "Avalan MCP"
        request.app.version = "3.0.0"

        orchestrator = DummyOrchestrator(SimpleNamespace(is_empty=False))

        response = await endpoint(
            request,
            logger=getLogger("test.rpc.list"),
            orchestrator=orchestrator,
        )
        payload = loads(response.body.decode("utf-8"))
        self.assertIn("tools", payload["result"])

    async def test_base_rpc_ping_dispatch(self) -> None:
        endpoint = self._get_route("/")
        message = {
            "jsonrpc": "2.0",
            "id": "ping-rpc",
            "method": "ping",
            "params": {"message": "still-here"},
        }
        body = (dumps(message) + mcp_router.RS).encode("utf-8")
        request = DummyRequest(body)
        orchestrator = DummyOrchestrator(SimpleNamespace(is_empty=True))

        response = await endpoint(
            request,
            logger=getLogger("test.rpc.ping"),
            orchestrator=orchestrator,
        )

        payload = loads(response.body.decode("utf-8"))
        self.assertEqual(payload["id"], "ping-rpc")
        self.assertEqual(payload["result"], {})

    async def test_base_rpc_handles_initialized_notification(self) -> None:
        endpoint = self._get_route("/")
        message = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        }
        body = (dumps(message) + mcp_router.RS).encode("utf-8")
        request = DummyRequest(body)
        orchestrator = DummyOrchestrator(SimpleNamespace(is_empty=True))

        response = await endpoint(
            request,
            logger=getLogger("test.rpc.initialized"),
            orchestrator=orchestrator,
        )

        self.assertEqual(response.status_code, 204)
        self.assertEqual(response.body, b"")

    async def test_base_rpc_ping_dispatch_generates_id(self) -> None:
        endpoint = self._get_route("/")
        message = {"jsonrpc": "2.0", "method": "ping"}
        body = (dumps(message) + mcp_router.RS).encode("utf-8")
        request = DummyRequest(body)
        orchestrator = DummyOrchestrator(SimpleNamespace(is_empty=True))
        generated_id = UUID("00000000-0000-0000-0000-000000000099")

        with patch.object(mcp_router, "uuid4", return_value=generated_id):
            response = await endpoint(
                request,
                logger=getLogger("test.rpc.ping"),
                orchestrator=orchestrator,
            )

        payload = loads(response.body.decode("utf-8"))
        self.assertEqual(payload["id"], str(generated_id))
        self.assertEqual(payload["result"], {})

    async def test_base_rpc_allowed_method_without_handler(self) -> None:
        endpoint = self._get_route("/")
        request = DummyRequest(b"")
        request.app.title = "Avalan MCP"
        request.app.version = "4.0.0"

        orchestrator = DummyOrchestrator(SimpleNamespace(is_empty=True))

        method = OneTimeMatchStr("initialize")
        method._first_compare = False  # type: ignore[attr-defined]
        message = {"jsonrpc": "2.0", "id": "weird", "method": method}

        async def empty_iter():
            if False:
                yield {}

        with patch.object(
            mcp_router,
            "_expect_jsonrpc_message",
            AsyncMock(return_value=(message, empty_iter())),
        ):
            with self.assertRaises(mcp_router.HTTPException) as exc:
                await endpoint(
                    request,
                    logger=getLogger("test.rpc.unhandled"),
                    orchestrator=orchestrator,
                )
        self.assertIn("Unsupported MCP method", str(exc.exception.detail))

    async def test_parse_call_request(self) -> None:
        message = {
            "jsonrpc": "2.0",
            "id": "call-2",
            "method": "tools/call",
            "params": {
                "name": "run",
                "arguments": {
                    "input_string": "Hello",
                },
            },
        }
        req = DummyRequest(b"")
        req.app.state.mcp_resource_base_path = "/m"

        # empty iterator of remaining messages
        async def _empty():
            if False:
                yield {}

        rid, req_model, token = mcp_router._parse_call_request(
            req, message, _empty()
        )
        self.assertEqual(rid, "call-2")
        self.assertEqual(req_model.input_string, "Hello")
        self.assertTrue(token)

    async def test_tool_event_notifications_without_item(self) -> None:
        tool_summaries: dict[str, dict[str, Any]] = {}
        resources: dict[str, mcp_router.MCPResource] = {}
        store = mcp_router.MCPResourceStore()
        event = Event(type=EventType.TOOL_PROCESS, payload=None)
        items = []
        async for item in mcp_router._tool_event_notifications(
            event=event,
            tool_summaries=tool_summaries,
            resources=resources,
            resource_store=store,
            base_path="/base",
        ):
            items.append(item)
        self.assertFalse(items)
        self.assertFalse(tool_summaries)
        self.assertFalse(resources)

    async def test_tool_event_notifications_ignores_non_string_call_id(
        self,
    ) -> None:
        tool_summaries: dict[str, dict[str, Any]] = {}
        resources: dict[str, mcp_router.MCPResource] = {}
        store = mcp_router.MCPResourceStore()
        event = Event(type=EventType.TOOL_PROCESS, payload=None)

        with patch.object(
            mcp_router, "_tool_call_event_item", return_value={"id": 1}
        ):
            items = []
            async for item in mcp_router._tool_event_notifications(
                event=event,
                tool_summaries=tool_summaries,
                resources=resources,
                resource_store=store,
                base_path="/base",
            ):
                items.append(item)

        self.assertFalse(items)
        self.assertFalse(tool_summaries)
        self.assertFalse(resources)

    async def test_tool_event_notifications_append_existing_resource(
        self,
    ) -> None:
        call = ToolCall(id="c1", name="run", arguments={})
        result = ToolCallResult(
            id="r1",
            call=call,
            name="run",
            arguments={},
            result={"stdout": "one"},
        )
        event = Event(
            type=EventType.TOOL_RESULT,
            payload={"result": result},
            started=1.0,
            finished=2.0,
            elapsed=1.0,
        )
        store = mcp_router.MCPResourceStore()
        tool_summaries: dict[str, dict[str, Any]] = {}
        resources: dict[str, mcp_router.MCPResource] = {}
        first_notifications = []
        async for item in mcp_router._tool_event_notifications(
            event=event,
            tool_summaries=tool_summaries,
            resources=resources,
            resource_store=store,
            base_path="/base",
        ):
            first_notifications.append(item)
        self.assertTrue(first_notifications)
        self.assertIn(str(call.id), tool_summaries)
        self.assertEqual(
            tool_summaries[str(call.id)]["result"]["stdout"], "one"
        )

        result_second = ToolCallResult(
            id="r2",
            call=call,
            name="run",
            arguments={},
            result={"stdout": "two"},
        )
        event_second = Event(
            type=EventType.TOOL_RESULT,
            payload={"result": result_second},
            started=2.0,
            finished=3.0,
            elapsed=1.0,
        )
        second_notifications = []
        async for item in mcp_router._tool_event_notifications(
            event=event_second,
            tool_summaries=tool_summaries,
            resources=resources,
            resource_store=store,
            base_path="/base",
        ):
            second_notifications.append(item)
        self.assertTrue(second_notifications)
        resource = next(iter(resources.values()))
        stored = await store.get(resource.id)
        self.assertEqual(stored.text, "onetwo")
        self.assertEqual(len(tool_summaries[str(call.id)]["resources"]), 2)

    async def test_tool_event_notifications_serializes_result(self) -> None:
        call = ToolCall(id="c2", name="run", arguments={})
        result = ToolCallResult(
            id="r3",
            call=call,
            name="run",
            arguments={},
            result=SimpleNamespace(payload="value"),
        )
        event = Event(
            type=EventType.TOOL_RESULT,
            payload={"result": result},
            started=3.0,
            finished=4.0,
            elapsed=1.0,
        )
        store = mcp_router.MCPResourceStore()
        tool_summaries: dict[str, dict[str, Any]] = {}
        resources: dict[str, mcp_router.MCPResource] = {}
        with patch.object(
            mcp_router, "to_json", return_value='{"payload":"value"}'
        ):
            notifications = []
            async for item in mcp_router._tool_event_notifications(
                event=event,
                tool_summaries=tool_summaries,
                resources=resources,
                resource_store=store,
                base_path="/base",
            ):
                notifications.append(item)
        self.assertFalse(resources)
        self.assertIn(str(call.id), tool_summaries)
        summary = tool_summaries[str(call.id)]
        self.assertEqual(summary["result"], '{"payload":"value"}')
        message = notifications[-1]["params"]["message"]
        self.assertEqual(message["resultDelta"], '{"payload":"value"}')

    async def test_tool_event_notifications_serializes_diagnostic(
        self,
    ) -> None:
        call = ToolCall(id="c-d", name="missing", arguments={})
        diagnostic = ToolCallDiagnostic(
            id="diag-d",
            call_id=call.id,
            requested_name="missing",
            code=ToolCallDiagnosticCode.UNKNOWN_TOOL,
            stage=ToolCallDiagnosticStage.RESOLVE,
            message="Unknown tool.",
        )
        event = Event(
            type=EventType.TOOL_DIAGNOSTIC,
            payload={"call": call, "diagnostic": diagnostic},
            started=4.0,
            finished=5.0,
            elapsed=1.0,
        )
        store = mcp_router.MCPResourceStore()
        tool_summaries: dict[str, dict[str, Any]] = {}
        resources: dict[str, mcp_router.MCPResource] = {}
        notifications = []
        async for item in mcp_router._tool_event_notifications(
            event=event,
            tool_summaries=tool_summaries,
            resources=resources,
            resource_store=store,
            base_path="/base",
        ):
            notifications.append(item)

        self.assertFalse(resources)
        self.assertIn(str(call.id), tool_summaries)
        summary = tool_summaries[str(call.id)]
        self.assertEqual(summary["diagnostic"]["code"], "tool.unknown")
        message = notifications[-1]["params"]["message"]
        self.assertEqual(message["type"], "tool.diagnostic")
        self.assertEqual(message["toolCallId"], "c-d")
        self.assertEqual(message["diagnostic"]["message"], "Unknown tool.")


class MCPRouterEdgeCaseAsyncTestCase(IsolatedAsyncioTestCase):
    def _chat_request(self, stream: bool = False) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="hi")],
            stream=stream,
        )

    async def test_cleanup_mcp_stream_sources_logs_single_failure(
        self,
    ) -> None:
        class ClosingSource:
            async def aclose(self) -> None:
                raise RuntimeError("close failed")

        logger = MagicMock()

        await mcp_router._cleanup_mcp_stream_sources(
            logger, ClosingSource(), cancelled=False
        )

        logger.exception.assert_called_once()
        self.assertEqual(
            logger.exception.call_args.args[0],
            "MCP stream source cleanup failed",
        )
        self.assertIsInstance(
            logger.exception.call_args.kwargs["exc_info"], RuntimeError
        )

    async def test_cleanup_mcp_stream_sources_logs_group_failure(
        self,
    ) -> None:
        class CancellingSource:
            async def cancel(self) -> None:
                raise CancelledError()

            async def aclose(self) -> None:
                raise RuntimeError("close failed")

        logger = MagicMock()

        await mcp_router._cleanup_mcp_stream_sources(
            logger, CancellingSource(), cancelled=True
        )

        logger.exception.assert_called_once()
        self.assertEqual(
            logger.exception.call_args.args[0],
            "MCP stream source cleanup failed",
        )
        self.assertIsInstance(
            logger.exception.call_args.kwargs["exc_info"], BaseExceptionGroup
        )

    async def test_expect_jsonrpc_message_invalid_payload(self) -> None:
        body = b"[1]"
        request = DummyRequest(body)
        with self.assertRaises(mcp_router.HTTPException) as exc:
            await mcp_router._expect_jsonrpc_message(request, {"tools/list"})
        self.assertIn("Invalid MCP payload", str(exc.exception.detail))

    async def test_expect_jsonrpc_message_unsupported_method(self) -> None:
        message = {"jsonrpc": "2.0", "id": "x", "method": "bad"}
        body = (dumps(message) + mcp_router.RS).encode("utf-8")
        request = DummyRequest(body)
        with self.assertRaises(mcp_router.HTTPException) as exc:
            await mcp_router._expect_jsonrpc_message(request, {"tools/list"})
        self.assertIn("Unsupported MCP method", str(exc.exception.detail))

    async def test_consume_call_request_missing_params(self) -> None:
        message = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": "bad",
        }
        body = (dumps(message) + mcp_router.RS).encode("utf-8")
        request = DummyRequest(body)
        with self.assertRaises(mcp_router.HTTPException) as exc:
            await mcp_router._consume_call_request(request)
        self.assertIn("Missing MCP params", str(exc.exception.detail))

    async def test_consume_call_request_invalid_arguments(self) -> None:
        message = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": "run", "arguments": []},
        }
        body = (dumps(message) + mcp_router.RS).encode("utf-8")
        request = DummyRequest(body)
        with self.assertRaises(mcp_router.HTTPException) as exc:
            await mcp_router._consume_call_request(request)
        self.assertIn("Invalid tool arguments", str(exc.exception.detail))

    async def test_consume_call_request_none_arguments(self) -> None:
        message = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": "run", "arguments": None},
        }
        body = (dumps(message) + mcp_router.RS).encode("utf-8")
        request = DummyRequest(body)
        with patch.object(
            mcp_router, "_extract_call_arguments", return_value=None
        ) as extractor:
            with self.assertRaises(mcp_router.HTTPException) as exc:
                await mcp_router._consume_call_request(request)
        extractor.assert_called_once()
        self.assertIn("Invalid tool arguments", str(exc.exception.detail))

    async def test_consume_call_request_with_progress_token(self) -> None:
        message = {
            "jsonrpc": "2.0",
            "id": "call-progress",
            "method": "tools/call",
            "params": {
                "name": "run",
                "arguments": {
                    "input_string": "hi",
                },
                "progressToken": "tok-1",
            },
        }
        body = (dumps(message) + mcp_router.RS).encode("utf-8")
        request = DummyRequest(body)
        request.app.state.mcp_resource_base_path = "/m"
        (
            request_id,
            tool_request,
            progress,
        ) = await mcp_router._consume_call_request(request)
        self.assertEqual(request_id, "call-progress")
        self.assertEqual(progress, "tok-1")
        self.assertEqual(tool_request.input_string, "hi")

    async def test_consume_call_request_validation_error(self) -> None:
        message = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "run",
                "arguments": {
                    "input_string": "hi",
                },
            },
        }
        body = (dumps(message) + mcp_router.RS).encode("utf-8")
        request = DummyRequest(body)
        with patch.object(
            mcp_router.MCPToolRequest,
            "model_validate",
            side_effect=ValueError("bad"),
        ):
            with self.assertRaises(mcp_router.HTTPException) as exc:
                await mcp_router._consume_call_request(request)
        self.assertIn("Invalid MCP arguments", str(exc.exception.detail))

    async def test_consume_call_request_generates_defaults(self) -> None:
        message = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "run",
                "arguments": {
                    "input_string": "hi",
                },
            },
        }
        follow_up = {"jsonrpc": "2.0", "method": "ping"}
        body = (
            dumps(message) + mcp_router.RS + dumps(follow_up) + mcp_router.RS
        ).encode("utf-8")
        request = DummyRequest(body)
        request.app.state.mcp_resource_base_path = "/m"
        with patch.object(mcp_router, "uuid4", return_value=UUID(int=1234)):
            (
                request_id,
                tool_request,
                progress,
            ) = await mcp_router._consume_call_request(request)
        self.assertEqual(request_id, str(UUID(int=1234)))
        self.assertEqual(progress, str(UUID(int=1234)))
        self.assertEqual(tool_request.input_string, "hi")
        remaining = []
        async for item in mcp_router._iter_jsonrpc_messages(request):
            remaining.append(item)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["method"], "ping")

    async def test_parse_call_request_invalid_method(
        self,
    ) -> None:
        request = DummyRequest(b"")

        async def _empty():
            if False:
                yield {}

        with self.assertRaises(mcp_router.HTTPException) as exc:
            mcp_router._parse_call_request(
                request,
                {"jsonrpc": "2.0", "method": "bad"},
                _empty(),
            )
        self.assertIn("Unsupported MCP method", str(exc.exception.detail))

    async def test_parse_call_request_missing_params(
        self,
    ) -> None:
        request = DummyRequest(b"")

        async def _empty():
            if False:
                yield {}

        with self.assertRaises(mcp_router.HTTPException) as exc:
            mcp_router._parse_call_request(
                request,
                {"jsonrpc": "2.0", "method": "tools/call", "params": "bad"},
                _empty(),
            )
        self.assertIn("Missing MCP params", str(exc.exception.detail))

    async def test_parse_call_request_invalid_arguments(
        self,
    ) -> None:
        request = DummyRequest(b"")

        async def _empty():
            if False:
                yield {}

        with self.assertRaises(mcp_router.HTTPException) as exc:
            mcp_router._parse_call_request(
                request,
                {
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {"name": "run", "arguments": []},
                },
                _empty(),
            )
        self.assertIn("Invalid tool arguments", str(exc.exception.detail))

    async def test_parse_call_request_none_arguments(
        self,
    ) -> None:
        request = DummyRequest(b"")

        async def _empty():
            if False:
                yield {}

        with patch.object(
            mcp_router, "_extract_call_arguments", return_value=None
        ) as extractor:
            with self.assertRaises(mcp_router.HTTPException) as exc:
                mcp_router._parse_call_request(
                    request,
                    {
                        "jsonrpc": "2.0",
                        "method": "tools/call",
                        "params": {"name": "run", "arguments": None},
                    },
                    _empty(),
                )
        extractor.assert_called_once()
        self.assertIn("Invalid tool arguments", str(exc.exception.detail))

    async def test_parse_call_request_validation_error(
        self,
    ) -> None:
        request = DummyRequest(b"")

        async def _empty():
            if False:
                yield {}

        with patch.object(
            mcp_router.MCPToolRequest,
            "model_validate",
            side_effect=ValueError("bad"),
        ):
            with self.assertRaises(mcp_router.HTTPException) as exc:
                mcp_router._parse_call_request(
                    request,
                    {
                        "jsonrpc": "2.0",
                        "method": "tools/call",
                        "params": {
                            "name": "run",
                            "arguments": {
                                "input_string": "hi",
                            },
                        },
                    },
                    _empty(),
                )
        self.assertIn("Invalid MCP arguments", str(exc.exception.detail))

    async def test_parse_call_request_progress_token(
        self,
    ) -> None:
        request = DummyRequest(b"")

        async def _empty():
            if False:
                yield {}

        rid, tool_request, token = mcp_router._parse_call_request(
            request,
            {
                "jsonrpc": "2.0",
                "id": "call-3",
                "method": "tools/call",
                "params": {
                    "name": "run",
                    "arguments": {
                        "input_string": "hi",
                    },
                    "progressToken": "tok-2",
                },
            },
            _empty(),
        )
        self.assertEqual(rid, "call-3")
        self.assertEqual(token, "tok-2")
        self.assertEqual(tool_request.input_string, "hi")

    async def test_parse_call_request_stores_iterator(
        self,
    ) -> None:
        request = DummyRequest(b"")

        async def remaining() -> AsyncIterator[dict[str, Any]]:
            yield {"jsonrpc": "2.0", "method": "follow"}

        message = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "run",
                "arguments": {
                    "input_string": "hi",
                },
            },
        }
        mcp_router._parse_call_request(
            request,
            message,
            remaining(),
        )
        messages = []
        async for item in mcp_router._iter_jsonrpc_messages(request):
            messages.append(item)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["method"], "follow")

    async def test_iter_jsonrpc_messages_invalid_json(self) -> None:
        request = DummyRequest(b"{invalid}")
        with self.assertRaises(mcp_router.HTTPException) as exc:
            async for _ in mcp_router._iter_jsonrpc_messages(request):
                pass
        self.assertIn("Invalid MCP payload", str(exc.exception.detail))

    async def test_iter_jsonrpc_messages_invalid_segment(self) -> None:
        payload = b"{invalid}" + mcp_router.RS.encode("utf-8")
        request = DummyRequest(payload)
        with self.assertRaises(mcp_router.HTTPException) as exc:
            async for _ in mcp_router._iter_jsonrpc_messages(request):
                pass
        self.assertIn("Invalid MCP payload", str(exc.exception.detail))

    async def test_iter_jsonrpc_messages_without_delimiter(self) -> None:
        message = {"jsonrpc": "2.0", "method": "tools/list"}
        body = dumps(message).encode("utf-8")
        request = DummyRequest(body)
        messages = []
        async for item in mcp_router._iter_jsonrpc_messages(request):
            messages.append(item)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["method"], "tools/list")

    async def test_iter_jsonrpc_messages_uses_stored_iterator(self) -> None:
        async def stored() -> AsyncIterator[dict[str, Any]]:
            yield {"jsonrpc": "2.0", "method": "tools/list"}

        request = DummyRequest(b"")
        request.state._mcp_message_iter = stored()
        messages = []
        async for item in mcp_router._iter_jsonrpc_messages(request):
            messages.append(item)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["method"], "tools/list")

    async def test_iter_jsonrpc_messages_skips_empty_chunks(self) -> None:
        message = {"jsonrpc": "2.0", "method": "tools/list"}
        body = [
            b"",
            b" \x1e",
            dumps(message).encode("utf-8"),
        ]
        request = DummyRequest(body)
        messages = []
        async for item in mcp_router._iter_jsonrpc_messages(request):
            messages.append(item)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["method"], "tools/list")

    async def test_watch_for_cancellation_sets_event(self) -> None:
        async def generator() -> AsyncIterator[dict[str, Any]]:
            yield {"method": "notifications/cancelled"}

        cancel_event = AsyncEvent()
        logger = MagicMock()
        await mcp_router._watch_for_cancellation(
            generator(), cancel_event, logger
        )
        self.assertTrue(cancel_event.is_set())

    async def test_watch_for_cancellation_ignores_non_dict(self) -> None:
        async def generator() -> AsyncIterator[Any]:
            yield "noise"
            yield {"method": "notifications/cancelled"}

        cancel_event = AsyncEvent()
        logger = MagicMock()
        await mcp_router._watch_for_cancellation(
            generator(), cancel_event, logger
        )
        self.assertTrue(cancel_event.is_set())
        logger.debug.assert_called_once()

    async def test_close_response_iterator_handles_missing_aclose(
        self,
    ) -> None:
        class Holder:
            def __init__(self) -> None:
                self._response_iterator = object()

        holder = Holder()
        await mcp_router._close_response_iterator(holder)

    async def test_close_response_iterator_closes_iterator(self) -> None:
        class Iterator:
            def __init__(self) -> None:
                self.closed = False

            async def aclose(self) -> None:
                self.closed = True

        class Holder:
            def __init__(self) -> None:
                self._response_iterator = Iterator()

        holder = Holder()
        await mcp_router._close_response_iterator(holder)
        self.assertTrue(holder._response_iterator.closed)

    async def test_start_tool_streaming_response_non_stream(self) -> None:
        request = DummyRequest(b"")
        tool_request = MCPToolRequest(input_string="hi")
        response_object = DummyResponse([])
        response_object.text = "answer"
        response_object.input_token_count = 1
        response_object.output_token_count = 2
        orchestrator = MagicMock()
        orchestrator.sync_messages = AsyncMock()
        logger = getLogger("test.stream.non")

        async def no_messages():
            if False:
                yield {}

        request.state._mcp_message_iter = no_messages()
        with (
            patch.object(
                mcp_router,
                "orchestrate",
                AsyncMock(return_value=(response_object, UUID(int=1), 42)),
            ),
            patch.object(
                mcp_router,
                "_build_chat_request",
                return_value=self._chat_request(stream=False),
            ),
            patch.object(
                mcp_router, "create_task", side_effect=fake_create_task
            ),
        ):
            result = await mcp_router._start_tool_streaming_response(
                request,
                logger,
                orchestrator,
                "req-1",
                tool_request,
                "progress",
            )
        self.assertIsInstance(result, mcp_router.JSONResponse)
        payload = loads(result.body.decode("utf-8"))
        self.assertEqual(
            payload["result"]["structuredContent"]["model"], "test"
        )
        self.assertEqual(
            payload["result"]["structuredContent"]["usage"]["total_tokens"],
            3,
        )

    async def test_start_tool_streaming_response_empty_text(self) -> None:
        request = DummyRequest(b"")
        tool_request = MCPToolRequest(input_string="hi")
        response_object = DummyResponse([])
        response_object.text = ""
        response_object.input_token_count = 0
        response_object.output_token_count = 0
        orchestrator = MagicMock()
        orchestrator.sync_messages = AsyncMock()
        logger = getLogger("test.stream.empty")

        async def no_messages():
            if False:
                yield {}

        request.state._mcp_message_iter = no_messages()
        with (
            patch.object(
                mcp_router,
                "orchestrate",
                AsyncMock(return_value=(response_object, UUID(int=3), 7)),
            ),
            patch.object(
                mcp_router,
                "_build_chat_request",
                return_value=self._chat_request(stream=False),
            ),
            patch.object(
                mcp_router, "create_task", side_effect=fake_create_task
            ),
        ):
            result = await mcp_router._start_tool_streaming_response(
                request,
                logger,
                orchestrator,
                "req-empty",
                tool_request,
                "progress",
            )
        payload = loads(result.body.decode("utf-8"))
        self.assertEqual(payload["result"]["content"], [])
        self.assertEqual(
            payload["result"]["structuredContent"]["usage"]["total_tokens"],
            0,
        )

    async def test_start_tool_streaming_response_streaming(self) -> None:
        async def stream(**_: Any) -> AsyncIterator[bytes]:
            yield b'{"chunk":1}\n'

        request = DummyRequest(b"")
        tool_request = MCPToolRequest(input_string="hi")
        response_object = DummyResponse([])
        orchestrator = MagicMock()
        orchestrator.sync_messages = AsyncMock()
        logger = getLogger("test.stream.yes")

        async def no_messages():
            if False:
                yield {}

        request.state._mcp_message_iter = no_messages()
        with (
            patch.object(
                mcp_router,
                "orchestrate",
                AsyncMock(return_value=(response_object, UUID(int=2), 99)),
            ),
            patch.object(
                mcp_router,
                "_build_chat_request",
                return_value=self._chat_request(stream=True),
            ),
            patch.object(
                mcp_router, "_stream_mcp_response", side_effect=stream
            ),
            patch.object(
                mcp_router, "create_task", side_effect=fake_create_task
            ),
        ):
            result = await mcp_router._start_tool_streaming_response(
                request,
                logger,
                orchestrator,
                "req-2",
                tool_request,
                "progress",
            )
        self.assertIsInstance(result, mcp_router.StreamingResponse)
        chunks = []
        try:
            async for chunk in result.body_iterator:  # type: ignore[attr-defined]
                chunks.append(chunk)
        except CancelledError:
            pass
        self.assertTrue(chunks)
        self.assertTrue(any(b"answer.completed" in chunk for chunk in chunks))
        self.assertTrue(any(b"structuredContent" in chunk for chunk in chunks))

    async def test_stream_mcp_response_handles_tool_error(self) -> None:
        call = ToolCall(id="c", name="run", arguments={})
        tool_error = ToolCallError(
            id="err",
            call=call,
            name="run",
            arguments={},
            error=Exception("x"),
            message="bad",
        )
        response = DummyResponse(
            [
                Event(
                    type=EventType.TOOL_RESULT,
                    payload={"result": tool_error},
                    started=1.0,
                    finished=2.0,
                    elapsed=1.0,
                ),
                ToolCallToken(token="input", call=None),
            ]
        )
        response.input_token_count = 0
        response.output_token_count = 0
        orchestrator = MagicMock()
        orchestrator.sync_messages = AsyncMock()
        cancel_event = AsyncEvent()
        store = mcp_router.MCPResourceStore()
        request_model = self._chat_request(stream=True)
        payloads = []
        async for chunk in mcp_router._stream_mcp_response(
            request_id="id",
            request_model=request_model,
            response=response,
            response_id=uuid4(),
            timestamp=1,
            progress_token="tok",
            orchestrator=orchestrator,
            logger=MagicMock(),
            resource_store=store,
            base_path="/m",
            cancel_event=cancel_event,
        ):
            payloads.append(loads(chunk.decode("utf-8")))
        orchestrator.sync_messages.assert_awaited()
        errors = [
            item
            for item in payloads
            if isinstance(item, dict)
            and item.get("method") == "notifications/message"
        ]
        self.assertTrue(any("error" in e["params"]["message"] for e in errors))

    async def test_stream_mcp_response_handles_tool_diagnostic(self) -> None:
        call = ToolCall(id="c-d", name="missing", arguments={})
        diagnostic = ToolCallDiagnostic(
            id="diag-d",
            call_id=call.id,
            requested_name="missing",
            code=ToolCallDiagnosticCode.UNKNOWN_TOOL,
            stage=ToolCallDiagnosticStage.RESOLVE,
            message="Unknown tool.",
        )
        response = DummyResponse(
            [
                Event(
                    type=EventType.TOOL_DIAGNOSTIC,
                    payload={"call": call, "diagnostic": diagnostic},
                    started=1.0,
                    finished=2.0,
                    elapsed=1.0,
                ),
                Token(token="done"),
            ]
        )
        response.input_token_count = 0
        response.output_token_count = 0
        orchestrator = MagicMock()
        orchestrator.sync_messages = AsyncMock()
        cancel_event = AsyncEvent()
        store = mcp_router.MCPResourceStore()
        request_model = self._chat_request(stream=True)
        payloads = []
        async for chunk in mcp_router._stream_mcp_response(
            request_id="id",
            request_model=request_model,
            response=response,
            response_id=uuid4(),
            timestamp=1,
            progress_token="tok",
            orchestrator=orchestrator,
            logger=MagicMock(),
            resource_store=store,
            base_path="/m",
            cancel_event=cancel_event,
        ):
            payloads.append(loads(chunk.decode("utf-8")))
        orchestrator.sync_messages.assert_awaited()
        diagnostics = [
            item
            for item in payloads
            if isinstance(item, dict)
            and item.get("method") == "notifications/message"
            and item["params"]["message"].get("type") == "tool.diagnostic"
        ]
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(
            diagnostics[0]["params"]["message"]["diagnostic"]["code"],
            "tool.unknown",
        )

    async def test_stream_mcp_response_cancellation_closes_resources(
        self,
    ) -> None:
        cancel_event = AsyncEvent()
        store = mcp_router.MCPResourceStore()
        call = ToolCall(id="c", name="run", arguments={})
        tool_result = ToolCallResult(
            id="res",
            call=call,
            name="run",
            arguments={},
            result={"stdout": "log"},
        )

        def cancel() -> Token:
            cancel_event.set()
            return Token(token="ignored")

        response = DummyResponse(
            [
                Event(
                    type=EventType.TOOL_RESULT,
                    payload={"result": tool_result},
                ),
                cancel,
            ]
        )
        response.input_token_count = 1
        response.output_token_count = 1
        orchestrator = MagicMock()
        orchestrator.sync_messages = AsyncMock()
        request_model = self._chat_request(stream=True)
        payloads = []
        async for chunk in mcp_router._stream_mcp_response(
            request_id="id",
            request_model=request_model,
            response=response,
            response_id=uuid4(),
            timestamp=1,
            progress_token="tok",
            orchestrator=orchestrator,
            logger=MagicMock(),
            resource_store=store,
            base_path="/base",
            cancel_event=cancel_event,
        ):
            payloads.append(loads(chunk.decode("utf-8")))
        orchestrator.sync_messages.assert_awaited()
        self.assertTrue(response._closed)
        resource_updates = [
            item
            for item in payloads
            if isinstance(item, dict)
            and item.get("method") == "notifications/resources/updated"
        ]
        self.assertTrue(resource_updates)
        self.assertTrue(
            resource_updates[-1]["params"]["resources"][0]["closed"]
        )
        errors = [
            item
            for item in payloads
            if isinstance(item, dict) and "error" in item
        ]
        self.assertEqual(errors[-1]["error"]["code"], -32000)

    async def test_stream_mcp_response_logs_cleanup_failure(
        self,
    ) -> None:
        class FailingCleanupResponse(DummyResponse):
            async def cancel(self) -> None:
                raise CancelledError()

            async def aclose(self) -> None:
                raise RuntimeError("close failed")

        cancel_event = AsyncEvent()
        store = mcp_router.MCPResourceStore()
        tool_call = ToolCall(id="c", name="run", arguments={})
        tool_result = ToolCallResult(
            id="res",
            call=tool_call,
            name="run",
            arguments={},
            result={"stdout": "partial"},
        )
        response = FailingCleanupResponse(
            [
                Event(
                    type=EventType.TOOL_RESULT,
                    payload={"result": tool_result},
                ),
                Token(token="ignored"),
            ]
        )
        orchestrator = MagicMock()
        orchestrator.sync_messages = AsyncMock()
        logger = MagicMock()
        stream = mcp_router._stream_mcp_response(
            request_id="id",
            request_model=self._chat_request(stream=True),
            response=response,
            response_id=uuid4(),
            timestamp=1,
            progress_token="tok",
            orchestrator=orchestrator,
            logger=logger,
            resource_store=store,
            base_path="/base",
            cancel_event=cancel_event,
        )

        first = await anext(stream)
        payloads: list[dict[str, Any]] = [loads(first.decode("utf-8"))]
        cancel_event.set()
        async for chunk in stream:
            payloads.extend(
                loads(part)
                for part in chunk.decode("utf-8").splitlines()
                if part
            )

        orchestrator.sync_messages.assert_awaited()
        self.assertTrue(
            any(
                item.get("method") == "notifications/resources/updated"
                and item["params"]["resources"][0].get("closed") is True
                for item in payloads
            )
        )
        self.assertEqual(payloads[-1]["error"]["code"], -32000)
        self.assertTrue(
            any(
                logged_call.args[0] == "MCP stream source cleanup failed"
                for logged_call in logger.exception.call_args_list
            )
        )

    async def test_create_router_mcp_rpc_tools_call(self) -> None:
        router = mcp_router.create_router()
        endpoint = None
        for route in router.routes:
            if getattr(route, "path", None) == "/":
                endpoint = route.endpoint
                break
        self.assertIsNotNone(endpoint)

        request = DummyRequest(
            (
                dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": "call",
                        "method": "tools/call",
                        "params": {
                            "name": "run",
                            "arguments": {
                                "input_string": "hello",
                            },
                        },
                    }
                )
                + mcp_router.RS
            ).encode("utf-8")
        )
        request.app.state.mcp_resource_base_path = "/base"

        async def fake_start(*args, **kwargs):
            return mcp_router.PlainTextResponse("ok")

        orchestrator = DummyOrchestrator(SimpleNamespace(is_empty=False))
        with patch.object(
            mcp_router,
            "_start_tool_streaming_response",
            side_effect=fake_start,
        ) as starter:
            response = await endpoint(  # type: ignore[operator]
                request,
                logger=getLogger("test.mcp.rpc.call"),
                orchestrator=orchestrator,
            )
        self.assertIsInstance(response, mcp_router.PlainTextResponse)
        args, kwargs = starter.call_args
        self.assertEqual(args[3], "call")

    async def test_create_router_mcp_rpc_unsupported_method(self) -> None:
        router = mcp_router.create_router()
        endpoint = None
        for route in router.routes:
            if getattr(route, "path", None) == "/":
                endpoint = route.endpoint
                break
        self.assertIsNotNone(endpoint)

        request = DummyRequest(
            (
                dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": "x",
                        "method": "unknown",
                    }
                )
                + mcp_router.RS
            ).encode("utf-8")
        )
        orchestrator = DummyOrchestrator(SimpleNamespace(is_empty=True))
        with self.assertRaises(mcp_router.HTTPException):
            await endpoint(
                request,
                logger=getLogger("test.mcp.rpc"),
                orchestrator=orchestrator,
            )

    async def test_mcp_get_resource_endpoint(self) -> None:
        request = DummyRequest(b"")
        store = mcp_router.MCPResourceStore()
        request.app.state.mcp_resource_store = store
        created = await store.create(base_path="/base", initial_text="hello")
        router = mcp_router.create_router()
        endpoint = None
        for route in router.routes:
            if getattr(route, "path", None) == "/resources/{resource_id}":
                endpoint = route.endpoint
                break
        self.assertIsNotNone(endpoint)
        response = await endpoint(  # type: ignore[operator]
            request,
            resource_id=created.id,
        )
        self.assertIsInstance(response, mcp_router.PlainTextResponse)
        self.assertEqual(response.body.decode("utf-8"), "hello")
