from avalan.entities import ReasoningToken, ToolCall, ToolCallResult, Token
from avalan.event import Event, EventType
from avalan.server.entities import ResponsesRequest
from avalan.server.routers import mcp as mcp_router
from asyncio import Event as AsyncEvent
from asyncio import run
from logging import getLogger
from json import dumps, loads
from types import SimpleNamespace
from typing import Any
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4


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

    def test_extract_append_streams(self) -> None:
        streams = mcp_router._extract_append_streams(
            "call-1", {"stdout": "out", "stderr": "", "logs": "log"}
        )
        self.assertIn("call-1:stdout", streams)
        self.assertIn("call-1:logs", streams)


class DummyRequest:
    def __init__(self, body: bytes) -> None:
        self._body = body
        self.state = SimpleNamespace()
        self.app = SimpleNamespace(state=SimpleNamespace())

    async def stream(self):
        yield self._body


class DummyResponse:
    def __init__(self, items: list[Any]) -> None:
        self._items = items
        self._index = 0
        self.input_token_count = 3
        self.output_token_count = 2
        self._response_iterator = None
        self._closed = False

    def __aiter__(self):
        self._index = 0
        self._response_iterator = self
        return self

    async def __anext__(self):
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item

    async def aclose(self) -> None:
        self._closed = True


class DummyOrchestrator(mcp_router.Orchestrator):
    def __init__(self, tool: Any) -> None:
        self._tool = tool

    @property
    def tool(self) -> Any:
        return self._tool


class MCPRouterAsyncTestCase(IsolatedAsyncioTestCase):
    def _get_route(self, path: str):
        router = mcp_router.create_router()
        for route in router.routes:
            if getattr(route, "path", None) == path:
                return route.endpoint
        raise AssertionError(f"Route {path} not found")

    async def test_consume_call_request(self) -> None:
        message = {
            "jsonrpc": "2.0",
            "id": "call-1",
            "method": "tools/call",
            "params": {
                "name": "run",
                "arguments": {
                    "model": "gpt",
                    "input": [{"role": "user", "content": "Hello"}],
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
        self.assertTrue(req_model.stream)
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
        ]
        response = DummyResponse(items)
        request_model = ResponsesRequest.model_validate(
            {
                "model": "gpt",
                "input": [{"role": "user", "content": "hi"}],
            }
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
            loads(part)
            for part in "".join(chunks).split(mcp_router.RS)
            if part
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

    async def test_stream_response_handles_cancellation(self) -> None:
        response = DummyResponse([Token(token="Hi")])
        request_model = ResponsesRequest.model_validate(
            {
                "model": "gpt",
                "input": [{"role": "user", "content": "hi"}],
            }
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
            loads(part)
            for part in "".join(chunks).split(mcp_router.RS)
            if part
        ]
        dict_messages = [msg for msg in messages if isinstance(msg, dict)]
        errors = [msg["error"] for msg in dict_messages if "error" in msg]
        self.assertTrue(errors)
        self.assertEqual(errors[-1]["code"], -32000)

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
        self.assertFalse(result["capabilities"]["tools"]["call"])

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

        schema = {
            "type": "function",
            "function": {
                "name": "demo.tool",
                "description": "Demo tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        tool_manager = SimpleNamespace(
            is_empty=False,
            json_schemas=MagicMock(return_value=[schema]),
        )
        orchestrator = DummyOrchestrator(tool_manager)

        response = await endpoint(
            request,
            logger=getLogger("test.list"),
            orchestrator=orchestrator,
        )

        payload = loads(response.body.decode("utf-8"))
        result = payload["result"]
        self.assertNotIn("nextCursor", result)
        self.assertEqual(
            result["tools"],
            [
                {
                    "name": "demo.tool",
                    "description": "Demo tool",
                    "inputSchema": {"type": "object", "properties": {}},
                }
            ],
        )

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

    async def test_consume_call_request_from_message(self) -> None:
        message = {
            "jsonrpc": "2.0",
            "id": "call-2",
            "method": "tools/call",
            "params": {
                "name": "run",
                "arguments": {
                    "model": "gpt",
                    "input": [{"role": "user", "content": "Hello"}],
                },
            },
        }
        req = DummyRequest(b"")
        req.app.state.mcp_resource_base_path = "/m"

        # empty iterator of remaining messages
        async def _empty():
            if False:
                yield {}

        rid, req_model, token = mcp_router._consume_call_request_from_message(
            req, message, _empty()
        )
        self.assertEqual(rid, "call-2")
        self.assertTrue(req_model.stream)
        self.assertTrue(token)
