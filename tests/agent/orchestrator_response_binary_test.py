from base64 import b64encode
from dataclasses import dataclass
from io import StringIO
from json import loads
from logging import getLogger
from typing import cast
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from avalan.agent import AgentOperation, EngineEnvironment, Specification
from avalan.agent.engine import EngineAgent
from avalan.agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
)
from avalan.entities import (
    EngineUri,
    GenerationSettings,
    Input,
    Message,
    MessageRole,
    ToolCall,
    ToolCallContext,
    ToolCallResult,
    TransformerEngineSettings,
)
from avalan.event.manager import EventManager
from avalan.model import TextGenerationResponse
from avalan.model.call import ModelCallContext
from avalan.model.capability import (
    DomainCapabilitySeed,
    ModelCapabilityCatalog,
    ModelCapabilityDescriptor,
)
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemKind,
    StreamTerminalOutcome,
)
from avalan.tool.manager import ToolManager


class _DummyEngine:
    def __init__(self):
        self.model_id = "m"
        self.tokenizer = MagicMock()


def _dummy_operation() -> AgentOperation:
    env = EngineEnvironment(
        engine_uri=EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="m",
            params={},
        ),
        settings=TransformerEngineSettings(),
    )
    spec = Specification(role="assistant", goal=None)
    return AgentOperation(specification=spec, environment=env)


def _string_response(text: str, *, async_gen: bool = False):
    def output_fn(*args, **kwargs):
        if async_gen:

            async def gen():
                sequence = 0
                yield CanonicalStreamItem(
                    stream_session_id="binary-test-stream",
                    run_id="binary-test-run",
                    turn_id="binary-test-turn",
                    sequence=sequence,
                    kind=StreamItemKind.STREAM_STARTED,
                    channel=StreamChannel.CONTROL,
                )
                sequence += 1
                for ch in text:
                    yield CanonicalStreamItem(
                        stream_session_id="binary-test-stream",
                        run_id="binary-test-run",
                        turn_id="binary-test-turn",
                        sequence=sequence,
                        kind=StreamItemKind.ANSWER_DELTA,
                        channel=StreamChannel.ANSWER,
                        text_delta=ch,
                    )
                    sequence += 1
                if text:
                    yield CanonicalStreamItem(
                        stream_session_id="binary-test-stream",
                        run_id="binary-test-run",
                        turn_id="binary-test-turn",
                        sequence=sequence,
                        kind=StreamItemKind.ANSWER_DONE,
                        channel=StreamChannel.ANSWER,
                    )
                    sequence += 1
                yield CanonicalStreamItem(
                    stream_session_id="binary-test-stream",
                    run_id="binary-test-run",
                    turn_id="binary-test-turn",
                    sequence=sequence,
                    kind=StreamItemKind.STREAM_COMPLETED,
                    channel=StreamChannel.CONTROL,
                    usage={},
                    terminal_outcome=StreamTerminalOutcome.COMPLETED,
                )

            return gen()
        return text

    return TextGenerationResponse(
        output_fn,
        logger=getLogger(),
        use_async_generator=async_gen,
        inputs={"input_ids": [[1, 2, 3]]},
        generation_settings=GenerationSettings(),
        settings=GenerationSettings(),
    )


def _make_response(
    input_value: Input,
    response: TextGenerationResponse,
    agent: EngineAgent,
    operation: AgentOperation,
    engine_args: dict,
    **kwargs,
) -> OrchestratorResponse:
    kwargs.setdefault(
        "enable_tool_parsing", kwargs.get("capability") is not None
    )
    agent_id = kwargs.get("agent_id")
    participant_id = kwargs.get("participant_id")
    session_id = kwargs.get("session_id")
    context = ModelCallContext(
        specification=operation.specification,
        input=input_value,
        capability=kwargs.get("capability"),
        engine_args=dict(engine_args),
        agent_id=agent_id,
        participant_id=participant_id,
        session_id=session_id,
    )
    return OrchestratorResponse(
        input_value,
        response,
        agent,
        operation,
        engine_args,
        context,
        **kwargs,
    )


def _posthoc_capability(
    calls_by_text: dict[str, list[ToolCall]],
) -> ModelCapabilityCatalog:
    names = tuple(
        sorted(
            {call.name for calls in calls_by_text.values() for call in calls}
        )
    )
    catalog = ModelCapabilityCatalog.create(
        DomainCapabilitySeed(
            descriptors=tuple(
                ModelCapabilityDescriptor(
                    canonical_name=name,
                    description=f"Invoke {name}.",
                    parameter_schema={
                        "type": "object",
                        "additionalProperties": True,
                    },
                )
                for name in names
            )
        )
    )
    capability = MagicMock(spec=ModelCapabilityCatalog)
    capability.get_calls.side_effect = calls_by_text.get
    capability.project.side_effect = catalog.project
    capability.classify_batch.side_effect = catalog.classify_batch
    return cast(ModelCapabilityCatalog, capability)


class OrchestratorResponseBinaryDataTestCase(IsolatedAsyncioTestCase):
    """Test that binary data in tool results is properly JSON serialized."""

    async def test_to_str_with_binary_result_in_tool_call(self):
        """Binary data in tool results should be base64 encoded."""
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()

        outer_response = _string_response("query", async_gen=True)

        binary_data = b"\x89PDF\x0d\x0a\x1a\x0a\x00\x00\x00\x0d"
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        capability = _posthoc_capability(
            {
                "query": [
                    ToolCall(
                        id=uuid4(),
                        name="database.sample",
                        arguments=None,
                    )
                ]
            }
        )

        async def tool_exec(call, context: ToolCallContext):
            return ToolCallResult(
                id=uuid4(),
                call=call,
                name=call.name,
                arguments=call.arguments,
                result=[{"id": 1, "content": binary_data}],
            )

        tool.side_effect = tool_exec

        inner_response = _string_response("ok", async_gen=True)
        agent.return_value = inner_response

        TextGenerationResponse._buffer = StringIO()

        resp = _make_response(
            Message(role=MessageRole.USER, content="get data"),
            outer_response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            capability=capability,
            enable_tool_parsing=False,
        )

        result = await resp.to_str()
        self.assertEqual(result, "ok")

        agent.assert_awaited_once()
        tool.assert_awaited_once()

        call_args = agent.await_args
        context = call_args[0][0]
        messages = context.input
        tool_message = next(
            (m for m in messages if m.role == MessageRole.TOOL), None
        )
        self.assertIsNotNone(tool_message)

        parsed_content = loads(tool_message.content)
        self.assertIsInstance(parsed_content, list)
        self.assertEqual(len(parsed_content), 1)
        self.assertEqual(parsed_content[0]["id"], 1)
        self.assertIsInstance(parsed_content[0]["content"], str)
        self.assertEqual(
            parsed_content[0]["content"], b64encode(binary_data).decode()
        )

    async def test_to_str_with_bytearray_result_in_tool_call(self):
        """Bytearray data in tool results should be base64 encoded."""
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()

        outer_response = _string_response("fetch", async_gen=True)

        bytearray_data = bytearray(b"test bytearray content")
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        capability = _posthoc_capability(
            {"fetch": [ToolCall(id=uuid4(), name="read_file", arguments=None)]}
        )

        async def tool_exec(call, context: ToolCallContext):
            return ToolCallResult(
                id=uuid4(),
                call=call,
                name=call.name,
                arguments=call.arguments,
                result={"data": bytearray_data},
            )

        tool.side_effect = tool_exec

        inner_response = _string_response("done", async_gen=True)
        agent.return_value = inner_response

        TextGenerationResponse._buffer = StringIO()

        resp = _make_response(
            Message(role=MessageRole.USER, content="read file"),
            outer_response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            capability=capability,
            enable_tool_parsing=False,
        )

        result = await resp.to_str()
        self.assertEqual(result, "done")

        call_args = agent.await_args
        context = call_args[0][0]
        messages = context.input
        tool_message = next(
            (m for m in messages if m.role == MessageRole.TOOL), None
        )
        self.assertIsNotNone(tool_message)

        parsed_content = loads(tool_message.content)
        self.assertIsInstance(parsed_content, dict)
        self.assertIsInstance(parsed_content["data"], str)
        self.assertEqual(
            parsed_content["data"], b64encode(bytearray_data).decode()
        )

    async def test_to_str_with_nested_binary_data(self):
        """Nested binary data in complex structures should be encoded."""
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()

        outer_response = _string_response("get", async_gen=True)

        binary1 = b"\x00\x01\x02\x03"
        binary2 = b"\xff\xfe\xfd\xfc"
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        capability = _posthoc_capability(
            {
                "get": [
                    ToolCall(id=uuid4(), name="complex_query", arguments=None)
                ]
            }
        )

        async def tool_exec(call, context: ToolCallContext):
            return ToolCallResult(
                id=uuid4(),
                call=call,
                name=call.name,
                arguments=call.arguments,
                result={
                    "records": [
                        {"id": 1, "data": binary1},
                        {"id": 2, "data": binary2},
                    ],
                    "metadata": {"signature": b"\xaa\xbb\xcc\xdd"},
                },
            )

        tool.side_effect = tool_exec

        inner_response = _string_response("ok", async_gen=True)
        agent.return_value = inner_response

        TextGenerationResponse._buffer = StringIO()

        resp = _make_response(
            Message(role=MessageRole.USER, content="query"),
            outer_response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            capability=capability,
            enable_tool_parsing=False,
        )

        result = await resp.to_str()
        self.assertEqual(result, "ok")

        call_args = agent.await_args
        context = call_args[0][0]
        messages = context.input
        tool_message = next(
            (m for m in messages if m.role == MessageRole.TOOL), None
        )
        self.assertIsNotNone(tool_message)

        parsed_content = loads(tool_message.content)
        self.assertIsInstance(parsed_content, dict)
        self.assertIn("records", parsed_content)
        self.assertEqual(len(parsed_content["records"]), 2)
        self.assertEqual(parsed_content["records"][0]["id"], 1)
        self.assertEqual(
            parsed_content["records"][0]["data"], b64encode(binary1).decode()
        )
        self.assertEqual(parsed_content["records"][1]["id"], 2)
        self.assertEqual(
            parsed_content["records"][1]["data"], b64encode(binary2).decode()
        )
        self.assertEqual(
            parsed_content["metadata"]["signature"],
            b64encode(b"\xaa\xbb\xcc\xdd").decode(),
        )

    async def test_to_str_with_empty_binary_data(self):
        """Empty binary data should be encoded as empty base64 string."""
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()

        outer_response = _string_response("x", async_gen=True)

        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        capability = _posthoc_capability(
            {"x": [ToolCall(id=uuid4(), name="test", arguments=None)]}
        )

        async def tool_exec(call, context: ToolCallContext):
            return ToolCallResult(
                id=uuid4(),
                call=call,
                name=call.name,
                arguments=call.arguments,
                result={"empty": b""},
            )

        tool.side_effect = tool_exec

        inner_response = _string_response("y", async_gen=True)
        agent.return_value = inner_response

        TextGenerationResponse._buffer = StringIO()

        resp = _make_response(
            Message(role=MessageRole.USER, content="test"),
            outer_response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            capability=capability,
            enable_tool_parsing=False,
        )

        result = await resp.to_str()
        self.assertEqual(result, "y")

        call_args = agent.await_args
        context = call_args[0][0]
        messages = context.input
        tool_message = next(
            (m for m in messages if m.role == MessageRole.TOOL), None
        )
        self.assertIsNotNone(tool_message)

        parsed_content = loads(tool_message.content)
        self.assertIsInstance(parsed_content, dict)
        self.assertIsInstance(parsed_content["empty"], str)
        self.assertEqual(parsed_content["empty"], b64encode(b"").decode())

    async def test_to_str_with_mixed_types_including_binary(self):
        """Tool results with mixed types including binary are encoded."""
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()

        outer_response = _string_response("call", async_gen=True)

        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        capability = _posthoc_capability(
            {"call": [ToolCall(id=uuid4(), name="mixed", arguments=None)]}
        )

        async def tool_exec(call, context: ToolCallContext):
            return ToolCallResult(
                id=uuid4(),
                call=call,
                name=call.name,
                arguments=call.arguments,
                result={
                    "string": "text",
                    "number": 42,
                    "float": 3.14,
                    "boolean": True,
                    "null": None,
                    "binary": b"\x01\x02\x03",
                    "list": [1, "two", b"\x04\x05"],
                },
            )

        tool.side_effect = tool_exec

        inner_response = _string_response("done", async_gen=True)
        agent.return_value = inner_response

        TextGenerationResponse._buffer = StringIO()

        resp = _make_response(
            Message(role=MessageRole.USER, content="test"),
            outer_response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            capability=capability,
            enable_tool_parsing=False,
        )

        result = await resp.to_str()
        self.assertEqual(result, "done")

        call_args = agent.await_args
        context = call_args[0][0]
        messages = context.input
        tool_message = next(
            (m for m in messages if m.role == MessageRole.TOOL), None
        )
        self.assertIsNotNone(tool_message)

        parsed_content = loads(tool_message.content)
        self.assertEqual(parsed_content["string"], "text")
        self.assertEqual(parsed_content["number"], 42)
        self.assertEqual(parsed_content["float"], 3.14)
        self.assertEqual(parsed_content["boolean"], True)
        self.assertIsNone(parsed_content["null"])
        self.assertEqual(
            parsed_content["binary"], b64encode(b"\x01\x02\x03").decode()
        )
        self.assertEqual(parsed_content["list"][0], 1)
        self.assertEqual(parsed_content["list"][1], "two")
        self.assertEqual(
            parsed_content["list"][2], b64encode(b"\x04\x05").decode()
        )

    async def test_to_str_with_memoryview_result_in_tool_call(self):
        """Memoryview data in tool results should be base64 encoded."""
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()

        outer_response = _string_response("query", async_gen=True)

        memview_data = memoryview(b"memoryview content")
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        capability = _posthoc_capability(
            {
                "query": [
                    ToolCall(
                        id=uuid4(),
                        name="database.fetch",
                        arguments=None,
                    )
                ]
            }
        )

        async def tool_exec(call, context: ToolCallContext):
            return ToolCallResult(
                id=uuid4(),
                call=call,
                name=call.name,
                arguments=call.arguments,
                result={"id": 1, "buffer": memview_data},
            )

        tool.side_effect = tool_exec

        inner_response = _string_response("ok", async_gen=True)
        agent.return_value = inner_response

        TextGenerationResponse._buffer = StringIO()

        resp = _make_response(
            Message(role=MessageRole.USER, content="fetch data"),
            outer_response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            capability=capability,
            enable_tool_parsing=False,
        )

        result = await resp.to_str()
        self.assertEqual(result, "ok")

        call_args = agent.await_args
        context = call_args[0][0]
        messages = context.input
        tool_message = next(
            (m for m in messages if m.role == MessageRole.TOOL), None
        )
        self.assertIsNotNone(tool_message)

        parsed_content = loads(tool_message.content)
        self.assertIsInstance(parsed_content, dict)
        self.assertEqual(parsed_content["id"], 1)
        self.assertIsInstance(parsed_content["buffer"], str)
        self.assertEqual(
            parsed_content["buffer"], b64encode(memview_data).decode()
        )

    async def test_to_str_with_nested_memoryview_data(self):
        """Nested memoryview data in complex structures should be encoded."""
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()

        outer_response = _string_response("fetch", async_gen=True)

        memview1 = memoryview(b"\x11\x22\x33\x44")
        memview2 = memoryview(b"\xaa\xbb\xcc\xdd")
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        capability = _posthoc_capability(
            {
                "fetch": [
                    ToolCall(id=uuid4(), name="complex_fetch", arguments=None)
                ]
            }
        )

        async def tool_exec(call, context: ToolCallContext):
            return ToolCallResult(
                id=uuid4(),
                call=call,
                name=call.name,
                arguments=call.arguments,
                result={
                    "records": [
                        {"id": 1, "buffer": memview1},
                        {"id": 2, "buffer": memview2},
                    ]
                },
            )

        tool.side_effect = tool_exec

        inner_response = _string_response("done", async_gen=True)
        agent.return_value = inner_response

        TextGenerationResponse._buffer = StringIO()

        resp = _make_response(
            Message(role=MessageRole.USER, content="query"),
            outer_response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            capability=capability,
            enable_tool_parsing=False,
        )

        result = await resp.to_str()
        self.assertEqual(result, "done")

        call_args = agent.await_args
        context = call_args[0][0]
        messages = context.input
        tool_message = next(
            (m for m in messages if m.role == MessageRole.TOOL), None
        )
        self.assertIsNotNone(tool_message)

        parsed_content = loads(tool_message.content)
        self.assertIsInstance(parsed_content, dict)
        self.assertIn("records", parsed_content)
        self.assertEqual(len(parsed_content["records"]), 2)
        self.assertEqual(parsed_content["records"][0]["id"], 1)
        self.assertEqual(
            parsed_content["records"][0]["buffer"],
            b64encode(memview1).decode(),
        )
        self.assertEqual(parsed_content["records"][1]["id"], 2)
        self.assertEqual(
            parsed_content["records"][1]["buffer"],
            b64encode(memview2).decode(),
        )


@dataclass
class BinaryDataClass:
    """Test dataclass with binary field."""

    id: int
    data: bytes


class OrchestratorResponseBinaryDataclassTestCase(IsolatedAsyncioTestCase):
    """Test that binary data in dataclass results is properly serialized."""

    async def test_to_str_with_dataclass_containing_binary(self):
        """Dataclass with binary fields should be properly serialized."""
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()

        outer_response = _string_response("dc", async_gen=True)

        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        capability = _posthoc_capability(
            {
                "dc": [
                    ToolCall(id=uuid4(), name="dataclass_tool", arguments=None)
                ]
            }
        )

        async def tool_exec(call, context: ToolCallContext):
            return ToolCallResult(
                id=uuid4(),
                call=call,
                name=call.name,
                arguments=call.arguments,
                result=BinaryDataClass(id=1, data=b"\xde\xad\xbe\xef"),
            )

        tool.side_effect = tool_exec

        inner_response = _string_response("ok", async_gen=True)
        agent.return_value = inner_response

        TextGenerationResponse._buffer = StringIO()

        resp = _make_response(
            Message(role=MessageRole.USER, content="get dataclass"),
            outer_response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            capability=capability,
            enable_tool_parsing=False,
        )

        result = await resp.to_str()
        self.assertEqual(result, "ok")

        call_args = agent.await_args
        context = call_args[0][0]
        messages = context.input
        tool_message = next(
            (m for m in messages if m.role == MessageRole.TOOL), None
        )
        self.assertIsNotNone(tool_message)

        parsed_content = loads(tool_message.content)
        self.assertIsInstance(parsed_content, dict)
        self.assertEqual(parsed_content["id"], 1)
        self.assertEqual(
            parsed_content["data"], b64encode(b"\xde\xad\xbe\xef").decode()
        )
