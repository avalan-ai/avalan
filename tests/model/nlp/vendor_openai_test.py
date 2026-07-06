import importlib
import sys
import types
from asyncio import CancelledError, Event, create_task, wait_for
from collections.abc import AsyncIterable
from dataclasses import dataclass
from importlib.machinery import ModuleSpec
from json import loads
from logging import getLogger
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import ANY, AsyncMock, MagicMock, patch
from uuid import UUID

from avalan.agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
)
from avalan.entities import (
    GenerationSettings,
    Message,
    MessageContentFile,
    MessageContentImage,
    MessageContentText,
    MessageRole,
    MessageToolCall,
    PromptCacheRetention,
    ReasoningEffort,
    ReasoningSettings,
    ToolCall,
    ToolCallDiagnostic,
    ToolCallDiagnosticCode,
    ToolCallDiagnosticStage,
    ToolCallError,
    ToolCallResult,
    ToolManagerSettings,
    ToolNamePolicyMode,
    ToolNamePolicySettings,
    TransformerEngineSettings,
)
from avalan.model.response.text import TextGenerationResponse
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemKind,
    StreamTerminalOutcome,
    accumulate_canonical_stream_items,
)
from avalan.task.usage import (
    usage_observation_from_response,
    usage_totals_from_response,
)
from avalan.tool import ToolSet
from avalan.tool.manager import ToolManager

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "tool_parsing"


class AsyncIter:
    def __init__(self, items):
        self._iter = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


async def _stream_items(
    stream: AsyncIterable[CanonicalStreamItem],
) -> list[CanonicalStreamItem]:
    return [item async for item in stream]


class TrackedAsyncIter:
    def __init__(self, items):
        self._iter = iter(items)
        self.read_count = 0
        self.close_count = 0
        self.cancel_count = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.read_count += 1
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def aclose(self) -> None:
        self.close_count += 1

    async def cancel(self) -> None:
        self.cancel_count += 1


class CloseOnlyAsyncIter:
    def __init__(self, items):
        self._iter = iter(items)
        self.read_count = 0
        self.close_count = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.read_count += 1
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def close(self) -> None:
        self.close_count += 1


class SyncCloseOnlyAsyncIter(CloseOnlyAsyncIter):
    def close(self) -> None:
        self.close_count += 1


class FailingCloseOnlyAsyncIter(CloseOnlyAsyncIter):
    def __init__(self, items, error: BaseException):
        super().__init__(items)
        self._error = error

    async def close(self) -> None:
        raise self._error


class AcloseOnlyAsyncIter:
    def __init__(self, items):
        self._iter = iter(items)
        self.read_count = 0
        self.close_count = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.read_count += 1
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def aclose(self) -> None:
        self.close_count += 1


class PendingAsyncIter:
    def __init__(self):
        self.started = Event()
        self.pull_cancelled = False
        self.close_count = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.started.set()
        try:
            await Event().wait()
        except CancelledError:
            self.pull_cancelled = True
            raise
        return SimpleNamespace(type="response.output_text.delta", delta="late")

    async def aclose(self) -> None:
        self.close_count += 1


class PolicyAdder:
    def __init__(self) -> None:
        self.__name__ = "adder"

    async def __call__(self, a: int, b: int) -> int:
        """Return the sum of two integers."""
        return a + b


def _sanitized_policy_manager() -> ToolManager:
    return ToolManager.create_instance(
        enable_tools=["math.adder"],
        available_toolsets=[ToolSet(namespace="math", tools=[PolicyAdder()])],
        settings=ToolManagerSettings(
            tool_name_policy=ToolNamePolicySettings(
                mode=ToolNamePolicyMode.SANITIZED
            )
        ),
    )


def patch_openai_imports():
    class Omit:
        def __bool__(self):
            return False

    openai_stub = types.ModuleType("openai")
    openai_stub.__spec__ = ModuleSpec("openai", loader=None)
    openai_stub.AsyncOpenAI = MagicMock()
    openai_stub.AsyncStream = MagicMock()
    openai_stub.Omit = Omit
    openai_stub.AsyncOpenAI.return_value.responses = MagicMock()

    transformers_stub = types.ModuleType("transformers")
    transformers_stub.__spec__ = ModuleSpec("transformers", loader=None)
    transformers_stub.PreTrainedModel = MagicMock()
    transformers_stub.PreTrainedTokenizer = MagicMock()
    transformers_stub.PreTrainedTokenizerFast = MagicMock()
    transformers_stub.__getattr__ = lambda name: MagicMock()

    transformers_utils_stub = types.ModuleType("transformers.utils")
    transformers_utils_stub.get_json_schema = MagicMock()
    transformers_logging_stub = types.ModuleType("transformers.utils.logging")
    transformers_logging_stub.disable_progress_bar = MagicMock()
    transformers_logging_stub.enable_progress_bar = MagicMock()
    transformers_utils_stub.logging = transformers_logging_stub
    transformers_tokenization_stub = types.ModuleType(
        "transformers.tokenization_utils_base"
    )
    transformers_tokenization_stub.BatchEncoding = MagicMock()
    transformers_stub.tokenization_utils_base = transformers_tokenization_stub
    transformers_generation_stub = types.ModuleType("transformers.generation")
    transformers_generation_stub.__spec__ = ModuleSpec(
        "transformers.generation", loader=None, is_package=True
    )
    transformers_generation_stub.__path__ = []
    transformers_generation_stub.StoppingCriteria = MagicMock()
    transformers_generation_stub.StoppingCriteriaList = MagicMock()
    transformers_stopping_criteria_stub = types.ModuleType(
        "transformers.generation.stopping_criteria"
    )
    transformers_stopping_criteria_stub.__spec__ = ModuleSpec(
        "transformers.generation.stopping_criteria", loader=None
    )
    transformers_stopping_criteria_stub.StoppingCriteria = MagicMock()
    transformers_stopping_criteria_stub.StoppingCriteriaList = MagicMock()
    transformers_generation_stub.stopping_criteria = (
        transformers_stopping_criteria_stub
    )
    transformers_stub.generation = transformers_generation_stub
    transformers_stub.utils = transformers_utils_stub

    diffusers_stub = types.ModuleType("diffusers")
    diffusers_stub.__spec__ = ModuleSpec("diffusers", loader=None)
    diffusers_stub.DiffusionPipeline = MagicMock()

    patcher = patch.dict(
        sys.modules,
        {
            "openai": openai_stub,
            "transformers": transformers_stub,
            "transformers.utils": transformers_utils_stub,
            "transformers.utils.logging": transformers_logging_stub,
            "transformers.tokenization_utils_base": (
                transformers_tokenization_stub
            ),
            "transformers.generation": transformers_generation_stub,
            "transformers.generation.stopping_criteria": (
                transformers_stopping_criteria_stub
            ),
            "diffusers": diffusers_stub,
        },
    )
    patcher.start()
    return openai_stub, patcher


class OpenAITestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.openai_stub, self.patch = patch_openai_imports()
        importlib.reload(
            importlib.import_module("avalan.model.nlp.text.vendor.openai")
        )
        self.mod = importlib.import_module(
            "avalan.model.nlp.text.vendor.openai"
        )

    def tearDown(self):
        self.patch.stop()

    async def test_stream_client_and_model(self):
        chunks = [
            SimpleNamespace(type="response.output_text.delta", delta="x"),
            SimpleNamespace(type="response.output_text.delta", delta="y"),
        ]
        stream = self.mod.OpenAIStream(AsyncIter(chunks))
        items = await _stream_items(stream)
        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(
            [
                item.text_delta
                for item in items
                if item.kind is StreamItemKind.ANSWER_DELTA
            ],
            ["x", "y"],
        )
        self.assertEqual(
            accumulate_canonical_stream_items(items).answer_text,
            "xy",
        )

        stream_instance = AsyncIter([])
        self.openai_stub.AsyncOpenAI.return_value.responses.create = AsyncMock(
            return_value=stream_instance
        )
        client = self.mod.OpenAIClient(api_key="k", base_url="b")
        client._template_messages = MagicMock(return_value=[{"c": 1}])
        with patch.object(self.mod, "OpenAIStream") as StreamMock:
            result = await client("m", [])
        self.openai_stub.AsyncOpenAI.assert_called_once_with(
            base_url="b", api_key="k"
        )
        client._client.responses.create.assert_awaited_once_with(
            extra_headers={
                "X-Title": "Avalan",
                "HTTP-Referer": "https://github.com/avalan-ai/avalan",
            },
            model="m",
            input=[{"c": 1}],
            store=False,
            stream=True,
        )
        StreamMock.assert_called_once_with(
            stream=stream_instance,
            provider_family="openai",
            output_item_sink=client._record_stateless_response_item,
            output_item_rollback=client._rollback_stateless_response_items,
            stream_factory=ANY,
            stream_retry_delay_seconds=(
                client._STREAM_RESPONSE_FAILED_RETRY_DELAY_SECONDS
            ),
            stream_retries=client._STREAM_RESPONSE_FAILED_RETRIES,
        )
        self.assertIs(result, StreamMock.return_value)

    async def test_stream_public_iterator_yields_canonical_items(self):
        stream = self.mod.OpenAIStream(
            AsyncIter(
                [
                    SimpleNamespace(
                        type="response.output_text.delta", delta="hi"
                    )
                ]
            )
        )

        items = [item async for item in stream]

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(items[1].text_delta, "hi")
        self.assertEqual({item.provider_family for item in items}, {"openai"})

    async def test_stream_ignores_text_events_after_answer_done(self):
        late_events = [
            SimpleNamespace(type="response.output_text.delta", delta="late"),
            SimpleNamespace(type="response.output_text.done", text="late"),
        ]

        for late_event in late_events:
            with self.subTest(late_event=late_event.type):
                stream = self.mod.OpenAIStream(
                    AsyncIter(
                        [
                            SimpleNamespace(
                                type="response.output_text.delta",
                                delta="done",
                            ),
                            SimpleNamespace(type="response.output_text.done"),
                            late_event,
                        ]
                    )
                )

                items = await _stream_items(stream)

                self.assertEqual(
                    [
                        item.kind
                        for item in items
                        if item.kind
                        in {
                            StreamItemKind.ANSWER_DELTA,
                            StreamItemKind.ANSWER_DONE,
                        }
                    ],
                    [
                        StreamItemKind.ANSWER_DELTA,
                        StreamItemKind.ANSWER_DONE,
                    ],
                )
                self.assertEqual(
                    accumulate_canonical_stream_items(items).answer_text,
                    "done",
                )

    async def test_stream_deduplicates_text_delta_alias_events(self):
        stream = self.mod.OpenAIStream(
            AsyncIter(
                [
                    SimpleNamespace(
                        type="response.output_text.delta",
                        delta="{",
                        item_id="msg_1",
                        output_index=0,
                        content_index=0,
                        sequence_number=1,
                    ),
                    SimpleNamespace(
                        type="response.text.delta",
                        delta="{",
                        item_id="msg_1",
                        output_index=0,
                        content_index=0,
                        sequence_number=1,
                    ),
                    SimpleNamespace(
                        type="response.output_text.delta",
                        delta="x",
                        item_id="msg_1",
                        output_index=0,
                        content_index=0,
                        sequence_number=2,
                    ),
                    SimpleNamespace(
                        type="response.text.delta",
                        delta="x",
                        item_id="msg_1",
                        output_index=0,
                        content_index=0,
                        sequence_number=2,
                    ),
                ]
            )
        )

        items = await _stream_items(stream)

        self.assertEqual(
            accumulate_canonical_stream_items(items).answer_text,
            "{x",
        )

    async def test_stream_preserves_repeated_text_delta_tokens(self):
        stream = self.mod.OpenAIStream(
            AsyncIter(
                [
                    SimpleNamespace(
                        type="response.output_text.delta",
                        delta="0",
                        item_id="msg_1",
                        output_index=0,
                        content_index=0,
                        sequence_number=1,
                    ),
                    SimpleNamespace(
                        type="response.output_text.delta",
                        delta="0",
                        item_id="msg_1",
                        output_index=0,
                        content_index=0,
                        sequence_number=2,
                    ),
                ]
            )
        )

        items = await _stream_items(stream)

        self.assertEqual(
            accumulate_canonical_stream_items(items).answer_text,
            "00",
        )

    async def test_stream_direct_anext_yields_canonical_items(self):
        stream = self.mod.OpenAIStream(
            AsyncIter(
                [
                    SimpleNamespace(
                        type="response.output_text.delta", delta="hi"
                    )
                ]
            )
        )

        started = await stream.__anext__()
        delta = await stream.__anext__()

        self.assertIs(started.kind, StreamItemKind.STREAM_STARTED)
        self.assertIs(delta.kind, StreamItemKind.ANSWER_DELTA)
        self.assertEqual(delta.text_delta, "hi")
        self.assertEqual(delta.provider_family, "openai")

    async def test_client_sends_top_level_instructions(self):
        response = SimpleNamespace(
            output=[SimpleNamespace(content=[SimpleNamespace(text="x")])]
        )
        self.openai_stub.AsyncOpenAI.return_value.responses.create = AsyncMock(
            return_value=response
        )
        client = self.mod.OpenAIClient(api_key="k", base_url="b")
        message = Message(role=MessageRole.USER, content="hi")

        await client(
            "m",
            [message],
            instructions="top-level",
            use_async_generator=False,
        )

        kwargs = client._client.responses.create.await_args.kwargs
        self.assertEqual(kwargs["instructions"], "top-level")
        self.assertEqual(kwargs["input"], [{"role": "user", "content": "hi"}])
        self.assertNotIn("top-level", str(kwargs["input"]))

    async def test_responses_payload_combines_instructions_cache_and_options(
        self,
    ):
        response = SimpleNamespace(
            output=[SimpleNamespace(content=[SimpleNamespace(text="ok")])]
        )
        create_mock = AsyncMock(return_value=response)
        self.openai_stub.AsyncOpenAI.return_value.responses.create = (
            create_mock
        )
        client = self.mod.OpenAIClient(api_key="k", base_url="b")
        tool = MagicMock()
        tool.json_schemas.return_value = [
            {"type": "function", "function": {"name": "pkg.lookup"}}
        ]
        messages = [
            Message(role=MessageRole.SYSTEM, content="system path"),
            Message(role=MessageRole.DEVELOPER, content="developer path"),
            Message(
                role=MessageRole.USER,
                content=[
                    MessageContentText(type="text", text="Summarize"),
                    MessageContentFile(
                        type="file",
                        file={
                            "file_data": "YWJj",
                            "filename": "report.pdf",
                        },
                    ),
                    MessageContentImage(
                        type="image_url",
                        image_url={
                            "data": "aW1n",
                            "mime_type": "image/png",
                            "detail": "high",
                        },
                    ),
                ],
            ),
        ]
        settings = GenerationSettings(
            max_new_tokens=10,
            prompt_cache_retention=PromptCacheRetention.EXTENDED_24H,
            reasoning=ReasoningSettings(effort=ReasoningEffort.HIGH),
            response_format={"type": "json_object"},
            stop_strings=["END"],
            tool_choice="pkg.lookup",
        )

        await client(
            "gpt-5",
            messages,
            settings=settings,
            instructions="top-level policy",
            tool=tool,
            use_async_generator=False,
        )

        create_mock.assert_awaited_once_with(
            extra_headers={
                "X-Title": "Avalan",
                "HTTP-Referer": "https://github.com/avalan-ai/avalan",
            },
            model="gpt-5",
            input=[
                {"role": "system", "content": "system path"},
                {"role": "developer", "content": "developer path"},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Summarize"},
                        {
                            "type": "input_file",
                            "file_data": "data:application/pdf;base64,YWJj",
                            "filename": "report.pdf",
                        },
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,aW1n",
                            "detail": "high",
                        },
                    ],
                },
            ],
            store=False,
            stream=False,
            instructions="top-level policy",
            max_output_tokens=10,
            text={"format": {"type": "json_object"}, "stop": ["END"]},
            reasoning={"effort": "high"},
            include=["reasoning.encrypted_content"],
            prompt_cache_retention="24h",
            tools=[{"type": "function", "name": "avl_cGtnLmxvb2t1cA"}],
            tool_choice={"type": "function", "name": "avl_cGtnLmxvb2t1cA"},
        )
        self.assertNotIn(
            "top-level policy", str(create_mock.await_args.kwargs["input"])
        )

    async def test_responses_payload_preserves_tool_history(self):
        response = SimpleNamespace(
            output=[SimpleNamespace(content=[SimpleNamespace(text="ok")])]
        )
        create_mock = AsyncMock(return_value=response)
        self.openai_stub.AsyncOpenAI.return_value.responses.create = (
            create_mock
        )
        client = self.mod.OpenAIClient(api_key="k", base_url="b")
        call = ToolCall(
            id="call1",
            name="shell.rg",
            arguments={"pattern": "needle"},
        )
        result = ToolCallResult(
            id="result1",
            name="shell.rg",
            arguments=call.arguments,
            call=call,
            result="match",
        )
        messages = [
            Message(role=MessageRole.USER, content="search"),
            Message(role=MessageRole.TOOL, tool_call_result=result),
        ]

        await client("gpt-5", messages, use_async_generator=False)

        kwargs = create_mock.await_args.kwargs
        self.assertNotIn("truncation", kwargs)
        self.assertEqual(
            kwargs["input"][0],
            {"role": "user", "content": "search"},
        )
        self.assertEqual(kwargs["input"][1]["type"], "function_call")
        self.assertEqual(kwargs["input"][2]["type"], "function_call_output")

    async def test_rejects_tool_choice_missing_from_schemas(self):
        create_mock = AsyncMock()
        self.openai_stub.AsyncOpenAI.return_value.responses.create = (
            create_mock
        )
        client = self.mod.OpenAIClient(api_key="k", base_url="b")
        tool = MagicMock()
        tool.json_schemas.return_value = [
            {"type": "function", "function": {"name": "pkg.lookup"}}
        ]

        with self.assertRaisesRegex(AssertionError, "tool_choice"):
            await client(
                "gpt-5",
                [Message(role=MessageRole.USER, content="hi")],
                settings=GenerationSettings(tool_choice="pkg.other"),
                tool=tool,
                use_async_generator=False,
            )

        create_mock.assert_not_awaited()

    async def test_rejects_invalid_instructions_before_provider_call(self):
        create_mock = AsyncMock()
        self.openai_stub.AsyncOpenAI.return_value.responses.create = (
            create_mock
        )
        client = self.mod.OpenAIClient(api_key="k", base_url="b")

        with self.assertRaisesRegex(AssertionError, "instructions"):
            await client(
                "m",
                [],
                instructions={"raw": "prompt"},  # type: ignore[arg-type]
            )

        create_mock.assert_not_awaited()

    async def test_client_consumes_tokens(self):
        chunks = [
            SimpleNamespace(type="response.output_text.delta", delta="a"),
            SimpleNamespace(type="response.output_text.delta", delta="b"),
        ]
        stream_instance = AsyncIter(chunks)
        self.openai_stub.AsyncOpenAI.return_value.responses.create = AsyncMock(
            return_value=stream_instance
        )
        client = self.mod.OpenAIClient(api_key="k", base_url="b")
        client._template_messages = MagicMock(return_value=[{"c": 1}])
        result = await client("m", [])
        items = await _stream_items(result)
        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(
            accumulate_canonical_stream_items(items).answer_text,
            "ab",
        )

        with patch.object(self.mod, "OpenAIClient") as ClientMock:
            settings = TransformerEngineSettings(
                auto_load_model=False,
                auto_load_tokenizer=False,
                access_token="t",
                base_url="u",
            )
            model = self.mod.OpenAIModel("m", settings)
            loaded = model._load_model()
        ClientMock.assert_called_once_with(base_url="u", api_key="t")
        self.assertIs(loaded, ClientMock.return_value)

    async def test_stream_records_terminal_usage_after_completion(self):
        usage = SimpleNamespace(
            input_tokens=3,
            input_tokens_details=SimpleNamespace(cached_tokens=1),
            cache_creation_input_tokens=2,
            output_tokens=4,
            output_tokens_details=SimpleNamespace(reasoning_tokens=2),
            total_tokens=9,
        )
        chunks = [
            SimpleNamespace(type="response.output_text.delta", delta="a"),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(usage=usage),
            ),
        ]
        stream = self.mod.OpenAIStream(AsyncIter(chunks))

        iterator = stream.__aiter__()
        started = await anext(iterator)
        delta = await anext(iterator)
        self.assertIs(started.kind, StreamItemKind.STREAM_STARTED)
        self.assertIs(delta.kind, StreamItemKind.ANSWER_DELTA)
        self.assertEqual(delta.text_delta, "a")
        self.assertIsNone(stream.usage)
        items = [started, delta, *[item async for item in iterator]]

        self.assertIs(stream.usage, usage)
        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.USAGE_COMPLETED,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(stream.provider_family, "openai")
        observation = usage_observation_from_response(stream)
        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(observation.metadata, {"provider_family": "openai"})
        totals = usage_totals_from_response(stream)
        self.assertIsNotNone(totals)
        assert totals is not None
        self.assertEqual(totals.input_tokens, 3)
        self.assertEqual(totals.cached_input_tokens, 1)
        self.assertEqual(totals.cache_creation_input_tokens, 2)
        self.assertEqual(totals.output_tokens, 4)
        self.assertEqual(totals.reasoning_tokens, 2)
        self.assertEqual(totals.total_tokens, 9)

    async def test_stream_records_dict_terminal_usage_after_exhaustion(self):
        stream = self.mod.OpenAIStream(
            AsyncIter(
                [
                    {
                        "type": "response.completed",
                        "response": {
                            "usage": {
                                "input_tokens": 0,
                                "prompt_tokens_details": {"cached_tokens": 0},
                                "completion_tokens": 0,
                                "completion_tokens_details": {
                                    "reasoning_tokens": 0
                                },
                                "total_tokens": 0,
                            }
                        },
                    }
                ]
            )
        )

        items = await _stream_items(stream)
        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.USAGE_COMPLETED,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        observation = usage_observation_from_response(stream)
        totals = usage_totals_from_response(stream)

        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(observation.metadata, {"provider_family": "openai"})
        self.assertIsNotNone(totals)
        assert totals is not None
        self.assertEqual(totals.input_tokens, 0)
        self.assertEqual(totals.cached_input_tokens, 0)
        self.assertIsNone(totals.cache_creation_input_tokens)
        self.assertEqual(totals.output_tokens, 0)
        self.assertEqual(totals.reasoning_tokens, 0)
        self.assertEqual(totals.total_tokens, 0)

    async def test_stream_keeps_null_or_interrupted_usage_unavailable(self):
        null_usage = self.mod.OpenAIStream(
            AsyncIter(
                [
                    SimpleNamespace(
                        type="response.completed",
                        response=SimpleNamespace(usage=None),
                    )
                ]
            )
        )
        null_usage_items = await _stream_items(null_usage)
        self.assertEqual(
            [item.kind for item in null_usage_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertIsNone(null_usage.usage)

        class FailingIter:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise RuntimeError("provider failure")

        interrupted = self.mod.OpenAIStream(FailingIter())
        interrupted_items = await _stream_items(interrupted)
        self.assertEqual(
            [item.kind for item in interrupted_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.STREAM_ERRORED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        error_data = interrupted_items[1].data
        assert isinstance(error_data, dict)
        self.assertEqual(error_data["error_type"], "RuntimeError")
        self.assertIsNone(interrupted.usage)

        class FailingAfterUsageIter:
            def __init__(self):
                self._count = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                self._count += 1
                if self._count == 1:
                    return {
                        "type": "response.completed",
                        "response": {"usage": {"input_tokens": 1}},
                    }
                raise RuntimeError("provider failure")

        interrupted_after_usage = self.mod.OpenAIStream(
            FailingAfterUsageIter()
        )
        interrupted_after_usage_items = await _stream_items(
            interrupted_after_usage
        )
        self.assertEqual(
            [item.kind for item in interrupted_after_usage_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.USAGE_COMPLETED,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(interrupted_after_usage.usage, {"input_tokens": 1})

    async def test_client_omits_auth_header_without_api_key(self):
        client = self.mod.OpenAIClient(
            api_key=None, base_url="http://localhost:9001/v1"
        )

        self.assertIsInstance(client, self.mod.OpenAIClient)
        self.openai_stub.AsyncOpenAI.assert_called_once()
        kwargs = self.openai_stub.AsyncOpenAI.call_args.kwargs
        self.assertEqual(kwargs["base_url"], "http://localhost:9001/v1")
        self.assertEqual(kwargs["api_key"], "")
        self.assertIsInstance(
            kwargs["default_headers"]["Authorization"], self.mod.Omit
        )

    async def test_client_uses_default_model_id_when_missing(self):
        response = SimpleNamespace(output=[])
        self.openai_stub.AsyncOpenAI.return_value.responses.create = AsyncMock(
            return_value=response
        )
        client = self.mod.OpenAIClient(
            api_key="key", base_url="http://localhost:9001/v1"
        )
        client._template_messages = MagicMock(return_value=[{"c": 1}])

        await client("", [], use_async_generator=False)

        self.openai_stub.AsyncOpenAI.return_value.responses.create.assert_awaited_once_with(
            extra_headers={
                "X-Title": "Avalan",
                "HTTP-Referer": "https://github.com/avalan-ai/avalan",
            },
            model="default",
            input=[{"c": 1}],
            store=False,
            stream=False,
        )

    async def test_model_loads_with_base_url_without_access_token(self):
        with patch.object(self.mod, "OpenAIClient") as ClientMock:
            settings = TransformerEngineSettings(
                auto_load_model=False,
                auto_load_tokenizer=False,
                base_url="http://localhost:9001/v1",
            )
            model = self.mod.OpenAIModel("m", settings)
            loaded = model._load_model()

        ClientMock.assert_called_once_with(
            base_url="http://localhost:9001/v1", api_key=None
        )
        self.assertIs(loaded, ClientMock.return_value)

    async def test_model_loads_with_azure_api_version(self):
        with patch.object(self.mod, "OpenAIClient") as ClientMock:
            settings = TransformerEngineSettings(
                auto_load_model=False,
                auto_load_tokenizer=False,
                access_token="token",
                base_url=(
                    "https://tenant.openai.azure.com/openai/deployments/d"
                ),
                provider_options={
                    "azure_api_version": "2025-04-01-preview",
                },
            )
            model = self.mod.OpenAIModel("deployment", settings)
            loaded = model._load_model()

        ClientMock.assert_called_once_with(
            base_url="https://tenant.openai.azure.com/openai/deployments/d",
            api_key="token",
            azure_api_version="2025-04-01-preview",
        )
        self.assertIs(loaded, ClientMock.return_value)

    async def test_model_rejects_invalid_provider_api_version(self):
        settings = TransformerEngineSettings(
            auto_load_model=False,
            auto_load_tokenizer=False,
            access_token="token",
            base_url="https://tenant.openai.azure.com/openai/deployments/d",
            provider_options={"azure_api_version": 20250401},
        )
        model = self.mod.OpenAIModel("deployment", settings)

        with self.assertRaises(AssertionError):
            model._load_model()

    async def test_model_loads_with_provider_retry_options(self):
        settings = TransformerEngineSettings(
            auto_load_model=False,
            auto_load_tokenizer=False,
            access_token="token",
            base_url="https://api.openai.com/v1",
            provider_options={
                "openai_max_retries": 0,
                "openai_response_failed_retries": 2,
                "openai_response_failed_retry_delay_seconds": 0.25,
                "openai_timeout_seconds": 45,
            },
        )
        model = self.mod.OpenAIModel("deployment", settings)
        loaded = model._load_model()

        self.openai_stub.AsyncOpenAI.assert_called_once_with(
            base_url="https://api.openai.com/v1",
            api_key="token",
            max_retries=0,
            timeout=45.0,
        )
        self.assertEqual(loaded._stream_response_failed_retries, 2)
        self.assertEqual(
            loaded._stream_response_failed_retry_delay_seconds,
            0.25,
        )

    async def test_client_response_failed_retry_default_budget(self):
        client = self.mod.OpenAIClient(api_key="token", base_url="b")

        self.assertEqual(client._stream_response_failed_retries, 24)
        self.assertEqual(
            client._stream_response_failed_retry_delay_seconds,
            1.0,
        )

    async def test_model_rejects_invalid_provider_retry_options(self):
        cases: list[dict[str, Any]] = [
            {"openai_max_retries": -1},
            {"openai_max_retries": 1.5},
            {"openai_response_failed_retries": -1},
            {"openai_response_failed_retries": 1.5},
            {"openai_response_failed_retry_delay_seconds": -0.1},
            {"openai_response_failed_retry_delay_seconds": False},
            {"openai_timeout_seconds": 0},
            {"openai_timeout_seconds": -0.1},
            {"openai_timeout_seconds": False},
        ]

        for provider_options in cases:
            with self.subTest(provider_options=provider_options):
                settings = TransformerEngineSettings(
                    auto_load_model=False,
                    auto_load_tokenizer=False,
                    access_token="token",
                    base_url="https://api.openai.com/v1",
                    provider_options=provider_options,
                )
                model = self.mod.OpenAIModel("deployment", settings)

                with self.assertRaises(AssertionError):
                    model._load_model()

    async def test_stream_event_types(self):
        events = [
            SimpleNamespace(type="response.output_item.added"),
            SimpleNamespace(type="response.content_part.added"),
            SimpleNamespace(type="response.reasoning_text.delta", delta="r1"),
            SimpleNamespace(type="response.reasoning_text.delta", delta="r2"),
            SimpleNamespace(type="response.reasoning_text.done"),
            SimpleNamespace(type="response.output_item.done"),
            SimpleNamespace(
                type="response.output_item.added",
                item=SimpleNamespace(
                    id="c1",
                    custom_tool_call=SimpleNamespace(
                        id="c1", name="pkg__func"
                    ),
                ),
            ),
            SimpleNamespace(
                type="response.function_call_arguments.delta",
                id="c1",
                delta="{",
            ),
            SimpleNamespace(
                type="response.function_call_arguments.delta",
                id="c1",
                delta="}",
            ),
            SimpleNamespace(
                type="response.output_item.done", item=SimpleNamespace(id="c1")
            ),
            SimpleNamespace(type="response.output_item.added"),
            SimpleNamespace(type="response.content_part.added"),
            SimpleNamespace(type="response.output_text.delta", delta="hi"),
            SimpleNamespace(type="response.output_item.done"),
        ]
        stream = self.mod.OpenAIStream(AsyncIter(events))
        items = await _stream_items(stream)

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.REASONING_DELTA,
                StreamItemKind.REASONING_DELTA,
                StreamItemKind.REASONING_DONE,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        accumulator = accumulate_canonical_stream_items(items)
        self.assertEqual(accumulator.reasoning_text, "r1r2")
        self.assertEqual(
            accumulator.tool_call_arguments,
            {"c1": "{}"},
        )
        self.assertEqual(accumulator.answer_text, "hi")
        ready = next(
            item
            for item in items
            if item.kind is StreamItemKind.TOOL_CALL_READY
        )
        self.assertEqual(ready.data, {"name": "pkg__func"})

    async def test_stream_ignores_message_output_item_done_as_tool_call(self):
        stream = self.mod.OpenAIStream(
            AsyncIter(
                [
                    SimpleNamespace(
                        type="response.output_text.delta", delta="done"
                    ),
                    SimpleNamespace(
                        type="response.output_item.done",
                        item=SimpleNamespace(type="message", id="msg_1"),
                    ),
                ]
            )
        )

        items = await _stream_items(stream)

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertFalse(
            any(item.channel is StreamChannel.TOOL_CALL for item in items)
        )
        self.assertEqual(
            accumulate_canonical_stream_items(items).answer_text,
            "done",
        )

    async def test_stream_records_stateless_reasoning_output_items(self):
        collected = []
        reasoning_item = {
            "type": "reasoning",
            "id": "rs_1",
            "content": [],
            "encrypted_content": "encrypted",
            "status": None,
        }
        call_item = {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": "{}",
            "namespace": None,
            "metadata": {
                "status": "ignored",
                "items": [
                    {"namespace": None, "value": 1},
                ],
            },
            "status": "completed",
        }
        expected_reasoning = dict(reasoning_item)
        expected_reasoning.pop("id")
        expected_reasoning.pop("status")
        expected_reasoning.pop("content")
        expected_call = dict(call_item)
        expected_call.pop("id")
        expected_call.pop("status")
        expected_call.pop("namespace")
        expected_call["metadata"] = {"items": [{"value": 1}]}
        stream = self.mod.OpenAIStream(
            AsyncIter(
                [
                    {
                        "type": "response.output_item.done",
                        "item": reasoning_item,
                    },
                    {
                        "type": "response.output_item.done",
                        "item": call_item,
                    },
                ]
            ),
            output_item_sink=collected.append,
        )

        await _stream_items(stream)

        self.assertEqual(collected, [expected_reasoning, expected_call])

    async def test_stream_ignores_non_replayable_stateless_output_items(self):
        collected = []
        stream = self.mod.OpenAIStream(
            AsyncIter(
                [
                    {
                        "type": "response.output_item.done",
                        "item": "raw",
                    },
                    {
                        "type": "response.output_item.done",
                        "item": {"type": "message", "id": "msg_1"},
                    },
                ]
            ),
            output_item_sink=collected.append,
        )

        await _stream_items(stream)

        self.assertEqual(collected, [])

    async def test_stream_retries_empty_response_failed_before_output(self):
        retry_streams = []

        async def retry_stream():
            retry_streams.append("retry")
            return AsyncIter(
                [
                    SimpleNamespace(
                        type="response.output_text.delta", delta="ok"
                    ),
                    SimpleNamespace(
                        type="response.completed",
                        response=SimpleNamespace(usage={}),
                    ),
                ]
            )

        stream = self.mod.OpenAIStream(
            AsyncIter(
                [
                    SimpleNamespace(
                        type="response.failed",
                        response=SimpleNamespace(
                            status="failed",
                            error=None,
                            output=[],
                        ),
                    )
                ]
            ),
            stream_factory=retry_stream,
            stream_retries=1,
        )

        items = await _stream_items(stream)

        self.assertEqual(retry_streams, ["retry"])
        self.assertEqual(
            accumulate_canonical_stream_items(items).answer_text,
            "ok",
        )

    async def test_stream_retries_response_failed_error_code_before_output(
        self,
    ):
        cases = [
            SimpleNamespace(
                type="response.failed",
                response=SimpleNamespace(
                    status="failed",
                    error={
                        "code": "response_failed",
                        "message": "response failed",
                    },
                    output=[],
                ),
            ),
            SimpleNamespace(
                type="response.failed",
                error={
                    "code": "response_failed",
                    "message": "response failed",
                },
            ),
        ]

        for failed_event in cases:
            with self.subTest(failed_event=failed_event):
                retry_streams = []

                async def retry_stream():
                    retry_streams.append("retry")
                    return AsyncIter(
                        [
                            SimpleNamespace(
                                type="response.output_text.delta",
                                delta="ok",
                            ),
                            SimpleNamespace(
                                type="response.completed",
                                response=SimpleNamespace(usage={}),
                            ),
                        ]
                    )

                stream = self.mod.OpenAIStream(
                    AsyncIter([failed_event]),
                    stream_factory=retry_stream,
                    stream_retries=1,
                )

                items = await _stream_items(stream)

                self.assertEqual(retry_streams, ["retry"])
                self.assertEqual(
                    accumulate_canonical_stream_items(items).answer_text,
                    "ok",
                )

    async def test_stream_retries_reasoning_only_response_failed(self):
        retry_streams = []
        collected: list[dict[str, Any]] = []
        reasoning_item = {
            "type": "reasoning",
            "id": "rs_1",
            "encrypted_content": "ciphertext",
            "content": [],
        }

        def rollback(count: int) -> None:
            del collected[-count:]

        async def retry_stream():
            retry_streams.append("retry")
            return AsyncIter(
                [
                    SimpleNamespace(
                        type="response.output_text.delta", delta="ok"
                    ),
                    SimpleNamespace(
                        type="response.completed",
                        response=SimpleNamespace(usage={}),
                    ),
                ]
            )

        stream = self.mod.OpenAIStream(
            AsyncIter(
                [
                    SimpleNamespace(
                        type="response.output_item.done",
                        item=reasoning_item,
                    ),
                    SimpleNamespace(
                        type="response.failed",
                        response=SimpleNamespace(
                            status="failed",
                            error=None,
                            output=[reasoning_item],
                        ),
                    ),
                ]
            ),
            output_item_sink=collected.append,
            output_item_rollback=rollback,
            stream_factory=retry_stream,
            stream_retries=1,
        )

        items = await _stream_items(stream)

        self.assertEqual(retry_streams, ["retry"])
        self.assertEqual(collected, [])
        self.assertEqual(
            accumulate_canonical_stream_items(items).answer_text,
            "ok",
        )

    async def test_stream_retry_keeps_output_items_without_rollback(self):
        retry_streams = []
        collected: list[dict[str, Any]] = []
        reasoning_item = {
            "type": "reasoning",
            "id": "rs_1",
            "encrypted_content": "ciphertext",
            "content": [],
        }

        async def retry_stream():
            retry_streams.append("retry")
            return AsyncIter(
                [
                    SimpleNamespace(
                        type="response.output_text.delta", delta="ok"
                    ),
                    SimpleNamespace(
                        type="response.completed",
                        response=SimpleNamespace(usage={}),
                    ),
                ]
            )

        stream = self.mod.OpenAIStream(
            AsyncIter(
                [
                    SimpleNamespace(
                        type="response.output_item.done",
                        item=reasoning_item,
                    ),
                    SimpleNamespace(
                        type="response.failed",
                        response=SimpleNamespace(
                            status="failed",
                            error=None,
                            output=[reasoning_item],
                        ),
                    ),
                ]
            ),
            output_item_sink=collected.append,
            stream_factory=retry_stream,
            stream_retries=1,
        )

        items = await _stream_items(stream)

        self.assertEqual(retry_streams, ["retry"])
        self.assertEqual(
            collected,
            [
                {
                    "type": "reasoning",
                    "encrypted_content": "ciphertext",
                }
            ],
        )
        self.assertEqual(
            accumulate_canonical_stream_items(items).answer_text,
            "ok",
        )

    async def test_stream_retries_unstreamed_failed_response_output(self):
        cases = [
            [{"type": "message", "id": "msg_1"}],
            [{"type": "function_call", "call_id": "call_1"}],
        ]

        for output in cases:
            with self.subTest(output=output):
                retry_streams = []

                async def retry_stream():
                    retry_streams.append("retry")
                    return AsyncIter(
                        [
                            SimpleNamespace(
                                type="response.output_text.delta",
                                delta="ok",
                            ),
                            SimpleNamespace(
                                type="response.completed",
                                response=SimpleNamespace(usage={}),
                            ),
                        ]
                    )

                stream = self.mod.OpenAIStream(
                    AsyncIter(
                        [
                            SimpleNamespace(
                                type="response.failed",
                                response=SimpleNamespace(
                                    status="failed",
                                    error=None,
                                    output=output,
                                ),
                            ),
                        ]
                    ),
                    stream_factory=retry_stream,
                    stream_retries=1,
                )

                items = await _stream_items(stream)

                self.assertEqual(retry_streams, ["retry"])
                self.assertEqual(
                    accumulate_canonical_stream_items(items).answer_text,
                    "ok",
                )

    async def test_client_retry_stream_factory_after_empty_response_failed(
        self,
    ):
        failed_stream = TrackedAsyncIter(
            [
                SimpleNamespace(
                    type="response.failed",
                    response=SimpleNamespace(
                        status="failed",
                        error=None,
                        output=[],
                    ),
                )
            ]
        )
        recovered_stream = AsyncIter(
            [SimpleNamespace(type="response.output_text.delta", delta="ok")]
        )
        create_mock = AsyncMock(side_effect=[failed_stream, recovered_stream])
        self.openai_stub.AsyncOpenAI.return_value.responses.create = (
            create_mock
        )
        client = self.mod.OpenAIClient(api_key="k", base_url="b")
        client._template_messages = MagicMock(
            return_value=[{"role": "user", "content": "hi"}]
        )

        with patch.object(self.mod, "sleep", new=AsyncMock()) as sleep_mock:
            stream = await client("m", [])
            items = await _stream_items(stream)

        self.assertGreaterEqual(failed_stream.close_count, 1)
        sleep_mock.assert_awaited_once_with(1.0)
        self.assertEqual(create_mock.await_count, 2)
        self.assertEqual(
            accumulate_canonical_stream_items(items).answer_text,
            "ok",
        )

    async def test_stream_retry_delay_caps_exponential_backoff(self):
        failed_streams = [
            TrackedAsyncIter(
                [
                    SimpleNamespace(
                        type="response.failed",
                        response=SimpleNamespace(
                            status="failed",
                            error=None,
                            output=[],
                        ),
                    )
                ]
            )
            for _ in range(5)
        ]
        recovered_stream = AsyncIter(
            [SimpleNamespace(type="response.output_text.delta", delta="ok")]
        )
        streams = [*failed_streams, recovered_stream]

        async def retry_stream() -> AsyncIter | TrackedAsyncIter:
            return streams.pop(0)

        stream = self.mod.OpenAIStream(
            streams.pop(0),
            stream_factory=retry_stream,
            stream_retries=5,
            stream_retry_delay_seconds=1,
        )

        with patch.object(self.mod, "sleep", new=AsyncMock()) as sleep_mock:
            items = await _stream_items(stream)

        self.assertEqual(
            [call.args[0] for call in sleep_mock.await_args_list],
            [1, 2, 4, 8, 8],
        )
        self.assertEqual(
            accumulate_canonical_stream_items(items).answer_text,
            "ok",
        )

    async def test_client_retry_rolls_back_failed_reasoning_items(self):
        reasoning_item = {
            "type": "reasoning",
            "id": "rs_1",
            "encrypted_content": "ciphertext",
            "content": [],
        }
        failed_stream = TrackedAsyncIter(
            [
                SimpleNamespace(
                    type="response.output_item.done",
                    item=reasoning_item,
                ),
                SimpleNamespace(
                    type="response.failed",
                    response=SimpleNamespace(
                        status="failed",
                        error=None,
                        output=[reasoning_item],
                    ),
                ),
            ]
        )
        recovered_stream = AsyncIter(
            [SimpleNamespace(type="response.output_text.delta", delta="ok")]
        )
        create_mock = AsyncMock(side_effect=[failed_stream, recovered_stream])
        self.openai_stub.AsyncOpenAI.return_value.responses.create = (
            create_mock
        )
        client = self.mod.OpenAIClient(api_key="k", base_url="b")
        client._template_messages = MagicMock(
            return_value=[{"role": "user", "content": "hi"}]
        )

        with patch.object(self.mod, "sleep", new=AsyncMock()):
            stream = await client("m", [])
            items = await _stream_items(stream)

        self.assertEqual(create_mock.await_count, 2)
        self.assertEqual(client._stateless_response_items, [])
        self.assertEqual(
            accumulate_canonical_stream_items(items).answer_text,
            "ok",
        )

    async def test_stream_retry_aclose_closes_active_retried_stream(self):
        failed_stream = CloseOnlyAsyncIter(
            [
                SimpleNamespace(
                    type="response.failed",
                    response=SimpleNamespace(
                        status="failed",
                        error=None,
                        output=[],
                    ),
                )
            ]
        )
        retried_stream = CloseOnlyAsyncIter(
            [
                SimpleNamespace(type="response.output_text.delta", delta="ok"),
                SimpleNamespace(
                    type="response.output_text.delta", delta="later"
                ),
            ]
        )

        async def retry_stream():
            return retried_stream

        stream = self.mod.OpenAIStream(
            failed_stream,
            stream_factory=retry_stream,
            stream_retries=1,
        )
        canonical = stream.canonical_stream(
            stream_session_id="responses-stream",
            run_id="run-1",
            turn_id="turn-1",
        )

        started = await anext(canonical)
        delta = await anext(canonical)
        await canonical.aclose()

        self.assertIs(started.kind, StreamItemKind.STREAM_STARTED)
        self.assertIs(delta.kind, StreamItemKind.ANSWER_DELTA)
        self.assertEqual(delta.text_delta, "ok")
        self.assertEqual(retried_stream.read_count, 1)
        self.assertGreaterEqual(failed_stream.close_count, 1)
        self.assertGreaterEqual(retried_stream.close_count, 1)

    async def test_stream_retry_cancel_closes_active_retried_stream(self):
        failed_stream = CloseOnlyAsyncIter(
            [
                SimpleNamespace(
                    type="response.failed",
                    response=SimpleNamespace(
                        status="failed",
                        error=None,
                        output=[],
                    ),
                )
            ]
        )
        retried_stream = CloseOnlyAsyncIter(
            [
                SimpleNamespace(type="response.output_text.delta", delta="ok"),
                SimpleNamespace(
                    type="response.output_text.delta", delta="later"
                ),
            ]
        )

        async def retry_stream():
            return retried_stream

        stream = self.mod.OpenAIStream(
            failed_stream,
            stream_factory=retry_stream,
            stream_retries=1,
        )
        canonical = stream.canonical_stream(
            stream_session_id="responses-stream",
            run_id="run-1",
            turn_id="turn-1",
        )

        started = await anext(canonical)
        delta = await anext(canonical)
        try:
            await stream.cancel()
            self.assertEqual(retried_stream.close_count, 1)
        finally:
            await canonical.aclose()

        self.assertIs(started.kind, StreamItemKind.STREAM_STARTED)
        self.assertIs(delta.kind, StreamItemKind.ANSWER_DELTA)
        self.assertEqual(delta.text_delta, "ok")
        self.assertEqual(retried_stream.read_count, 1)
        self.assertGreaterEqual(failed_stream.close_count, 1)
        self.assertGreaterEqual(retried_stream.close_count, 1)

    async def test_stream_cancel_during_retry_factory_closes_replacement(
        self,
    ):
        factory_started = Event()
        factory_release = Event()
        failed_stream = CloseOnlyAsyncIter(
            [
                SimpleNamespace(
                    type="response.failed",
                    response=SimpleNamespace(
                        status="failed",
                        error=None,
                        output=[],
                    ),
                )
            ]
        )
        replacement_stream = CloseOnlyAsyncIter(
            [
                SimpleNamespace(
                    type="response.output_text.delta",
                    delta="leaked",
                )
            ]
        )

        async def retry_stream():
            factory_started.set()
            await factory_release.wait()
            return replacement_stream

        stream = self.mod.OpenAIStream(
            failed_stream,
            stream_factory=retry_stream,
            stream_retries=1,
        )
        canonical = stream.canonical_stream(
            stream_session_id="responses-stream",
            run_id="run-1",
            turn_id="turn-1",
        )

        started = await anext(canonical)
        pull = create_task(anext(canonical))
        try:
            await wait_for(factory_started.wait(), 1.0)
            await stream.cancel()
            factory_release.set()
            item = await wait_for(pull, 1.0)
        finally:
            factory_release.set()
            if not pull.done():
                pull.cancel()
            await canonical.aclose()

        self.assertIs(started.kind, StreamItemKind.STREAM_STARTED)
        self.assertIs(item.kind, StreamItemKind.STREAM_CANCELLED)
        self.assertGreaterEqual(failed_stream.close_count, 1)
        self.assertEqual(replacement_stream.read_count, 0)
        self.assertEqual(replacement_stream.close_count, 1)

    async def test_stream_aclose_during_retry_factory_closes_replacement(
        self,
    ):
        factory_started = Event()
        factory_release = Event()
        failed_stream = CloseOnlyAsyncIter(
            [
                SimpleNamespace(
                    type="response.failed",
                    response=SimpleNamespace(
                        status="failed",
                        error=None,
                        output=[],
                    ),
                )
            ]
        )
        replacement_stream = CloseOnlyAsyncIter(
            [
                SimpleNamespace(
                    type="response.output_text.delta",
                    delta="leaked",
                )
            ]
        )

        async def retry_stream():
            factory_started.set()
            await factory_release.wait()
            return replacement_stream

        stream = self.mod.OpenAIStream(
            failed_stream,
            stream_factory=retry_stream,
            stream_retries=1,
        )
        canonical = stream.canonical_stream(
            stream_session_id="responses-stream",
            run_id="run-1",
            turn_id="turn-1",
        )

        started = await anext(canonical)
        pull = create_task(anext(canonical))
        try:
            await wait_for(factory_started.wait(), 1.0)
            await stream.aclose()
            factory_release.set()
            item = await wait_for(pull, 1.0)
        finally:
            factory_release.set()
            if not pull.done():
                pull.cancel()
            await canonical.aclose()

        self.assertIs(started.kind, StreamItemKind.STREAM_STARTED)
        self.assertIs(item.kind, StreamItemKind.STREAM_CANCELLED)
        self.assertGreaterEqual(failed_stream.close_count, 1)
        self.assertEqual(replacement_stream.read_count, 0)
        self.assertEqual(replacement_stream.close_count, 1)

    async def test_stream_cancel_closes_aclose_only_source(self):
        source = AcloseOnlyAsyncIter(
            [SimpleNamespace(type="response.output_text.delta", delta="late")]
        )
        stream = self.mod.OpenAIStream(source)

        await stream.cancel()

        self.assertEqual(source.read_count, 0)
        self.assertEqual(source.close_count, 1)

    async def test_stream_aclose_accepts_sync_close_source(self):
        source = SyncCloseOnlyAsyncIter(
            [SimpleNamespace(type="response.output_text.delta", delta="late")]
        )
        stream = self.mod.OpenAIStream(source)

        await stream.aclose()

        self.assertEqual(source.read_count, 0)
        self.assertEqual(source.close_count, 1)

    async def test_stream_aclose_propagates_single_cleanup_error(self):
        error = RuntimeError("close failed")
        source = FailingCloseOnlyAsyncIter(
            [SimpleNamespace(type="response.output_text.delta", delta="late")],
            error,
        )
        stream = self.mod.OpenAIStream(source)

        with self.assertRaises(RuntimeError) as context:
            await stream.aclose()

        self.assertIs(context.exception, error)

    async def test_stream_aclose_base_exception_skips_later_sources(self):
        error = BaseException("close interrupted")
        first = FailingCloseOnlyAsyncIter([], error)
        second = CloseOnlyAsyncIter([])
        stream = self.mod.OpenAIStream(first)
        stream._stream_sources = (first, second)

        with self.assertRaises(BaseException) as context:
            await stream.aclose()

        self.assertIs(context.exception, error)
        self.assertEqual(second.close_count, 0)

    async def test_stream_aclose_groups_multiple_cleanup_errors(self):
        first_error = RuntimeError("first")
        second_error = ValueError("second")
        first = FailingCloseOnlyAsyncIter([], first_error)
        second = FailingCloseOnlyAsyncIter([], second_error)
        stream = self.mod.OpenAIStream(first)
        stream._stream_sources = (first, second)

        with self.assertRaises(BaseExceptionGroup) as context:
            await stream.aclose()

        self.assertEqual(
            context.exception.exceptions,
            (first_error, second_error),
        )

    async def test_client_response_failed_retry_settings_override_defaults(
        self,
    ):
        stream_instance = AsyncIter([])
        self.openai_stub.AsyncOpenAI.return_value.responses.create = AsyncMock(
            return_value=stream_instance
        )
        client = self.mod.OpenAIClient(api_key="k", base_url="b")
        client._template_messages = MagicMock(return_value=[])
        settings = GenerationSettings(
            openai_response_failed_retries=0,
            openai_response_failed_retry_delay_seconds=2.5,
        )

        with patch.object(self.mod, "OpenAIStream") as stream_mock:
            await client("m", [], settings=settings)

        stream_mock.assert_called_once_with(
            stream=stream_instance,
            provider_family="openai",
            output_item_sink=client._record_stateless_response_item,
            output_item_rollback=client._rollback_stateless_response_items,
            stream_factory=ANY,
            stream_retry_delay_seconds=2.5,
            stream_retries=0,
        )

    async def test_client_openai_request_settings_override_defaults(self):
        stream_instance = AsyncIter([])
        request_client = MagicMock()
        request_client.responses.create = AsyncMock(
            return_value=stream_instance
        )
        sdk_client = self.openai_stub.AsyncOpenAI.return_value
        sdk_client.with_options.return_value = request_client
        client = self.mod.OpenAIClient(api_key="k", base_url="b")
        client._template_messages = MagicMock(return_value=[{"c": 1}])
        settings = GenerationSettings(
            openai_max_retries=0,
            openai_timeout_seconds=30,
        )

        with patch.object(self.mod, "OpenAIStream"):
            await client("m", [], settings=settings)

        sdk_client.with_options.assert_called_once_with(max_retries=0)
        request_client.responses.create.assert_awaited_once_with(
            extra_headers={
                "X-Title": "Avalan",
                "HTTP-Referer": "https://github.com/avalan-ai/avalan",
            },
            model="m",
            input=[{"c": 1}],
            store=False,
            stream=True,
            temperature=1.0,
            top_p=1.0,
            timeout=30.0,
        )

    async def test_stream_does_not_retry_failed_response_with_error(self):
        cases = [
            SimpleNamespace(
                status="failed",
                error=SimpleNamespace(message="boom"),
                output=[],
            ),
        ]

        for response in cases:
            with self.subTest(response=response):
                retry_stream = AsyncMock()
                stream = self.mod.OpenAIStream(
                    AsyncIter(
                        [
                            SimpleNamespace(
                                type="response.failed",
                                response=response,
                            )
                        ]
                    ),
                    stream_factory=retry_stream,
                    stream_retries=1,
                )

                items = await _stream_items(stream)

                retry_stream.assert_not_awaited()
                self.assertIn(
                    StreamItemKind.STREAM_ERRORED,
                    [item.kind for item in items],
                )

    async def test_stream_does_not_retry_response_failed_after_output(self):
        retry_stream = AsyncMock()
        stream = self.mod.OpenAIStream(
            AsyncIter(
                [
                    SimpleNamespace(
                        type="response.output_text.delta", delta="partial"
                    ),
                    SimpleNamespace(
                        type="response.failed",
                        response=SimpleNamespace(
                            status="failed",
                            error=None,
                            output=[],
                        ),
                    ),
                ]
            ),
            stream_factory=retry_stream,
            stream_retries=1,
        )

        items = await _stream_items(stream)

        retry_stream.assert_not_awaited()
        self.assertIn(
            StreamItemKind.STREAM_ERRORED, [item.kind for item in items]
        )

    async def test_stream_completion_output_emits_structured_answer(self):
        events = [
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    output=[
                        SimpleNamespace(
                            type="message",
                            content=[
                                SimpleNamespace(
                                    type="output_text",
                                    text='{"answer":"ok"}',
                                )
                            ],
                        )
                    ],
                    usage={
                        "input_tokens": 3,
                        "output_tokens": 5,
                        "total_tokens": 8,
                    },
                ),
            )
        ]
        stream = self.mod.OpenAIStream(AsyncIter(events))

        items = await _stream_items(stream)

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.USAGE_COMPLETED,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        accumulator = accumulate_canonical_stream_items(items)
        self.assertEqual(accumulator.answer_text, '{"answer":"ok"}')
        self.assertEqual(
            accumulator.usage_items[0].usage,
            {
                "input_tokens": 3,
                "output_tokens": 5,
                "total_tokens": 8,
            },
        )

    async def test_stream_completion_output_does_not_duplicate_deltas(self):
        events = [
            SimpleNamespace(type="response.output_text.delta", delta="ok"),
            SimpleNamespace(type="response.output_text.done"),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    output=[
                        SimpleNamespace(
                            type="message",
                            content=[SimpleNamespace(text="ignored")],
                        )
                    ]
                ),
            ),
        ]
        stream = self.mod.OpenAIStream(AsyncIter(events))

        items = await _stream_items(stream)

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(
            accumulate_canonical_stream_items(items).answer_text,
            "ok",
        )

    async def test_stream_output_text_done_emits_text_without_delta(self):
        events = [
            SimpleNamespace(
                type="response.output_text.done",
                text='{"answer":"done"}',
            ),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(output_text='{"answer":"ignored"}'),
            ),
        ]
        stream = self.mod.OpenAIStream(AsyncIter(events))

        items = await _stream_items(stream)

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(
            accumulate_canonical_stream_items(items).answer_text,
            '{"answer":"done"}',
        )

    async def test_stream_empty_output_text_done_uses_completion_output(self):
        events = [
            SimpleNamespace(type="response.output_text.done"),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(output_text='{"answer":"done"}'),
            ),
        ]
        stream = self.mod.OpenAIStream(AsyncIter(events))

        items = await _stream_items(stream)

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(
            accumulate_canonical_stream_items(items).answer_text,
            '{"answer":"done"}',
        )

    async def test_stream_completion_output_prefers_output_text(self):
        events = [
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    output_text='{"answer":"ok"}',
                    output=[
                        SimpleNamespace(
                            type="message",
                            content=[SimpleNamespace(text='{"answer":"ok"}')],
                        )
                    ],
                ),
            )
        ]
        stream = self.mod.OpenAIStream(AsyncIter(events))

        items = await _stream_items(stream)

        self.assertEqual(
            accumulate_canonical_stream_items(items).answer_text,
            '{"answer":"ok"}',
        )

    async def test_stream_completion_output_ignores_tool_calls(self):
        events = [
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    output=[
                        SimpleNamespace(
                            type="function_call",
                            id="call-1",
                            name="database.run",
                            arguments='{"sql":"SELECT 1"}',
                        )
                    ]
                ),
            )
        ]
        stream = self.mod.OpenAIStream(AsyncIter(events))

        items = await _stream_items(stream)

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(
            accumulate_canonical_stream_items(items).answer_text,
            "",
        )

    async def test_stream_message_output_item_done_emits_structured_answer(
        self,
    ):
        events = [
            SimpleNamespace(
                type="response.output_item.done",
                item=SimpleNamespace(
                    type="message",
                    id="msg_1",
                    content=[
                        SimpleNamespace(
                            type="output_text",
                            text='{"answer":"done"}',
                        )
                    ],
                ),
            ),
        ]
        stream = self.mod.OpenAIStream(AsyncIter(events))

        items = await _stream_items(stream)

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(
            accumulate_canonical_stream_items(items).answer_text,
            '{"answer":"done"}',
        )

    async def test_stream_message_output_item_done_emits_direct_text(self):
        events = [
            SimpleNamespace(
                type="response.output_item.done",
                item=SimpleNamespace(
                    type="message",
                    id="msg_1",
                    text='{"answer":"direct"}',
                ),
            ),
        ]
        stream = self.mod.OpenAIStream(AsyncIter(events))

        items = await _stream_items(stream)

        self.assertEqual(
            accumulate_canonical_stream_items(items).answer_text,
            '{"answer":"direct"}',
        )

    async def test_stream_message_output_item_done_without_text_is_ignored(
        self,
    ):
        events = [
            SimpleNamespace(
                type="response.output_item.done",
                item=SimpleNamespace(
                    type="message",
                    id="msg_1",
                    content=[],
                ),
            ),
        ]
        stream = self.mod.OpenAIStream(AsyncIter(events))

        items = await _stream_items(stream)

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )

    async def test_stream_ignores_tool_added_without_id(self):
        stream = self.mod.OpenAIStream(
            AsyncIter(
                [
                    SimpleNamespace(
                        type="response.output_item.added",
                        item=SimpleNamespace(type="function_call"),
                    ),
                    SimpleNamespace(
                        type="response.output_text.delta", delta="done"
                    ),
                ]
            )
        )

        items = await _stream_items(stream)

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertFalse(
            any(item.channel is StreamChannel.TOOL_CALL for item in items)
        )
        self.assertEqual(
            accumulate_canonical_stream_items(items).answer_text,
            "done",
        )

    async def test_canonical_stream_maps_responses_events(self):
        usage = {
            "input_tokens": 2,
            "output_tokens": 3,
            "total_tokens": 5,
        }
        events = [
            SimpleNamespace(
                type="response.reasoning_text.delta", delta="plan "
            ),
            SimpleNamespace(type="response.reasoning_text.done"),
            SimpleNamespace(
                type="response.output_item.added",
                item=SimpleNamespace(
                    id="call-1",
                    custom_tool_call=SimpleNamespace(
                        id="call-1", name="pkg.lookup"
                    ),
                ),
            ),
            SimpleNamespace(
                type="response.function_call_arguments.delta",
                item_id="call-1",
                delta='{"city"',
            ),
            SimpleNamespace(
                type="response.function_call_arguments.delta",
                item_id="call-1",
                delta=':"Paris"}',
            ),
            SimpleNamespace(
                type="response.function_call_arguments.done",
                item_id="call-1",
            ),
            SimpleNamespace(
                type="response.output_item.done",
                item=SimpleNamespace(id="call-1"),
            ),
            SimpleNamespace(type="response.output_text.delta", delta="hi "),
            SimpleNamespace(type="response.output_text.done"),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(usage=usage),
            ),
        ]
        stream = self.mod.OpenAIStream(AsyncIter(events))

        items = [
            item
            async for item in stream.canonical_stream(
                stream_session_id="responses-stream",
                run_id="run-1",
                turn_id="turn-1",
            )
        ]

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.REASONING_DELTA,
                StreamItemKind.REASONING_DONE,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.USAGE_COMPLETED,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual([item.sequence for item in items], list(range(12)))
        self.assertEqual({item.provider_family for item in items}, {"openai"})
        self.assertEqual(
            items[0].metadata["capabilities"]["provider_family"], "openai"
        )
        self.assertIs(items[1].channel, StreamChannel.REASONING)
        self.assertEqual(
            items[1].provider_event_type, "response.reasoning_text.delta"
        )
        self.assertEqual(
            {
                item.correlation.tool_call_id
                for item in items
                if item.channel is StreamChannel.TOOL_CALL
            },
            {"call-1"},
        )
        self.assertEqual(items[5].data, {"name": "pkg.lookup"})

        accumulator = accumulate_canonical_stream_items(items)
        self.assertEqual(accumulator.answer_text, "hi ")
        self.assertEqual(accumulator.reasoning_text, "plan ")
        self.assertEqual(
            accumulator.tool_call_arguments,
            {"call-1": '{"city":"Paris"}'},
        )
        self.assertEqual(accumulator.final_usage, usage)
        self.assertIs(
            accumulator.terminal_outcome,
            StreamTerminalOutcome.COMPLETED,
        )

    async def test_canonical_stream_preserves_hosted_reasoning_whitespace(
        self,
    ):
        usage = {
            "input_tokens": 1,
            "output_tokens": 4,
            "total_tokens": 5,
        }
        events = [
            SimpleNamespace(
                type="response.output_text.delta", delta="pre <thi"
            ),
            SimpleNamespace(
                type="response.output_text.delta",
                delta="nk> stays answer ",
            ),
            SimpleNamespace(
                type="response.reasoning_text.delta", delta="  plan"
            ),
            SimpleNamespace(type="response.reasoning_text.delta", delta="\n"),
            SimpleNamespace(type="response.reasoning_text.done"),
            SimpleNamespace(
                type="response.output_text.delta", delta="post </think>"
            ),
            SimpleNamespace(type="response.output_text.done"),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(usage=usage),
            ),
        ]
        stream = self.mod.OpenAIStream(AsyncIter(events))

        items = [
            item
            async for item in stream.canonical_stream(
                stream_session_id="responses-stream",
                run_id="run-1",
                turn_id="turn-1",
            )
        ]

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.REASONING_DELTA,
                StreamItemKind.REASONING_DELTA,
                StreamItemKind.REASONING_DONE,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.USAGE_COMPLETED,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(
            [
                item.text_delta
                for item in items
                if item.kind is StreamItemKind.REASONING_DELTA
            ],
            ["  plan", "\n"],
        )
        accumulator = accumulate_canonical_stream_items(items)
        self.assertEqual(
            accumulator.answer_text, "pre <think> stays answer post </think>"
        )
        self.assertEqual(accumulator.reasoning_text, "  plan\n")
        self.assertEqual(accumulator.final_usage, usage)

    async def test_canonical_disconnect_closes_provider_no_read_ahead(
        self,
    ):
        source = TrackedAsyncIter(
            [SimpleNamespace(type="response.output_text.delta", delta="late")]
        )
        stream = self.mod.OpenAIStream(source)
        canonical = stream.canonical_stream(
            stream_session_id="responses-stream",
            run_id="run-1",
            turn_id="turn-1",
        )

        started = await anext(canonical)
        await canonical.aclose()

        self.assertIs(started.kind, StreamItemKind.STREAM_STARTED)
        self.assertEqual(source.read_count, 0)
        self.assertEqual(source.close_count, 1)

    async def test_canonical_cancel_closes_pending_provider_pull(
        self,
    ):
        source = PendingAsyncIter()
        stream = self.mod.OpenAIStream(source)
        canonical = stream.canonical_stream(
            stream_session_id="responses-stream",
            run_id="run-1",
            turn_id="turn-1",
        )

        started = await anext(canonical)
        pull = create_task(anext(canonical))
        try:
            await wait_for(source.started.wait(), 1.0)
            pull.cancel()
            try:
                cancelled = await wait_for(pull, 1.0)
            except CancelledError:
                cancelled = None
        finally:
            await canonical.aclose()

        self.assertIs(started.kind, StreamItemKind.STREAM_STARTED)
        if cancelled is not None:
            self.assertIs(cancelled.kind, StreamItemKind.STREAM_CANCELLED)
            self.assertIs(
                cancelled.terminal_outcome,
                StreamTerminalOutcome.CANCELLED,
            )
        self.assertTrue(source.pull_cancelled)
        self.assertEqual(source.close_count, 1)

    async def test_canonical_stream_maps_done_function_call_arguments(self):
        events = [
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "id": "call-2",
                    "name": "pkg.search",
                    "arguments": '{"q": "avalan"}',
                },
            }
        ]
        stream = self.mod.OpenAIStream(AsyncIter(events))

        items = [
            item
            async for item in stream.canonical_stream(
                stream_session_id="responses-stream",
                run_id="run-1",
                turn_id="turn-1",
                close_after_terminal=False,
            )
        ]

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.STREAM_COMPLETED,
            ],
        )
        accumulator = accumulate_canonical_stream_items(items)
        self.assertEqual(
            accumulator.tool_call_arguments,
            {"call-2": '{"q": "avalan"}'},
        )
        self.assertEqual(items[2].data, {"name": "pkg.search"})
        self.assertEqual(items[1].provider_payload, events[0])

    async def test_response_uses_openai_canonical_stream_for_multiple_tools(
        self,
    ):
        events = [
            SimpleNamespace(
                type="response.output_item.added",
                item=SimpleNamespace(
                    type="function_call",
                    id="fc_1",
                    name="math.calculator",
                ),
            ),
            SimpleNamespace(
                type="response.function_call_arguments.delta",
                item_id="fc_1",
                delta='{"expression":"4 + 6"}',
            ),
            SimpleNamespace(
                type="response.output_item.done",
                item=SimpleNamespace(
                    type="function_call",
                    id="fc_1",
                    name="math.calculator",
                    arguments='{"expression":"4 + 6"}',
                ),
            ),
            SimpleNamespace(
                type="response.output_item.added",
                item=SimpleNamespace(
                    type="function_call",
                    id="fc_2",
                    name="math.calculator",
                ),
            ),
            SimpleNamespace(
                type="response.function_call_arguments.delta",
                item_id="fc_2",
                delta='{"expression":"10 * 5 / 2"}',
            ),
            SimpleNamespace(
                type="response.output_item.done",
                item=SimpleNamespace(
                    type="function_call",
                    id="fc_2",
                    name="math.calculator",
                    arguments='{"expression":"10 * 5 / 2"}',
                ),
            ),
            SimpleNamespace(type="response.output_text.delta", delta="25"),
            SimpleNamespace(type="response.output_text.done"),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    usage={
                        "input_tokens": 1,
                        "output_tokens": 2,
                        "total_tokens": 3,
                    }
                ),
            ),
        ]
        stream = self.mod.OpenAIStream(AsyncIter(events))
        response = TextGenerationResponse(
            stream,
            logger=getLogger(),
            use_async_generator=True,
        )

        items = [
            item
            async for item in response.canonical_stream(
                stream_session_id="responses-stream",
                run_id="run-1",
                turn_id="turn-1",
            )
        ]

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.USAGE_COMPLETED,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        accumulator = accumulate_canonical_stream_items(items)
        self.assertEqual(accumulator.answer_text, "25")
        self.assertEqual(
            accumulator.tool_call_arguments,
            {
                "fc_1": '{"expression":"4 + 6"}',
                "fc_2": '{"expression":"10 * 5 / 2"}',
            },
        )

    async def test_canonical_stream_preserves_done_item_provider_payload(self):
        payload = {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "id": "call-2",
                "name": "pkg.search",
                "arguments": '{"q": "avalan"}',
            },
        }
        modes: list[str] = []

        class ModelDumpEvent:
            type = "response.output_item.done"
            item = SimpleNamespace(
                type="function_call",
                id="call-2",
                name="pkg.search",
                arguments='{"q": "avalan"}',
            )

            def model_dump(self, *, mode: str) -> dict[str, object]:
                modes.append(mode)
                return payload

        stream = self.mod.OpenAIStream(AsyncIter([ModelDumpEvent()]))

        items = [
            item
            async for item in stream.canonical_stream(
                stream_session_id="responses-stream",
                run_id="run-1",
                turn_id="turn-1",
                close_after_terminal=False,
            )
        ]

        self.assertEqual(modes, ["json"])
        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.STREAM_COMPLETED,
            ],
        )
        self.assertEqual(items[1].provider_payload, payload)
        self.assertEqual(items[2].provider_payload, payload)
        self.assertEqual(items[3].provider_payload, payload)

    async def test_canonical_stream_ignores_non_object_provider_payload(self):
        class ModelDumpEvent:
            type = "response.output_text.delta"
            delta = "hi"

            def model_dump(self, *, mode: str) -> object:
                return ["not", "an", "event", mode]

        stream = self.mod.OpenAIStream(AsyncIter([ModelDumpEvent()]))

        items = [
            item
            async for item in stream.canonical_stream(
                stream_session_id="responses-stream",
                run_id="run-1",
                turn_id="turn-1",
                close_after_terminal=False,
            )
        ]

        self.assertEqual(items[1].text_delta, "hi")
        self.assertIsNone(items[1].provider_payload)

    async def test_canonical_stream_uses_response_call_id_for_function_call(
        self,
    ):
        events = [
            SimpleNamespace(
                type="response.output_item.added",
                item=SimpleNamespace(
                    type="function_call",
                    id="item-1",
                    call_id="call-1",
                    name="pkg.search",
                ),
            ),
            SimpleNamespace(
                type="response.function_call_arguments.delta",
                item_id="item-1",
                delta='{"q"',
            ),
            SimpleNamespace(
                type="response.function_call_arguments.delta",
                item_id="item-1",
                delta=':"avalan"}',
            ),
            SimpleNamespace(
                type="response.output_item.done",
                item=SimpleNamespace(
                    type="function_call",
                    id="item-1",
                    name="pkg.search",
                    arguments='{"q":"avalan"}',
                ),
            ),
        ]
        stream = self.mod.OpenAIStream(AsyncIter(events))

        items = [
            item
            async for item in stream.canonical_stream(
                stream_session_id="responses-stream",
                run_id="run-1",
                turn_id="turn-1",
                close_after_terminal=False,
            )
        ]

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.STREAM_COMPLETED,
            ],
        )
        self.assertEqual(
            {
                item.correlation.tool_call_id
                for item in items
                if item.channel is StreamChannel.TOOL_CALL
            },
            {"call-1"},
        )
        self.assertEqual(
            {
                item.correlation.protocol_item_id
                for item in items
                if item.channel is StreamChannel.TOOL_CALL
            },
            {"item-1"},
        )
        accumulator = accumulate_canonical_stream_items(items)
        self.assertEqual(
            accumulator.tool_call_arguments,
            {"call-1": '{"q":"avalan"}'},
        )
        self.assertEqual(items[3].data, {"name": "pkg.search"})

    async def test_canonical_stream_rejects_mismatched_function_call_id(
        self,
    ):
        events = [
            SimpleNamespace(
                type="response.output_item.added",
                item=SimpleNamespace(
                    type="function_call",
                    id="item-1",
                    call_id="call-1",
                    name="pkg.search",
                ),
            ),
            SimpleNamespace(
                type="response.function_call_arguments.delta",
                item_id="item-1",
                delta='{"q":"avalan"}',
            ),
            SimpleNamespace(
                type="response.output_item.done",
                item=SimpleNamespace(
                    type="function_call",
                    id="item-1",
                    call_id="call-2",
                    name="pkg.search",
                ),
            ),
        ]
        stream = self.mod.OpenAIStream(AsyncIter(events))

        items = [
            item
            async for item in stream.canonical_stream(
                stream_session_id="responses-stream",
                run_id="run-1",
                turn_id="turn-1",
            )
        ]

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.STREAM_ERRORED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(items[2].metadata["tool_call.close_reason"], "error")
        error_data = items[3].data
        assert isinstance(error_data, dict)
        self.assertEqual(error_data["error_type"], "StreamValidationError")
        self.assertIn("call-1", error_data["message"])

    async def test_canonical_stream_handles_empty_responses_control_events(
        self,
    ):
        events = [
            SimpleNamespace(type="response.in_progress"),
            SimpleNamespace(type="response.output_item.added"),
            SimpleNamespace(type="response.output_item.added", item=None),
            SimpleNamespace(
                type="response.output_item.added", item=SimpleNamespace()
            ),
            SimpleNamespace(type="response.output_item.done"),
            SimpleNamespace(type="response.output_item.done", id="call-1"),
            SimpleNamespace(
                type="response.output_item.done",
                item=SimpleNamespace(id="call-2"),
            ),
        ]
        stream = self.mod.OpenAIStream(AsyncIter(events))

        items = [
            item
            async for item in stream.canonical_stream(
                stream_session_id="responses-stream",
                run_id="run-1",
                turn_id="turn-1",
            )
        ]

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(
            {
                item.correlation.tool_call_id
                for item in items
                if item.channel is StreamChannel.TOOL_CALL
            },
            {"call-1", "call-2"},
        )
        self.assertEqual(items[1].data, {"name": None})
        self.assertEqual(items[3].data, {"name": None})

    async def test_canonical_stream_ignores_message_done_as_tool_call(self):
        events = [
            SimpleNamespace(type="response.output_text.delta", delta="done"),
            SimpleNamespace(
                type="response.output_item.done",
                item=SimpleNamespace(type="message", id="msg_1"),
            ),
        ]
        stream = self.mod.OpenAIStream(AsyncIter(events))

        items = [
            item
            async for item in stream.canonical_stream(
                stream_session_id="responses-stream",
                run_id="run-1",
                turn_id="turn-1",
                close_after_terminal=False,
            )
        ]

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_COMPLETED,
            ],
        )
        self.assertFalse(
            any(item.channel is StreamChannel.TOOL_CALL for item in items)
        )
        self.assertEqual(
            accumulate_canonical_stream_items(items).answer_text,
            "done",
        )

    async def test_canonical_stream_maps_custom_done_mapping_input(self):
        events = [
            SimpleNamespace(
                type="response.output_item.done",
                item=SimpleNamespace(
                    custom_tool_call=SimpleNamespace(
                        id="call-3",
                        name="pkg.custom",
                        input={"x": 1},
                    )
                ),
            )
        ]
        stream = self.mod.OpenAIStream(AsyncIter(events))

        items = [
            item
            async for item in stream.canonical_stream(
                stream_session_id="responses-stream",
                run_id="run-1",
                turn_id="turn-1",
                close_after_terminal=False,
            )
        ]

        accumulator = accumulate_canonical_stream_items(items)
        self.assertEqual(
            loads(accumulator.tool_call_arguments["call-3"]), {"x": 1}
        )
        self.assertEqual(items[2].data, {"name": "pkg.custom"})

    async def test_canonical_stream_maps_responses_errors_and_cancellation(
        self,
    ):
        class OpaqueError:
            def __str__(self):
                return "opaque response"

        cases = (
            (
                [
                    SimpleNamespace(
                        type="response.failed", error={"code": "bad"}
                    )
                ],
                StreamItemKind.STREAM_ERRORED,
                StreamTerminalOutcome.ERRORED,
                {"error": {"code": "bad"}},
            ),
            (
                [
                    SimpleNamespace(
                        type="response.cancelled", reason="disconnect"
                    )
                ],
                StreamItemKind.STREAM_CANCELLED,
                StreamTerminalOutcome.CANCELLED,
                {"reason": "disconnect"},
            ),
            (
                [SimpleNamespace(type="response.cancelled")],
                StreamItemKind.STREAM_CANCELLED,
                StreamTerminalOutcome.CANCELLED,
                {},
            ),
            (
                [
                    SimpleNamespace(
                        type="response.failed",
                        error=SimpleNamespace(message="bad response"),
                    )
                ],
                StreamItemKind.STREAM_ERRORED,
                StreamTerminalOutcome.ERRORED,
                {"error": {"message": "bad response"}},
            ),
            (
                [SimpleNamespace(type="response.failed", error=OpaqueError())],
                StreamItemKind.STREAM_ERRORED,
                StreamTerminalOutcome.ERRORED,
                {"error": {"message": "opaque response"}},
            ),
            (
                [
                    SimpleNamespace(
                        type="response.failed",
                        response=SimpleNamespace(
                            id="resp-1",
                            status="failed",
                            error=None,
                        ),
                    )
                ],
                StreamItemKind.STREAM_ERRORED,
                StreamTerminalOutcome.ERRORED,
                {
                    "error": {
                        "code": "response_failed",
                        "message": "response failed",
                        "status": "failed",
                        "response_id": "resp-1",
                    }
                },
            ),
        )

        for events, kind, outcome, data in cases:
            with self.subTest(kind=kind):
                stream = self.mod.OpenAIStream(AsyncIter(events))

                items = [
                    item
                    async for item in stream.canonical_stream(
                        stream_session_id="responses-stream",
                        run_id="run-1",
                        turn_id="turn-1",
                    )
                ]

                self.assertEqual(
                    [item.kind for item in items],
                    [
                        StreamItemKind.STREAM_STARTED,
                        kind,
                        StreamItemKind.STREAM_CLOSED,
                    ],
                )
                self.assertEqual(items[1].data, data)
                self.assertIs(items[1].terminal_outcome, outcome)

    async def test_canonical_stream_maps_incomplete_response_to_error(self):
        cases = (
            (
                [
                    SimpleNamespace(
                        type="response.incomplete",
                        response=SimpleNamespace(
                            id="resp-1",
                            status="incomplete",
                            incomplete_details=SimpleNamespace(
                                reason="max_output_tokens"
                            ),
                            usage={"input_tokens": 3, "output_tokens": 4},
                        ),
                    )
                ],
                "response.incomplete",
            ),
            (
                [
                    {
                        "type": "response.completed",
                        "response": {
                            "id": "resp-2",
                            "status": "incomplete",
                            "incomplete_details": {"reason": "content_filter"},
                            "usage": {
                                "input_tokens": 5,
                                "output_tokens": 6,
                            },
                        },
                    }
                ],
                "response.completed",
            ),
        )

        for events, provider_event_type in cases:
            with self.subTest(provider_event_type=provider_event_type):
                stream = self.mod.OpenAIStream(AsyncIter(events))

                items = await _stream_items(stream)

                self.assertEqual(
                    [item.kind for item in items],
                    [
                        StreamItemKind.STREAM_STARTED,
                        StreamItemKind.USAGE_COMPLETED,
                        StreamItemKind.STREAM_ERRORED,
                        StreamItemKind.STREAM_CLOSED,
                    ],
                )
                self.assertEqual(
                    items[1].provider_event_type, provider_event_type
                )
                self.assertEqual(
                    items[2].provider_event_type, provider_event_type
                )
                self.assertIs(
                    items[2].terminal_outcome,
                    StreamTerminalOutcome.ERRORED,
                )
                data = items[2].data
                assert isinstance(data, dict)
                error = data["error"]
                assert isinstance(error, dict)
                self.assertEqual(error["code"], "response_incomplete")
                self.assertEqual(error["status"], "incomplete")
                self.assertIn("reason", error)
                self.assertIs(stream.usage, items[1].usage)

    async def test_canonical_stream_maps_malformed_responses_events_to_error(
        self,
    ):
        cases = (
            (SimpleNamespace(type=3), "type"),
            (
                SimpleNamespace(
                    type="response.output_text.delta", delta=object()
                ),
                "delta",
            ),
            (
                SimpleNamespace(
                    type="response.output_text.done", text=object()
                ),
                "text",
            ),
            (
                SimpleNamespace(
                    type="response.function_call_arguments.delta",
                    delta="{}",
                ),
                "id is missing",
            ),
            (
                SimpleNamespace(
                    type="response.function_call_arguments.delta",
                    item_id="",
                    delta="{}",
                ),
                "id",
            ),
            (
                SimpleNamespace(
                    type="response.output_item.added",
                    item=SimpleNamespace(
                        id=object(),
                        custom_tool_call=SimpleNamespace(id=object()),
                    ),
                ),
                "id",
            ),
            (
                SimpleNamespace(
                    type="response.output_item.added",
                    item=SimpleNamespace(
                        id=object(),
                        custom_tool_call=SimpleNamespace(id="call-1"),
                    ),
                ),
                "item id",
            ),
            (
                SimpleNamespace(
                    type="response.output_item.added",
                    item=SimpleNamespace(
                        custom_tool_call=SimpleNamespace(
                            id="call-1", name=object()
                        )
                    ),
                ),
                "name",
            ),
            (
                SimpleNamespace(
                    type="response.output_item.done",
                    item=SimpleNamespace(
                        type="function_call",
                        id="call-1",
                        arguments=object(),
                    ),
                ),
                "arguments",
            ),
        )

        for event, message in cases:
            with self.subTest(message=message):
                stream = self.mod.OpenAIStream(AsyncIter([event]))

                items = [
                    item
                    async for item in stream.canonical_stream(
                        stream_session_id="responses-stream",
                        run_id="run-1",
                        turn_id="turn-1",
                    )
                ]

                self.assertEqual(
                    [item.kind for item in items],
                    [
                        StreamItemKind.STREAM_STARTED,
                        StreamItemKind.STREAM_ERRORED,
                        StreamItemKind.STREAM_CLOSED,
                    ],
                )
                error_data = items[1].data
                assert isinstance(error_data, dict)
                self.assertEqual(error_data["error_type"], "ValueError")
                self.assertIn(message, error_data["message"])

    async def test_canonical_stream_preserves_malformed_response_payload(
        self,
    ):
        event = {
            "type": "response.output_text.delta",
            "delta": {"not": "text"},
            "response": {"id": "resp_1"},
        }
        stream = self.mod.OpenAIStream(AsyncIter([event]))

        items = [
            item
            async for item in stream.canonical_stream(
                stream_session_id="responses-stream",
                run_id="run-1",
                turn_id="turn-1",
            )
        ]

        self.assertTrue(
            all(isinstance(item, CanonicalStreamItem) for item in items)
        )
        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.STREAM_ERRORED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        error_item = items[1]
        error_data = error_item.data
        assert isinstance(error_data, dict)
        self.assertEqual(error_data["error_type"], "ValueError")
        self.assertIn("delta must be a string", error_data["message"])
        self.assertEqual(error_item.provider_payload, event)
        self.assertEqual(
            error_item.provider_event_type,
            "response.output_text.delta",
        )
        self.assertIs(
            error_item.terminal_outcome,
            StreamTerminalOutcome.ERRORED,
        )
        self.assertNotIn(
            StreamItemKind.ANSWER_DELTA,
            [item.kind for item in items],
        )

    async def test_canonical_stream_maps_duplicate_tool_done_to_error(self):
        events = [
            SimpleNamespace(
                type="response.output_item.done",
                item=SimpleNamespace(id="call-1"),
            ),
            SimpleNamespace(
                type="response.output_item.done",
                item=SimpleNamespace(id="call-1"),
            ),
        ]
        stream = self.mod.OpenAIStream(AsyncIter(events))

        items = [
            item
            async for item in stream.canonical_stream(
                stream_session_id="responses-stream",
                run_id="run-1",
                turn_id="turn-1",
            )
        ]

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.STREAM_ERRORED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        error_data = items[3].data
        assert isinstance(error_data, dict)
        self.assertEqual(error_data["error_type"], "ValueError")
        self.assertIn("already completed", error_data["message"])

    async def test_function_call_events(self):
        events = [
            SimpleNamespace(
                type="response.function_call_arguments.delta",
                call_id="c2",
                delta="{",
            ),
            SimpleNamespace(
                type="response.function_call_arguments.delta",
                id="c2",
                delta="}",
            ),
            SimpleNamespace(
                type="response.output_item.done", item=SimpleNamespace(id="c2")
            ),
            SimpleNamespace(
                type="response.output_item.done",
                item=SimpleNamespace(
                    type="function_call",
                    id="c3",
                    name="pkg__f",
                    arguments='{"p": 1}',
                ),
            ),
        ]
        stream = self.mod.OpenAIStream(AsyncIter(events))
        items = await _stream_items(stream)

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        accumulator = accumulate_canonical_stream_items(items)
        self.assertEqual(
            accumulator.tool_call_arguments,
            {"c2": "{}", "c3": '{"p": 1}'},
        )
        ready_items = [
            item
            for item in items
            if item.kind is StreamItemKind.TOOL_CALL_READY
        ]
        self.assertEqual(
            [
                (
                    item.correlation.tool_call_id,
                    item.data,
                )
                for item in ready_items
            ],
            [("c2", {"name": None}), ("c3", {"name": "pkg__f"})],
        )

    async def test_function_call_events_accept_item_id_deltas(self):
        events = [
            SimpleNamespace(
                type="response.output_item.added",
                item=SimpleNamespace(
                    id="item-4",
                    custom_tool_call=SimpleNamespace(
                        id="item-4", name="pkg__search"
                    ),
                ),
            ),
            SimpleNamespace(
                type="response.function_call_arguments.delta",
                item_id="item-4",
                delta='{"q"',
            ),
            SimpleNamespace(
                type="response.function_call_arguments.delta",
                item_id="item-4",
                delta=':"avalan"}',
            ),
            SimpleNamespace(
                type="response.output_item.done",
                item=SimpleNamespace(
                    type="function_call",
                    id="item-4",
                    call_id="call-4",
                    name="pkg__search",
                ),
            ),
        ]
        stream = self.mod.OpenAIStream(AsyncIter(events))

        items = await _stream_items(stream)

        self.assertEqual(
            [
                item.text_delta
                for item in items
                if item.kind is StreamItemKind.TOOL_CALL_ARGUMENT_DELTA
            ],
            ['{"q"', ':"avalan"}'],
        )
        self.assertEqual(
            {
                item.correlation.tool_call_id
                for item in items
                if item.channel is StreamChannel.TOOL_CALL
            },
            {"item-4"},
        )
        self.assertEqual(
            accumulate_canonical_stream_items(items).tool_call_arguments,
            {"item-4": '{"q":"avalan"}'},
        )
        ready = next(
            item
            for item in items
            if item.kind is StreamItemKind.TOOL_CALL_READY
        )
        self.assertEqual(ready.data, {"name": "pkg__search"})

    async def test_function_call_added_item_decodes_provider_call_name(self):
        provider_name = self.mod.TextGenerationVendor.encode_tool_name(
            "math.calculator"
        )
        events = [
            SimpleNamespace(
                type="response.output_item.added",
                item=SimpleNamespace(
                    type="function_call",
                    id="fc_1",
                    name=provider_name,
                ),
            ),
            SimpleNamespace(
                type="response.function_call_arguments.delta",
                item_id="fc_1",
                delta='{"expression"',
            ),
            SimpleNamespace(
                type="response.function_call_arguments.delta",
                item_id="fc_1",
                delta=':"4 + 6"}',
            ),
            SimpleNamespace(
                type="response.output_item.done",
                item=SimpleNamespace(type="function_call", id="fc_1"),
            ),
        ]
        stream = self.mod.OpenAIStream(AsyncIter(events))

        items = await _stream_items(stream)

        self.assertEqual(
            [
                item.text_delta
                for item in items
                if item.kind is StreamItemKind.TOOL_CALL_ARGUMENT_DELTA
            ],
            ['{"expression"', ':"4 + 6"}'],
        )
        self.assertEqual(
            accumulate_canonical_stream_items(items).tool_call_arguments,
            {"fc_1": '{"expression":"4 + 6"}'},
        )
        ready = next(
            item
            for item in items
            if item.kind is StreamItemKind.TOOL_CALL_READY
        )
        self.assertEqual(ready.correlation.tool_call_id, "fc_1")
        self.assertEqual(ready.data, {"name": "math.calculator"})

    async def test_function_call_added_item_uses_tool_name_policy(self):
        manager = _sanitized_policy_manager()
        events = [
            SimpleNamespace(
                type="response.output_item.added",
                item=SimpleNamespace(
                    type="function_call",
                    id="fc_1",
                    name="math_adder",
                ),
            ),
            SimpleNamespace(
                type="response.function_call_arguments.delta",
                item_id="fc_1",
                delta='{"a":1',
            ),
            SimpleNamespace(
                type="response.function_call_arguments.delta",
                item_id="fc_1",
                delta=',"b":2}',
            ),
            SimpleNamespace(
                type="response.output_item.done",
                item=SimpleNamespace(type="function_call", id="fc_1"),
            ),
        ]
        stream = self.mod.OpenAIStream(AsyncIter(events), tool=manager)

        items = await _stream_items(stream)

        self.assertEqual(
            accumulate_canonical_stream_items(items).tool_call_arguments,
            {"fc_1": '{"a":1,"b":2}'},
        )
        ready = next(
            item
            for item in items
            if item.kind is StreamItemKind.TOOL_CALL_READY
        )
        self.assertEqual(ready.data, {"name": "math.adder"})

    async def test_function_call_name_falls_back_when_policy_rejects(self):
        manager = _sanitized_policy_manager()
        events = [
            SimpleNamespace(
                type="response.output_item.added",
                item=SimpleNamespace(
                    type="function_call",
                    id="fc_1",
                    name="bad.name",
                ),
            ),
            SimpleNamespace(
                type="response.output_item.done",
                item=SimpleNamespace(type="function_call", id="fc_1"),
            ),
        ]
        stream = self.mod.OpenAIStream(AsyncIter(events), tool=manager)

        items = await _stream_items(stream)

        ready = next(
            item
            for item in items
            if item.kind is StreamItemKind.TOOL_CALL_READY
        )
        self.assertEqual(ready.data, {"name": "bad.name"})

    async def test_stream_client_passes_tool_manager_to_stream(self):
        stream_instance = AsyncIter([])
        self.openai_stub.AsyncOpenAI.return_value.responses.create = AsyncMock(
            return_value=stream_instance
        )
        client = self.mod.OpenAIClient(api_key="k", base_url="b")
        manager = _sanitized_policy_manager()

        with patch.object(self.mod, "OpenAIStream") as StreamMock:
            result = await client("m", [], tool=manager)

        self.assertIs(result, StreamMock.return_value)
        StreamMock.assert_called_once()
        self.assertIs(StreamMock.call_args.kwargs["tool"], manager)

    async def test_function_call_events_reject_invalid_delta_id(self):
        stream = self.mod.OpenAIStream(
            AsyncIter(
                [
                    SimpleNamespace(
                        type="response.function_call_arguments.delta",
                        item_id="",
                        delta="{}",
                    )
                ]
            )
        )

        items = await _stream_items(stream)

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.STREAM_ERRORED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        error_data = items[1].data
        assert isinstance(error_data, dict)
        self.assertIn("id", error_data["message"])

    async def test_provider_argument_deltas_match_serialized_call(self):
        fixture = loads(
            (FIXTURES / "provider_openai_argument_deltas.json").read_text(
                encoding="utf-8"
            )
        )
        call_id = fixture["call_id"]
        events = [
            SimpleNamespace(
                type="response.output_item.added",
                item=SimpleNamespace(
                    id=call_id,
                    custom_tool_call=SimpleNamespace(
                        id=call_id,
                        name=fixture["provider_name"],
                    ),
                ),
            ),
            *[
                SimpleNamespace(
                    type="response.function_call_arguments.delta",
                    id=call_id,
                    delta=delta,
                )
                for delta in fixture["argument_deltas"]
            ],
            SimpleNamespace(
                type="response.output_item.done",
                item=SimpleNamespace(id=call_id),
            ),
            SimpleNamespace(
                type="response.output_text.delta",
                delta=fixture["assistant_after"],
            ),
        ]

        stream = self.mod.OpenAIStream(AsyncIter(events))
        outputs = []
        async for output in stream:
            outputs.append(output)

        delta_outputs = [
            output
            for output in outputs
            if output.kind is StreamItemKind.TOOL_CALL_ARGUMENT_DELTA
        ]
        self.assertEqual(len(delta_outputs), len(fixture["argument_deltas"]))
        self.assertTrue(
            all(
                output.correlation.tool_call_id == call_id
                for output in delta_outputs
            )
        )
        self.assertEqual(
            [output.text_delta for output in delta_outputs],
            fixture["argument_deltas"],
        )

        ready = next(
            output
            for output in outputs
            if output.kind is StreamItemKind.TOOL_CALL_READY
        )
        self.assertEqual(ready.correlation.tool_call_id, call_id)
        self.assertEqual(
            ready.data,
            {"name": fixture["provider_name"]},
        )

        done = next(
            output
            for output in outputs
            if output.kind is StreamItemKind.TOOL_CALL_DONE
        )
        self.assertEqual(done.correlation.tool_call_id, call_id)
        self.assertEqual(
            loads("".join(fixture["argument_deltas"])),
            {"city": "Paris", "unit": "c"},
        )
        self.assertEqual(
            accumulate_canonical_stream_items(outputs).answer_text,
            fixture["assistant_after"],
        )

    async def test_generation_settings_and_tools(self):
        stream_instance = AsyncIter([])
        self.openai_stub.AsyncOpenAI.return_value.responses.create = AsyncMock(
            return_value=stream_instance
        )
        client = self.mod.OpenAIClient(api_key="k", base_url="b")
        client._template_messages = MagicMock(return_value=[{"c": 1}])
        tool = MagicMock()
        tool.json_schemas.return_value = [
            {"type": "function", "function": {"name": "pkg.func"}}
        ]
        settings = GenerationSettings(
            temperature=0.5,
            top_p=0.8,
            max_new_tokens=10,
            stop_strings=["stop"],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "result",
                    "schema": {"type": "object"},
                    "strict": True,
                },
            },
        )
        await client(
            "m",
            [],
            settings=settings,
            tool=tool,
        )
        self.openai_stub.AsyncOpenAI.return_value.responses.create.assert_awaited_once_with(
            extra_headers={
                "X-Title": "Avalan",
                "HTTP-Referer": "https://github.com/avalan-ai/avalan",
            },
            model="m",
            input=[{"c": 1}],
            store=False,
            stream=True,
            max_output_tokens=10,
            temperature=0.5,
            top_p=0.8,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "result",
                    "schema": {"type": "object"},
                    "strict": True,
                },
                "stop": ["stop"],
            },
            tools=[{"type": "function", "name": "avl_cGtnLmZ1bmM"}],
        )

    def test_generation_settings_rejects_invalid_openai_retry_values(self):
        cases: list[dict[str, Any]] = [
            {"openai_max_retries": -1},
            {"openai_max_retries": 1.5},
            {"openai_response_failed_retries": -1},
            {"openai_response_failed_retries": 1.5},
            {"openai_response_failed_retry_delay_seconds": -0.1},
            {"openai_response_failed_retry_delay_seconds": False},
            {"openai_timeout_seconds": 0},
            {"openai_timeout_seconds": -0.1},
            {"openai_timeout_seconds": False},
        ]

        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(AssertionError):
                    GenerationSettings(**kwargs)

    async def test_azure_responses_payload_uses_text_format(self):
        response = SimpleNamespace(
            output=[SimpleNamespace(content=[SimpleNamespace(text="ok")])]
        )
        create_mock = AsyncMock(return_value=response)
        self.openai_stub.AsyncOpenAI.return_value.responses.create = (
            create_mock
        )
        client = self.mod.OpenAIClient(
            api_key="key",
            base_url="https://tenant.openai.azure.com/openai/v1/",
        )
        client._template_messages = MagicMock(return_value=[{"c": 1}])
        settings = GenerationSettings(
            max_new_tokens=20,
            temperature=0.3,
            top_p=0.4,
            stop_strings="END",
            response_format={"type": "json_object"},
            reasoning=ReasoningSettings(effort=ReasoningEffort.HIGH),
        )

        await client(
            "claims-extractor-prod",
            [],
            settings=settings,
            use_async_generator=False,
        )

        create_mock.assert_awaited_once_with(
            extra_headers={
                "X-Title": "Avalan",
                "HTTP-Referer": "https://github.com/avalan-ai/avalan",
            },
            model="claims-extractor-prod",
            input=[{"c": 1}],
            store=False,
            stream=False,
            max_output_tokens=20,
            text={"format": {"type": "json_object"}, "stop": "END"},
            reasoning={"effort": "high"},
            include=["reasoning.encrypted_content"],
        )

    async def test_azure_legacy_api_version_uses_extra_query(self):
        response = SimpleNamespace(output=[])
        create_mock = AsyncMock(return_value=response)
        self.openai_stub.AsyncOpenAI.return_value.responses.create = (
            create_mock
        )
        client = self.mod.OpenAIClient(
            api_key="key",
            base_url="https://tenant.openai.azure.com/openai/deployments/dep",
            azure_api_version="2025-04-01-preview",
        )
        client._template_messages = MagicMock(return_value=[])

        await client("dep", [], use_async_generator=False)

        self.assertEqual(
            create_mock.await_args.kwargs["extra_query"],
            {"api-version": "2025-04-01-preview"},
        )

    async def test_rejects_invalid_responses_options_before_provider_call(
        self,
    ):
        create_mock = AsyncMock()
        self.openai_stub.AsyncOpenAI.return_value.responses.create = (
            create_mock
        )
        client = self.mod.OpenAIClient(api_key="key", base_url="url")
        invalid_settings = GenerationSettings(
            response_format={
                "type": "json_schema",
                "json_schema": {"schema": {"type": "object"}},
                "schema": {"type": "object"},
            }
        )

        with self.assertRaisesRegex(AssertionError, "ambiguous"):
            await client("model", [], settings=invalid_settings)

        create_mock.assert_not_awaited()

    async def test_rejects_invalid_prompt_cache_retention_before_provider_call(
        self,
    ):
        create_mock = AsyncMock()
        self.openai_stub.AsyncOpenAI.return_value.responses.create = (
            create_mock
        )
        client = self.mod.OpenAIClient(api_key="key", base_url="url")

        with self.assertRaisesRegex(AssertionError, "not supported"):
            await client(
                "model",
                [],
                settings=GenerationSettings(
                    prompt_cache_retention="forever",
                ),
            )
        with self.assertRaisesRegex(AssertionError, "must be a string"):
            await client(
                "model",
                [],
                settings=GenerationSettings(
                    prompt_cache_retention=object(),  # type: ignore[arg-type]
                ),
            )

        create_mock.assert_not_awaited()

    async def test_rejects_invalid_reasoning_effort(self):
        create_mock = AsyncMock()
        self.openai_stub.AsyncOpenAI.return_value.responses.create = (
            create_mock
        )
        client = self.mod.OpenAIClient(
            api_key="key",
            base_url="https://tenant.openai.azure.com/openai/v1/",
        )
        settings = GenerationSettings(
            reasoning=ReasoningSettings(
                effort="ultra",  # type: ignore[arg-type]
            )
        )

        with self.assertRaisesRegex(AssertionError, "reasoning effort"):
            await client("deployment", [], settings=settings)

        create_mock.assert_not_awaited()

    async def test_gpt_55_reasoning_xhigh_effort_is_forwarded(self):
        response = SimpleNamespace(
            output=[SimpleNamespace(content=[SimpleNamespace(text="x")])]
        )
        create_mock = AsyncMock(return_value=response)
        self.openai_stub.AsyncOpenAI.return_value.responses.create = (
            create_mock
        )
        client = self.mod.OpenAIClient(api_key="key", base_url="url")
        settings = GenerationSettings(
            reasoning=ReasoningSettings(effort=ReasoningEffort.XHIGH)
        )

        await client(
            "gpt-5.5",
            [Message(role=MessageRole.USER, content="hi")],
            settings,
            use_async_generator=False,
        )

        kwargs = create_mock.await_args.kwargs
        self.assertEqual(kwargs["reasoning"], {"effort": "xhigh"})
        self.assertEqual(kwargs["include"], ["reasoning.encrypted_content"])

    def test_response_text_format_normalizes_supported_shapes(self):
        self.assertEqual(
            self.mod.OpenAIClient._prompt_cache_retention_config(
                GenerationSettings(prompt_cache_retention="in_memory")
            ),
            "in_memory",
        )
        self.assertEqual(
            self.mod.OpenAIClient._response_text_format({"type": "text"}),
            {"type": "text"},
        )
        self.assertEqual(
            self.mod.OpenAIClient._response_text_format(
                {
                    "type": "json_schema",
                    "name": "document",
                    "schema": {"type": "object"},
                    "strict": False,
                }
            ),
            {
                "type": "json_schema",
                "name": "document",
                "schema": {"type": "object"},
                "strict": False,
            },
        )
        with self.assertRaisesRegex(AssertionError, "not supported"):
            self.mod.OpenAIClient._response_text_format({"type": "xml"})
        with self.assertRaises(AssertionError):
            self.mod.OpenAIClient._response_text_format(
                {
                    "type": "json_schema",
                    "schema": {"type": "object"},
                }
            )
        with self.assertRaises(AssertionError):
            self.mod.OpenAIClient._response_text_format(
                {
                    "type": "json_schema",
                    "json_schema": {"name": "bad"},
                }
            )

    def test_azure_configuration_validation(self):
        self.assertFalse(self.mod.OpenAIClient._is_azure_base_url(None))

        with self.assertRaisesRegex(AssertionError, "/openai/v1/"):
            self.mod.OpenAIClient(
                api_key="key",
                base_url=(
                    "https://tenant.openai.azure.com/openai/deployments/dep"
                ),
            )

        try:
            self.mod.OpenAIClient(
                api_key="key",
                base_url=(
                    "https://tenant.openai.azure.com/openai/v1/"
                    "?api-version=super-secret"
                ),
            )
        except AssertionError as exc:
            self.assertIn("query parameters", str(exc))
            self.assertNotIn("super-secret", str(exc))

        with self.assertRaisesRegex(AssertionError, "api-key"):
            self.mod.OpenAIClient(
                api_key=None,
                base_url="https://tenant.openai.azure.com/openai/v1/",
            )

        with self.assertRaisesRegex(AssertionError, "only supported"):
            self.mod.OpenAIClient(
                api_key="key",
                base_url="https://api.openai.com/v1",
                azure_api_version="2025-04-01-preview",
            )


class VendorClientsTestCase(TestCase):
    def setUp(self):
        self.openai_stub, self.patch = patch_openai_imports()
        importlib.reload(
            importlib.import_module("avalan.model.nlp.text.vendor.openai")
        )

    def tearDown(self):
        self.patch.stop()

    def test_openrouter_client_and_model(self):
        mod = importlib.import_module(
            "avalan.model.nlp.text.vendor.openrouter"
        )
        importlib.reload(mod)
        self.openai_stub.AsyncOpenAI.reset_mock()
        client = mod.OpenRouterClient(api_key="k", base_url=None)
        self.openai_stub.AsyncOpenAI.assert_called_once_with(
            base_url="https://openrouter.ai/api/v1", api_key="k"
        )
        self.assertIsInstance(client, mod.OpenRouterClient)
        self.assertEqual(client._usage_provider_family, "openai_compatible")
        with patch.object(mod, "OpenRouterClient") as ClientMock:
            settings = TransformerEngineSettings(
                auto_load_model=False,
                auto_load_tokenizer=False,
                access_token="t",
                base_url="b",
            )
            model = mod.OpenRouterModel("m", settings)
            loaded = model._load_model()
        ClientMock.assert_called_once_with(base_url="b", api_key="t")
        self.assertIs(loaded, ClientMock.return_value)

    def test_together_client_and_model(self):
        mod = importlib.import_module("avalan.model.nlp.text.vendor.together")
        importlib.reload(mod)
        self.openai_stub.AsyncOpenAI.reset_mock()
        mod.TogetherClient(api_key="k", base_url=None)
        self.openai_stub.AsyncOpenAI.assert_called_once_with(
            base_url="https://api.together.xyz/v1", api_key="k"
        )
        with patch.object(mod, "TogetherClient") as ClientMock:
            settings = TransformerEngineSettings(
                auto_load_model=False,
                auto_load_tokenizer=False,
                access_token="t",
                base_url="b",
            )
            model = mod.TogetherModel("m", settings)
            loaded = model._load_model()
        ClientMock.assert_called_once_with(base_url="b", api_key="t")
        self.assertIs(loaded, ClientMock.return_value)


class NonStreamingResponseTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.openai_stub, self.patch = patch_openai_imports()
        importlib.reload(
            importlib.import_module("avalan.model.nlp.text.vendor.openai")
        )
        self.mod = importlib.import_module(
            "avalan.model.nlp.text.vendor.openai"
        )

    def tearDown(self):
        self.patch.stop()

    async def test_response_single_stream(self):
        resp = SimpleNamespace(
            output=[SimpleNamespace(content=[SimpleNamespace(text="ok")])],
            usage=SimpleNamespace(input_tokens=1),
        )
        self.openai_stub.AsyncOpenAI.return_value.responses.create = AsyncMock(
            return_value=resp
        )
        settings = TransformerEngineSettings(
            auto_load_model=False,
            auto_load_tokenizer=False,
            access_token="tok",
            base_url="url",
        )
        model = self.mod.OpenAIModel("m", settings)
        model._model = model._load_model()
        gen = GenerationSettings(use_async_generator=False)
        response = await model("hi", settings=gen)
        self.openai_stub.AsyncOpenAI.assert_called_once_with(
            base_url="url", api_key="tok"
        )
        self.openai_stub.AsyncOpenAI.return_value.responses.create.assert_awaited_once_with(
            extra_headers={
                "X-Title": "Avalan",
                "HTTP-Referer": "https://github.com/avalan-ai/avalan",
            },
            model="m",
            input=[{"role": "user", "content": "hi"}],
            store=False,
            stream=False,
            temperature=1.0,
            top_p=1.0,
        )
        from avalan.model.stream import TextGenerationSingleStream

        self.assertIsInstance(response._output_fn, TextGenerationSingleStream)
        self.assertFalse(response._use_async_generator)
        self.assertEqual(response.usage.input_tokens, 1)
        self.assertEqual(response.provider_family, "openai")
        observation = usage_observation_from_response(response)
        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(observation.metadata, {"provider_family": "openai"})
        self.assertEqual(await response.to_str(), "ok")

    async def test_azure_response_usage_preserves_safe_metadata(self):
        usage = SimpleNamespace(
            input_tokens=11,
            input_tokens_details=SimpleNamespace(cached_tokens=4),
            output_tokens=7,
            output_tokens_details=SimpleNamespace(reasoning_tokens=3),
            total_tokens=18,
            model="private-deployment-name",
            response_id="private-response-id",
        )
        resp = SimpleNamespace(
            output=[
                SimpleNamespace(content=[SimpleNamespace(text="azure ok")])
            ],
            usage=usage,
        )
        self.openai_stub.AsyncOpenAI.return_value.responses.create = AsyncMock(
            return_value=resp
        )
        client = self.mod.OpenAIClient(
            api_key="key",
            base_url="https://acct.openai.azure.com/openai/v1/",
        )
        response = await client(
            "private-deployment-name",
            [Message(role=MessageRole.USER, content="hi")],
            use_async_generator=False,
        )

        observation = usage_observation_from_response(response)

        self.assertEqual(response.provider_family, "azure_openai")
        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(
            observation.metadata, {"provider_family": "azure_openai"}
        )
        self.assertEqual(observation.totals.input_tokens, 11)
        self.assertEqual(observation.totals.cached_input_tokens, 4)
        self.assertIsNone(observation.totals.cache_creation_input_tokens)
        self.assertEqual(observation.totals.output_tokens, 7)
        self.assertEqual(observation.totals.reasoning_tokens, 3)
        self.assertEqual(observation.totals.total_tokens, 18)
        self.assertNotIn("private-deployment-name", str(observation))
        self.assertNotIn("private-response-id", str(observation))


class TemplateMessagesFormatTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.openai_stub, self.patch = patch_openai_imports()
        importlib.reload(
            importlib.import_module("avalan.model.nlp.text.vendor.openai")
        )
        self.mod = importlib.import_module(
            "avalan.model.nlp.text.vendor.openai"
        )

    def tearDown(self):
        self.patch.stop()

    async def _assert_messages(self, content, expected_content):
        resp = SimpleNamespace(
            output=[SimpleNamespace(content=[SimpleNamespace(text="x")])]
        )
        self.openai_stub.AsyncOpenAI.return_value.responses.create = AsyncMock(
            return_value=resp
        )
        client = self.mod.OpenAIClient(api_key="key", base_url="url")
        message = Message(role=MessageRole.USER, content=content)
        await client("model", [message], use_async_generator=False)
        create_mock = (
            self.openai_stub.AsyncOpenAI.return_value.responses.create
        )
        create_mock.assert_awaited_once()
        kwargs = create_mock.await_args.kwargs
        self.assertEqual(
            kwargs["input"], [{"role": "user", "content": expected_content}]
        )

    async def test_string_message(self):
        await self._assert_messages("hi", "hi")

    async def test_text_message_content(self):
        content = MessageContentText(type="text", text="hi")
        await self._assert_messages(content, "hi")

    async def test_image_message_content(self):
        content = MessageContentImage(type="image_url", image_url={"url": "u"})
        await self._assert_messages(
            content, [{"type": "input_image", "image_url": "u"}]
        )

    async def test_image_message_content_from_file_id(self):
        content = MessageContentImage(
            type="image_url", image_url={"file_id": "file-img"}
        )
        await self._assert_messages(
            content, [{"type": "input_image", "file_id": "file-img"}]
        )

    async def test_image_message_content_from_base64(self):
        content = MessageContentImage(
            type="image_url",
            image_url={
                "data": "YWJj",
                "detail": "high",
                "mime_type": "image/jpeg",
            },
        )
        await self._assert_messages(
            content,
            [
                {
                    "type": "input_image",
                    "image_url": "data:image/jpeg;base64,YWJj",
                    "detail": "high",
                }
            ],
        )

    async def test_image_message_content_preserves_data_url(self):
        content = MessageContentImage(
            type="image_url",
            image_url={
                "data": "data:image/png;base64,YWJj",
                "mime_type": "image/png",
            },
        )
        await self._assert_messages(
            content,
            [
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,YWJj",
                }
            ],
        )

    async def test_image_message_content_rejects_non_image_mime(self):
        resp = SimpleNamespace(
            output=[SimpleNamespace(content=[SimpleNamespace(text="x")])]
        )
        create_mock = AsyncMock(return_value=resp)
        self.openai_stub.AsyncOpenAI.return_value.responses.create = (
            create_mock
        )
        client = self.mod.OpenAIClient(api_key="key", base_url="url")
        message = Message(
            role=MessageRole.USER,
            content=MessageContentImage(
                type="image_url",
                image_url={"data": "YWJj", "mime_type": "application/pdf"},
            ),
        )

        with self.assertRaisesRegex(AssertionError, "image MIME type"):
            await client("model", [message], use_async_generator=False)

        create_mock.assert_not_awaited()

    async def test_image_message_content_rejects_invalid_mime_type(self):
        resp = SimpleNamespace(
            output=[SimpleNamespace(content=[SimpleNamespace(text="x")])]
        )
        create_mock = AsyncMock(return_value=resp)
        self.openai_stub.AsyncOpenAI.return_value.responses.create = (
            create_mock
        )
        client = self.mod.OpenAIClient(api_key="key", base_url="url")
        message = Message(
            role=MessageRole.USER,
            content=MessageContentImage(
                type="image_url",
                image_url={"data": "YWJj", "mime_type": object()},
            ),
        )

        with self.assertRaisesRegex(AssertionError, "image MIME type"):
            await client("model", [message], use_async_generator=False)

        create_mock.assert_not_awaited()

    async def test_mixed_message_content(self):
        content = [
            MessageContentText(type="text", text="hi"),
            MessageContentImage(type="image_url", image_url={"url": "u"}),
        ]
        await self._assert_messages(
            content,
            [
                {"type": "input_text", "text": "hi"},
                {"type": "input_image", "image_url": "u"},
            ],
        )

    async def test_file_message_content(self):
        content = MessageContentFile(
            type="file", file={"file_url": "https://example.com/a.pdf"}
        )
        await self._assert_messages(
            content,
            [
                {
                    "type": "input_file",
                    "file_url": "https://example.com/a.pdf",
                }
            ],
        )

    async def test_mixed_message_content_with_file_data(self):
        content = [
            MessageContentText(type="text", text="hi"),
            MessageContentFile(
                type="file",
                file={"file_data": "YWJj", "filename": "report.pdf"},
            ),
            MessageContentText(
                type="text",
                text=(
                    "Attached files available to tools:\n"
                    "Use these path values as tool arguments.\n"
                    '- "attachment/report.pdf"'
                ),
            ),
        ]
        await self._assert_messages(
            content,
            [
                {"type": "input_text", "text": "hi"},
                {
                    "type": "input_file",
                    "file_data": "data:application/pdf;base64,YWJj",
                    "filename": "report.pdf",
                },
                {
                    "type": "input_text",
                    "text": (
                        "Attached files available to tools:\n"
                        "Use these path values as tool arguments.\n"
                        '- "attachment/report.pdf"'
                    ),
                },
            ],
        )

    async def test_mixed_message_content_with_pdf_file_data_uses_data_url(
        self,
    ):
        content = [
            MessageContentText(type="text", text="hi"),
            MessageContentFile(
                type="file",
                file={
                    "file_data": "YWJj",
                    "filename": "report.pdf",
                    "mime_type": "application/pdf",
                },
            ),
        ]
        await self._assert_messages(
            content,
            [
                {"type": "input_text", "text": "hi"},
                {
                    "type": "input_file",
                    "file_data": "data:application/pdf;base64,YWJj",
                    "filename": "report.pdf",
                },
            ],
        )

    async def test_pdf_file_data_uses_data_url_when_inferred_from_filename(
        self,
    ):
        content = MessageContentFile(
            type="file",
            file={"file_data": "YWJj", "filename": "report.pdf"},
        )
        await self._assert_messages(
            content,
            [
                {
                    "type": "input_file",
                    "file_data": "data:application/pdf;base64,YWJj",
                    "filename": "report.pdf",
                }
            ],
        )

    async def test_reasoning_effort_is_forwarded(self):
        response = SimpleNamespace(
            output=[SimpleNamespace(content=[SimpleNamespace(text="x")])]
        )
        create_mock = AsyncMock(return_value=response)
        self.openai_stub.AsyncOpenAI.return_value.responses.create = (
            create_mock
        )
        client = self.mod.OpenAIClient(api_key="key", base_url="url")
        settings = GenerationSettings(
            reasoning=ReasoningSettings(effort=ReasoningEffort.XHIGH)
        )

        await client(
            "model",
            [Message(role=MessageRole.USER, content="hi")],
            settings,
            use_async_generator=False,
        )

        kwargs = create_mock.await_args.kwargs
        self.assertEqual(kwargs["reasoning"], {"effort": "xhigh"})

    def test_helper_methods_cover_additional_reasoning_and_file_paths(self):
        self.assertEqual(
            self.mod.OpenAIClient._reasoning_config(
                GenerationSettings(
                    reasoning=ReasoningSettings(effort=ReasoningEffort.MAX)
                )
            ),
            {"effort": "xhigh"},
        )
        self.assertEqual(
            self.mod.OpenAIClient._content_block(
                {
                    "type": "unknown",
                    "value": 1,
                }
            ),
            {"type": "unknown", "value": 1},
        )
        self.assertEqual(
            self.mod.OpenAIClient._file_block(
                {
                    "file_id": "file-1",
                    "filename": "report.pdf",
                }
            ),
            {
                "type": "input_file",
                "file_id": "file-1",
                "filename": "report.pdf",
            },
        )
        with self.assertRaises(AssertionError):
            self.mod.OpenAIClient._file_block({})
        with self.assertRaises(AssertionError):
            self.mod.OpenAIClient._image_block({})
        self.assertEqual(
            self.mod.OpenAIClient._non_stream_response_content(
                {"output": None}
            ),
            "",
        )

    def test_template_messages_skips_entries_without_content_key(self):
        client = self.mod.OpenAIClient(api_key="key", base_url="url")

        with patch.object(
            self.mod.TextGenerationVendor,
            "_template_messages",
            return_value=[{"role": "assistant"}],
        ):
            templated = client._template_messages([])

        self.assertEqual(templated, [{"role": "assistant"}])

    async def test_non_stream_tool_call_output(self):
        response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="output_text",
                    content=[SimpleNamespace(text="hello ")],
                ),
                SimpleNamespace(
                    type="tool_call",
                    call=SimpleNamespace(
                        id="call1",
                        function=SimpleNamespace(
                            name="pkg__tool", arguments='{"a":1}'
                        ),
                    ),
                ),
            ]
        )

        create_mock = AsyncMock(return_value=response)
        self.openai_stub.AsyncOpenAI.return_value.responses.create = (
            create_mock
        )

        with patch.object(
            self.mod.TextGenerationVendor,
            "build_tool_call_text",
            return_value="<tool_call />",
        ) as build_token:
            client = self.mod.OpenAIClient(api_key="key", base_url="url")
            message = Message(role=MessageRole.USER, content="hi")
            stream = await client(
                "model", [message], use_async_generator=False
            )

        from avalan.model.stream import TextGenerationSingleStream

        self.assertIsInstance(stream, TextGenerationSingleStream)
        self.assertEqual(stream.content, "hello <tool_call />")
        build_token.assert_called_once_with("call1", "pkg__tool", '{"a":1}')


class TemplateAndToolSchemaTestCase(TestCase):
    def setUp(self):
        self.openai_stub, self.patch = patch_openai_imports()
        importlib.reload(
            importlib.import_module("avalan.model.nlp.text.vendor.openai")
        )
        self.mod = importlib.import_module(
            "avalan.model.nlp.text.vendor.openai"
        )

    def tearDown(self):
        self.patch.stop()

    def test_tool_schemas_none(self):
        tool = MagicMock()
        tool.json_schemas.return_value = None
        self.assertIsNone(self.mod.OpenAIClient._tool_schemas(tool))
        tool.json_schemas.return_value = [{"type": "x"}]
        self.assertEqual(self.mod.OpenAIClient._tool_schemas(tool), [])

    def test_tool_schemas_use_tool_manager_name_policy(self):
        schemas = self.mod.OpenAIClient._tool_schemas(
            _sanitized_policy_manager()
        )

        self.assertIsNotNone(schemas)
        assert schemas is not None
        self.assertEqual(schemas[0]["name"], "math_adder")

    def test_template_messages_tool_results(self):
        client = self.mod.OpenAIClient(api_key="k", base_url="b")

        @dataclass
        class R:
            v: int

        call1 = ToolCall(id="c1", name="pkg.func", arguments={"a": 1})
        result1 = ToolCallResult(
            id="c1",
            name="pkg.func",
            arguments={"a": 1},
            call=call1,
            result=R(v=2),
        )
        msg1 = Message(role=MessageRole.TOOL, tool_call_result=result1)

        call2 = ToolCall(id="c2", name="pkg.func2")
        result2 = ToolCallResult(
            id="c2", name="pkg.func2", call=call2, result={"x": 3}
        )
        msg2 = Message(role=MessageRole.TOOL, tool_call_result=result2)

        templated = client._template_messages([msg1, msg2])
        self.assertEqual(
            templated,
            [
                {
                    "type": "function_call",
                    "name": "avl_cGtnLmZ1bmM",
                    "call_id": "c1",
                    "arguments": '{"a": 1}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "c1",
                    "output": '{"v": 2}',
                },
                {
                    "type": "function_call",
                    "name": "avl_cGtnLmZ1bmMy",
                    "call_id": "c2",
                    "arguments": "null",
                },
                {
                    "type": "function_call_output",
                    "call_id": "c2",
                    "output": '{"x": 3}',
                },
            ],
        )

    def test_has_function_call_context(self):
        self.assertFalse(
            self.mod.OpenAIClient._has_function_call_context(
                [{"role": "user", "content": "hi"}]
            )
        )
        self.assertFalse(
            self.mod.OpenAIClient._has_function_call_context(
                ["raw", {"role": "user", "content": "hi"}]  # type: ignore[arg-type]
            )
        )
        self.assertTrue(
            self.mod.OpenAIClient._has_function_call_context(
                [
                    {"role": "user", "content": "hi"},
                    {"type": "function_call_output", "call_id": "c1"},
                ]
            )
        )

    def test_template_messages_ignores_stateless_items_without_tool_outputs(
        self,
    ):
        client = self.mod.OpenAIClient(api_key="k", base_url="b")
        client._record_stateless_response_item(
            {
                "type": "reasoning",
                "id": "rs_1",
                "encrypted_content": "encrypted",
            }
        )

        templated = client._template_messages(
            [Message(role=MessageRole.USER, content="hi")]
        )

        self.assertEqual(templated, [{"role": "user", "content": "hi"}])

    def test_template_messages_replays_stateless_reasoning_items(self):
        client = self.mod.OpenAIClient(api_key="k", base_url="b")
        reasoning_item = {
            "type": "reasoning",
            "id": "rs_1",
            "encrypted_content": "encrypted",
        }
        call_item = {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "rg",
            "arguments": '{"pattern": "needle"}',
        }
        client._record_stateless_response_item(reasoning_item)
        client._record_stateless_response_item(call_item)
        call = ToolCall(
            id="call_1",
            name="shell.rg",
            arguments={"pattern": "needle"},
        )
        result = ToolCallResult(
            id="result_1",
            name="shell.rg",
            arguments=call.arguments,
            call=call,
            result="match",
        )

        templated = client._template_messages(
            [
                Message(role=MessageRole.USER, content="search"),
                Message(role=MessageRole.TOOL, tool_call_result=result),
            ]
        )

        self.assertEqual(
            templated,
            [
                {"role": "user", "content": "search"},
                {
                    "type": "reasoning",
                    "encrypted_content": "encrypted",
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "rg",
                    "arguments": '{"pattern": "needle"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": '"match"',
                },
            ],
        )

    def test_template_messages_omits_stateless_calls_without_outputs(self):
        client = self.mod.OpenAIClient(api_key="k", base_url="b")
        reasoning_item = {
            "type": "reasoning",
            "id": "rs_1",
            "encrypted_content": "encrypted",
        }
        missing_call_item = {
            "type": "function_call",
            "id": "fc_missing",
            "call_id": "call_missing",
            "name": "rg",
            "arguments": "{}",
        }
        call_item_a = {
            "type": "function_call",
            "id": "fc_a",
            "call_id": "call_a",
            "name": "rg",
            "arguments": '{"pattern": "a"}',
        }
        call_item_b = {
            "type": "function_call",
            "id": "fc_b",
            "call_id": "call_b",
            "name": "rg",
            "arguments": '{"pattern": "b"}',
        }
        client._record_stateless_response_item(reasoning_item)
        client._record_stateless_response_item(missing_call_item)
        client._record_stateless_response_item(call_item_a)
        client._record_stateless_response_item(call_item_b)
        call_a = ToolCall(
            id="call_a",
            name="shell.rg",
            arguments={"pattern": "a"},
        )
        call_b = ToolCall(
            id="call_b",
            name="shell.rg",
            arguments={"pattern": "b"},
        )
        result_a = ToolCallResult(
            id="result_a",
            name="shell.rg",
            arguments=call_a.arguments,
            call=call_a,
            result="match a",
        )
        result_b = ToolCallResult(
            id="result_b",
            name="shell.rg",
            arguments=call_b.arguments,
            call=call_b,
            result="match b",
        )

        templated = client._template_messages(
            [
                Message(role=MessageRole.USER, content="search"),
                Message(role=MessageRole.TOOL, tool_call_result=result_b),
                Message(role=MessageRole.TOOL, tool_call_result=result_a),
            ]
        )

        self.assertEqual(
            templated,
            [
                {"role": "user", "content": "search"},
                {
                    "type": "reasoning",
                    "encrypted_content": "encrypted",
                },
                {
                    "type": "function_call",
                    "call_id": "call_a",
                    "name": "rg",
                    "arguments": '{"pattern": "a"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_a",
                    "output": '"match a"',
                },
                {
                    "type": "function_call",
                    "call_id": "call_b",
                    "name": "rg",
                    "arguments": '{"pattern": "b"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_b",
                    "output": '"match b"',
                },
            ],
        )

    def test_template_messages_falls_back_after_unmatched_stateless_items(
        self,
    ):
        client = self.mod.OpenAIClient(api_key="k", base_url="b")
        client._record_stateless_response_item(
            {
                "type": "message",
                "id": "msg_1",
            }
        )
        client._record_stateless_response_item(
            {
                "type": "function_call",
                "name": "rg",
                "arguments": "{}",
            }
        )
        call = ToolCall(id="call_1", name="shell.rg", arguments={})
        result = ToolCallResult(
            id="result_1",
            name="shell.rg",
            arguments=call.arguments,
            call=call,
            result="match",
        )

        templated = client._template_messages(
            [Message(role=MessageRole.TOOL, tool_call_result=result)]
        )

        self.assertEqual(
            templated,
            [
                {
                    "type": "function_call",
                    "name": "avl_c2hlbGwucmc",
                    "call_id": "call_1",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": '"match"',
                },
            ],
        )

    def test_template_messages_use_tool_manager_name_policy(self):
        client = self.mod.OpenAIClient(api_key="k", base_url="b")
        call = ToolCall(id="c1", name="math.adder", arguments={"a": 1, "b": 2})
        result = ToolCallResult(
            id="c1",
            name="math.adder",
            arguments=call.arguments,
            call=call,
            result=3,
        )

        templated = client._template_messages(
            [Message(role=MessageRole.TOOL, tool_call_result=result)],
            tool=_sanitized_policy_manager(),
        )

        self.assertEqual(templated[0]["type"], "function_call")
        self.assertEqual(templated[0]["name"], "math_adder")

    def test_template_messages_skips_tool_only_assistant_placeholder(self):
        client = self.mod.OpenAIClient(api_key="k", base_url="b")
        call = ToolCall(
            id="call1",
            name="math.calculator",
            arguments={"expression": "(4 + 6) * 5 / 2"},
        )
        result = ToolCallResult(
            id="call1",
            name="math.calculator",
            call=call,
            result="25",
        )

        templated = client._template_messages(
            [
                Message(role=MessageRole.USER, content="calculate"),
                Message(
                    role=MessageRole.ASSISTANT,
                    tool_calls=[
                        MessageToolCall(
                            id="call1",
                            name="math.calculator",
                            arguments={"expression": "(4 + 6) * 5 / 2"},
                        )
                    ],
                ),
                Message(role=MessageRole.TOOL, tool_call_result=result),
            ]
        )

        self.assertEqual(
            templated[0], {"role": "user", "content": "calculate"}
        )
        self.assertNotIn(
            {"role": "assistant", "content": "None"},
            templated,
        )
        self.assertEqual(templated[1]["type"], "function_call")
        self.assertEqual(templated[1]["call_id"], "call1")
        self.assertEqual(templated[2]["type"], "function_call_output")
        self.assertEqual(templated[2]["call_id"], "call1")

    def test_template_messages_tool_results_stringify_uuid_call_id(self):
        client = self.mod.OpenAIClient(api_key="k", base_url="b")
        call_id = UUID("11111111-1111-1111-1111-111111111111")
        call = ToolCall(
            id=call_id,
            name="pkg.func",
            arguments={"entity_id": call_id},
        )
        result = ToolCallResult(
            id="c1",
            name="pkg.func",
            call=call,
            result={"entity_id": call_id},
        )
        message = Message(role=MessageRole.TOOL, tool_call_result=result)

        templated = client._template_messages([message])

        self.assertEqual(
            templated,
            [
                {
                    "type": "function_call",
                    "name": "avl_cGtnLmZ1bmM",
                    "call_id": str(call_id),
                    "arguments": (
                        '{"entity_id": "11111111-1111-1111-1111-111111111111"}'
                    ),
                },
                {
                    "type": "function_call_output",
                    "call_id": str(call_id),
                    "output": (
                        '{"entity_id": "11111111-1111-1111-1111-111111111111"}'
                    ),
                },
            ],
        )

    def test_template_messages_uses_model_facing_tool_result(self):
        client = self.mod.OpenAIClient(api_key="k", base_url="b")
        limit = OrchestratorResponse._MAXIMUM_MODEL_TOOL_OUTPUT_CHARS
        output = "x" * (limit + 5_000)
        call = ToolCall(
            id="c1", name="shell.cat", arguments={"path": "big.py"}
        )
        result = ToolCallResult(
            id="c1",
            name="shell.cat",
            arguments=call.arguments,
            call=call,
            result=output,
        )
        messages = OrchestratorResponse._tool_observation_messages(
            result,
            json_output=False,
        )

        templated = client._template_messages(messages)

        payload = loads(templated[1]["output"])
        self.assertTrue(payload["truncated"])
        self.assertEqual(payload["original_output_chars"], limit + 5_000)
        self.assertIn("truncated 5000 characters", payload["output"])
        self.assertLess(len(templated[1]["output"]), len(output))

    def test_template_messages_tool_error(self):
        client = self.mod.OpenAIClient(api_key="k", base_url="b")
        call = ToolCall(id="c1", name="pkg.func", arguments={"a": 1})
        error = ToolCallError(
            id="c1",
            name="pkg.func",
            call=call,
            error=ValueError("boom"),
            message="boom",
        )
        msg = Message(role=MessageRole.TOOL, tool_call_error=error)
        templated = client._template_messages([msg])
        self.assertEqual(
            templated,
            [
                {
                    "type": "function_call",
                    "name": "avl_cGtnLmZ1bmM",
                    "call_id": "c1",
                    "arguments": '{"a": 1}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "c1",
                    "output": '{"error": "boom"}',
                },
            ],
        )

    def test_template_messages_tool_diagnostic(self):
        client = self.mod.OpenAIClient(api_key="k", base_url="b")
        diagnostic = ToolCallDiagnostic(
            id="d1",
            call_id="c1",
            requested_name="missing",
            code=ToolCallDiagnosticCode.UNKNOWN_TOOL,
            stage=ToolCallDiagnosticStage.RESOLVE,
            message="Tool is unknown.",
        )
        msg = Message(
            role=MessageRole.TOOL,
            name="missing",
            arguments={"a": 1},
            tool_call_diagnostic=diagnostic,
        )

        templated = client._template_messages([msg])

        self.assertEqual(templated[0]["call_id"], "c1")
        self.assertEqual(templated[0]["name"], "missing")
        self.assertEqual(templated[0]["arguments"], '{"a": 1}')
        output = loads(templated[1]["output"])
        self.assertEqual(output["code"], "tool.unknown")
        self.assertEqual(output["requested_name"], "missing")

    def test_template_messages_unanchored_tool_diagnostic(self):
        client = self.mod.OpenAIClient(api_key="k", base_url="b")
        diagnostic = ToolCallDiagnostic(
            id="d1",
            code=ToolCallDiagnosticCode.MALFORMED_CALL,
            stage=ToolCallDiagnosticStage.PARSE,
            message="Tool call could not be parsed.",
        )
        msg = Message(
            role=MessageRole.TOOL,
            tool_call_diagnostic=diagnostic,
        )

        templated = client._template_messages([msg])

        self.assertEqual(templated[0]["role"], "assistant")
        output = loads(templated[0]["content"])
        self.assertEqual(output["code"], "tool_call.malformed")


class OpenAIAdditionalCoverageTestCase(TestCase):
    def setUp(self):
        self.openai_stub, self.patch = patch_openai_imports()
        importlib.reload(
            importlib.import_module("avalan.model.nlp.text.vendor.openai")
        )
        self.mod = importlib.import_module(
            "avalan.model.nlp.text.vendor.openai"
        )

    def tearDown(self):
        self.patch.stop()

    def test_non_stream_response_content_handles_dict(self):
        response = {
            "output": [
                {
                    "type": "message",
                    "content": [{"text": "hello"}],
                },
                {
                    "type": "tool_call",
                    "call": {
                        "id": "call-id",
                        "function": {
                            "name": "pkg.tool",
                            "arguments": '{"a":1}',
                        },
                    },
                },
            ]
        }
        with patch.object(
            self.mod.TextGenerationVendor,
            "build_tool_call_text",
            return_value="<tool>",
        ) as build:
            text = self.mod.OpenAIClient._non_stream_response_content(response)

        self.assertEqual(text, "hello<tool>")
        build.assert_called_once_with("call-id", "pkg.tool", '{"a":1}')

    def test_non_stream_response_content_uses_tool_name_policy(self):
        response = {
            "output": [
                {
                    "type": "tool_call",
                    "call": {
                        "id": "call-id",
                        "function": {
                            "name": "math_adder",
                            "arguments": '{"a":1,"b":2}',
                        },
                    },
                },
            ]
        }

        text = self.mod.OpenAIClient._non_stream_response_content(
            response,
            tool=_sanitized_policy_manager(),
        )

        self.assertIn('"name": "math.adder"', text)

    def test_non_streaming_response_str_variants(self):
        settings = GenerationSettings()
        response = self.mod.OpenAINonStreamingResponse(
            lambda **_: "value",
            logger=MagicMock(),
            generation_settings=settings,
            settings=settings,
            use_async_generator=False,
            static_response_text="cached",
        )
        self.assertEqual(str(response), "cached")

        buffered = self.mod.OpenAINonStreamingResponse(
            lambda **_: "value",
            logger=MagicMock(),
            generation_settings=settings,
            settings=settings,
            use_async_generator=False,
        )
        buffered._buffer.write("buffered")
        self.assertEqual(str(buffered), "buffered")

        fallback = self.mod.OpenAINonStreamingResponse(
            lambda **_: "value",
            logger=MagicMock(),
            generation_settings=settings,
            settings=settings,
            use_async_generator=False,
        )
        fallback._buffer = SimpleNamespace(getvalue=lambda: None)
        self.assertIn("OpenAINonStreamingResponse", str(fallback))


class OpenAIModelStreamingFlagTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.openai_stub, self.patch = patch_openai_imports()
        importlib.reload(
            importlib.import_module("avalan.model.nlp.text.vendor.openai")
        )
        self.mod = importlib.import_module(
            "avalan.model.nlp.text.vendor.openai"
        )

    def tearDown(self):
        self.patch.stop()

    async def test_call_returns_streaming_response(self):
        settings = TransformerEngineSettings(
            access_token="tok",
            auto_load_model=False,
            auto_load_tokenizer=False,
        )
        model = self.mod.OpenAIModel("model-id", settings)
        model._model = AsyncMock(
            return_value=lambda *_args, **_kwargs: AsyncIter([])
        )

        response = await model(
            "prompt",
            system_prompt="sys",
            developer_prompt="dev",
            settings=GenerationSettings(),
        )

        self.assertIsInstance(response, self.mod.TextGenerationResponse)


if __name__ == "__main__":
    from unittest import main

    main()
