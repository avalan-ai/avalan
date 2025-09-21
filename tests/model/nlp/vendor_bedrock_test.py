from contextlib import AsyncExitStack
from dataclasses import dataclass
from importlib import import_module, reload
from importlib.machinery import ModuleSpec
from sys import modules
from types import ModuleType, SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch

from avalan.entities import (
    GenerationSettings,
    Message,
    MessageContentImage,
    MessageContentText,
    MessageRole,
    ReasoningToken,
    Token,
    ToolCall,
    ToolCallError,
    ToolCallResult,
    ToolCallToken,
    TransformerEngineSettings,
)


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


def patch_bedrock_imports():
    aioboto3_stub = ModuleType("aioboto3")
    aioboto3_stub.__spec__ = ModuleSpec("aioboto3", loader=None)
    session_mock = MagicMock()
    aioboto3_stub.Session = MagicMock(return_value=session_mock)

    transformers_stub = ModuleType("transformers")
    transformers_stub.__spec__ = ModuleSpec("transformers", loader=None)
    transformers_stub.PreTrainedModel = type("PreTrainedModel", (), {})
    transformers_stub.PreTrainedTokenizer = type("PreTrainedTokenizer", (), {})
    transformers_stub.PreTrainedTokenizerFast = type(
        "PreTrainedTokenizerFast", (), {}
    )
    transformers_stub.__getattr__ = lambda name: MagicMock()

    transformers_utils_stub = ModuleType("transformers.utils")
    transformers_utils_stub.__spec__ = ModuleSpec(
        "transformers.utils", loader=None
    )
    transformers_utils_stub.get_json_schema = MagicMock()
    transformers_logging_stub = ModuleType("transformers.utils.logging")
    transformers_logging_stub.__spec__ = ModuleSpec(
        "transformers.utils.logging", loader=None
    )
    transformers_logging_stub.disable_progress_bar = MagicMock()
    transformers_logging_stub.enable_progress_bar = MagicMock()
    transformers_utils_stub.logging = transformers_logging_stub

    tokenization_stub = ModuleType("transformers.tokenization_utils_base")
    tokenization_stub.__spec__ = ModuleSpec(
        "transformers.tokenization_utils_base", loader=None
    )
    tokenization_stub.BatchEncoding = MagicMock()

    generation_stub = ModuleType("transformers.generation")
    generation_stub.__spec__ = ModuleSpec(
        "transformers.generation", loader=None
    )
    generation_stub.StoppingCriteria = MagicMock()
    generation_stub.StoppingCriteriaList = MagicMock()

    diffusers_stub = ModuleType("diffusers")
    diffusers_stub.__spec__ = ModuleSpec("diffusers", loader=None)
    diffusers_stub.DiffusionPipeline = MagicMock()

    patcher = patch.dict(
        modules,
        {
            "aioboto3": aioboto3_stub,
            "transformers": transformers_stub,
            "transformers.utils": transformers_utils_stub,
            "transformers.utils.logging": transformers_logging_stub,
            "transformers.tokenization_utils_base": tokenization_stub,
            "transformers.generation": generation_stub,
            "diffusers": diffusers_stub,
        },
    )
    patcher.start()
    return aioboto3_stub, session_mock, patcher


class ClientContext:
    def __init__(self, client):
        self._client = client

    async def __aenter__(self):
        return self._client

    async def __aexit__(self, exc_type, exc, tb):
        return False


class BedrockTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.aioboto3_stub, self.session_mock, self.patch = (
            patch_bedrock_imports()
        )
        reload(import_module("avalan.model.nlp.text.vendor.bedrock"))
        self.mod = import_module("avalan.model.nlp.text.vendor.bedrock")
        self.client = SimpleNamespace(
            converse_stream=AsyncMock(), converse=AsyncMock()
        )
        self.session_mock.client.return_value = ClientContext(self.client)

    def tearDown(self):
        self.patch.stop()

    async def test_stream_processing(self):
        events = [
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"reasoning": {"text": "think"}},
                }
            },
            {
                "contentBlockStart": {
                    "contentBlockIndex": 1,
                    "contentBlock": {
                        "toolUse": {
                            "toolUseId": "id1",
                            "name": "pkg__tool",
                        }
                    },
                }
            },
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 1,
                    "delta": {"toolUse": {"input": "{"}},
                }
            },
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 1,
                    "delta": {"toolUse": {"input": '"a":1}'}},
                }
            },
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"text": {"text": "hi"}},
                }
            },
            {"contentBlockStop": {"contentBlockIndex": 1}},
            {"messageStop": {}},
        ]
        with patch.object(
            self.mod.TextGenerationVendor,
            "build_tool_call_token",
            return_value="tool",
        ) as build:
            stream = self.mod.BedrockStream(AsyncIter(events))
            out = []
            while True:
                try:
                    out.append(await stream.__anext__())
                except StopAsyncIteration:
                    break

        self.assertEqual(len(out), 5)
        self.assertIsInstance(out[0], ReasoningToken)
        self.assertEqual(out[0].token, "think")
        self.assertIsInstance(out[1], ToolCallToken)
        self.assertEqual(out[1].token, "{")
        self.assertIsInstance(out[2], ToolCallToken)
        self.assertEqual(out[2].token, '"a":1}')
        self.assertIsInstance(out[3], Token)
        self.assertEqual(out[3].token, "hi")
        self.assertEqual(out[4], "tool")
        build.assert_called_once_with("id1", "pkg__tool", '{"a":1}')

    async def test_client_stream_invocation(self):
        self.client.converse_stream.return_value = {"stream": AsyncIter([])}
        exit_stack = AsyncExitStack()
        client = self.mod.BedrockClient(
            exit_stack=exit_stack,
            region_name="us-east-1",
            endpoint_url="https://example.com",
        )
        client._system_prompt = MagicMock(return_value=None)
        client._template_messages = MagicMock(
            return_value=[{"role": "user", "content": []}]
        )
        client._inference_config = MagicMock(return_value=None)
        client._tool_config = MagicMock(return_value=None)

        with patch.object(self.mod, "BedrockStream") as StreamMock:
            result = await client("model", [], GenerationSettings())

        self.session_mock.client.assert_called_once_with(
            "bedrock-runtime",
            region_name="us-east-1",
            endpoint_url="https://example.com",
        )
        self.client.converse_stream.assert_awaited_once_with(
            modelId="model", messages=[{"role": "user", "content": []}]
        )
        StreamMock.assert_called_once_with(
            events=self.client.converse_stream.return_value["stream"]
        )
        self.assertIs(result, StreamMock.return_value)
        await exit_stack.aclose()

    async def test_client_without_stream(self):
        self.client.converse.return_value = {
            "output": {
                "message": {
                    "content": [
                        {"text": {"text": "hello"}},
                        {"text": {"text": " world"}},
                    ]
                }
            }
        }
        exit_stack = AsyncExitStack()
        client = self.mod.BedrockClient(exit_stack=exit_stack)
        client._system_prompt = MagicMock(return_value=None)
        client._template_messages = MagicMock(
            return_value=[{"role": "user", "content": []}]
        )
        client._inference_config = MagicMock(return_value=None)
        client._tool_config = MagicMock(return_value=None)

        result = await client(
            "model",
            [],
            GenerationSettings(),
            use_async_generator=False,
        )

        text = await result.__anext__()
        self.assertEqual(text, "hello world")
        self.client.converse.assert_awaited_once()
        await exit_stack.aclose()

    def test_template_messages_and_tool_config(self):
        client = self.mod.BedrockClient(exit_stack=AsyncExitStack())

        @dataclass
        class Payload:
            value: int

        tool_call = ToolCall(id="id1", name="pkg.tool", arguments={"a": 1})
        tool_result = ToolCallResult(
            id="id1", name="pkg.tool", call=tool_call, result=Payload(2)
        )
        tool_error = ToolCallError(
            id="id2",
            name="pkg.tool",
            call=tool_call,
            error=ValueError("bad"),
            message="bad",
        )

        messages = [
            Message(role=MessageRole.USER, content="hello"),
            Message(
                role=MessageRole.DEVELOPER,
                content=MessageContentText(type="text", text="dev"),
            ),
            Message(
                role=MessageRole.ASSISTANT,
                content=[
                    MessageContentText(type="text", text="chunk"),
                    MessageContentImage(
                        type="image_url",
                        image_url={"url": "https://example.com"},
                    ),
                ],
            ),
            Message(role=MessageRole.TOOL, tool_call_result=tool_result),
            Message(role=MessageRole.TOOL, tool_call_error=tool_error),
        ]

        templated = client._template_messages(messages)
        self.assertEqual(templated[0]["role"], "user")
        self.assertEqual(templated[0]["content"][0]["text"]["text"], "hello")
        self.assertEqual(templated[1]["role"], "user")
        self.assertEqual(templated[1]["content"][0]["text"]["text"], "dev")
        self.assertEqual(
            templated[2]["content"][1]["image"]["source"]["url"],
            "https://example.com",
        )
        self.assertEqual(
            templated[3]["content"][0]["toolResult"]["toolUseId"],
            "id1",
        )
        self.assertEqual(
            templated[4]["content"][0]["toolResult"]["status"],
            "error",
        )

        tool_manager = MagicMock()
        tool_manager.json_schemas.return_value = [
            {
                "type": "function",
                "function": {
                    "name": "pkg.tool",
                    "description": "desc",
                    "parameters": {"type": "object"},
                },
            }
        ]
        config = client._tool_config(tool_manager)
        self.assertEqual(
            config["tools"][0]["toolSpec"]["name"],
            "pkg__tool",
        )

    def test_model_loads_client(self):
        settings = TransformerEngineSettings(
            access_token="https://endpoint",
            base_url="us-east-1",
        )
        model = self.mod.BedrockModel("model", settings)
        client = model._load_model()
        self.assertIsInstance(client, self.mod.BedrockClient)
        self.assertEqual(client._region_name, "us-east-1")
        self.assertEqual(client._endpoint_url, "https://endpoint")


class BedrockStreamHelpersTest(TestCase):
    def test_string_helper(self):
        _, _, patcher = patch_bedrock_imports()
        self.addCleanup(patcher.stop)
        module = import_module("avalan.model.nlp.text.vendor.bedrock")
        self.assertEqual(module._string({"text": {"text": "value"}}), "value")
        self.assertIsNone(module._string(123))
