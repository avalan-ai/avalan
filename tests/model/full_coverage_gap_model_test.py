import asyncio
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch

import torch

from avalan.entities import (
    EngineSettings,
    GenerationSettings,
    Message,
    MessageContentImage,
    MessageContentText,
    MessageRole,
    Modality,
    ReasoningToken,
    Token,
    TransformerEngineSettings,
)
from avalan.model.engine import Engine
from avalan.model.manager import ModelManager
from avalan.model.modalities import ModalityRegistry
from avalan.model.nlp.text.generation import TextGenerationModel
from avalan.model.nlp.text.vendor.ollama import OllamaStream
from avalan.model.nlp.text.vllm import VllmModel, VllmStream
from avalan.model.response.text import TextGenerationResponse
from avalan.model.transformer import TransformerModel
from avalan.model.vendor import TextGenerationVendor


class MinimalEngine(Engine):
    async def __call__(self, input, **kwargs):
        return "ok"

    def _load_model(self):
        return object()


class MinimalTransformer(TransformerModel):
    async def __call__(self, input, **kwargs):
        return "ok"

    def _load_model(self):
        return object()

    def _tokenize_input(
        self, input, context=None, tensor_format="pt", **kwargs
    ):
        del input, context, tensor_format, kwargs
        return {"input_ids": []}


class EngineCoverageTestCase(IsolatedAsyncioTestCase):
    async def test_wait_closed_handles_pending_task(self) -> None:
        engine = MinimalEngine(
            "id",
            EngineSettings(auto_load_model=False, auto_load_tokenizer=False),
        )
        engine._pending_exit_task = asyncio.create_task(asyncio.sleep(0))
        await engine.wait_closed()
        self.assertIsNone(engine._pending_exit_task)


class ModelManagerCoverageTestCase(IsolatedAsyncioTestCase):
    async def test_aexit_awaits_pending_task(self) -> None:
        manager = ModelManager(MagicMock(), MagicMock())
        manager._pending_exit_task = asyncio.create_task(asyncio.sleep(0))
        manager._stack.__aexit__ = AsyncMock(return_value=False)  # type: ignore[method-assign]
        result = await manager.__aexit__(None, None, None)
        self.assertFalse(result)
        self.assertIsNone(manager._pending_exit_task)


class ModalityRegistryCoverageTestCase(TestCase):
    def test_register_non_class_handler(self) -> None:
        modality = Modality.VISION_TEXT_TO_IMAGE
        original = dict(ModalityRegistry._handlers)

        async def handler(*args, **kwargs):
            del args, kwargs
            return "ok"

        try:
            decorated = ModalityRegistry.register(modality)(handler)
            self.assertIs(decorated, handler)
            self.assertIs(ModalityRegistry.get(modality), handler)
        finally:
            ModalityRegistry._handlers = original


class TextGenerationModelCoverageTestCase(TestCase):
    def test_messages_accepts_list_of_strings(self) -> None:
        model = TextGenerationModel(
            "id",
            TransformerEngineSettings(
                auto_load_model=False,
                auto_load_tokenizer=False,
            ),
        )
        messages = model._messages(["a", "b"], None)
        self.assertEqual([m.content for m in messages], ["a", "b"])


class VllmCoverageTestCase(IsolatedAsyncioTestCase):
    async def test_vllm_stream_async_generator_token_branch(self) -> None:
        async def agen():
            yield Token(token="z")

        stream = VllmStream(agen())
        self.assertIsNone(stream._iterator)
        self.assertEqual(await stream.__anext__(), "z")

    async def test_stream_generator_object_chunk(self) -> None:
        model = VllmModel(
            "id",
            TransformerEngineSettings(
                auto_load_model=False,
                auto_load_tokenizer=False,
            ),
        )
        model._build_sampling_params = MagicMock(return_value="params")
        model._model = MagicMock()
        model._model.generate.return_value = iter(
            [SimpleNamespace(outputs=[SimpleNamespace(text="chunk")])]
        )
        chunks = [
            c
            async for c in model._stream_generator(
                "prompt", GenerationSettings()
            )
        ]
        self.assertEqual(chunks, ["chunk"])

    async def test_call_returns_stream_generator_instance(self) -> None:
        model = VllmModel(
            "id",
            TransformerEngineSettings(
                auto_load_model=False,
                auto_load_tokenizer=False,
            ),
        )
        model._prompt = MagicMock(return_value="prompt")

        async def simple_stream(*args, **kwargs):
            del args, kwargs
            yield "x"

        stream = simple_stream()
        with patch.object(model, "_stream_generator", return_value=stream):
            result = await model(
                "input", settings=GenerationSettings(use_async_generator=True)
            )
        self.assertIs(result, stream)


class OllamaCoverageTestCase(IsolatedAsyncioTestCase):
    async def test_ollama_stream_non_dict_chunk(self) -> None:
        async def agen():
            yield "raw"

        stream = OllamaStream(agen())
        self.assertEqual(await stream.__anext__(), "")


class VendorCoverageTestCase(TestCase):
    def test_system_prompt_message_content_text(self) -> None:
        vendor = TextGenerationVendor()
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content=MessageContentText(type="text", text="sys"),
            )
        ]
        self.assertEqual(vendor._system_prompt(messages), "sys")

    def test_system_prompt_message_content_image_returns_none(self) -> None:
        vendor = TextGenerationVendor()
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content=MessageContentImage(
                    type="image_url", image_url="http://image"
                ),
            )
        ]
        self.assertIsNone(vendor._system_prompt(messages))


class TransformerCoverageTestCase(TestCase):
    def test_token_count_input_ids_non_list_is_zero(self) -> None:
        model = MinimalTransformer(
            "id",
            TransformerEngineSettings(
                auto_load_model=False,
                auto_load_tokenizer=False,
            ),
        )
        model._tokenizer = MagicMock()
        with patch.object(
            model, "_tokenize_input", return_value={"input_ids": 1}
        ):
            self.assertEqual(model.input_token_count("x"), 0)

    def test_token_count_empty_tensor_is_zero(self) -> None:
        model = MinimalTransformer(
            "id",
            TransformerEngineSettings(
                auto_load_model=False,
                auto_load_tokenizer=False,
            ),
        )
        model._tokenizer = MagicMock()
        with patch.object(
            model,
            "_tokenize_input",
            return_value={"input_ids": torch.tensor([])},
        ):
            self.assertEqual(model.input_token_count("x"), 0)


class TextGenerationResponseCoverageTestCase(IsolatedAsyncioTestCase):
    async def test_reasoning_token_without_id_defaults_minus_one(self) -> None:
        async def output_fn(**kwargs):
            del kwargs
            for value in [Token(token="<think>"), Token(token="x")]:
                yield value

        response = TextGenerationResponse(
            output_fn,
            logger=MagicMock(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
        )
        response._reasoning_parser.push = AsyncMock(  # type: ignore[union-attr]
            return_value=[ReasoningToken(token="x")]
        )

        response.__aiter__()
        token = await response.__anext__()
        self.assertIsInstance(token, ReasoningToken)
        self.assertEqual(token.id, -1)


class TransformerAdditionalCoverageTestCase(TestCase):
    def test_input_token_count_non_mapping_inputs(self) -> None:
        model = MinimalTransformer(
            "id",
            TransformerEngineSettings(
                auto_load_model=False,
                auto_load_tokenizer=False,
            ),
        )
        model._tokenizer = MagicMock()
        with patch.object(
            model, "_tokenize_input", return_value=torch.tensor([1])
        ):
            self.assertEqual(model.input_token_count("x"), 0)

    def test_input_token_count_flat_list(self) -> None:
        model = MinimalTransformer(
            "id",
            TransformerEngineSettings(
                auto_load_model=False,
                auto_load_tokenizer=False,
            ),
        )
        model._tokenizer = MagicMock()
        with patch.object(
            model, "_tokenize_input", return_value={"input_ids": [1, 2, 3]}
        ):
            self.assertEqual(model.input_token_count("x"), 3)


class EngineAdditionalCoverageTestCase(IsolatedAsyncioTestCase):
    async def test_wait_closed_no_pending_task(self) -> None:
        engine = MinimalEngine(
            "id",
            EngineSettings(auto_load_model=False, auto_load_tokenizer=False),
        )
        engine._pending_exit_task = None
        await engine.wait_closed()
        self.assertIsNone(engine._pending_exit_task)


class TextGenerationModelAdditionalCoverageTestCase(TestCase):
    def test_tokenize_input_handles_none_content(self) -> None:
        model = TextGenerationModel(
            "id",
            TransformerEngineSettings(
                auto_load_model=False,
                auto_load_tokenizer=False,
            ),
        )
        model._model = object()
        model._tokenizer = MagicMock()
        model._tokenizer.chat_template = "template"
        model._tokenizer.apply_chat_template.return_value = {
            "input_ids": [[1]]
        }
        messages = [Message(role=MessageRole.USER, content=None)]
        with patch.object(model, "_messages", return_value=messages):
            model._tokenize_input("ignored")
        template_messages = (
            model._tokenizer.apply_chat_template.call_args.args[0]
        )
        self.assertEqual(template_messages[0]["content"], "")


class VllmAdditionalCoverageTestCase(IsolatedAsyncioTestCase):
    async def test_vllm_stream_async_generator_string_branch(self) -> None:
        async def agen():
            yield "s"

        stream = VllmStream(agen())
        self.assertEqual(await stream.__anext__(), "s")


class DummyVendorModel:
    pass


class VendorModuleAdditionalCoverageTestCase(TestCase):
    def test_vendor_model_count_tokens_non_string_content(self) -> None:
        from avalan.model.nlp.text.vendor import TextGenerationVendorModel

        class DummyVendorModel(TextGenerationVendorModel):
            def _load_model(self):
                return object()

        model = DummyVendorModel(
            "id",
            TransformerEngineSettings(
                auto_load_model=False,
                auto_load_tokenizer=False,
                access_token="x",
            ),
        )
        model._resolve_encoding = MagicMock(
            return_value=SimpleNamespace(encode=lambda v: [1] * len(v))
        )
        model._messages = MagicMock(
            return_value=[
                Message(
                    role=MessageRole.USER,
                    content=MessageContentText(type="text", text="abc"),
                )
            ]
        )
        self.assertEqual(
            model.input_token_count("q"),
            len(str(MessageContentText(type="text", text="abc"))),
        )


class BedrockAdditionalCoverageTestCase(TestCase):
    def test_bedrock_helper_functions_and_messages(self) -> None:
        from avalan.model.nlp.text.vendor import bedrock as mod

        self.assertIsNone(mod._bedrock_error_code(Exception("x")))
        err = Exception("x")
        err.response = {"x": 1}
        self.assertIsNone(mod._bedrock_error_code(err))
        self.assertEqual(mod._bedrock_error_message(Exception("y")), "y")
        self.assertIsNone(mod._geo_inference_prefix(None))
        self.assertEqual(mod._geo_inference_prefix("eu-west-1"), "eu.")
        self.assertIsNone(mod._geo_inference_prefix("ap-south-1"))

        client = mod.BedrockClient(
            exit_stack=MagicMock(), region_name="ap-south-1"
        )
        with self.assertRaises(ValueError) as exc:
            client._raise_invalid_model_identifier(
                "anthropic.model", Exception("e")
            )
        self.assertIn("us.anthropic", str(exc.exception))

        client2 = mod.BedrockClient(
            exit_stack=MagicMock(), region_name="eu-west-1"
        )
        with self.assertRaises(ValueError) as exc2:
            client2._raise_end_of_life_model_error(
                "anthropic.model", Exception("e")
            )
        self.assertIn("'eu.'", str(exc2.exception))


class OpenAiAdditionalCoverageTestCase(TestCase):
    def test_non_stream_response_content_empty_output(self) -> None:
        from avalan.model.nlp.text.vendor.openai import (
            OpenAIClient,
            OpenAIStream,
        )

        self.assertEqual(
            OpenAIClient._non_stream_response_content({"output": {}}), ""
        )

        async def agen():
            yield SimpleNamespace(
                type="response.output_item.added",
                item=SimpleNamespace(
                    custom_tool_call=SimpleNamespace(id=1), id=None
                ),
            )
            yield SimpleNamespace(type="response.completed")

        stream = OpenAIStream(agen())
        with self.assertRaises(StopAsyncIteration):
            asyncio.run(stream.__anext__())
