from types import SimpleNamespace
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import MagicMock, patch

import torch

from avalan.entities import (
    GenerationSettings,
    Message,
    MessageContentFile,
    MessageContentImage,
    MessageContentText,
    MessageRole,
    MessageToolCall,
    TransformerEngineSettings,
)
from avalan.model import (
    DomainCapabilitySeed,
    ModelCapabilityCatalog,
    ModelCapabilityDescriptor,
)
from avalan.model.nlp.text.generation import TextGenerationModel
from avalan.model.nlp.text.local_protocol import (
    LOCAL_STRUCTURED_OUTPUT_PROTOCOL,
    LOCAL_STRUCTURED_OUTPUT_PROTOCOL_ID,
    LOCAL_STRUCTURED_OUTPUT_TEMPLATE_NAME,
)
from avalan.model.stream import StreamItemKind


class TokenGeneratorPickTestCase(IsolatedAsyncioTestCase):
    async def test_pick_creates_tokens(self) -> None:
        model = TextGenerationModel(
            "m",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
        )
        model._tokenizer = MagicMock()
        model._tokenizer.decode.side_effect = lambda i, **_: f"t{i}"
        model._log = MagicMock()

        outputs = SimpleNamespace(
            sequences=torch.tensor([[5, 1, 2]]),
            scores=[
                torch.tensor([[1.0, 2.0, 3.0]]),
                torch.tensor([[0.5, 0.4, 0.1]]),
            ],
        )

        with (
            patch.object(
                TextGenerationModel, "_generate_output", return_value=outputs
            ),
            patch(
                "avalan.model.nlp.text.generation.softmax",
                return_value=torch.tensor([0.2, 0.3, 0.5]),
            ),
            patch("avalan.model.nlp.text.generation.topk") as topk_mock,
        ):
            topk_mock.return_value = SimpleNamespace(
                indices=torch.tensor([2, 1]),
                values=torch.tensor([0.5, 0.4]),
            )
            settings = GenerationSettings(max_new_tokens=2, temperature=1.0)
            inputs = {"input_ids": torch.tensor([[5]])}
            result = []
            async for t in model._token_generator(
                inputs,
                settings,
                None,
                False,
                pick=2,
            ):
                result.append(t)

        deltas = [
            item for item in result if item.kind is StreamItemKind.ANSWER_DELTA
        ]
        self.assertEqual(len(deltas), 2)
        self.assertEqual(
            [item.metadata["token_id"] for item in deltas], [1, 2]
        )
        self.assertTrue(all("tokens" in item.metadata for item in deltas))


class TokenizeInputPrefixTestCase(TestCase):
    def test_prefix_added_when_no_user_message(self) -> None:
        model = TextGenerationModel(
            "m",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
        )
        model._model = MagicMock(device="cpu")
        model._tokenizer = MagicMock(chat_template=None)
        token_out = MagicMock()
        token_out.to.return_value = token_out
        model._tokenizer.return_value = token_out
        model._messages = MagicMock(
            return_value=[Message(role=MessageRole.ASSISTANT, content="a")]
        )
        model._log = MagicMock()

        result = model._tokenize_input("in", "sys", context=None)

        expected_prompt = "sys\n\nAssistant: a\n"
        model._tokenizer.assert_called_once_with(
            expected_prompt, add_special_tokens=True, return_tensors="pt"
        )
        token_out.to.assert_called_once_with("cpu")
        self.assertIs(result, token_out)


class TokenizeInputWrapperTestCase(TestCase):
    def test_mapping_inputs_move_to_device(self) -> None:
        inputs = {"input_ids": torch.tensor([1, 2])}

        result = TextGenerationModel._move_inputs_to_device(inputs, "cpu")

        self.assertIsInstance(result, dict)
        self.assertEqual(result["input_ids"].device.type, "cpu")

    def test_tokenize_prompt_rejects_tokenizer_without_encode(self) -> None:
        class BadTokenizer:
            pass

        model = TextGenerationModel(
            "m",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
        )
        model._tokenizer = BadTokenizer()

        with self.assertRaisesRegex(TypeError, "does not provide encode"):
            model._tokenize_prompt("hi")

    def test_wrapper_chat_template_flag_uses_template(self) -> None:
        class WrapperLikeTokenizer:
            chat_template = None
            has_chat_template = True

            def __init__(self) -> None:
                self.calls: list[tuple[object, dict[str, object]]] = []

            def apply_chat_template(
                self, *args: object, **kwargs: object
            ) -> list[int]:
                self.calls.append((args, kwargs))
                return [4, 5]

        model = TextGenerationModel(
            "m",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
        )
        tokenizer = WrapperLikeTokenizer()
        model._tokenizer = tokenizer
        model._log = MagicMock()

        result = model._tokenize_input("hi", None, context=None)

        self.assertEqual(len(tokenizer.calls), 1)
        args, kwargs = tokenizer.calls[0]
        template_messages = args[0]
        assert isinstance(template_messages, list)
        self.assertEqual(template_messages[0]["content"], "hi")
        self.assertEqual(kwargs["return_tensors"], "pt")
        self.assertIsInstance(result, dict)
        input_ids = result["input_ids"] if isinstance(result, dict) else None
        assert input_ids is not None
        self.assertTrue(torch.equal(input_ids, torch.tensor([[4, 5]])))


class LocalCapabilitySupportTestCase(TestCase):
    @staticmethod
    def _capability() -> ModelCapabilityCatalog:
        return ModelCapabilityCatalog.create(
            DomainCapabilitySeed(
                descriptors=(
                    ModelCapabilityDescriptor(
                        canonical_name="math.calculate",
                        description="Calculate one expression.",
                        parameter_schema={
                            "type": "object",
                            "properties": {"expression": {"type": "string"}},
                            "required": ("expression",),
                            "additionalProperties": False,
                        },
                    ),
                )
            )
        )

    @staticmethod
    def _model(template: object) -> TextGenerationModel:
        model = TextGenerationModel(
            "m",
            TransformerEngineSettings(
                auto_load_model=False,
                auto_load_tokenizer=False,
            ),
        )
        tokenizer = MagicMock()
        tokenizer.chat_template = template
        tokenizer.apply_chat_template.return_value = [1, 2]
        model._tokenizer = tokenizer
        model._log = MagicMock()
        return model

    def test_capability_requires_explicit_exact_protocol_template(
        self,
    ) -> None:
        capability = self._capability()
        supported = self._model(
            {LOCAL_STRUCTURED_OUTPUT_TEMPLATE_NAME: "{{ messages }}"}
        )
        native_tools = self._model(
            "{% if tools %}{{ tools | length }}{% endif %}{{ messages }}"
        )
        unknown_adapter = self._model(
            {"tool_use": "{{ tools }}{{ messages }}"}
        )
        empty_adapter = self._model(
            {LOCAL_STRUCTURED_OUTPUT_TEMPLATE_NAME: " "}
        )

        self.assertIs(
            supported._effective_local_capability(capability), capability
        )
        self.assertIsNone(native_tools._effective_local_capability(capability))
        self.assertIsNone(
            unknown_adapter._effective_local_capability(capability)
        )
        self.assertIsNone(
            empty_adapter._effective_local_capability(capability)
        )
        self.assertEqual(
            capability.descriptors[0].canonical_name,
            "math.calculate",
        )

    def test_capability_requires_callable_template_application(self) -> None:
        model = self._model(
            {LOCAL_STRUCTURED_OUTPUT_TEMPLATE_NAME: "{{ messages }}"}
        )
        model._tokenizer = SimpleNamespace(
            chat_template={
                LOCAL_STRUCTURED_OUTPUT_TEMPLATE_NAME: "{{ messages }}"
            }
        )

        self.assertFalse(model._tokenizer_supports_structured_capabilities())

    def test_unsupported_template_omits_structured_schemas(self) -> None:
        capability = self._capability()
        model = self._model("{{ messages }}")

        model._tokenize_input("hello", capability=capability)

        tokenizer = cast(MagicMock, model._tokenizer)
        kwargs = tokenizer.apply_chat_template.call_args.kwargs
        self.assertNotIn("tools", kwargs)

    def test_exact_adapter_receives_projection_and_protocol_instruction(
        self,
    ) -> None:
        capability = self._capability()
        model = self._model(
            {
                "default": "{{ messages }}",
                LOCAL_STRUCTURED_OUTPUT_TEMPLATE_NAME: "{{ messages }}",
            }
        )

        model._tokenize_input("hello", capability=capability)

        tokenizer = cast(MagicMock, model._tokenizer)
        call = tokenizer.apply_chat_template.call_args
        tools = call.kwargs["tools"]
        assert tools is not None
        self.assertEqual(tools[0]["function"]["name"], "math.calculate")
        self.assertEqual(
            call.kwargs["chat_template"],
            "{{ messages }}",
        )
        messages = call.args[0]
        instruction = messages[0]["content"]
        assert isinstance(instruction, str)
        self.assertIn(LOCAL_STRUCTURED_OUTPUT_PROTOCOL_ID, instruction)
        self.assertIn(
            "<tool_call id=JSON_STRING name=JSON_STRING>"
            "JSON_OBJECT</tool_call>",
            instruction,
        )
        self.assertIn('"name":"math.calculate"', instruction)
        self.assertEqual(messages[-1]["content"], "hello")

    def test_explicit_native_template_override_disables_protocol(self) -> None:
        capability = self._capability()
        model = self._model(
            {
                "default": "native {{ messages }}",
                LOCAL_STRUCTURED_OUTPUT_TEMPLATE_NAME: "exact {{ messages }}",
            }
        )

        model._tokenize_input(
            "hello",
            capability=capability,
            chat_template="native {{ messages }}",
        )

        tokenizer = cast(MagicMock, model._tokenizer)
        call = tokenizer.apply_chat_template.call_args
        self.assertNotIn("tools", call.kwargs)
        self.assertEqual(call.kwargs["chat_template"], "native {{ messages }}")
        self.assertNotIn(
            LOCAL_STRUCTURED_OUTPUT_PROTOCOL_ID,
            str(call.args[0]),
        )

    def test_protocol_rejects_non_json_schema_values(self) -> None:
        with self.assertRaisesRegex(
            TypeError,
            "schemas must contain only JSON values",
        ):
            LOCAL_STRUCTURED_OUTPUT_PROTOCOL.instruction(
                cast(Any, ({"invalid": object()},))
            )


class TokenizeInputContentTextTestCase(TestCase):
    def test_message_content_text_handled(self) -> None:
        model = TextGenerationModel(
            "m",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
        )
        model._model = MagicMock(device="cpu")
        model._tokenizer = MagicMock(chat_template=None)
        token_out = MagicMock()
        token_out.to.return_value = token_out
        model._tokenizer.return_value = token_out
        model._log = MagicMock()

        message = Message(
            role=MessageRole.USER,
            content=MessageContentText(type="text", text="hi"),
        )

        result = model._tokenize_input(message, None, context=None)

        model._tokenizer.assert_called_once()
        args, kwargs = model._tokenizer.call_args
        self.assertTrue(args[0].endswith("hi\n"))
        self.assertEqual(
            kwargs, {"add_special_tokens": True, "return_tensors": "pt"}
        )
        token_out.to.assert_called_once_with("cpu")
        self.assertIs(result, token_out)


class TokenizeInputContentImageTestCase(TestCase):
    def _setup(
        self, has_template: bool
    ) -> tuple[TextGenerationModel, MagicMock, MagicMock]:
        model = TextGenerationModel(
            "m",
            TransformerEngineSettings(
                auto_load_model=False,
                auto_load_tokenizer=False,
            ),
        )
        model._model = MagicMock(device="cpu")
        tokenizer = MagicMock()
        token_out = MagicMock()
        token_out.to.return_value = token_out
        if has_template:
            tokenizer.chat_template = "tpl"
            tokenizer.apply_chat_template.return_value = token_out
        else:
            tokenizer.chat_template = None
            tokenizer.return_value = token_out
        model._tokenizer = tokenizer
        model._log = MagicMock()
        return model, tokenizer, token_out

    def test_image_content_with_template(self) -> None:
        model, tokenizer, token_out = self._setup(True)
        message = Message(
            role=MessageRole.USER,
            content=MessageContentImage(
                type="image_url", image_url={"url": "u"}
            ),
        )
        result = model._tokenize_input(message, None, context=None)
        tokenizer.apply_chat_template.assert_called_once()
        args, kwargs = tokenizer.apply_chat_template.call_args
        self.assertEqual(
            args[0],
            [
                {
                    "role": MessageRole.USER,
                    "content": [
                        {"type": "image_url", "image_url": {"url": "u"}},
                    ],
                    "thinking": "",
                    "arguments": None,
                    "name": None,
                    "tool_calls": [],
                    "tool_call_result": None,
                    "tool_call_error": None,
                }
            ],
        )
        token_out.to.assert_called_once_with("cpu")
        self.assertIs(result, token_out)

    def test_image_content_plain(self) -> None:
        model, tokenizer, token_out = self._setup(False)
        message = Message(
            role=MessageRole.USER,
            content=MessageContentImage(
                type="image_url", image_url={"url": "u"}
            ),
        )
        result = model._tokenize_input(message, None, context=None)
        tokenizer.assert_called_once()
        self.assertEqual(tokenizer.call_args[0][0], "None\n\n\n")
        token_out.to.assert_called_once_with("cpu")
        self.assertIs(result, token_out)

    def test_file_content_plain(self) -> None:
        model, tokenizer, token_out = self._setup(False)
        message = Message(
            role=MessageRole.USER,
            content=MessageContentFile(
                type="file", file={"file_id": "file-1"}
            ),
        )
        result = model._tokenize_input(message, None, context=None)
        tokenizer.assert_called_once()
        self.assertEqual(tokenizer.call_args[0][0], "None\n\n\n")
        token_out.to.assert_called_once_with("cpu")
        self.assertIs(result, token_out)


class TokenizeInputContentListTestCase(TestCase):
    def _setup(
        self, has_template: bool
    ) -> tuple[TextGenerationModel, MagicMock, MagicMock]:
        model = TextGenerationModel(
            "m",
            TransformerEngineSettings(
                auto_load_model=False,
                auto_load_tokenizer=False,
            ),
        )
        model._model = MagicMock(device="cpu")
        tokenizer = MagicMock()
        token_out = MagicMock()
        token_out.to.return_value = token_out
        if has_template:
            tokenizer.chat_template = "tpl"
            tokenizer.apply_chat_template.return_value = token_out
        else:
            tokenizer.chat_template = None
            tokenizer.return_value = token_out
        model._tokenizer = tokenizer
        model._log = MagicMock()
        return model, tokenizer, token_out

    def test_list_content_with_template(self) -> None:
        model, tokenizer, token_out = self._setup(True)
        message = Message(
            role=MessageRole.USER,
            content=[
                MessageContentImage(type="image_url", image_url={"url": "u"}),
                MessageContentText(type="text", text="hi"),
            ],
        )
        result = model._tokenize_input(message, None, context=None)
        tokenizer.apply_chat_template.assert_called_once()
        args, kwargs = tokenizer.apply_chat_template.call_args
        self.assertEqual(
            args[0],
            [
                {
                    "role": MessageRole.USER,
                    "content": [
                        {"type": "image_url", "image_url": {"url": "u"}},
                        {"type": "text", "text": "hi"},
                    ],
                    "thinking": "",
                    "arguments": None,
                    "name": None,
                    "tool_calls": [],
                    "tool_call_result": None,
                    "tool_call_error": None,
                }
            ],
        )
        token_out.to.assert_called_once_with("cpu")
        self.assertIs(result, token_out)

    def test_list_content_plain(self) -> None:
        model, tokenizer, token_out = self._setup(False)
        message = Message(
            role=MessageRole.USER,
            content=[
                MessageContentText(type="text", text="a"),
                MessageContentImage(type="image_url", image_url={"url": "u"}),
                MessageContentText(type="text", text="b"),
            ],
        )
        result = model._tokenize_input(message, None, context=None)
        tokenizer.assert_called_once()
        self.assertEqual(tokenizer.call_args[0][0], "None\n\na\nb\n")
        token_out.to.assert_called_once_with("cpu")
        self.assertIs(result, token_out)

    def test_list_content_with_file_template_ignores_file(self) -> None:
        model, tokenizer, token_out = self._setup(True)
        message = Message(
            role=MessageRole.USER,
            content=[
                MessageContentText(type="text", text="a"),
                MessageContentFile(
                    type="file", file={"file_url": "http://file"}
                ),
                MessageContentText(type="text", text="b"),
            ],
        )
        result = model._tokenize_input(message, None, context=None)
        tokenizer.apply_chat_template.assert_called_once()
        args, kwargs = tokenizer.apply_chat_template.call_args
        self.assertEqual(
            args[0],
            [
                {
                    "role": MessageRole.USER,
                    "content": [
                        {"type": "text", "text": "a"},
                        {"type": "text", "text": "b"},
                    ],
                    "thinking": "",
                    "arguments": None,
                    "name": None,
                    "tool_calls": [],
                    "tool_call_result": None,
                    "tool_call_error": None,
                }
            ],
        )
        token_out.to.assert_called_once_with("cpu")
        self.assertIs(result, token_out)


class TokenizeInputHarmonyContentTestCase(TestCase):
    def test_harmony_tags_split_into_fields(self) -> None:
        model = TextGenerationModel(
            "m",
            TransformerEngineSettings(
                auto_load_model=False,
                auto_load_tokenizer=False,
            ),
        )
        model._model = MagicMock(device="cpu")
        tokenizer = MagicMock()
        token_out = MagicMock()
        token_out.to.return_value = token_out
        tokenizer.chat_template = "tpl"
        tokenizer.apply_chat_template.return_value = token_out
        model._tokenizer = tokenizer
        model._log = MagicMock()

        message = Message(
            role=MessageRole.ASSISTANT,
            content=(
                "<|start|>assistant<|channel|>analysis<|message|>think<|end|>"
                "<|start|>assistant<|channel|>final<|message|>answer<|end|>"
            ),
        )

        result = model._tokenize_input(message, None, context=None)

        tokenizer.apply_chat_template.assert_called_once()
        template_messages = tokenizer.apply_chat_template.call_args[0][0]
        self.assertEqual(template_messages[0]["thinking"], "think")
        self.assertEqual(template_messages[0]["content"], "answer")
        token_out.to.assert_called_once_with("cpu")
        self.assertIs(result, token_out)

    def test_harmony_tool_call_sets_empty_content(self) -> None:
        model = TextGenerationModel(
            "m",
            TransformerEngineSettings(
                auto_load_model=False,
                auto_load_tokenizer=False,
            ),
        )
        model._model = MagicMock(device="cpu")
        tokenizer = MagicMock()
        token_out = MagicMock()
        token_out.to.return_value = token_out
        tokenizer.chat_template = "tpl"
        tokenizer.apply_chat_template.return_value = token_out
        model._tokenizer = tokenizer
        model._log = MagicMock()

        tool_call = MessageToolCall(id="c1", name="fn", arguments=[])
        message = Message(
            role=MessageRole.ASSISTANT,
            content=(
                "<|start|>assistant<|channel|>analysis<|message|>think<|end|>"
                "<|start|>assistant<|channel|>commentary to=fn<|message|>{}"
                "<|call|>"
            ),
            tool_calls=[tool_call],
        )

        result = model._tokenize_input(message, None, context=None)

        tokenizer.apply_chat_template.assert_called_once()
        template_messages = tokenizer.apply_chat_template.call_args[0][0]
        self.assertEqual(template_messages[0]["thinking"], "think")
        self.assertEqual(template_messages[0]["content"], "")
        token_out.to.assert_called_once_with("cpu")
        self.assertIs(result, token_out)

    def test_harmony_tool_call_inferred_from_content(self) -> None:
        model = TextGenerationModel(
            "m",
            TransformerEngineSettings(
                auto_load_model=False,
                auto_load_tokenizer=False,
            ),
        )
        model._model = MagicMock(device="cpu")
        tokenizer = MagicMock()
        token_out = MagicMock()
        token_out.to.return_value = token_out
        tokenizer.chat_template = "tpl"
        tokenizer.apply_chat_template.return_value = token_out
        model._tokenizer = tokenizer
        model._log = MagicMock()

        message = Message(
            role=MessageRole.ASSISTANT,
            content=(
                "<|start|>assistant<|channel|>analysis<|message|>think<|end|>"
                "<|start|>assistant<|channel|>commentary to=fn<|message|>{}"
                "<|call|>"
            ),
        )

        result = model._tokenize_input(message, None, context=None)

        tokenizer.apply_chat_template.assert_called_once()
        template_messages = tokenizer.apply_chat_template.call_args[0][0]
        self.assertEqual(template_messages[0]["thinking"], "think")
        self.assertEqual(template_messages[0]["content"], "")
        tool_calls = template_messages[0]["tool_calls"]
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["name"], "fn")
        self.assertEqual(tool_calls[0]["arguments"], {})
        self.assertEqual(tool_calls[0]["content_type"], "json")
        self.assertIsInstance(tool_calls[0]["id"], str)
        token_out.to.assert_called_once_with("cpu")
        self.assertIs(result, token_out)

    def test_harmony_message_content_text_processed(self) -> None:
        model = TextGenerationModel(
            "m",
            TransformerEngineSettings(
                auto_load_model=False,
                auto_load_tokenizer=False,
            ),
        )
        model._model = MagicMock(device="cpu")
        tokenizer = MagicMock()
        token_out = MagicMock()
        token_out.to.return_value = token_out
        tokenizer.chat_template = "tpl"
        tokenizer.apply_chat_template.return_value = token_out
        model._tokenizer = tokenizer
        model._log = MagicMock()

        message = Message(
            role=MessageRole.ASSISTANT,
            content=MessageContentText(
                type="text",
                text=(
                    "<|start|>assistant<|channel|>analysis<|message|>plan<|end|>"
                    "<|start|>assistant<|channel|>final<|message|>done<|end|>"
                ),
            ),
        )

        result = model._tokenize_input(message, None, context=None)

        tokenizer.apply_chat_template.assert_called_once()
        template_message = tokenizer.apply_chat_template.call_args[0][0][0]
        self.assertEqual(template_message["thinking"], "plan")
        self.assertEqual(template_message["content"], "done")
        self.assertNotIn("<|channel|>", template_message["thinking"])
        self.assertNotIn("<|channel|>", template_message["content"])
        token_out.to.assert_called_once_with("cpu")
        self.assertIs(result, token_out)

    def test_harmony_message_content_list_processed(self) -> None:
        model = TextGenerationModel(
            "m",
            TransformerEngineSettings(
                auto_load_model=False,
                auto_load_tokenizer=False,
            ),
        )
        model._model = MagicMock(device="cpu")
        tokenizer = MagicMock()
        token_out = MagicMock()
        token_out.to.return_value = token_out
        tokenizer.chat_template = "tpl"
        tokenizer.apply_chat_template.return_value = token_out
        model._tokenizer = tokenizer
        model._log = MagicMock()

        message = Message(
            role=MessageRole.ASSISTANT,
            content=[
                MessageContentText(
                    type="text",
                    text=(
                        "<|start|>assistant<|channel|>analysis<|message|>reason<|end|>"
                        "<|start|>assistant<|channel|>final<|message|>result<|end|>"
                    ),
                )
            ],
        )

        result = model._tokenize_input(message, None, context=None)

        tokenizer.apply_chat_template.assert_called_once()
        template_message = tokenizer.apply_chat_template.call_args[0][0][0]
        self.assertEqual(template_message["thinking"], "reason")
        self.assertEqual(template_message["content"], "result")
        self.assertNotIn("<|channel|>", template_message["thinking"])
        self.assertNotIn("<|channel|>", template_message["content"])
        token_out.to.assert_called_once_with("cpu")
        self.assertIs(result, token_out)


class TokenizeInputUnknownContentTestCase(TestCase):
    def test_unknown_content_converted_to_string(self) -> None:
        model = TextGenerationModel(
            "m",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
        )
        model._model = MagicMock(device="cpu")
        tokenizer = MagicMock(chat_template=None)
        token_out = MagicMock()
        token_out.to.return_value = token_out
        tokenizer.return_value = token_out
        model._tokenizer = tokenizer
        model._log = MagicMock()

        message = Message(role=MessageRole.USER, content=123)
        result = model._tokenize_input(message, None, context=None)
        tokenizer.assert_called_once()
        self.assertEqual(tokenizer.call_args[0][0], "None\n\n123\n")
        token_out.to.assert_called_once_with("cpu")
        self.assertIs(result, token_out)


class MessagesInvalidInputTestCase(TestCase):
    def test_invalid_input_raises(self) -> None:
        model = TextGenerationModel(
            "m",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
        )
        with self.assertRaises(ValueError):
            model._messages(1, None)


if __name__ == "__main__":
    from unittest import main

    main()
