from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import MagicMock, patch

import torch

from avalan.entities import (
    GenerationSettings,
    Message,
    MessageContentImage,
    MessageContentText,
    MessageRole,
    TransformerEngineSettings,
)
from avalan.model.nlp.text.generation import TextGenerationModel


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

        self.assertEqual(len(result), 2)
        self.assertEqual([t.id for t in result], [1, 2])
        self.assertTrue(all(t.tokens is not None for t in result))


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
