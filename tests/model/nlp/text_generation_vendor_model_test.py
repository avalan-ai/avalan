from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import AsyncMock, MagicMock, patch

from avalan.entities import (
    GenerationSettings,
    Message,
    MessageRole,
    TransformerEngineSettings,
)
from avalan.model import TextGenerationResponse
from avalan.model.nlp.text.vendor import TextGenerationVendorModel


class DummyVendorModel(TextGenerationVendorModel):
    pass


# allow instantiation without implementing abstract methods
DummyVendorModel.__abstractmethods__ = set()


class ConstructorTestCase(TestCase):
    def test_no_settings_requires_token(self):
        with self.assertRaises(AssertionError):
            DummyVendorModel("m")

    def test_settings_without_token_raises(self):
        settings = TransformerEngineSettings(
            auto_load_model=False, auto_load_tokenizer=False
        )
        with self.assertRaises(AssertionError):
            DummyVendorModel("m", settings)

    def test_settings_with_token_success(self):
        settings = TransformerEngineSettings(
            auto_load_model=False,
            auto_load_tokenizer=False,
            access_token="tok",
            enable_eval=True,
        )
        model = DummyVendorModel("m", settings)
        self.assertIsInstance(model, DummyVendorModel)
        self.assertFalse(model._settings.enable_eval)


class PropertyTestCase(TestCase):
    def setUp(self):
        self.settings = TransformerEngineSettings(
            auto_load_model=False,
            auto_load_tokenizer=False,
            access_token="tok",
        )
        self.model = DummyVendorModel("m", self.settings)

    def test_properties(self):
        self.assertFalse(self.model.supports_sample_generation)
        self.assertTrue(self.model.supports_token_streaming)
        self.assertFalse(self.model.uses_tokenizer)

    def test_load_model_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.model._load_model()

    def test_tokenize_input_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.model._tokenize_input("in")


class InputTokenCountTestCase(TestCase):
    def setUp(self):
        self.settings = TransformerEngineSettings(
            auto_load_model=False,
            auto_load_tokenizer=False,
            access_token="tok",
        )
        self.model = DummyVendorModel("m", self.settings)

    def test_input_token_count_with_model_encoding(self):
        encoding = MagicMock()
        encoding.encode.side_effect = lambda text: list(text)
        messages = [
            Message(role=MessageRole.USER, content="hi"),
            Message(role=MessageRole.USER, content="there"),
        ]
        with (
            patch(
                "avalan.model.nlp.text.vendor.encoding_for_model",
                return_value=encoding,
            ) as efm,
            patch("avalan.model.nlp.text.vendor.get_encoding") as ge,
        ):
            self.model._messages = MagicMock(return_value=messages)
            count = self.model.input_token_count("in")
        efm.assert_called_once_with("m")
        ge.assert_not_called()
        self.assertEqual(count, len("hi") + len("there"))

    def test_input_token_count_fallback_encoding(self):
        encoding = MagicMock()
        encoding.encode.side_effect = lambda text: list(text)
        messages = [Message(role=MessageRole.USER, content="a")]
        with (
            patch(
                "avalan.model.nlp.text.vendor.encoding_for_model",
                side_effect=KeyError,
            ),
            patch(
                "avalan.model.nlp.text.vendor.get_encoding",
                return_value=encoding,
            ) as ge,
        ):
            self.model._messages = MagicMock(return_value=messages)
            count = self.model.input_token_count("in")
        ge.assert_called_once()
        self.assertEqual(count, 1)


class CallTestCase(IsolatedAsyncioTestCase):
    async def test_call(self):
        settings = TransformerEngineSettings(
            auto_load_model=False,
            auto_load_tokenizer=False,
            access_token="tok",
        )
        model = DummyVendorModel("m", settings)
        messages = [Message(role=MessageRole.USER, content="hi")]
        model._messages = MagicMock(return_value=messages)

        async def fake_model(
            model_id, msgs, opts, *, tool=None, use_async_generator=True
        ):
            self.assertEqual(model_id, "m")
            self.assertIs(msgs, messages)
            self.assertIs(opts, gen_settings)
            self.assertTrue(use_async_generator)
            return "streamer"

        model._model = AsyncMock(side_effect=fake_model)
        gen_settings = GenerationSettings()
        response = await model(
            "input",
            system_prompt="sys",
            settings=gen_settings,
            tool=None,
        )
        model._messages.assert_called_once_with("input", "sys", None)
        model._model.assert_awaited_once_with(
            "m",
            messages,
            gen_settings,
            tool=None,
            use_async_generator=True,
        )
        self.assertIsInstance(response, TextGenerationResponse)
        self.assertIs(response._output_fn, "streamer")
        self.assertIs(response._kwargs["settings"], gen_settings)
        self.assertTrue(response._use_async_generator)


if __name__ == "__main__":
    main()
