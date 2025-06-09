from avalan.entities import GenerationSettings, TransformerEngineSettings
from avalan.model.nlp.text.generation import TextGenerationModel
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import MagicMock, patch, PropertyMock


class TextGenerationModelCallTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.settings = TransformerEngineSettings(
            auto_load_model=False,
            auto_load_tokenizer=False,
        )
        self.model = TextGenerationModel("m", self.settings)
        self.model._model = MagicMock()
        self.model._tokenizer = MagicMock()
        self.model._tokenizer.eos_token_id = 99

    async def test_string_output_selected(self):
        tok_inputs = {"input_ids": [[1, 2]]}
        tokenize_mock = MagicMock(return_value=tok_inputs)
        string_output = MagicMock()
        self.model._tokenize_input = tokenize_mock
        self.model._string_output = string_output

        settings = GenerationSettings(
            temperature=0.5,
            use_async_generator=False,
        )
        response = await self.model("hi", settings=settings)

        tokenize_mock.assert_called_once_with(
            "hi",
            None,
            context=None,
            tool=None,
            chat_template_settings=settings.chat_template_settings,
        )
        self.assertIs(response._output_fn, string_output)
        self.assertEqual(response._kwargs["inputs"], tok_inputs)
        self.assertTrue(response._kwargs["settings"].do_sample)
        self.assertEqual(
            response._kwargs["settings"].pad_token_id,
            self.model._tokenizer.eos_token_id,
        )
        self.assertFalse(response._use_async_generator)

    async def test_stream_output_selected(self):
        tok_inputs = {"input_ids": [[3]]}
        self.model._tokenize_input = MagicMock(return_value=tok_inputs)
        stream_output = MagicMock()
        self.model._stream_generator = stream_output

        settings = GenerationSettings(
            temperature=None, use_async_generator=True
        )
        response = await self.model("go", settings=settings)

        self.assertIs(response._output_fn, stream_output)
        self.assertFalse(response._kwargs["settings"].do_sample)
        self.assertTrue(response._use_async_generator)

    async def test_manual_sampling_token_generator(self):
        tok_inputs = {"input_ids": [[4]]}
        self.model._tokenize_input = MagicMock(return_value=tok_inputs)
        token_gen = MagicMock()
        self.model._token_generator = token_gen

        settings = GenerationSettings(
            temperature=None, use_async_generator=True
        )
        response = await self.model(
            "ok", settings=settings, manual_sampling=True
        )

        self.assertIs(response._output_fn, token_gen)
        self.assertTrue(response._use_async_generator)

    async def test_do_sample_without_temperature_raises(self):
        self.model._tokenize_input = MagicMock(
            return_value={"input_ids": [[1]]}
        )
        with self.assertRaises(AssertionError):
            await self.model(
                "bad",
                settings=GenerationSettings(do_sample=True, temperature=None),
            )

    async def test_temperature_without_support_raises(self):
        self.model._tokenize_input = MagicMock(
            return_value={"input_ids": [[1]]}
        )
        with patch.object(
            TextGenerationModel,
            "supports_sample_generation",
            new_callable=PropertyMock,
            return_value=False,
        ):
            with self.assertRaises(AssertionError):
                await self.model(
                    "bad",
                    settings=GenerationSettings(temperature=0.2),
                )

    async def test_default_settings_used_when_none_provided(self):
        tok_inputs = {"input_ids": [[5]]}
        self.model._tokenize_input = MagicMock(return_value=tok_inputs)
        stream_output = MagicMock()
        self.model._stream_generator = stream_output

        response = await self.model("hey")

        self.assertIs(response._output_fn, stream_output)
        self.assertTrue(response._use_async_generator)
        self.assertTrue(response._kwargs["settings"].do_sample)
        self.assertEqual(
            response._kwargs["settings"].pad_token_id,
            self.model._tokenizer.eos_token_id,
        )

    async def test_manual_sampling_selects_token_generator(self):
        tok_inputs = {"input_ids": [[6]]}
        self.model._tokenize_input = MagicMock(return_value=tok_inputs)
        token_gen = MagicMock()
        self.model._token_generator = token_gen

        settings = GenerationSettings(
            use_async_generator=True, temperature=None
        )
        response = await self.model(
            "x", settings=settings, manual_sampling=True
        )

        self.assertIs(response._output_fn, token_gen)
        self.assertTrue(response._use_async_generator)

    async def test_use_async_generator_true_stream_output(self):
        tok_inputs = {"input_ids": [[7]]}
        self.model._tokenize_input = MagicMock(return_value=tok_inputs)
        stream_output = MagicMock()
        self.model._stream_generator = stream_output

        settings = GenerationSettings(
            use_async_generator=True, temperature=None
        )
        response = await self.model("y", settings=settings)

        self.assertIs(response._output_fn, stream_output)
        self.assertTrue(response._use_async_generator)

    async def test_use_async_generator_false_string_output(self):
        tok_inputs = {"input_ids": [[8]]}
        self.model._tokenize_input = MagicMock(return_value=tok_inputs)
        string_output = MagicMock()
        self.model._string_output = string_output

        settings = GenerationSettings(
            use_async_generator=False,
            temperature=0.7,
        )
        response = await self.model("z", settings=settings)

        self.assertIs(response._output_fn, string_output)
        self.assertFalse(response._use_async_generator)
        self.assertTrue(response._kwargs["settings"].do_sample)


if __name__ == "__main__":
    main()
