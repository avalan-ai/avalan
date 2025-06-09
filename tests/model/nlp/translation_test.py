from avalan.entities import GenerationSettings, TransformerEngineSettings
from avalan.model.nlp.sequence import TranslationModel
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import MagicMock, patch


class TranslationModelCallTestCase(IsolatedAsyncioTestCase):
    async def test_call_translates_and_restores_language(self):
        settings = TransformerEngineSettings(
            auto_load_model=False,
            auto_load_tokenizer=False,
        )
        model = TranslationModel("dummy", settings=settings)
        model._model = MagicMock()
        model._tokenizer = MagicMock()
        model._tokenizer.src_lang = "en_US"
        model._tokenizer.lang_code_to_id = {"es_XX": 1}
        model._tokenizer.decode.return_value = "hola"

        inputs = MagicMock()
        with (
            patch.object(
                TranslationModel, "_tokenize_input", return_value=inputs
            ) as tok_mock,
            patch.object(
                TranslationModel, "_generate_output", return_value=[[42]]
            ) as gen_mock,
        ):
            gen_settings = GenerationSettings(max_length=5)
            result = await model(
                "hi",
                source_language="en_US",
                destination_language="es_XX",
                settings=gen_settings,
                stopping_criterias=None,
                skip_special_tokens=True,
            )

        self.assertEqual(result, "hola")
        tok_mock.assert_called_once_with(
            "hi", system_prompt=None, context=None
        )
        gen_mock.assert_called_once()
        args = gen_mock.call_args.args
        self.assertIs(args[0], inputs)
        settings_arg = args[1]
        self.assertTrue(settings_arg.early_stopping)
        self.assertEqual(settings_arg.repetition_penalty, 1.0)
        self.assertTrue(settings_arg.use_cache)
        self.assertIsNone(settings_arg.temperature)
        self.assertEqual(settings_arg.forced_bos_token_id, 1)
        self.assertEqual(model._tokenizer.src_lang, "en_US")
        model._tokenizer.decode.assert_called_once_with(
            [42], skip_special_tokens=True
        )

    async def test_skip_special_tokens_parameter(self):
        settings = TransformerEngineSettings(
            auto_load_model=False,
            auto_load_tokenizer=False,
        )
        model = TranslationModel("dummy", settings=settings)
        model._model = MagicMock()
        model._tokenizer = MagicMock()
        model._tokenizer.src_lang = "en_US"
        model._tokenizer.lang_code_to_id = {"es_XX": 1}
        model._tokenizer.decode.return_value = "hola"

        with (
            patch.object(
                TranslationModel, "_tokenize_input", return_value=MagicMock()
            ),
            patch.object(
                TranslationModel, "_generate_output", return_value=[[42]]
            ),
        ):
            await model(
                "hi",
                source_language="en_US",
                destination_language="es_XX",
                settings=GenerationSettings(),
                stopping_criterias=None,
                skip_special_tokens=False,
            )

        model._tokenizer.decode.assert_called_once_with(
            [42], skip_special_tokens=False
        )


if __name__ == "__main__":
    main()
