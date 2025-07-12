from avalan.entities import GenerationSettings, TransformerEngineSettings
from avalan.model.engine import Engine
from avalan.model.nlp.sequence import SequenceToSequenceModel
from logging import Logger
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)
from unittest import TestCase, IsolatedAsyncioTestCase, main
from unittest.mock import MagicMock, patch, PropertyMock


class SequenceToSequenceModelInstantiationTestCase(TestCase):
    model_id = "dummy/model"

    def test_instantiation_no_load(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(
                AutoTokenizer, "from_pretrained"
            ) as auto_tokenizer_mock,
            patch.object(
                AutoModelForSeq2SeqLM, "from_pretrained"
            ) as auto_model_mock,
        ):
            settings = TransformerEngineSettings(
                auto_load_model=False,
                auto_load_tokenizer=False,
            )
            model = SequenceToSequenceModel(
                self.model_id,
                settings,
                logger=logger_mock,
            )
            self.assertIsInstance(model, SequenceToSequenceModel)
            auto_model_mock.assert_not_called()
            auto_tokenizer_mock.assert_not_called()

    def test_instantiation_with_load_model_and_tokenizer(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(
                AutoTokenizer, "from_pretrained"
            ) as auto_tokenizer_mock,
            patch.object(
                AutoModelForSeq2SeqLM, "from_pretrained"
            ) as auto_model_mock,
        ):
            model_instance = MagicMock(spec=PreTrainedModel)
            type(model_instance).name_or_path = PropertyMock(
                return_value=self.model_id
            )
            auto_model_mock.return_value = model_instance

            tokenizer_instance = MagicMock(spec=PreTrainedTokenizerFast)
            tokenizer_instance.__len__.return_value = 1
            tokenizer_instance.model_max_length = 10
            tokenizer_instance.all_special_tokens = []
            type(tokenizer_instance).name_or_path = PropertyMock(
                return_value=self.model_id
            )
            auto_tokenizer_mock.return_value = tokenizer_instance

            model = SequenceToSequenceModel(
                self.model_id,
                TransformerEngineSettings(),
                logger=logger_mock,
            )
            self.assertIs(model._model, model_instance)

            auto_model_mock.assert_called_once_with(
                self.model_id,
                cache_dir=None,
                subfolder=None,
                attn_implementation=None,
                trust_remote_code=False,
                torch_dtype="auto",
                state_dict=None,
                local_files_only=False,
                token=None,
                device_map=Engine.get_default_device(),
                tp_plan=None,
            )
            auto_tokenizer_mock.assert_called_once_with(
                self.model_id,
                use_fast=True,
                subfolder=None,
            )


class SequenceToSequenceModelCallTestCase(IsolatedAsyncioTestCase):
    async def test_call(self):
        settings = TransformerEngineSettings(
            auto_load_model=False,
            auto_load_tokenizer=False,
        )
        model = SequenceToSequenceModel("dummy", settings=settings)
        model._model = MagicMock()
        model._tokenizer = MagicMock()
        model._tokenizer.decode.return_value = "summary"

        inputs = MagicMock()
        with (
            patch.object(
                SequenceToSequenceModel, "_tokenize_input", return_value=inputs
            ) as tok_mock,
            patch.object(
                SequenceToSequenceModel,
                "_generate_output",
                return_value=[[42]],
            ) as gen_mock,
        ):
            gen_settings = GenerationSettings(max_length=5)
            result = await model("text", gen_settings, stopping_criterias=None)

        self.assertEqual(result, "summary")
        tok_mock.assert_called_once_with(
            "text", system_prompt=None, context=None
        )
        gen_mock.assert_called_once_with(inputs, gen_settings, None)
        model._tokenizer.decode.assert_called_once_with(
            [42], skip_special_tokens=True
        )


if __name__ == "__main__":
    main()
