from avalan.model.transformer import AutoTokenizer
from avalan.model.entities import GenerationSettings, TransformerEngineSettings
from avalan.model.nlp.sequence import SequenceToSequenceModel
from logging import Logger
from transformers import AutoModelForSeq2SeqLM, PreTrainedModel, PreTrainedTokenizerFast
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import MagicMock, PropertyMock, patch


class SequenceToSequenceModelInstantiationTestCase(TestCase):
    model_id = "dummy/model"

    def test_instantiation_no_load(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(AutoTokenizer, "from_pretrained") as auto_tokenizer_mock,
            patch.object(AutoModelForSeq2SeqLM, "from_pretrained") as auto_model_mock,
        ):
            settings = TransformerEngineSettings(
                auto_load_model=False,
                auto_load_tokenizer=False,
            )
            model = SequenceToSequenceModel(self.model_id, settings, logger=logger_mock)
            self.assertIsInstance(model, SequenceToSequenceModel)
            auto_tokenizer_mock.assert_not_called()
            auto_model_mock.assert_not_called()

    def test_instantiation_with_load_tokenizer(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(AutoTokenizer, "from_pretrained") as auto_tokenizer_mock,
            patch.object(AutoModelForSeq2SeqLM, "from_pretrained") as auto_model_mock,
        ):
            tokenizer_mock = MagicMock(spec=PreTrainedTokenizerFast)
            type(tokenizer_mock).name_or_path = PropertyMock(return_value=self.model_id)
            auto_tokenizer_mock.return_value = tokenizer_mock

            settings = TransformerEngineSettings(
                auto_load_model=False,
                auto_load_tokenizer=True,
            )
            model = SequenceToSequenceModel(self.model_id, settings, logger=logger_mock)
            self.assertIsInstance(model, SequenceToSequenceModel)
            auto_tokenizer_mock.assert_called_once_with(self.model_id, use_fast=True)
            auto_model_mock.assert_not_called()

    def test_instantiation_with_load_model_and_tokenizer(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(AutoTokenizer, "from_pretrained") as auto_tokenizer_mock,
            patch.object(AutoModelForSeq2SeqLM, "from_pretrained") as auto_model_mock,
        ):
            model_instance = MagicMock(spec=PreTrainedModel)
            type(model_instance).name_or_path = PropertyMock(return_value=self.model_id)
            auto_model_mock.return_value = model_instance

            tokenizer_mock = MagicMock(spec=PreTrainedTokenizerFast)
            type(tokenizer_mock).name_or_path = PropertyMock(return_value=self.model_id)
            auto_tokenizer_mock.return_value = tokenizer_mock

            settings = TransformerEngineSettings(
                auto_load_model=True,
                auto_load_tokenizer=True,
            )
            model = SequenceToSequenceModel(self.model_id, settings, logger=logger_mock)
            self.assertIsInstance(model, SequenceToSequenceModel)
            auto_model_mock.assert_called_once_with(
                self.model_id,
                cache_dir=None,
                attn_implementation=None,
                trust_remote_code=False,
                torch_dtype="auto",
                state_dict=None,
                local_files_only=False,
                token=None,
            )
            auto_tokenizer_mock.assert_called_once_with(self.model_id, use_fast=True)


class SequenceToSequenceModelCallTestCase(IsolatedAsyncioTestCase):
    async def test_call_returns_decoded_text(self):
        model = SequenceToSequenceModel(
            "dummy/model",
            TransformerEngineSettings(auto_load_model=False, auto_load_tokenizer=False),
        )
        model._model = MagicMock()
        model._tokenizer = MagicMock()
        model._tokenizer.decode.return_value = "decoded"

        with (
            patch.object(model, "_tokenize_input", return_value="encoded") as token_mock,
            patch.object(model, "_generate_output", return_value=[[1, 2]]) as gen_mock,
        ):
            result = await model("in", GenerationSettings())

            self.assertEqual(result, "decoded")
            token_mock.assert_called_once_with("in", system_prompt=None, context=None)
            gen_mock.assert_called_once()
            model._tokenizer.decode.assert_called_once_with([1, 2], skip_special_tokens=True)


if __name__ == "__main__":
    main()
