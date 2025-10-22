from logging import Logger
from unittest import TestCase
from unittest.mock import MagicMock

from avalan.entities import TransformerEngineSettings
from avalan.model.nlp.question import QuestionAnsweringModel


class QuestionAnsweringModelPropertyTestCase(TestCase):
    def test_properties(self):
        model = QuestionAnsweringModel(
            "id",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
            logger=MagicMock(spec=Logger),
        )
        self.assertFalse(model.supports_sample_generation)
        self.assertFalse(model.supports_token_streaming)


class QuestionAnsweringTokenizeInputTestCase(TestCase):
    def test_tokenize_input(self):
        model = QuestionAnsweringModel(
            "id",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
            logger=MagicMock(spec=Logger),
        )
        model._tokenizer = MagicMock()
        token_out = MagicMock()
        token_out.to.return_value = "t"
        model._tokenizer.return_value = token_out
        model._model = MagicMock(device="cpu")
        model._log = MagicMock()

        result = model._tokenize_input("q", None, context="c")
        self.assertEqual(result, "t")
        model._tokenizer.assert_called_once_with("q", "c", return_tensors="pt")
        token_out.to.assert_called_once_with("cpu")
        model._log.assert_called_once()

    def test_tokenize_input_with_system_prompt(self):
        model = QuestionAnsweringModel(
            "id",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
            logger=MagicMock(spec=Logger),
        )
        with self.assertRaises(AssertionError):
            model._tokenize_input("q", system_prompt="sp", context=None)
