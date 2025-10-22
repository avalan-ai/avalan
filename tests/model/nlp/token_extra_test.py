from logging import Logger
from unittest import TestCase
from unittest.mock import MagicMock

from avalan.entities import TransformerEngineSettings
from avalan.model.nlp.token import TokenClassificationModel


class TokenClassificationModelPropertyTestCase(TestCase):
    def test_properties(self) -> None:
        model = TokenClassificationModel(
            "id",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
            logger=MagicMock(spec=Logger),
        )
        self.assertFalse(model.supports_sample_generation)
        self.assertFalse(model.supports_token_streaming)
        self.assertTrue(model.uses_tokenizer)


class TokenClassificationTokenizeInputTestCase(TestCase):
    def test_tokenize_input(self) -> None:
        model = TokenClassificationModel(
            "id",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
            logger=MagicMock(spec=Logger),
        )
        model._tokenizer = MagicMock()
        token_out = MagicMock()
        token_out.to.return_value = {"input_ids": "ids"}
        model._tokenizer.return_value = token_out
        model._model = MagicMock(device="cpu")
        model._log = MagicMock()

        result = model._tokenize_input("hi", None, context=None)

        model._tokenizer.assert_called_once_with("hi", return_tensors="pt")
        token_out.to.assert_called_once_with("cpu")
        model._log.assert_called_once()
        self.assertEqual(result, {"input_ids": "ids"})

    def test_tokenize_input_with_system_prompt(self) -> None:
        model = TokenClassificationModel(
            "id",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
            logger=MagicMock(spec=Logger),
        )
        with self.assertRaises(AssertionError):
            model._tokenize_input("hi", system_prompt="sp", context=None)
