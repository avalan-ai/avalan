from logging import Logger
from unittest import TestCase
from unittest.mock import MagicMock

from avalan.entities import TransformerEngineSettings
from avalan.model.nlp.sentence import SentenceTransformerModel


class SentenceTransformerModelPropertyTestCase(TestCase):
    def test_properties(self):
        model = SentenceTransformerModel(
            "id",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
            logger=MagicMock(spec=Logger),
        )
        self.assertFalse(model.supports_sample_generation)
        self.assertFalse(model.supports_token_streaming)
        self.assertTrue(model.uses_tokenizer)


class SentenceTransformerModelTokenizeTestCase(TestCase):
    def test_tokenize_not_implemented(self):
        model = SentenceTransformerModel(
            "id",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
            logger=MagicMock(spec=Logger),
        )
        with self.assertRaises(NotImplementedError):
            model._tokenize_input("q", None, None)
