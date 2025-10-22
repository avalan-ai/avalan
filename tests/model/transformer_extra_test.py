from logging import Logger
from unittest import TestCase
from unittest.mock import MagicMock

from avalan.entities import TransformerEngineSettings
from avalan.model.transformer import TransformerModel


class DummyTransformerModel(TransformerModel):
    def _load_model(self):
        return MagicMock()

    def _tokenize_input(self, *args, **kwargs):
        return super()._tokenize_input(*args, **kwargs)

    async def __call__(self, *args, **kwargs):
        return None


class TransformerModelPropertyTestCase(TestCase):
    def test_properties_and_tokenize(self) -> None:
        model = DummyTransformerModel(
            "id",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
            logger=MagicMock(spec=Logger),
        )
        self.assertFalse(model.supports_sample_generation)
        self.assertFalse(model.supports_token_streaming)
        self.assertTrue(model.uses_tokenizer)
        with self.assertRaises(NotImplementedError):
            model._tokenize_input("in")
