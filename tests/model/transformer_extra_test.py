import sys
from logging import Logger
from unittest import TestCase
from unittest.mock import MagicMock, patch

from avalan.entities import TransformerEngineSettings
from avalan.model.transformer import AutoTokenizer, TransformerModel


class DummyTransformerModel(TransformerModel):
    def _load_model(self):
        return MagicMock()

    def _tokenize_input(self, *args, **kwargs):
        return super()._tokenize_input(*args, **kwargs)

    async def __call__(self, *args, **kwargs):
        return None


class TransformerModelPropertyTestCase(TestCase):
    def test_lazy_auto_tokenizer_ignores_pytest_introspection(self) -> None:
        sys.modules.pop("transformers.models.auto.tokenization_auto", None)

        self.assertFalse(getattr(AutoTokenizer, "__test__", False))
        self.assertNotIn(
            "transformers.models.auto.tokenization_auto", sys.modules
        )

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

    def test_load_tokenizer_disables_tokenization_cleanup(self) -> None:
        tokenizer = MagicMock()
        tokenizer.clean_up_tokenization_spaces = True
        model = DummyTransformerModel(
            "id",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
            logger=MagicMock(spec=Logger),
        )

        with patch.object(
            AutoTokenizer, "from_pretrained", return_value=tokenizer
        ):
            loaded_tokenizer = model._load_tokenizer_with_tokens("tok")

        self.assertIs(loaded_tokenizer, tokenizer)
        self.assertFalse(tokenizer.clean_up_tokenization_spaces)
