from avalan.entities import TransformerEngineSettings
from avalan.model.nlp.sequence import (
    SequenceClassificationModel,
    SequenceToSequenceModel,
)
from logging import Logger
from unittest import TestCase
from unittest.mock import MagicMock


class SequenceModelPropertyTestCase(TestCase):
    def test_properties(self) -> None:
        sc_model = SequenceClassificationModel(
            "id",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
            logger=MagicMock(spec=Logger),
        )
        ss_model = SequenceToSequenceModel(
            "id",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
            logger=MagicMock(spec=Logger),
        )
        self.assertFalse(sc_model.supports_sample_generation)
        self.assertFalse(sc_model.supports_token_streaming)
        self.assertFalse(ss_model.supports_sample_generation)
        self.assertFalse(ss_model.supports_token_streaming)


class SequenceToSequenceTokenizeInputTestCase(TestCase):
    def test_tokenize_input(self) -> None:
        model = SequenceToSequenceModel(
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
        self.assertEqual(result, "ids")


if __name__ == "__main__":
    from unittest import main

    main()
