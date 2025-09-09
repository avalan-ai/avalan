from unittest import TestCase
from unittest.mock import MagicMock, patch

from avalan.entities import TransformerEngineSettings
from avalan.model.transformer import TransformerModel


class DummyTransformerModel(TransformerModel):
    def _load_model(self):
        return MagicMock()

    def _tokenize_input(
        self,
        input,
        system_prompt=None,
        context=None,
        tensor_format="pt",
        **kwargs,
    ):
        return {"input_ids": [[1, 2, 3]]}

    async def __call__(self, *args, **kwargs):
        return None


class TransformerTokenizerFullTestCase(TestCase):
    def setUp(self) -> None:
        settings = TransformerEngineSettings(
            auto_load_model=False,
            auto_load_tokenizer=False,
            tokens=["tok1", "tok2"],
            special_tokens=["<S1>", "<S2>"],
        )
        self.model = DummyTransformerModel("m", settings, logger=MagicMock())
        self.tokenizer = MagicMock()
        self.tokenizer.name_or_path = "tok-name"
        self.tokenizer.save_pretrained.return_value = ("f1", "f2")
        self.tokenizer.encode.return_value = [10, 11]
        self.tokenizer.decode.side_effect = (
            lambda token_id, skip_special_tokens=False: f"t{token_id}"
        )
        self.model._tokenizer = self.tokenizer
        self.model._loaded_tokenizer = True

    def test_save_tokenizer_and_tokenize(self) -> None:
        files = self.model.save_tokenizer("/tmp/path")
        self.tokenizer.save_pretrained.assert_called_once_with("/tmp/path")
        self.assertEqual(files, ["f1", "f2"])

        tokens = self.model.tokenize("hi")
        self.tokenizer.encode.assert_called_once_with(
            "hi", add_special_tokens=True
        )
        self.assertEqual(len(tokens), 2)
        self.assertEqual(tokens[0].token, "t10")

        count = self.model.input_token_count("ignored")
        self.assertEqual(count, 3)

    @patch("avalan.model.transformer.AutoTokenizer")
    def test_load_tokenizer_with_tokens(self, auto_tokenizer) -> None:
        tok = MagicMock()
        tok.name_or_path = "base"
        auto_tokenizer.from_pretrained.return_value = tok
        result = self.model._load_tokenizer_with_tokens("base")
        auto_tokenizer.from_pretrained.assert_called_once_with(
            "base",
            use_fast=True,
            subfolder=self.model._settings.tokenizer_subfolder or "",
        )
        tok.add_tokens.assert_called_once_with(["tok1", "tok2"])
        tok.add_special_tokens.assert_called_once()
        self.assertIs(result, tok)
