from avalan.entities import EngineSettings
from avalan.model import TokenizerNotSupportedException
from avalan.model.audio import BaseAudioModel
from logging import Logger
from unittest import IsolatedAsyncioTestCase
from unittest.mock import MagicMock
from typing import Literal


class DummyAudioModel(BaseAudioModel):
    def _load_model(self):
        return MagicMock()

    async def __call__(
        self, image_source: str | object, tensor_format: Literal["pt"] = "pt"
    ) -> str:
        return await super().__call__(image_source, tensor_format)


class BaseAudioModelTestCase(IsolatedAsyncioTestCase):
    async def test_base_methods_raise(self):
        model = DummyAudioModel(
            None,
            EngineSettings(auto_load_model=False),
            logger=MagicMock(spec=Logger),
        )
        with self.assertRaises(NotImplementedError):
            await model("img")
        with self.assertRaises(TokenizerNotSupportedException):
            model._load_tokenizer(None)
        with self.assertRaises(TokenizerNotSupportedException):
            model._load_tokenizer_with_tokens(None)
