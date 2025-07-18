from avalan.entities import EngineSettings
from avalan.model.vision import BaseVisionModel
from logging import Logger
from unittest import IsolatedAsyncioTestCase
from unittest.mock import MagicMock


class DummyVisionModel(BaseVisionModel):
    def _load_model(self):
        return MagicMock()

    async def __call__(
        self,
        image_source: str | object,
    ):
        return await super().__call__(image_source)


class BaseVisionModelTestCase(IsolatedAsyncioTestCase):
    async def test_call_not_implemented(self) -> None:
        model = DummyVisionModel(
            None,
            EngineSettings(auto_load_model=False),
            logger=MagicMock(spec=Logger),
        )
        with self.assertRaises(NotImplementedError):
            await model("img")
