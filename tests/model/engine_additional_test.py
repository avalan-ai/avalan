from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from avalan.entities import EngineSettings
from avalan.model.engine import Engine, TokenizerNotSupportedException


class DummyEngine(Engine):
    def uses_tokenizer(self) -> bool:
        return False

    async def __call__(self, input, **kwargs):
        return await super().__call__(input, **kwargs)

    def _load_model(self):
        return super()._load_model()


class EngineAdditionalTestCase(IsolatedAsyncioTestCase):
    async def test_base_call_raises(self):
        engine = DummyEngine(
            "id",
            EngineSettings(auto_load_model=False, auto_load_tokenizer=False),
        )
        with self.assertRaises(NotImplementedError):
            await engine("hi")

    def test_base_load_model_raises(self):
        engine = DummyEngine(
            "id",
            EngineSettings(auto_load_model=False, auto_load_tokenizer=False),
        )
        with self.assertRaises(NotImplementedError):
            engine._load_model()

    def test_load_tokenizer_with_tokens_raises(self):
        engine = DummyEngine(
            "id",
            EngineSettings(auto_load_model=False, auto_load_tokenizer=False),
        )
        with self.assertRaises(TokenizerNotSupportedException):
            engine._load_tokenizer_with_tokens("tok")

    def test_get_device_memory_cuda_unavailable(self):
        with patch(
            "avalan.model.engine.cuda.is_available", return_value=False
        ):
            self.assertEqual(Engine._get_device_memory("cuda"), 0)
