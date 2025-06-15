import sys
import importlib
import types
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

import avalan.model  # noqa: F401


from avalan.entities import GenerationSettings, TransformerEngineSettings


class MlxLmStreamTestCase(IsolatedAsyncioTestCase):
    async def test_stream_iteration(self) -> None:
        from avalan.model.nlp.text import generation as gen_mod

        sys.modules["avalan.model"].TextGenerationModel = (
            gen_mod.TextGenerationModel
        )
        from avalan.model.nlp.text.mlxlm import MlxLmStream

        async def agen():
            yield "a"
            yield "b"

        stream = MlxLmStream(iter(["a", "b"]))
        self.assertEqual(await stream.__anext__(), "a")
        self.assertEqual(await stream.__anext__(), "b")
        with self.assertRaises(StopAsyncIteration):
            await stream.__anext__()
        del sys.modules["avalan.model"].TextGenerationModel


class MlxLmModelTestCase(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        stub = types.ModuleType("mlx_lm")
        stub.load = MagicMock(return_value=("model", "tokenizer"))
        stub.generate = MagicMock(return_value="out")
        self.patch = patch.dict(sys.modules, {"mlx_lm": stub})
        self.patch.start()
        from avalan.model.nlp.text import generation as gen_mod

        sys.modules["avalan.model"].TextGenerationModel = (
            gen_mod.TextGenerationModel
        )
        importlib.reload(
            importlib.import_module("avalan.model.nlp.text.mlxlm")
        )
        self.mod = importlib.import_module("avalan.model.nlp.text.mlxlm")
        self.stub = stub

    async def asyncTearDown(self) -> None:
        self.patch.stop()
        del sys.modules["avalan.model"].TextGenerationModel

    async def test_load_and_call(self) -> None:
        model = self.mod.MlxLmModel(
            "id",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
        )
        model._model = MagicMock()
        model._tokenizer = MagicMock()
        model._tokenizer.decode.return_value = "prompt"
        with (
            patch.object(
                self.mod.TextGenerationModel,
                "_tokenize_input",
                return_value={"input_ids": [1]},
            ) as tok_mock,
            patch.object(
                self.mod.MlxLmModel,
                "_stream_generator",
                new_callable=AsyncMock,
                return_value=self.mod.MlxLmStream(iter(["x"])),
            ) as stream_mock,
        ):
            out = await model(
                "in", settings=GenerationSettings(use_async_generator=True)
            )
        tok_mock.assert_called_once()
        stream_mock.assert_called_once()
        self.assertIsInstance(out, self.mod.MlxLmStream)

    def test_build_params(self) -> None:
        model = self.mod.MlxLmModel(
            "id",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
        )
        params = model._build_params(
            GenerationSettings(
                temperature=0.5,
                top_p=0.1,
                top_k=2,
                max_new_tokens=3,
                stop_strings=["a"],
            )
        )
        self.assertEqual(
            params,
            {
                "temperature": 0.5,
                "top_p": 0.1,
                "top_k": 2,
                "max_tokens": 3,
                "stop": ["a"],
            },
        )


if __name__ == "__main__":
    from unittest import main

    main()
