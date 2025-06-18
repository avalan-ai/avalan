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


class MlxLmModelAdditionalTestCase(IsolatedAsyncioTestCase):
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

    async def test_init_disables_auto_tokenizer(self) -> None:
        model = self.mod.MlxLmModel(
            "id",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=True
            ),
        )
        self.assertFalse(model._settings.auto_load_tokenizer)

    def test_load_model_sets_tokenizer(self) -> None:
        model = self.mod.MlxLmModel(
            "id",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
        )
        out = model._load_model()
        self.stub.load.assert_called_once_with("id")
        self.assertEqual(out, "model")
        self.assertEqual(model._tokenizer, "tokenizer")
        self.assertTrue(model._loaded_tokenizer)

    async def test_stream_generator(self) -> None:
        model = self.mod.MlxLmModel(
            "id",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
        )
        model._model = "m"
        model._tokenizer = "tok"
        self.stub.generate.side_effect = lambda *a, **kw: iter(["a", "b"])
        chunks = []
        async for c in model._stream_generator("p", GenerationSettings()):
            chunks.append(c)
        self.assertEqual(chunks, ["a", "b"])

    def test_string_output(self) -> None:
        model = self.mod.MlxLmModel(
            "id",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
        )
        model._model = "m"
        model._tokenizer = "tok"
        self.stub.generate.return_value = "text"
        out = model._string_output("p", GenerationSettings())
        self.assertEqual(out, "text")
        self.stub.generate.assert_called_with(
            "m",
            "tok",
            "p",
            temperature=1.0,
            top_p=1.0,
            top_k=50,
            max_tokens=None,
            stop=None,
        )

    async def test_call_string_path(self) -> None:
        model = self.mod.MlxLmModel(
            "id",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
        )
        model._model = "m"
        model._tokenizer = MagicMock()
        model._tokenizer.decode.return_value = "p"
        with (
            patch.object(
                self.mod.TextGenerationModel,
                "_tokenize_input",
                return_value={"input_ids": [1]},
            ) as tok_mock,
            patch.object(
                self.mod.MlxLmModel, "_string_output", return_value="out"
            ) as str_mock,
        ):
            result = await model(
                "hi",
                settings=GenerationSettings(use_async_generator=False),
            )
        tok_mock.assert_called_once()
        str_mock.assert_called_once_with("p", GenerationSettings(use_async_generator=False))
        self.assertEqual(result, "out")

    def test_supports_sample_generation(self) -> None:
        model = self.mod.MlxLmModel(
            "id",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
        )
        self.assertFalse(model.supports_sample_generation)
