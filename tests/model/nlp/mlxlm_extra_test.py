import sys
import importlib
import types
from unittest import IsolatedAsyncioTestCase
from unittest.mock import MagicMock, patch
from logging import getLogger

import avalan.model  # noqa: F401


from avalan.entities import GenerationSettings, TransformerEngineSettings
from avalan.model.response.text import TextGenerationResponse


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
        stub.stream_generate = MagicMock(return_value=iter([]))
        sampler_mod = types.ModuleType("mlx_lm.sample_utils")
        sampler_mod.make_sampler = MagicMock(return_value="sampler")
        self.patch = patch.dict(
            sys.modules,
            {"mlx_lm": stub, "mlx_lm.sample_utils": sampler_mod},
        )
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
            logger=getLogger(),
        )
        model._model = MagicMock()
        model._tokenizer = MagicMock()
        model._tokenizer.decode.return_value = "prompt"
        with (
            patch.object(
                self.mod.TextGenerationModel,
                "_tokenize_input",
                return_value={"input_ids": [[1]]},
            ) as tok_mock,
            patch.object(
                self.mod.MlxLmModel,
                "_stream_generator",
                return_value=self.mod.MlxLmStream(iter(["x"])),
            ) as stream_mock,
        ):
            resp = await model(
                "in", settings=GenerationSettings(use_async_generator=True)
            )
        tok_mock.assert_called_once()
        stream_mock.assert_not_called()
        self.assertIsInstance(resp, TextGenerationResponse)
        self.assertIs(resp._output_fn, stream_mock)
        self.assertEqual(resp.input_token_count, 1)
        self.assertFalse(resp._kwargs["settings"].do_sample)
        self.assertTrue(resp._use_async_generator)

    def test_get_sampler_and_prompt(self) -> None:
        model = self.mod.MlxLmModel(
            "id",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
            logger=getLogger(),
        )
        model._tokenizer = MagicMock()
        model._tokenizer.decode.return_value = "prompt"
        inputs = {"input_ids": [[1]]}
        with patch("avalan.model.nlp.text.mlxlm.make_sampler") as make_sampler:
            sampler, prompt = model._get_sampler_and_prompt(
                inputs,
                GenerationSettings(
                    temperature=0.5,
                    top_p=0.1,
                    top_k=2,
                ),
                True,
            )
            make_sampler.assert_called_once_with(temp=0.5, top_p=0.1, top_k=2)
        self.assertEqual(prompt, "prompt")
        self.assertEqual(sampler, make_sampler.return_value)


class MlxLmModelAdditionalTestCase(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        stub = types.ModuleType("mlx_lm")
        stub.load = MagicMock(return_value=("model", "tokenizer"))
        stub.generate = MagicMock(return_value="out")
        stub.stream_generate = MagicMock(return_value=iter([]))
        sampler_mod = types.ModuleType("mlx_lm.sample_utils")
        sampler_mod.make_sampler = MagicMock(return_value="sampler")
        self.patch = patch.dict(
            sys.modules,
            {"mlx_lm": stub, "mlx_lm.sample_utils": sampler_mod},
        )
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
            logger=getLogger(),
        )
        self.assertFalse(model._settings.auto_load_tokenizer)

    def test_load_model_sets_tokenizer(self) -> None:
        model = self.mod.MlxLmModel(
            "id",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
            logger=getLogger(),
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
            logger=getLogger(),
        )
        model._model = "m"
        model._tokenizer = MagicMock()
        model._tokenizer.decode.return_value = "p"
        self.stub.stream_generate.side_effect = lambda *a, **kw: iter(
            [
                MagicMock(text="a"),
                MagicMock(text="b"),
            ]
        )
        chunks = []
        async for c in model._stream_generator(
            {"input_ids": [[1]]}, GenerationSettings(), False
        ):
            chunks.append(c)
        self.assertEqual(chunks, ["a", "b"])

    def test_string_output(self) -> None:
        model = self.mod.MlxLmModel(
            "id",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
            logger=getLogger(),
        )
        model._model = "m"
        model._tokenizer = MagicMock()
        model._tokenizer.decode.return_value = "p"
        self.stub.generate.return_value = "text"
        out = model._string_output(
            {"input_ids": [[1]]}, GenerationSettings(), False
        )
        self.assertEqual(out, "text")
        self.stub.generate.assert_called_with(
            "m",
            model._tokenizer,
            "p",
            sampler="sampler",
            max_tokens=None,
        )

    async def test_call_string_path(self) -> None:
        model = self.mod.MlxLmModel(
            "id",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
            logger=getLogger(),
        )
        model._model = "m"
        model._tokenizer = MagicMock()
        model._tokenizer.decode.return_value = "p"
        with (
            patch.object(
                self.mod.TextGenerationModel,
                "_tokenize_input",
                return_value={"input_ids": [[1]]},
            ) as tok_mock,
            patch.object(
                self.mod.MlxLmModel, "_string_output", return_value="out"
            ) as str_mock,
        ):
            resp = await model(
                "hi",
                settings=GenerationSettings(use_async_generator=False),
            )
        tok_mock.assert_called_once()
        str_mock.assert_not_called()
        self.assertIsInstance(resp, TextGenerationResponse)
        self.assertIs(resp._output_fn, str_mock)
        self.assertEqual(resp.input_token_count, 1)
        self.assertFalse(resp._kwargs["settings"].do_sample)
        self.assertFalse(resp._use_async_generator)

    def test_supports_sample_generation(self) -> None:
        model = self.mod.MlxLmModel(
            "id",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
            logger=getLogger(),
        )
        self.assertFalse(model.supports_sample_generation)
