import importlib
import sys
import types
from logging import getLogger
from unittest import IsolatedAsyncioTestCase
from unittest.mock import MagicMock, patch

from transformers.tokenization_utils_base import BatchEncoding

import avalan.model  # noqa: F401
from avalan.entities import GenerationSettings, TransformerEngineSettings
from avalan.model.response.text import TextGenerationResponse


class MlxLmStreamTestCase(IsolatedAsyncioTestCase):
    async def test_stream_iteration(self) -> None:
        stub = types.ModuleType("mlx_lm")
        stub.generate = MagicMock()
        stub.load = MagicMock()
        stub.stream_generate = MagicMock()
        sampler_mod = types.ModuleType("mlx_lm.sample_utils")
        sampler_mod.make_sampler = MagicMock()
        from avalan.model.nlp.text import generation as gen_mod

        sys.modules["avalan.model"].TextGenerationModel = (
            gen_mod.TextGenerationModel
        )
        with patch.dict(
            sys.modules,
            {"mlx_lm": stub, "mlx_lm.sample_utils": sampler_mod},
        ):
            from avalan.model.nlp.text.mlxlm import MlxLmStream

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

    def test_get_sampler_and_prompt_batch_encoding(self) -> None:
        model = self.mod.MlxLmModel(
            "id",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
            logger=getLogger(),
        )
        model._tokenizer = MagicMock()
        model._tokenizer.decode.return_value = "prompt"
        inputs = BatchEncoding({"input_ids": [[1]]})
        settings = GenerationSettings(
            temperature=0.3,
            top_p=0.2,
            top_k=4,
        )
        with patch("avalan.model.nlp.text.mlxlm.make_sampler") as make_sampler:
            sampler, prompt = model._get_sampler_and_prompt(
                inputs,
                settings,
                False,
            )
            make_sampler.assert_called_once_with(temp=0.3, top_p=0.2, top_k=4)
        self.assertEqual(prompt, "prompt")
        self.assertEqual(sampler, make_sampler.return_value)

    def test_get_sampler_and_prompt_tensor(self) -> None:
        model = self.mod.MlxLmModel(
            "id",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
            logger=getLogger(),
        )
        model._tokenizer = MagicMock()
        model._tokenizer.decode.return_value = "prompt"
        tensor = getattr(sys.modules["torch"], "tensor")
        inputs = tensor([[1, 2, 3]])
        settings = GenerationSettings()
        with patch("avalan.model.nlp.text.mlxlm.make_sampler") as make_sampler:
            sampler, prompt = model._get_sampler_and_prompt(
                inputs,
                settings,
                False,
            )
            make_sampler.assert_called_once_with(temp=1.0, top_p=1.0, top_k=50)

        prompt_ids = model._tokenizer.decode.call_args.args[0]
        self.assertEqual(len(prompt_ids), 3)
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

    def test_input_ids_from_inputs_validation(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "include input_ids"
        ):
            self.mod.MlxLmModel._input_ids_from_inputs({})
        with self.assertRaisesRegex(
            ValueError, "mapping or tensor"
        ):
            self.mod.MlxLmModel._input_ids_from_inputs("bad")

    def test_first_prompt_sequence_fallbacks(self) -> None:
        class BadShape:
            shape = [2, 1]

            def __getitem__(self, _):
                raise TypeError("bad index")

        bad_shape = BadShape()
        self.assertIs(
            self.mod.MlxLmModel._first_prompt_sequence(bad_shape), bad_shape
        )
        self.assertEqual(
            self.mod.MlxLmModel._first_prompt_sequence([[1], [2]]), [1]
        )
        self.assertEqual(self.mod.MlxLmModel._first_prompt_sequence([1]), [1])
        self.assertEqual(self.mod.MlxLmModel._first_prompt_sequence("x"), "x")

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


class MlxLmCoverageGapTestCase(IsolatedAsyncioTestCase):
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

    async def test_stream_handles_token_and_text_chunks(self) -> None:
        stream = self.mod.MlxLmStream(
            iter(
                [
                    self.mod.Token(token="tok"),
                    types.SimpleNamespace(text="txt"),
                ]
            )
        )

        self.assertEqual(await stream.__anext__(), "tok")
        self.assertEqual(await stream.__anext__(), "txt")

        only_text = []
        async for chunk in self.mod.MlxLmStream(
            iter([self.mod.Token(token="a"), types.SimpleNamespace(text="b")])
        )._generator:
            only_text.append(chunk)
        self.assertIsInstance(only_text[0], self.mod.Token)
        self.assertEqual(only_text[1], "b")

    async def test_stream_generator_normalizes_token_and_text(self) -> None:
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
                self.mod.Token(token="z"),
                types.SimpleNamespace(text="y"),
            ]
        )
        chunks = []
        async for chunk in model._stream_generator(
            {"input_ids": [[1]]}, GenerationSettings(), False
        ):
            chunks.append(chunk)
        self.assertEqual(chunks, ["z", "y"])

    def test_get_sampler_and_prompt_rejects_non_mapping_inputs(self) -> None:
        model = self.mod.MlxLmModel(
            "id",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
            logger=getLogger(),
        )
        model._tokenizer = MagicMock()

        with self.assertRaisesRegex(
            ValueError, "Expected tokenized inputs to be a mapping or tensor"
        ):
            model._get_sampler_and_prompt(
                object(),
                GenerationSettings(),
                False,
            )


class MlxImportGuardTestCase(IsolatedAsyncioTestCase):
    async def test_import_guard_behaviors(self) -> None:
        mod = importlib.import_module("avalan.model.nlp.text.mlxlm")
        mod._mlx_lm_import_is_safe.cache_clear()
        with patch.object(mod, "find_spec", return_value=None):
            self.assertFalse(mod._mlx_lm_import_is_safe())
        mod._mlx_lm_import_is_safe.cache_clear()
        self.assertIn("mlx-lm", mod._mlx_unavailable_message())

        with patch.object(mod, "find_spec", side_effect=ValueError("bad")):
            self.assertFalse(mod._mlx_lm_import_is_safe())
        mod._mlx_lm_import_is_safe.cache_clear()

        with patch.object(mod, "find_spec", return_value=True), patch.object(
            mod, "run", return_value=MagicMock(returncode=1)
        ):
            self.assertFalse(mod._mlx_lm_import_is_safe())
        mod._mlx_lm_import_is_safe.cache_clear()

        with patch.object(mod, "_mlx_lm_import_is_safe", return_value=False):
            with self.assertRaises(ModuleNotFoundError):
                mod._require_mlx_lm()
            with self.assertRaises(ModuleNotFoundError):
                mod.make_sampler()

    async def test_first_prompt_sequence_remaining_paths(self) -> None:
        mod = importlib.import_module("avalan.model.nlp.text.mlxlm")

        class OneDimensional:
            shape = (2,)

        one_dimensional = OneDimensional()
        self.assertIs(
            mod.MlxLmModel._first_prompt_sequence(one_dimensional),
            one_dimensional,
        )

        class KeyErrorIndex:
            def __getitem__(self, _):
                raise KeyError("k")

        key_error_index = KeyErrorIndex()
        self.assertIs(
            mod.MlxLmModel._first_prompt_sequence(key_error_index),
            key_error_index,
        )
