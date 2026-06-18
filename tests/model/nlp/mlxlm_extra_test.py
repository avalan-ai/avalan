import importlib
import sys
import types
from asyncio import CancelledError, create_task, sleep, to_thread
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from logging import getLogger
from threading import Event as ThreadEvent
from threading import get_ident
from unittest import IsolatedAsyncioTestCase
from unittest.mock import MagicMock, patch

import torch  # noqa: F401
from transformers.tokenization_utils_base import BatchEncoding

import avalan.model  # noqa: F401
from avalan.entities import (
    GenerationSettings,
    Token,
    TransformerEngineSettings,
)
from avalan.model.response.text import TextGenerationResponse
from avalan.model.stream import StreamItemKind


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
            items = [await stream.__anext__() for _ in range(6)]
            self.assertEqual(
                [item.kind for item in items],
                [
                    StreamItemKind.STREAM_STARTED,
                    StreamItemKind.ANSWER_DELTA,
                    StreamItemKind.ANSWER_DELTA,
                    StreamItemKind.ANSWER_DONE,
                    StreamItemKind.STREAM_COMPLETED,
                    StreamItemKind.STREAM_CLOSED,
                ],
            )
            self.assertEqual(
                [items[1].text_delta, items[2].text_delta],
                ["a", "b"],
            )
            with self.assertRaises(StopAsyncIteration):
                await stream.__anext__()
        del sys.modules["avalan.model"].TextGenerationModel

    async def test_stream_rejects_legacy_token_chunk(self) -> None:
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

            stream = MlxLmStream(iter([Token(token="z")]), use_executor=False)
            started = await stream.__anext__()
            errored = await stream.__anext__()

        del sys.modules["avalan.model"].TextGenerationModel

        self.assertIs(started.kind, StreamItemKind.STREAM_STARTED)
        self.assertIs(errored.kind, StreamItemKind.STREAM_ERRORED)
        assert isinstance(errored.data, dict)
        self.assertEqual(
            errored.data["message"],
            "unsupported legacy local stream item",
        )

    async def test_stream_rejects_legacy_token_subclass_chunk(self) -> None:
        stub = types.ModuleType("mlx_lm")
        stub.generate = MagicMock()
        stub.load = MagicMock()
        stub.stream_generate = MagicMock()
        sampler_mod = types.ModuleType("mlx_lm.sample_utils")
        sampler_mod.make_sampler = MagicMock()
        from avalan.model.nlp.text import generation as gen_mod

        class LegacyTokenSubclass(Token):
            pass

        sys.modules["avalan.model"].TextGenerationModel = (
            gen_mod.TextGenerationModel
        )
        with patch.dict(
            sys.modules,
            {"mlx_lm": stub, "mlx_lm.sample_utils": sampler_mod},
        ):
            from avalan.model.nlp.text.mlxlm import MlxLmStream

            stream = MlxLmStream(
                iter([LegacyTokenSubclass(token="legacy")]),
                use_executor=False,
            )
            started = await stream.__anext__()
            errored = await stream.__anext__()

        del sys.modules["avalan.model"].TextGenerationModel

        self.assertIs(started.kind, StreamItemKind.STREAM_STARTED)
        self.assertIs(errored.kind, StreamItemKind.STREAM_ERRORED)
        assert isinstance(errored.data, dict)
        self.assertEqual(
            errored.data["message"],
            "unsupported legacy local stream item",
        )

    async def test_stream_accepts_non_legacy_token_and_text_chunks(
        self,
    ) -> None:
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

            stream = MlxLmStream(
                iter(
                    [
                        types.SimpleNamespace(token="tok", id=3),
                        types.SimpleNamespace(text=" text"),
                    ]
                ),
                use_executor=False,
            )
            items = [item async for item in stream]

        del sys.modules["avalan.model"].TextGenerationModel

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(
            [
                item.text_delta
                for item in items
                if item.kind is StreamItemKind.ANSWER_DELTA
            ],
            ["tok", " text"],
        )
        self.assertEqual(items[1].metadata, {"token_id": 3})
        self.assertEqual(items[2].metadata, {})

    async def test_stream_factory_stays_on_worker_thread(self) -> None:
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
        owner_threads: list[int] = []
        next_threads: list[int] = []

        class ThreadBoundIterator:
            def __init__(self) -> None:
                self._owner_thread = get_ident()
                self._items = iter(["a", "b"])
                owner_threads.append(self._owner_thread)

            def __iter__(self) -> "ThreadBoundIterator":
                return self

            def __next__(self) -> str:
                thread_id = get_ident()
                next_threads.append(thread_id)
                if thread_id != self._owner_thread:
                    raise RuntimeError("iterator moved threads")
                return next(self._items)

        with patch.dict(
            sys.modules,
            {"mlx_lm": stub, "mlx_lm.sample_utils": sampler_mod},
        ):
            from avalan.model.nlp.text.mlxlm import MlxLmStream

            stream = MlxLmStream(lambda: ThreadBoundIterator())
            items = [await stream.__anext__() for _ in range(6)]
            self.assertEqual(
                [item.kind for item in items],
                [
                    StreamItemKind.STREAM_STARTED,
                    StreamItemKind.ANSWER_DELTA,
                    StreamItemKind.ANSWER_DELTA,
                    StreamItemKind.ANSWER_DONE,
                    StreamItemKind.STREAM_COMPLETED,
                    StreamItemKind.STREAM_CLOSED,
                ],
            )
            self.assertEqual(
                [items[1].text_delta, items[2].text_delta],
                ["a", "b"],
            )
            with self.assertRaises(StopAsyncIteration):
                await stream.__anext__()

        del sys.modules["avalan.model"].TextGenerationModel
        self.assertTrue(owner_threads)
        self.assertEqual(len(set(owner_threads)), 1)
        self.assertEqual(set(next_threads), set(owner_threads))

    async def test_stream_does_not_close_external_executor(self) -> None:
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

            executor = ThreadPoolExecutor(max_workers=1)
            try:
                stream = MlxLmStream(iter(["a"]), executor=executor)
                await stream.aclose()
                result = executor.submit(lambda: "alive").result()
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

        del sys.modules["avalan.model"].TextGenerationModel
        self.assertEqual(result, "alive")

    async def test_stream_close_short_circuits_iteration(self) -> None:
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

            stream = MlxLmStream(iter(["late"]))
            stream.close()
            items = [await stream.__anext__() for _ in range(3)]
            self.assertEqual(
                [item.kind for item in items],
                [
                    StreamItemKind.STREAM_STARTED,
                    StreamItemKind.STREAM_COMPLETED,
                    StreamItemKind.STREAM_CLOSED,
                ],
            )

        del sys.modules["avalan.model"].TextGenerationModel

    async def test_stream_placeholder_generator_is_empty(self) -> None:
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

            stream = MlxLmStream(iter([]))
            with self.assertRaises(StopAsyncIteration):
                await stream._generator.__anext__()

        del sys.modules["avalan.model"].TextGenerationModel

    async def test_stream_maps_token_metadata_and_skips_empty_chunks(
        self,
    ) -> None:
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

        token_chunk = types.SimpleNamespace(
            token="tok",
            id=7,
            probability=0.25,
            step=3,
            probability_distribution={"tok": 0.25},
            tokens=[
                types.SimpleNamespace(token="alt", id=8, probability=0.1),
                types.SimpleNamespace(token=None, id=9, probability=0.2),
                types.SimpleNamespace(token="plain", id=-1),
            ],
        )

        with patch.dict(
            sys.modules,
            {"mlx_lm": stub, "mlx_lm.sample_utils": sampler_mod},
        ):
            from avalan.model.nlp.text.mlxlm import MlxLmStream

            stream = MlxLmStream(
                iter([types.SimpleNamespace(), token_chunk]),
                use_executor=False,
            )
            items = [await stream.__anext__() for _ in range(5)]

        del sys.modules["avalan.model"].TextGenerationModel

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(items[1].text_delta, "tok")
        self.assertEqual(
            items[1].metadata,
            {
                "token_id": 7,
                "probability": 0.25,
                "step": 3,
                "probability_distribution": {"tok": 0.25},
                "tokens": [
                    {
                        "token": "alt",
                        "token_id": 8,
                        "probability": 0.1,
                    },
                    {"token": "plain"},
                ],
            },
        )

    async def test_stream_async_close_and_cancel_close_before_pull(
        self,
    ) -> None:
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

            close_stream = MlxLmStream(iter(["late"]))
            self.assertFalse(close_stream._closed)
            await close_stream.aclose()
            self.assertTrue(close_stream._closed)

            cancel_stream = MlxLmStream(iter(["late"]))
            self.assertFalse(cancel_stream._closed)
            await cancel_stream.cancel()
            self.assertTrue(cancel_stream._closed)

        del sys.modules["avalan.model"].TextGenerationModel

    async def test_stream_propagates_generation_error(self) -> None:
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

        class BrokenIterator:
            def __iter__(self) -> "BrokenIterator":
                return self

            def __next__(self) -> str:
                raise ValueError("bad generation")

        with patch.dict(
            sys.modules,
            {"mlx_lm": stub, "mlx_lm.sample_utils": sampler_mod},
        ):
            from avalan.model.nlp.text.mlxlm import MlxLmStream

            stream = MlxLmStream(lambda: BrokenIterator())
            started = await stream.__anext__()
            errored = await stream.__anext__()

            self.assertIs(started.kind, StreamItemKind.STREAM_STARTED)
            self.assertIs(errored.kind, StreamItemKind.STREAM_ERRORED)
            assert isinstance(errored.data, dict)
            self.assertEqual(errored.data["message"], "bad generation")

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
        calling_thread = get_ident()
        load_threads: list[int] = []

        def load_model(_model_id: str) -> tuple[str, str]:
            load_threads.append(get_ident())
            return "model", "tokenizer"

        self.stub.load.side_effect = load_model
        out = model._load_model()
        try:
            self.stub.load.assert_called_once_with("id")
            self.assertEqual(out, "model")
            self.assertEqual(model._tokenizer, "tokenizer")
            self.assertTrue(model._loaded_tokenizer)
            self.assertEqual(len(load_threads), 1)
            self.assertNotEqual(load_threads[0], calling_thread)
        finally:
            model.close()

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
        try:
            async for c in model._stream_generator(
                {"input_ids": [[1]]}, GenerationSettings(), False
            ):
                chunks.append(c)
            self.assertEqual(
                [chunk.kind for chunk in chunks],
                [
                    StreamItemKind.STREAM_STARTED,
                    StreamItemKind.ANSWER_DELTA,
                    StreamItemKind.ANSWER_DELTA,
                    StreamItemKind.ANSWER_DONE,
                    StreamItemKind.STREAM_COMPLETED,
                    StreamItemKind.STREAM_CLOSED,
                ],
            )
            self.assertEqual(
                [
                    chunk.text_delta
                    for chunk in chunks
                    if chunk.kind is StreamItemKind.ANSWER_DELTA
                ],
                ["a", "b"],
            )
        finally:
            model.close()

    async def test_stream_generator_uses_worker_thread(self) -> None:
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
        event_loop_thread = get_ident()
        factory_threads: list[int] = []
        next_threads: list[int] = []

        class ThreadRecorder:
            def __init__(self) -> None:
                self._items = iter(
                    [
                        types.SimpleNamespace(text="a"),
                        types.SimpleNamespace(text="b"),
                    ]
                )

            def __iter__(self) -> "ThreadRecorder":
                return self

            def __next__(self) -> object:
                next_threads.append(get_ident())
                return next(self._items)

        def stream_generate(*_args: object, **_kwargs: object) -> object:
            factory_threads.append(get_ident())
            return ThreadRecorder()

        self.stub.stream_generate.side_effect = stream_generate
        chunks = []
        try:
            async for c in model._stream_generator(
                {"input_ids": [[1]]}, GenerationSettings(), False
            ):
                chunks.append(c)

            self.assertEqual(
                [
                    chunk.text_delta
                    for chunk in chunks
                    if chunk.kind is StreamItemKind.ANSWER_DELTA
                ],
                ["a", "b"],
            )
            self.assertEqual(len(factory_threads), 1)
            worker_thread = factory_threads[0]
            self.assertNotEqual(worker_thread, event_loop_thread)
            self.assertEqual(set(next_threads), {worker_thread})
        finally:
            model.close()

    async def test_stream_generator_does_not_block_event_loop(self) -> None:
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
        next_started = ThreadEvent()
        release_next = ThreadEvent()

        class BlockingIterator:
            def __init__(self) -> None:
                self._sent = False

            def __iter__(self) -> "BlockingIterator":
                return self

            def __next__(self) -> object:
                if self._sent:
                    raise StopIteration
                self._sent = True
                next_started.set()
                release_next.wait(1)
                return types.SimpleNamespace(text="a")

        self.stub.stream_generate.return_value = BlockingIterator()
        stream = model._stream_generator(
            {"input_ids": [[1]]}, GenerationSettings(), False
        )
        second_item = None
        try:
            started = await stream.__anext__()
            self.assertIs(started.kind, StreamItemKind.STREAM_STARTED)
            second_item = create_task(stream.__anext__())
            self.assertTrue(await to_thread(next_started.wait, 1))
            loop_moved = False

            async def mark_loop_progress() -> None:
                nonlocal loop_moved
                await sleep(0)
                loop_moved = True

            marker = create_task(mark_loop_progress())
            await marker
            self.assertTrue(loop_moved)
            self.assertFalse(second_item.done())

            release_next.set()
            delta = await second_item
            self.assertIs(delta.kind, StreamItemKind.ANSWER_DELTA)
            self.assertEqual(delta.text_delta, "a")
        finally:
            release_next.set()
            if second_item is not None and not second_item.done():
                second_item.cancel()
                with suppress(CancelledError):
                    await second_item
            await stream.aclose()
            model.close()

    def test_input_ids_from_inputs_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "include input_ids"):
            self.mod.MlxLmModel._input_ids_from_inputs({})
        with self.assertRaisesRegex(ValueError, "mapping or tensor"):
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
        event_loop_thread = get_ident()
        generate_threads: list[int] = []

        def generate(*_args: object, **_kwargs: object) -> str:
            generate_threads.append(get_ident())
            return "text"

        self.stub.generate.side_effect = generate
        try:
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
            self.assertEqual(len(generate_threads), 1)
            self.assertNotEqual(generate_threads[0], event_loop_thread)
        finally:
            model.close()

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

    async def test_call_forwards_instructions_to_tokenizer(self) -> None:
        model = self.mod.MlxLmModel(
            "id",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
            logger=getLogger(),
        )
        model._model = "m"
        model._tokenizer = MagicMock()
        with (
            patch.object(
                self.mod.TextGenerationModel,
                "_tokenize_input",
                return_value={"input_ids": [[1]]},
            ) as tok_mock,
            patch.object(
                self.mod.MlxLmModel, "_string_output", return_value="out"
            ),
        ):
            await model(
                "hi",
                instructions="provider",
                settings=GenerationSettings(use_async_generator=False),
            )

        tok_mock.assert_called_once()
        self.assertEqual(tok_mock.call_args.kwargs["instructions"], "provider")

    async def test_call_tokenizes_non_callable_wrapper(self) -> None:
        class WrapperLikeTokenizer:
            chat_template = None
            bos_token = "<s>"

            def __init__(self) -> None:
                self.prompts: list[tuple[str, bool]] = []

            def encode(
                self, prompt: str, *, add_special_tokens: bool
            ) -> list[int]:
                self.prompts.append((prompt, add_special_tokens))
                return [7, 8]

            def decode(
                self, _token_ids: object, *, skip_special_tokens: bool
            ) -> str:
                return "prompt"

        model = self.mod.MlxLmModel(
            "id",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
            logger=getLogger(),
        )
        tokenizer = WrapperLikeTokenizer()
        model._model = "m"
        model._tokenizer = tokenizer
        with patch.object(
            self.mod.MlxLmModel, "_string_output", return_value="out"
        ) as str_mock:
            resp = await model(
                "hi",
                settings=GenerationSettings(use_async_generator=False),
            )

        self.assertEqual(tokenizer.prompts, [("None\n\nhi\n", True)])
        self.assertIs(resp._output_fn, str_mock)
        self.assertEqual(resp.input_token_count, 2)

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
                    types.SimpleNamespace(token="tok", id=7),
                    types.SimpleNamespace(text="txt"),
                ]
            )
        )

        items = [await stream.__anext__() for _ in range(6)]

        self.assertEqual(
            [
                item.text_delta
                for item in items
                if item.kind is StreamItemKind.ANSWER_DELTA
            ],
            ["tok", "txt"],
        )
        self.assertEqual(items[1].metadata["token_id"], 7)

    async def test_stream_raises_base_exception_chunk(self) -> None:
        stream = self.mod.MlxLmStream(iter([RuntimeError("bad chunk")]))

        started = await stream.__anext__()
        errored = await stream.__anext__()

        self.assertIs(started.kind, StreamItemKind.STREAM_STARTED)
        self.assertIs(errored.kind, StreamItemKind.STREAM_ERRORED)
        assert isinstance(errored.data, dict)
        self.assertEqual(errored.data["message"], "bad chunk")
        self.assertTrue(stream._closed)

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
                types.SimpleNamespace(token="z"),
                types.SimpleNamespace(text="y"),
            ]
        )
        chunks = []
        try:
            async for chunk in model._stream_generator(
                {"input_ids": [[1]]}, GenerationSettings(), False
            ):
                chunks.append(chunk)
            self.assertEqual(
                [
                    chunk.text_delta
                    for chunk in chunks
                    if chunk.kind is StreamItemKind.ANSWER_DELTA
                ],
                ["z", "y"],
            )
        finally:
            model.close()

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

        with (
            patch.object(mod, "find_spec", return_value=True),
            patch.object(mod, "run", return_value=MagicMock(returncode=1)),
        ):
            self.assertFalse(mod._mlx_lm_import_is_safe())
        mod._mlx_lm_import_is_safe.cache_clear()

        with (
            patch.object(mod, "find_spec", return_value=True),
            patch.object(mod, "run", side_effect=OSError("missing python")),
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
