import asyncio
import time
from logging import getLogger
from threading import Event
from types import ModuleType, SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import MagicMock, patch

import torch

from avalan.entities import (
    GenerationSettings,
    Message,
    MessageRole,
    QuantizationSettings,
    TransformerEngineSettings,
)
from avalan.model.nlp.text import generation as generation_module
from avalan.model.nlp.text.generation import (
    TextGenerationModel,
    _configure_lossless_streamer_handoff,
    _is_event_loop_closed_error,
)


class EventLoopClosedErrorTestCase(TestCase):
    def test_identifies_event_loop_closed_runtime_error(self):
        self.assertTrue(
            _is_event_loop_closed_error(RuntimeError("Event loop is closed"))
        )


class LazyExternalProxyTestCase(TestCase):
    def test_async_text_iterator_streamer_loads_transformers_class(
        self,
    ) -> None:
        class LoadedStreamer:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        transformers_module = SimpleNamespace(
            AsyncTextIteratorStreamer=LoadedStreamer
        )

        with patch.object(
            generation_module,
            "import_module",
            return_value=transformers_module,
        ):
            streamer = generation_module.AsyncTextIteratorStreamer(
                "tokenizer", skip_prompt=True
            )

        self.assertIsInstance(streamer, LoadedStreamer)
        self.assertEqual(streamer.args, ("tokenizer",))
        self.assertEqual(streamer.kwargs, {"skip_prompt": True})

    def test_torch_function_proxies_delegate_to_lazy_modules(self) -> None:
        torch_module = SimpleNamespace(
            log_softmax=MagicMock(return_value="log"),
            softmax=MagicMock(return_value="soft"),
            topk=MagicMock(return_value="top"),
        )
        functional_module = SimpleNamespace(
            gumbel_softmax=MagicMock(return_value="gumbel")
        )

        def load_module(module_name: str):
            if module_name == "torch.nn.functional":
                return functional_module
            return torch_module

        with patch.object(
            generation_module, "import_module", side_effect=load_module
        ):
            self.assertEqual(
                generation_module.log_softmax("scores", dim=-1), "log"
            )
            self.assertEqual(
                generation_module.softmax("scores", dim=-1), "soft"
            )
            self.assertEqual(generation_module.topk("scores", 2), "top")
            self.assertEqual(
                generation_module.gumbel_softmax("scores", tau=1.0),
                "gumbel",
            )

        torch_module.log_softmax.assert_called_once_with("scores", dim=-1)
        torch_module.softmax.assert_called_once_with("scores", dim=-1)
        torch_module.topk.assert_called_once_with("scores", 2)
        functional_module.gumbel_softmax.assert_called_once_with(
            "scores", tau=1.0
        )


class SupportsTokenStreamingTestCase(TestCase):
    def test_property(self):
        model = TextGenerationModel(
            "m",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
            logger=getLogger(),
        )
        self.assertTrue(model.supports_token_streaming)


class LoadModelQuantizationTestCase(TestCase):
    def test_load_with_quantization_and_bitsandbytes(self):
        settings = TransformerEngineSettings(
            auto_load_model=False,
            auto_load_tokenizer=False,
            quantization=QuantizationSettings(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            ),
        )
        model = TextGenerationModel("m", settings, logger=getLogger())
        model._device = "cpu"
        loader = MagicMock()
        loader.from_pretrained.return_value = "loaded"
        with (
            patch(
                "avalan.model.nlp.text.generation.find_spec", return_value=True
            ),
            patch("transformers.BitsAndBytesConfig") as bnb,
            patch.object(TextGenerationModel, "_loaders", {"auto": loader}),
        ):
            result = model._load_model()
        self.assertEqual(result, "loaded")
        bnb.assert_called_once_with(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        loader.from_pretrained.assert_called_once()
        self.assertIs(
            loader.from_pretrained.call_args.kwargs["quantization_config"],
            bnb.return_value,
        )
        call_kwargs = loader.from_pretrained.call_args.kwargs
        self.assertEqual(call_kwargs["dtype"], "auto")
        self.assertNotIn("torch_dtype", call_kwargs)


class StreamGeneratorTestCase(IsolatedAsyncioTestCase):
    def _model(self) -> TextGenerationModel:
        model = TextGenerationModel(
            "m",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
            logger=getLogger(),
        )
        model._model = MagicMock()
        model._tokenizer = MagicMock()
        model._log = MagicMock()
        return model

    async def test_lossless_streamer_handoff_uses_bounded_queue(self):
        stop_event = Event()

        class DummyStreamer:
            def __init__(self):
                self.stop_signal = object()
                self.text_queue = asyncio.Queue()
                self.loop = asyncio.get_running_loop()

        streamer = DummyStreamer()
        queue_size = _configure_lossless_streamer_handoff(
            streamer, stop_event, max_queue_size=1
        )

        self.assertEqual(queue_size, 1)
        self.assertEqual(streamer.text_queue.maxsize, 1)

        await asyncio.to_thread(streamer.on_finalized_text, "a")
        self.assertEqual(await streamer.text_queue.get(), "a")

        ended = asyncio.create_task(
            asyncio.to_thread(streamer.on_finalized_text, "", stream_end=True)
        )
        self.assertEqual(await streamer.text_queue.get(), "")
        await ended
        self.assertIs(await streamer.text_queue.get(), streamer.stop_signal)

    async def test_lossless_streamer_handoff_blocks_until_consumed(self):
        stop_event = Event()

        class DummyStreamer:
            def __init__(self):
                self.stop_signal = object()
                self.text_queue = asyncio.Queue()
                self.loop = asyncio.get_running_loop()

        streamer = DummyStreamer()
        _configure_lossless_streamer_handoff(
            streamer, stop_event, max_queue_size=1
        )
        await asyncio.to_thread(streamer.on_finalized_text, "first")

        second = asyncio.create_task(
            asyncio.to_thread(streamer.on_finalized_text, "second")
        )
        await asyncio.sleep(0.01)

        self.assertFalse(second.done())
        self.assertEqual(await streamer.text_queue.get(), "first")
        await second
        self.assertEqual(await streamer.text_queue.get(), "second")

    async def test_lossless_streamer_handoff_stops_blocked_put(self):
        stop_event = Event()

        class DummyStreamer:
            def __init__(self):
                self.stop_signal = object()
                self.text_queue = asyncio.Queue()
                self.loop = asyncio.get_running_loop()

        streamer = DummyStreamer()
        _configure_lossless_streamer_handoff(
            streamer, stop_event, max_queue_size=1
        )
        await asyncio.to_thread(streamer.on_finalized_text, "first")
        stopped = asyncio.create_task(
            asyncio.to_thread(streamer.on_finalized_text, "second")
        )

        await asyncio.sleep(0.01)
        stop_event.set()

        with self.assertRaisesRegex(RuntimeError, "Event loop is closed"):
            await stopped

    async def test_lossless_streamer_handoff_skips_unknown_streamer(self):
        class DummyStreamer:
            pass

        streamer = DummyStreamer()

        self.assertIsNone(
            _configure_lossless_streamer_handoff(streamer, Event())
        )
        self.assertFalse(hasattr(streamer, "on_finalized_text"))

    async def test_stream_generator(self):
        model = TextGenerationModel(
            "m",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
            logger=getLogger(),
        )
        model._model = MagicMock()
        model._tokenizer = MagicMock()
        model._log = MagicMock()

        class DummyStreamer:
            def __init__(self, *args, **kwargs):
                self.stop_signal = object()
                self.queue = asyncio.Queue()

            def put(self, value):
                self.queue.put_nowait(value)

            def __aiter__(self):
                return self

            async def __anext__(self):
                val = await self.queue.get()
                if val is self.stop_signal:
                    raise StopAsyncIteration
                return val

        class DummyThread:
            def __init__(self, target, name=None, daemon=None):
                self.target = target
                self.name = name
                self.daemon = daemon
                self.ident = 1
                self._alive = False

            def start(self):
                self._alive = True
                self.target()
                self._alive = False

            def is_alive(self):
                return self._alive

        def gen_side_effect(*args, streamer=None, **kwargs):
            streamer.put("a")
            streamer.put("b")
            streamer.put(streamer.stop_signal)

        with (
            patch(
                "avalan.model.nlp.text.generation.AsyncTextIteratorStreamer",
                DummyStreamer,
            ),
            patch("avalan.model.nlp.text.generation.Thread", DummyThread),
            patch.object(
                TextGenerationModel,
                "_generate_output",
                side_effect=gen_side_effect,
            ) as gen,
        ):
            gen_settings = GenerationSettings(max_new_tokens=2)
            inputs = {"input_ids": torch.tensor([[1, 2]])}
            out = []
            async for token in model._stream_generator(
                inputs, gen_settings, None, False
            ):
                out.append(token)

        gen.assert_called_once()
        self.assertEqual(out, ["a", "b"])

    async def test_stream_generator_uses_bounded_handoff(self):
        model = self._model()

        class DummyStreamer:
            def __init__(self, *args, **kwargs):
                self.stop_signal = object()
                self.text_queue = asyncio.Queue()
                self.loop = asyncio.get_running_loop()

            def __aiter__(self):
                return self

            async def __anext__(self):
                val = await self.text_queue.get()
                if val is self.stop_signal:
                    raise StopAsyncIteration
                return val

        def gen_side_effect(*args, streamer=None, **kwargs):
            streamer.on_finalized_text("a")
            streamer.on_finalized_text("", stream_end=True)

        with (
            patch(
                "avalan.model.nlp.text.generation.AsyncTextIteratorStreamer",
                DummyStreamer,
            ),
            patch.object(
                TextGenerationModel,
                "_generate_output",
                side_effect=gen_side_effect,
            ) as gen,
        ):
            chunks = []
            async for chunk in model._stream_generator(
                {"input_ids": torch.tensor([[1, 2]])},
                GenerationSettings(max_new_tokens=2),
                None,
                False,
            ):
                chunks.append(chunk)

        gen.assert_called_once()
        self.assertEqual(chunks, ["a", ""])
        model._log.assert_any_call(
            "Created generator async text token streamer with 64 queued chunks"
        )

    async def test_stream_generator_stops_thread_when_closed(self):
        model = TextGenerationModel(
            "m",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
            logger=getLogger(),
        )
        model._model = MagicMock()
        model._tokenizer = MagicMock()
        model._log = MagicMock()
        stopped = Event()

        class DummyStreamer:
            def __init__(self, *args, **kwargs):
                self.stop_signal = object()
                self.queue = asyncio.Queue()
                self.loop = asyncio.get_running_loop()

            def put(self, value):
                self.loop.call_soon_threadsafe(self.queue.put_nowait, value)

            def on_finalized_text(self, text, stream_end=False):
                self.loop.call_soon_threadsafe(self.queue.put_nowait, text)
                if stream_end:
                    self.loop.call_soon_threadsafe(
                        self.queue.put_nowait, self.stop_signal
                    )

            def __aiter__(self):
                return self

            async def __anext__(self):
                val = await self.queue.get()
                if val is self.stop_signal:
                    raise StopAsyncIteration
                return val

        def gen_side_effect(*args, streamer=None, **kwargs):
            stopping_criterias = args[2]
            streamer.put("a")
            while not stopping_criterias[-1](None, None):
                time.sleep(0.01)
            stopped.set()
            streamer.put(streamer.stop_signal)

        with (
            patch(
                "avalan.model.nlp.text.generation.AsyncTextIteratorStreamer",
                DummyStreamer,
            ),
            patch.object(
                TextGenerationModel,
                "_generate_output",
                side_effect=gen_side_effect,
            ),
        ):
            gen_settings = GenerationSettings(max_new_tokens=2)
            inputs = {"input_ids": torch.tensor([[1, 2]])}
            stream = model._stream_generator(inputs, gen_settings, None, False)
            self.assertEqual(await stream.__anext__(), "a")
            await stream.aclose()

        self.assertTrue(stopped.wait(1))

    async def test_stream_generator_raises_worker_error(self):
        model = TextGenerationModel(
            "m",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
            logger=getLogger(),
        )
        model._model = MagicMock()
        model._tokenizer = MagicMock()
        model._log = MagicMock()

        class DummyStreamer:
            def __init__(self, *args, **kwargs):
                self.stop_signal = object()
                self.queue = asyncio.Queue()
                self.loop = asyncio.get_running_loop()

            def on_finalized_text(self, text, stream_end=False):
                self.loop.call_soon_threadsafe(self.queue.put_nowait, text)
                if stream_end:
                    self.loop.call_soon_threadsafe(
                        self.queue.put_nowait, self.stop_signal
                    )

            def __aiter__(self):
                return self

            async def __anext__(self):
                val = await self.queue.get()
                if val is self.stop_signal:
                    raise StopAsyncIteration
                return val

        with (
            patch(
                "avalan.model.nlp.text.generation.AsyncTextIteratorStreamer",
                DummyStreamer,
            ),
            patch.object(
                TextGenerationModel,
                "_generate_output",
                side_effect=ValueError("bad generation"),
            ),
        ):
            gen_settings = GenerationSettings(max_new_tokens=2)
            inputs = {"input_ids": torch.tensor([[1, 2]])}
            stream = model._stream_generator(inputs, gen_settings, None, False)
            with self.assertRaisesRegex(ValueError, "bad generation"):
                await stream.__anext__()

    async def test_stream_generator_ignores_closed_loop_after_stop(self):
        model = self._model()

        class DummyStreamer:
            def __init__(self, *args, **kwargs):
                pass

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise TimeoutError()

        class DummyThread:
            def __init__(self, target, name=None, daemon=None):
                self.target = target
                self.name = name
                self.daemon = daemon
                self.ident = 1
                self._alive = False

            def start(self):
                self._alive = True
                self.target()
                self._alive = False

            def is_alive(self):
                return self._alive

        def gen_side_effect(*args, **kwargs):
            stopping_criterias = args[2]
            stopping_criterias[-1]._event.set()
            raise RuntimeError("Event loop is closed")

        with (
            patch(
                "avalan.model.nlp.text.generation.AsyncTextIteratorStreamer",
                DummyStreamer,
            ),
            patch("avalan.model.nlp.text.generation.Thread", DummyThread),
            patch.object(
                TextGenerationModel,
                "_generate_output",
                side_effect=gen_side_effect,
            ),
        ):
            chunks = []
            async for chunk in model._stream_generator(
                {"input_ids": torch.tensor([[1, 2]])},
                GenerationSettings(max_new_tokens=2),
                None,
                False,
            ):
                chunks.append(chunk)

        self.assertEqual(chunks, [])

    async def test_stream_generator_raises_worker_error_after_timeout(self):
        model = self._model()
        threads = []

        class DummyStreamer:
            def __init__(self, *args, **kwargs):
                pass

            def on_finalized_text(self, text, stream_end=False):
                pass

            def __aiter__(self):
                return self

            async def __anext__(self):
                thread = threads[0]
                thread.target()
                thread._alive = False
                raise TimeoutError()

        class DummyThread:
            def __init__(self, target, name=None, daemon=None):
                self.target = target
                self.name = name
                self.daemon = daemon
                self.ident = 1
                self._alive = False
                threads.append(self)

            def start(self):
                self._alive = True

            def is_alive(self):
                return self._alive

        with (
            patch(
                "avalan.model.nlp.text.generation.AsyncTextIteratorStreamer",
                DummyStreamer,
            ),
            patch("avalan.model.nlp.text.generation.Thread", DummyThread),
            patch.object(
                TextGenerationModel,
                "_generate_output",
                side_effect=ValueError("late generation failure"),
            ),
        ):
            stream = model._stream_generator(
                {"input_ids": torch.tensor([[1, 2]])},
                GenerationSettings(max_new_tokens=2),
                None,
                False,
            )
            with self.assertRaisesRegex(ValueError, "late generation failure"):
                await stream.__anext__()

    async def test_stream_generator_raises_worker_error_after_chunk(self):
        model = self._model()
        threads = []

        class DummyStreamer:
            def __init__(self, *args, **kwargs):
                pass

            def on_finalized_text(self, text, stream_end=False):
                pass

            def __aiter__(self):
                return self

            async def __anext__(self):
                thread = threads[0]
                thread.target()
                thread._alive = False
                return "partial"

        class DummyThread:
            def __init__(self, target, name=None, daemon=None):
                self.target = target
                self.name = name
                self.daemon = daemon
                self.ident = 1
                self._alive = False
                threads.append(self)

            def start(self):
                self._alive = True

            def is_alive(self):
                return self._alive

        with (
            patch(
                "avalan.model.nlp.text.generation.AsyncTextIteratorStreamer",
                DummyStreamer,
            ),
            patch("avalan.model.nlp.text.generation.Thread", DummyThread),
            patch.object(
                TextGenerationModel,
                "_generate_output",
                side_effect=ValueError("chunk generation failure"),
            ),
        ):
            stream = model._stream_generator(
                {"input_ids": torch.tensor([[1, 2]])},
                GenerationSettings(max_new_tokens=2),
                None,
                False,
            )
            with self.assertRaisesRegex(
                ValueError, "chunk generation failure"
            ):
                await stream.__anext__()

    async def test_stream_generator_raises_worker_error_after_stop(self):
        model = self._model()
        threads = []

        class DummyStreamer:
            def __init__(self, *args, **kwargs):
                pass

            def on_finalized_text(self, text, stream_end=False):
                pass

            def __aiter__(self):
                return self

            async def __anext__(self):
                thread = threads[0]
                thread.target()
                thread._alive = False
                raise StopAsyncIteration

        class DummyThread:
            def __init__(self, target, name=None, daemon=None):
                self.target = target
                self.name = name
                self.daemon = daemon
                self.ident = 1
                self._alive = False
                threads.append(self)

            def start(self):
                self._alive = True

            def is_alive(self):
                return self._alive

        with (
            patch(
                "avalan.model.nlp.text.generation.AsyncTextIteratorStreamer",
                DummyStreamer,
            ),
            patch("avalan.model.nlp.text.generation.Thread", DummyThread),
            patch.object(
                TextGenerationModel,
                "_generate_output",
                side_effect=ValueError("stopped generation failure"),
            ),
        ):
            stream = model._stream_generator(
                {"input_ids": torch.tensor([[1, 2]])},
                GenerationSettings(max_new_tokens=2),
                None,
                False,
            )
            with self.assertRaisesRegex(
                ValueError, "stopped generation failure"
            ):
                await stream.__anext__()

    async def test_stream_generator_reraises_finish_stream_runtime_error(self):
        model = self._model()

        class DummyStreamer:
            def __init__(self, *args, **kwargs):
                pass

            def on_finalized_text(self, text, stream_end=False):
                raise RuntimeError("stream finalization failed")

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        class DummyThread:
            def __init__(self, target, name=None, daemon=None):
                self.target = target
                self.name = name
                self.daemon = daemon
                self.ident = 1

            def start(self):
                self.target()

            def is_alive(self):
                return False

        with (
            patch(
                "avalan.model.nlp.text.generation.AsyncTextIteratorStreamer",
                DummyStreamer,
            ),
            patch("avalan.model.nlp.text.generation.Thread", DummyThread),
            patch.object(
                TextGenerationModel,
                "_generate_output",
                side_effect=RuntimeError("worker failed"),
            ),
        ):
            stream = model._stream_generator(
                {"input_ids": torch.tensor([[1, 2]])},
                GenerationSettings(max_new_tokens=2),
                None,
                False,
            )
            with self.assertRaisesRegex(
                RuntimeError, "stream finalization failed"
            ):
                await stream.__anext__()


class StringOutputTestCase(TestCase):
    def test_string_output(self):
        model = TextGenerationModel(
            "m",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
            logger=getLogger(),
        )
        model._tokenizer = MagicMock()
        model._tokenizer.decode.return_value = "ok"
        model._log = MagicMock()
        inputs = {"input_ids": torch.tensor([[1, 2]])}
        with patch.object(
            TextGenerationModel,
            "_generate_output",
            return_value=[[1, 2, 3, 4]],
        ) as gen:
            result = model._string_output(
                inputs, GenerationSettings(), None, False
            )
        gen.assert_called_once()
        model._tokenizer.decode.assert_called_once_with(
            [3, 4], skip_special_tokens=False
        )
        self.assertEqual(result, "ok")

    def test_string_output_skip_special_tokens_true(self):
        model = TextGenerationModel(
            "m",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
            logger=getLogger(),
        )
        model._tokenizer = MagicMock()
        model._tokenizer.decode.return_value = "ok"
        model._log = MagicMock()
        inputs = {"input_ids": torch.tensor([[1, 2]])}
        with patch.object(
            TextGenerationModel,
            "_generate_output",
            return_value=[[1, 2, 3, 4]],
        ) as gen:
            result = model._string_output(
                inputs, GenerationSettings(), None, True
            )
        gen.assert_called_once()
        model._tokenizer.decode.assert_called_once_with(
            [3, 4], skip_special_tokens=True
        )
        self.assertEqual(result, "ok")


class TokenGeneratorTestCase(IsolatedAsyncioTestCase):
    async def _setup(self, entmax_available: bool):
        settings = TransformerEngineSettings(
            auto_load_model=False, auto_load_tokenizer=False
        )
        model = TextGenerationModel("m", settings, logger=getLogger())
        model._tokenizer = MagicMock()
        model._tokenizer.decode.side_effect = (
            lambda i, skip_special_tokens=False: f"t{i}"
        )
        model._log = MagicMock()

        outputs = SimpleNamespace(
            sequences=torch.tensor([[5, 1, 2]]),
            scores=[
                torch.tensor([[2.0, 1.0, 0.0]]),
                torch.tensor([[0.5, 0.5, 0.0]]),
            ],
        )

        def gen_side_effect(*args, **kwargs):
            return outputs

        patches = [
            patch.object(
                TextGenerationModel,
                "_generate_output",
                side_effect=gen_side_effect,
            ),
        ]

        if entmax_available:
            em = ModuleType("entmax")
            em.entmax15 = MagicMock(return_value=torch.tensor([0.6, 0.3, 0.1]))
            patches.append(patch.dict("sys.modules", {"entmax": em}))
            patches.append(
                patch(
                    "avalan.model.nlp.text.generation.find_spec",
                    return_value=True,
                )
            )
            patches.append(
                patch("avalan.model.nlp.text.generation.softmax", MagicMock())
            )
        else:
            patches.append(
                patch(
                    "avalan.model.nlp.text.generation.find_spec",
                    return_value=False,
                )
            )
            soft = MagicMock(return_value=torch.tensor([0.7, 0.2, 0.1]))
            patches.append(
                patch("avalan.model.nlp.text.generation.softmax", soft)
            )
            patches.append(
                patch.dict("sys.modules", {"entmax": ModuleType("entmax")})
            )

        return model, outputs, patches

    async def test_token_generator_with_entmax(self):
        model, outputs, patches = await self._setup(True)
        for p in patches:
            p.start()
        try:
            settings = GenerationSettings(max_new_tokens=2, temperature=1.0)
            inputs = {"input_ids": torch.tensor([[5]])}
            result = []
            async for t in model._token_generator(
                inputs,
                settings,
                None,
                False,
                pick=0,
                probability_distribution="entmax",
            ):
                result.append(t)
        finally:
            for p in reversed(patches):
                p.stop()
        self.assertEqual(len(result), 2)
        self.assertEqual([t.id for t in result], [1, 2])
        self.assertEqual([t.token for t in result], ["t1", "t2"])
        for got, expected in zip([t.probability for t in result], [0.3, 0.1]):
            self.assertAlmostEqual(got, expected, places=3)
        self.assertTrue(
            all(t.probability_distribution == "entmax" for t in result)
        )

    async def test_token_generator_without_entmax(self):
        model, outputs, patches = await self._setup(False)
        for p in patches:
            p.start()
        try:
            settings = GenerationSettings(max_new_tokens=2, temperature=1.0)
            inputs = {"input_ids": torch.tensor([[5]])}
            result = []
            async for t in model._token_generator(
                inputs,
                settings,
                None,
                False,
                pick=0,
                probability_distribution="entmax",
            ):
                result.append(t)
        finally:
            for p in reversed(patches):
                p.stop()
        self.assertEqual(len(result), 2)
        self.assertEqual([t.id for t in result], [1, 2])
        self.assertTrue(
            all(t.probability_distribution == "entmax" for t in result)
        )


class TokenizeInputTestCase(TestCase):
    def _setup(self, has_template: bool):
        model = TextGenerationModel(
            "m",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
            logger=getLogger(),
        )
        model._model = MagicMock()
        model._model.device = "cpu"
        model._log = MagicMock()
        model._messages = MagicMock(
            return_value=[Message(role=MessageRole.USER, content="hi")]
        )
        tokenizer = MagicMock()
        tokenizer.chat_template = "tpl" if has_template else None
        model._tokenizer = tokenizer
        return model, tokenizer

    def test_tokenize_no_template(self):
        model, tokenizer = self._setup(False)
        inputs = MagicMock()
        tokenizer.return_value = inputs
        inputs.to.return_value = inputs
        result = model._tokenize_input("in", "sys", context=None)
        model._messages.assert_called_once_with("in", "sys", None, None)
        tokenizer.assert_called_once()
        inputs.to.assert_called_once_with(model._model.device)
        self.assertIs(result, inputs)

    def test_tokenize_with_template(self):
        model, tokenizer = self._setup(True)
        inputs = MagicMock()
        tokenizer.apply_chat_template.return_value = inputs
        inputs.to.return_value = inputs
        result = model._tokenize_input("in", "sys", context=None)
        model._messages.assert_called_once_with("in", "sys", None, None)
        tokenizer.apply_chat_template.assert_called_once()
        inputs.to.assert_called_once_with(model._model.device)
        self.assertIs(result, inputs)


class MessagesTestCase(TestCase):
    def test_messages_from_string_and_system(self):
        model = TextGenerationModel(
            "m",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
            logger=getLogger(),
        )
        result = model._messages("hi", "sys")
        self.assertEqual(
            result,
            [
                Message(role=MessageRole.SYSTEM, content="sys"),
                Message(role=MessageRole.USER, content="hi"),
            ],
        )

    def test_messages_from_string_and_developer(self):
        model = TextGenerationModel(
            "m",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
            logger=getLogger(),
        )
        result = model._messages("hi", None, "dev")
        self.assertEqual(
            result,
            [
                Message(role=MessageRole.DEVELOPER, content="dev"),
                Message(role=MessageRole.USER, content="hi"),
            ],
        )

    def test_messages_from_string_system_and_developer(self):
        model = TextGenerationModel(
            "m",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
            logger=getLogger(),
        )
        result = model._messages("hi", "sys", "dev")
        self.assertEqual(
            result,
            [
                Message(role=MessageRole.SYSTEM, content="sys"),
                Message(role=MessageRole.DEVELOPER, content="dev"),
                Message(role=MessageRole.USER, content="hi"),
            ],
        )

    def test_messages_from_list(self):
        model = TextGenerationModel(
            "m",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
            logger=getLogger(),
        )
        messages = [
            Message(role=MessageRole.USER, content="a"),
            Message(role=MessageRole.ASSISTANT, content="b"),
        ]
        result = model._messages(messages, None)
        self.assertEqual(result, messages)

    def test_messages_from_list_with_developer(self):
        model = TextGenerationModel(
            "m",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
            logger=getLogger(),
        )
        messages = [
            Message(role=MessageRole.USER, content="a"),
            Message(role=MessageRole.ASSISTANT, content="b"),
        ]
        result = model._messages(messages, None, "dev")
        self.assertEqual(
            result,
            [
                Message(role=MessageRole.DEVELOPER, content="dev"),
                *messages,
            ],
        )


if __name__ == "__main__":
    main()
