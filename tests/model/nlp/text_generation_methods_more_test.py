import asyncio
from types import SimpleNamespace, ModuleType
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import MagicMock, patch

import torch

from avalan.entities import (
    GenerationSettings,
    QuantizationSettings,
    TransformerEngineSettings,
    Message,
    MessageRole,
)
from avalan.model.nlp.text.generation import TextGenerationModel


class SupportsTokenStreamingTestCase(TestCase):
    def test_property(self):
        model = TextGenerationModel(
            "m",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
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
        model = TextGenerationModel("m", settings)
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


class StreamGeneratorTestCase(IsolatedAsyncioTestCase):
    async def test_stream_generator(self):
        model = TextGenerationModel(
            "m",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
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

            async def __aiter__(self):
                while True:
                    val = await self.queue.get()
                    if val is self.stop_signal:
                        break
                    yield val

        class DummyThread:
            def __init__(self, target, name=None):
                self.target = target
                self.name = name
                self.ident = 1

            def start(self):
                self.target()

            def join(self):
                return

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


class StringOutputTestCase(TestCase):
    def test_string_output(self):
        model = TextGenerationModel(
            "m",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
        )
        model._tokenizer = MagicMock()
        model._tokenizer.decode.return_value = "ok"
        model._log = MagicMock()
        inputs = {"input_ids": torch.tensor([[1, 2]])}
        with patch.object(
            TextGenerationModel, "_generate_output", return_value=[[1, 2, 3, 4]]
        ) as gen:
            result = model._string_output(
                inputs, GenerationSettings(), None, False
            )
        gen.assert_called_once()
        model._tokenizer.decode.assert_called_once_with(
            [3, 4], skip_special_tokens=False
        )
        self.assertEqual(result, "ok")


class TokenGeneratorTestCase(IsolatedAsyncioTestCase):
    async def _setup(self, entmax_available: bool):
        settings = TransformerEngineSettings(
            auto_load_model=False, auto_load_tokenizer=False
        )
        model = TextGenerationModel("m", settings)
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
        model._messages.assert_called_once_with("in", "sys", None)
        tokenizer.assert_called_once()
        inputs.to.assert_called_once_with(model._model.device)
        self.assertIs(result, inputs)

    def test_tokenize_with_template(self):
        model, tokenizer = self._setup(True)
        inputs = MagicMock()
        tokenizer.apply_chat_template.return_value = inputs
        inputs.to.return_value = inputs
        result = model._tokenize_input("in", "sys", context=None)
        model._messages.assert_called_once_with("in", "sys", None)
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
        )
        result = model._messages("hi", "sys")
        self.assertEqual(
            result,
            [
                Message(role=MessageRole.SYSTEM, content="sys"),
                Message(role=MessageRole.USER, content="hi"),
            ],
        )

    def test_messages_from_list(self):
        model = TextGenerationModel(
            "m",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
        )
        messages = [
            Message(role=MessageRole.USER, content="a"),
            Message(role=MessageRole.ASSISTANT, content="b"),
        ]
        result = model._messages(messages, None)
        self.assertEqual(result, messages)


if __name__ == "__main__":
    main()
