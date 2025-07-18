from avalan.entities import GenerationSettings, TransformerEngineSettings
from avalan.model.nlp.text.vllm import VllmModel, VllmStream
from avalan.model.nlp.text.generation import TextGenerationModel
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from transformers import PreTrainedModel, PreTrainedTokenizerFast


class VllmStreamTestCase(IsolatedAsyncioTestCase):
    async def test_constructor_and_anext(self):
        iterator = iter(["a", "b"])
        stream = VllmStream(iterator)
        self.assertIs(stream._iterator, iterator)
        self.assertIs(stream._generator, iterator)

        self.assertEqual(await stream.__anext__(), "a")
        self.assertEqual(await stream.__anext__(), "b")
        with self.assertRaises(StopAsyncIteration):
            await stream.__anext__()


class VllmModelTestCase(IsolatedAsyncioTestCase):
    model_id = "test-model"

    def _make_model(self):
        settings = TransformerEngineSettings(
            auto_load_model=False, auto_load_tokenizer=False
        )
        return VllmModel(self.model_id, settings)

    def test_constructor_loads_model_and_tokenizer(self):
        llm_instance = MagicMock(spec=PreTrainedModel)
        vllm_mock = MagicMock()
        vllm_mock.LLM.return_value = llm_instance
        vllm_mock.SamplingParams = MagicMock()

        tokenizer_mock = MagicMock(spec=PreTrainedTokenizerFast)
        type(tokenizer_mock).chat_template = PropertyMock(return_value=None)
        type(tokenizer_mock).name_or_path = PropertyMock(
            return_value=self.model_id
        )

        with patch.dict("sys.modules", {"vllm": vllm_mock}):
            with (
                patch(
                    "avalan.model.transformer.AutoTokenizer.from_pretrained",
                    return_value=tokenizer_mock,
                ) as auto_tok,
                patch("avalan.model.nlp.text.vllm.LLM", vllm_mock.LLM),
            ):
                settings = TransformerEngineSettings()
                model = VllmModel(self.model_id, settings)

        self.assertIs(model._model, llm_instance)
        auto_tok.assert_called_once_with(
            self.model_id, use_fast=True, subfolder=""
        )
        vllm_mock.LLM.assert_called_once()

    def test_supports_sample_generation(self):
        model = self._make_model()
        self.assertFalse(model.supports_sample_generation)

    def test_load_model_without_vllm(self):
        model = self._make_model()
        with patch("avalan.model.nlp.text.vllm.LLM", None):
            with self.assertRaises(AssertionError):
                model._load_model()

    def test_load_model_with_vllm(self):
        model = self._make_model()
        llm_mock = MagicMock(return_value="llm")
        with patch("avalan.model.nlp.text.vllm.LLM", llm_mock):
            loaded = model._load_model()
        self.assertEqual(loaded, "llm")
        llm_mock.assert_called_once_with(
            model=self.model_id,
            tokenizer=self.model_id,
            trust_remote_code=False,
        )

    def test_build_sampling_params_without_vllm(self):
        model = self._make_model()
        with patch("avalan.model.nlp.text.vllm.SamplingParams", None):
            with self.assertRaises(AssertionError):
                model._build_sampling_params(GenerationSettings())

    def test_build_sampling_params_with_vllm(self):
        model = self._make_model()
        sp_mock = MagicMock(return_value="params")
        with patch("avalan.model.nlp.text.vllm.SamplingParams", sp_mock):
            params = model._build_sampling_params(GenerationSettings())
        self.assertEqual(params, "params")
        sp_mock.assert_called_once()

    def test_prompt(self):
        model = self._make_model()
        model._tokenizer = MagicMock(spec=PreTrainedTokenizerFast)
        type(model._tokenizer).chat_template = PropertyMock(return_value=None)
        model._tokenizer.decode.return_value = "decoded"

        with patch.object(
            TextGenerationModel,
            "_tokenize_input",
            return_value={"input_ids": [[1, 2]]},
        ) as tok:
            prompt = model._prompt("hello", "sys", None, None)
        tok.assert_called_once_with(
            "hello",
            "sys",
            context=None,
            tensor_format="pt",
            tool=None,
            chat_template_settings=None,
        )
        model._tokenizer.decode.assert_called_once_with(
            [1, 2], skip_special_tokens=False
        )
        self.assertEqual(prompt, "decoded")

    async def test_stream_generator(self):
        model = self._make_model()
        model._model = MagicMock()
        model._build_sampling_params = MagicMock(return_value="params")
        iterator = iter(["x", "y"])
        model._model.generate.return_value = iterator

        out = []
        async for chunk in model._stream_generator("p", GenerationSettings()):
            out.append(chunk)

        model._build_sampling_params.assert_called_once()
        model._model.generate.assert_called_once_with(
            ["p"], "params", stream=True
        )
        self.assertEqual(out, ["x", "y"])

    def test_string_output(self):
        model = self._make_model()
        model._model = MagicMock()
        model._build_sampling_params = MagicMock(return_value="params")
        result_obj = SimpleNamespace(outputs=[SimpleNamespace(text="done")])
        model._model.generate.return_value = [result_obj]
        out = model._string_output("p", GenerationSettings())
        self.assertEqual(out, "done")
        model._model.generate.assert_called_once_with(["p"], "params")

    async def test_call_use_async_generator_true(self):
        model = self._make_model()

        async def stream_gen(prompt, settings):
            return "stream"

        model._stream_generator = AsyncMock(side_effect=stream_gen)
        model._string_output = MagicMock()
        model._prompt = MagicMock(return_value="p")
        settings = GenerationSettings(use_async_generator=True)
        result = await model("input", settings=settings)
        model._stream_generator.assert_awaited_once()
        self.assertEqual(result, "stream")
        model._string_output.assert_not_called()

    async def test_call_use_async_generator_false(self):
        model = self._make_model()
        model._stream_generator = AsyncMock(return_value="stream")
        model._string_output = MagicMock(return_value="string")
        model._prompt = MagicMock(return_value="p")
        settings = GenerationSettings(use_async_generator=False)
        result = await model("input", settings=settings)
        model._stream_generator.assert_not_called()
        model._string_output.assert_called_once()
        self.assertEqual(result, "string")


if __name__ == "__main__":
    main()
