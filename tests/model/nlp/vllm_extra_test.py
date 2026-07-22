from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import MagicMock, PropertyMock, patch

from transformers import PreTrainedModel, PreTrainedTokenizerFast

import avalan.model.nlp.text.vllm as vllm_module
from avalan.entities import (
    GenerationSettings,
    Token,
    TransformerEngineSettings,
)
from avalan.model.nlp.text.generation import TextGenerationModel
from avalan.model.nlp.text.local_protocol import (
    LOCAL_STRUCTURED_OUTPUT_PROTOCOL,
)
from avalan.model.nlp.text.vllm import (
    VllmModel,
    VllmStream,
    _llm_class,
    _sampling_params_class,
    _vllm_attribute,
)
from avalan.model.response.text import TextGenerationResponse
from avalan.model.stream import (
    StreamItemKind,
    StreamProviderEvent,
    accumulate_canonical_stream_items,
)


class VllmStreamTestCase(IsolatedAsyncioTestCase):
    async def test_constructor_and_anext(self):
        iterator = iter(["a", "b"])
        stream = VllmStream(iterator)
        self.assertIs(stream._iterator, iterator)

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

    async def test_sync_iterator_does_not_advertise_cancellation(self):
        stream = VllmStream(iter(["a"]))

        started = await stream.__anext__()

        capabilities = started.metadata["capabilities"]
        assert isinstance(capabilities, dict)
        self.assertFalse(capabilities["supports_cancellation"])

    async def test_async_iterator_advertises_cancellation(self):
        async def generator():
            yield "a"

        stream = VllmStream(generator())

        started = await stream.__anext__()

        capabilities = started.metadata["capabilities"]
        assert isinstance(capabilities, dict)
        self.assertTrue(capabilities["supports_cancellation"])

    async def test_disabled_tool_parser_preserves_literal_output(self):
        text = '<tool_call name="lookup">{"q":"v"}</tool_call>'
        stream = VllmStream(iter([text]))

        items = [item async for item in stream]

        started_capabilities = items[0].metadata["capabilities"]
        assert isinstance(started_capabilities, dict)
        self.assertFalse(started_capabilities["supports_tool_calls"])
        self.assertEqual(
            "".join(
                item.text_delta or ""
                for item in items
                if item.kind is StreamItemKind.ANSWER_DELTA
            ),
            text,
        )
        self.assertFalse(
            any(
                item.kind
                in {
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    StreamItemKind.TOOL_CALL_READY,
                    StreamItemKind.TOOL_CALL_DONE,
                }
                for item in items
            )
        )

    async def test_placeholder_generator_is_empty(self):
        stream = VllmStream(iter([]))

        with self.assertRaises(StopAsyncIteration):
            await stream._generator.__anext__()

    async def test_chunk_text_uses_text_attribute_and_string_fallback(self):
        async def generator():
            yield SimpleNamespace(text="attr")
            yield 42

        stream = VllmStream(generator())

        items = [await stream.__anext__() for _ in range(6)]

        self.assertEqual(
            [items[1].text_delta, items[2].text_delta],
            ["attr", "42"],
        )

    def test_chunk_text_compatibility_helper(self):
        self.assertEqual(
            VllmStream._chunk_text(SimpleNamespace(text="attr")), "attr"
        )

    async def test_stream_skips_none_text_and_empty_delta(self):
        stream = VllmStream(iter(["none", "empty"]))

        with patch.object(
            VllmStream,
            "_chunk_text_and_metadata",
            side_effect=[(None, {}), ("", {})],
        ):
            items = [item async for item in stream]

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )

    async def test_stream_flushes_pending_parser_text(self):
        stream = VllmStream(iter(["<"]))

        items = [item async for item in stream]

        self.assertEqual(
            [
                item.text_delta
                for item in items
                if item.kind is StreamItemKind.ANSWER_DELTA
            ],
            ["<"],
        )

    async def test_stream_rejects_legacy_token_chunk(self) -> None:
        stream = VllmStream(iter([Token(token="legacy")]))

        started = await stream.__anext__()
        errored = await stream.__anext__()

        self.assertIs(started.kind, StreamItemKind.STREAM_STARTED)
        self.assertIs(errored.kind, StreamItemKind.STREAM_ERRORED)
        assert isinstance(errored.data, dict)
        self.assertEqual(
            errored.data["message"],
            "unsupported legacy local stream item",
        )

    async def test_stream_rejects_legacy_token_subclass_chunk(self) -> None:
        class LegacyTokenSubclass(Token):
            pass

        stream = VllmStream(iter([LegacyTokenSubclass(token="legacy")]))

        started = await stream.__anext__()
        errored = await stream.__anext__()

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
        stream = VllmStream(
            iter(
                [
                    SimpleNamespace(token="tok", id=3),
                    SimpleNamespace(text=" text"),
                ]
            )
        )

        items = [item async for item in stream]

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

    async def test_stream_parses_split_local_events(self):
        stream = VllmStream(
            iter(
                [
                    "a<think>r",
                    '</think><tool_call id="lookup-call" name="lookup">{}',
                    "</tool_call>b",
                ]
            ),
            local_structured_output_protocol=(
                LOCAL_STRUCTURED_OUTPUT_PROTOCOL
            ),
        )
        items = [item async for item in stream]

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(
            accumulate_canonical_stream_items(items).answer_text,
            'a<think>r</think><tool_call id="lookup-call" '
            'name="lookup">{}</tool_call>b',
        )

    async def test_stream_preserves_token_metadata(self):
        stream = VllmStream(
            iter(
                [
                    SimpleNamespace(
                        token="tok",
                        id=7,
                        probability=0.25,
                        step=3,
                        probability_distribution="softmax",
                        tokens=[
                            SimpleNamespace(token="alt", id=8, probability=0.1)
                        ],
                    )
                ]
            )
        )

        started = await stream.__anext__()
        delta = await stream.__anext__()

        self.assertIs(started.kind, StreamItemKind.STREAM_STARTED)
        self.assertEqual(delta.text_delta, "tok")
        self.assertEqual(
            delta.metadata,
            {
                "token_id": 7,
                "probability": 0.25,
                "step": 3,
                "probability_distribution": "softmax",
                "tokens": [
                    {"token": "alt", "token_id": 8, "probability": 0.1}
                ],
            },
        )

    async def test_stream_drops_invalid_token_metadata(self):
        stream = VllmStream(
            iter(
                [
                    SimpleNamespace(
                        token="tok",
                        id=True,
                        probability=True,
                        step=True,
                        tokens=[
                            SimpleNamespace(token="", id=8, probability=0.1),
                            SimpleNamespace(
                                token="alt", id=True, probability=True
                            ),
                        ],
                    )
                ]
            )
        )

        _started = await stream.__anext__()
        delta = await stream.__anext__()

        self.assertEqual(delta.text_delta, "tok")
        self.assertEqual(delta.metadata, {"tokens": [{"token": "alt"}]})

    async def test_stream_emits_error_terminal_for_iterator_failure(self):
        stream = VllmStream(iter([RuntimeError("bad vllm chunk")]))

        started = await stream.__anext__()
        errored = await stream.__anext__()

        self.assertIs(started.kind, StreamItemKind.STREAM_STARTED)
        self.assertIs(errored.kind, StreamItemKind.STREAM_ERRORED)
        assert isinstance(errored.data, dict)
        self.assertEqual(errored.data["message"], "bad vllm chunk")

    async def test_stream_flushes_buffered_text_before_iterator_failure(
        self,
    ) -> None:
        stream = VllmStream(iter(["<", RuntimeError("bad vllm chunk")]))

        items = [item async for item in stream]

        self.assertEqual(items[1].kind, StreamItemKind.ANSWER_DELTA)
        self.assertEqual(items[1].text_delta, "<")
        self.assertEqual(items[-2].kind, StreamItemKind.STREAM_ERRORED)
        assert isinstance(items[-2].data, dict)
        self.assertEqual(items[-2].data["message"], "bad vllm chunk")

    def test_non_delta_event_preserves_its_own_metadata(self) -> None:
        event = StreamProviderEvent(
            kind=StreamItemKind.USAGE_COMPLETED,
            usage={"output_tokens": 1},
            metadata={"source": "event"},
            provider_event_type="native.usage",
        )

        result = VllmStream._event_with_metadata(
            event,
            {"source": "chunk", "token_id": 7},
            provider_event_type="vllm.delta",
        )

        self.assertEqual(result.metadata, {"source": "event"})
        self.assertEqual(result.provider_event_type, "native.usage")


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

    def test_vllm_attribute_returns_none_when_dependency_missing(self):
        def fail_import(module_name: str) -> object:
            self.assertEqual(module_name, "vllm")
            raise ImportError(module_name)

        with patch.object(vllm_module, "import_module", fail_import):
            self.assertIsNone(_vllm_attribute("LLM"))

    def test_lazy_vllm_classes_resolve_from_imported_module(self):
        fake_module = SimpleNamespace(LLM="llm-class", SamplingParams="params")

        def import_module(module_name: str) -> object:
            self.assertEqual(module_name, "vllm")
            return fake_module

        with (
            patch.object(vllm_module, "LLM", vllm_module._UNSET),
            patch.object(vllm_module, "SamplingParams", vllm_module._UNSET),
            patch.object(vllm_module, "import_module", import_module),
        ):
            self.assertEqual(_llm_class(), "llm-class")
            self.assertEqual(_sampling_params_class(), "params")

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
            prompt = model._prompt("hello", "sys")
        tok.assert_called_once_with(
            "hello",
            system_prompt="sys",
            developer_prompt=None,
            context=None,
            tensor_format="pt",
            capability=None,
            chat_template_settings=None,
            instructions=None,
        )
        model._tokenizer.decode.assert_called_once_with(
            [1, 2], skip_special_tokens=False
        )
        self.assertEqual(prompt, "decoded")

    def test_prompt_forwards_instructions(self):
        model = self._make_model()
        model._tokenizer = MagicMock(spec=PreTrainedTokenizerFast)
        model._tokenizer.decode.return_value = "decoded"

        with patch.object(
            TextGenerationModel,
            "_tokenize_input",
            return_value={"input_ids": [[1, 2]]},
        ) as tok:
            prompt = model._prompt("hello", "sys", instructions="provider")

        tok.assert_called_once()
        self.assertEqual(tok.call_args.kwargs["instructions"], "provider")
        self.assertEqual(prompt, "decoded")

    async def test_stream_generator(self):
        model = self._make_model()
        model._model = MagicMock()
        model._build_sampling_params = MagicMock(return_value="params")
        iterator = iter(["x", "y"])
        model._model.generate.return_value = iterator

        stream = model._stream_generator("p", GenerationSettings())
        self.assertIsInstance(stream, VllmStream)
        out = [item async for item in stream]

        model._build_sampling_params.assert_called_once()
        model._model.generate.assert_called_once_with(
            ["p"], "params", stream=True
        )
        self.assertEqual(
            [
                item.text_delta
                for item in out
                if item.kind is StreamItemKind.ANSWER_DELTA
            ],
            ["x", "y"],
        )

    async def test_stream_generator_deltas_cumulative_output_text(self):
        model = self._make_model()
        model._model = MagicMock()
        model._build_sampling_params = MagicMock(return_value="params")
        model._model.generate.return_value = iter(
            [
                SimpleNamespace(outputs=[SimpleNamespace(text="hel")]),
                SimpleNamespace(outputs=[SimpleNamespace(text="hello")]),
            ]
        )

        stream = model._stream_generator("p", GenerationSettings())
        out = [item async for item in stream]

        self.assertEqual(
            [
                item.text_delta
                for item in out
                if item.kind is StreamItemKind.ANSWER_DELTA
            ],
            ["hel", "lo"],
        )

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
        stream = VllmStream(iter(["stream"]))
        model._stream_generator = MagicMock(return_value=stream)
        model._string_output = MagicMock()
        model._prompt = MagicMock(return_value="p")
        settings = GenerationSettings(use_async_generator=True)
        result = await model("input", settings=settings)
        model._stream_generator.assert_called_once()
        self.assertIs(result, stream)
        model._string_output.assert_not_called()

    async def test_call_returns_canonical_stream_items(self):
        model = self._make_model()
        model._stream_generator = MagicMock(
            return_value=VllmStream(iter(["stream"]))
        )
        model._string_output = MagicMock()
        model._prompt = MagicMock(return_value="p")
        settings = GenerationSettings(use_async_generator=True)
        result = await model("input", settings=settings)

        self.assertIsInstance(result, VllmStream)
        assert isinstance(result, VllmStream)
        items = [await result.__anext__() for _ in range(5)]
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
        self.assertEqual(items[1].text_delta, "stream")
        model._string_output.assert_not_called()

    async def test_call_use_async_generator_false(self):
        model = self._make_model()
        model._stream_generator = MagicMock(return_value=VllmStream(iter([])))
        model._string_output = MagicMock(return_value="string")
        model._prompt = MagicMock(return_value="p")
        settings = GenerationSettings(use_async_generator=False)
        result = await model("input", settings=settings)
        model._stream_generator.assert_not_called()
        model._string_output.assert_called_once()
        self.assertIsInstance(result, TextGenerationResponse)
        self.assertEqual(await result.to_str(), "string")


if __name__ == "__main__":
    main()
