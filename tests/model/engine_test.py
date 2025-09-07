from avalan.entities import (
    EngineSettings,
    ModelConfig,
    SentenceTransformerModelConfig,
    TransformerEngineSettings,
)
from avalan.model.engine import (
    Engine,
    ModelAlreadyLoadedException,
    TokenizerAlreadyLoadedException,
)
from unittest import TestCase
from contextlib import AsyncExitStack
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
import logging
import sys
import types
import importlib.machinery
from transformers import PreTrainedModel, PreTrainedTokenizerFast


class DummyEngine(Engine):
    def __init__(
        self, model_id: str = "id", settings: EngineSettings | None = None
    ):
        settings = settings or EngineSettings(
            auto_load_model=False, auto_load_tokenizer=False
        )
        self.fake_model = MagicMock(spec=PreTrainedModel)
        self.fake_model.parameters.return_value = []
        self.fake_model.eval = MagicMock()
        self.fake_model.resize_token_embeddings = MagicMock()
        self.fake_model.state_dict.return_value = {}
        self.fake_model.config = ModelConfig(
            architectures=None,
            attribute_map={},
            bos_token_id=None,
            bos_token=None,
            decoder_start_token_id=None,
            eos_token_id=None,
            eos_token=None,
            finetuning_task=None,
            hidden_size=None,
            hidden_sizes=None,
            keys_to_ignore_at_inference=[],
            loss_type=None,
            max_position_embeddings=None,
            model_type="type",
            num_attention_heads=None,
            num_hidden_layers=None,
            num_labels=None,
            output_attentions=False,
            output_hidden_states=False,
            pad_token_id=None,
            pad_token=None,
            prefix=None,
            sep_token_id=None,
            sep_token=None,
            state_size=0,
            task_specific_params=None,
            torch_dtype="float32",
            vocab_size=None,
            tokenizer_class=None,
        )

        class DummyTokenizer(PreTrainedTokenizerFast):
            def __init__(self):
                pass

        self.fake_tokenizer = DummyTokenizer()
        type(self.fake_tokenizer).all_special_tokens = PropertyMock(
            return_value=[]
        )
        type(self.fake_tokenizer).name_or_path = PropertyMock(
            return_value="tok"
        )
        type(self.fake_tokenizer).model_max_length = PropertyMock(
            return_value=10
        )
        self.fake_tokenizer.__len__ = MagicMock(return_value=0)
        self.fake_tokenizer._tokenizer = MagicMock()

        super().__init__(model_id, settings)

    @property
    def uses_tokenizer(self) -> bool:
        return True

    async def __call__(self, input, **kwargs):
        return "out"

    def _load_model(self):
        return self.fake_model

    def _load_tokenizer_with_tokens(
        self, tokenizer_name_or_path: str | None, use_fast: bool = True
    ):
        return self.fake_tokenizer


class EnginePropertyTestCase(TestCase):
    def test_properties(self):
        engine = DummyEngine(
            model_id="id",
            settings=EngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
        )
        engine._config = "cfg"
        engine._parameter_count = 5
        engine._parameter_types = {"float32"}
        engine._tokenizer_config = "tok_cfg"
        self.assertEqual(engine.config, "cfg")
        self.assertEqual(engine.model_id, "id")
        self.assertEqual(engine.parameter_count, 5)
        self.assertEqual(engine.parameter_types, {"float32"})
        self.assertEqual(engine.tokenizer_config, "tok_cfg")


class EngineIsRunnableTestCase(TestCase):
    def test_returns_none_without_params(self):
        engine = DummyEngine(
            settings=EngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            )
        )
        self.assertIsNone(engine.is_runnable())

    def test_runnable_no_device(self):
        engine = DummyEngine(
            settings=EngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            )
        )
        engine._parameter_types = {"float32"}
        engine._parameter_count = 10
        with (
            patch.object(
                Engine, "get_default_device", return_value="cpu"
            ) as gdd,
            patch.object(Engine, "_get_device_memory", return_value=80) as gdm,
        ):
            self.assertTrue(engine.is_runnable())
            gdd.assert_called_once()
            gdm.assert_called_once_with("cpu")

    def test_runnable_with_device_not_enough_memory(self):
        engine = DummyEngine(
            settings=EngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            )
        )
        engine._parameter_types = {"float32"}
        engine._parameter_count = 30
        with patch.object(Engine, "_get_device_memory", return_value=40):
            self.assertFalse(engine.is_runnable("cpu"))

    def test_runnable_device_memory_zero(self):
        engine = DummyEngine(
            settings=EngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            )
        )
        engine._parameter_types = {"float32"}
        engine._parameter_count = 1
        with patch.object(Engine, "_get_device_memory", return_value=0):
            self.assertFalse(engine.is_runnable("cpu"))


class EngineContextTestCase(TestCase):
    def test_enter_exit_change_level(self):
        settings = EngineSettings(
            auto_load_model=False, auto_load_tokenizer=False
        )
        engine = DummyEngine(settings=settings)
        engine._exit_stack = AsyncMock(spec=AsyncExitStack)
        engine._transformers_logging_logger = MagicMock(level=logging.WARNING)
        engine._transformers_logging_level = logging.INFO
        with patch(
            "avalan.model.engine.transformers_logging.set_verbosity_error"
        ) as sve:
            result = engine.__enter__()
            self.assertIs(result, engine)
            sve.assert_called_once()
        with patch.object(
            engine._transformers_logging_logger, "setLevel"
        ) as sl:
            engine.__exit__(None, None, None)
            sl.assert_called_once_with(logging.INFO)
            engine._exit_stack.aclose.assert_awaited_once()

    def test_enter_exit_no_change(self):
        settings = EngineSettings(
            auto_load_model=False, auto_load_tokenizer=False
        )
        engine = DummyEngine(settings=settings)
        engine._exit_stack = AsyncMock(spec=AsyncExitStack)
        engine._transformers_logging_logger = None
        with patch(
            "avalan.model.engine.transformers_logging.set_verbosity_error"
        ) as sve:
            self.assertIs(engine.__enter__(), engine)
            sve.assert_not_called()
        engine.__exit__(None, None, None)
        engine._exit_stack.aclose.assert_awaited_once()


class EngineLoadTestCase(TestCase):
    def _setup_engine(self, **settings_kwargs):
        defaults = dict(auto_load_model=False, auto_load_tokenizer=False)
        defaults.update(settings_kwargs)
        settings = TransformerEngineSettings(**defaults)
        engine = DummyEngine(settings=settings)
        return engine

    def test_model_already_loaded(self):
        engine = self._setup_engine(auto_load_model=True)
        engine._loaded_model = True
        with self.assertRaises(ModelAlreadyLoadedException):
            engine._load(load_tokenizer=False, tokenizer_name_or_path=None)

    def test_tokenizer_already_loaded(self):
        engine = self._setup_engine()
        engine._loaded_tokenizer = True
        with self.assertRaises(TokenizerAlreadyLoadedException):
            engine._load(load_tokenizer=True, tokenizer_name_or_path=None)

    def test_progress_bar_disabled(self):
        engine = self._setup_engine(disable_loading_progress_bar=True)
        with (
            patch("avalan.model.engine.disable_progress_bar") as dpb,
            patch("avalan.model.engine.enable_progress_bar") as epb,
        ):
            engine._load(load_tokenizer=True, tokenizer_name_or_path=None)
            dpb.assert_called_once()
            epb.assert_called_once()

    def test_resize_token_embeddings_when_tokens(self):
        engine = self._setup_engine(
            tokens=["a"],
            disable_loading_progress_bar=False,
            auto_load_model=True,
        )
        engine._loaded_model = False
        engine._model = None
        with patch.object(
            engine.fake_tokenizer.__class__, "__len__", return_value=5
        ):
            engine._load(load_tokenizer=True, tokenizer_name_or_path=None)
        engine.fake_model.resize_token_embeddings.assert_called_once_with(5)

    def test_no_resize_without_tokens(self):
        engine = self._setup_engine(
            disable_loading_progress_bar=False, auto_load_model=True
        )
        engine._loaded_model = False
        engine._model = None
        engine._load(load_tokenizer=True, tokenizer_name_or_path=None)
        engine.fake_model.resize_token_embeddings.assert_not_called()

    def test_sentence_transformer_model_config(self):
        class DummyST:
            backend = "torch"
            similarity_fn_name = "cosine"
            truncate_dim = 1

            def __init__(self):
                self.config = engine.fake_model.config

            def parameters(self):
                return []

            def eval(self):
                pass

            def resize_token_embeddings(self, *_):
                pass

            def state_dict(self):
                return {}

        engine = self._setup_engine(auto_load_model=True)
        engine.fake_model = DummyST()
        engine._loaded_model = False
        engine._model = None
        engine._config = None
        with patch("importlib.util.find_spec", return_value=True):
            module = types.SimpleNamespace(
                SentenceTransformer=DummyST,
                __spec__=importlib.machinery.ModuleSpec(
                    "sentence_transformers", None
                ),
            )
            with patch.dict("sys.modules", {"sentence_transformers": module}):
                engine._load(load_tokenizer=False, tokenizer_name_or_path=None)
        self.assertIsInstance(engine._config, SentenceTransformerModelConfig)


class GetDeviceMemoryTestCase(TestCase):
    def test_cuda(self):
        with (
            patch("avalan.model.engine.cuda.is_available", return_value=True),
            patch("avalan.model.engine.cuda.current_device", return_value=0),
            patch("avalan.model.engine.cuda.get_device_properties") as gdp,
        ):
            gdp.return_value.total_memory = 100
            self.assertEqual(Engine._get_device_memory("cuda"), 100)

    def test_mps(self):
        vm = MagicMock()
        vm.total = 50
        dummy_psutil = types.SimpleNamespace(virtual_memory=lambda: vm)
        with (
            patch("avalan.model.engine.mps.is_available", return_value=True),
            patch.dict(sys.modules, {"psutil": dummy_psutil}, clear=False),
        ):
            self.assertEqual(Engine._get_device_memory("mps"), 50)

    def test_cpu(self):
        vm = MagicMock()
        vm.total = 75
        dummy_psutil = types.SimpleNamespace(virtual_memory=lambda: vm)
        with patch.dict(sys.modules, {"psutil": dummy_psutil}, clear=False):
            self.assertEqual(Engine._get_device_memory("cpu"), 75)
