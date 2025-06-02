from avalan.entities import TransformerEngineSettings
from avalan.model.nlp.sentence import SentenceTransformerModel
from avalan.model.transformer import AutoTokenizer
from transformers import PreTrainedTokenizerFast
from avalan.model.engine import Engine
from logging import Logger
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import MagicMock, patch, PropertyMock
import types
from importlib.machinery import ModuleSpec


class SentenceTransformerModelTestCase(IsolatedAsyncioTestCase):
    model_id = "sentence-transformers/all-MiniLM-L6-v2"

    def test_instantiation_no_load(self):
        logger_mock = MagicMock(spec=Logger)
        model = SentenceTransformerModel(
            self.model_id,
            TransformerEngineSettings(
                auto_load_model=False,
                auto_load_tokenizer=False,
            ),
            logger=logger_mock,
        )
        self.assertIsInstance(model, SentenceTransformerModel)
        logger_mock.assert_not_called()

    def test_instantiation_with_load_model_and_tokenizer(self):
        logger_mock = MagicMock(spec=Logger)

        class DummySentenceTransformer:
            def __init__(self, *args, **kwargs):
                self.init_args = args
                self.init_kwargs = kwargs
                self.called_with = None

            def encode(self, *args, **kwargs):
                self.called_with = (args, kwargs)
                return [0.0]

            def eval(self):
                pass

            def __getitem__(self, idx):
                class Wrapper:
                    auto_model = types.SimpleNamespace(config=None)

                return Wrapper()

            def state_dict(self):
                return {}

        dummy = types.ModuleType("sentence_transformers")
        dummy.SentenceTransformer = DummySentenceTransformer
        dummy.__spec__ = ModuleSpec("sentence_transformers", loader=None)
        with patch.dict("sys.modules", {"sentence_transformers": dummy}):
            with patch.object(
                AutoTokenizer, "from_pretrained"
            ) as auto_tokenizer_mock:
                tokenizer_mock = MagicMock(spec=PreTrainedTokenizerFast)
                type(tokenizer_mock).all_special_tokens = PropertyMock(
                    return_value=[]
                )
                type(tokenizer_mock).name_or_path = PropertyMock(
                    return_value=self.model_id
                )
                auto_tokenizer_mock.return_value = tokenizer_mock

                model = SentenceTransformerModel(
                    self.model_id,
                    TransformerEngineSettings(
                        auto_load_model=True,
                        auto_load_tokenizer=True,
                    ),
                    logger=logger_mock,
                )

                self.assertIsInstance(model._model, DummySentenceTransformer)
                self.assertEqual(model._model.init_args[0], self.model_id)
                self.assertEqual(
                    model._model.init_kwargs,
                    {
                        "cache_folder": None,
                        "device": Engine.get_default_device(),
                        "trust_remote_code": False,
                        "local_files_only": False,
                        "token": None,
                        "model_kwargs": {
                            "attn_implementation": None,
                            "torch_dtype": "auto",
                            "low_cpu_mem_usage": True,
                            "device_map": Engine.get_default_device(),
                        },
                        "backend": "torch",
                        "similarity_fn_name": None,
                        "truncate_dim": None,
                    },
                )
                auto_tokenizer_mock.assert_called_once_with(
                    self.model_id,
                    use_fast=True,
                )

    def test_token_count(self):
        logger_mock = MagicMock(spec=Logger)
        dummy = types.ModuleType("sentence_transformers")
        dummy.SentenceTransformer = MagicMock()
        dummy.__spec__ = ModuleSpec("sentence_transformers", loader=None)
        with patch.dict("sys.modules", {"sentence_transformers": dummy}):
            with patch.object(
                AutoTokenizer, "from_pretrained"
            ) as auto_tokenizer_mock:
                tokenizer_mock = MagicMock(spec=PreTrainedTokenizerFast)
                tokenizer_mock.encode.return_value = [1, 2, 3]
                type(tokenizer_mock).all_special_tokens = PropertyMock(
                    return_value=[]
                )
                type(tokenizer_mock).name_or_path = PropertyMock(
                    return_value=self.model_id
                )
                auto_tokenizer_mock.return_value = tokenizer_mock

                model = SentenceTransformerModel(
                    self.model_id,
                    TransformerEngineSettings(
                        auto_load_model=False,
                        auto_load_tokenizer=True,
                    ),
                    logger=logger_mock,
                )
                count = model.token_count("hi")
                tokenizer_mock.encode.assert_called_once_with(
                    "hi", add_special_tokens=False
                )
                self.assertEqual(count, 3)

    async def test_call(self):
        logger_mock = MagicMock(spec=Logger)

        class DummySentenceTransformer:
            def __init__(self, *args, **kwargs):
                self.init_args = args
                self.init_kwargs = kwargs
                self.called_with = None

            def encode(self, *args, **kwargs):
                self.called_with = (args, kwargs)
                return [0.1, 0.2]

            def eval(self):
                pass

            def __getitem__(self, idx):
                class Wrapper:
                    auto_model = types.SimpleNamespace(config=None)

                return Wrapper()

            def state_dict(self):
                return {}

        dummy = types.ModuleType("sentence_transformers")
        dummy.SentenceTransformer = DummySentenceTransformer
        dummy.__spec__ = ModuleSpec("sentence_transformers", loader=None)
        with patch.dict("sys.modules", {"sentence_transformers": dummy}):
            with patch.object(
                AutoTokenizer, "from_pretrained"
            ) as auto_tokenizer_mock:
                tokenizer_mock = MagicMock(spec=PreTrainedTokenizerFast)
                type(tokenizer_mock).all_special_tokens = PropertyMock(
                    return_value=[]
                )
                type(tokenizer_mock).name_or_path = PropertyMock(
                    return_value=self.model_id
                )
                auto_tokenizer_mock.return_value = tokenizer_mock

                model = SentenceTransformerModel(
                    self.model_id,
                    TransformerEngineSettings(
                        auto_load_model=True,
                        auto_load_tokenizer=True,
                    ),
                    logger=logger_mock,
                )
                result = await model("hello")
                self.assertIsInstance(model._model, DummySentenceTransformer)
                self.assertEqual(
                    model._model.called_with,
                    (("hello",), {"convert_to_numpy": True}),
                )
                self.assertEqual(result, [0.1, 0.2])


if __name__ == "__main__":
    main()
