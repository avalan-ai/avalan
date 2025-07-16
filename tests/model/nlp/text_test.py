from avalan.entities import TransformerEngineSettings, Token
from avalan.model.transformer import AutoTokenizer
from avalan.model.nlp.text.generation import (
    AutoModelForCausalLM,
    TextGenerationModel,
)
from avalan.model.engine import Engine
from logging import Logger
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)
from torch import tensor
from unittest import main, TestCase
from unittest.mock import patch, MagicMock, PropertyMock


class TextGenerationModelTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_ids = [
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "deepseek-ai/deepseek-llm-7b-chat",
            "google/owlvit-base-patch16",
            "hf-internal-testing/tiny-random-bert",
            "qingy2024/UwU-7B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
        ]

    def test_instantiation_no_load(self):
        logger_mock = MagicMock(spec=Logger)
        for model_id in self.model_ids:
            with self.subTest():
                model = TextGenerationModel(
                    model_id,
                    TransformerEngineSettings(
                        auto_load_model=False, auto_load_tokenizer=False
                    ),
                    logger=logger_mock,
                )
                self.assertIsInstance(model, TextGenerationModel)
                logger_mock.assert_not_called()

    def test_instantiation_with_load_tokenizer(self):
        logger_mock = MagicMock(spec=Logger)
        for model_id in self.model_ids:
            with (
                self.subTest(),
                patch.object(
                    AutoTokenizer, "from_pretrained"
                ) as auto_tokenizer_mock,
            ):
                tokenizer_mock = MagicMock(spec=PreTrainedTokenizerFast)
                type(tokenizer_mock).name_or_path = PropertyMock(
                    return_value=model_id
                )
                auto_tokenizer_mock.reset_mock()
                auto_tokenizer_mock.return_value = tokenizer_mock
                model = TextGenerationModel(
                    model_id,
                    TransformerEngineSettings(
                        auto_load_model=False, auto_load_tokenizer=True
                    ),
                    logger=logger_mock,
                )
                self.assertIsInstance(model, TextGenerationModel)
                auto_tokenizer_mock.assert_called_once_with(
                    model_id, use_fast=True, subfolder=None
                )

    def test_instantiation_with_load_model_and_tokenizer(self):
        logger_mock = MagicMock(spec=Logger)
        for model_id in self.model_ids:
            with (
                self.subTest(),
                patch.object(
                    AutoTokenizer, "from_pretrained"
                ) as auto_tokenizer_mock,
                patch.object(
                    AutoModelForCausalLM, "from_pretrained"
                ) as auto_model_mock,
            ):
                auto_model_mock.reset_mock()
                model_mock = MagicMock(spec=PreTrainedModel)
                config_mock = MagicMock(autospec=PretrainedConfig)
                model_mock.config = PropertyMock(return_value=config_mock)
                type(model_mock).name_or_path = PropertyMock(
                    return_value=model_id
                )
                auto_model_mock.return_value = model_mock

                auto_tokenizer_mock.reset_mock()
                tokenizer_mock = MagicMock(spec=PreTrainedTokenizerFast)
                type(tokenizer_mock).all_special_tokens = PropertyMock(
                    return_value=[]
                )
                type(tokenizer_mock).name_or_path = PropertyMock(
                    return_value=model_id
                )
                auto_tokenizer_mock.return_value = tokenizer_mock

                model = TextGenerationModel(
                    model_id,
                    TransformerEngineSettings(
                        auto_load_model=True, auto_load_tokenizer=True
                    ),
                    logger=logger_mock,
                )
                self.assertIsInstance(model, TextGenerationModel)

                auto_model_mock.assert_called_once_with(
                    model_id,
                    cache_dir=None,
                    subfolder="",
                    attn_implementation=None,
                    trust_remote_code=False,
                    torch_dtype="auto",
                    state_dict=None,
                    local_files_only=False,
                    low_cpu_mem_usage=True,
                    device_map=Engine.get_default_device(),
                    token=None,
                    quantization_config=None,
                    revision=None,
                    tp_plan=None,
                )
                auto_tokenizer_mock.assert_called_once_with(
                    model_id, use_fast=True, subfolder=None
                )


class TextGenerationModelMethodsTestCase(TestCase):
    def setUp(self):
        self.settings = TransformerEngineSettings(
            auto_load_model=False,
            auto_load_tokenizer=False,
        )
        self.model = TextGenerationModel("m", self.settings)

    def test_tokenize(self):
        self.model._tokenizer = MagicMock()
        self.model._tokenizer.encode.return_value = [1, 2]
        self.model._tokenizer.decode.side_effect = (
            lambda i, skip_special_tokens=False: f"t{i}"
        )
        self.model._loaded_tokenizer = False
        self.model.load = MagicMock()

        result = self.model.tokenize("hi")

        self.model.load.assert_called_once_with(
            load_model=False,
            load_tokenizer=True,
            tokenizer_name_or_path=None,
        )
        self.model._tokenizer.encode.assert_called_once_with(
            "hi", add_special_tokens=True
        )
        self.assertEqual(
            result,
            [
                Token(id=1, token="t1", probability=1),
                Token(id=2, token="t2", probability=1),
            ],
        )

    def test_input_token_count(self):
        self.model._tokenizer = MagicMock()
        self.model._loaded_tokenizer = True
        with patch.object(
            TextGenerationModel,
            "_tokenize_input",
            return_value={"input_ids": tensor([[1, 2, 3]])},
        ) as tok:
            count = self.model.input_token_count("hi")

        tok.assert_called_once_with("hi", None, context=None)
        self.assertEqual(count, 3)

    def test_save_tokenizer(self):
        path = "/tmp/tok"
        tokenizer = MagicMock(spec=PreTrainedTokenizerFast)
        tokenizer.save_pretrained.return_value = ["f1", "f2"]
        type(tokenizer).name_or_path = PropertyMock(return_value="m")
        tokenizer.__len__.return_value = 1

        self.model._tokenizer = tokenizer

        files = self.model.save_tokenizer(path)

        tokenizer.save_pretrained.assert_called_once_with(path)
        self.assertEqual(files, ["f1", "f2"])

    def test_special_tokens_added(self):
        special_tokens = ["<s>", "</s>"]
        with patch.object(AutoTokenizer, "from_pretrained") as auto_tok:
            tokenizer = MagicMock(spec=PreTrainedTokenizerFast)
            tokenizer.all_special_tokens = []
            tokenizer.model_max_length = 10
            type(tokenizer).name_or_path = PropertyMock(return_value="m")
            tokenizer.__len__.return_value = 1
            auto_tok.return_value = tokenizer

            TextGenerationModel(
                "m",
                TransformerEngineSettings(
                    auto_load_model=False,
                    auto_load_tokenizer=True,
                    special_tokens=special_tokens,
                ),
            )

            auto_tok.assert_called_once_with(
                "m", use_fast=True, subfolder=None
            )
            tokenizer.add_special_tokens.assert_called_once()
            args = tokenizer.add_special_tokens.call_args.args[0]
            self.assertIn("additional_special_tokens", args)
            self.assertEqual(
                [t.content for t in args["additional_special_tokens"]],
                special_tokens,
            )

    def test_tokens_added(self):
        tokens = ["a", "b"]
        with (
            patch.object(AutoTokenizer, "from_pretrained") as auto_tok,
            patch.object(TextGenerationModel, "_tokens", tokens, create=True),
        ):
            tokenizer = MagicMock(spec=PreTrainedTokenizerFast)
            tokenizer.all_special_tokens = []
            tokenizer.model_max_length = 10
            type(tokenizer).name_or_path = PropertyMock(return_value="m")
            tokenizer.__len__.return_value = 1
            auto_tok.return_value = tokenizer

            TextGenerationModel(
                "m",
                TransformerEngineSettings(
                    auto_load_model=False,
                    auto_load_tokenizer=True,
                    tokens=tokens,
                ),
            )

            auto_tok.assert_called_once_with(
                "m", use_fast=True, subfolder=None
            )
            tokenizer.add_tokens.assert_called_once_with(tokens)


if __name__ == "__main__":
    main()
