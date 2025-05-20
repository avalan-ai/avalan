from avalan.model.transformer import AutoTokenizer
from avalan.model.entities import TransformerEngineSettings
from avalan.model.nlp.text.generation import (
    AutoModelForCausalLM,
    TextGenerationModel
)
from logging import Logger
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerFast
)
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
            "Qwen/Qwen2.5-7B-Instruct"
        ]

    def test_instantiation_no_load(self):
        logger_mock = MagicMock(spec=Logger)
        for model_id in self.model_ids:
            with self.subTest():
                model = TextGenerationModel(
                    model_id,
                    TransformerEngineSettings(
                        auto_load_model=False,
                        auto_load_tokenizer=False
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
                patch.object(AutoTokenizer, "from_pretrained")
                    as auto_tokenizer_mock
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
                        auto_load_model=False,
                        auto_load_tokenizer=True
                    ),
                    logger=logger_mock,
                )
                self.assertIsInstance(model, TextGenerationModel)
                auto_tokenizer_mock.assert_called_once_with(
                    model_id,
                    use_fast=True
                )

    def test_instantiation_with_load_model_and_tokenizer(self):
        logger_mock = MagicMock(spec=Logger)
        for model_id in self.model_ids:
            with (
                self.subTest(),
                patch.object(AutoTokenizer, "from_pretrained")
                    as auto_tokenizer_mock,
                patch.object(AutoModelForCausalLM, "from_pretrained")
                    as auto_model_mock
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
                        auto_load_model=True,
                        auto_load_tokenizer=True
                    ),
                    logger=logger_mock,
                )
                self.assertIsInstance(model, TextGenerationModel)

                auto_model_mock.assert_called_once_with(
                    model_id,
                    cache_dir=None,
                    attn_implementation=None,
                    trust_remote_code=False,
                    torch_dtype="auto",
                    state_dict=None,
                    local_files_only=False,
                    low_cpu_mem_usage=True,
                    device_map="mps",
                    token=None,
                    quantization_config=None,
                    revision=None
                )
                auto_tokenizer_mock.assert_called_once_with(
                    model_id,
                    use_fast=True
                )

if __name__ == '__main__':
    main()
