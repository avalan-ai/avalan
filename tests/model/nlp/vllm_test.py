from logging import Logger
from unittest import TestCase, main
from unittest.mock import MagicMock, PropertyMock, patch

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

from avalan.entities import TransformerEngineSettings
from avalan.model.nlp.text.vllm import VllmModel
from avalan.model.transformer import AutoTokenizer


class VllmModelTestCase(TestCase):
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    def test_instantiation_with_load_model_and_tokenizer(self):
        vllm_mock = MagicMock()
        llm_instance = MagicMock(spec=PreTrainedModel)
        vllm_mock.LLM.return_value = llm_instance
        vllm_mock.SamplingParams.return_value = MagicMock()

        with (
            self.subTest(),
            patch.object(
                AutoTokenizer, "from_pretrained"
            ) as auto_tokenizer_mock,
            patch("avalan.model.nlp.text.vllm.LLM", vllm_mock.LLM),
            patch(
                "avalan.model.nlp.text.vllm.SamplingParams",
                vllm_mock.SamplingParams,
            ),
        ):
            tokenizer_mock = MagicMock(spec=PreTrainedTokenizerFast)
            type(tokenizer_mock).name_or_path = PropertyMock(
                return_value=self.model_id
            )
            auto_tokenizer_mock.reset_mock()
            auto_tokenizer_mock.return_value = tokenizer_mock

            logger_mock = MagicMock(spec=Logger)
            model = VllmModel(
                self.model_id,
                TransformerEngineSettings(),
                logger=logger_mock,
            )
            self.assertIs(model._model, llm_instance)
            auto_tokenizer_mock.assert_called_once_with(
                self.model_id, use_fast=True, subfolder=""
            )
            vllm_mock.LLM.assert_called_once()


if __name__ == "__main__":
    main()
