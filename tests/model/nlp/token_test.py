from avalan.entities import TransformerEngineSettings
from avalan.model.transformer import AutoTokenizer
from avalan.model.engine import Engine
from avalan.model.nlp.token import (
    TokenClassificationModel,
    AutoModelForTokenClassification,
)
from logging import Logger
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from unittest import TestCase, IsolatedAsyncioTestCase, main
from unittest.mock import MagicMock, patch, PropertyMock
from contextlib import nullcontext
from torch import tensor


class TokenClassificationModelInstantiationTestCase(TestCase):
    model_id = "dummy/model"

    def test_instantiation_no_load(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(
                AutoTokenizer, "from_pretrained"
            ) as auto_tokenizer_mock,
            patch.object(
                AutoModelForTokenClassification, "from_pretrained"
            ) as auto_model_mock,
        ):
            model = TokenClassificationModel(
                self.model_id,
                TransformerEngineSettings(
                    auto_load_model=False,
                    auto_load_tokenizer=False,
                ),
                logger=logger_mock,
            )
            self.assertIsInstance(model, TokenClassificationModel)
            auto_model_mock.assert_not_called()
            auto_tokenizer_mock.assert_not_called()

    def test_instantiation_with_load_tokenizer(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(
                AutoTokenizer, "from_pretrained"
            ) as auto_tokenizer_mock,
            patch.object(
                AutoModelForTokenClassification, "from_pretrained"
            ) as auto_model_mock,
        ):
            tokenizer_mock = MagicMock(spec=PreTrainedTokenizerFast)
            tokenizer_mock.name_or_path = self.model_id
            tokenizer_mock.__len__.return_value = 1
            tokenizer_mock.model_max_length = 77
            auto_tokenizer_mock.return_value = tokenizer_mock

            model = TokenClassificationModel(
                self.model_id,
                TransformerEngineSettings(
                    auto_load_model=False,
                    auto_load_tokenizer=True,
                ),
                logger=logger_mock,
            )
            self.assertIsInstance(model, TokenClassificationModel)
            auto_tokenizer_mock.assert_called_once_with(
                self.model_id, use_fast=True
            )
            auto_model_mock.assert_not_called()

    def test_instantiation_with_load_model_and_tokenizer(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(
                AutoTokenizer, "from_pretrained"
            ) as auto_tokenizer_mock,
            patch.object(
                AutoModelForTokenClassification, "from_pretrained"
            ) as auto_model_mock,
        ):
            tokenizer_mock = MagicMock(spec=PreTrainedTokenizerFast)
            tokenizer_mock.name_or_path = self.model_id
            tokenizer_mock.__len__.return_value = 1
            tokenizer_mock.model_max_length = 77
            auto_tokenizer_mock.return_value = tokenizer_mock

            model_instance = MagicMock(spec=PreTrainedModel)
            config_mock = MagicMock()
            type(model_instance).config = PropertyMock(
                return_value=config_mock
            )
            auto_model_mock.return_value = model_instance

            model = TokenClassificationModel(
                self.model_id,
                TransformerEngineSettings(
                    auto_load_model=True,
                    auto_load_tokenizer=True,
                ),
                logger=logger_mock,
            )
            self.assertIs(model._model, model_instance)
            auto_model_mock.assert_called_once_with(
                self.model_id,
                cache_dir=None,
                attn_implementation=None,
                trust_remote_code=False,
                torch_dtype="auto",
                state_dict=None,
                local_files_only=False,
                token=None,
                device_map=Engine.get_default_device(),
            )
            auto_tokenizer_mock.assert_called_once_with(
                self.model_id, use_fast=True
            )


class TokenClassificationModelCallTestCase(IsolatedAsyncioTestCase):
    model_id = "dummy/model"

    async def test_call(self):
        logger_mock = MagicMock(spec=Logger)
        inputs = {"input_ids": tensor([[1, 2, 3]])}
        with (
            patch.object(
                AutoTokenizer, "from_pretrained"
            ) as auto_tokenizer_mock,
            patch.object(
                AutoModelForTokenClassification, "from_pretrained"
            ) as auto_model_mock,
            patch.object(
                TokenClassificationModel,
                "_tokenize_input",
                return_value=inputs,
            ) as tokenize_mock,
            patch("avalan.model.nlp.token.argmax") as argmax_mock,
            patch(
                "avalan.model.nlp.token.inference_mode",
                return_value=nullcontext(),
            ) as inference_mode_mock,
        ):
            tokenizer_mock = MagicMock(spec=PreTrainedTokenizerFast)
            tokenizer_mock.convert_ids_to_tokens.return_value = ["a", "b", "c"]
            tokenizer_mock.name_or_path = self.model_id
            tokenizer_mock.__len__.return_value = 1
            tokenizer_mock.model_max_length = 77
            auto_tokenizer_mock.return_value = tokenizer_mock

            model_instance = MagicMock(spec=PreTrainedModel)
            type(model_instance).config = PropertyMock(
                return_value=MagicMock(id2label={0: "A", 1: "B"})
            )
            model_instance.device = "cpu"
            call_result = MagicMock(logits="logits")
            model_instance.return_value = call_result
            auto_model_mock.return_value = model_instance

            argmax_mock.return_value = tensor([[0, 1, 0]])

            model = TokenClassificationModel(
                self.model_id,
                TransformerEngineSettings(),
                logger=logger_mock,
            )

            result = await model("text")

            self.assertEqual(result, {"a": "A", "b": "B", "c": "A"})
            tokenize_mock.assert_called_once_with(
                "text", system_prompt=None, context=None
            )
            model_instance.assert_called_once()
            tokenizer_mock.convert_ids_to_tokens.assert_called_once()
            auto_tokenizer_mock.assert_called_once_with(
                self.model_id, use_fast=True
            )
            auto_model_mock.assert_called_once()
            inference_mode_mock.assert_called_once_with()


if __name__ == "__main__":
    main()
