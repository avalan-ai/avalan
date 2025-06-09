from avalan.entities import TransformerEngineSettings
from avalan.model.engine import Engine
from avalan.model.nlp.sequence import SequenceClassificationModel
from logging import Logger
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)
from unittest import TestCase, IsolatedAsyncioTestCase, main
from unittest.mock import MagicMock, patch, PropertyMock
from contextlib import nullcontext


class SequenceClassificationModelInstantiationTestCase(TestCase):
    model_id = "dummy/model"

    def test_instantiation_no_load(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(
                AutoTokenizer, "from_pretrained"
            ) as auto_tokenizer_mock,
            patch.object(
                AutoModelForSequenceClassification, "from_pretrained"
            ) as auto_model_mock,
        ):
            settings = TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            )
            model = SequenceClassificationModel(
                self.model_id,
                settings,
                logger=logger_mock,
            )
            self.assertIsInstance(model, SequenceClassificationModel)
            auto_model_mock.assert_not_called()
            auto_tokenizer_mock.assert_not_called()

    def test_instantiation_with_load_model_and_tokenizer(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(
                AutoTokenizer, "from_pretrained"
            ) as auto_tokenizer_mock,
            patch.object(
                AutoModelForSequenceClassification, "from_pretrained"
            ) as auto_model_mock,
        ):
            model_instance = MagicMock(spec=PreTrainedModel)
            config_mock = MagicMock()
            config_mock.id2label = {0: "LABEL_0"}
            type(model_instance).config = PropertyMock(return_value=config_mock)
            type(model_instance).name_or_path = PropertyMock(
                return_value=self.model_id
            )
            auto_model_mock.return_value = model_instance

            tokenizer_instance = MagicMock(spec=PreTrainedTokenizerFast)
            tokenizer_instance.__len__.return_value = 1
            tokenizer_instance.model_max_length = 10
            tokenizer_instance.all_special_tokens = []
            type(tokenizer_instance).name_or_path = PropertyMock(
                return_value=self.model_id
            )
            auto_tokenizer_mock.return_value = tokenizer_instance

            model = SequenceClassificationModel(
                self.model_id,
                TransformerEngineSettings(),
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
                self.model_id,
                use_fast=True,
            )


class SequenceClassificationModelCallTestCase(IsolatedAsyncioTestCase):
    model_id = "dummy/model"

    async def test_call(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(
                AutoTokenizer, "from_pretrained"
            ) as auto_tokenizer_mock,
            patch.object(
                AutoModelForSequenceClassification, "from_pretrained"
            ) as auto_model_mock,
            patch(
                "avalan.model.nlp.sequence.inference_mode",
                return_value=nullcontext(),
            ) as inference_mode_mock,
            patch("avalan.model.nlp.sequence.softmax") as softmax_mock,
            patch("avalan.model.nlp.sequence.argmax") as argmax_mock,
        ):
            tokenizer_instance = MagicMock(spec=PreTrainedTokenizerFast)
            tokenizer_instance.__len__.return_value = 1
            tokenizer_instance.model_max_length = 10
            tokenizer_instance.all_special_tokens = []
            inputs_obj = MagicMock()
            inputs_obj.to.return_value = {"input_ids": "ids"}
            tokenizer_instance.return_value = inputs_obj
            type(tokenizer_instance).name_or_path = PropertyMock(
                return_value=self.model_id
            )
            auto_tokenizer_mock.return_value = tokenizer_instance

            model_instance = MagicMock(spec=PreTrainedModel)
            model_instance.device = "device"
            outputs = MagicMock(logits="logits")
            model_instance.return_value = outputs
            config_mock = MagicMock()
            config_mock.id2label = {0: "OK"}
            type(model_instance).config = PropertyMock(return_value=config_mock)
            auto_model_mock.return_value = model_instance

            softmax_mock.return_value = "probs"
            argmax_value = MagicMock()
            argmax_value.item.return_value = 0
            argmax_mock.return_value = argmax_value

            model = SequenceClassificationModel(
                self.model_id,
                TransformerEngineSettings(),
                logger=logger_mock,
            )

            result = await model("hi")

            self.assertEqual(result, "OK")
            tokenizer_instance.assert_called_once_with(
                "hi", return_tensors="pt"
            )
            inputs_obj.to.assert_called_once_with(model_instance.device)
            model_instance.assert_called_once()
            self.assertEqual(
                model_instance.call_args.kwargs, {"input_ids": "ids"}
            )
            softmax_mock.assert_called_once_with(outputs.logits, dim=-1)
            argmax_mock.assert_called_once_with("probs", dim=-1)
            inference_mode_mock.assert_called_once_with()


if __name__ == "__main__":
    main()
