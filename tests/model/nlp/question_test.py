from avalan.entities import TransformerEngineSettings
from avalan.model.transformer import AutoTokenizer
from avalan.model.engine import Engine
from avalan.model.nlp.question import (
    AutoModelForQuestionAnswering,
    QuestionAnsweringModel,
)
from logging import Logger
from torch import tensor
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from unittest import TestCase, IsolatedAsyncioTestCase, main
from contextlib import nullcontext
from unittest.mock import patch, MagicMock, PropertyMock


class TruthyMagicMock(MagicMock):
    def __bool__(self):
        return True


class QuestionAnsweringModelInstantiationTestCase(TestCase):
    model_id = "dummy/model"

    def test_instantiation_no_load(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(AutoTokenizer, "from_pretrained") as tokenizer_mock,
            patch.object(
                AutoModelForQuestionAnswering, "from_pretrained"
            ) as model_mock,
        ):
            settings = TransformerEngineSettings(
                auto_load_model=False,
                auto_load_tokenizer=False,
            )
            model = QuestionAnsweringModel(
                self.model_id,
                settings,
                logger=logger_mock,
            )
            self.assertIsInstance(model, QuestionAnsweringModel)
            tokenizer_mock.assert_not_called()
            model_mock.assert_not_called()

    def test_instantiation_with_load_model_and_tokenizer(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(AutoTokenizer, "from_pretrained") as tokenizer_mock,
            patch.object(
                AutoModelForQuestionAnswering, "from_pretrained"
            ) as model_mock,
        ):
            tokenizer_instance = TruthyMagicMock(spec=PreTrainedTokenizerFast)
            type(tokenizer_instance).name_or_path = PropertyMock(
                return_value=self.model_id
            )
            type(tokenizer_instance).all_special_tokens = PropertyMock(
                return_value=[]
            )
            type(tokenizer_instance).model_max_length = PropertyMock(
                return_value=1000
            )
            tokenizer_mock.return_value = tokenizer_instance

            model_instance = TruthyMagicMock()
            model_instance.__class__ = PreTrainedModel
            type(model_instance).name_or_path = PropertyMock(
                return_value=self.model_id
            )
            model_mock.return_value = model_instance

            settings = TransformerEngineSettings()
            model = QuestionAnsweringModel(
                self.model_id,
                settings,
                logger=logger_mock,
            )
            self.assertIsInstance(model, QuestionAnsweringModel)
            model_mock.assert_called_once_with(
                self.model_id,
                cache_dir=None,
                subfolder="",
                attn_implementation=None,
                output_hidden_states=False,
                trust_remote_code=False,
                torch_dtype="auto",
                state_dict=None,
                local_files_only=False,
                token=None,
                device_map=Engine.get_default_device(),
                tp_plan=None,
            )
            tokenizer_mock.assert_called_once_with(
                self.model_id,
                use_fast=True,
                subfolder="",
            )


class QuestionAnsweringModelCallTestCase(IsolatedAsyncioTestCase):
    model_id = "dummy/model"

    async def test_call(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(AutoTokenizer, "from_pretrained") as tokenizer_mock,
            patch.object(
                AutoModelForQuestionAnswering, "from_pretrained"
            ) as model_mock,
            patch("avalan.model.nlp.question.argmax") as argmax_mock,
            patch.object(
                QuestionAnsweringModel, "_tokenize_input"
            ) as tokenize_mock,
            patch(
                "avalan.model.nlp.question.inference_mode",
                return_value=nullcontext(),
            ) as inference_mode_mock,
        ):
            tokenizer_instance = TruthyMagicMock(spec=PreTrainedTokenizerFast)
            tokenizer_instance.decode.return_value = "ans"
            type(tokenizer_instance).name_or_path = PropertyMock(
                return_value=self.model_id
            )
            type(tokenizer_instance).all_special_tokens = PropertyMock(
                return_value=[]
            )
            type(tokenizer_instance).model_max_length = PropertyMock(
                return_value=1000
            )
            tokenizer_mock.return_value = tokenizer_instance

            outputs_instance = TruthyMagicMock()
            model_instance = TruthyMagicMock()
            model_instance.__class__ = PreTrainedModel
            model_instance.return_value = outputs_instance
            type(model_instance).config = PropertyMock(
                return_value=MagicMock()
            )
            type(model_instance).name_or_path = PropertyMock(
                return_value=self.model_id
            )
            model_mock.return_value = model_instance

            settings = TransformerEngineSettings()
            model = QuestionAnsweringModel(
                self.model_id,
                settings,
                logger=logger_mock,
            )

            inputs = {"input_ids": tensor([[0, 1, 2, 3]])}
            tokenize_mock.return_value = inputs
            outputs_instance.start_logits = tensor([0.1, 0.2, 0.3, 0.4])
            outputs_instance.end_logits = tensor([0.1, 0.2, 0.5, 0.2])
            argmax_mock.side_effect = [2, 3]

            result = await model("q", context="ctx")

            self.assertEqual(result, "ans")
            tokenize_mock.assert_called_once_with(
                "q",
                system_prompt=None,
                context="ctx",
            )
            model_instance.assert_called_once()
            called_tensor = tokenizer_instance.decode.call_args_list[-1][0][0]
            self.assertEqual(called_tensor.tolist(), [2, 3])
            tokenizer_instance.decode.assert_called_with(
                called_tensor, skip_special_tokens=True
            )
            inference_mode_mock.assert_called_once_with()

    async def test_call_with_system_prompt(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(AutoTokenizer, "from_pretrained") as tokenizer_mock,
            patch.object(
                AutoModelForQuestionAnswering, "from_pretrained"
            ) as model_mock,
            patch("avalan.model.nlp.question.argmax") as argmax_mock,
            patch.object(
                QuestionAnsweringModel, "_tokenize_input"
            ) as tokenize_mock,
            patch(
                "avalan.model.nlp.question.inference_mode",
                return_value=nullcontext(),
            ) as inference_mode_mock,
        ):
            tokenizer_instance = TruthyMagicMock(spec=PreTrainedTokenizerFast)
            tokenizer_instance.decode.return_value = "ans"
            type(tokenizer_instance).name_or_path = PropertyMock(
                return_value=self.model_id
            )
            type(tokenizer_instance).all_special_tokens = PropertyMock(
                return_value=[]
            )
            type(tokenizer_instance).model_max_length = PropertyMock(
                return_value=1000
            )
            tokenizer_mock.return_value = tokenizer_instance

            outputs_instance = TruthyMagicMock()
            model_instance = TruthyMagicMock()
            model_instance.__class__ = PreTrainedModel
            model_instance.return_value = outputs_instance
            type(model_instance).config = PropertyMock(
                return_value=MagicMock()
            )
            type(model_instance).name_or_path = PropertyMock(
                return_value=self.model_id
            )
            model_mock.return_value = model_instance

            settings = TransformerEngineSettings()
            model = QuestionAnsweringModel(
                self.model_id,
                settings,
                logger=logger_mock,
            )

            inputs = {"input_ids": tensor([[0, 1, 2, 3]])}
            tokenize_mock.return_value = inputs
            outputs_instance.start_logits = tensor([0.1, 0.2, 0.3, 0.4])
            outputs_instance.end_logits = tensor([0.1, 0.2, 0.5, 0.2])
            argmax_mock.side_effect = [2, 3]

            result = await model("q", context="ctx", system_prompt="sp")

            self.assertEqual(result, "ans")
            tokenize_mock.assert_called_once_with(
                "q",
                system_prompt="sp",
                context="ctx",
            )
            inference_mode_mock.assert_called_once_with()


if __name__ == "__main__":
    main()
