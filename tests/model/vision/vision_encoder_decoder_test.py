from avalan.entities import TransformerEngineSettings
from avalan.model.engine import Engine
from avalan.model.vision.decoder import (
    AutoImageProcessor,
    VisionEncoderDecoderModel,
    VisionEncoderDecoderModelImpl,
)
from avalan.model.vision import BaseVisionModel
from logging import Logger
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)
from unittest import TestCase, IsolatedAsyncioTestCase, main
from unittest.mock import MagicMock, patch, PropertyMock


class DummyInputs(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.to_called_with = None

    def to(self, device):
        self.to_called_with = device
        return self


class PTMWithGenerate(PreTrainedModel):
    def generate(self, *args, **kwargs):
        raise NotImplementedError


class VisionEncoderDecoderModelInstantiationTestCase(TestCase):
    model_id = "dummy/model"

    def test_instantiation_with_load_model_and_tokenizer(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(
                AutoImageProcessor, "from_pretrained"
            ) as processor_mock,
            patch.object(
                VisionEncoderDecoderModelImpl, "from_pretrained"
            ) as model_mock,
            patch.object(AutoTokenizer, "from_pretrained") as tokenizer_mock,
        ):
            processor_instance = MagicMock()
            processor_mock.return_value = processor_instance

            model_instance = MagicMock(spec=PTMWithGenerate)
            model_mock.return_value = model_instance

            tokenizer_instance = MagicMock(spec=PreTrainedTokenizerFast)
            type(tokenizer_instance).all_special_tokens = PropertyMock(
                return_value=[]
            )
            type(tokenizer_instance).name_or_path = PropertyMock(
                return_value=self.model_id
            )
            tokenizer_mock.return_value = tokenizer_instance

            model = VisionEncoderDecoderModel(
                self.model_id,
                TransformerEngineSettings(),
                logger=logger_mock,
            )

            self.assertIs(model.model, model_instance)
            processor_mock.assert_called_once_with(
                self.model_id, use_fast=True
            )
            model_mock.assert_called_once_with(
                self.model_id,
                device_map=Engine.get_default_device(),
                tp_plan=None,
            )
            model_instance.eval.assert_called_once()
            tokenizer_mock.assert_called_once_with(
                self.model_id, use_fast=True, subfolder=""
            )


class VisionEncoderDecoderModelCallTestCase(IsolatedAsyncioTestCase):
    model_id = "dummy/model"

    async def test_call(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(
                AutoImageProcessor, "from_pretrained"
            ) as processor_mock,
            patch.object(
                VisionEncoderDecoderModelImpl, "from_pretrained"
            ) as model_mock,
            patch.object(AutoTokenizer, "from_pretrained") as tokenizer_mock,
            patch("avalan.model.vision.decoder.Image.open") as image_open_mock,
        ):
            processor_instance = MagicMock()
            processor_instance.return_value = DummyInputs(pixel_values="t")
            processor_mock.return_value = processor_instance

            model_instance = MagicMock(spec=PTMWithGenerate)
            output_ids = [[1, 2, 3]]
            model_instance.generate.return_value = output_ids

            model_mock.return_value = model_instance

            tokenizer_instance = MagicMock(spec=PreTrainedTokenizerFast)
            tokenizer_instance.decode.return_value = "caption"
            type(tokenizer_instance).all_special_tokens = PropertyMock(
                return_value=[]
            )
            type(tokenizer_instance).name_or_path = PropertyMock(
                return_value=self.model_id
            )
            tokenizer_mock.return_value = tokenizer_instance

            image_instance = MagicMock()
            image_open_mock.return_value = image_instance
            image_instance.convert.return_value = image_instance

            model = VisionEncoderDecoderModel(
                self.model_id,
                TransformerEngineSettings(),
                logger=logger_mock,
            )

            caption = await model("img.jpg", None)

            self.assertEqual(caption, "caption")
            image_open_mock.assert_called_once_with("img.jpg")
            processor_instance.assert_called_with(
                images=image_instance, return_tensors="pt"
            )
            image_instance.convert.assert_called_once_with("RGB")
            model_instance.generate.assert_called_once_with(
                **processor_instance.return_value
            )
            self.assertEqual(
                processor_instance.return_value.to_called_with,
                model._device,
            )


class VisionEncoderDecoderModelPromptCallTestCase(IsolatedAsyncioTestCase):
    model_id = "dummy/model"

    async def test_call_with_prompt(self) -> None:
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(
                AutoImageProcessor, "from_pretrained"
            ) as processor_mock,
            patch.object(
                VisionEncoderDecoderModelImpl, "from_pretrained"
            ) as model_mock,
            patch.object(AutoTokenizer, "from_pretrained") as tokenizer_mock,
            patch.object(BaseVisionModel, "_get_image") as get_image_mock,
        ):
            processor_instance = MagicMock()
            pixel_values = DummyInputs()
            processor_result = DummyInputs(pixel_values=pixel_values)
            processor_instance.return_value = processor_result
            processor_mock.return_value = processor_instance

            model_instance = MagicMock(spec=PTMWithGenerate)
            decoder = MagicMock()
            decoder.config = MagicMock(max_position_embeddings=10)
            type(model_instance).decoder = PropertyMock(return_value=decoder)
            outputs = MagicMock()
            outputs.sequences = [[1, 2]]
            model_instance.generate.return_value = outputs
            model_mock.return_value = model_instance

            tokenizer_instance = MagicMock(spec=PreTrainedTokenizerFast)
            decoder_ids = DummyInputs()
            encode_result = DummyInputs(input_ids=decoder_ids)
            tokenizer_instance.return_value = encode_result
            tokenizer_instance.batch_decode.return_value = ["caption"]
            tokenizer_instance.pad_token_id = 0
            tokenizer_instance.eos_token_id = 1
            tokenizer_instance.unk_token_id = 2
            type(tokenizer_instance).all_special_tokens = PropertyMock(
                return_value=[]
            )
            type(tokenizer_instance).name_or_path = PropertyMock(
                return_value=self.model_id
            )
            tokenizer_mock.return_value = tokenizer_instance

            image_mock = MagicMock()
            get_image_mock.return_value = image_mock

            model = VisionEncoderDecoderModel(
                self.model_id,
                TransformerEngineSettings(),
                logger=logger_mock,
            )

            caption = await model(
                "img.jpg",
                "prompt",
                early_stopping=False,
                num_beams=2,
                use_cache=False,
                skip_special_tokens=False,
            )

            self.assertEqual(caption, "caption")
            get_image_mock.assert_called_once_with("img.jpg")
            processor_instance.assert_called_once_with(
                image_mock, return_tensors="pt"
            )
            self.assertEqual(pixel_values.to_called_with, model._device)
            tokenizer_instance.assert_called_once_with(
                "prompt", add_special_tokens=False, return_tensors="pt"
            )
            self.assertEqual(decoder_ids.to_called_with, model._device)
            model_instance.generate.assert_called_once_with(
                pixel_values=processor_result.pixel_values,
                decoder_input_ids=encode_result.input_ids,
                max_length=10,
                early_stopping=False,
                pad_token_id=0,
                eos_token_id=1,
                use_cache=False,
                num_beams=2,
                bad_words_ids=[[2]],
                return_dict_in_generate=True,
            )
            tokenizer_instance.batch_decode.assert_called_once_with(
                outputs.sequences, skip_special_tokens=False
            )


if __name__ == "__main__":
    main()
