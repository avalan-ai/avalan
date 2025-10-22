from contextlib import nullcontext
from logging import Logger
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import MagicMock, PropertyMock, patch

from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

from avalan.entities import (
    GenerationSettings,
    MessageRole,
    TransformerEngineSettings,
)
from avalan.model.engine import Engine
from avalan.model.vision import BaseVisionModel
from avalan.model.vision.text import (
    AutoImageProcessor,
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
    AutoProcessor,
    ImageTextToTextModel,
    ImageToTextModel,
)


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


class ImageToTextModelInstantiationTestCase(TestCase):
    model_id = "dummy/model"

    def test_instantiation_with_load_model(self):
        logger_mock = MagicMock(spec=Logger)
        settings = TransformerEngineSettings()
        with (
            patch.object(
                AutoImageProcessor, "from_pretrained"
            ) as processor_mock,
            patch.object(
                AutoModelForVision2Seq, "from_pretrained"
            ) as model_mock,
            patch.object(AutoTokenizer, "from_pretrained") as tokenizer_mock,
            patch.object(Engine, "_get_tp_plan", return_value="tp") as tp_mock,
            patch.object(
                Engine, "_get_distributed_config", return_value="dc"
            ) as dist_mock,
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

            model = ImageToTextModel(
                self.model_id,
                settings,
                logger=logger_mock,
            )

            self.assertIs(model.model, model_instance)
            self.assertIs(model._processor, processor_instance)
            processor_mock.assert_called_once_with(
                self.model_id, use_fast=True
            )
            tp_mock.assert_called_once_with(settings.parallel)
            dist_mock.assert_called_once_with(settings.distributed_config)
            model_mock.assert_called_once_with(
                self.model_id,
                device_map=model._device,
                tp_plan="tp",
                distributed_config="dc",
            )


class ImageToTextModelTokenizeInputTestCase(TestCase):
    model_id = "dummy/model"

    def test_tokenize_input_not_implemented(self):
        with patch.object(AutoTokenizer, "from_pretrained") as tokenizer_mock:
            tokenizer_instance = MagicMock(spec=PreTrainedTokenizerFast)
            type(tokenizer_instance).all_special_tokens = PropertyMock(
                return_value=[]
            )
            type(tokenizer_instance).name_or_path = PropertyMock(
                return_value=self.model_id
            )
            tokenizer_mock.return_value = tokenizer_instance

            model = ImageToTextModel(
                self.model_id,
                TransformerEngineSettings(auto_load_model=False),
            )
            with self.assertRaises(NotImplementedError):
                model._tokenize_input("input")


class ImageToTextModelCallTestCase(IsolatedAsyncioTestCase):
    model_id = "dummy/model"

    async def test_call(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(
                AutoImageProcessor, "from_pretrained"
            ) as processor_mock,
            patch.object(
                AutoModelForVision2Seq, "from_pretrained"
            ) as model_mock,
            patch.object(AutoTokenizer, "from_pretrained") as tokenizer_mock,
            patch(
                "avalan.model.vision.text.inference_mode",
                return_value=nullcontext(),
            ) as inference_mode_mock,
            patch.object(BaseVisionModel, "_get_image") as get_image_mock,
            patch.object(Engine, "_get_tp_plan", return_value=None),
            patch.object(Engine, "_get_distributed_config", return_value=None),
        ):
            processor_instance = MagicMock()
            processor_instance.return_value = DummyInputs(
                pixel_values="inputs"
            )
            processor_mock.return_value = processor_instance

            model_instance = MagicMock(spec=PTMWithGenerate)
            model_instance.generate.return_value = [[1, 2, 3]]
            model_mock.return_value = model_instance

            tokenizer_instance = MagicMock(spec=PreTrainedTokenizerFast)
            tokenizer_instance.decode.return_value = "decoded"
            type(tokenizer_instance).all_special_tokens = PropertyMock(
                return_value=[]
            )
            type(tokenizer_instance).name_or_path = PropertyMock(
                return_value=self.model_id
            )
            tokenizer_mock.return_value = tokenizer_instance

            image = MagicMock()
            get_image_mock.return_value = image

            model = ImageToTextModel(
                self.model_id,
                TransformerEngineSettings(),
                logger=logger_mock,
            )

            result = await model("img.jpg")

            self.assertEqual(result, "decoded")
            processor_instance.assert_called_once_with(
                images=image, return_tensors="pt"
            )
            self.assertEqual(
                processor_instance.return_value.to_called_with, model._device
            )
            model_instance.generate.assert_called_once_with(
                **processor_instance.return_value
            )
            tokenizer_instance.decode.assert_called_once_with(
                [1, 2, 3], skip_special_tokens=True
            )
            get_image_mock.assert_called_once_with("img.jpg")
            inference_mode_mock.assert_called_once_with()


class ImageTextToTextModelDeveloperPromptTestCase(IsolatedAsyncioTestCase):
    model_id = "dummy/model"

    async def test_call_with_developer_prompt(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(AutoProcessor, "from_pretrained") as processor_mock,
            patch.object(
                AutoModelForImageTextToText, "from_pretrained"
            ) as model_mock,
            patch.object(AutoTokenizer, "from_pretrained") as tokenizer_mock,
            patch.object(BaseVisionModel, "_get_image") as get_image_mock,
        ):
            processor_instance = MagicMock()
            processor_instance.apply_chat_template.return_value = "chat"
            inputs = DummyInputs(input_ids=[[1, 2]])
            processor_instance.return_value = inputs
            processor_instance.batch_decode.return_value = ["out"]
            processor_mock.return_value = processor_instance

            tokenizer_instance = MagicMock(spec=PreTrainedTokenizerFast)
            type(tokenizer_instance).all_special_tokens = PropertyMock(
                return_value=[]
            )
            type(tokenizer_instance).name_or_path = PropertyMock(
                return_value=self.model_id
            )
            tokenizer_mock.return_value = tokenizer_instance

            model_instance = MagicMock(spec=PTMWithGenerate)
            model_instance.generate.return_value = [[1, 2, 3, 4]]
            model_mock.return_value = model_instance

            image = MagicMock()
            rgb_image = MagicMock()
            image.convert.return_value = rgb_image
            rgb_image.width = 10
            rgb_image.height = 5
            get_image_mock.return_value = image

            model = ImageTextToTextModel(
                self.model_id,
                TransformerEngineSettings(),
                logger=logger_mock,
            )

            gen_settings = GenerationSettings(max_new_tokens=5)
            result = await model(
                "img.jpg",
                "prompt",
                developer_prompt="dev",
                settings=gen_settings,
            )

            self.assertEqual(result, "out")
            get_image_mock.assert_called_once_with("img.jpg")
            image.convert.assert_called_once_with("RGB")
            expected_messages = [
                {
                    "role": str(MessageRole.DEVELOPER),
                    "content": [{"type": "text", "text": "dev"}],
                },
                {
                    "role": str(MessageRole.USER),
                    "content": [
                        {"type": "image", "image": rgb_image},
                        {"type": "text", "text": "prompt"},
                    ],
                },
            ]
            processor_instance.apply_chat_template.assert_called_once_with(
                expected_messages,
                tokenize=False,
                add_generation_prompt=gen_settings.chat_settings.add_generation_prompt,
            )
            processor_instance.assert_called_once_with(
                text=["chat"],
                images=rgb_image,
                videos=None,
                padding=True,
                return_tensors="pt",
            )
            self.assertEqual(inputs.to_called_with, model._device)
            model_instance.generate.assert_called_once_with(
                **inputs, max_new_tokens=5
            )
            processor_instance.batch_decode.assert_called_once_with(
                [[3, 4]],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )


if __name__ == "__main__":
    main()
