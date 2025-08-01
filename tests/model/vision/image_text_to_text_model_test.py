from avalan.entities import (
    GenerationSettings,
    TransformerEngineSettings,
    MessageRole,
)
from avalan.model.vision.text import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Gemma3ForConditionalGeneration,
    ImageTextToTextModel,
)
from avalan.model.vision import BaseVisionModel
from avalan.model.engine import Engine
from logging import Logger
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)
from unittest import TestCase, IsolatedAsyncioTestCase, main
from unittest.mock import MagicMock, patch, PropertyMock
from PIL import Image


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


class ImageTextToTextModelInstantiationTestCase(TestCase):
    model_id = "dummy/model"

    def test_instantiation_with_load_model(self):
        logger_mock = MagicMock(spec=Logger)
        settings = TransformerEngineSettings()
        with (
            patch.object(AutoProcessor, "from_pretrained") as processor_mock,
            patch.object(
                AutoModelForImageTextToText, "from_pretrained"
            ) as model_mock,
            patch.object(AutoTokenizer, "from_pretrained") as tokenizer_mock,
            patch.object(Engine, "weight", return_value="dtype") as wt_mock,
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

            model = ImageTextToTextModel(
                self.model_id,
                settings,
                logger=logger_mock,
            )

            self.assertIs(model.model, model_instance)
            self.assertIs(model._processor, processor_instance)
            processor_mock.assert_called_once_with(
                self.model_id, use_fast=True
            )
            wt_mock.assert_called_once_with(settings.weight_type)
            model_mock.assert_called_once_with(
                self.model_id,
                torch_dtype="dtype",
                device_map=model._device,
                tp_plan=None,
            )

    def test_instantiation_default_loader(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(AutoProcessor, "from_pretrained") as processor_mock,
            patch.object(
                AutoModelForImageTextToText, "from_pretrained"
            ) as model_mock,
            patch.object(AutoTokenizer, "from_pretrained") as tokenizer_mock,
            patch.object(Engine, "weight", return_value="dtype") as wt_mock,
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

            settings = TransformerEngineSettings()
            model = ImageTextToTextModel(
                self.model_id,
                settings,
                logger=logger_mock,
            )

            self.assertIs(model.model, model_instance)
            self.assertIs(model._processor, processor_instance)
            processor_mock.assert_called_once_with(
                self.model_id, use_fast=True
            )
            wt_mock.assert_called_once_with(settings.weight_type)
            model_mock.assert_called_once_with(
                self.model_id,
                torch_dtype="dtype",
                device_map=model._device,
                tp_plan=None,
            )

    def test_instantiation_gemma3_loader(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(AutoProcessor, "from_pretrained") as processor_mock,
            patch.object(
                Gemma3ForConditionalGeneration, "from_pretrained"
            ) as gemma_mock,
            patch.object(AutoTokenizer, "from_pretrained") as tokenizer_mock,
            patch.object(Engine, "weight", return_value="dtype") as wt_mock,
        ):
            processor_instance = MagicMock()
            processor_mock.return_value = processor_instance

            model_instance = MagicMock(spec=PTMWithGenerate)
            gemma_mock.return_value = model_instance

            tokenizer_instance = MagicMock(spec=PreTrainedTokenizerFast)
            type(tokenizer_instance).all_special_tokens = PropertyMock(
                return_value=[]
            )
            type(tokenizer_instance).name_or_path = PropertyMock(
                return_value=self.model_id
            )
            tokenizer_mock.return_value = tokenizer_instance

            settings = TransformerEngineSettings(loader_class="gemma3")
            model = ImageTextToTextModel(
                self.model_id,
                settings,
                logger=logger_mock,
            )

            self.assertIs(model.model, model_instance)
            processor_mock.assert_called_once_with(
                self.model_id, use_fast=True
            )
            wt_mock.assert_called_once_with(settings.weight_type)
            gemma_mock.assert_called_once_with(
                self.model_id,
                torch_dtype="dtype",
                device_map=model._device,
                tp_plan=None,
            )

    def test_instantiation_invalid_loader(self):
        with (
            patch.object(AutoProcessor, "from_pretrained") as processor_mock,
            patch.object(AutoTokenizer, "from_pretrained") as tokenizer_mock,
        ):
            processor_mock.return_value = MagicMock()
            tokenizer_mock.return_value = MagicMock()
            with self.assertRaises(AssertionError):
                ImageTextToTextModel(
                    self.model_id,
                    TransformerEngineSettings(loader_class="bad"),
                )


class ImageTextToTextModelCallTestCase(IsolatedAsyncioTestCase):
    model_id = "dummy/model"

    async def _run_call(
        self, batch_decode_return, system_prompt=None, width=None
    ):
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
            processor_instance.batch_decode.return_value = batch_decode_return
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
            get_image_mock.return_value = image

            resized_image = MagicMock()
            rgb_image.resize.return_value = resized_image
            rgb_image.width = 10
            rgb_image.height = 5

            model = ImageTextToTextModel(
                self.model_id,
                TransformerEngineSettings(),
                logger=logger_mock,
            )

            gen_settings = GenerationSettings(max_new_tokens=5)
            result = await model(
                "img.jpg",
                "prompt",
                system_prompt=system_prompt,
                settings=gen_settings,
                width=width,
            )

            self.assertEqual(
                result,
                (
                    batch_decode_return[0]
                    if isinstance(batch_decode_return, list)
                    else batch_decode_return
                ),
            )
            get_image_mock.assert_called_once_with("img.jpg")
            image.convert.assert_called_once_with("RGB")
            if width:
                expected_height = int(
                    width / rgb_image.width * rgb_image.height
                )
                rgb_image.resize.assert_called_once_with(
                    (width, expected_height),
                    Image.Resampling.LANCZOS,
                )
                expected_image = resized_image
            else:
                rgb_image.resize.assert_not_called()
                expected_image = rgb_image

            expected_messages = []
            if system_prompt:
                expected_messages.append(
                    {
                        "role": str(MessageRole.SYSTEM),
                        "content": [{"type": "text", "text": system_prompt}],
                    }
                )
            expected_messages.append(
                {
                    "role": str(MessageRole.USER),
                    "content": [
                        {"type": "image", "image": expected_image},
                        {"type": "text", "text": "prompt"},
                    ],
                }
            )
            processor_instance.apply_chat_template.assert_called_once_with(
                expected_messages,
                tokenize=False,
                add_generation_prompt=gen_settings.chat_settings.add_generation_prompt,
            )
            processor_instance.assert_called_once_with(
                text=["chat"],
                images=expected_image,
                videos=None,
                padding=True,
                return_tensors="pt",
            )
            self.assertEqual(inputs.to_called_with, model._device)
            model_instance.generate.assert_called_once_with(
                **inputs,
                max_new_tokens=5,
            )
            processor_instance.batch_decode.assert_called_once_with(
                [[3, 4]],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

    async def test_call_list_output(self):
        await self._run_call(["ok"])

    async def test_call_string_output(self):
        await self._run_call("ok")

    async def test_call_without_system_prompt(self):
        await self._run_call("ok", system_prompt=None)

    async def test_call_with_system_prompt(self):
        await self._run_call("ok", system_prompt="sys")

    async def test_call_with_width(self):
        await self._run_call("ok", width=20)


if __name__ == "__main__":
    main()
