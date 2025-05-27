from ...compat import override
from ...model import TextGenerationVendor
from ...model.entities import (
    GenerationSettings,
    ImageEntity,
    Input,
    MessageRole
)
from ...model.nlp import BaseNLPModel
from ...model.vision import BaseVisionModel
from ...model.transformer import TransformerModel
from PIL import Image
from torch import no_grad, Tensor
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoModelForVision2Seq,
    AutoProcessor,
    PreTrainedModel,
    Qwen2VLForConditionalGeneration,
    VisionEncoderDecoderModel as HFVisionEncoderDecoderModel,
)
from transformers.tokenization_utils_base import BatchEncoding
from typing import Literal

# model predicts one of the 1000 ImageNet classes
class ImageClassificationModel(BaseVisionModel):
    def _load_model(self) -> PreTrainedModel | TextGenerationVendor:
        self._processor = AutoImageProcessor.from_pretrained(
            self._model_id,
            # default behavior in transformers v4.48
            use_fast=True
        )
        model = AutoModelForImageClassification.from_pretrained(self._model_id)
        return model

    async def __call__(
        self,
        image_source: str | Image.Image,
        tensor_format: Literal["pt"]="pt"
    ) -> ImageEntity:
        image = BaseVisionModel._get_image(image_source)
        inputs = self._processor(image, return_tensors=tensor_format)

        with no_grad():
            logits = self._model(**inputs).logits

        label_index = logits.argmax(dim=1).item()
        return ImageEntity(
            label=self._model.config.id2label[label_index]
        )

class ImageToTextModel(TransformerModel):
    def _load_model(self) -> PreTrainedModel | TextGenerationVendor:
        self._processor = AutoImageProcessor.from_pretrained(
            self._model_id,
            # default behavior in transformers v4.48
            use_fast=True
        )
        model = AutoModelForVision2Seq.from_pretrained(self._model_id)
        return model

    def _tokenize_input(
        self,
        input: Input,
        context: str | None=None,
        tensor_format: Literal["pt"]="pt",
        **kwargs
    ) -> (
        dict[str,Tensor] | BatchEncoding | Tensor
    ):
        raise NotImplementedError()

    async def __call__(
        self,
        image_source: str | Image.Image,
        *,
        skip_special_tokens: bool=True,
        tensor_format: Literal["pt"]="pt"
    ) -> str:
        image = BaseVisionModel._get_image(image_source)
        inputs = self._processor(images=image, return_tensors=tensor_format)
        output_ids = self._model.generate(**inputs)
        caption = self._tokenizer.decode(
            output_ids[0],
            skip_special_tokens=skip_special_tokens
        )
        return caption


class ConditionalVisionGenerationModel(ImageToTextModel):
    def _load_model(self) -> PreTrainedModel | TextGenerationVendor:
        self._processor = AutoProcessor.from_pretrained(
            self._model_id,
            use_fast=True,
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self._model_id,
            torch_dtype=BaseNLPModel._get_weight_type(
                self._settings.weight_type
            ),
            device_map=self._device,
        )
        return model

    @override
    async def __call__(
        self,
        image_source: str | Image.Image,
        prompt: str,
        settings: GenerationSettings | None=None,
        *,
        skip_special_tokens: bool=True,
        tensor_format: Literal["pt"]="pt"
    ) -> str:
        image = BaseVisionModel._get_image(image_source).convert("RGB")

        messages = [
            {
                "role": str(MessageRole.USER),
                "content": [
                    {
                        "type": "image",
                        "image": image
                    },
                    {
                        "type": "text",
                        "text": prompt
                    },
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text],
            images=image,
            videos=None,
            padding=True,
            return_tensors=tensor_format,
        )
        inputs = inputs.to(self._device)
        generated_ids = self._model.generate(
            **inputs,
            max_new_tokens=settings.max_new_tokens
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False
        )
        return output_text


class VisionEncoderDecoderModel(ImageToTextModel):
    def _load_model(self) -> PreTrainedModel | TextGenerationVendor:
        self._processor = AutoImageProcessor.from_pretrained(
            self._model_id,
            use_fast=True,
        )
        model = HFVisionEncoderDecoderModel.from_pretrained(self._model_id)
        return model


