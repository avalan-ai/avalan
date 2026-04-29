from ...entities import (
    GenerationSettings,
    Input,
    MessageRole,
)
from ...model.engine import Engine
from ...model.transformer import TransformerModel
from ...model.vision import BaseVisionModel

from typing import Any, Literal, cast

from PIL import Image
from torch import Tensor, inference_mode
from transformers import (
    AutoImageProcessor,
    AutoModelForImageTextToText,
    AutoProcessor,
    Gemma3ForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
)
from transformers.tokenization_utils_base import BatchEncoding

AutoModelForVision2Seq = AutoModelForImageTextToText


class ImageToTextModel(TransformerModel):
    _processor: Any

    @staticmethod
    def _generation_kwargs(
        settings: GenerationSettings | None,
    ) -> dict[str, int]:
        if not settings:
            return {}
        if settings.max_new_tokens is not None:
            return {"max_new_tokens": settings.max_new_tokens}
        if settings.max_length is not None:
            return {"max_length": settings.max_length}
        return {}

    def _load_model(
        self,
    ) -> Any:
        self._processor = cast(
            Any,
            cast(Any, AutoImageProcessor).from_pretrained(
                self._model_id,
                backend="torchvision",
            ),
        )
        model = cast(
            Any,
            cast(Any, AutoModelForVision2Seq).from_pretrained(
                self._model_id,
                device_map=self._device,
                tp_plan=Engine._get_tp_plan(self._settings.parallel),
                distributed_config=Engine._get_distributed_config(
                    self._settings.distributed_config
                ),
            ),
        )
        return model

    def _tokenize_input(
        self,
        input: Input,
        context: str | None = None,
        tensor_format: Literal["pt"] = "pt",
        **kwargs: object,
    ) -> dict[str, Tensor] | BatchEncoding | Tensor:
        raise NotImplementedError()

    async def __call__(
        self,
        image_source: object,
        *,
        settings: GenerationSettings | None = None,
        skip_special_tokens: bool = True,
        tensor_format: Literal["pt"] = "pt",
    ) -> str:
        image = BaseVisionModel._get_image(
            cast(str | Image.Image, image_source)
        )

        inputs = self._processor(images=image, return_tensors=tensor_format)
        inputs.to(self._device)

        with inference_mode():
            output_ids = self._model.generate(
                **inputs, **self._generation_kwargs(settings)
            )

        output = self._tokenizer.decode(
            output_ids[0], skip_special_tokens=skip_special_tokens
        )
        return cast(str, output)


class ImageTextToTextModel(ImageToTextModel):
    _loaders: dict[str, Any] = {
        "auto": AutoModelForImageTextToText,
        "qwen2": Qwen2VLForConditionalGeneration,
        "gemma3": Gemma3ForConditionalGeneration,
    }

    def _load_model(
        self,
    ) -> Any:
        assert (
            self._settings.loader_class in self._loaders
        ), f"Unrecognized loader {self._settings.loader_class}"

        self._processor = cast(
            Any,
            cast(Any, AutoProcessor).from_pretrained(
                self._model_id,
                backend="torchvision",
            ),
        )

        loader = self._loaders[self._settings.loader_class]
        model = cast(
            Any,
            cast(Any, loader).from_pretrained(
                self._model_id,
                torch_dtype=Engine.weight(self._settings.weight_type),
                device_map=self._device,
                tp_plan=Engine._get_tp_plan(self._settings.parallel),
                distributed_config=Engine._get_distributed_config(
                    self._settings.distributed_config
                ),
            ),
        )
        return model

    async def __call__(
        self,
        image_source: object,
        prompt: object,
        system_prompt: str | None = None,
        developer_prompt: str | None = None,
        settings: GenerationSettings | None = None,
        width: int | None = None,
        *,
        skip_special_tokens: bool = True,
        tensor_format: Literal["pt"] = "pt",
    ) -> str:
        generation_settings = settings or GenerationSettings()
        image = BaseVisionModel._get_image(
            cast(str | Image.Image, image_source)
        ).convert("RGB")
        assert image.width
        prompt_text = cast(str, prompt or "")

        if width:
            ratio = width / image.width
            height = int(ratio * image.height)
            image = image.resize((width, height), Image.Resampling.LANCZOS)

        messages = []
        if system_prompt:
            messages.append(
                {
                    "role": str(MessageRole.SYSTEM),
                    "content": [{"type": "text", "text": system_prompt}],
                }
            )
        if developer_prompt:
            messages.append(
                {
                    "role": str(MessageRole.DEVELOPER),
                    "content": [{"type": "text", "text": developer_prompt}],
                }
            )
        messages.append(
            {
                "role": str(MessageRole.USER),
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        )

        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=generation_settings.chat_settings.add_generation_prompt,
        )
        inputs = self._processor(
            text=[text],
            images=image,
            videos=None,
            padding=True,
            return_tensors=tensor_format,
        )
        if generation_settings.use_inputs_attention_mask:
            inputs.pop("attention_mask", None)

        inputs.to(self._device)
        with inference_mode():
            generated_ids = self._model.generate(
                **inputs, max_new_tokens=generation_settings.max_new_tokens
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )
        return cast(
            str,
            output_text[0] if isinstance(output_text, list) else output_text,
        )
