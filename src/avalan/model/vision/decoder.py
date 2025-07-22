from ...compat import override
from ...model.vendor import TextGenerationVendor
from ...model.engine import Engine
from ...model.vision import BaseVisionModel
from ...model.vision.text import ImageToTextModel
from diffusers import DiffusionPipeline
from PIL import Image
from torch import inference_mode
from transformers import (
    AutoImageProcessor,
    PreTrainedModel,
    VisionEncoderDecoderModel as VisionEncoderDecoderModelImpl,
)
from typing import Literal


class VisionEncoderDecoderModel(ImageToTextModel):
    def _load_model(
        self,
    ) -> PreTrainedModel | TextGenerationVendor | DiffusionPipeline:
        self._processor = AutoImageProcessor.from_pretrained(
            self._model_id,
            use_fast=True,
        )
        model = VisionEncoderDecoderModelImpl.from_pretrained(
            self._model_id,
            device_map=self._device,
            tp_plan=Engine._get_tp_plan(self._settings.parallel),
        )
        return model

    @override
    async def __call__(
        self,
        image_source: str | Image.Image,
        prompt: str | None,
        *,
        early_stopping: bool = True,
        num_beams: int = 1,
        skip_special_tokens: bool = True,
        tensor_format: Literal["pt"] = "pt",
        use_cache: bool = True,
    ) -> str:
        if not prompt:
            return await super().__call__(
                image_source=image_source,
                skip_special_tokens=skip_special_tokens,
                tensor_format=tensor_format,
            )

        image = BaseVisionModel._get_image(image_source)
        pixel_values = self._processor(
            image, return_tensors=tensor_format
        ).pixel_values.to(self._device)
        decoder_input_ids = self._tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self._device)

        with inference_mode():
            outputs = self._model.generate(
                pixel_values=pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=self._model.decoder.config.max_position_embeddings,
                early_stopping=early_stopping,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                use_cache=use_cache,
                num_beams=num_beams,
                bad_words_ids=[[self._tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

        output = self._tokenizer.batch_decode(
            outputs.sequences, skip_special_tokens=skip_special_tokens
        )[0]
        return output
