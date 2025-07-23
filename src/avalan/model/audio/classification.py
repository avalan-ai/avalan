from ...compat import override
from ...model.vendor import TextGenerationVendor
from ...model.audio import BaseAudioModel
from ...model.engine import Engine
from diffusers import DiffusionPipeline
from torch import inference_mode
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    PreTrainedModel,
)
from typing import Literal


class AudioClassificationModel(BaseAudioModel):
    _extractor: AutoFeatureExtractor

    def _load_model(
        self,
    ) -> PreTrainedModel | TextGenerationVendor | DiffusionPipeline:
        self._extractor = AutoFeatureExtractor.from_pretrained(self._model_id)
        model = AutoModelForAudioClassification.from_pretrained(
            self._model_id,
            device_map=self._device,
            tp_plan=Engine._get_tp_plan(self._settings.parallel),
            subfolder=self._settings.subfolder or "",
        ).to(self._device)

        return model

    @override
    async def __call__(
        self,
        path: str,
        *,
        padding: bool = True,
        sampling_rate: int = 16_000,
        tensor_format: Literal["pt"] = "pt",
    ) -> dict[str, float]:
        assert path

        wave = self._resample_mono(path, sampling_rate)
        inputs = self._extractor(
            wave,
            sampling_rate=sampling_rate,
            return_tensors=tensor_format,
            padding=padding,
        ).to(self._device)

        id2label = {int(k): v for k, v in self._model.config.id2label.items()}
        labels: dict[str, float] = {}

        with inference_mode():
            logits = self._model(**inputs).logits
            probs = logits.softmax(dim=-1)[0]
            for idx, p in sorted(enumerate(probs), key=lambda x: -x[1]):
                labels[id2label[idx]] = p.item()

        return labels
