from ...compat import override
from ...model.audio import BaseAudioModel
from ...model.engine import Engine
from ...model.vendor import TextGenerationVendor

from typing import Any, Literal

from diffusers import DiffusionPipeline
from torch import inference_mode
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    PreTrainedModel,
)


class AudioClassificationModel(BaseAudioModel):
    _extractor: Any  # AutoFeatureExtractor with model-specific methods

    def _load_model(
        self,
    ) -> PreTrainedModel | TextGenerationVendor | DiffusionPipeline:
        assert self._model_id, "model_id is required"
        self._extractor = AutoFeatureExtractor.from_pretrained(self._model_id)
        model: PreTrainedModel = (
            AutoModelForAudioClassification.from_pretrained(
                self._model_id,
                device_map=self._device,
                tp_plan=Engine._get_tp_plan(self._settings.parallel),
                distributed_config=Engine._get_distributed_config(
                    self._settings.distributed_config
                ),
                subfolder=self._settings.subfolder or "",
            ).to(self._device)
        )

        return model

    @override
    async def __call__(  # type: ignore[override]
        self,
        path: str,
        *,
        padding: bool = True,
        sampling_rate: int = 16_000,
        tensor_format: Literal["pt"] = "pt",
    ) -> dict[str, float]:
        assert path
        assert self._model is not None, "Model must be loaded"
        assert isinstance(self._model, PreTrainedModel)

        wave = self._resample_mono(path, sampling_rate)
        inputs = self._extractor(
            wave,
            sampling_rate=sampling_rate,
            return_tensors=tensor_format,
            padding=padding,
        ).to(self._device)

        id2label_raw = self._model.config.id2label or {}
        id2label: dict[int, str] = {
            int(k): str(v) for k, v in id2label_raw.items()
        }
        labels: dict[str, float] = {}

        with inference_mode():
            logits = self._model(**inputs).logits  # type: ignore[operator]
            probs = logits.softmax(dim=-1)[0]
            for idx, p in sorted(enumerate(probs), key=lambda x: -x[1]):
                labels[id2label[idx]] = p.item()

        return labels
