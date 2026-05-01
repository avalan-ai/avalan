from ...model.audio import BaseAudioModel
from ...model.engine import Engine
from ...model.vendor import TextGenerationVendor

from copy import deepcopy
from typing import Any, Literal, cast

from diffusers import DiffusionPipeline
from torch import inference_mode
from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    DiaConfig,
    DiaForConditionalGeneration,
    DiaProcessor,
    PreTrainedModel,
)

_DIA_TOKEN_ID_FIELDS = ("pad_token_id", "eos_token_id", "bos_token_id")


class TextToSpeechModel(BaseAudioModel):
    _processor: Any

    def _hub_kwargs(self, subfolder: str | None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"subfolder": subfolder or ""}
        if self._settings.cache_dir is not None:
            kwargs["cache_dir"] = self._settings.cache_dir
        if self._settings.access_token is not None:
            kwargs["token"] = self._settings.access_token
        if self._settings.revision is not None:
            kwargs["revision"] = self._settings.revision
        return kwargs

    @staticmethod
    def _normalize_dia_config_dict(
        config_dict: dict[str, Any],
    ) -> dict[str, Any]:
        normalized = deepcopy(config_dict)
        decoder_config = normalized.get("decoder_config") or {}
        assert isinstance(decoder_config, dict)
        decoder_config = deepcopy(decoder_config)

        for token_id_field in _DIA_TOKEN_ID_FIELDS:
            token_id = normalized.pop(token_id_field, None)
            if token_id is not None:
                decoder_config[token_id_field] = token_id

        normalized["decoder_config"] = decoder_config
        return normalized

    @staticmethod
    def _sync_dia_config_token_ids(config: DiaConfig) -> None:
        for token_id_field in _DIA_TOKEN_ID_FIELDS:
            token_id = getattr(config.decoder_config, token_id_field, None)
            if token_id is not None:
                setattr(config, token_id_field, token_id)

    def _load_dia_config(self) -> DiaConfig:
        assert self._model_id
        config_dict, _ = DiaConfig.get_config_dict(
            self._model_id,
            **self._hub_kwargs(self._settings.subfolder),
        )
        config = DiaConfig.from_dict(
            self._normalize_dia_config_dict(config_dict)
        )
        self._sync_dia_config_token_ids(config)
        return config

    def _load_processor(self, config: DiaConfig) -> DiaProcessor:
        assert self._model_id
        processor_kwargs = {
            **self._hub_kwargs(self._settings.tokenizer_subfolder),
            "trust_remote_code": self._settings.trust_remote_code,
        }
        processor_dict, processor_init_kwargs = (
            DiaProcessor.get_processor_dict(
                self._model_id,
                **processor_kwargs,
            )
        )
        feature_extractor = cast(Any, AutoFeatureExtractor).from_pretrained(
            self._model_id,
            **processor_kwargs,
        )
        tokenizer = cast(Any, AutoTokenizer).from_pretrained(
            self._model_id,
            config=config,
            **processor_kwargs,
        )
        return cast(
            DiaProcessor,
            DiaProcessor.from_args_and_dict(
                [feature_extractor, tokenizer],
                processor_dict,
                **processor_init_kwargs,
            ),
        )

    def _load_model(
        self,
    ) -> PreTrainedModel | TextGenerationVendor | DiffusionPipeline:
        config = self._load_dia_config()
        self._processor = self._load_processor(config)
        model = cast(
            PreTrainedModel,
            cast(Any, DiaForConditionalGeneration).from_pretrained(
                self._model_id,
                config=config,
                trust_remote_code=self._settings.trust_remote_code,
                device_map=self._device,
                tp_plan=Engine._get_tp_plan(self._settings.parallel),
                distributed_config=Engine._get_distributed_config(
                    self._settings.distributed_config
                ),
                **self._hub_kwargs(self._settings.subfolder),
            ),
        )
        return model

    async def __call__(
        self,
        prompt: str,
        path: str,
        max_new_tokens: int,
        *,
        padding: bool = True,
        reference_path: str | None = None,
        reference_text: str | None = None,
        sampling_rate: int = 44_100,
        tensor_format: Literal["pt"] = "pt",
    ) -> str:
        assert (not reference_path and not reference_text) or (
            reference_path and reference_text
        )

        reference_voice = None
        if reference_path and reference_text:
            reference_voice = self._resample(reference_path, sampling_rate)

        text = (
            f"{reference_text}\n{prompt}"
            if reference_voice is not None
            else prompt
        )

        inputs = self._processor(
            text=text,
            audio=reference_voice,
            padding=padding,
            return_tensors=tensor_format,
            sampling_rate=sampling_rate,
        ).to(self._device)

        prompt_len = (
            self._processor.get_audio_prompt_len(
                inputs["decoder_attention_mask"]
            )
            if reference_voice is not None
            else None
        )

        with inference_mode():
            outputs = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens
            )

        wave = (
            self._processor.batch_decode(outputs, audio_prompt_len=prompt_len)
            if prompt_len and outputs.shape[1] >= prompt_len
            else self._processor.batch_decode(outputs)
        )

        self._processor.save_audio(wave, path)
        return path
