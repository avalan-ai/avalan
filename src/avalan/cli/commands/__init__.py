from ...entities import (
    AttentionImplementation,
    Backend,
    EngineUri,
    Modality,
    ParallelStrategy,
    TextGenerationLoaderClass,
    WeightType,
)
from ...model.manager import ModelManager

from argparse import Namespace
from logging import Logger
from typing import Any, NotRequired, TypedDict


class ModelSettings(TypedDict):
    """Define typed settings used by ModelManager.load."""

    base_url: str | None
    engine_uri: EngineUri
    attention: AttentionImplementation | None
    output_hidden_states: bool | None
    device: str | None
    disable_loading_progress_bar: bool
    modality: Modality
    loader_class: TextGenerationLoaderClass | None
    backend: Backend
    low_cpu_mem_usage: bool
    quiet: bool
    revision: str | None
    parallel: ParallelStrategy | None
    base_model_id: str | None
    checkpoint: str | None
    refiner_model_id: str | None
    upsampler_model_id: str | None
    special_tokens: list[str] | None
    tokenizer: str | None
    tokens: list[str] | None
    subfolder: str | None
    tokenizer_subfolder: str | None
    trust_remote_code: bool | None
    weight_type: WeightType
    backend_config: NotRequired[dict[str, object] | None]


def _normalize_modality(modality: Modality | str) -> Modality:
    return modality if isinstance(modality, Modality) else Modality(modality)


def is_ds4_backend_selected(args: Namespace, engine_uri: EngineUri) -> bool:
    """Return whether CLI or URI settings select the DS4 backend."""
    params = getattr(engine_uri, "params", {})
    uri_backend = params.get("backend") if isinstance(params, dict) else None
    cli_backend = getattr(args, "backend", None)
    return uri_backend == Backend.DS4.value or cli_backend in {
        Backend.DS4,
        Backend.DS4.value,
    }


def get_model_settings(
    args: Namespace,
    hub: Any,
    logger: Logger,
    engine_uri: EngineUri,
    modality: Modality | None = None,
) -> ModelSettings:
    """Return settings used to load a model."""
    modality = (
        modality
        or getattr(args, "modality", None)
        or (
            Modality.EMBEDDING
            if hasattr(args, "sentence_transformer")
            and args.sentence_transformer
            else None
        )
        or Modality.TEXT_GENERATION
    )
    modality = _normalize_modality(modality)
    settings: ModelSettings = dict(
        base_url=getattr(args, "base_url", None),
        engine_uri=engine_uri,
        attention=getattr(args, "attention", None),
        output_hidden_states=getattr(args, "output_hidden_states", False),
        device=args.device,
        disable_loading_progress_bar=args.disable_loading_progress_bar,
        modality=modality,
        loader_class=args.loader_class,
        backend=args.backend,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        quiet=args.quiet,
        revision=args.revision,
        parallel=getattr(args, "parallel", None),
        base_model_id=getattr(args, "base_model", None),
        checkpoint=getattr(args, "checkpoint", None),
        refiner_model_id=getattr(args, "refiner_model", None),
        upsampler_model_id=getattr(args, "upsampler_model", None),
        special_tokens=(
            args.special_token
            if args.special_token and isinstance(args.special_token, list)
            else None
        ),
        tokenizer=args.tokenizer or None,
        tokens=(
            args.token if args.token and isinstance(args.token, list) else None
        ),
        subfolder=getattr(args, "subfolder", None),
        tokenizer_subfolder=getattr(args, "tokenizer_subfolder", None),
        trust_remote_code=getattr(args, "trust_remote_code", None),
        weight_type=args.weight_type,
    )

    uri_backend = engine_uri.params.get("backend")
    resolved_backend = (
        uri_backend if isinstance(uri_backend, str) else settings["backend"]
    )
    if resolved_backend == Backend.DS4.value:
        backend_config = ModelManager.ds4_backend_config_from_mapping(
            vars(args)
        )
        if backend_config:
            settings["backend_config"] = backend_config

    return settings
