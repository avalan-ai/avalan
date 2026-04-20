from ...entities import (
    AttentionImplementation,
    Backend,
    EngineUri,
    Modality,
    ParallelStrategy,
    TextGenerationLoaderClass,
    WeightType,
)
from ...model.hubs.huggingface import HuggingfaceHub

from argparse import Namespace
from logging import Logger
from typing import TypedDict


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


def get_model_settings(
    args: Namespace,
    hub: HuggingfaceHub,
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
    return dict(
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
