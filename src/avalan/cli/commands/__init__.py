from ...entities import EngineUri, Modality
from ...model.hubs.huggingface import HuggingfaceHub
from argparse import Namespace
from logging import Logger


def get_model_settings(
    args: Namespace,
    hub: HuggingfaceHub,
    logger: Logger,
    engine_uri: EngineUri,
    modality: Modality | None = None,
) -> dict:
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
        base_url=args.base_url if hasattr(args, "base_url") else None,
        engine_uri=engine_uri,
        attention=args.attention if hasattr(args, "attention") else None,
        device=args.device,
        disable_loading_progress_bar=args.disable_loading_progress_bar,
        modality=modality,
        loader_class=args.loader_class,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        quiet=args.quiet,
        revision=args.revision,
        parallel=args.parallel if hasattr(args, "parallel") else None,
        special_tokens=(
            args.special_token
            if args.special_token and isinstance(args.special_token, list)
            else None
        ),
        tokenizer=args.tokenizer or None,
        tokens=(
            args.token if args.token and isinstance(args.token, list) else None
        ),
        trust_remote_code=(
            args.trust_remote_code
            if hasattr(args, "trust_remote_code")
            else None
        ),
        weight_type=args.weight_type,
    )
