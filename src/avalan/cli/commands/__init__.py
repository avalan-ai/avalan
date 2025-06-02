from ...entities import EngineUri
from ...model.hubs.huggingface import HuggingfaceHub
from argparse import Namespace
from logging import Logger


def get_model_settings(
    args: Namespace,
    hub: HuggingfaceHub,
    logger: Logger,
    engine_uri: EngineUri,
    is_sentence_transformer: bool | None = None,
) -> dict:
    """Return settings used to load a model."""

    return dict(
        engine_uri=engine_uri,
        attention=args.attention if hasattr(args, "attention") else None,
        device=args.device,
        disable_loading_progress_bar=args.disable_loading_progress_bar,
        is_sentence_transformer=is_sentence_transformer
        or (
            hasattr(args, "sentence_transformer") and args.sentence_transformer
        ),
        loader_class=args.loader_class,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        quiet=args.quiet,
        revision=args.revision,
        special_tokens=args.special_token
        if args.special_token and isinstance(args.special_token, list)
        else None,
        tokenizer=args.tokenizer or None,
        tokens=args.token
        if args.token and isinstance(args.token, list)
        else None,
        trust_remote_code=(
            args.trust_remote_code
            if hasattr(args, "trust_remote_code")
            else None
        ),
        weight_type=args.weight_type,
    )
