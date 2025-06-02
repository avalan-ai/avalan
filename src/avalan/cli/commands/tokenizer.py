from argparse import Namespace
from ...cli import get_input
from ...entities import Token, TransformerEngineSettings
from ...model.hubs.huggingface import HuggingfaceHub
from ...model.nlp.text.generation import TextGenerationModel
from logging import Logger
from rich.console import Console
from rich.theme import Theme
from typing import Optional


async def tokenize(
    args: Namespace,
    console: Console,
    theme: Theme,
    hub: HuggingfaceHub,
    logger: Logger,
) -> Optional[list[Token]]:
    assert args.tokenizer

    _, _i, _n = theme._, theme.icons, theme._n

    tokenizer_name_or_path = args.tokenizer
    with TextGenerationModel(
        tokenizer_name_or_path,
        settings=TransformerEngineSettings(
            args.device,
            hub.cache_dir,
            tokenizer_name_or_path=tokenizer_name_or_path,
            tokens=args.token
            if args.token and isinstance(args.token, list)
            else None,
            special_tokens=args.special_token
            if args.special_token and isinstance(args.special_token, list)
            else None,
            auto_load_model=False,
            auto_load_tokenizer=True,
            disable_loading_progress_bar=args.disable_loading_progress_bar,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            loader_class=args.loader_class,
            weight_type=args.weight_type,
        ),
        logger=logger,
    ) as lm:
        logger.debug(f"Loaded tokenizer {lm.tokenizer_config.__repr__()}")
        console.print(theme.tokenizer_config(lm.tokenizer_config))

        if args.save:
            paths = lm.save_tokenizer(args.save)
            total_files = len(paths)
            console.print(theme.saved_tokenizer_files(args.save, total_files))
            return

        input_string = get_input(
            console,
            _i["user_input"] + " ",
            echo_stdin=not args.no_repl,
            is_quiet=args.quiet,
        )
        if input_string:
            logger.debug(f"Loaded model {lm.config.__repr__()}")
            tokens = lm.tokenize(input_string)

            panel = theme.tokenizer_tokens(
                tokens,
                lm.tokenizer_config.tokens,
                lm.tokenizer_config.special_tokens,
                display_details=True,
            )
            console.print(panel)
