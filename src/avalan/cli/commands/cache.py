from argparse import Namespace
from ...cli import confirm
from ...cli.download import create_live_tqdm_class
from ...model.hubs import HubAccessDeniedException
from ...model.hubs.huggingface import HuggingfaceHub
from rich.console import Console
from rich.padding import Padding
from rich.theme import Theme


def cache_delete(
    args: Namespace,
    console: Console,
    theme: Theme,
    hub: HuggingfaceHub,
    is_full_deletion: bool = False,
) -> None:
    assert args.model
    _ = theme._
    model_id = args.model
    cache_deletion, execute_deletion = hub.cache_delete(
        model_id, None if is_full_deletion else args.delete_revision
    )
    if not cache_deletion or not execute_deletion:
        console.print(theme.cache_delete(cache_deletion, False))
        return

    console.print(theme.cache_delete(cache_deletion))
    if not args.delete and not confirm(console, theme.ask_delete_paths()):
        return
    execute_deletion()
    console.print(
        Padding(theme.cache_delete(cache_deletion, True), pad=(1, 0, 0, 0))
    )


def cache_download(
    args: Namespace, console: Console, theme: Theme, hub: HuggingfaceHub
) -> None:
    assert args.model
    model_id = args.model
    can_access = args.skip_hub_access_check or hub.can_access(model_id)
    model = hub.model(model_id)
    console.print(theme.model(model, can_access=can_access))
    console.print(theme.download_start(model_id))
    progress_template = theme.download_progress()
    try:
        path = hub.download(
            model_id, tqdm_class=create_live_tqdm_class(progress_template)
        )
        console.print(theme.download_finished(model_id, path))
    except HubAccessDeniedException:
        model_url = hub.model_url(model_id)
        console.print(theme.download_access_denied(model_id, model_url))


def cache_list(
    args: Namespace, console: Console, theme: Theme, hub: HuggingfaceHub
) -> None:
    cached_models = hub.cache_scan()
    console.print(
        theme.cache_list(
            hub.cache_dir, cached_models, args.model, args.summary
        )
    )
