from argparse import Namespace
from asyncio import as_completed, create_task, gather, sleep, to_thread
from ...agent.orchestrator import Orchestrator
from ...event import Event, EventType
from ...cli import get_input, confirm
from ...cli.commands.cache import cache_delete, cache_download
from ...entities import GenerationSettings, Model, Token, TokenDetail
from ...event import EventStats
from ...model import TextGenerationResponse
from ...model.hubs.huggingface import HuggingfaceHub
from ...model.manager import ModelManager
from ...model.criteria import KeywordStoppingCriteria
from ...model.nlp.sentence import SentenceTransformerModel
from ...model.nlp.text.generation import TextGenerationModel
from ...secrets import KeyringSecrets
from . import get_model_settings
from rich.prompt import Prompt
from logging import Logger
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.padding import Padding
from rich.spinner import Spinner
from rich.theme import Theme
from time import perf_counter
from typing import Tuple, Union


def model_display(
    args: Namespace,
    console: Console,
    theme: Theme,
    hub: HuggingfaceHub,
    logger: Logger,
    *vargs,
    is_sentence_transformer: bool | None = None,
    load: bool | None = None,
    model: Union[SentenceTransformerModel, TextGenerationModel] | None = None,
    summary: bool | None = None,
) -> None:
    assert args.model
    _ = theme._

    with ModelManager(hub, logger) as manager:
        engine_uri = manager.parse_uri(args.model)
        model_id = args.model
        can_access = args.skip_hub_access_check or hub.can_access(model_id)
        hub_model = hub.model(model_id)
        console.print(
            theme.model(
                hub_model,
                can_access=can_access,
                expand=(summary is not None and not summary)
                or (summary is None and not args.summary),
                summary=False,
            )
        )

        if not model and (
            (load is not None and load) or (load is None and args.load)
        ):
            model_settings = get_model_settings(
                args,
                hub,
                logger,
                engine_uri,
                is_sentence_transformer=is_sentence_transformer,
            )
            with manager.load(**model_settings) as lm:
                logger.debug(f"Loaded model {lm.config.__repr__()}")
                console.print(
                    Padding(
                        theme.model_display(
                            lm.config,
                            lm.tokenizer_config,
                            summary=summary or False,
                        ),
                        pad=(0, 0, 0, 0),
                    )
                )
        elif model:
            console.print(
                Padding(
                    theme.model_display(
                        model.config,
                        model.tokenizer_config,
                        summary=summary or False,
                    ),
                    pad=(0, 0, 0, 0),
                )
            )


def model_install(
    args: Namespace, console: Console, theme: Theme, hub: HuggingfaceHub
) -> None:
    assert args.model
    engine_uri = ModelManager.parse_uri(args.model)
    if (
        engine_uri.vendor
        and engine_uri.password
        and engine_uri.user == "secret"
    ):
        secrets = KeyringSecrets()
        token = secrets.read(engine_uri.password)
        if token is None:
            secret_value = Prompt.ask(
                theme.ask_secret_password(engine_uri.password)
            )
            secrets.write(engine_uri.password, secret_value)
        elif confirm(console, theme.ask_override_secret(engine_uri.password)):
            secret_value = Prompt.ask(
                theme.ask_secret_password(engine_uri.password)
            )
            secrets.write(engine_uri.password, secret_value)

    cache_download(args, console, theme, hub)


async def model_run(
    args: Namespace,
    console: Console,
    theme: Theme,
    hub: HuggingfaceHub,
    refresh_per_second: int,
    logger: Logger,
) -> None:
    assert args.model and args.device and args.max_new_tokens
    _, _i = theme._, theme.icons

    system_prompt = args.system or None

    with ModelManager(hub, logger) as manager:
        engine_uri = manager.parse_uri(args.model)
        model_settings = get_model_settings(
            args, hub, logger, engine_uri, is_sentence_transformer=False
        )

        if not args.quiet:
            if engine_uri.is_local:
                can_access = (
                    args.quiet
                    or args.skip_hub_access_check
                    or hub.can_access(engine_uri.model_id)
                )

                model = hub.model(engine_uri.model_id)
                console.print(
                    Padding(
                        theme.model(model, can_access=can_access, summary=True),
                        pad=(0, 0, 1, 0),
                    )
                )

        with manager.load(**model_settings) as lm:
            logger.debug(f"Loaded model {lm.config.__repr__()}")

            input_string = get_input(
                console,
                _i["user_input"] + " ",
                echo_stdin=not args.no_repl,
                is_quiet=args.quiet,
            )
            if not input_string:
                return

            settings = GenerationSettings(
                do_sample=args.do_sample,
                enable_gradient_calculation=args.enable_gradient_calculation,
                max_new_tokens=args.max_new_tokens,
                min_p=args.min_p,
                repetition_penalty=args.repetition_penalty,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                use_cache=args.use_cache,
            )

            display_tokens = args.display_tokens or 0
            dtokens_pick = 10 if display_tokens > 0 else 0

            if engine_uri.is_local:
                stopping_criteria = (
                    KeywordStoppingCriteria(args.stop_on_keyword, lm.tokenizer)
                    if args.stop_on_keyword
                    else None
                )
                output_generator = lm(
                    input_string,
                    system_prompt=system_prompt,
                    settings=settings,
                    stopping_criterias=[stopping_criteria]
                    if stopping_criteria
                    else None,
                    manual_sampling=display_tokens,
                    pick=dtokens_pick,
                    skip_special_tokens=args.quiet or args.skip_special_tokens,
                )
            else:
                output_generator = lm(
                    input_string,
                    system_prompt=system_prompt,
                    settings=settings,
                )

            await token_generation(
                args=args,
                console=console,
                theme=theme,
                logger=logger,
                orchestrator=None,
                event_stats=None,
                lm=lm,
                input_string=input_string,
                refresh_per_second=refresh_per_second,
                response=await output_generator,
                dtokens_pick=dtokens_pick,
                display_tokens=display_tokens,
                with_stats=not args.quiet,
                tool_events_limit=args.tool_events,
            )


async def model_search(
    args: Namespace,
    console: Console,
    theme: Theme,
    hub: HuggingfaceHub,
    refresh_per_second: int,
) -> None:
    assert args.limit
    _ = theme._

    models: list[Model] = []
    model_access: dict[str, bool] = {}

    # Fetch matching models
    with console.status(
        _("Loading models..."),
        spinner=theme.get_spinner("downloading"),
        refresh_per_second=refresh_per_second,
    ):
        models = [
            model
            for model in hub.models(
                filter=args.filter or None,
                search=args.search or None,
                limit=args.limit,
            )
        ]

    # Tasks to check model access
    tasks = [
        create_task(to_thread(lambda id=model.id: (id, hub.can_access(id))))
        for model in models
    ]

    def _render(
        models: list[Model], model_access: dict[str, bool]
    ) -> list[RenderableType]:
        return [
            theme.model(
                model,
                can_access=model_access[model.id]
                if model.id in model_access
                else None,
            )
            for model in models
        ]

    # Keep list of models updated as tasks are completed
    with Live(
        Group(*_render(models, model_access)),
        console=console,
        refresh_per_second=refresh_per_second,
    ) as live:
        for completed_task in as_completed(tasks):
            model_id, can_access = await completed_task
            model_access[model_id] = can_access

            live.update(Group(*_render(models, model_access)))

        await gather(*tasks)


def model_uninstall(
    args: Namespace, console: Console, theme: Theme, hub: HuggingfaceHub
) -> None:
    assert args.model
    engine_uri = ModelManager.parse_uri(args.model)
    if (
        engine_uri.vendor
        and engine_uri.password
        and engine_uri.user == "secret"
    ):
        secrets = KeyringSecrets()
        secrets.delete(engine_uri.password)

    cache_delete(args, console, theme, hub, is_full_deletion=True)


async def token_generation(
    args: Namespace,
    console: Console,
    theme: Theme,
    logger: Logger,
    orchestrator: Orchestrator | None,
    event_stats: EventStats | None,
    lm: TextGenerationModel,
    input_string: str,
    response: TextGenerationResponse,
    *,
    display_tokens: int,
    dtokens_pick: int,
    refresh_per_second: int,
    tool_events_limit: int | None,
    with_stats: bool = True,
):
    # If no statistics needed, return as early as possible
    if not with_stats:
        async for token in response:
            if isinstance(token,Event):
                continue
            text_token = token.token if isinstance(token, Token) else token
            console.print(text_token, end="")
        return

    # From here on, display includes stats and may include token probabilities

    display_time_to_n_token = args.display_time_to_n_token or 256
    display_pause = (
        args.display_pause
        if args.display_pause and args.display_pause > 0
        else 0
    )
    start_thinking = (
        args.start_thinking if hasattr(args, "start_thinking") else False
    )
    tokens = []
    text_tokens: list[Union[Token, TokenDetail, str]] = []
    tool_events: list[Event] = []
    tool_event_calls: list[Event] = []
    tool_event_results: list[Event] = []
    total_tokens = 0
    frame_minimum_pause_ms = (
        100 if display_pause > 0 and display_tokens > 0 else 0
    )

    with Live(refresh_per_second=refresh_per_second) as live:
        start = perf_counter()
        input_token_count = (
            response.input_token_count
            if response.input_token_count
            else orchestrator.input_token_count
            if orchestrator
            else lm.input_token_count(input_string)
        )
        ttft: float | None = None
        ttnt: float | None = None
        token_frame_list: list[Tuple[Token | None, RenderableType]] = None
        last_current_dtoken: Token | None = None
        tool_running_spinner: Spinner | None = None

        async for token in response:
            if isinstance(token, Event):
                event = token
                tool_events.append(event)
                if event.type == EventType.TOOL_MODEL_RESPONSE:
                    tokens = []
                    text_tokens = []
                    inner_response = event.payload["response"]
                    assert isinstance(inner_response, TextGenerationResponse)
                    if inner_response.input_token_count:
                        input_token_count = inner_response.input_token_count
                elif event.type == EventType.TOOL_RESULT:
                    tool_event_results.append(event)
                else:
                    tool_event_calls.append(event)

            else:
                text_token = token.token if isinstance(token, Token) else token
                text_tokens.append(text_token)

            tool_running_spinner = None
            if tool_event_calls or tool_event_results:
                tool_calling_names = [
                    c.name
                    for e in tool_event_calls
                    for c in e.payload
                    if c.id not in {
                        r.payload["call"].id
                        for r in tool_event_results
                        if r.type == EventType.TOOL_RESULT and "call" in r.payload
                    }
                ]

                tool_running_spinner = Spinner(
                    theme.get_spinner("tool_running"),
                    text="[cyan]"+theme._n(
                        "Running tool {tool_names}...",
                        "Running tools {tool_names}...",
                        len(tool_calling_names)
                    ).format(
                        tool_names=", ".join(tool_calling_names)
                    ) + "[/cyan]",
                    style="cyan",
                    speed=1.0
                ) if tool_calling_names else None

            total_tokens = total_tokens + 1
            ellapsed = perf_counter() - start
            if ttft is None:
                ttft = ellapsed
            if ttnt is None and total_tokens >= display_time_to_n_token:
                ttnt = ellapsed

            if display_tokens and isinstance(token, Token):
                tokens.append(token)

            token_frames_promise = theme.tokens(
                lm.model_id,
                lm.tokenizer_config.tokens if lm.tokenizer_config else None,
                lm.tokenizer_config.special_tokens
                if lm.tokenizer_config
                else None,
                display_tokens,
                args.display_probabilities if dtokens_pick > 0 else False,
                dtokens_pick,
                # Which tokens to mark as interesting
                lambda dtoken: (
                    dtoken.probability < args.display_probabilities_maximum
                    or len(
                        [
                            t
                            for t in dtoken.tokens
                            if t.id != dtoken.id
                            and t.probability
                            >= args.display_probabilities_sample_minimum
                        ]
                    )
                    > 0
                )
                if display_tokens
                and args.display_probabilities
                and args.display_probabilities_maximum > 0
                and args.display_probabilities_maximum > 0
                else None,
                text_tokens,
                tokens or None,
                input_token_count,
                total_tokens,
                tool_events,
                tool_event_calls,
                tool_event_results,
                tool_running_spinner,
                ttft,
                ttnt,
                ellapsed,
                console.width,
                logger,
                event_stats,
                tool_events_limit=tool_events_limit,
                height=6,
                maximum_frames=1,
                start_thinking=start_thinking,
            )

            token_frame_list = [
                token_frame async for token_frame in token_frames_promise
            ]

            # We prioritize a single selected dtoken at a time, it being
            # the leftmost  selected which is also guaranteed by setting
            # minimum_frames=1 when calling theme.tokens()
            token_frames = [token_frame_list[0]]

            for current_dtoken, frame in token_frames:
                live.update(frame)
                if current_dtoken and current_dtoken != last_current_dtoken:
                    last_current_dtoken = current_dtoken
                    if display_pause > 0:
                        await sleep(display_pause / 1000)
                    elif frame_minimum_pause_ms > 0:
                        await sleep(frame_minimum_pause_ms / 1000)
                elif (
                    dtokens_pick > 0
                    and not args.display_probabilities
                    and display_pause > 0
                ):
                    await sleep(display_pause / 1000)

        if (
            dtokens_pick > 0
            and args.display_probabilities
            and token_frame_list
            and len(token_frame_list) > 0
        ):
            for current_dtoken, frame in token_frame_list[1:]:
                live.update(frame)
                if current_dtoken and display_pause > 0:
                    await sleep(display_pause / 1000)
                elif frame_minimum_pause_ms > 0:
                    await sleep(frame_minimum_pause_ms / 1000)
