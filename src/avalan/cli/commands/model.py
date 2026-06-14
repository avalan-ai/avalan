from ...agent import Specification
from ...agent.orchestrator import Orchestrator
from ...cli import confirm, get_input, has_input
from ...cli.commands.cache import cache_delete, cache_download
from ...cli.theme import Theme
from ...entities import (
    GenerationSettings,  # noqa: F401
    Input,
    Message,
    Modality,
    Model,
    Token,
    ToolCallToken,
)
from ...event import TOOL_TYPES, Event, EventStats, EventType
from ...model.call import ModelCall, ModelCallContext
from ...model.criteria import KeywordStoppingCriteria  # noqa: F401
from ...model.input import input_files
from ...model.manager import ModelManager
from ...model.nlp.text.generation import TextGenerationModel
from ...model.response.text import TextGenerationResponse
from ...model.stream import (
    CanonicalStreamAccumulator,
    CanonicalStreamItem,
    StreamChannel,
    StreamConsumerProjection,
    StreamValidationError,
    canonical_item_from_consumer_projection,
    project_canonical_stream_item,
    stream_consumer_projection_from_token,
    stream_projection_is_reasoning,
    stream_projection_is_tool_call,
    stream_projection_text_delta,
)
from ...secrets import KeyringSecrets
from . import ModelSettings, get_model_settings, is_ds4_backend_selected

from argparse import Namespace
from asyncio import (
    CancelledError,
    as_completed,
    create_task,
    gather,
    sleep,
    to_thread,
)
from asyncio import (
    Event as EventSignal,
)
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from functools import partial
from logging import Logger
from time import perf_counter
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    TypeAlias,
    cast,
)

if TYPE_CHECKING:
    from ...model.nlp.sentence import SentenceTransformerModel
else:
    SentenceTransformerModel: TypeAlias = Any

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.padding import Padding
from rich.prompt import Prompt
from rich.spinner import Spinner

_HAS_INPUT = has_input


@dataclass(frozen=True, slots=True)
class _StreamRenderItem:
    projection: StreamConsumerProjection | None = None
    event: Event | None = None
    source_token: Token | None = None


class _FrameRateRenderer:
    def __init__(
        self,
        args: Namespace,
        console: Console,
        live: Live,
        group: Group | None,
        group_index: int | None,
        *,
        refresh_per_second: int,
    ) -> None:
        assert refresh_per_second > 0
        self._args = args
        self._console = console
        self._live = live
        self._group = group
        self._group_index = group_index
        self._interval = 1 / refresh_per_second
        self._dirty = EventSignal()
        self._latest_frame: RenderableType | None = None
        self._latest_version = 0
        self._rendered_version = 0
        self._stopped = False
        self._task = create_task(self._run())

    def mark_dirty(self, frame: RenderableType) -> None:
        self._latest_frame = frame
        self._latest_version += 1
        self._dirty.set()

    async def close(self) -> None:
        self._stopped = True
        self._dirty.set()
        await self._task

    async def _run(self) -> None:
        while True:
            await self._dirty.wait()
            self._dirty.clear()
            if self._latest_version == self._rendered_version:
                if self._stopped:
                    return
                continue

            frame = self._latest_frame
            assert frame is not None
            version = self._latest_version
            await to_thread(
                _render_frame,
                self._args,
                self._console,
                self._live,
                frame,
                self._group,
                self._group_index,
            )
            self._rendered_version = version

            if (
                self._stopped
                and self._rendered_version == self._latest_version
            ):
                return
            await sleep(self._interval)


def _supports_optional_stdin(modality: Modality) -> bool:
    return modality in {
        Modality.VISION_ENCODER_DECODER,
    }


async def _text_generation_input(
    input_string: str | None, file_paths: list[str] | None
) -> Message | str | None:
    return await input_files(input_string, file_paths)


def model_display(
    args: Namespace,
    console: Console,
    theme: Theme,
    hub: Any,
    logger: Logger,
    *vargs: object,
    modality: Modality | None = None,
    load: bool | None = None,
    model: SentenceTransformerModel | TextGenerationModel | None = None,
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

        is_runnable = not engine_uri.is_local
        if not model and (
            (load is not None and load) or (load is None and args.load)
        ):
            model_settings: ModelSettings = get_model_settings(
                args,
                hub,
                logger,
                engine_uri,
                modality=modality,
            )
            with manager.load(**model_settings) as lm:
                logger.debug("Loaded model %s", lm.config.__repr__())
                is_runnable = bool(
                    lm.is_runnable(getattr(args, "device", None))
                )
                console.print(
                    Padding(
                        theme.model_display(
                            lm.config,
                            lm.tokenizer_config,
                            is_runnable=is_runnable,
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
                        is_runnable=is_runnable,
                        summary=summary or False,
                    ),
                    pad=(0, 0, 0, 0),
                )
            )


def model_install(
    args: Namespace, console: Console, theme: Theme, hub: Any
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
    hub: Any,
    refresh_per_second: int,
    logger: Logger,
) -> None:
    assert args.model and args.device and args.max_new_tokens
    _, _i = theme._, theme.icons

    with ModelManager(hub, logger) as manager:
        engine_uri = manager.parse_uri(args.model)
        model_settings: ModelSettings = get_model_settings(
            args, hub, logger, engine_uri
        )
        modality = model_settings["modality"]

        if not args.quiet:
            if engine_uri.is_local:
                if is_ds4_backend_selected(args, engine_uri):
                    can_access = True
                    hub_model_summary = None
                else:
                    can_access = (
                        args.quiet
                        or args.skip_hub_access_check
                        or hub.can_access(cast(str, engine_uri.model_id))
                    )
                    hub_model_summary = hub.model(
                        cast(str, engine_uri.model_id)
                    )

                if hub_model_summary is not None:
                    console.print(
                        Padding(
                            theme.model(
                                hub_model_summary,
                                can_access=can_access,
                                summary=True,
                            ),
                            pad=(0, 0, 1, 0),
                        )
                    )

        operation = ModelManager.get_operation_from_arguments(
            modality, args, None
        )

        with manager.load(**model_settings) as model:
            logger.debug("Loaded model %s", model.config.__repr__())

            tty_path = getattr(args, "tty", "/dev/tty") or "/dev/tty"
            input_file_paths = cast(
                list[str] | None, getattr(args, "input_file", None)
            )
            if input_file_paths:
                assert (
                    operation.modality == Modality.TEXT_GENERATION
                ), "--input-file is only supported for text generation"

            should_read_input = operation.requires_input or (
                _supports_optional_stdin(operation.modality)
                and has_input(console)
            )
            input_string: str | None = None
            if should_read_input:
                input_string = get_input(
                    console,
                    _i["user_input"] + " ",
                    echo_stdin=not args.no_repl,
                    is_quiet=args.quiet,
                    tty_path=tty_path,
                )
                if (
                    operation.requires_input
                    and not input_string
                    and not input_file_paths
                ):
                    return

            if operation.modality == Modality.TEXT_GENERATION:
                operation = replace(
                    operation,
                    input=await _text_generation_input(
                        input_string, input_file_paths
                    ),
                )
            elif input_string:
                operation = replace(operation, input=input_string)

            context = ModelCallContext(
                specification=Specification(role=None, goal=None),
                input=operation.input,
                engine_args={},
            )
            task = ModelCall(
                engine_uri=engine_uri,
                model=model,
                operation=operation,
                tool=None,
                context=context,
            )
            output = await manager(task)

            if operation.modality in {
                Modality.AUDIO_SPEECH_RECOGNITION,
                Modality.TEXT_QUESTION_ANSWERING,
                Modality.TEXT_SEQUENCE_CLASSIFICATION,
                Modality.TEXT_SEQUENCE_TO_SEQUENCE,
                Modality.TEXT_TRANSLATION,
                Modality.VISION_IMAGE_TO_TEXT,
                Modality.VISION_ENCODER_DECODER,
                Modality.VISION_IMAGE_TEXT_TO_TEXT,
            }:
                console.print(output)

            elif operation.modality == Modality.AUDIO_CLASSIFICATION:
                console.print(
                    theme.display_audio_labels(cast(dict[str, float], output))
                )

            elif operation.modality == Modality.AUDIO_TEXT_TO_SPEECH:
                console.print(f"Audio generated in {output}")

            elif operation.modality == Modality.AUDIO_GENERATION:
                console.print(f"Audio generated in {output}")

            elif operation.modality == Modality.TEXT_TOKEN_CLASSIFICATION:
                console.print(
                    theme.display_token_labels([cast(dict[str, str], output)])
                )

            elif operation.modality == Modality.TEXT_GENERATION:
                await token_generation(
                    args=args,
                    console=console,
                    theme=theme,
                    logger=logger,
                    orchestrator=None,
                    event_stats=None,
                    lm=cast(TextGenerationModel, model),
                    input_string=input_string or "",
                    refresh_per_second=refresh_per_second,
                    response=cast(TextGenerationResponse, output),
                    dtokens_pick=(
                        operation.parameters["text"].pick_tokens or 0
                        if operation.parameters
                        and operation.parameters["text"]
                        else 0
                    ),
                    display_tokens=args.display_tokens or 0,
                    with_stats=not args.quiet,
                    tool_events_limit=args.display_tools_events,
                )

            elif operation.modality == Modality.VISION_IMAGE_CLASSIFICATION:
                console.print(theme.display_image_entity(cast(Any, output)))

            elif operation.modality == Modality.VISION_OBJECT_DETECTION:
                console.print(
                    theme.display_image_entities(
                        cast(list[Any], output), sort=True
                    )
                )

            elif operation.modality == Modality.VISION_SEMANTIC_SEGMENTATION:
                console.print(
                    theme.display_image_labels(cast(list[str], output))
                )

            elif operation.modality == Modality.VISION_TEXT_TO_IMAGE:
                console.print(output)

            elif operation.modality == Modality.VISION_TEXT_TO_ANIMATION:
                console.print(output)

            elif operation.modality == Modality.VISION_TEXT_TO_VIDEO:
                console.print(output)

            else:
                raise NotImplementedError(
                    f"Modality {operation.modality} not supported"
                )


async def model_search(
    args: Namespace,
    console: Console,
    theme: Theme,
    hub: Any,
    refresh_per_second: int,
) -> None:
    assert args.limit
    _ = theme._

    models: list[Model] = []
    model_access: dict[str, bool] = {}

    # Fetch matching models
    with console.status(
        _("Loading models..."),
        spinner=theme.get_spinner("downloading") or "dots",
        refresh_per_second=refresh_per_second,
    ):
        models = [
            model
            for model in hub.models(
                filter=args.filter or None,
                search=args.search or None,
                library=args.library or None,
                author=args.author,
                gated=True if args.gated else False if args.open else None,
                language=args.language or None,
                name=args.name or None,
                task=args.task or None,
                tags=args.tag or None,
                limit=args.limit,
            )
        ]

    # Tasks to check model access
    def _model_access_check(model_id: str) -> tuple[str, bool]:
        return model_id, hub.can_access(model_id)

    tasks = [
        create_task(to_thread(partial(_model_access_check, model.id)))
        for model in models
    ]

    def _render(
        models: list[Model], model_access: dict[str, bool]
    ) -> list[RenderableType]:
        return [
            theme.model(
                model,
                can_access=(
                    model_access[model.id]
                    if model.id in model_access
                    else None
                ),
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
    args: Namespace, console: Console, theme: Theme, hub: Any
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
    live_container: dict[str, Live | None] | None = None,
) -> None:
    # If no statistics needed, return as early as possible
    if not with_stats:
        async for token in _plain_stdout_projections(response):
            if token.channel != StreamChannel.ANSWER:
                continue
            text_token = stream_projection_text_delta(token)
            if text_token is None:
                continue
            console.print(text_token, end="")
            await sleep(0)
        return

    stop_signal = EventSignal()

    # From here on, display includes stats and may include token probabilities

    if not orchestrator or (
        not args.display_events and not args.display_tools
    ):
        try:
            with Live(
                refresh_per_second=refresh_per_second,
                screen=args.record,
                console=console,
            ) as live:
                if live_container is not None:
                    live_container["live"] = live
                await _token_stream(
                    args,
                    console,
                    live,
                    None,
                    None,
                    theme,
                    logger,
                    orchestrator,
                    event_stats,
                    lm,
                    input_string,
                    response,
                    display_tokens=display_tokens,
                    dtokens_pick=dtokens_pick,
                    refresh_per_second=refresh_per_second,
                    stop_signal=stop_signal,
                    tool_events_limit=tool_events_limit,
                    with_stats=with_stats,
                )
        finally:
            if live_container is not None:
                live_container["live"] = None
    else:
        events_height = 6
        tools_height = 10
        empty = ""
        group = Group(empty, empty, empty)
        events_group_index = 0
        tools_group_index = 1
        tokens_group_index = 2

        try:
            with Live(
                group,
                refresh_per_second=refresh_per_second,
                screen=args.record,
                console=console,
            ) as live:
                if live_container is not None:
                    live_container["live"] = live
                event_task = create_task(
                    _event_stream(
                        args,
                        console,
                        live,
                        group,
                        events_group_index,
                        tools_group_index,
                        orchestrator,
                        theme,
                        events_height=events_height,
                        tools_height=tools_height,
                        stop_signal=stop_signal,
                    )
                )
                token_task = create_task(
                    _token_stream(
                        args,
                        console,
                        live,
                        group,
                        tokens_group_index,
                        theme,
                        logger,
                        orchestrator,
                        event_stats,
                        lm,
                        input_string,
                        response,
                        display_tokens=display_tokens,
                        dtokens_pick=dtokens_pick,
                        refresh_per_second=refresh_per_second,
                        stop_signal=stop_signal,
                        tool_events_limit=tool_events_limit,
                        with_stats=with_stats,
                    )
                )
                token_error: BaseException | None = None
                try:
                    await token_task
                except BaseException as exc:
                    token_error = exc
                    raise
                finally:
                    stop_signal.set()
                    if not event_task.done():
                        event_task.cancel()
                    event_results = await gather(
                        event_task, return_exceptions=True
                    )
                    if token_error is None:
                        for result in event_results:
                            if isinstance(
                                result, BaseException
                            ) and not isinstance(result, CancelledError):
                                raise result
        finally:
            if live_container is not None:
                live_container["live"] = None


async def _event_stream(
    args: Namespace,
    console: Console,
    live: Live,
    group: Group,
    events_group_index: int,
    tools_group_index: int,
    orchestrator: Orchestrator,
    theme: Theme,
    *,
    events_height: int = 6,
    tools_height: int = 10,
    stop_signal: EventSignal,
) -> None:
    event_manager = orchestrator.event_manager
    if not event_manager or (
        not args.display_events and not args.display_tools
    ):
        return

    async for e in event_manager.listen(stop_signal=stop_signal):
        tool_view = e.type in TOOL_TYPES
        if (tool_view and not args.display_tools) or (
            not tool_view and not args.display_events
        ):
            continue

        events_renderable = theme.events(
            event_manager.history,
            events_limit=6 if tool_view else 4,
            height=tools_height if tool_view else events_height,
            include_tokens=False,
            include_tools=tool_view,
            include_tool_detect=False,
            include_non_tools=not tool_view,
            tool_view=tool_view,
        )
        if not events_renderable:
            continue

        _render_frame(
            args,
            console,
            live,
            events_renderable,
            group,
            tools_group_index if tool_view else events_group_index,
        )


async def _token_stream(
    args: Namespace,
    console: Console,
    live: Live,
    group: Group | None,
    tokens_group_index: int | None,
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
    stop_signal: EventSignal | None,
    tool_events_limit: int | None,
    with_stats: bool = True,
) -> None:
    display_time_to_n_token = args.display_time_to_n_token
    display_reasoning_time = not getattr(args, "skip_display_reasoning_time")
    display_pause = (
        args.display_pause
        if args.display_pause and args.display_pause > 0
        else 0
    )
    start_thinking = (
        args.start_thinking if hasattr(args, "start_thinking") else False
    )
    tokens: list[Token] = []
    answer_text_tokens: list[str] = []
    thinking_text_tokens: list[str] = []
    tool_text_tokens: list[str] = []
    tool_events: list[Event] = []
    tool_event_calls: list[Event] = []
    tool_event_results: list[Event] = []
    completed_call_ids: set[str] = set()
    total_tokens = 0
    tool_tokens = 0
    frame_minimum_pause_ms = (
        100 if display_pause > 0 and display_tokens > 0 else 0
    )

    def _input_token_count(input_value: Input) -> int | None:
        count = lm.input_token_count(input_value)
        return count or None

    def _tool_token_count(token_text: str) -> int:
        if not token_text:
            return 0
        return _input_token_count(token_text) or 1

    input_token_count = (
        response.input_token_count
        or (orchestrator.input_token_count if orchestrator else None)
        or _input_token_count(input_string)
    )
    display_input_token_count = input_token_count or 0
    assert lm.model_id is not None
    ttft: float | None = None
    ttnt: float | None = None
    last_current_dtoken: Token | None = None
    tool_running_spinner: Spinner | None = None

    if start_thinking and response.can_think and not response.is_thinking:
        response.set_thinking(start_thinking)

    start = perf_counter()
    started_reasoning = perf_counter() if response.is_thinking else None
    reasoning_time = None
    frame_renderer = _FrameRateRenderer(
        args,
        console,
        live,
        group,
        tokens_group_index,
        refresh_per_second=refresh_per_second,
    )

    try:
        async for render_item in _stream_render_items(
            response,
            stream_session_id="cli-render-stream",
            run_id="cli-render-run",
            turn_id="cli-render-turn",
        ):
            is_event = False
            source_token = render_item.source_token
            if render_item.event is not None:
                token: Event | StreamConsumerProjection = render_item.event
            else:
                assert render_item.projection is not None
                token = render_item.projection
            is_reasoning_token = _is_reasoning_stream_item(token)

            if isinstance(token, Event):
                is_event = True
                event = token
                tool_events.append(event)
                if event.type == EventType.TOOL_MODEL_RESPONSE:
                    tokens = []
                    answer_text_tokens = []
                    tool_text_tokens = []
                    thinking_text_tokens = []
                    assert event.payload is not None
                    inner_response = event.payload["response"]
                    assert isinstance(inner_response, TextGenerationResponse)
                    next_input_token_count = (
                        inner_response.input_token_count
                        or cast(
                            int | None,
                            event.payload.get("input_token_count"),
                        )
                        or (
                            _input_token_count(
                                cast(Input, event.payload["messages"])
                            )
                            if "messages" in event.payload
                            else None
                        )
                    )
                    if next_input_token_count:
                        display_input_token_count += next_input_token_count
                        input_token_count = next_input_token_count
                elif event.type in (
                    EventType.TOOL_DIAGNOSTIC,
                    EventType.TOOL_RESULT,
                ):
                    tool_event_results.append(event)
                    if event.payload and "call" in event.payload:
                        completed_call_ids.add(event.payload["call"].id)
                else:
                    tool_event_calls.append(event)
            else:
                if (
                    display_reasoning_time
                    and not is_reasoning_token
                    and started_reasoning is not None
                ):
                    reasoning_time = perf_counter() - started_reasoning
                    started_reasoning = None

                text_token = _stream_text(token)
                if text_token is None:
                    continue
                if _is_tool_call_stream_item(token):
                    tool_text_tokens.append(text_token)
                    tool_tokens += _tool_token_count(text_token)
                elif is_reasoning_token:
                    if not started_reasoning:
                        started_reasoning = perf_counter()
                    thinking_text_tokens.append(text_token)
                else:
                    answer_text_tokens.append(text_token)

            tool_running_spinner = None
            if tool_event_calls or tool_event_results:
                tool_calling_names = [
                    str(getattr(c, "name", ""))
                    for e in tool_event_calls
                    for c in cast(list[object], e.payload or [])
                    if str(getattr(c, "id", "")) not in completed_call_ids
                ]
                if tool_calling_names:
                    tool_running_spinner = Spinner(
                        theme.get_spinner("tool_running") or "dots",
                        text="[cyan]"
                        + theme._n(
                            "Running tool {tool_names}...",
                            "Running tools {tool_names}...",
                            len(tool_calling_names),
                        ).format(tool_names=", ".join(tool_calling_names))
                        + "[/cyan]",
                        style="cyan",
                        speed=1.0,
                    )

            elapsed = perf_counter() - start
            is_tool_call_token = _is_tool_call_stream_item(token)
            if not is_event and not is_tool_call_token:
                total_tokens += 1

            if not is_event and not is_tool_call_token and ttft is None:
                ttft = elapsed
            if (
                not is_event
                and not is_tool_call_token
                and ttnt is None
                and display_time_to_n_token
                and total_tokens >= display_time_to_n_token
            ):
                ttnt = elapsed

            ttsr = None
            if display_reasoning_time and reasoning_time:
                ttsr = reasoning_time

            if (
                display_tokens
                and source_token is not None
                and not isinstance(source_token, ToolCallToken)
            ):
                tokens.append(source_token)
            limit_answer_height = not getattr(
                args, "display_answer_height_expand", False
            )
            answer_height = getattr(args, "display_answer_height", 12)

            token_frames_result = theme.tokens(
                lm.model_id,
                lm.tokenizer_config.tokens if lm.tokenizer_config else None,
                (
                    lm.tokenizer_config.special_tokens
                    if lm.tokenizer_config
                    else None
                ),
                display_tokens,
                args.display_probabilities if dtokens_pick > 0 else False,
                dtokens_pick,
                # Which tokens to mark as interesting
                lambda dtoken: (
                    (
                        dtoken.probability is not None
                        and dtoken.probability
                        < args.display_probabilities_maximum
                        or len(
                            [
                                t
                                for t in cast(
                                    list[Token],
                                    getattr(dtoken, "tokens", []) or [],
                                )
                                if t.id != dtoken.id
                                and t.probability is not None
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
                    else False
                ),
                thinking_text_tokens,
                tool_text_tokens,
                answer_text_tokens,
                tokens or None,
                display_input_token_count,
                total_tokens,
                tool_events,
                tool_event_calls,
                tool_event_results,
                cast(Any, tool_running_spinner),
                ttft,
                ttnt,
                ttsr,
                elapsed,
                console.width,
                logger,
                event_stats,
                tool_token_count=tool_tokens,
                height=answer_height,
                tool_events_limit=tool_events_limit,
                limit_answer_height=limit_answer_height,
                maximum_frames=1,
                start_thinking=start_thinking,
            )

            token_frames_stream: AsyncGenerator[
                tuple[Token | None, RenderableType], None
            ]
            if isinstance(token_frames_result, Awaitable):
                token_frames_stream = await cast(
                    Awaitable[
                        AsyncGenerator[
                            tuple[Token | None, RenderableType], None
                        ]
                    ],
                    token_frames_result,
                )
            else:
                token_frames_stream = token_frames_result

            token_frame_list = [
                token_frame async for token_frame in token_frames_stream
            ]

            token_frames = [token_frame_list[0]]

            for current_dtoken, frame in token_frames:
                frame_renderer.mark_dirty(frame)

                if current_dtoken and current_dtoken != last_current_dtoken:
                    last_current_dtoken = current_dtoken
                    if display_pause > 0:
                        await sleep(display_pause / 1000)
                    elif frame_minimum_pause_ms > 0:
                        await sleep(
                            frame_minimum_pause_ms / 1000
                        )  # pragma: no cover - unreachable
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
                    frame_renderer.mark_dirty(frame)

                    if current_dtoken and display_pause > 0:
                        await sleep(display_pause / 1000)
                    elif frame_minimum_pause_ms > 0:
                        await sleep(frame_minimum_pause_ms / 1000)
            await sleep(0)
    except (CancelledError, KeyboardInterrupt):
        raise
    finally:
        await frame_renderer.close()
        if stop_signal:
            stop_signal.set()


async def _plain_stdout_projections(
    response: TextGenerationResponse | AsyncIterator[Any],
) -> AsyncIterator[StreamConsumerProjection]:
    consumer_projections = getattr(response, "consumer_projections", None)
    if callable(consumer_projections):
        accumulator = CanonicalStreamAccumulator()
        async for projection in consumer_projections(
            stream_session_id="cli-stdout-stream",
            run_id="cli-stdout-run",
            turn_id="cli-stdout-turn",
        ):
            if not isinstance(projection, StreamConsumerProjection):
                raise StreamValidationError(
                    "consumer projection stream item must be "
                    "StreamConsumerProjection"
                )
            accumulator.add(
                canonical_item_from_consumer_projection(projection)
            )
            yield projection
        accumulator.validate_complete()
        return

    async for item in _stream_render_items(
        response,
        stream_session_id="cli-stdout-stream",
        run_id="cli-stdout-run",
        turn_id="cli-stdout-turn",
    ):
        if item.event is not None:
            continue
        assert item.projection is not None
        yield item.projection


async def _stream_render_items(
    response: TextGenerationResponse | AsyncIterator[Any],
    *,
    stream_session_id: str,
    run_id: str,
    turn_id: str,
) -> AsyncIterator[_StreamRenderItem]:
    accumulator: CanonicalStreamAccumulator | None = None
    sequence = 0
    legacy_stream_seen = False

    async for token in response:
        if isinstance(token, Event):
            yield _StreamRenderItem(event=token)
            continue

        if isinstance(token, CanonicalStreamItem):
            if legacy_stream_seen:
                raise StreamValidationError(
                    "canonical stream item after legacy stream item"
                )
            if accumulator is None:
                accumulator = CanonicalStreamAccumulator()
            accumulator.add(token)
            yield _StreamRenderItem(
                projection=project_canonical_stream_item(token)
            )
            continue

        if isinstance(token, StreamConsumerProjection):
            if legacy_stream_seen:
                raise StreamValidationError(
                    "canonical stream item after legacy stream item"
                )
            if accumulator is None:
                accumulator = CanonicalStreamAccumulator()
            accumulator.add(canonical_item_from_consumer_projection(token))
            yield _StreamRenderItem(projection=token)
            continue

        if accumulator is not None:
            raise StreamValidationError(
                "legacy stream item after canonical stream item"
            )

        source_token = token if isinstance(token, Token) else None
        yield _StreamRenderItem(
            projection=stream_consumer_projection_from_token(
                token,
                sequence,
                stream_session_id=stream_session_id,
                run_id=run_id,
                turn_id=turn_id,
            ),
            source_token=source_token,
        )
        legacy_stream_seen = True
        sequence += 1

    if accumulator is not None:
        accumulator.validate_complete()


def _stream_text(
    token: StreamConsumerProjection,
) -> str | None:
    assert isinstance(token, StreamConsumerProjection)
    return stream_projection_text_delta(token)


def _is_reasoning_stream_item(token: object) -> bool:
    if not isinstance(token, StreamConsumerProjection):
        return False
    return stream_projection_is_reasoning(token)


def _is_tool_call_stream_item(token: object) -> bool:
    if not isinstance(token, StreamConsumerProjection):
        return False
    return stream_projection_is_tool_call(token)


def _stream_projection(
    token: CanonicalStreamItem | StreamConsumerProjection | Token | str,
) -> StreamConsumerProjection:
    return stream_consumer_projection_from_token(token, 0)


def _render_frame(
    args: Namespace,
    console: Console,
    live: Live,
    frame: RenderableType,
    group: Group | None = None,
    group_index: int | None = None,
) -> None:
    if group and group_index is not None:
        group.renderables[group_index] = frame
        live.refresh()
    else:
        live.update(frame)

    if args.record:
        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y%m%d%H%M%S")
        ms = now.microsecond // 1000
        filename = f"avalan-screenshot-{ts}-{ms:03d}.svg"
        console.save_svg(filename, clear=True)
