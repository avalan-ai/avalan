from ...agent import Specification
from ...agent.orchestrator import Orchestrator
from ...cli import confirm, get_input, has_input
from ...cli.commands.cache import cache_delete, cache_download
from ...cli.theme import (
    Theme,
    TokenRenderDisplayToken,
    TokenRenderDisplayTokenCandidate,
    TokenRenderFrame,
    TokenRenderState,
)
from ...entities import (
    GenerationSettings,  # noqa: F401
    Input,
    Message,
    Modality,
    Model,
)
from ...event import TOOL_TYPES, Event, EventStats
from ...model.call import ModelCall, ModelCallContext
from ...model.criteria import KeywordStoppingCriteria  # noqa: F401
from ...model.input import input_files
from ...model.manager import ModelManager
from ...model.nlp.text.generation import TextGenerationModel
from ...model.response.text import TextGenerationResponse
from ...model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamConsumerProjection,
    StreamItemKind,
    project_stream_consumer_item,
    stream_consumer_iterator,
    stream_projection_is_reasoning,
    stream_projection_is_tool_call,
    stream_projection_text_delta,
)
from ...secrets import KeyringSecrets
from . import ModelSettings, get_model_settings, is_ds4_backend_selected

from argparse import Namespace
from asyncio import (
    CancelledError,
    Lock,
    as_completed,
    create_task,
    gather,
    sleep,
    to_thread,
)
from asyncio import (
    Event as EventSignal,
)
from asyncio import (
    run as asyncio_run,
)
from collections.abc import AsyncIterable, Mapping
from dataclasses import replace
from datetime import datetime, timezone
from functools import partial
from logging import Logger
from time import perf_counter
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Generic,
    TypeAlias,
    TypeVar,
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

_HAS_INPUT = has_input
_T = TypeVar("_T")


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
        render_lock: Lock | None = None,
    ) -> None:
        assert refresh_per_second > 0
        self._args = args
        self._console = console
        self._live = live
        self._group = group
        self._group_index = group_index
        self._interval = 1 / refresh_per_second
        self._render_lock = render_lock
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
            await self._render(frame)
            self._rendered_version = version

            if (
                self._stopped
                and self._rendered_version == self._latest_version
            ):
                return
            await sleep(self._interval)

    async def _render(self, frame: RenderableType) -> None:
        if self._render_lock is None:
            await self._render_unlocked(frame)
            return
        async with self._render_lock:
            await self._render_unlocked(frame)

    async def _render_unlocked(self, frame: RenderableType) -> None:
        await to_thread(
            _render_frame,
            self._args,
            self._console,
            self._live,
            frame,
            self._group,
            self._group_index,
        )


class _LatestTokenFrameBuilder:
    def __init__(
        self,
        args: Namespace,
        console: Console,
        theme: Theme,
        logger: Logger,
        frame_renderer: _FrameRateRenderer,
        *,
        refresh_per_second: int,
        display_pause: int,
        frame_minimum_pause_ms: int,
        tool_events_limit: int | None,
        height: int,
        limit_answer_height: bool,
        start_thinking: bool,
    ) -> None:
        assert refresh_per_second > 0
        assert display_pause >= 0
        assert frame_minimum_pause_ms >= 0
        self._args = args
        self._console = console
        self._theme = theme
        self._logger = logger
        self._frame_renderer = frame_renderer
        self._interval = 1 / refresh_per_second
        self._display_pause = display_pause
        self._frame_minimum_pause_ms = frame_minimum_pause_ms
        self._tool_events_limit = tool_events_limit
        self._height = height
        self._limit_answer_height = limit_answer_height
        self._start_thinking = start_thinking
        self._dirty = EventSignal()
        self._latest_state: TokenRenderState | None = None
        self._latest_version = 0
        self._built_version = 0
        self._stopped = False
        self._last_current_dtoken: TokenRenderDisplayToken | None = None
        self._task = create_task(self._run())

    def mark_dirty(self, state: TokenRenderState) -> None:
        self._latest_state = state
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
            if self._latest_version == self._built_version:
                if self._stopped:
                    return
                continue

            if not self._stopped:
                await sleep(self._interval)
                self._dirty.clear()

            state = self._latest_state
            assert state is not None
            version = self._latest_version
            await self._build_and_render(state, version=version)
            self._built_version = version

            if self._stopped and self._built_version == self._latest_version:
                return

    async def _build_and_render(
        self, state: TokenRenderState, *, version: int
    ) -> None:
        token_frames = await to_thread(
            _collect_token_frames,
            self._theme,
            state,
            console_width=self._console.width,
            logger=self._logger,
            height=self._height,
            tool_events_limit=self._tool_events_limit,
            limit_answer_height=self._limit_answer_height,
            maximum_frames=1,
            start_thinking=self._start_thinking,
        )
        if not token_frames:
            await sleep(0)
            return

        for current_dtoken, frame in token_frames:
            if self._latest_version != version:
                break
            self._frame_renderer.mark_dirty(frame)
            await self._pause_after_frame(state, current_dtoken)
        await sleep(0)

    async def _pause_after_frame(
        self,
        state: TokenRenderState,
        current_dtoken: TokenRenderDisplayToken | None,
    ) -> None:
        if current_dtoken and current_dtoken != self._last_current_dtoken:
            self._last_current_dtoken = current_dtoken
            if self._display_pause > 0:
                await sleep(self._display_pause / 1000)
            elif self._frame_minimum_pause_ms > 0:
                await sleep(
                    self._frame_minimum_pause_ms / 1000
                )  # pragma: no cover - unreachable
        elif (
            state.pick > 0
            and not state.display_probabilities
            and self._display_pause > 0
        ):
            await sleep(self._display_pause / 1000)
        elif (
            state.pick > 0
            and state.display_probabilities
            and self._frame_minimum_pause_ms > 0
        ):
            await sleep(self._frame_minimum_pause_ms / 1000)


class _TokenTupleSnapshot(Generic[_T]):
    def __init__(self) -> None:
        self._length = -1
        self._tokens: tuple[_T, ...] = ()

    def get(self, source: list[_T]) -> tuple[_T, ...]:
        assert isinstance(source, list)
        if len(source) != self._length:
            self._tokens = tuple(source)
            self._length = len(source)
        return self._tokens


class _TokenRenderSnapshotCache:
    def __init__(
        self,
        *,
        added_tokens: Any,
        special_tokens: Any,
    ) -> None:
        self._added_token_source = added_tokens
        self._special_token_source = special_tokens
        self._added_tokens_built = False
        self._special_tokens_built = False
        self._added_tokens: tuple[str, ...] | None = None
        self._special_tokens: tuple[str, ...] | None = None
        self._reasoning_text_tokens = _TokenTupleSnapshot[str]()
        self._tool_text_tokens = _TokenTupleSnapshot[str]()
        self._answer_text_tokens = _TokenTupleSnapshot[str]()
        self._display_tokens = _TokenTupleSnapshot[TokenRenderDisplayToken]()

    def added_tokens(self) -> tuple[str, ...] | None:
        if not self._added_tokens_built:
            self._added_tokens = self._optional_token_tuple(
                self._added_token_source
            )
            self._added_tokens_built = True
        return self._added_tokens

    def special_tokens(self) -> tuple[str, ...] | None:
        if not self._special_tokens_built:
            self._special_tokens = self._optional_token_tuple(
                self._special_token_source
            )
            self._special_tokens_built = True
        return self._special_tokens

    def reasoning_text_tokens(self, source: list[str]) -> tuple[str, ...]:
        return self._reasoning_text_tokens.get(source)

    def tool_text_tokens(self, source: list[str]) -> tuple[str, ...]:
        return self._tool_text_tokens.get(source)

    def answer_text_tokens(self, source: list[str]) -> tuple[str, ...]:
        return self._answer_text_tokens.get(source)

    def display_tokens(
        self, source: list[TokenRenderDisplayToken]
    ) -> tuple[TokenRenderDisplayToken, ...]:
        return self._display_tokens.get(source)

    @staticmethod
    def _optional_token_tuple(tokens: Any) -> tuple[str, ...] | None:
        if not tokens:
            return None
        return cast(tuple[str, ...], tuple(tokens))


def _collect_token_frames(
    theme: Theme,
    state: TokenRenderState,
    *,
    console_width: int,
    logger: Logger,
    maximum_frames: int | None = None,
    tool_events_limit: int | None = None,
    height: int = 12,
    limit_answer_height: bool = False,
    start_thinking: bool = False,
) -> tuple[TokenRenderFrame, ...]:
    frames = theme.token_frames(
        state,
        console_width=console_width,
        logger=logger,
        height=height,
        tool_events_limit=tool_events_limit,
        limit_answer_height=limit_answer_height,
        maximum_frames=maximum_frames,
        start_thinking=start_thinking,
    )
    if isinstance(frames, tuple):
        return frames
    if isinstance(frames, AsyncIterable):
        return asyncio_run(_collect_async_token_frames(frames))
    return tuple(frames)


async def _collect_async_token_frames(
    frames: AsyncIterable[TokenRenderFrame],
) -> tuple[TokenRenderFrame, ...]:
    return tuple([frame async for frame in frames])


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
        async for projection in _plain_stdout_projections(response):
            if projection.kind is not StreamItemKind.ANSWER_DELTA:
                continue
            text_token = stream_projection_text_delta(projection)
            assert text_token is not None
            console.print(text_token, end="")
            await sleep(0)
        return

    stop_signal = EventSignal()
    render_lock = Lock()

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
                    render_lock=render_lock,
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
                        refresh_per_second=refresh_per_second,
                        render_lock=render_lock,
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
                        render_lock=render_lock,
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
    refresh_per_second: int = 12,
    render_lock: Lock | None = None,
) -> None:
    event_manager = orchestrator.event_manager
    if not event_manager or (
        not args.display_events and not args.display_tools
    ):
        return

    events_renderer = (
        _FrameRateRenderer(
            args,
            console,
            live,
            group,
            events_group_index,
            refresh_per_second=refresh_per_second,
            render_lock=render_lock,
        )
        if args.display_events
        else None
    )
    tools_renderer = (
        _FrameRateRenderer(
            args,
            console,
            live,
            group,
            tools_group_index,
            refresh_per_second=refresh_per_second,
            render_lock=render_lock,
        )
        if args.display_tools
        else None
    )
    try:
        async for e in event_manager.listen(stop_signal=stop_signal):
            tool_view = _event_targets_tool_panel(e)
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

            renderer = tools_renderer if tool_view else events_renderer
            assert renderer is not None
            renderer.mark_dirty(events_renderable)
    finally:
        close_tasks = [
            renderer.close()
            for renderer in (events_renderer, tools_renderer)
            if renderer is not None
        ]
        if close_tasks:
            await gather(*close_tasks)


_CANONICAL_TOOL_EVENT_CHANNELS = frozenset(
    {
        "tool_call",
        "tool_execution",
        "tool.call",
        "tool.execution",
    }
)
_CANONICAL_TOOL_EVENT_KIND_PREFIXES = (
    "tool.call.",
    "tool_call.",
    "tool.execution.",
    "tool_execution.",
    "model.continuation.",
    "model_continuation.",
)


def _event_targets_tool_panel(event: Event) -> bool:
    canonical_payload = _canonical_event_payload(event)
    if canonical_payload is None:
        return event.type in TOOL_TYPES
    channel = canonical_payload.get("channel")
    if channel in _CANONICAL_TOOL_EVENT_CHANNELS:
        return True
    kind = canonical_payload.get("kind")
    return isinstance(kind, str) and kind.startswith(
        _CANONICAL_TOOL_EVENT_KIND_PREFIXES
    )


def _canonical_event_payload(event: Event) -> Mapping[str, Any] | None:
    payload = event.payload
    if not isinstance(payload, Mapping):
        return None
    if not all(
        isinstance(payload.get(key), str)
        for key in ("stream_session_id", "run_id", "turn_id", "kind")
    ):
        return None
    if not isinstance(payload.get("channel"), str):
        return None
    return cast(Mapping[str, Any], payload)


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
    render_lock: Lock | None = None,
) -> None:
    display_time_to_n_token = args.display_time_to_n_token
    display_reasoning = getattr(args, "display_reasoning", False)
    display_reasoning_time = not getattr(
        args, "skip_display_reasoning_time", False
    )
    display_pause = (
        args.display_pause
        if args.display_pause and args.display_pause > 0
        else 0
    )
    start_thinking = (
        args.start_thinking if hasattr(args, "start_thinking") else False
    )
    collect_display_tokens = display_tokens > 0 or dtokens_pick > 0
    display_token_details: list[TokenRenderDisplayToken] = []
    answer_text_tokens: list[str] = []
    thinking_text_tokens: list[str] = []
    tool_text_tokens: list[str] = []
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

    if start_thinking and response.can_think and not response.is_thinking:
        response.set_thinking(start_thinking)

    tokenizer_tokens = None
    tokenizer_special_tokens = None
    if collect_display_tokens:
        tokenizer_config = lm.tokenizer_config
        if tokenizer_config:
            tokenizer_tokens = tokenizer_config.tokens
            tokenizer_special_tokens = tokenizer_config.special_tokens
    render_snapshots = _TokenRenderSnapshotCache(
        added_tokens=tokenizer_tokens,
        special_tokens=tokenizer_special_tokens,
    )

    start = perf_counter()
    started_reasoning = perf_counter() if response.is_thinking else None
    reasoning_time = None
    limit_answer_height = not getattr(
        args, "display_answer_height_expand", False
    )
    answer_height = getattr(args, "display_answer_height", 12)
    frame_renderer = _FrameRateRenderer(
        args,
        console,
        live,
        group,
        tokens_group_index,
        refresh_per_second=refresh_per_second,
        render_lock=render_lock,
    )
    frame_builder = _LatestTokenFrameBuilder(
        args,
        console,
        theme,
        logger,
        frame_renderer,
        refresh_per_second=refresh_per_second,
        display_pause=display_pause,
        frame_minimum_pause_ms=frame_minimum_pause_ms,
        tool_events_limit=tool_events_limit,
        height=answer_height,
        limit_answer_height=limit_answer_height,
        start_thinking=start_thinking,
    )

    def _focus_on_display_token(
        dtoken: TokenRenderDisplayToken,
    ) -> bool:
        if (
            not display_tokens
            or not args.display_probabilities
            or args.display_probabilities_maximum <= 0
        ):
            return False
        low_probability = (
            dtoken.probability is not None
            and dtoken.probability < args.display_probabilities_maximum
        )
        sampled_alternative = any(
            candidate.id != dtoken.id
            and candidate.probability is not None
            and candidate.probability
            >= args.display_probabilities_sample_minimum
            for candidate in dtoken.tokens
        )
        return low_probability or sampled_alternative

    try:
        async for projection in _stream_render_projections(
            response,
            stream_session_id="cli-render-stream",
            run_id="cli-render-run",
            turn_id="cli-render-turn",
        ):
            is_reasoning_token = _is_reasoning_stream_item(projection)
            if (
                display_reasoning_time
                and not is_reasoning_token
                and started_reasoning is not None
            ):
                reasoning_time = perf_counter() - started_reasoning
                started_reasoning = None

            text_token = _stream_text(projection)
            if text_token is None:
                continue
            is_tool_token = _is_tool_stream_item(projection)
            if is_tool_token:
                tool_text_tokens.append(text_token)
                tool_tokens += _tool_token_count(text_token)
            elif is_reasoning_token:
                if not started_reasoning:
                    started_reasoning = perf_counter()
                thinking_text_tokens.append(text_token)
            else:
                assert projection.kind is StreamItemKind.ANSWER_DELTA
                answer_text_tokens.append(text_token)

            elapsed = perf_counter() - start
            if not is_tool_token:
                total_tokens += 1

            if not is_tool_token and ttft is None:
                ttft = elapsed
            if (
                not is_tool_token
                and ttnt is None
                and display_time_to_n_token
                and total_tokens >= display_time_to_n_token
            ):
                ttnt = elapsed

            ttsr = None
            if display_reasoning_time and reasoning_time:
                ttsr = reasoning_time

            display_token = (
                _projection_display_token(projection)
                if collect_display_tokens
                else None
            )
            if (
                display_tokens
                and display_token is not None
                and (
                    projection.kind is StreamItemKind.ANSWER_DELTA
                    or (
                        display_reasoning
                        and projection.kind is StreamItemKind.REASONING_DELTA
                    )
                )
            ):
                display_token_details.append(display_token)

            render_state = TokenRenderState(
                model_id=lm.model_id,
                projection_sequence=projection.sequence,
                projection_kind=projection.kind,
                projection_channel=projection.channel,
                added_tokens=render_snapshots.added_tokens(),
                special_tokens=render_snapshots.special_tokens(),
                display_token_size=display_tokens or None,
                display_probabilities=(
                    args.display_probabilities if dtokens_pick > 0 else False
                ),
                pick=dtokens_pick,
                focus_on_token_when=_focus_on_display_token,
                reasoning_text_tokens=(
                    render_snapshots.reasoning_text_tokens(
                        thinking_text_tokens
                    )
                ),
                tool_text_tokens=render_snapshots.tool_text_tokens(
                    tool_text_tokens
                ),
                answer_text_tokens=render_snapshots.answer_text_tokens(
                    answer_text_tokens
                ),
                display_tokens=render_snapshots.display_tokens(
                    display_token_details
                ),
                display_reasoning=display_reasoning,
                display_tools=getattr(args, "display_tools", False),
                input_token_count=display_input_token_count,
                total_tokens=total_tokens,
                tool_token_count=tool_tokens,
                ttft=ttft,
                ttnt=ttnt,
                ttsr=ttsr,
                elapsed=elapsed,
                event_stats=event_stats,
                start_thinking=start_thinking,
            )
            frame_builder.mark_dirty(render_state)
            await sleep(0)
    except (CancelledError, KeyboardInterrupt):
        raise
    finally:
        try:
            await frame_builder.close()
        finally:
            await frame_renderer.close()
            if stop_signal:
                stop_signal.set()


async def _plain_stdout_projections(
    response: TextGenerationResponse | AsyncIterator[Any],
) -> AsyncIterator[StreamConsumerProjection]:
    async for projection in stream_consumer_iterator(
        response,
        stream_session_id="cli-stdout-stream",
        run_id="cli-stdout-run",
        turn_id="cli-stdout-turn",
        unsupported_message="unsupported CLI stream item",
    ):
        yield projection


async def _stream_render_projections(
    response: TextGenerationResponse | AsyncIterator[Any],
    *,
    stream_session_id: str,
    run_id: str,
    turn_id: str,
) -> AsyncIterator[StreamConsumerProjection]:
    async for projection in stream_consumer_iterator(
        response,
        stream_session_id=stream_session_id,
        run_id=run_id,
        turn_id=turn_id,
        unsupported_message="unsupported CLI stream item",
    ):
        yield projection


def _projection_display_token(
    projection: StreamConsumerProjection,
) -> TokenRenderDisplayToken | None:
    assert isinstance(projection, StreamConsumerProjection)
    if projection.kind not in {
        StreamItemKind.ANSWER_DELTA,
        StreamItemKind.REASONING_DELTA,
    }:
        return None

    text = stream_projection_text_delta(projection)
    assert text is not None

    metadata = projection.metadata
    if not any(
        key in metadata
        for key in (
            "token_id",
            "probability",
            "step",
            "probability_distribution",
            "tokens",
        )
    ):
        return None

    token_id = metadata.get("token_id")
    probability = metadata.get("probability")
    step = metadata.get("step")
    probability_distribution = metadata.get("probability_distribution")
    return TokenRenderDisplayToken(
        sequence=projection.sequence,
        kind=projection.kind,
        channel=projection.channel,
        token=text,
        id=token_id if isinstance(token_id, int) else None,
        probability=(
            float(probability)
            if isinstance(probability, int | float)
            else None
        ),
        step=step if isinstance(step, int) else None,
        probability_distribution=(
            probability_distribution
            if isinstance(probability_distribution, str)
            else None
        ),
        tokens=_projection_display_token_candidates(metadata.get("tokens")),
    )


def _projection_display_token_candidates(
    value: object,
) -> tuple[TokenRenderDisplayTokenCandidate, ...]:
    if not isinstance(value, list):
        return ()
    candidates: list[TokenRenderDisplayTokenCandidate] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        text = item.get("token")
        if not isinstance(text, str):
            continue
        token_id = item.get("token_id")
        probability = item.get("probability")
        candidates.append(
            TokenRenderDisplayTokenCandidate(
                token=text,
                id=token_id if isinstance(token_id, int) else None,
                probability=(
                    float(probability)
                    if isinstance(probability, int | float)
                    else None
                ),
            )
        )
    return tuple(candidates)


def _stream_text(
    token: StreamConsumerProjection,
) -> str | None:
    assert isinstance(token, StreamConsumerProjection)
    return stream_projection_text_delta(token)


def _is_reasoning_stream_item(token: object) -> bool:
    if not isinstance(token, StreamConsumerProjection):
        return False
    return stream_projection_is_reasoning(token)


def _is_tool_stream_item(token: object) -> bool:
    if not isinstance(token, StreamConsumerProjection):
        return False
    return token.channel in {
        StreamChannel.TOOL_CALL,
        StreamChannel.TOOL_EXECUTION,
    } or stream_projection_is_tool_call(token)


def _stream_projection(
    token: CanonicalStreamItem | StreamConsumerProjection,
) -> StreamConsumerProjection:
    return project_stream_consumer_item(
        token,
        0,
        stream_session_id="cli-helper-stream",
        run_id="cli-helper-run",
        turn_id="cli-helper-turn",
        unsupported_message="unsupported CLI stream item",
    )


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
