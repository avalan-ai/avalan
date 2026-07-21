from ...agent import Specification
from ...agent.orchestrator import Orchestrator
from ...cli import confirm, get_input, has_input
from ...cli.commands.cache import cache_delete, cache_download
from ...cli.display import CliStreamDisplayConfig, cli_stream_display_config
from ...cli.display_reducer import (
    CliSideChannelEvent,
    CliStreamSnapshotReducer,
)
from ...cli.stream_coordinator import CliStreamCoordinator
from ...cli.stream_presenter import (
    CliStreamPresenterContext,
    CliStreamPresenterRequest,
    StreamPresenterMode,
)
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
from ...event import TOOL_TYPES, EventStats, EventType
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
    StreamItemCorrelation,
    StreamItemKind,
    StreamTerminalOutcome,
    StreamValidationError,
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
    FIRST_COMPLETED,
    CancelledError,
    Lock,
    Task,
    as_completed,
    create_task,
    current_task,
    gather,
    sleep,
    to_thread,
    wait,
)
from asyncio import (
    Event as EventSignal,
)
from asyncio import (
    run as asyncio_run,
)
from collections.abc import (
    AsyncIterable,
    Callable,
    Iterable,
    Mapping,
)
from dataclasses import dataclass, field, replace
from functools import partial
from logging import Logger, getLogger
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


@dataclass(slots=True)
class _CliStreamPresentationGate:
    """Gate snapshot presentation to refresh interval ticks."""

    refresh_per_second: int
    clock: Callable[[], float] = field(default_factory=lambda: perf_counter)
    last_presented_at: float | None = None
    last_checked_at: float | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.refresh_per_second, int)
        assert self.refresh_per_second > 0
        assert callable(self.clock)

    def should_present(self, *, force: bool = False) -> bool:
        """Return whether presentation is due now."""
        now = self.clock()
        self.last_checked_at = now
        if force or self.last_presented_at is None:
            self.last_presented_at = now
            return True
        if now - self.last_presented_at < 1 / self.refresh_per_second:
            return False
        self.last_presented_at = now
        return True

    def delay_until_next_tick(self) -> float:
        """Return seconds until the next presentation tick."""
        if self.last_presented_at is None:
            return 0.0
        now = (
            self.last_checked_at
            if self.last_checked_at is not None
            else self.clock()
        )
        interval = 1 / self.refresh_per_second
        elapsed = now - self.last_presented_at
        return max(0.0, interval - elapsed)


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
    display_config = cli_stream_display_config(
        args,
        refresh_per_second=refresh_per_second,
        interactive=bool(getattr(console, "is_terminal", True)),
    )

    with ModelManager(hub, logger) as manager:
        engine_uri = manager.parse_uri(args.model)
        model_settings: ModelSettings = get_model_settings(
            args, hub, logger, engine_uri
        )
        modality = model_settings["modality"]

        if not display_config.answer_stdout_only:
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
                    is_quiet=display_config.answer_stdout_only,
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
                    refresh_per_second=display_config.refresh_per_second,
                    response=cast(TextGenerationResponse, output),
                    dtokens_pick=(
                        operation.parameters["text"].pick_tokens or 0
                        if operation.parameters
                        and operation.parameters["text"]
                        else 0
                    ),
                    display_tokens=display_config.display_tokens,
                    with_stats=display_config.show_stats,
                    tool_events_limit=display_config.display_tools_events,
                    display_config=display_config,
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
    coordinator_container: (
        dict[str, CliStreamCoordinator | None] | None
    ) = None,
    display_config: CliStreamDisplayConfig | None = None,
    answer_prefix: str | None = None,
) -> None:
    assert answer_prefix is None or isinstance(answer_prefix, str)
    display_config = display_config or _legacy_stream_display_config(
        args,
        display_tokens=display_tokens,
        refresh_per_second=refresh_per_second,
        tool_events_limit=tool_events_limit,
        with_stats=with_stats,
    )
    display_config = _theme_default_stream_display_config(
        theme,
        display_config,
        orchestrator,
    )
    stop_signal = EventSignal()
    render_lock = Lock()
    reducer = CliStreamSnapshotReducer(display_config)
    stream_logger = (
        logger if isinstance(logger, Logger) else getLogger(__name__)
    )
    stream_presenter_factory = getattr(type(theme), "stream_presenter", None)
    if not callable(stream_presenter_factory):
        stream_presenter_factory = Theme.stream_presenter
    if answer_prefix is None:
        presenter = stream_presenter_factory(
            theme,
            stream_logger,
            event_stats=event_stats,
        )
    else:
        presenter = stream_presenter_factory(
            theme,
            stream_logger,
            event_stats=event_stats,
            answer_prefix=answer_prefix,
        )
    presenter_context = _stream_presenter_context(
        args,
        console,
        orchestrator,
        lm,
        input_string,
        response,
        display_config=display_config,
        dtokens_pick=dtokens_pick,
    )
    diagnostics_enabled = (
        display_config.diagnostic_channel != "none"
        and (
            display_config.show_stats
            or display_config.show_tools
            or display_config.show_events
            or display_config.show_reasoning
        )
        and (
            display_config.live_enabled
            or _stream_presenter_supports_stderr_diagnostics(presenter)
        )
    )
    presenter_mode: StreamPresenterMode = (
        "live" if diagnostics_enabled else "answer"
    )
    event_listen: Any = None
    if (
        diagnostics_enabled
        and orchestrator is not None
        and (display_config.show_events or display_config.show_tools)
    ):
        event_manager = getattr(orchestrator, "event_manager", None)
        listen = getattr(event_manager, "listen", None)
        if callable(listen):
            event_listen = listen
    coordinator = CliStreamCoordinator(console, display_config)
    side_channel_events_enabled = True
    snapshot_revision = 0
    presented_snapshot_revision = -1
    last_projection_sequence: int | None = None
    presentation_gate = _CliStreamPresentationGate(
        display_config.refresh_per_second
    )
    presentation_retry_task: Task[None] | None = None
    presentation_retry_failure: BaseException | None = None
    supervisor_task = current_task()
    basic_pre_answer_diagnostics_flushed = False

    def record_presentation_retry_failure(exc: BaseException) -> None:
        nonlocal presentation_retry_failure
        presentation_retry_failure = exc
        stop_signal.set()

    def raise_presentation_retry_failure() -> None:
        if presentation_retry_failure is not None:
            raise presentation_retry_failure

    def schedule_presentation_retry(delay: float) -> None:
        nonlocal presentation_retry_failure, presentation_retry_task
        assert isinstance(delay, float)
        assert delay >= 0
        if presentation_retry_task is not None:
            if not presentation_retry_task.done():
                return
            presentation_retry_task = None
        presentation_retry_task = create_task(
            retry_pending_presentation(delay)
        )

    async def retry_pending_presentation(delay: float) -> None:
        nonlocal presentation_retry_failure
        pending_delay = delay
        try:
            while True:
                if pending_delay > 0:
                    await sleep(pending_delay)
                async with render_lock:
                    await render_snapshot()
                    if presented_snapshot_revision == snapshot_revision:
                        return
                    pending_delay = presentation_gate.delay_until_next_tick()
        except CancelledError:
            raise
        except BaseException as exc:
            record_presentation_retry_failure(exc)
            if supervisor_task is not None and not supervisor_task.done():
                supervisor_task.cancel()
            raise

    async def stop_presentation_retry(
        *,
        raise_failure: bool = True,
    ) -> None:
        nonlocal presentation_retry_task
        assert isinstance(raise_failure, bool)
        task = presentation_retry_task
        if task is None:
            if raise_failure:
                raise_presentation_retry_failure()
            return
        presentation_retry_task = None
        if not task.done():
            task.cancel()
        try:
            await task
        except CancelledError:
            pass
        except BaseException as exc:
            record_presentation_retry_failure(exc)
        if raise_failure:
            raise_presentation_retry_failure()

    async def render_snapshot(
        *,
        force: bool = False,
        flush: bool = False,
    ) -> None:
        nonlocal presented_snapshot_revision
        raise_presentation_retry_failure()
        if presented_snapshot_revision == snapshot_revision:
            return
        if not presentation_gate.should_present(force=force):
            schedule_presentation_retry(
                presentation_gate.delay_until_next_tick()
            )
            return
        snapshot = reducer.snapshot()
        request = CliStreamPresenterRequest(
            snapshot=snapshot,
            display_config=display_config,
            context=presenter_context,
            mode=presenter_mode,
        )
        async for item in presenter.present(request):
            await coordinator.handle_item(item)
        if flush:
            await coordinator.flush()
        presented_snapshot_revision = snapshot_revision

    async def reduce_projection(
        projection: StreamConsumerProjection,
    ) -> None:
        nonlocal basic_pre_answer_diagnostics_flushed
        nonlocal side_channel_events_enabled, snapshot_revision
        nonlocal last_projection_sequence
        async with render_lock:
            last_projection_sequence = projection.sequence
            if (
                not basic_pre_answer_diagnostics_flushed
                and _should_flush_basic_diagnostics_before_answer(
                    theme,
                    display_config,
                    projection,
                )
            ):
                basic_pre_answer_diagnostics_flushed = True
                if not reducer.snapshot().answer_text:
                    await render_snapshot(force=True, flush=True)
            changed = reducer.apply_projection(projection)
            if projection.terminal_outcome is not None:
                side_channel_events_enabled = False
                stop_signal.set()
            if changed:
                snapshot_revision += 1
                await render_snapshot(
                    force=(
                        projection.terminal_outcome is not None
                        or _stream_projection_forces_presentation(projection)
                    ),
                    flush=projection.terminal_outcome is not None,
                )

    async def reduce_event(event: CliSideChannelEvent) -> None:
        nonlocal snapshot_revision
        async with render_lock:
            if not side_channel_events_enabled:
                return
            if not reducer.apply_event(event):
                return
            snapshot_revision += 1
            await render_snapshot()

    async def event_stream() -> None:
        async for event in event_listen(stop_signal=stop_signal):
            assert _has_side_channel_event_fields(event)
            await reduce_event(cast(CliSideChannelEvent, event))

    async def response_stream() -> None:
        nonlocal side_channel_events_enabled, snapshot_revision
        try:
            async for projection in _stream_render_projections(
                response,
                stream_session_id="cli-render-stream",
                run_id="cli-render-run",
                turn_id="cli-render-turn",
            ):
                if projection.terminal_outcome is not None:
                    side_channel_events_enabled = False
                    stop_signal.set()
                await reduce_projection(projection)
                if (
                    projection.terminal_outcome
                    is StreamTerminalOutcome.INPUT_REQUIRED
                ):
                    raise StreamValidationError(
                        "CLI input-required projection is unavailable"
                    )
        except BaseException:
            stop_signal.set()
            raise
        else:
            side_channel_events_enabled = False
            stop_signal.set()
            async with render_lock:
                if (
                    _stream_presenter_requires_completion_snapshot(presenter)
                    and not reducer.terminal_completed
                ):
                    changed = reducer.apply_projection(
                        _stream_completed_projection(
                            (
                                last_projection_sequence + 1
                                if last_projection_sequence is not None
                                else 0
                            ),
                        )
                    )
                    if changed:
                        snapshot_revision += 1
                await render_snapshot(force=True, flush=True)
                await coordinator.flush()

    if coordinator_container is not None:
        coordinator_container["coordinator"] = None

    def _raise_stream_failures(
        failures: list[BaseException],
    ) -> None:
        assert failures
        if len(failures) == 1:
            raise failures[0]
        raise BaseExceptionGroup("CLI stream tasks failed", failures)

    async with coordinator:
        if not callable(event_listen):
            if coordinator_container is not None:
                coordinator_container["coordinator"] = coordinator
            try:
                await response_stream()
            except (CancelledError, KeyboardInterrupt, StreamValidationError):
                stop_signal.set()
                raise
            finally:
                await stop_presentation_retry()
                if coordinator_container is not None:
                    coordinator_container["coordinator"] = None
            return

        event_task = create_task(event_stream())
        response_task = create_task(response_stream())

        def _completed_task_failures(
            done_tasks: set[Task[Any]],
        ) -> list[BaseException]:
            failures: list[BaseException] = []
            for task in (event_task, response_task):
                if task not in done_tasks:
                    continue
                if task.cancelled():
                    failures.append(CancelledError())
                    continue
                exc = task.exception()
                if exc is not None:
                    failures.append(exc)
            return failures

        try:
            if coordinator_container is not None:
                coordinator_container["coordinator"] = coordinator
            if display_config.show_events or display_config.show_tools:
                await sleep(0)

            pending: set[Task[Any]] = {
                event_task,
                response_task,
            }
            while pending:
                done, pending = await wait(
                    pending,
                    return_when=FIRST_COMPLETED,
                )
                failures = _completed_task_failures(done)
                if failures:
                    stop_signal.set()
                    for pending_task in pending:
                        pending_task.cancel()
                    pending_results = await gather(
                        *pending,
                        return_exceptions=True,
                    )
                    failures.extend(
                        result
                        for result in pending_results
                        if isinstance(result, BaseException)
                        and not isinstance(result, CancelledError)
                    )
                    _raise_stream_failures(failures)

                if response_task in done:
                    stop_signal.set()
                    if event_task in pending:
                        event_task.cancel()
                        event_results = await gather(
                            event_task,
                            return_exceptions=True,
                        )
                        for result in event_results:
                            if isinstance(
                                result, BaseException
                            ) and not isinstance(result, CancelledError):
                                raise result
                    return
        except (CancelledError, KeyboardInterrupt, StreamValidationError):
            stop_signal.set()
            raise
        finally:
            stop_signal.set()
            if coordinator_container is not None:
                coordinator_container["coordinator"] = None
            for task in (event_task, response_task):
                if not task.done():
                    task.cancel()
            await gather(event_task, response_task, return_exceptions=True)
            await stop_presentation_retry()


def _stream_completed_projection(sequence: int) -> StreamConsumerProjection:
    assert isinstance(sequence, int)
    assert sequence >= 0
    return StreamConsumerProjection(
        stream_session_id="cli-render-stream",
        run_id="cli-render-run",
        turn_id="cli-render-turn",
        sequence=sequence,
        kind=StreamItemKind.STREAM_COMPLETED,
        channel=StreamChannel.CONTROL,
        correlation=StreamItemCorrelation(),
        terminal_outcome=StreamTerminalOutcome.COMPLETED,
    )


def _stream_projection_forces_presentation(
    projection: StreamConsumerProjection,
) -> bool:
    return projection.kind in (
        StreamItemKind.TOOL_EXECUTION_STARTED,
        StreamItemKind.TOOL_EXECUTION_OUTPUT,
        StreamItemKind.TOOL_EXECUTION_PROGRESS,
        StreamItemKind.TOOL_EXECUTION_COMPLETED,
        StreamItemKind.TOOL_EXECUTION_ERROR,
        StreamItemKind.TOOL_EXECUTION_CANCELLED,
    )


def _should_flush_basic_diagnostics_before_answer(
    theme: Theme,
    display_config: CliStreamDisplayConfig,
    projection: StreamConsumerProjection,
) -> bool:
    if not isinstance(theme, Theme) or not theme.default_display_tools:
        return False
    if display_config.diagnostic_channel == "none":
        return False
    if projection.channel != StreamChannel.ANSWER:
        return False
    return bool(stream_projection_text_delta(projection))


def _stream_presenter_requires_completion_snapshot(
    presenter: object,
) -> bool:
    return bool(getattr(presenter, "requires_completion_snapshot", False))


def _stream_presenter_supports_stderr_diagnostics(
    presenter: object,
) -> bool:
    return bool(getattr(presenter, "supports_stderr_diagnostics", False))


def _stream_presenter_context(
    args: Namespace,
    console: Console,
    orchestrator: Orchestrator | None,
    lm: TextGenerationModel,
    input_string: str,
    response: TextGenerationResponse,
    *,
    display_config: CliStreamDisplayConfig,
    dtokens_pick: int,
) -> CliStreamPresenterContext:
    start_thinking = bool(getattr(args, "start_thinking", False))
    if (
        start_thinking
        and bool(getattr(response, "can_think", False))
        and not bool(getattr(response, "is_thinking", False))
    ):
        set_thinking = getattr(response, "set_thinking", None)
        assert callable(set_thinking)
        set_thinking(start_thinking)

    tokenizer_tokens: tuple[str, ...] | None = None
    tokenizer_special_tokens: tuple[str, ...] | None = None
    if display_config.show_token_details or dtokens_pick > 0:
        tokenizer_config = getattr(lm, "tokenizer_config", None)
        if tokenizer_config is not None:
            tokenizer_tokens = _presenter_tokenizer_tuple(
                getattr(tokenizer_config, "tokens", None)
            )
            tokenizer_special_tokens = _presenter_tokenizer_tuple(
                getattr(tokenizer_config, "special_tokens", None)
            )

    return CliStreamPresenterContext(
        model_id=_presenter_model_id(args, lm),
        console_width=_stream_console_width(console),
        input_token_count=_stream_input_token_count(
            orchestrator,
            lm,
            input_string,
            response,
        ),
        tokenizer_tokens=tokenizer_tokens,
        tokenizer_special_tokens=tokenizer_special_tokens,
        token_probability_pick=dtokens_pick,
        start_thinking=start_thinking,
    )


def _presenter_model_id(args: Namespace, lm: TextGenerationModel) -> str:
    model_id = getattr(lm, "model_id", None)
    if isinstance(model_id, str) and model_id.strip():
        return model_id

    args_model = getattr(args, "model", None)
    if isinstance(args_model, str) and args_model.strip():
        return args_model

    return "model"


def _stream_console_width(console: Console) -> int:
    width = getattr(console, "width", 80)
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        return 80
    return width


def _stream_input_token_count(
    orchestrator: Orchestrator | None,
    lm: TextGenerationModel,
    input_string: str,
    response: TextGenerationResponse,
) -> int:
    response_count = _stream_nonzero_int(
        getattr(response, "input_token_count", None)
    )
    if response_count is not None:
        return response_count

    orchestrator_count = _stream_nonzero_int(
        getattr(orchestrator, "input_token_count", None)
        if orchestrator is not None
        else None
    )
    if orchestrator_count is not None:
        return orchestrator_count

    input_token_count = getattr(lm, "input_token_count", None)
    if callable(input_token_count):
        counted = input_token_count(input_string)
        lm_count = _stream_nonzero_int(counted)
        if lm_count is not None:
            return lm_count

    return 0


def _stream_nonzero_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value <= 0:
        return None
    return value


def _presenter_tokenizer_tuple(value: object) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        values: Iterable[object] = value.keys()
    elif isinstance(value, Iterable):
        values = value
    else:
        return None

    tokens = tuple(str(token) for token in values if isinstance(token, str))
    return tokens or None


def _legacy_stream_display_config(
    args: Namespace,
    *,
    display_tokens: int,
    refresh_per_second: int,
    tool_events_limit: int | None,
    with_stats: bool,
) -> CliStreamDisplayConfig:
    if bool(getattr(args, "quiet", False)):
        return cli_stream_display_config(
            args,
            refresh_per_second=refresh_per_second,
            interactive=True,
        )

    return CliStreamDisplayConfig(
        quiet=False,
        stats=with_stats,
        display_tools=bool(getattr(args, "display_tools", False)),
        display_events=bool(getattr(args, "display_events", False)),
        display_tools_events=tool_events_limit,
        record=bool(getattr(args, "record", False)),
        interactive=True,
        refresh_per_second=refresh_per_second,
        answer_height=int(getattr(args, "display_answer_height", 12)),
        answer_height_expand=bool(
            getattr(args, "display_answer_height_expand", False)
        ),
        display_tokens=display_tokens,
        display_pause=int(getattr(args, "display_pause", None) or 0),
        display_probabilities=bool(
            getattr(args, "display_probabilities", False)
        ),
        display_probabilities_maximum=float(
            getattr(args, "display_probabilities_maximum", 0.8)
        ),
        display_probabilities_sample_minimum=float(
            getattr(args, "display_probabilities_sample_minimum", 0.1)
        ),
        display_time_to_n_token=getattr(args, "display_time_to_n_token", None),
        display_reasoning_time=not bool(
            getattr(args, "skip_display_reasoning_time", False)
        ),
        display_reasoning=bool(getattr(args, "display_reasoning", False)),
        display_reasoning_raw=bool(
            getattr(args, "display_reasoning_raw", False)
        ),
        display_reasoning_simple=bool(
            getattr(args, "display_reasoning_simple", False)
        ),
    )


def _theme_default_stream_display_config(
    theme: Theme,
    display_config: CliStreamDisplayConfig,
    orchestrator: Orchestrator | None,
) -> CliStreamDisplayConfig:
    if display_config.quiet or display_config.display_tools:
        return display_config
    if not isinstance(theme, Theme) or not theme.default_display_tools:
        return display_config
    if not _orchestrator_has_tools(orchestrator):
        return display_config
    return replace(display_config, display_tools=True)


def _orchestrator_has_tools(orchestrator: Orchestrator | None) -> bool:
    if orchestrator is None:
        return False
    tool = getattr(orchestrator, "tool", None)
    if tool is None:
        return False
    is_empty = getattr(tool, "is_empty", True)
    return isinstance(is_empty, bool) and not is_empty


async def _print_plain_stdout_response(
    console: Console,
    response: TextGenerationResponse | AsyncIterator[Any],
) -> None:
    async for token in _plain_stdout_projections(response):
        if token.channel != StreamChannel.ANSWER:
            continue
        text_token = stream_projection_text_delta(token)
        if text_token is None:
            continue
        console.print(text_token, end="")
        await sleep(0)


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
    display_config: CliStreamDisplayConfig | None = None,
) -> None:
    display_config = display_config or cli_stream_display_config(
        args,
        refresh_per_second=refresh_per_second,
        interactive=True,
    )
    refresh_per_second = display_config.refresh_per_second
    event_manager = orchestrator.event_manager
    if not event_manager or (
        not display_config.show_events and not display_config.show_tools
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
        if display_config.show_events
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
        if display_config.show_tools
        else None
    )
    try:
        async for e in event_manager.listen(stop_signal=stop_signal):
            tool_view = _event_targets_tool_panel(e)
            if (tool_view and not display_config.show_tools) or (
                not tool_view and not display_config.show_events
            ):
                continue

            events_renderable = theme.events(
                event_manager.history,
                events_limit=(
                    display_config.display_tools_events if tool_view else 4
                ),
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


def _has_side_channel_event_fields(event: object) -> bool:
    return hasattr(event, "type") and hasattr(event, "payload")


def _event_targets_tool_panel(event: object) -> bool:
    canonical_payload = _canonical_event_payload(event)
    if canonical_payload is None:
        return getattr(event, "type") in TOOL_TYPES
    channel = canonical_payload.get("channel")
    if channel in _CANONICAL_TOOL_EVENT_CHANNELS:
        return True
    kind = canonical_payload.get("kind")
    return isinstance(kind, str) and kind.startswith(
        _CANONICAL_TOOL_EVENT_KIND_PREFIXES
    )


def _canonical_event_payload(event: object) -> Mapping[str, Any] | None:
    payload = getattr(event, "payload", None)
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
    display_config: CliStreamDisplayConfig | None = None,
) -> None:
    display_config = display_config or _legacy_stream_display_config(
        args,
        display_tokens=display_tokens,
        refresh_per_second=refresh_per_second,
        tool_events_limit=tool_events_limit,
        with_stats=with_stats,
    )
    display_tokens = display_config.display_tokens
    refresh_per_second = display_config.refresh_per_second
    tool_events_limit = display_config.display_tools_events
    display_time_to_n_token = (
        display_config.display_time_to_n_token
        if display_config
        else args.display_time_to_n_token
    )
    display_reasoning_time = (
        display_config.display_reasoning_time
        if display_config
        else not getattr(args, "skip_display_reasoning_time")
    )
    display_reasoning = getattr(args, "display_reasoning", False)
    display_pause = (
        display_config.display_pause
        if display_config
        else (
            args.display_pause
            if args.display_pause and args.display_pause > 0
            else 0
        )
    )
    display_probabilities = (
        display_config.show_probabilities
        if display_config
        else args.display_probabilities
    )
    display_probabilities_maximum = (
        display_config.display_probabilities_maximum
        if display_config
        else args.display_probabilities_maximum
    )
    display_probabilities_sample_minimum = (
        display_config.display_probabilities_sample_minimum
        if display_config
        else args.display_probabilities_sample_minimum
    )
    display_answer_height_expand = (
        display_config.answer_height_expand
        if display_config
        else getattr(args, "display_answer_height_expand", False)
    )
    display_answer_height = (
        display_config.answer_height
        if display_config
        else getattr(args, "display_answer_height", 12)
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
    limit_answer_height = not display_answer_height_expand
    answer_height = display_answer_height
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
            or not display_probabilities
            or display_probabilities_maximum <= 0
        ):
            return False
        low_probability = (
            dtoken.probability is not None
            and dtoken.probability < display_probabilities_maximum
        )
        sampled_alternative = any(
            candidate.id != dtoken.id
            and candidate.probability is not None
            and candidate.probability >= display_probabilities_sample_minimum
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
                    display_probabilities if dtokens_pick > 0 else False
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
                display_tools=display_config.show_tools,
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


def _side_channel_event_type(event: object) -> str:
    assert _has_side_channel_event_fields(event)
    event_type = getattr(event, "type")
    if isinstance(event_type, EventType):
        return event_type.value
    return str(event_type)


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
    assert not getattr(args, "record", False)
    if group and group_index is not None:
        group.renderables[group_index] = frame
        live.refresh()
    else:
        live.update(frame)
