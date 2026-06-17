from ...agent.orchestrator import Orchestrator
from ...entities import (
    EngineMessage,
    EngineMessageScored,
    HubCache,
    HubCacheDeletion,
    ImageEntity,
    Model,
    ModelConfig,
    SearchMatch,
    SentenceTransformerModelConfig,
    Similarity,
    TextPartition,
    Token,
    TokenizerConfig,
    User,
)
from ...event import Event, EventStats
from ...memory.permanent import Memory as Memory
from ...memory.permanent import PermanentMemoryPartition
from ...model.stream import StreamChannel, StreamItemKind
from ..download import DownloadCompleteColumn

from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import datetime
from logging import Logger
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Literal,
    TypeAlias,
    cast,
)
from uuid import UUID

from humanize import intcomma, intword, naturalsize, naturaltime
from rich.console import RenderableType
from rich.progress import BarColumn, SpinnerColumn, TimeElapsedColumn
from rich.spinner import Spinner as RichSpinner

if TYPE_CHECKING:
    from numpy import ndarray
else:

    class ndarray:  # noqa: D101
        def __class_getitem__(cls, _: Any) -> Any:
            return Any


Formatter: TypeAlias = Callable[[datetime | float | int], str]
Formatters = dict[Literal["datetime", "number", "quantity", "size"], Formatter]
Spinner = Literal[
    "agent_loading",
    "cache_accessing",
    "connecting",
    "downloading",
    "thinking",
    "tool_running",
]
Data = str
DataValue = datetime | float | int | str | UUID | None
Styler: TypeAlias = Callable[[Data, DataValue, str | None, bool | str], str]
Stylers = dict[Data, Styler]


@dataclass(frozen=True, kw_only=True, slots=True)
class TokenRenderDisplayTokenCandidate:
    token: str
    id: int | None = None
    probability: float | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class TokenRenderDisplayToken:
    sequence: int
    kind: StreamItemKind
    channel: StreamChannel
    token: str
    id: int | None = None
    probability: float | None = None
    step: int | None = None
    probability_distribution: str | None = None
    tokens: tuple[TokenRenderDisplayTokenCandidate, ...] = ()


@dataclass(frozen=True, kw_only=True, slots=True)
class TokenRenderState:
    model_id: str
    projection_sequence: int | None = None
    projection_kind: StreamItemKind | None = None
    projection_channel: StreamChannel | None = None
    added_tokens: tuple[str, ...] | None = None
    special_tokens: tuple[str, ...] | None = None
    display_token_size: int | None = None
    display_probabilities: bool = False
    pick: int = 0
    focus_on_token_when: Callable[[TokenRenderDisplayToken], bool] | None = (
        None
    )
    reasoning_text_tokens: tuple[str, ...] = ()
    tool_text_tokens: tuple[str, ...] = ()
    answer_text_tokens: tuple[str, ...] = ()
    display_tokens: tuple[TokenRenderDisplayToken, ...] = ()
    display_reasoning: bool = False
    display_tools: bool = False
    input_token_count: int = 0
    total_tokens: int = 0
    tool_token_count: int = 0
    tool_running: bool = False
    tool_running_spinner: RichSpinner | None = None
    ttft: float | None = None
    ttnt: float | None = None
    ttsr: float | None = None
    elapsed: float = 0.0
    event_stats: EventStats | None = None
    start_thinking: bool = False


TokenRenderFrame: TypeAlias = tuple[
    TokenRenderDisplayToken | None, RenderableType
]


class Theme:
    _all_spinners: dict[Spinner, str | None]
    _all_stylers: Stylers
    _all_styles: dict[str, str]
    _icons: dict[Data, str]
    _: Callable[[str], str]

    @property
    def formatters(self) -> Formatters:
        return {}

    @property
    def icons(self) -> dict[Data, str]:
        return {
            "agent_output": "",
            "user_input": "",
        }

    @property
    def quantity_data(self) -> list[str]:
        return []

    @property
    def spinners(self) -> dict[Spinner, str]:
        return {}

    @property
    def stylers(self) -> Stylers:
        return {}

    @property
    def styles(self) -> dict[str, str]:
        return {}

    def action(
        self,
        name: str,
        description: str,
        author: str,
        model_id: str,
        library_name: str,
        highlight: bool,
        finished: bool,
    ) -> RenderableType:
        status = self._("finished") if finished else self._("running")
        marker = "*" if highlight else "-"
        return (
            f"{marker} {name}: {description}\n"
            f"{self._('Author')}: {author}\n"
            f"{self._('Model')}: {model_id}\n"
            f"{self._('Library')}: {library_name}\n"
            f"{self._('Status')}: {status}"
        )

    def agent(
        self,
        agent: Orchestrator,
        *args: object,
        models: list[Model | str],
        cans_access: bool | None = None,
        can_access: bool | None = None,
    ) -> RenderableType:
        _ = args, agent, cans_access, can_access
        model_ids = ", ".join(
            model.id if isinstance(model, Model) else model for model in models
        )
        return self._("{agent}\n{models_label}: {models}").format(
            agent=self._("Agent"),
            models_label=self._("Models"),
            models=model_ids or self._("none"),
        )

    def ask_access_token(self) -> str:
        return self._("Enter your Huggingface access token")

    def ask_delete_paths(self) -> str:
        return self._("Delete selected paths?")

    def ask_login_to_hub(self) -> str:
        return self._("Login to huggingface?")

    def ask_secret_password(self, key: str) -> str:
        return self._("Enter secret for {key}").format(key=key)

    def ask_override_secret(self, key: str) -> str:
        return self._("Secret {key} exists, override?").format(key=key)

    def bye(self) -> RenderableType:
        return self._("bye :)")

    def cache_delete(
        self,
        cache_deletion: HubCacheDeletion | None,
        deleted: bool = False,
    ) -> RenderableType:
        _ = cache_deletion
        return (
            self._("Deleted cache entry.")
            if deleted
            else self._("Cache entry selected.")
        )

    def cache_list(
        self,
        cache_dir: str,
        cached_models: list[HubCache],
        display_models: list[str] | None = None,
        show_summary: bool = False,
    ) -> RenderableType:
        _ = display_models, show_summary
        model_ids = ", ".join(cache.model_id for cache in cached_models)
        return self._("Cache: {cache_dir}\nModels: {models}").format(
            cache_dir=cache_dir,
            models=model_ids or self._("none"),
        )

    def download_access_denied(
        self, model_id: str, model_url: str
    ) -> RenderableType:
        return self._(
            "Access denied while downloading {model_id}: {model_url}"
        ).format(model_id=model_id, model_url=model_url)

    def download_start(self, model_id: str) -> RenderableType:
        return self._("Downloading model {model_id}.").format(
            model_id=model_id
        )

    def download_progress(self) -> tuple[str | RenderableType, ...]:
        return (
            cast(RenderableType, SpinnerColumn()),
            (
                "[progress.description]{task.description}"
                "[progress.percentage]{task.percentage:>4.0f}%"
            ),
            cast(RenderableType, BarColumn(bar_width=None)),
            "[",
            cast(RenderableType, DownloadCompleteColumn()),
            "-",
            cast(RenderableType, TimeElapsedColumn()),
            "]",
        )

    def download_finished(self, model_id: str, path: str) -> RenderableType:
        return self._("Downloaded model {model_id} to {path}.").format(
            model_id=model_id,
            path=path,
        )

    def events(
        self,
        events: list[Event],
        *,
        events_limit: int | None = None,
        height: int | None = None,
        include_tokens: bool = True,
        include_tool_detect: bool = True,
        include_tools: bool = True,
        include_non_tools: bool = True,
        tool_view: bool = False,
    ) -> RenderableType | None:
        _ = (
            events,
            events_limit,
            height,
            include_tokens,
            include_tool_detect,
            include_tools,
            include_non_tools,
            tool_view,
        )
        return None

    def logging_in(self, domain: str) -> str:
        return self._("Logging in to {domain}...").format(domain=domain)

    def memory_embeddings(
        self,
        input_string: str,
        embeddings: ndarray[Any, Any],
        *args: object,
        total_tokens: int,
        minv: float,
        maxv: float,
        meanv: float,
        stdv: float,
        normv: float,
        embedding_peek: int | None = 3,
        horizontal: bool = True,
        input_string_peek: int = 40,
        show_stats: bool = True,
        partition: int | None = None,
        total_partitions: int | None = None,
    ) -> RenderableType:
        _ = (
            embeddings,
            args,
            minv,
            maxv,
            meanv,
            stdv,
            normv,
            embedding_peek,
            horizontal,
            input_string_peek,
            show_stats,
            partition,
            total_partitions,
        )
        return self._("{input_string}\nTokens: {total_tokens}").format(
            input_string=input_string,
            total_tokens=total_tokens,
        )

    def memory_embeddings_comparison(
        self, similarities: dict[str, Similarity], most_similar: str
    ) -> RenderableType:
        _ = similarities
        return self._("Most similar: {most_similar}").format(
            most_similar=most_similar
        )

    def memory_embeddings_search(
        self,
        matches: list[SearchMatch],
        *args: object,
        match_preview_length: int = 300,
    ) -> RenderableType:
        _ = args, match_preview_length
        return "\n".join(match.match for match in matches)

    def memory_partitions(
        self,
        partitions: list[TextPartition],
        *args: object,
        display_partitions: int,
    ) -> RenderableType:
        _ = args
        return "\n".join(
            partition.data for partition in partitions[:display_partitions]
        )

    def model(
        self,
        model: Model,
        *args: object,
        can_access: bool | None = None,
        expand: bool = False,
        summary: bool = False,
    ) -> RenderableType:
        _ = args, can_access, expand, summary
        return model.id

    def model_display(
        self,
        model_config: ModelConfig | SentenceTransformerModelConfig | None,
        tokenizer_config: TokenizerConfig | None,
        *args: object,
        is_runnable: bool | None = None,
        summary: bool = False,
    ) -> RenderableType:
        _ = tokenizer_config, args, is_runnable, summary
        model_type = getattr(model_config, "model_type", None)
        return self._("Model type: {model_type}").format(
            model_type=model_type or self._("unknown")
        )

    def recent_messages(
        self,
        participant_id: UUID,
        agent: Orchestrator,
        messages: list[EngineMessage],
    ) -> RenderableType:
        _ = agent
        return self._("{participant_id}: {messages} recent messages").format(
            participant_id=participant_id,
            messages=len(messages),
        )

    def saved_tokenizer_files(
        self,
        directory_path_or_total: str | int,
        total_files: int | None = None,
    ) -> RenderableType:
        total = (
            total_files if total_files is not None else directory_path_or_total
        )
        return self._("Saved tokenizer files: {total}").format(total=total)

    def search_message_matches(
        self,
        participant_id: UUID,
        agent: Orchestrator,
        messages: list[EngineMessageScored],
    ) -> RenderableType:
        _ = agent
        return self._("{participant_id}: {messages} message matches").format(
            participant_id=participant_id,
            messages=len(messages),
        )

    def memory_search_matches(
        self,
        participant_id: UUID,
        namespace: str,
        memories: list[PermanentMemoryPartition],
    ) -> RenderableType:
        return self._(
            "{participant_id}/{namespace}: {memories} memory matches"
        ).format(
            participant_id=participant_id,
            namespace=namespace,
            memories=len(memories),
        )

    def tokenizer_config(self, config: TokenizerConfig) -> RenderableType:
        return config.name_or_path

    def tokenizer_tokens(
        self,
        dtokens: list[Token],
        added_tokens: list[str] | None,
        special_tokens: list[str] | None,
        display_details: bool = False,
        current_dtoken: Token | None = None,
        dtokens_selected: list[Token] | None = None,
    ) -> RenderableType:
        _ = (
            added_tokens,
            special_tokens,
            display_details,
            current_dtoken,
            dtokens_selected,
        )
        return "\n".join(dtoken.token for dtoken in dtokens)

    def display_image_entities(
        self, entities: list[ImageEntity], sort: bool
    ) -> RenderableType:
        sorted_entities = (
            sorted(entities, key=lambda entity: entity.label)
            if sort
            else entities
        )
        return "\n".join(entity.label for entity in sorted_entities)

    def display_image_entity(
        self, image_entity: ImageEntity
    ) -> RenderableType:
        return image_entity.label

    def display_audio_labels(
        self, audio_labels: dict[str, float]
    ) -> RenderableType:
        return "\n".join(
            f"{label}: {score}" for label, score in audio_labels.items()
        )

    def display_image_labels(self, labels: list[str]) -> RenderableType:
        return "\n".join(labels)

    def display_token_labels(
        self, token_labels: list[dict[str, str]]
    ) -> RenderableType:
        return "\n".join(
            f"{token}: {label}"
            for token_label in token_labels
            for token, label in token_label.items()
        )

    def token_frames(
        self,
        state: TokenRenderState,
        *,
        console_width: int,
        logger: Logger,
        maximum_frames: int | None = None,
        logits_count: int | None = None,
        tool_events_limit: int | None = None,
        think_height: int = 6,
        think_padding: int = 1,
        tool_height: int = 6,
        tool_padding: int = 1,
        height: int = 12,
        padding: int = 1,
        wrap_padding: int = 4,
        limit_think_height: bool = True,
        limit_tool_height: bool = True,
        limit_answer_height: bool = False,
        start_thinking: bool = False,
    ) -> tuple[TokenRenderFrame, ...]:
        _ = (
            state,
            console_width,
            logger,
            maximum_frames,
            logits_count,
            tool_events_limit,
            think_height,
            think_padding,
            tool_height,
            tool_padding,
            height,
            padding,
            wrap_padding,
            limit_think_height,
            limit_tool_height,
            limit_answer_height,
            start_thinking,
        )
        return ()

    async def tokens(
        self,
        state: TokenRenderState,
        *,
        console_width: int,
        logger: Logger,
        maximum_frames: int | None = None,
        logits_count: int | None = None,
        tool_events_limit: int | None = None,
        think_height: int = 6,
        think_padding: int = 1,
        tool_height: int = 6,
        tool_padding: int = 1,
        height: int = 12,
        padding: int = 1,
        wrap_padding: int = 4,
        limit_think_height: bool = True,
        limit_tool_height: bool = True,
        limit_answer_height: bool = False,
        start_thinking: bool = False,
    ) -> AsyncGenerator[TokenRenderFrame, None]:
        for frame in self.token_frames(
            state,
            console_width=console_width,
            logger=logger,
            maximum_frames=maximum_frames,
            logits_count=logits_count,
            tool_events_limit=tool_events_limit,
            think_height=think_height,
            think_padding=think_padding,
            tool_height=tool_height,
            tool_padding=tool_padding,
            height=height,
            padding=padding,
            wrap_padding=wrap_padding,
            limit_think_height=limit_think_height,
            limit_tool_height=limit_tool_height,
            limit_answer_height=limit_answer_height,
            start_thinking=start_thinking,
        ):
            yield frame

    def welcome(
        self,
        url: str,
        name: str,
        version: str,
        license: str,
        user: User | None,
    ) -> RenderableType:
        user_name = (
            self._("\nUser: {user_name}").format(user_name=user.name)
            if user
            else ""
        )
        return self._(
            "{name} {version}\n{url}\nLicense: {license}{user}"
        ).format(
            name=name,
            version=version,
            url=url,
            license=license,
            user=user_name,
        )

    def __init__(
        self,
        translator: Callable[[str], str],
        translator_plurals: Callable[[str, str, int], str],
        formatters: Formatters | None = None,
        stylers: Stylers | None = None,
        styles: dict[str, str] | None = None,
        spinners: dict[Spinner, str] | None = None,
        icons: dict[str, str] | None = None,
        quantity_data: list[str] | None = None,
    ) -> None:
        data_keys = sorted(
            set(field.name for field in fields(Model))
            | set(field.name for field in fields(User))
            | set(self.icons.keys())
            | set((icons or {}).keys())
        )

        formatters = {
            **{
                "datetime": lambda v: naturaltime(cast(datetime, v)),
                "number": lambda v: intcomma(cast(float | int, v)),
                "quantity": lambda v: intword(cast(float | int, v)),
                "size": lambda v: naturalsize(cast(float | int, v)),
            },
            **self.formatters,
            **(formatters or {}),
        }

        all_icons = {
            **{data: "" for data in data_keys},
            **self.icons,
            **(icons or {}),
        }

        data_keys.extend(all_icons.keys())
        data_keys.extend(self.styles.keys())

        self._all_spinners = {
            **{
                "agent_loading": None,
                "cache_accessing": None,
                "connecting": None,
                "thinking": None,
                "downloading": None,
                "tool_running": None,
            },
            **self.spinners,
            **(spinners or {}),
        }
        quantity_keys = quantity_data or self.quantity_data

        def _build_styler(data_key: Data) -> Styler:
            def _styler(
                data: Data,
                value: DataValue,
                prefix: str | None = None,
                icon: bool | str = True,
            ) -> str:
                return "".join(
                    [
                        prefix or "",
                        (
                            f"{all_icons[data]} "
                            if isinstance(icon, bool)
                            and icon
                            and data in self._icons
                            and all_icons[data]
                            else icon if isinstance(icon, str) else ""
                        ),
                        f"[{data}]",
                        (
                            formatters["quantity"](cast(float | int, value))
                            if data_key in quantity_keys
                            else (
                                formatters["datetime"](
                                    cast(datetime | float | int, value)
                                )
                                if isinstance(value, datetime)
                                else (
                                    formatters["number"](value)
                                    if isinstance(value, int)
                                    or isinstance(value, float)
                                    else str(value)
                                )
                            )
                        ),
                        f"[/{data}]",
                    ]
                )

            return _styler

        self._all_stylers = {
            **{data: _build_styler(data) for data in data_keys},
            **self.stylers,
            **(stylers or {}),
        }
        self._all_styles = {
            **{data: "" for data in data_keys},
            **self.styles,
            **(styles or {}),
        }
        self._icons = all_icons
        self._ = translator
        self._n = translator_plurals

    def get_styles(self) -> dict[str, str]:
        return self._all_styles

    def get_spinner(self, spinner_name: Spinner) -> str | None:
        return self._all_spinners[spinner_name]

    def __call__(self, item: Model | str) -> RenderableType:
        return self.model(item) if isinstance(item, Model) else str(item)

    def flow_run_progress_message(
        self,
        event_type: str,
        *,
        node: str | None = None,
        status: str | None = None,
        attempt: int | None = None,
        flow_name: str | None = None,
    ) -> str:
        """Return the live flow run status line."""
        _ = status, flow_name
        if node:
            if event_type == "flow_node_started":
                suffix = (
                    f" (attempt {attempt})"
                    if attempt is not None and attempt > 1
                    else ""
                )
                return f"Running {node}{suffix}."
            if event_type == "flow_node_retrying":
                suffix = (
                    f" after attempt {attempt}" if attempt is not None else ""
                )
                return f"Retrying {node}{suffix}."
            if event_type == "flow_node_completed":
                return f"Finished {node}."
            if event_type == "flow_node_failed":
                return f"{node} failed."
            if event_type == "flow_node_skipped":
                return f"Skipped {node}."
            if event_type == "flow_node_paused":
                return f"Paused {node}."
            if event_type == "flow_node_resumed":
                return f"Resumed {node}."
            if event_type == "flow_node_cancelled":
                return f"Cancelled {node}."
        if event_type == "flow_started":
            return "Flow run started."
        if event_type == "flow_completed":
            return "Flow run completed."
        if event_type == "flow_cancelled":
            return "Flow run cancelled."
        return "Flow run is active."

    def flow_run_progress(
        self,
        mermaid_source: str,
        *,
        node_states: Mapping[str, str],
        active_nodes: tuple[str, ...],
        message: str,
        console_width: int,
        flow_stats: Mapping[str, Mapping[str, int | float]] | None = None,
    ) -> RenderableType:
        """Return the live flow run progress renderable."""
        _ = node_states, active_nodes, console_width, flow_stats
        return f"{message}\n\n{mermaid_source}"

    def _f(
        self,
        data: Data,
        value: DataValue,
        prefix: str | None = None,
        icon: bool | str = True,
    ) -> str:
        return (
            self._all_stylers[data](data, value, prefix, icon)
            if data in self._all_stylers
            else str(value)
        )
