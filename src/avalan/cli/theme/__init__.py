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

from abc import ABC, abstractmethod
from dataclasses import fields
from datetime import datetime
from logging import Logger
from typing import Any, Callable, Generator, Literal, TypeAlias, cast
from uuid import UUID

from humanize import intcomma, intword, naturalsize, naturaltime
from numpy import ndarray
from rich.console import RenderableType

Formatter: TypeAlias = Callable[[datetime | float | int], str]
Formatters = dict[Literal["datetime", "number", "quantity", "size"], Formatter]
Spinner = Literal["cache_accessing", "connecting", "thinking", "downloading"]
Data = str
DataValue = datetime | float | int | str | None
Styler: TypeAlias = Callable[[Data, DataValue, str | None, bool | str], str]
Stylers = dict[Data, Styler]


class Theme(ABC):
    _all_spinners: dict[Spinner, str | None]
    _all_stylers: Stylers
    _all_styles: dict[str, str]
    _icons: dict[Data, str | None]
    _: Callable[[str], str]

    @property
    def formatters(self) -> Formatters:
        return {}

    @property
    def icons(self) -> dict[str, str]:
        return {}

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

    @abstractmethod
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
        raise NotImplementedError()

    @abstractmethod
    def agent(
        self,
        agent: Orchestrator,
        *args: object,
        models: list[Model | str],
        cans_access: bool | None = None,
        can_access: bool | None = None,
    ) -> RenderableType:
        raise NotImplementedError()

    @abstractmethod
    def ask_access_token(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def ask_delete_paths(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def ask_login_to_hub(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def ask_secret_password(self, key: str) -> str:
        raise NotImplementedError()

    @abstractmethod
    def ask_override_secret(self, key: str) -> str:
        raise NotImplementedError()

    @abstractmethod
    def bye(self) -> RenderableType:
        raise NotImplementedError()

    @abstractmethod
    def cache_delete(
        self, cache_deletion: HubCacheDeletion | None, deleted: bool
    ) -> RenderableType:
        raise NotImplementedError()

    @abstractmethod
    def cache_list(
        self,
        cache_dir: str,
        cached_models: list[HubCache],
        display_models: list[str] | None = None,
        show_summary: bool = False,
    ) -> RenderableType:
        raise NotImplementedError()

    @abstractmethod
    def download_access_denied(
        self, model_id: str, model_url: str
    ) -> RenderableType:
        raise NotImplementedError()

    @abstractmethod
    def download_start(self, model_id: str) -> RenderableType:
        raise NotImplementedError()

    @abstractmethod
    def download_progress(self) -> tuple[str | RenderableType]:
        raise NotImplementedError()

    @abstractmethod
    def download_finished(self, model_id: str, path: str) -> RenderableType:
        raise NotImplementedError()

    @abstractmethod
    def events(
        self,
        events: list[Event],
        *,
        events_limit: int | None = None,
        include_tokens: bool = True,
        include_tool_detect: bool = True,
        include_tools: bool = True,
        include_non_tools: bool = True,
        tool_view: bool = False,
    ) -> RenderableType:
        raise NotImplementedError()

    @abstractmethod
    def logging_in(self, domain: str) -> str:
        raise NotImplementedError()

    @abstractmethod
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
        raise NotImplementedError()

    @abstractmethod
    def memory_embeddings_comparison(
        self, similarities: dict[str, Similarity], most_similar: str
    ) -> RenderableType:
        raise NotImplementedError()

    @abstractmethod
    def memory_embeddings_search(
        self,
        matches: list[SearchMatch],
        *args: object,
        match_preview_length: int = 300,
    ) -> RenderableType:
        raise NotImplementedError()

    @abstractmethod
    def memory_partitions(
        self,
        partitions: list[TextPartition],
        *args: object,
        display_partitions: int,
    ) -> RenderableType:
        raise NotImplementedError()

    @abstractmethod
    def model(
        self,
        model: Model,
        *args: object,
        can_access: bool | None = None,
        expand: bool = False,
        summary: bool = False,
    ) -> RenderableType:
        raise NotImplementedError()

    @abstractmethod
    def model_display(
        self,
        model_config: ModelConfig | SentenceTransformerModelConfig | None,
        tokenizer_config: TokenizerConfig | None,
        *args: object,
        is_runnable: bool | None = None,
        summary: bool = False,
    ) -> RenderableType:
        raise NotImplementedError()

    @abstractmethod
    def recent_messages(
        self,
        participant_id: UUID,
        agent: Orchestrator,
        messages: list[EngineMessage],
    ) -> RenderableType:
        raise NotImplementedError()

    @abstractmethod
    def saved_tokenizer_files(
        self,
        directory_path_or_total: str | int,
        total_files: int | None = None,
    ) -> RenderableType:
        raise NotImplementedError()

    @abstractmethod
    def search_message_matches(
        self,
        participant_id: UUID,
        agent: Orchestrator,
        messages: list[EngineMessageScored],
    ) -> RenderableType:
        raise NotImplementedError()

    @abstractmethod
    def memory_search_matches(
        self,
        participant_id: UUID,
        namespace: str,
        memories: list[PermanentMemoryPartition],
    ) -> RenderableType:
        raise NotImplementedError()

    @abstractmethod
    def tokenizer_config(self, config: TokenizerConfig) -> RenderableType:
        raise NotImplementedError()

    @abstractmethod
    def tokenizer_tokens(
        self,
        dtokens: list[Token],
        added_tokens: list[str] | None,
        special_tokens: list[str] | None,
        display_details: bool = False,
        current_dtoken: Token | None = None,
        dtokens_selected: list[Token] | None = None,
    ) -> RenderableType:
        raise NotImplementedError()

    @abstractmethod
    def display_image_entities(
        self, entities: list[ImageEntity], sort: bool
    ) -> RenderableType:
        raise NotImplementedError()

    @abstractmethod
    def display_image_entity(
        self, image_entity: ImageEntity
    ) -> RenderableType:
        raise NotImplementedError()

    @abstractmethod
    def display_audio_labels(
        self, audio_labels: dict[str, float]
    ) -> RenderableType:
        raise NotImplementedError()

    @abstractmethod
    def display_image_labels(self, labels: list[str]) -> RenderableType:
        raise NotImplementedError()

    @abstractmethod
    def display_token_labels(
        self, token_labels: list[dict[str, str]]
    ) -> RenderableType:
        raise NotImplementedError()

    @abstractmethod
    async def tokens(
        self,
        model_id: str,
        added_tokens: list[str] | None,
        special_tokens: list[str] | None,
        display_token_size: int | None,
        display_probabilities: bool,
        pick: int,
        focus_on_token_when: Callable[[Token], bool] | None,
        thinking_text_tokens: list[str],
        tool_text_tokens: list[str],
        answer_text_tokens: list[str],
        tokens: list[Token] | None,
        input_token_count: int,
        total_tokens: int,
        tool_events: list[Event] | None,
        tool_event_calls: list[Event] | None,
        tool_event_results: list[Event] | None,
        ttft: float,
        ttnt: float | None,
        ttsr: float | None,
        elapsed: float,
        console_width: int,
        logger: Logger,
        event_stats: EventStats | None = None,
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
    ) -> Generator[tuple[Token | None, RenderableType], None, None]:
        raise NotImplementedError()

    @abstractmethod
    def welcome(
        self,
        url: str,
        name: str,
        version: str,
        license: str,
        user: User | None,
    ) -> RenderableType:
        raise NotImplementedError()

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
            **{data: None for data in data_keys},
            **self.icons,
            **(icons or {}),
        }

        data_keys.extend(all_icons.keys())
        data_keys.extend(self.styles.keys())

        self._all_spinners = {
            **{
                "cache_accessing": None,
                "connecting": None,
                "thinking": None,
                "downloading": None,
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

    def get_spinner(self, spinner_name: Spinner) -> str:
        return cast(str, self._all_spinners[spinner_name])

    def __call__(self, item: Model | str) -> RenderableType:
        return self.model(item) if isinstance(item, Model) else str(item)

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
