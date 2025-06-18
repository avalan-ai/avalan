from abc import ABC, abstractmethod
from ...agent.orchestrator import Orchestrator
from ...entities import (
    EngineMessage,
    EngineMessageScored,
    HubCache,
    HubCacheDeletion,
    Model,
    ModelConfig,
    SearchMatch,
    SentenceTransformerModelConfig,
    Similarity,
    Token,
    TokenizerConfig,
    User,
)
from ...event import Event, EventStats
from ...memory.partitioner.text import TextPartition
from ...memory.permanent import Memory
from dataclasses import fields
from datetime import datetime
from enum import StrEnum
from humanize import intcomma, intword, naturalsize, naturaltime
from logging import Logger
from numpy import ndarray
from rich.console import RenderableType
from typing import Callable, Generator, Literal
from uuid import UUID

Formatter = (
    Callable[[datetime], str] | Callable[[float], str] | Callable[[int], str]
)
Formatters = dict[Literal["datetime", "number", "quantity", "size"], Formatter]
Spinner = Literal["cache_accessing", "connecting", "thinking", "downloading"]
Data = StrEnum(
    "Data",
    {
        **{field.name: field.name for field in fields(Model)},
        **{field.name: field.name for field in fields(User)},
    },
)
DataValue = datetime | float | int | str | None
Stylers = dict[Data, Callable[[DataValue], str]]


class Theme(ABC):
    _all_spinners: dict[Spinner, str]
    _all_stylers: Stylers
    _all_styles: dict[Data, str]
    _icons: dict[Data, str]
    _: Callable[[str], str]

    @property
    def formatters(self) -> Formatters:
        return {}

    @property
    def icons(self) -> dict[Data, str]:
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
    def styles(self) -> dict[Data, str]:
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
        *args,
        models: list[Model | str],
        cans_access: bool | None,
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
        embeddings: ndarray,
        *args,
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
        *args,
        match_preview_length: int = 300,
    ) -> RenderableType:
        raise NotImplementedError()

    @abstractmethod
    def memory_partitions(
        self, partitions: list[TextPartition], *args, display_partitions: int
    ) -> RenderableType:
        raise NotImplementedError()

    @abstractmethod
    def model(
        self,
        model: Model,
        *args,
        can_access: bool | None = None,
        expand: bool = False,
        summary: bool = False,
    ) -> RenderableType:
        raise NotImplementedError()

    @abstractmethod
    def model_display(
        self,
        model_config: ModelConfig | SentenceTransformerModelConfig | None,
        tokenizer_config: TokenizerConfig,
        *args,
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
    ):
        raise NotImplementedError()

    @abstractmethod
    def saved_tokenizer_files(
        directory_path: str, total_files: int
    ) -> RenderableType:
        raise NotImplementedError()

    @abstractmethod
    def search_message_matches(
        self,
        participant_id: UUID,
        agent: Orchestrator,
        messages: list[EngineMessageScored],
    ):
        raise NotImplementedError()

    @abstractmethod
    def memory_search_matches(
        self, participant_id: UUID, namespace: str, memories: list[Memory]
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
        current_dtoken: Token | None = None,
        dtokens_selected: list[Token] = [],
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
        text_tokens: list[str],
        tokens: list[Token] | None,
        input_token_count: int,
        total_tokens: int,
        tool_events: list[Event] | None,
        tool_event_calls: list[Event] | None,
        tool_event_results: list[Event] | None,
        ttft: float,
        ttnt: float,
        ellapsed: float,
        console_width: int,
        logger: Logger,
        event_stats: EventStats | None = None,
        maximum_frames: int | None = None,
        logits_count: int | None = None,
        tool_events_limit: int | None = None,
        think_height: int = 6,
        think_padding: int = 1,
        height: int = 12,
        padding: int = 1,
        wrap_padding: int = 4,
        limit_think_height: bool = True,
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
        formatters: Formatters = {},
        stylers: Stylers = {},
        styles: dict[Data, str] = {},
        spinners: dict[Spinner, str] = {},
        icons: dict[Data, str] = {},
        quantity_data: list[str] = [],
    ):
        data_keys = list(map(str, Data))

        formatters = {
            **{
                "datetime": lambda v: naturaltime(v),
                "number": lambda v: intcomma(v),
                "quantity": lambda v: intword(v),
                "size": lambda v: naturalsize(v),
            },
            **self.formatters,
        }

        all_icons = {
            **{data: None for data in data_keys},
            **self.icons,
            **icons,
        }

        data_keys.extend(all_icons)
        data_keys.extend(self.styles.keys())

        self._all_spinners = {
            **{spinner: None for spinner in Spinner.__args__},
            **self.spinners,
        }
        self._all_stylers = {
            **{
                data: lambda data, value, prefix=None, icon=True: "".join(
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
                            formatters["quantity"](value)
                            if data in quantity_data
                            else (
                                formatters["datetime"](value)
                                if isinstance(value, datetime)
                                else (
                                    formatters["number"](value)
                                    if isinstance(value, int)
                                    or isinstance(value, float)
                                    else value
                                )
                            )
                        ),
                        f"[/{data}]",
                    ]
                )
                for data in data_keys
            },
            **self.stylers,
        }
        self._all_styles = {**{data: "" for data in data_keys}, **self.styles}
        self._icons = all_icons
        self._ = translator
        self._n = translator_plurals

    def get_styles(self) -> dict[Data, str]:
        return self._all_styles

    def get_spinner(self, spinner_name: str) -> Spinner:
        return self._all_spinners[spinner_name]

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
            if data in self._all_stylers and self._all_stylers[data]
            else str(value)
        )
