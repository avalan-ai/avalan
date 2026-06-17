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
    ToolCallDiagnostic,
    ToolCallError,
    User,
)
from ...event import Event, EventStats, EventType
from ...memory.permanent import Memory as Memory
from ...memory.permanent import PermanentMemoryPartition
from ...model.stream import StreamChannel, StreamItemKind
from ..display_safety import MAX_SUMMARY_ITEMS as _MAX_SUMMARY_ITEMS
from ..display_safety import event_type_value as _event_type_value
from ..display_safety import safe_summary as _safe_summary
from ..display_safety import safe_text as _safe_text
from ..display_safety import value_from as _value_from
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
    _n: Callable[[str, str, int], str]

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
            f"{marker} {_safe_text(name)}: {_safe_text(description)}\n"
            f"{self._('Author')}: {_safe_text(author)}\n"
            f"{self._('Model')}: {_safe_text(model_id)}\n"
            f"{self._('Library')}: {_safe_text(library_name)}\n"
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
            _safe_text(model.id if isinstance(model, Model) else model)
            for model in models
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
        return self._("Enter secret for {key}").format(key=_safe_text(key))

    def ask_override_secret(self, key: str) -> str:
        return self._("Secret {key} exists, override?").format(
            key=_safe_text(key)
        )

    def bye(self) -> RenderableType:
        return self._("bye :)")

    def cache_delete(
        self,
        cache_deletion: HubCacheDeletion | None,
        deleted: bool = False,
    ) -> RenderableType:
        if not cache_deletion or (
            not cache_deletion.deletable_blobs
            and not cache_deletion.deletable_refs
            and not cache_deletion.deletable_repos
            and not cache_deletion.deletable_snapshots
        ):
            return self._("Nothing found for deletion. No action taken.")

        total_revisions = len(cache_deletion.revisions)
        model_id = _safe_text(cache_deletion.model_id)
        disk_space = naturalsize(cache_deletion.deletable_size_on_disk)
        message = (
            self._n(
                "{disk_space} of disk space were freed after deleting "
                "{total_revisions} revision for {model_id}.",
                "{disk_space} of disk space were freed after deleting "
                "{total_revisions} revisions for {model_id}.",
                total_revisions,
            )
            if deleted
            else self._n(
                "{disk_space} of disk space will be freed after deleting "
                "{total_revisions} revision for {model_id}.",
                "{disk_space} of disk space will be freed after deleting "
                "{total_revisions} revisions for {model_id}.",
                total_revisions,
            )
        ).format(
            disk_space=disk_space,
            total_revisions=total_revisions,
            model_id=model_id,
        )
        if deleted:
            return message

        path_counts = [
            (self._("BLOBs"), len(cache_deletion.deletable_blobs)),
            (self._("refs"), len(cache_deletion.deletable_refs)),
            (self._("repositories"), len(cache_deletion.deletable_repos)),
            (self._("snapshots"), len(cache_deletion.deletable_snapshots)),
        ]
        populated = [
            f"{label}: {count}" for label, count in path_counts if count
        ]
        return "\n".join([message, *populated])

    def cache_list(
        self,
        cache_dir: str,
        cached_models: list[HubCache],
        display_models: list[str] | None = None,
        show_summary: bool = False,
    ) -> RenderableType:
        _ = show_summary
        selected = (
            [
                cache
                for cache in cached_models
                if cache.model_id in display_models
            ]
            if display_models
            else cached_models
        )
        if not selected:
            return self._("Cache: {cache_dir}\nModels: {models}").format(
                cache_dir=_safe_text(cache_dir),
                models=self._("none"),
            )

        lines = [
            self._("Cache: {cache_dir}").format(
                cache_dir=_safe_text(cache_dir)
            )
        ]
        for cache in selected:
            revisions = ", ".join(
                _safe_text(revision[:6]) for revision in cache.revisions
            )
            lines.append(
                self._(
                    "{model_id}: {total_revisions} revisions, "
                    "{total_files} files, {size} at {path}"
                ).format(
                    model_id=_safe_text(cache.model_id),
                    total_revisions=intcomma(cache.total_revisions),
                    total_files=intcomma(cache.total_files),
                    size=naturalsize(cache.size_on_disk),
                    path=_safe_text(cache.path),
                )
            )
            if display_models and cache.files:
                lines.append(
                    self._("Revisions: {revisions}").format(
                        revisions=revisions or self._("none")
                    )
                )
        return "\n".join(lines)

    def download_access_denied(
        self, model_id: str, model_url: str
    ) -> RenderableType:
        return self._(
            "Access denied while downloading {model_id}: {model_url}"
        ).format(
            model_id=_safe_text(model_id), model_url=_safe_text(model_url)
        )

    def download_start(self, model_id: str) -> RenderableType:
        return self._("Downloading model {model_id}.").format(
            model_id=_safe_text(model_id)
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
            model_id=_safe_text(model_id),
            path=_safe_text(path),
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
        _ = height, tool_view
        event_log = self._events_log(
            events,
            events_limit=events_limit,
            include_tokens=include_tokens,
            include_tool_detect=include_tool_detect,
            include_tools=include_tools,
            include_non_tools=include_non_tools,
        )
        return "\n".join(event_log) if event_log else None

    def _events_log(
        self,
        events: list[Event],
        *,
        events_limit: int | None,
        include_tokens: bool,
        include_tool_detect: bool,
        include_tools: bool,
        include_non_tools: bool,
    ) -> list[str] | None:
        if not events or events_limit == 0:
            return None

        selected_events: list[tuple[Event, str, bool]] = []
        for event in events:
            event_type = _event_type_value(event.type)
            is_tool = event_type.startswith("tool_")
            include_event = (
                include_tools
                and is_tool
                and (
                    event_type != EventType.TOOL_DETECT.value
                    or include_tool_detect
                )
            ) or (
                include_non_tools
                and not is_tool
                and (
                    event_type != EventType.TOKEN_GENERATED.value
                    or include_tokens
                )
            )
            if not include_event:
                continue
            selected_events.append((event, event_type, is_tool))

        if selected_events and events_limit:
            selected_events = selected_events[-events_limit:]

        event_log: list[str] = []
        for event, event_type, is_tool in selected_events:
            formatted = (
                self._format_tool_event(event, event_type) if is_tool else None
            )
            event_log.append(
                formatted
                if formatted is not None
                else self._format_generic_event(event, event_type)
            )

        return event_log or None

    def _format_tool_event(self, event: Event, event_type: str) -> str | None:
        if event_type == EventType.TOOL_EXECUTE.value:
            return self._format_tool_execute_event(event.payload)
        if event_type == EventType.TOOL_MODEL_RUN.value:
            return self._format_tool_model_run_event(event.payload)
        if event_type == EventType.TOOL_MODEL_RESPONSE.value:
            return self._format_tool_model_response_event(event.payload)
        if event_type == EventType.TOOL_PROCESS.value:
            return self._format_tool_process_event(event.payload)
        if event_type == EventType.TOOL_DIAGNOSTIC.value:
            diagnostic = self._tool_diagnostic_from_payload(event.payload)
            return (
                self._tool_diagnostic_log(diagnostic, event.payload)
                if diagnostic is not None
                else None
            )
        if event_type == EventType.TOOL_RESULT.value:
            return self._format_tool_result_event(event)
        return None

    def _format_tool_execute_event(self, payload: object) -> str | None:
        call = _value_from(payload, "call")
        if call is None:
            return None
        arguments = _value_from(call, "arguments")
        total_arguments = self._payload_size(arguments)
        return self._n(
            "Executing tool {tool} call #{call_id} with "
            "{total_arguments} argument: {arguments}.",
            "Executing tool {tool} call #{call_id} with "
            "{total_arguments} arguments: {arguments}.",
            total_arguments,
        ).format(
            tool=self._tool_name(call),
            call_id=self._tool_call_id(call),
            total_arguments=total_arguments,
            arguments=_safe_summary(arguments),
        )

    def _format_tool_model_run_event(self, payload: object) -> str | None:
        model_id = _value_from(payload, "model_id")
        messages = _value_from(payload, "messages")
        if model_id is None or messages is None:
            return None
        total_messages = self._payload_size(messages)
        return self._n(
            "Running ReACT model {model_id} with {total_messages} message.",
            "Running ReACT model {model_id} with {total_messages} messages.",
            total_messages,
        ).format(
            model_id=_safe_text(model_id),
            total_messages=total_messages,
        )

    def _format_tool_model_response_event(self, payload: object) -> str | None:
        model_id = _value_from(payload, "model_id")
        if model_id is None:
            return None
        return self._("Got ReACT response from model {model_id}.").format(
            model_id=_safe_text(model_id)
        )

    def _format_tool_process_event(self, payload: object) -> str | None:
        if not isinstance(payload, list | tuple):
            return None
        total_calls = len(payload)
        display_calls = payload[:_MAX_SUMMARY_ITEMS]
        tool_names = ", ".join(self._tool_name(call) for call in display_calls)
        remaining_calls = total_calls - len(display_calls)
        if remaining_calls:
            remaining_text = self._n(
                "{remaining_calls} more",
                "{remaining_calls} more",
                remaining_calls,
            ).format(remaining_calls=remaining_calls)
            tool_names = ", ".join([tool_names, remaining_text])
        return self._n(
            "Executing {total_calls} tool: {calls}.",
            "Executing {total_calls} tools: {calls}.",
            total_calls,
        ).format(
            total_calls=total_calls,
            calls=tool_names or self._("none"),
        )

    def _format_tool_result_event(self, event: Event) -> str | None:
        result = _value_from(event.payload, "result")
        if result is None:
            return None
        if isinstance(result, ToolCallDiagnostic):
            return self._tool_diagnostic_log(result, event.payload)

        call = _value_from(result, "call") or result
        arguments = _value_from(call, "arguments")
        total_arguments = self._payload_size(arguments)
        result_value = (
            _value_from(result, "message")
            if isinstance(result, ToolCallError)
            else _value_from(result, "result")
        )
        return self._n(
            "Executed tool {tool} call #{call_id} with "
            "{total_arguments} argument. Got result {result} in {elapsed}.",
            "Executed tool {tool} call #{call_id} with "
            "{total_arguments} arguments. Got result {result} in {elapsed}.",
            total_arguments,
        ).format(
            tool=self._tool_name(call),
            call_id=self._tool_call_id(call),
            total_arguments=total_arguments,
            result=_safe_summary(result_value),
            elapsed=self._elapsed_text(event.elapsed),
        )

    def _tool_diagnostic_from_payload(
        self, payload: object
    ) -> ToolCallDiagnostic | None:
        if not isinstance(payload, Mapping):
            return None
        for key in ("diagnostic", "result"):
            diagnostic = payload.get(key)
            if isinstance(diagnostic, ToolCallDiagnostic):
                return diagnostic
        diagnostics = payload.get("diagnostics")
        if isinstance(diagnostics, list | tuple):
            return next(
                (
                    item
                    for item in diagnostics
                    if isinstance(item, ToolCallDiagnostic)
                ),
                None,
            )
        return None

    def _tool_diagnostic_log(
        self, diagnostic: ToolCallDiagnostic, payload: object
    ) -> str:
        call = _value_from(payload, "call")
        diagnostic_details = _value_from(diagnostic, "details")
        details = (
            self._(" Details: {details}").format(
                details=_safe_summary(diagnostic_details)
            )
            if diagnostic_details
            else ""
        )
        return self._(
            "Tool diagnostic {code} at {stage} for {tool} call #{call_id}: "
            "{message}.{details}"
        ).format(
            code=_safe_text(diagnostic.code.value),
            stage=_safe_text(diagnostic.stage.value),
            tool=self._tool_name(call or diagnostic),
            call_id=self._tool_call_id(call or diagnostic),
            message=_safe_text(diagnostic.message),
            details=details,
        )

    def _format_generic_event(self, event: Event, event_type: str) -> str:
        prefix = (
            self._elapsed_text(event.elapsed)
            if event.elapsed is not None
            else self._("event")
        )
        suffix = (
            self._(": {payload}").format(payload=_safe_summary(event.payload))
            if event.payload is not None
            else ""
        )
        return f"{prefix} <{_safe_text(event_type)}>{suffix}"

    def _tool_name(self, tool: object) -> str:
        name = (
            _value_from(tool, "name")
            or _value_from(tool, "canonical_name")
            or _value_from(tool, "requested_name")
            or self._("tool")
        )
        return _safe_text(name)

    def _tool_call_id(self, tool: object) -> str:
        call_id = _value_from(tool, "call_id") or _value_from(tool, "id")
        return _safe_text(str(call_id)[:8] if call_id is not None else "?")

    def _payload_size(self, payload: object) -> int:
        if payload is None:
            return 0
        try:
            return len(cast(Any, payload))
        except Exception:
            return 1

    def _elapsed_text(self, elapsed: float | None) -> str:
        return f"{elapsed:.3f}s" if elapsed is not None else self._("unknown")

    def logging_in(self, domain: str) -> str:
        return self._("Logging in to {domain}...").format(
            domain=_safe_text(domain)
        )

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
            embedding_peek,
            horizontal,
            input_string_peek,
        )
        lines = [
            self._("Input: {input_string}").format(
                input_string=_safe_text(input_string, limit=input_string_peek)
            ),
            self._("Tokens: {total_tokens}").format(
                total_tokens=intcomma(total_tokens)
            ),
        ]
        if partition is not None and total_partitions is not None:
            lines.append(
                self._("Partition {partition} of {total_partitions}").format(
                    partition=intcomma(partition),
                    total_partitions=intcomma(total_partitions),
                )
            )
        if show_stats:
            lines.append(
                self._(
                    "Embedding stats: min {minv}, max {maxv}, mean {meanv}, "
                    "std {stdv}, norm {normv}"
                ).format(
                    minv=f"{minv:.4g}",
                    maxv=f"{maxv:.4g}",
                    meanv=f"{meanv:.4g}",
                    stdv=f"{stdv:.4g}",
                    normv=f"{normv:.4g}",
                )
            )
        return "\n".join(lines)

    def memory_embeddings_comparison(
        self, similarities: dict[str, Similarity], most_similar: str
    ) -> RenderableType:
        lines = [
            self._("Most similar: {most_similar}").format(
                most_similar=_safe_text(most_similar)
            )
        ]
        for label, similarity in similarities.items():
            lines.append(
                self._(
                    "{label}: cosine {cosine}, l2 {l2}, pearson {pearson}"
                ).format(
                    label=_safe_text(label),
                    cosine=f"{similarity.cosine_distance:.4g}",
                    l2=f"{similarity.l2_distance:.4g}",
                    pearson=f"{similarity.pearson:.4g}",
                )
            )
        return "\n".join(lines)

    def memory_embeddings_search(
        self,
        matches: list[SearchMatch],
        *args: object,
        match_preview_length: int = 300,
    ) -> RenderableType:
        _ = args, match_preview_length
        if not matches:
            return self._("No matches.")
        return "\n".join(
            self._("{query}: {match} (l2 {distance})").format(
                query=_safe_text(match.query),
                match=_safe_text(match.match, limit=match_preview_length),
                distance=f"{match.l2_distance:.4g}",
            )
            for match in matches
        )

    def memory_partitions(
        self,
        partitions: list[TextPartition],
        *args: object,
        display_partitions: int,
    ) -> RenderableType:
        _ = args
        selected = partitions[:display_partitions]
        if not selected:
            return self._("No partitions.")
        return "\n".join(
            self._("Partition {partition}: {data} ({tokens} tokens)").format(
                partition=index,
                data=_safe_text(partition.data),
                tokens=intcomma(partition.total_tokens),
            )
            for index, partition in enumerate(selected, start=1)
        )

    def model(
        self,
        model: Model,
        *args: object,
        can_access: bool | None = None,
        expand: bool = False,
        summary: bool = False,
    ) -> RenderableType:
        _ = args
        lines = [
            self._("Model: {model_id}").format(model_id=_safe_text(model.id)),
            self._("Author: {author}").format(author=_safe_text(model.author)),
            self._("Parameters: {parameters}").format(
                parameters=(
                    self._("unknown")
                    if model.parameters is None
                    else intword(model.parameters)
                )
            ),
        ]
        if can_access is not None:
            lines.append(
                self._("Access: {access}").format(
                    access=self._("yes") if can_access else self._("no")
                )
            )
        if not summary:
            lines.extend(
                [
                    self._("Downloads: {downloads}").format(
                        downloads=intcomma(model.downloads)
                    ),
                    self._("Likes: {likes}").format(
                        likes=intcomma(model.likes)
                    ),
                ]
            )
        optional_fields = [
            (self._("Library"), model.library_name),
            (self._("License"), model.license),
            (self._("Pipeline"), model.pipeline_tag),
            (self._("Type"), model.model_type),
        ]
        for label, value in optional_fields:
            if value:
                lines.append(f"{label}: {_safe_text(value)}")
        if expand:
            lines.extend(
                [
                    self._("Created: {created_at}").format(
                        created_at=naturaltime(model.created_at)
                    ),
                    self._("Updated: {updated_at}").format(
                        updated_at=naturaltime(model.updated_at)
                    ),
                ]
            )
        if model.tags:
            lines.append(
                self._("Tags: {tags}").format(
                    tags=", ".join(_safe_text(tag) for tag in model.tags)
                )
            )
        return "\n".join(lines)

    def model_display(
        self,
        model_config: ModelConfig | SentenceTransformerModelConfig | None,
        tokenizer_config: TokenizerConfig | None,
        *args: object,
        is_runnable: bool | None = None,
        summary: bool = False,
    ) -> RenderableType:
        _ = args, summary
        if isinstance(model_config, SentenceTransformerModelConfig):
            lines = [
                self._("Backend: {backend}").format(
                    backend=_safe_text(model_config.backend)
                ),
                self._("Transformer model type: {model_type}").format(
                    model_type=_safe_text(
                        model_config.transformer_model_config.model_type
                        or self._("unknown")
                    )
                ),
            ]
            if model_config.similarity_function is not None:
                lines.append(
                    self._("Similarity: {similarity}").format(
                        similarity=_safe_text(model_config.similarity_function)
                    )
                )
        else:
            model_type = getattr(model_config, "model_type", None)
            lines = [
                self._("Model type: {model_type}").format(
                    model_type=_safe_text(model_type or self._("unknown"))
                )
            ]
            if model_config is not None:
                vocab_size = getattr(model_config, "vocab_size", None)
                hidden_size = getattr(model_config, "hidden_size", None)
                lines.extend(
                    [
                        self._("Vocabulary: {vocab_size}").format(
                            vocab_size=(
                                self._("unknown")
                                if vocab_size is None
                                else intcomma(vocab_size)
                            )
                        ),
                        self._("Hidden size: {hidden_size}").format(
                            hidden_size=(
                                self._("unknown")
                                if hidden_size is None
                                else intcomma(hidden_size)
                            )
                        ),
                    ]
                )
        if tokenizer_config is not None:
            lines.append(
                self._("Tokenizer: {tokenizer}").format(
                    tokenizer=_safe_text(tokenizer_config.name_or_path)
                )
            )
        if is_runnable is not None:
            lines.append(
                self._("Runnable: {runnable}").format(
                    runnable=self._("yes") if is_runnable else self._("no")
                )
            )
        return "\n".join(lines)

    def recent_messages(
        self,
        participant_id: UUID,
        agent: Orchestrator,
        messages: list[EngineMessage],
    ) -> RenderableType:
        _ = agent
        total_messages = len(messages)
        return self._n(
            "{participant_id}: {messages} recent message",
            "{participant_id}: {messages} recent messages",
            total_messages,
        ).format(
            participant_id=_safe_text(participant_id),
            messages=intcomma(total_messages),
        )

    def saved_tokenizer_files(
        self,
        directory_path_or_total: str | int,
        total_files: int | None = None,
    ) -> RenderableType:
        total = (
            total_files if total_files is not None else directory_path_or_total
        )
        if (
            isinstance(directory_path_or_total, str)
            and total_files is not None
        ):
            return self._n(
                "Saved {total} tokenizer file to {path}.",
                "Saved {total} tokenizer files to {path}.",
                total_files,
            ).format(
                total=intcomma(total_files),
                path=_safe_text(directory_path_or_total),
            )
        total_label = intcomma(total) if isinstance(total, int) else total
        return self._("Saved tokenizer files: {total}").format(
            total=_safe_text(total_label)
        )

    def search_message_matches(
        self,
        participant_id: UUID,
        agent: Orchestrator,
        messages: list[EngineMessageScored],
    ) -> RenderableType:
        _ = agent
        total_messages = len(messages)
        return self._n(
            "{participant_id}: {messages} message match",
            "{participant_id}: {messages} message matches",
            total_messages,
        ).format(
            participant_id=_safe_text(participant_id),
            messages=intcomma(total_messages),
        )

    def memory_search_matches(
        self,
        participant_id: UUID,
        namespace: str,
        memories: list[PermanentMemoryPartition],
    ) -> RenderableType:
        total_memories = len(memories)
        return self._n(
            "{participant_id}/{namespace}: {memories} memory match",
            "{participant_id}/{namespace}: {memories} memory matches",
            total_memories,
        ).format(
            participant_id=_safe_text(participant_id),
            namespace=_safe_text(namespace),
            memories=intcomma(total_memories),
        )

    def tokenizer_config(self, config: TokenizerConfig) -> RenderableType:
        tokens = len(config.tokens or [])
        special_tokens = len(config.special_tokens or [])
        return "\n".join(
            [
                self._("Tokenizer: {tokenizer}").format(
                    tokenizer=_safe_text(config.name_or_path)
                ),
                self._("Max length: {max_length}").format(
                    max_length=intcomma(config.tokenizer_model_max_length)
                ),
                self._("Tokens: {tokens}").format(tokens=intcomma(tokens)),
                self._("Special tokens: {special_tokens}").format(
                    special_tokens=intcomma(special_tokens)
                ),
                self._("Fast: {fast}").format(
                    fast=self._("yes") if config.fast else self._("no")
                ),
            ]
        )

    def tokenizer_tokens(
        self,
        dtokens: list[Token],
        added_tokens: list[str] | None,
        special_tokens: list[str] | None,
        display_details: bool = False,
        current_dtoken: Token | None = None,
        dtokens_selected: list[Token] | None = None,
    ) -> RenderableType:
        selected_tokens = dtokens_selected or dtokens
        if not selected_tokens:
            return self._("No tokens.")

        lines: list[str] = []
        if added_tokens:
            lines.append(
                self._("Added tokens: {tokens}").format(
                    tokens=", ".join(
                        _safe_text(token) for token in added_tokens
                    )
                )
            )
        if special_tokens:
            lines.append(
                self._("Special tokens: {tokens}").format(
                    tokens=", ".join(
                        _safe_text(token) for token in special_tokens
                    )
                )
            )
        for token in selected_tokens:
            marker = "* " if token == current_dtoken else ""
            token_text = _safe_text(token.token)
            if display_details:
                token_text = self._(
                    "{token} (id {id}, p {probability})"
                ).format(
                    token=token_text,
                    id=_safe_text(token.id if token.id is not None else "?"),
                    probability=(
                        self._("unknown")
                        if token.probability is None
                        else f"{token.probability:.4g}"
                    ),
                )
            lines.append(marker + token_text)
        return "\n".join(lines)

    def display_image_entities(
        self, entities: list[ImageEntity], sort: bool
    ) -> RenderableType:
        sorted_entities = (
            sorted(entities, key=lambda entity: entity.label)
            if sort
            else entities
        )
        return "\n".join(
            self._("{label}{score}{box}").format(
                label=_safe_text(entity.label),
                score=(
                    self._(" ({score})").format(score=f"{entity.score:.4g}")
                    if entity.score is not None
                    else ""
                ),
                box=(
                    self._(" box {box}").format(box=_safe_summary(entity.box))
                    if entity.box is not None
                    else ""
                ),
            )
            for entity in sorted_entities
        )

    def display_image_entity(
        self, image_entity: ImageEntity
    ) -> RenderableType:
        return self.display_image_entities([image_entity], False)

    def display_audio_labels(
        self, audio_labels: dict[str, float]
    ) -> RenderableType:
        return "\n".join(
            f"{_safe_text(label)}: {score:.4g}"
            for label, score in audio_labels.items()
        )

    def display_image_labels(self, labels: list[str]) -> RenderableType:
        return "\n".join(_safe_text(label) for label in labels)

    def display_token_labels(
        self, token_labels: list[dict[str, str]]
    ) -> RenderableType:
        return "\n".join(
            f"{_safe_text(token)}: {_safe_text(label)}"
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
            self._("\nUser: {user_name}").format(
                user_name=_safe_text(user.name)
            )
            if user
            else ""
        )
        return self._(
            "{name} {version}\n{url}\nLicense: {license}{user}"
        ).format(
            name=_safe_text(name),
            version=_safe_text(version),
            url=_safe_text(url),
            license=_safe_text(license),
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
